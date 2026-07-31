"use client";

/**
 * The document library: listing, uploading, and following ingestion to completion.
 *
 * Upload is two distinct phases and the UI has to tell them apart, because they fail
 * differently and take different amounts of time:
 *
 * 1. **Transfer** — bytes to the API. Progress is real (XHR reports it), and a failure
 *    here is the user's: wrong file type, too large, no connection.
 * 2. **Ingestion** — a Celery job that extracts, chunks and embeds. Starts after the 202
 *    and has no progress signal at all, only a status column that ends `ready` or
 *    `failed`. Showing a fake percentage for this would be a lie, so it gets an
 *    indeterminate state instead.
 *
 * Polling stops on a terminal status, when nothing is left in flight, or after
 * `MAX_POLLS` — a job wedged in `processing` must not have the browser querying it
 * forever.
 */

import { useCallback, useEffect, useRef, useState } from "react";
import {
  ApiError,
  getDocument,
  listDocuments,
  uploadDocumentWithProgress,
  type DocumentSummary,
} from "@/lib/api";

/** Extensions the backend accepts (CLAUDE.md 4.2). Checked client-side purely to fail
 * fast with a clear message — the backend allowlist is the one that actually enforces. */
export const ACCEPTED_EXTENSIONS = ["pdf", "docx", "csv", "xlsx", "txt"] as const;

/** Mirrors the backend's `max_upload_bytes`. Same reasoning: a local check saves a
 * 25 MB round trip that would only be rejected, but it is not the enforcement point. */
export const MAX_UPLOAD_BYTES = 25 * 1024 * 1024;

const POLL_INTERVAL_MS = 2000;
/** ~2 minutes of polling. A job still running past that is not stuck for a reason the
 * UI can help with; the user can refresh. */
const MAX_POLLS = 60;

export type UploadPhase = "transferring" | "processing" | "ready" | "failed";

/** One upload as the UI follows it, from selection through ingestion. */
export interface UploadState {
  id: string;
  filename: string;
  sizeBytes: number;
  phase: UploadPhase;
  /** Transfer fraction, 0–1. Meaningful only while `phase === "transferring"`. */
  progress: number;
  /** Set once the API accepts the file; how the document is then polled. */
  documentId?: string;
  error?: string;
}

let uploadCounter = 0;

function extensionOf(filename: string): string {
  const dot = filename.lastIndexOf(".");
  return dot === -1 ? "" : filename.slice(dot + 1).toLowerCase();
}

/** Reject a file the backend certainly would, with a message naming the actual reason. */
export function validateFile(file: File): string | null {
  const extension = extensionOf(file.name);
  if (!ACCEPTED_EXTENSIONS.includes(extension as (typeof ACCEPTED_EXTENSIONS)[number])) {
    return `${extension ? `.${extension}` : "This"} files are not accepted. Allowed: ${ACCEPTED_EXTENSIONS.map((e) => `.${e}`).join(", ")}.`;
  }
  if (file.size > MAX_UPLOAD_BYTES) {
    return `This file is larger than the ${Math.round(MAX_UPLOAD_BYTES / 1024 / 1024)} MB limit.`;
  }
  if (file.size === 0) {
    return "This file is empty.";
  }
  return null;
}

export function useDocuments(token?: string) {
  const [documents, setDocuments] = useState<DocumentSummary[]>([]);
  const [uploads, setUploads] = useState<UploadState[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | undefined>();

  const pollCountRef = useRef(0);
  const mountedRef = useRef(true);

  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
    };
  }, []);

  const refresh = useCallback(async () => {
    if (!token) return;
    setIsLoading(true);
    try {
      const { documents: rows } = await listDocuments({ token, limit: 100 });
      if (mountedRef.current) {
        setDocuments(rows);
        setError(undefined);
      }
    } catch (caught) {
      if (mountedRef.current) {
        setError(
          caught instanceof ApiError
            ? caught.message
            : "Could not load the document library.",
        );
      }
    } finally {
      if (mountedRef.current) setIsLoading(false);
    }
  }, [token]);

  useEffect(() => {
    // Fetch-on-mount: `refresh` flips the loading flag before awaiting, which the rule
    // reads as a cascading render. The cascade is the point here — one extra render to
    // show the spinner, then one to show the rows.
    // eslint-disable-next-line react-hooks/set-state-in-effect
    void refresh();
  }, [refresh]);

  const upload = useCallback(
    async (file: File) => {
      uploadCounter += 1;
      const id = `upload-${uploadCounter}`;

      const rejection = validateFile(file);
      if (rejection) {
        // Surfaced as a failed upload row rather than a toast: it stays on screen next to
        // the filename it refers to, which is what makes the reason actionable.
        setUploads((current) => [
          ...current,
          {
            id,
            filename: file.name,
            sizeBytes: file.size,
            phase: "failed",
            progress: 0,
            error: rejection,
          },
        ]);
        return;
      }

      setUploads((current) => [
        ...current,
        {
          id,
          filename: file.name,
          sizeBytes: file.size,
          phase: "transferring",
          progress: 0,
        },
      ]);

      const patch = (change: Partial<UploadState>) => {
        if (!mountedRef.current) return;
        setUploads((current) =>
          current.map((item) => (item.id === id ? { ...item, ...change } : item)),
        );
      };

      try {
        const accepted = await uploadDocumentWithProgress(file, {
          token,
          onProgress: (fraction) => patch({ progress: fraction }),
        });
        patch({
          phase: "processing",
          progress: 1,
          documentId: accepted.document.id,
        });
        // Reset the budget: a new job deserves a full polling window even if an earlier
        // one exhausted it.
        pollCountRef.current = 0;
        setDocuments((current) => [accepted.document, ...current]);
      } catch (caught) {
        if ((caught as DOMException)?.name === "AbortError") return;
        patch({
          phase: "failed",
          error:
            caught instanceof ApiError
              ? caught.message
              : "The upload failed. Please try again.",
        });
      }
    },
    [token],
  );

  /** Discard a finished or failed upload row. */
  const dismissUpload = useCallback((id: string) => {
    setUploads((current) => current.filter((item) => item.id !== id));
  }, []);

  // Follow queued and in-flight ingestion jobs to a terminal status.
  const pending = documents.filter(
    (document) => document.status === "pending" || document.status === "processing",
  );
  const pendingKey = pending.map((document) => document.id).join(",");

  useEffect(() => {
    if (!token || !pendingKey) return;

    const ids = pendingKey.split(",");
    let cancelled = false;

    const timer = setInterval(async () => {
      if (pollCountRef.current >= MAX_POLLS) {
        clearInterval(timer);
        return;
      }
      pollCountRef.current += 1;

      const updated = await Promise.all(
        ids.map(async (id) => {
          try {
            return await getDocument(id, { token });
          } catch {
            // A transient failure must not clear a row from the library — the next tick
            // tries again, and the stale status is closer to the truth than nothing.
            return null;
          }
        }),
      );
      if (cancelled || !mountedRef.current) return;

      const byId = new Map(
        updated.filter((row): row is DocumentSummary => row !== null).map((row) => [row.id, row]),
      );
      if (!byId.size) return;

      setDocuments((current) => current.map((row) => byId.get(row.id) ?? row));
      setUploads((current) =>
        current.map((item) => {
          const row = item.documentId ? byId.get(item.documentId) : undefined;
          if (!row) return item;
          if (row.status === "ready") return { ...item, phase: "ready" };
          if (row.status === "failed") {
            return {
              ...item,
              phase: "failed",
              error: row.error_message ?? "Processing failed.",
            };
          }
          return item;
        }),
      );
    }, POLL_INTERVAL_MS);

    return () => {
      cancelled = true;
      clearInterval(timer);
    };
  }, [token, pendingKey]);

  return {
    documents,
    uploads,
    isLoading,
    error,
    refresh,
    upload,
    dismissUpload,
  };
}
