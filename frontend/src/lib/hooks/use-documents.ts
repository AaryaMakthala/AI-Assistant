"use client";

/**
 * The document library: listing, uploading, and managing documents.
 *
 * Upload is synchronous (Phase 3) — the backend performs extraction, chunking and
 * embedding inline. Owner uploads return READY immediately; member uploads return
 * PENDING. There is no separate ingestion job to poll.
 *
 * Owners can approve or reject PENDING member uploads (Phase 4), which triggers
 * inline ingestion on approval.
 */

import { useCallback, useEffect, useRef, useState } from "react";
import {
  ApiError,
  approveDocument,
  deleteDocument,
  getDocument,
  listDocuments,
  rejectDocument,
  uploadDocumentWithProgress,
  type DocumentSummary,
} from "@/lib/api";

/** Extensions the backend accepts. Checked client-side purely to fail fast — the
 * backend allowlist is the one that actually enforces. */
export const ACCEPTED_EXTENSIONS = [
  "pdf",
  "docx",
  "csv",
  "xlsx",
  "txt",
  "md",
  "markdown",
] as const;

/** Mirrors the backend's max upload size. */
export const MAX_UPLOAD_BYTES = 25 * 1024 * 1024;

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
  /** Documents with a delete in flight, so their row can show it and block a second click. */
  const [deletingIds, setDeletingIds] = useState<ReadonlySet<string>>(new Set());
  /** Documents with an approve in flight. */
  const [approvingIds, setApprovingIds] = useState<ReadonlySet<string>>(new Set());
  /** Documents with a reject in flight. */
  const [rejectingIds, setRejectingIds] = useState<ReadonlySet<string>>(new Set());

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
    // eslint-disable-next-line react-hooks/set-state-in-effect
    void refresh();
  }, [refresh]);

  const upload = useCallback(
    async (file: File) => {
      uploadCounter += 1;
      const id = `upload-${uploadCounter}`;

      const rejection = validateFile(file);
      if (rejection) {
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
        // Ingestion is synchronous — if the document is READY, it's done; if PENDING,
        // it waits for owner approval. No polling needed.
        const finalPhase: UploadPhase =
          accepted.document.status === "READY"
            ? "ready"
            : accepted.document.status === "FAILED"
              ? "failed"
              : accepted.document.status === "PENDING"
                ? "ready" // Show as "ready" in the upload list since the upload itself succeeded
                : "processing";
        patch({
          phase: finalPhase,
          progress: 1,
          documentId: accepted.document.id,
          error: accepted.document.error_message ?? undefined,
        });
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

  /**
   * Delete a document and everything derived from it.
   */
  const remove = useCallback(
    async (documentId: string) => {
      if (!token) return;
      setDeletingIds((current) => new Set(current).add(documentId));
      try {
        await deleteDocument(documentId, { token });
      } catch (caught) {
        if (!(caught instanceof ApiError && caught.status === 404)) {
          if (mountedRef.current) {
            setError(
              caught instanceof ApiError
                ? caught.message
                : "The document could not be deleted.",
            );
          }
          return;
        }
      } finally {
        if (mountedRef.current) {
          setDeletingIds((current) => {
            const next = new Set(current);
            next.delete(documentId);
            return next;
          });
        }
      }

      if (!mountedRef.current) return;
      setDocuments((current) => current.filter((row) => row.id !== documentId));
      setUploads((current) =>
        current.filter((item) => item.documentId !== documentId),
      );
      setError(undefined);
    },
    [token],
  );

  /**
   * Approve a pending document (owner only).
   *
   * PENDING → ingest inline → READY + chunks, atomically.
   */
  const approve = useCallback(
    async (documentId: string) => {
      if (!token) return;
      setApprovingIds((current) => new Set(current).add(documentId));
      try {
        const result = await approveDocument(documentId, { token });
        if (!mountedRef.current) return;
        setDocuments((current) =>
          current.map((row) =>
            row.id === documentId ? result.document : row,
          ),
        );
        setError(undefined);
      } catch (caught) {
        if (mountedRef.current) {
          setError(
            caught instanceof ApiError
              ? caught.message
              : "The document could not be approved.",
          );
        }
      } finally {
        if (mountedRef.current) {
          setApprovingIds((current) => {
            const next = new Set(current);
            next.delete(documentId);
            return next;
          });
        }
      }
    },
    [token],
  );

  /**
   * Reject a pending document (owner only).
   *
   * PENDING → REJECTED, never ingested.
   */
  const reject = useCallback(
    async (documentId: string) => {
      if (!token) return;
      setRejectingIds((current) => new Set(current).add(documentId));
      try {
        const updated = await rejectDocument(documentId, { token });
        if (!mountedRef.current) return;
        setDocuments((current) =>
          current.map((row) =>
            row.id === documentId ? updated : row,
          ),
        );
        setError(undefined);
      } catch (caught) {
        if (mountedRef.current) {
          setError(
            caught instanceof ApiError
              ? caught.message
              : "The document could not be rejected.",
          );
        }
      } finally {
        if (mountedRef.current) {
          setRejectingIds((current) => {
            const next = new Set(current);
            next.delete(documentId);
            return next;
          });
        }
      }
    },
    [token],
  );

  /** Re-run ingestion for a document that failed. (Not supported by the current
   * backend — the document would need to be re-uploaded. Included for forward
   * compatibility.) */
  const reprocess = useCallback(
    async (documentId: string) => {
      // The current backend does not have a reprocess endpoint for Phase 3/4.
      // A failed document needs to be re-uploaded.
      if (!token) return;
      try {
        const doc = await getDocument(documentId, { token });
        if (!mountedRef.current) return;
        setDocuments((current) =>
          current.map((row) => (row.id === documentId ? doc : row)),
        );
        setError(undefined);
      } catch (caught) {
        if (mountedRef.current) {
          setError(
            caught instanceof ApiError
              ? caught.message
              : "Could not refresh the document status.",
          );
        }
      }
    },
    [token],
  );

  return {
    documents,
    uploads,
    isLoading,
    error,
    deletingIds,
    approvingIds,
    rejectingIds,
    refresh,
    upload,
    dismissUpload,
    remove,
    approve,
    reject,
    reprocess,
    clearError: useCallback(() => setError(undefined), []),
  };
}
