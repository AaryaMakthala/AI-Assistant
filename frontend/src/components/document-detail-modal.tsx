"use client";

/**
 * Detail view for an existing document, opened from a card click.
 *
 * A centered modal over a dimmed backdrop. Clicking the card opens it; ×,
 * Escape, or a click on the backdrop dismisses it. The Download button fetches
 * the raw bytes from the backend and hands them to the browser.
 */

import { Download, Loader2, X } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { Button } from "./button";
import { StatusBadge } from "./status-badge";
import { downloadDocument } from "@/lib/api";
import type { DocumentSummary } from "@/lib/api";
import { cn, formatBytes } from "@/lib/utils";

export function DocumentDetailModal({
  document,
  token,
  workspaceId,
  onClose,
}: {
  document: DocumentSummary | null;
  token?: string;
  workspaceId?: string;
  onClose: () => void;
}) {
  const ref = useRef<HTMLDialogElement>(null);
  const [isDownloading, setIsDownloading] = useState(false);
  const [downloadError, setDownloadError] = useState<string | null>(null);

  useEffect(() => {
    const dialog = ref.current;
    if (!dialog) return;
    if (document && !dialog.open) dialog.showModal();
    if (!document && dialog.open) dialog.close();
  }, [document]);

  if (!document) return null;

  const handleDownload = async () => {
    if (!token) return;
    setIsDownloading(true);
    setDownloadError(null);
    try {
      await downloadDocument(document.id, document.filename, {
        token,
        workspaceId,
      });
    } catch (err) {
      setDownloadError(
        err instanceof Error ? err.message : "Download failed.",
      );
    } finally {
      setIsDownloading(false);
    }
  };

  return (
    <dialog
      ref={ref}
      onCancel={(event) => {
        event.preventDefault();
        onClose();
      }}
      aria-labelledby="document-detail-title"
      className={cn(
        "fixed inset-0 z-50 m-auto w-[min(28rem,calc(100vw-2rem))]",
        "rounded-2xl border border-[rgba(255,255,255,0.1)]",
        "bg-[rgba(20,40,30,0.9)] backdrop-blur-[20px] p-0 text-foreground shadow-2xl",
        "backdrop:bg-black/50 backdrop:backdrop-blur-sm",
      )}
    >
      <div className="flex items-start justify-between gap-3 border-b border-[rgba(255,255,255,0.1)] px-5 py-4">
        <div className="min-w-0">
          <h2
            id="document-detail-title"
            className="font-display text-base break-words"
          >
            {document.filename}
          </h2>
          <div className="mt-1.5 flex items-center gap-2">
            <StatusBadge status={document.status} />
          </div>
        </div>
        <button
          type="button"
          onClick={onClose}
          aria-label="Close"
          className={cn(
            "flex size-7 shrink-0 items-center justify-center rounded-md text-muted",
            "transition-colors hover:bg-surface-raised hover:text-foreground",
            "focus-visible:ring-2 focus-visible:ring-accent focus-visible:outline-none",
          )}
        >
          <X className="size-4" aria-hidden />
        </button>
      </div>

      <div className="space-y-3 px-5 py-4">
        {document.description && (
          <p className="text-sm leading-relaxed break-words text-foreground/90">
            {document.description}
          </p>
        )}
        <dl className="space-y-1.5 text-xs">
          <div className="flex justify-between">
            <dt className="text-muted">File size</dt>
            <dd className="font-medium">{formatBytes(document.file_size)}</dd>
          </div>
          <div className="flex justify-between">
            <dt className="text-muted">Upload date</dt>
            <dd className="font-medium">
              {new Date(document.created_at).toLocaleDateString(undefined, {
                year: "numeric",
                month: "short",
                day: "numeric",
              })}
            </dd>
          </div>
          {document.mime_type && (
            <div className="flex justify-between">
              <dt className="text-muted">Type</dt>
              <dd className="font-medium">{document.mime_type}</dd>
            </div>
          )}
        </dl>
      </div>

      <div className="flex flex-col gap-2 border-t border-[rgba(255,255,255,0.1)] px-5 py-4">
        {downloadError && (
          <p className="text-[0.6875rem] text-danger">{downloadError}</p>
        )}
        <div className="flex justify-end gap-2">
          <Button variant="ghost" onClick={onClose}>
            Close
          </Button>
          <Button
            variant="primary"
            onClick={() => void handleDownload()}
            disabled={!token || isDownloading}
          >
            {isDownloading ? (
              <Loader2 className="size-3.5 animate-spin" aria-hidden />
            ) : (
              <Download className="size-3.5" aria-hidden />
            )}
            Download
          </Button>
        </div>
      </div>
    </dialog>
  );
}
