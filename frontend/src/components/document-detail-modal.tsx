"use client";

/**
 * Detail view for an existing document, opened from a card click.
 *
 * A centered modal over a dimmed backdrop. Clicking the card opens it; ×,
 * Escape, or a click on the backdrop dismisses it. The Download button fetches
 * the raw bytes from the backend and hands them to the browser.
 *
 * The modal itself owns no delete logic: it renders the app's ConfirmDialog for
 * the destructive action and delegates to `onDelete`. The bug this guards
 * against is error *routing* — `useDocuments` publishes every failure through
 * its single shared `error` channel (a banner next to the chat transcript),
 * while a modal stacked above the dialog covers the workspace chrome. A delete
 * that failed inside this modal therefore looked like a dead button: the
 * confirm closed, nothing was removed, and the explaining banner sat hidden
 * underneath. The modal now captures the caller's failure locally and shows it
 * right where the click happened, and the confirm stays open unless the delete
 * actually succeeded — the modal's own success contract for removal.
 */

import { Download, Loader2, X } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { Button } from "./button";
import { ConfirmDialog } from "./confirm-dialog";
import { StatusBadge } from "./status-badge";
import { downloadDocument } from "@/lib/api";
import type { DocumentSummary } from "@/lib/api";
import { cn, formatBytes } from "@/lib/utils";

export function DocumentDetailModal({
  document,
  token,
  workspaceId,
  onClose,
  onDelete,
  isDeleting,
}: {
  document: DocumentSummary | null;
  token?: string;
  workspaceId?: string;
  onClose: () => void;
  /**
   * The actual delete call. Resolves to a user-facing failure message (null on
   * success) so the reason shows right here in the modal, where the click
   * happened — never in the shared page-level banner. On success the parent
   * closes this modal.
   */
  onDelete?: (id: string) => Promise<string | null> | string | null;
  /** The delete in flight, for the confirm's busy state. */
  isDeleting?: boolean;
}) {
  const ref = useRef<HTMLDialogElement>(null);
  const [isDownloading, setIsDownloading] = useState(false);
  const [downloadError, setDownloadError] = useState<string | null>(null);
  /** Delete confirm state + failure message, reset whenever a new document opens. */
  const [deleteUi, setDeleteUi] = useState<{ confirming: boolean; error: string | null }>(
    { confirming: false, error: null },
  );

  useEffect(() => {
    const dialog = ref.current;
    if (!dialog) return;
    if (document && !dialog.open) dialog.showModal();
    if (!document && dialog.open) dialog.close();
  }, [document]);

  // The modal stays mounted (its parent always renders it), so delete state must
  // reset when a different document is opened — a stale error or confirm must
  // never leak from the previous document into this one. (Single setState per
  // the repo's set-state-in-effect lint convention — see use-documents.ts.)
  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect
    setDeleteUi({ confirming: false, error: null });
  }, [document?.id]);

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

  /**
   * Perform the delete. The hook's `onDelete` resolves to a user-facing
   * failure message (null on success) instead of throwing, so the reason shows
   * right here in the modal — where the user clicked — rather than in a page
   * banner elsewhere.
   */
  const handleDelete = async () => {
    if (!onDelete) return;
    const failure = await onDelete(document.id);
    if (failure) {
      setDeleteUi({ confirming: false, error: failure });
    }
  };

  return (
    <dialog
      ref={ref}
      onCancel={(event) => {
        event.preventDefault();
        if (!deleteUi.confirming && !isDeleting) onClose();
        // While the confirm or a delete is in flight, Escape must not rip the
        // confirm out of the top layer from under the user.
        else event.preventDefault();
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
            id="document-detail-modal-title"
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

      {/* max-h + overflow-y: a long description must be readable in FULL here —
       * the card grid clamps to 2 lines, this modal never does. */}
      <div className="max-h-[50vh] space-y-3 overflow-y-auto px-5 py-4">
        {deleteUi.error && (
          <p role="alert" className="text-xs break-words text-danger">
            {deleteUi.error}
          </p>
        )}
        {/* Intentionally no line-clamp: the description renders in full. */}
        {document.description && (
          <p className="text-sm leading-relaxed break-words whitespace-pre-wrap text-foreground/90">
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
          {onDelete && (
            <Button
              variant="danger"
              onClick={() => setDeleteUi((current) => ({ ...current, confirming: true }))}
              disabled={isDeleting}
            >
              Delete
            </Button>
          )}
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

      <ConfirmDialog
        open={deleteUi.confirming}
        title="Delete this document?"
        description={
          <>
            <span className="font-medium text-foreground">
              {document.filename}
            </span>{" "}
            and everything indexed from it — chunks, embeddings and the stored
            file — will be removed permanently. Answers will no longer be able
            to cite it. This cannot be undone.
          </>
        }
        isBusy={isDeleting}
        onConfirm={() => void handleDelete()}
        onCancel={() => setDeleteUi((current) => ({ ...current, confirming: false }))}
      />
    </dialog>
  );
}
