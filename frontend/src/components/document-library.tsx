"use client";

/**
 * The document library: drag-and-drop upload, per-file progress, ingestion status,
 * and the owner's approval queue.
 *
 * Documents are split into Company (all READY documents) and Mine (the user's uploads).
 * Pending documents are shown separately in an approval queue visible only to owners.
 *
 * CLAUDE.md section 5: only READY documents are ever chunked or embedded. A PENDING
 * document structurally has zero rows in document_chunks — there is nothing to leak.
 * The owner approves or rejects pending uploads via the Phase 4 endpoints.
 */

import { AlertCircle, CheckCircle2, FileText, Loader2, RotateCw, Trash2, Upload, X } from "lucide-react";
import { useMemo, useState } from "react";
import { ConfirmDialog } from "./confirm-dialog";
import { StatusBadge } from "./status-badge";
import {
  type UploadState,
} from "@/lib/hooks/use-documents";
import type { DocumentSummary } from "@/lib/api";
import { cn, formatBytes, formatRelativeTime } from "@/lib/utils";

type Section = "company" | "mine";

export function DocumentLibrary({
  documents,
  uploads,
  deletingIds,
  approvingIds,
  rejectingIds,
  currentUserId,
  canManageOrg,
  onUpload,
  onDismissUpload,
  onDelete,
  onReprocess,
  onApprove,
  onReject,
  disabled,
}: {
  documents: DocumentSummary[];
  uploads: UploadState[];
  deletingIds?: ReadonlySet<string>;
  approvingIds?: ReadonlySet<string>;
  rejectingIds?: ReadonlySet<string>;
  /** Whose uploads count as "Mine". Undefined until /me resolves. */
  currentUserId?: string;
  /** Owners and admins only. Presentation; the server gates the action independently. */
  canManageOrg?: boolean;
  onUpload: () => void;
  onDismissUpload: (id: string) => void;
  onDelete: (id: string) => void;
  onReprocess: (id: string) => void;
  onApprove: (id: string) => void;
  onReject: (id: string) => void;
  disabled?: boolean;
}) {
  const [section, setSection] = useState<Section>("company");
  const [pendingDelete, setPendingDelete] = useState<DocumentSummary | null>(null);

  const { company, mine, pending } = useMemo(
    () => ({
      company: documents.filter((row) => row.status === "READY"),
      mine: documents.filter(
        (row) => row.status === "READY" && row.uploaded_by === currentUserId,
      ),
      pending: documents.filter((row) => row.status === "PENDING"),
    }),
    [documents, currentUserId],
  );

  const visible = section === "company" ? company : mine;

  // Uploads still in flight are listed separately above the library.
  const activeUploads = uploads.filter(
    (upload) => upload.phase !== "ready" || !upload.documentId,
  );

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <div
        role="tablist"
        aria-label="Document sections"
        className="flex gap-1 px-3 pb-2"
      >
        {(
          [
            ["company", "Company"],
            ["mine", "My Docs"],
          ] as const
        ).map(([value, label]) => (
          <button
            key={value}
            type="button"
            role="tab"
            aria-selected={section === value}
            onClick={() => setSection(value)}
            className={cn(
              "flex-1 rounded-md px-2 py-1 text-[0.6875rem] font-medium transition-colors",
              "focus-visible:ring-2 focus-visible:ring-accent focus-visible:outline-none",
              section === value
                ? "bg-surface-raised text-foreground"
                : "text-muted hover:text-foreground",
            )}
          >
            {label}
          </button>
        ))}
      </div>

      <div className="m-3 mt-0">
        <button
          type="button"
          disabled={disabled}
          onClick={onUpload}
          className={cn(
            "flex w-full items-center justify-center gap-1.5 rounded-lg border border-dashed p-4 text-center transition-colors",
            "border-border bg-background text-xs text-muted",
            "hover:border-accent hover:bg-accent-subtle hover:text-foreground",
            "focus-visible:ring-2 focus-visible:ring-accent focus-visible:outline-none",
            "disabled:cursor-not-allowed disabled:opacity-50",
          )}
        >
          <Upload className="size-4" aria-hidden />
          Upload files
        </button>
      </div>

      <div className="min-h-0 flex-1 overflow-y-auto px-3 pb-3">
        {/* Approval queue — owners only */}
        {canManageOrg && pending.length > 0 && (
          <div className="mb-3">
            <p className="mb-1.5 text-[0.6875rem] font-medium text-warning">
              Pending approval ({pending.length})
            </p>
            <ul className="space-y-1.5">
              {pending.map((row) => (
                <li key={row.id}>
                  <PendingRow
                    row={row}
                    isApproving={approvingIds?.has(row.id)}
                    isRejecting={rejectingIds?.has(row.id)}
                    onApprove={() => onApprove(row.id)}
                    onReject={() => onReject(row.id)}
                  />
                </li>
              ))}
            </ul>
          </div>
        )}

        {activeUploads.length > 0 && (
          <ul className="mb-3 space-y-1.5">
            {activeUploads.map((upload) => (
              <li key={upload.id}>
                <UploadRow upload={upload} onDismiss={onDismissUpload} />
              </li>
            ))}
          </ul>
        )}

        {visible.length === 0 && activeUploads.length === 0 && pending.length === 0 ? (
          <p className="px-1 py-6 text-center text-xs text-muted">
            {section === "company"
              ? "No company documents yet."
              : "No personal documents yet."}
          </p>
        ) : (
          <ul className="space-y-0.5">
            {visible.map((row) => (
              <li key={row.id}>
                <DocumentRow
                  row={row}
                  isDeleting={deletingIds?.has(row.id)}
                  onRequestDelete={() => setPendingDelete(row)}
                  onReprocess={() => onReprocess(row.id)}
                />
              </li>
            ))}
          </ul>
        )}
      </div>

      <ConfirmDialog
        open={pendingDelete !== null}
        title="Delete this document?"
        description={
          <>
            <span className="font-medium text-foreground">
              {pendingDelete?.filename}
            </span>{" "}
            and everything indexed from it will be removed permanently. Answers will
            no longer be able to cite it. This cannot be undone.
          </>
        }
        isBusy={pendingDelete ? deletingIds?.has(pendingDelete.id) : false}
        onConfirm={() => {
          if (pendingDelete) onDelete(pendingDelete.id);
          setPendingDelete(null);
        }}
        onCancel={() => setPendingDelete(null)}
      />
    </div>
  );
}

function PendingRow({
  row,
  isApproving,
  isRejecting,
  onApprove,
  onReject,
}: {
  row: DocumentSummary;
  isApproving?: boolean;
  isRejecting?: boolean;
  onApprove: () => void;
  onReject: () => void;
}) {
  return (
    <div className="rounded-lg border border-warning/30 bg-warning-subtle px-2.5 py-2">
      <div className="flex items-center gap-2">
        <FileText className="size-3.5 shrink-0 text-warning" aria-hidden />
        <span className="min-w-0 flex-1 truncate text-xs font-medium">
          {row.filename}
        </span>
        <span className="shrink-0 text-[0.6875rem] text-muted">
          {formatBytes(row.file_size)}
        </span>
      </div>
      <p className="mt-1 text-[0.6875rem] text-muted">
        Uploaded {formatRelativeTime(row.created_at)}
      </p>
      <div className="mt-1.5 flex gap-1.5">
        <button
          type="button"
          onClick={onApprove}
          disabled={isApproving || isRejecting}
          className={cn(
            "flex items-center gap-1 rounded-md bg-success px-2.5 py-1 text-[0.6875rem] font-medium text-white",
            "transition-opacity hover:opacity-90",
            "focus-visible:ring-2 focus-visible:ring-accent focus-visible:outline-none",
            "disabled:cursor-not-allowed disabled:opacity-50",
          )}
        >
          {isApproving ? (
            <Loader2 className="size-3 animate-spin" aria-hidden />
          ) : (
            <CheckCircle2 className="size-3" aria-hidden />
          )}
          Approve
        </button>
        <button
          type="button"
          onClick={onReject}
          disabled={isApproving || isRejecting}
          className={cn(
            "flex items-center gap-1 rounded-md border border-border px-2.5 py-1 text-[0.6875rem] font-medium text-muted",
            "transition-colors hover:text-foreground",
            "focus-visible:ring-2 focus-visible:ring-accent focus-visible:outline-none",
            "disabled:cursor-not-allowed disabled:opacity-50",
          )}
        >
          {isRejecting ? (
            <Loader2 className="size-3 animate-spin" aria-hidden />
          ) : (
            <X className="size-3" aria-hidden />
          )}
          Reject
        </button>
      </div>
    </div>
  );
}

function UploadRow({
  upload,
  onDismiss,
}: {
  upload: UploadState;
  onDismiss: (id: string) => void;
}) {
  const isFailed = upload.phase === "failed";

  return (
    <div
      className={cn(
        "rounded-lg border px-2.5 py-2",
        isFailed ? "border-danger/30 bg-danger-subtle" : "border-border bg-background",
      )}
    >
      <div className="flex items-center gap-2">
        <span className="min-w-0 flex-1 truncate text-xs font-medium">
          {upload.filename}
        </span>
        <span className="shrink-0 text-[0.6875rem] text-muted">
          {formatBytes(upload.sizeBytes)}
        </span>
        <button
          type="button"
          onClick={() => onDismiss(upload.id)}
          aria-label={`Dismiss ${upload.filename}`}
          className={cn(
            "flex size-5 shrink-0 items-center justify-center rounded text-muted",
            "transition-colors hover:text-foreground",
            "focus-visible:ring-2 focus-visible:ring-accent focus-visible:outline-none",
          )}
        >
          <X className="size-3" aria-hidden />
        </button>
      </div>

      {upload.phase === "transferring" && (
        <div className="mt-1.5">
          <div
            role="progressbar"
            aria-valuenow={Math.round(upload.progress * 100)}
            aria-valuemin={0}
            aria-valuemax={100}
            aria-label={`Uploading ${upload.filename}`}
            className="h-1 overflow-hidden rounded-full bg-surface-raised"
          >
            <div
              className="h-full rounded-full bg-accent transition-[width] duration-200"
              style={{ width: `${Math.round(upload.progress * 100)}%` }}
            />
          </div>
          <p className="mt-1 text-[0.6875rem] text-muted">
            Uploading · {Math.round(upload.progress * 100)}%
          </p>
        </div>
      )}

      {upload.phase === "processing" && (
        <p className="mt-1.5 flex items-center gap-1.5 text-[0.6875rem] text-muted">
          <Loader2 className="size-3 animate-spin" aria-hidden />
          Processing — extracting and indexing
        </p>
      )}

      {upload.phase === "ready" && (
        <p className="mt-1.5 flex items-center gap-1.5 text-[0.6875rem] text-success">
          <CheckCircle2 className="size-3" aria-hidden />
          {upload.error ? upload.error : "Ready to query"}
        </p>
      )}

      {isFailed && (
        <p className="mt-1.5 flex items-start gap-1.5 text-[0.6875rem] text-danger">
          <AlertCircle className="mt-px size-3 shrink-0" aria-hidden />
          <span>{upload.error ?? "Upload failed."}</span>
        </p>
      )}
    </div>
  );
}

function DocumentRow({
  row,
  isDeleting,
  onRequestDelete,
  onReprocess,
}: {
  row: DocumentSummary;
  isDeleting?: boolean;
  onRequestDelete: () => void;
  onReprocess: () => void;
}) {
  const isFailed = row.status === "FAILED";

  const details = [
    row.file_size ? formatBytes(row.file_size) : null,
    formatRelativeTime(row.created_at),
  ].filter(Boolean);

  return (
    <div
      className={cn(
        "group flex items-start gap-2 rounded-md px-2 py-1.5 hover:bg-surface-raised",
        isDeleting && "opacity-50",
      )}
    >
      <FileText className="mt-0.5 size-3.5 shrink-0 text-muted" aria-hidden />
      <div className="min-w-0 flex-1">
        <p className="truncate text-xs" title={row.filename}>
          {row.filename}
        </p>
        <p className="mt-0.5 flex flex-wrap items-center gap-1 text-[0.6875rem] text-muted">
          <StatusBadge status={row.status} />
          {details.join(" · ")}
        </p>
        {isFailed && row.error_message && (
          <p className="mt-0.5 text-[0.6875rem] break-words text-danger">
            {row.error_message}
          </p>
        )}
      </div>

      <span className="flex shrink-0 items-center gap-0.5">
        {isFailed && (
          <button
            type="button"
            onClick={onReprocess}
            disabled={isDeleting}
            aria-label={`Retry processing ${row.filename}`}
            title="Try processing again"
            className={cn(
              "flex size-6 items-center justify-center rounded text-muted opacity-0",
              "transition-opacity group-hover:opacity-100 focus-visible:opacity-100",
              "hover:text-accent focus-visible:ring-2 focus-visible:ring-accent",
              "focus-visible:outline-none disabled:cursor-not-allowed",
            )}
          >
            <RotateCw className="size-3" aria-hidden />
          </button>
        )}
        <button
          type="button"
          onClick={onRequestDelete}
          disabled={isDeleting}
          aria-label={`Delete ${row.filename}`}
          className={cn(
            "flex size-6 items-center justify-center rounded text-muted opacity-0",
            "transition-opacity group-hover:opacity-100 focus-visible:opacity-100",
            "hover:text-danger focus-visible:ring-2 focus-visible:ring-accent",
            "focus-visible:outline-none disabled:cursor-not-allowed",
          )}
        >
          {isDeleting ? (
            <Loader2 className="size-3 animate-spin" aria-hidden />
          ) : (
            <Trash2 className="size-3" aria-hidden />
          )}
        </button>
      </span>
    </div>
  );
}
