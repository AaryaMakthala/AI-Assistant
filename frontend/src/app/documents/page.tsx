"use client";

/**
 * The documents page: the owner's view of every document in the workspace.
 *
 * The sidebar's library is for glancing at while chatting; this is for managing. Same
 * hook, same data, different density — it has room for the timestamps, counts and errors
 * the rail has to leave out, and for a drop zone big enough to actually aim at.
 *
 * Owners and admins only, because it lists all documents (CLAUDE.md 4.6). That is an
 * oversight view, so a member is redirected rather than shown an emptier version of it
 * — and the backend gates the rows regardless of what this page renders.
 *
 * Auth is enforced by the backend on every call (CLAUDE.md 4.6).
 */

import { AlertCircle, ArrowLeft, CheckCircle2, FileText, Loader2, RotateCw, Trash2, Upload, X } from "lucide-react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { useEffect, useRef, useState } from "react";
import { ConfirmDialog } from "@/components/confirm-dialog";
import { StatusBadge } from "@/components/status-badge";
import type { DocumentSummary } from "@/lib/api";
import { useAuth } from "@/lib/auth";
import {
  ACCEPTED_EXTENSIONS,
  MAX_UPLOAD_BYTES,
  useDocuments,
  type UploadState,
} from "@/lib/hooks/use-documents";
import { cn, formatBytes, formatRelativeTime } from "@/lib/utils";

const ACCEPT_ATTRIBUTE = ACCEPTED_EXTENSIONS.map((ext) => `.${ext}`).join(",");
const ADMIN_ROLES = ["OWNER", "owner"];

export default function DocumentsPage() {
  const router = useRouter();
  const { token, isAuthenticated, isLoading: isAuthLoading, role } = useAuth();
  const library = useDocuments(token);
  const canManageOrg = Boolean(role && ADMIN_ROLES.includes(role));

  const [isDragging, setIsDragging] = useState(false);
  const [pendingDelete, setPendingDelete] = useState<DocumentSummary | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const dragDepth = useRef(0);

  useEffect(() => {
    if (isAuthLoading) return;
    if (!isAuthenticated) {
      router.replace("/login");
      return;
    }
    if (role && !canManageOrg) router.replace("/");
  }, [isAuthLoading, isAuthenticated, role, canManageOrg, router]);

  const handleFiles = (files: FileList | null) => {
    if (!files || !isAuthenticated || !canManageOrg) return;
    for (const file of Array.from(files)) void library.upload(file);
  };

  const activeUploads = library.uploads.filter(
    (upload) => upload.phase !== "ready" || !upload.documentId,
  );

  const pending = library.documents.filter((row) => row.status === "PENDING");

  return (
    <div className="mx-auto flex min-h-dvh w-full max-w-4xl flex-col gap-4 p-4 sm:p-6">
      <header className="flex items-center justify-between gap-3">
        <div className="min-w-0">
          <Link
            href="/"
            className={cn(
              "inline-flex items-center gap-1 text-xs text-muted transition-colors",
              "hover:text-foreground focus-visible:ring-2 focus-visible:ring-accent",
              "focus-visible:outline-none",
            )}
          >
            <ArrowLeft className="size-3" aria-hidden />
            Back to chat
          </Link>
          <h1 className="mt-1 text-lg font-semibold">All documents</h1>
          <p className="text-xs text-muted">
            {library.documents.length === 0
              ? "Nothing uploaded yet."
              : `${library.documents.length} document${library.documents.length === 1 ? "" : "s"} in your workspace.`}
          </p>
        </div>
        <button
          type="button"
          onClick={() => void library.refresh()}
          disabled={library.isLoading}
          className={cn(
            "flex shrink-0 items-center gap-1.5 rounded-md border border-border px-2.5",
            "py-1.5 text-xs text-muted transition-colors hover:text-foreground",
            "focus-visible:ring-2 focus-visible:ring-accent focus-visible:outline-none",
            "disabled:opacity-50",
          )}
        >
          <RotateCw
            className={cn("size-3", library.isLoading && "animate-spin")}
            aria-hidden
          />
          Refresh
        </button>
      </header>

      {library.error && (
        <div
          role="alert"
          className={cn(
            "flex items-start gap-2 rounded-lg border border-danger/30",
            "bg-danger-subtle px-3 py-2 text-xs text-danger",
          )}
        >
          <AlertCircle className="mt-px size-3.5 shrink-0" aria-hidden />
          <span className="min-w-0 flex-1 break-words">{library.error}</span>
          <button
            type="button"
            onClick={library.clearError}
            aria-label="Dismiss error"
            className="shrink-0 rounded p-0.5 hover:opacity-70"
          >
            <X className="size-3" aria-hidden />
          </button>
        </div>
      )}

      <div
        onDragEnter={(event) => {
          event.preventDefault();
          dragDepth.current += 1;
          setIsDragging(true);
        }}
        onDragOver={(event) => event.preventDefault()}
        onDragLeave={(event) => {
          event.preventDefault();
          dragDepth.current -= 1;
          if (dragDepth.current <= 0) {
            dragDepth.current = 0;
            setIsDragging(false);
          }
        }}
        onDrop={(event) => {
          event.preventDefault();
          dragDepth.current = 0;
          setIsDragging(false);
          handleFiles(event.dataTransfer.files);
        }}
        className={cn(
          "rounded-xl border border-dashed p-8 text-center transition-colors",
          isDragging ? "border-accent bg-accent-subtle" : "border-border bg-surface",
          !isAuthenticated && "opacity-50",
        )}
      >
        <Upload className="mx-auto size-5 text-muted" aria-hidden />
        <p className="mt-2 text-sm">
          Drop files here, or{" "}
          <button
            type="button"
            disabled={!isAuthenticated}
            onClick={() => inputRef.current?.click()}
            className={cn(
              "font-medium text-accent underline underline-offset-2",
              "focus-visible:ring-2 focus-visible:ring-accent focus-visible:outline-none",
              "disabled:cursor-not-allowed",
            )}
          >
            browse
          </button>
        </p>
        <p className="mt-1 text-xs text-muted">
          {ACCEPTED_EXTENSIONS.join(", ")} · up to{" "}
          {Math.round(MAX_UPLOAD_BYTES / 1024 / 1024)} MB
        </p>
        <input
          ref={inputRef}
          type="file"
          multiple
          accept={ACCEPT_ATTRIBUTE}
          className="hidden"
          onChange={(event) => {
            handleFiles(event.target.files);
            event.target.value = "";
          }}
        />
      </div>

      {/* Pending approval queue */}
      {canManageOrg && pending.length > 0 && (
        <div className="rounded-xl border border-warning/30 bg-warning-subtle p-4">
          <h2 className="text-sm font-semibold text-warning">
            Pending approval ({pending.length})
          </h2>
          <p className="mt-1 text-xs text-muted">
            Member uploads that need your review before they become searchable.
          </p>
          <ul className="mt-3 space-y-2">
            {pending.map((row) => (
              <li key={row.id}>
                <PendingDocumentRow
                  row={row}
                  isApproving={library.approvingIds.has(row.id)}
                  isRejecting={library.rejectingIds.has(row.id)}
                  onApprove={() => void library.approve(row.id)}
                  onReject={() => void library.reject(row.id)}
                />
              </li>
            ))}
          </ul>
        </div>
      )}

      {activeUploads.length > 0 && (
        <ul className="space-y-2">
          {activeUploads.map((upload) => (
            <li key={upload.id}>
              <UploadCard upload={upload} onDismiss={library.dismissUpload} />
            </li>
          ))}
        </ul>
      )}

      <div className="overflow-hidden rounded-xl border border-border">
        {library.documents.length === 0 ? (
          <p className="px-4 py-10 text-center text-xs text-muted">
            {library.isLoading
              ? "Loading your document library…"
              : "No documents yet. Upload one to start asking questions about it."}
          </p>
        ) : (
          <ul className="divide-y divide-border">
            {library.documents.map((row) => (
              <li key={row.id}>
                <DocumentCard
                  row={row}
                  isDeleting={library.deletingIds.has(row.id)}
                  onRequestDelete={() => setPendingDelete(row)}
                  onReprocess={() => void library.reprocess(row.id)}
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
            and everything indexed from it — chunks, embeddings and the stored file —
            will be removed permanently. Answers will no longer be able to cite it.
            This cannot be undone.
          </>
        }
        isBusy={pendingDelete ? library.deletingIds.has(pendingDelete.id) : false}
        onConfirm={() => {
          if (pendingDelete) void library.remove(pendingDelete.id);
          setPendingDelete(null);
        }}
        onCancel={() => setPendingDelete(null)}
      />
    </div>
  );
}

function PendingDocumentRow({
  row,
  isApproving,
  isRejecting,
  onApprove,
  onReject,
}: {
  row: DocumentSummary;
  isApproving: boolean;
  isRejecting: boolean;
  onApprove: () => void;
  onReject: () => void;
}) {
  return (
    <div className="flex items-start gap-3 rounded-lg border border-border bg-background px-4 py-3">
      <FileText className="mt-0.5 size-4 shrink-0 text-muted" aria-hidden />
      <div className="min-w-0 flex-1">
        <p className="truncate text-sm font-medium" title={row.filename}>
          {row.filename}
        </p>
        <p className="mt-0.5 text-xs text-muted">
          {formatBytes(row.file_size)} · Uploaded {formatRelativeTime(row.created_at)}
        </p>
      </div>
      <div className="flex shrink-0 gap-1.5">
        <button
          type="button"
          onClick={onApprove}
          disabled={isApproving || isRejecting}
          className={cn(
            "flex items-center gap-1 rounded-md bg-success px-3 py-1.5 text-xs font-medium text-white",
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
            "flex items-center gap-1 rounded-md border border-border px-3 py-1.5 text-xs font-medium text-muted",
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

function UploadCard({
  upload,
  onDismiss,
}: {
  upload: UploadState;
  onDismiss: (id: string) => void;
}) {
  const isFailed = upload.phase === "failed";
  const percent = Math.round(upload.progress * 100);

  return (
    <div
      className={cn(
        "rounded-lg border px-3 py-2.5",
        isFailed ? "border-danger/30 bg-danger-subtle" : "border-border bg-surface",
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
            "transition-colors hover:text-foreground focus-visible:ring-2",
            "focus-visible:ring-accent focus-visible:outline-none",
          )}
        >
          <X className="size-3" aria-hidden />
        </button>
      </div>

      {upload.phase === "transferring" && (
        <div className="mt-2">
          <div
            role="progressbar"
            aria-valuenow={percent}
            aria-valuemin={0}
            aria-valuemax={100}
            aria-label={`Uploading ${upload.filename}`}
            className="h-1.5 overflow-hidden rounded-full bg-surface-raised"
          >
            <div
              className="h-full rounded-full bg-accent transition-[width] duration-200"
              style={{ width: `${percent}%` }}
            />
          </div>
          <p className="mt-1 text-[0.6875rem] text-muted">Uploading · {percent}%</p>
        </div>
      )}

      {upload.phase === "processing" && (
        <p className="mt-2 flex items-center gap-1.5 text-[0.6875rem] text-muted">
          <Loader2 className="size-3 animate-spin" aria-hidden />
          Processing — extracting, chunking and indexing
        </p>
      )}

      {isFailed && (
        <p className="mt-2 flex items-start gap-1.5 text-[0.6875rem] text-danger">
          <AlertCircle className="mt-px size-3 shrink-0" aria-hidden />
          <span>{upload.error ?? "Upload failed."}</span>
        </p>
      )}
    </div>
  );
}

function DocumentCard({
  row,
  isDeleting,
  onRequestDelete,
  onReprocess,
}: {
  row: DocumentSummary;
  isDeleting: boolean;
  onRequestDelete: () => void;
  onReprocess: () => void;
}) {
  const details = [
    formatBytes(row.file_size),
    formatRelativeTime(row.created_at),
  ].filter(Boolean);

  return (
    <div
      className={cn(
        "flex items-start gap-3 px-4 py-3 transition-colors hover:bg-surface-raised",
        isDeleting && "opacity-50",
      )}
    >
      <FileText className="mt-0.5 size-4 shrink-0 text-muted" aria-hidden />
      <div className="min-w-0 flex-1">
        <div className="flex flex-wrap items-center gap-2">
          <span className="truncate text-sm font-medium" title={row.filename}>
            {row.filename}
          </span>
          <StatusBadge status={row.status} />
        </div>
        <p className="mt-1 text-xs text-muted">{details.join(" · ")}</p>
        {row.status === "FAILED" && row.error_message && (
          <p className="mt-1 text-xs break-words text-danger">{row.error_message}</p>
        )}
      </div>

      <div className="flex shrink-0 items-center gap-1">
        {row.status === "FAILED" && (
          <button
            type="button"
            onClick={onReprocess}
            disabled={isDeleting}
            title="Try processing again"
            aria-label={`Retry processing ${row.filename}`}
            className={cn(
              "flex size-7 items-center justify-center rounded-md text-muted",
              "transition-colors hover:bg-surface hover:text-accent",
              "focus-visible:ring-2 focus-visible:ring-accent focus-visible:outline-none",
              "disabled:cursor-not-allowed disabled:opacity-50",
            )}
          >
            <RotateCw className="size-3.5" aria-hidden />
          </button>
        )}
        <button
          type="button"
          onClick={onRequestDelete}
          disabled={isDeleting}
          aria-label={`Delete ${row.filename}`}
          className={cn(
            "flex size-7 items-center justify-center rounded-md text-muted",
            "transition-colors hover:bg-surface hover:text-danger",
            "focus-visible:ring-2 focus-visible:ring-accent focus-visible:outline-none",
            "disabled:cursor-not-allowed disabled:opacity-50",
          )}
        >
          {isDeleting ? (
            <Loader2 className="size-3.5 animate-spin" aria-hidden />
          ) : (
            <Trash2 className="size-3.5" aria-hidden />
          )}
        </button>
      </div>
    </div>
  );
}
