"use client";

/**
 * The document library: drag-and-drop upload, per-file progress, ingestion status.
 *
 * The progress bar tracks *transfer* and stops at 100% when the API accepts the file. What
 * follows is a Celery job with no progress signal, shown as an indeterminate "Processing"
 * state until the status endpoint reports `ready` or `failed`. Animating a fake bar through
 * that phase would imply knowledge the frontend does not have.
 *
 * Documents are split into Company and Mine. The split is presentation over rows the server
 * already filtered — RLS decides what arrives here, so this cannot reveal anything, and a
 * missing `visibility` would show a file in the wrong section rather than to the wrong
 * person. Only owners and admins can upload to Company; the API answers 403 regardless.
 */

import { AlertCircle, CheckCircle2, FileText, Loader2, RotateCw, Trash2, Upload, X } from "lucide-react";
import { useMemo, useRef, useState } from "react";
import { ConfirmDialog } from "./confirm-dialog";
import { StatusBadge } from "./status-badge";
import {
  ACCEPTED_EXTENSIONS,
  type UploadState,
} from "@/lib/hooks/use-documents";
import type { DocumentSummary, DocumentVisibility } from "@/lib/api";
import { cn, formatBytes } from "@/lib/utils";

const ACCEPT_ATTRIBUTE = ACCEPTED_EXTENSIONS.map((ext) => `.${ext}`).join(",");

type Section = "company" | "mine";

export function DocumentLibrary({
  documents,
  uploads,
  deletingIds,
  currentUserId,
  canManageOrg,
  onUpload,
  onDismissUpload,
  onDelete,
  onReprocess,
  disabled,
}: {
  documents: DocumentSummary[];
  uploads: UploadState[];
  deletingIds?: ReadonlySet<string>;
  /** Whose uploads count as "Mine". Undefined until /me resolves. */
  currentUserId?: string;
  /** Owners and admins only. Presentation; the server gates the action independently. */
  canManageOrg?: boolean;
  onUpload: (file: File, visibility: DocumentVisibility) => void;
  onDismissUpload: (id: string) => void;
  onDelete: (id: string) => void;
  onReprocess: (id: string) => void;
  disabled?: boolean;
}) {
  const [section, setSection] = useState<Section>("company");
  const [isDragging, setIsDragging] = useState(false);
  const [pendingDelete, setPendingDelete] = useState<DocumentSummary | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  // dragenter/dragleave fire for every child element crossed. Counting them means the
  // highlight survives moving across the zone's contents instead of flickering.
  const dragDepth = useRef(0);

  const { company, mine } = useMemo(
    () => ({
      company: documents.filter((row) => row.visibility === "org"),
      // An owner sees colleagues' personal files too, so "Mine" is by uploader rather than
      // by visibility — otherwise their own tab would fill with everyone else's uploads.
      mine: documents.filter(
        (row) => row.visibility === "personal" && row.uploaded_by === currentUserId,
      ),
    }),
    [documents, currentUserId],
  );

  const visible = section === "company" ? company : mine;
  // Uploading to Company publishes org-wide, which only an owner or admin may do.
  const canUploadHere = section === "mine" || Boolean(canManageOrg);
  const isUploadDisabled = disabled || !canUploadHere;

  const handleFiles = (files: FileList | null) => {
    if (!files || isUploadDisabled) return;
    const visibility: DocumentVisibility = section === "company" ? "org" : "personal";
    for (const file of Array.from(files)) onUpload(file, visibility);
  };

  // Uploads still in flight are listed separately above the library; once a document row
  // exists for a finished upload, the row is the better place to read its status from.
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
          "m-3 mt-0 rounded-lg border border-dashed p-4 text-center transition-colors",
          isDragging && !isUploadDisabled
            ? "border-accent bg-accent-subtle"
            : "border-border bg-background",
          isUploadDisabled && "opacity-50",
        )}
      >
        {canUploadHere ? (
          <>
            <Upload className="mx-auto size-4 text-muted" aria-hidden />
            <p className="mt-2 text-xs text-muted">
              Drop files here, or{" "}
              <button
                type="button"
                disabled={isUploadDisabled}
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
            <p className="mt-1 text-[0.6875rem] text-muted">
              {section === "company"
                ? "Visible to everyone in your organization"
                : "Visible only to you"}
            </p>
          </>
        ) : (
          <p className="text-[0.6875rem] text-muted">
            Only an owner or admin can add company documents. Use My Docs for your own
            files.
          </p>
        )}
        <input
          ref={inputRef}
          type="file"
          multiple
          accept={ACCEPT_ATTRIBUTE}
          className="hidden"
          onChange={(event) => {
            handleFiles(event.target.files);
            // Cleared so selecting the same file again still fires a change event.
            event.target.value = "";
          }}
        />
      </div>

      <div className="min-h-0 flex-1 overflow-y-auto px-3 pb-3">
        {activeUploads.length > 0 && (
          <ul className="mb-3 space-y-1.5">
            {activeUploads.map((upload) => (
              <li key={upload.id}>
                <UploadRow upload={upload} onDismiss={onDismissUpload} />
              </li>
            ))}
          </ul>
        )}

        {visible.length === 0 && activeUploads.length === 0 ? (
          <p className="px-1 py-6 text-center text-xs text-muted">
            {section === "company"
              ? "No company documents yet."
              : "No personal documents yet. Files you upload here stay private to you."}
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
          Ready to query
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

// Named `row`, not `document`: a prop called `document` shadows the DOM global inside
// this component, which turns any later use of it into a confusing type error.
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
  const isFailed = row.status === "failed";

  // Facts worth showing, in the order they become known. Assembled rather than
  // concatenated inline so a null field leaves no stray separator behind.
  const details = [
    row.page_count ? `${row.page_count} pages` : null,
    row.chunk_count ? `${row.chunk_count} chunks` : null,
    row.size_bytes ? formatBytes(row.size_bytes) : null,
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
          <StatusBadge status={row.status} chunkCount={row.chunk_count} />
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
