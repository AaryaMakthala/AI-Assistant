"use client";

/**
 * The Documents view: existing files and newly staged uploads share one
 * responsive grid of cards.
 *
 * - A "+ Add file" card is always the first tile. Selecting files adds staged
 *   cards right after it, so it is never replaced and files can keep being
 *   added.
 * - Each staged card carries its required description inline; the upload
 *   action bars stays pinned to the top of the grid and shows a live count
 *   ("Upload 3 files"), disabled while any file lacks a description.
 * - Already-uploaded documents render as the same style of card (icon, name,
 *   status, size, upload date) so staging and browsing are one view.
 *
 * Upload logic, validation, description requirement and API calls are shared
 * with the rest of the app — this is purely a presentation change.
 */

import { useCallback, useRef, useState } from "react";
import {
  AlertCircle,
  ArrowLeft,
  CheckCircle2,
  Download,
  FileSpreadsheet,
  FileText,
  Loader2,
  Plus,
  Upload,
  X,
} from "lucide-react";
import {
  ACCEPTED_EXTENSIONS,
  MAX_UPLOAD_BYTES,
  validateFile,
  type UploadState,
} from "@/lib/hooks/use-documents";
import { downloadDocument } from "@/lib/api";
import type { DocumentSummary } from "@/lib/api";
import { StatusBadge } from "./status-badge";
import { Button } from "./button";
import { DocumentDetailModal } from "./document-detail-modal";
import { cn, formatBytes, formatRelativeTime } from "@/lib/utils";

const ACCEPT_ATTRIBUTE = ACCEPTED_EXTENSIONS.map((ext) => `.${ext}`).join(",");

/** Shared glass-dark card shell for the grid. */
const CARD_BASE = cn(
  "rounded-xl border border-[rgba(255,255,255,0.08)]",
  "bg-[rgba(20,40,30,0.2)] backdrop-blur-md",
);

/** One file with its description, tracked in the upload list. */
interface UploadItem {
  id: string;
  file: File;
  description: string;
}

let itemCounter = 0;

interface UploadInterfaceProps {
  onUpload: (file: File, description?: string) => void;
  onDismissUpload: (id: string) => void;
  uploads: UploadState[];
  /** Existing documents in the workspace, shown as cards under the staged files. */
  documents: DocumentSummary[];
  onBack: () => void;
  disabled?: boolean;
  token?: string;
  workspaceId?: string;
}

export function UploadInterface({
  onUpload,
  onDismissUpload,
  uploads,
  documents,
  onBack,
  disabled,
  token,
  workspaceId,
}: UploadInterfaceProps) {
  const [items, setItems] = useState<UploadItem[]>([]);
  const [detailDoc, setDetailDoc] = useState<DocumentSummary | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const addFiles = useCallback((files: FileList | null) => {
    if (!files) return;
    const newItems: UploadItem[] = [];
    for (const file of Array.from(files)) {
      const rejection = validateFile(file);
      if (rejection) {
        // Skip invalid files — could add a toast here later
        continue;
      }
      itemCounter += 1;
      newItems.push({
        id: `item-${itemCounter}`,
        file,
        description: "",
      });
    }
    setItems((current) => [...current, ...newItems]);
  }, []);

  const updateDescription = useCallback((id: string, value: string) => {
    setItems((current) =>
      current.map((item) =>
        item.id === id ? { ...item, description: value } : item,
      ),
    );
  }, []);

  const removeItem = useCallback((id: string) => {
    setItems((current) => current.filter((item) => item.id !== id));
  }, []);

  const canUpload =
    !disabled &&
    items.length > 0 &&
    items.every((item) => item.description.trim().length > 0);

  const handleUpload = useCallback(() => {
    if (!canUpload) return;
    for (const item of items) {
      onUpload(item.file, item.description.trim());
    }
    setItems([]);
  }, [canUpload, items, onUpload]);

  const activeUploads = uploads.filter(
    (u) => u.phase !== "ready" || !u.documentId,
  );

  const stagedCount = items.length;
  const missingCount = items.filter(
    (item) => item.description.trim().length === 0,
  ).length;
  const isEmpty =
    stagedCount === 0 && activeUploads.length === 0 && documents.length === 0;
  const plural = (count: number) => (count === 1 ? "" : "s");

  return (
    <div className="flex h-full flex-col">
      {/* Header */}
      <div className="flex items-center gap-3 border-b border-border px-4 py-3">
        <Button
          variant="ghost"
          onClick={onBack}
          aria-label="Back to chat"
        >
          <ArrowLeft className="size-3" aria-hidden />
          Back to chat
        </Button>
        <div className="min-w-0">
          <h2 className="font-display text-sm">Documents</h2>
          <p className="truncate text-[0.6875rem] text-muted">
            {ACCEPTED_EXTENSIONS.map((ext) => ext.toUpperCase()).join(" · ")} · up to{" "}
            {Math.round(MAX_UPLOAD_BYTES / 1024 / 1024)} MB · each file requires a
            description
          </p>
        </div>
      </div>

      {/* Scrollable card grid */}
      <div className="min-h-0 flex-1 overflow-y-auto px-4 py-4">
        {/* Contextual upload action bar — sticky while the grid scrolls */}
        {(stagedCount > 0 || activeUploads.length > 0) && (
          <div
            className={cn(
              "sticky top-0 z-10 mb-4 flex flex-wrap items-center gap-3",
              "rounded-xl border border-border bg-[rgba(12,20,16,0.92)] px-4 py-3",
              "shadow-lg shadow-black/20 backdrop-blur-md",
            )}
          >
            <div className="min-w-0 flex-1">
              {stagedCount > 0 ? (
                <>
                  <p className="text-sm font-medium">
                    {stagedCount} file{plural(stagedCount)} staged
                  </p>
                  <p className="mt-0.5 text-[0.6875rem] text-muted">
                    {canUpload || missingCount === 0
                      ? "Ready to upload — every file has a description."
                      : `Add a description for ${missingCount} file${plural(missingCount)} to upload.`}
                  </p>
                </>
              ) : (
                <>
                  <p className="flex items-center gap-1.5 text-sm font-medium">
                    <Loader2 className="size-3 animate-spin" aria-hidden />
                    {activeUploads.length} file{plural(activeUploads.length)} uploading…
                  </p>
                  <p className="mt-0.5 text-[0.6875rem] text-muted">
                    Ingestion runs as each upload completes.
                  </p>
                </>
              )}
            </div>

            {stagedCount > 0 && (
              <Button
                variant="primary"
                onClick={handleUpload}
                disabled={!canUpload}
              >
                <Upload className="size-3.5" aria-hidden />
                Upload {stagedCount} file{plural(stagedCount)}
              </Button>
            )}
          </div>
        )}

        {isEmpty && <p className="mb-3 text-xs text-muted">Add files to your workspace</p>}

        <div className="grid grid-cols-[repeat(auto-fill,minmax(220px,1fr))] gap-4">
          {/* "+ Add file" tile — always first, never replaced */}
          <AddFileCard
            onClick={() => inputRef.current?.click()}
            disabled={disabled}
          />

          {/* Staged files, next to the add card */}
          {items.map((item) => (
            <StagedFileCard
              key={item.id}
              item={item}
              onChangeDescription={updateDescription}
              onRemove={removeItem}
            />
          ))}

          {/* Uploads in flight */}
          {activeUploads.map((upload) => (
            <UploadStatusCard
              key={upload.id}
              upload={upload}
              onDismiss={() => onDismissUpload(upload.id)}
            />
          ))}

          {/* Existing documents */}
          {documents.map((row) => (
            <ExistingFileCard
              key={row.id}
              row={row}
              token={token}
              workspaceId={workspaceId}
              onOpen={() => setDetailDoc(row)}
            />
          ))}
        </div>

        <input
          ref={inputRef}
          type="file"
          multiple
          accept={ACCEPT_ATTRIBUTE}
          className="hidden"
          onChange={(e) => {
            addFiles(e.target.files);
            e.target.value = "";
          }}
        />
      </div>

      <DocumentDetailModal
        document={detailDoc}
        token={token}
        workspaceId={workspaceId}
        onClose={() => setDetailDoc(null)}
      />
    </div>
  );
}

/** The persistent "+ Add file" tile that opens the file picker. */
function AddFileCard({
  onClick,
  disabled,
}: {
  onClick: () => void;
  disabled?: boolean;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      aria-label="Add file"
      className={cn(
        "group flex min-h-[220px] flex-col items-center justify-center gap-2.5",
        "rounded-xl border border-dashed border-[rgba(255,255,255,0.15)]",
        "bg-[rgba(255,255,255,0.04)] backdrop-blur-md transition-colors",
        "hover:border-[rgba(245,243,236,0.45)] hover:bg-[rgba(255,255,255,0.08)]",
        "focus-visible:ring-2 focus-visible:ring-accent focus-visible:outline-none",
        "disabled:cursor-not-allowed disabled:opacity-50",
      )}
    >
      <span
        className={cn(
          "flex size-11 items-center justify-center rounded-full bg-surface-raised",
          "text-muted transition-colors group-hover:bg-accent-subtle group-hover:text-accent",
        )}
      >
        <Plus className="size-5" aria-hidden />
      </span>
      <span
        className={cn(
          "text-xs font-medium text-muted transition-colors group-hover:text-foreground",
        )}
      >
        Add file
      </span>
      <span className="px-3 text-center text-[0.6875rem] text-muted/70">
        {ACCEPTED_EXTENSIONS.map((ext) => ext.toUpperCase()).join(" · ")}
      </span>
    </button>
  );
}

/** A staged (not yet uploaded) file with its required description inline. */
function StagedFileCard({
  item,
  onChangeDescription,
  onRemove,
}: {
  item: UploadItem;
  onChangeDescription: (id: string, value: string) => void;
  onRemove: (id: string) => void;
}) {
  const missing = item.description.trim().length === 0;

  return (
    <div className={cn(CARD_BASE, "flex flex-col p-3")}>
      <div className="flex items-start gap-2.5">
        <span className="flex size-10 shrink-0 items-center justify-center rounded-lg bg-surface-raised">
          <FileTypeIcon name={item.file.name} className="size-5 text-muted" />
        </span>
        <div className="min-w-0 flex-1">
          <p className="truncate text-xs font-medium" title={item.file.name}>
            {item.file.name}
          </p>
          <p className="mt-0.5 text-[0.6875rem] text-muted">
            {formatBytes(item.file.size)} · Not uploaded
          </p>
        </div>
        <button
          type="button"
          onClick={() => onRemove(item.id)}
          aria-label={`Remove ${item.file.name}`}
          className={cn(
            "flex size-5 shrink-0 items-center justify-center rounded text-muted",
            "transition-colors hover:bg-surface-raised hover:text-foreground",
            "focus-visible:ring-2 focus-visible:ring-accent focus-visible:outline-none",
          )}
        >
          <X className="size-3" aria-hidden />
        </button>
      </div>

      <div className="mt-3">
        <label
          htmlFor={`desc-${item.id}`}
          className="mb-1 block text-[0.6875rem] text-muted"
        >
          Description <span className="text-danger">*</span>
        </label>
        <input
          id={`desc-${item.id}`}
          type="text"
          value={item.description}
          onChange={(e) => onChangeDescription(item.id, e.target.value)}
          placeholder="Add a description..."
          aria-invalid={missing}
          className={cn(
            "w-full rounded-md border bg-[rgba(12,20,16,0.5)] px-2.5 py-1.5",
            "text-xs text-foreground placeholder:text-muted/50",
            "focus:border-accent focus:outline-none focus:ring-1 focus:ring-accent",
            missing ? "border-warning/50" : "border-[rgba(255,255,255,0.08)]",
          )}
        />
        {missing && (
          <p className="mt-1 text-[0.6875rem] text-warning">
            Description required
          </p>
        )}
      </div>
    </div>
  );
}

/** An upload in flight: transfer progress, processing, success, or failure. */
function UploadStatusCard({
  upload,
  onDismiss,
}: {
  upload: UploadState;
  onDismiss: () => void;
}) {
  const isFailed = upload.phase === "failed";
  const percent = Math.round(upload.progress * 100);

  return (
    <div
      className={cn(
        CARD_BASE,
        "flex flex-col p-3",
        isFailed && "border-danger/30 bg-danger-subtle",
      )}
    >
      <div className="flex items-start gap-2.5">
        <span className="flex size-10 shrink-0 items-center justify-center rounded-lg bg-surface-raised">
          <FileTypeIcon
            name={upload.filename}
            className="size-5 text-muted"
          />
        </span>
        <div className="min-w-0 flex-1">
          <p className="truncate text-xs font-medium" title={upload.filename}>
            {upload.filename}
          </p>
          <p className="mt-0.5 text-[0.6875rem] text-muted">
            {formatBytes(upload.sizeBytes)}
          </p>
        </div>
        <button
          type="button"
          onClick={onDismiss}
          aria-label={`Dismiss ${upload.filename}`}
          className={cn(
            "flex size-5 shrink-0 items-center justify-center rounded text-muted",
            "transition-colors hover:bg-surface-raised hover:text-foreground",
            "focus-visible:ring-2 focus-visible:ring-accent focus-visible:outline-none",
          )}
        >
          <X className="size-3" aria-hidden />
        </button>
      </div>

      {upload.phase === "transferring" && (
        <div className="mt-3">
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
          <p className="mt-1 text-[0.6875rem] text-muted">
            Uploading · {percent}%
          </p>
        </div>
      )}

      {upload.phase === "processing" && (
        <p className="mt-3 flex items-center gap-1.5 text-[0.6875rem] text-muted">
          <Loader2 className="size-3 animate-spin" aria-hidden />
          Processing — extracting and indexing
        </p>
      )}

      {upload.phase === "ready" && !upload.error && (
        <p className="mt-3 flex items-center gap-1.5 text-[0.6875rem] text-success">
          <CheckCircle2 className="size-3" aria-hidden />
          Ready
        </p>
      )}

      {isFailed && (
        <p className="mt-3 flex items-start gap-1.5 text-[0.6875rem] text-danger">
          <AlertCircle className="mt-px size-3 shrink-0" aria-hidden />
          <span className="min-w-0 flex-1 break-words">
            {upload.error ?? "Upload failed."}
          </span>
        </p>
      )}
    </div>
  );
}

/** An already-uploaded document, displayed with its stored metadata. */
function ExistingFileCard({
  row,
  token,
  workspaceId,
  onOpen,
}: {
  row: DocumentSummary;
  token?: string;
  workspaceId?: string;
  onOpen: () => void;
}) {
  const isFailed = row.status === "FAILED";
  const [isDownloading, setIsDownloading] = useState(false);
  const details = [
    formatBytes(row.file_size),
    formatRelativeTime(row.created_at),
  ].filter(Boolean);

  const handleDownload = async (event: React.MouseEvent) => {
    event.stopPropagation();
    if (!token) return;
    setIsDownloading(true);
    try {
      await downloadDocument(row.id, row.filename, { token, workspaceId });
    } catch {
      // Swallow — the detail modal offers a surfaced retry path.
    } finally {
      setIsDownloading(false);
    }
  };

  return (
    <div
      onClick={onOpen}
      role="button"
      tabIndex={0}
      onKeyDown={(event) => {
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          onOpen();
        }
      }}
      aria-label={`Open details for ${row.filename}`}
      className={cn(
        CARD_BASE,
        "group relative flex cursor-pointer flex-col p-3",
        "transition-all duration-200 hover:-translate-y-0.5",
        "hover:border-[rgba(255,255,255,0.18)] hover:shadow-lg hover:shadow-black/20",
        "focus-visible:ring-2 focus-visible:ring-accent focus-visible:outline-none",
        isFailed && "border-danger/30 bg-danger-subtle",
      )}
    >
      <div className="flex items-start gap-2.5">
        <span className="flex size-10 shrink-0 items-center justify-center rounded-lg bg-surface-raised">
          <FileTypeIcon name={row.filename} className="size-5 text-muted" />
        </span>
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-1.5">
            <p
              className="min-w-0 flex-1 truncate text-xs font-medium"
              title={row.filename}
            >
              {row.filename}
            </p>
            <StatusBadge status={row.status} className="shrink-0" />
          </div>
          <p className="mt-0.5 text-[0.6875rem] text-muted">
            {details.join(" · ")}
          </p>
        </div>

        <button
          type="button"
          onClick={handleDownload}
          disabled={!token || isDownloading}
          aria-label={`Download ${row.filename}`}
          title="Download"
          className={cn(
            "absolute top-2 right-2 flex size-7 items-center justify-center rounded-md",
            "border border-[rgba(255,255,255,0.12)] bg-[rgba(12,20,16,0.6)] text-muted",
            "opacity-0 transition-opacity group-hover:opacity-100",
            "hover:border-[rgba(255,255,255,0.3)] hover:text-foreground",
            "focus-visible:opacity-100 focus-visible:ring-2 focus-visible:ring-accent focus-visible:outline-none",
            "disabled:cursor-not-allowed disabled:opacity-40",
          )}
        >
          {isDownloading ? (
            <Loader2 className="size-3.5 animate-spin" aria-hidden />
          ) : (
            <Download className="size-3.5" aria-hidden />
          )}
        </button>
      </div>

      {row.description && (
        <p className="mt-2 line-clamp-2 text-[0.6875rem] text-muted">
          {row.description}
        </p>
      )}
      {isFailed && row.error_message && (
        <p className="mt-2 text-[0.6875rem] break-words text-danger">
          {row.error_message}
        </p>
      )}
    </div>
  );
}

/** File-type icon: spreadsheet formats get a sheet glyph, everything else a document. */
function FileTypeIcon({
  name,
  className,
}: {
  name: string;
  className?: string;
}) {
  const dot = name.lastIndexOf(".");
  const ext = dot === -1 ? "" : name.slice(dot + 1).toLowerCase();
  const Icon = ext === "csv" || ext === "xlsx" ? FileSpreadsheet : FileText;
  return <Icon className={className} aria-hidden />;
}