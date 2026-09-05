"use client";

/**
 * Dedicated upload interface: a file list where every file requires a description
 * before upload is allowed.
 *
 * Replaces the inline drag-and-drop upload in the sidebar and documents page.
 * The upload button is disabled until every file has a non-empty description.
 */

import { useCallback, useRef, useState } from "react";
import { ArrowLeft, FileText, Plus, Upload, X } from "lucide-react";
import {
  ACCEPTED_EXTENSIONS,
  MAX_UPLOAD_BYTES,
  validateFile,
  type UploadState,
} from "@/lib/hooks/use-documents";
import { cn } from "@/lib/utils";

const ACCEPT_ATTRIBUTE = ACCEPTED_EXTENSIONS.map((ext) => `.${ext}`).join(",");

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
  onBack: () => void;
  disabled?: boolean;
}

export function UploadInterface({
  onUpload,
  onDismissUpload,
  uploads,
  onBack,
  disabled,
}: UploadInterfaceProps) {
  const [items, setItems] = useState<UploadItem[]>([]);
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

  return (
    <div className="flex h-full flex-col">
      {/* Header */}
      <div className="flex items-center gap-3 border-b border-border px-4 py-3">
        <button
          type="button"
          onClick={onBack}
          className={cn(
            "flex items-center gap-1.5 rounded-md px-2 py-1 text-xs text-muted",
            "transition-colors hover:text-foreground",
            "focus-visible:ring-2 focus-visible:ring-accent focus-visible:outline-none",
          )}
        >
          <ArrowLeft className="size-3" aria-hidden />
          Back to chat
        </button>
        <h2 className="font-display text-sm font-semibold">Upload Documents</h2>
      </div>

      {/* File list */}
      <div className="min-h-0 flex-1 overflow-y-auto px-4 py-3">
        {items.length === 0 && activeUploads.length === 0 && (
          <div className="flex flex-col items-center justify-center py-12 text-center">
            <Upload className="size-8 text-muted" aria-hidden />
            <p className="mt-3 text-sm text-muted">
              Add files to upload to your workspace.
            </p>
            <p className="mt-1 text-xs text-muted">
              Each file requires a description before uploading.
            </p>
          </div>
        )}

        {items.length > 0 && (
          <ul className="space-y-3">
            {items.map((item) => (
              <li
                key={item.id}
                className="rounded-lg border border-border bg-surface p-3"
              >
                <div className="flex items-center gap-2">
                  <FileText
                    className="size-4 shrink-0 text-muted"
                    aria-hidden
                  />
                  <span className="min-w-0 flex-1 truncate text-sm font-medium">
                    {item.file.name}
                  </span>
                  <span className="shrink-0 text-xs text-muted">
                    {formatBytes(item.file.size)}
                  </span>
                  <button
                    type="button"
                    onClick={() => removeItem(item.id)}
                    aria-label={`Remove ${item.file.name}`}
                    className={cn(
                      "flex size-5 shrink-0 items-center justify-center rounded text-muted",
                      "transition-colors hover:text-foreground",
                      "focus-visible:ring-2 focus-visible:ring-accent focus-visible:outline-none",
                    )}
                  >
                    <X className="size-3" aria-hidden />
                  </button>
                </div>
                <div className="mt-2">
                  <label
                    htmlFor={`desc-${item.id}`}
                    className="mb-1 block text-xs text-muted"
                  >
                    Description <span className="text-danger">*</span>
                  </label>
                  <input
                    id={`desc-${item.id}`}
                    type="text"
                    value={item.description}
                    onChange={(e) =>
                      updateDescription(item.id, e.target.value)
                    }
                    placeholder="e.g. Company employee policies"
                    className={cn(
                      "w-full rounded-md border bg-background px-2.5 py-1.5 text-sm text-foreground",
                      "placeholder:text-muted focus:border-accent focus:outline-none focus:ring-1 focus:ring-accent",
                      item.description.trim().length === 0
                        ? "border-warning/50"
                        : "border-border",
                    )}
                  />
                </div>
              </li>
            ))}
          </ul>
        )}

        {/* Active uploads (in-flight) */}
        {activeUploads.length > 0 && (
          <ul className="mt-4 space-y-2">
            {activeUploads.map((upload) => (
              <li
                key={upload.id}
                className={cn(
                  "rounded-lg border px-3 py-2",
                  upload.phase === "failed"
                    ? "border-danger/30 bg-danger-subtle"
                    : "border-border bg-surface",
                )}
              >
                <div className="flex items-center gap-2">
                  <span className="min-w-0 flex-1 truncate text-xs font-medium">
                    {upload.filename}
                  </span>
                  <button
                    type="button"
                    onClick={() => onDismissUpload(upload.id)}
                    aria-label={`Dismiss ${upload.filename}`}
                    className="flex size-5 shrink-0 items-center justify-center rounded text-muted hover:text-foreground"
                  >
                    <X className="size-3" aria-hidden />
                  </button>
                </div>
                {upload.phase === "transferring" && (
                  <div className="mt-1.5">
                    <div className="h-1 overflow-hidden rounded-full bg-surface-raised">
                      <div
                        className="h-full rounded-full bg-accent transition-[width] duration-200"
                        style={{
                          width: `${Math.round(upload.progress * 100)}%`,
                        }}
                      />
                    </div>
                    <p className="mt-1 text-[0.6875rem] text-muted">
                      Uploading · {Math.round(upload.progress * 100)}%
                    </p>
                  </div>
                )}
                {upload.phase === "processing" && (
                  <p className="mt-1.5 text-[0.6875rem] text-muted">
                    Processing…
                  </p>
                )}
                {upload.phase === "ready" && !upload.error && (
                  <p className="mt-1.5 text-[0.6875rem] text-success">
                    Ready
                  </p>
                )}
                {upload.phase === "failed" && (
                  <p className="mt-1.5 text-[0.6875rem] text-danger">
                    {upload.error ?? "Upload failed."}
                  </p>
                )}
              </li>
            ))}
          </ul>
        )}
      </div>

      {/* Footer with Add File + Upload buttons */}
      <div className="flex items-center gap-2 border-t border-border px-4 py-3">
        <button
          type="button"
          onClick={() => inputRef.current?.click()}
          disabled={disabled}
          className={cn(
            "flex items-center gap-1.5 rounded-md border border-border px-3 py-2 text-xs font-medium text-muted",
            "transition-colors hover:text-foreground",
            "focus-visible:ring-2 focus-visible:ring-accent focus-visible:outline-none",
            "disabled:cursor-not-allowed disabled:opacity-50",
          )}
        >
          <Plus className="size-3.5" aria-hidden />
          Add File
        </button>
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

        <div className="flex-1" />

        <button
          type="button"
          onClick={handleUpload}
          disabled={!canUpload}
          className={cn(
            "flex items-center gap-1.5 rounded-md bg-accent px-4 py-2 text-xs font-medium text-accent-foreground",
            "transition-opacity hover:opacity-90",
            "focus-visible:ring-2 focus-visible:ring-accent focus-visible:outline-none",
            "disabled:cursor-not-allowed disabled:opacity-50",
          )}
        >
          <Upload className="size-3.5" aria-hidden />
          Upload{items.length > 0 ? ` (${items.length})` : ""}
        </button>
      </div>
    </div>
  );
}

function formatBytes(bytes: number): string {
  if (bytes === 0) return "0 B";
  const units = ["B", "KB", "MB", "GB"];
  const i = Math.min(
    Math.floor(Math.log(bytes) / Math.log(1024)),
    units.length - 1,
  );
  const value = bytes / Math.pow(1024, i);
  return `${i === 0 ? value : value.toFixed(1)} ${units[i]}`;
}
