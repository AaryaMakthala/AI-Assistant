"use client";

/**
 * Ingestion status as a badge.
 *
 * The four backend statuses map to four visually distinct states rather than a spinner
 * plus text, because the library is scanned rather than read — a user looking for the one
 * document that failed should find it without reading every row.
 *
 * `ready` deliberately splits in two. A document that finished with no chunks is not
 * searchable, and showing it as "Ready" would promise something the retriever cannot
 * deliver (CLAUDE.md section 6: never imply confidence the answer doesn't have).
 */

import { AlertCircle, CheckCircle2, Clock, Loader2 } from "lucide-react";
import type { DocumentStatus } from "@/lib/api";
import { cn } from "@/lib/utils";

const STYLES: Record<DocumentStatus, string> = {
  PENDING: "bg-surface-raised text-muted",
  PROCESSING: "bg-warning-subtle text-warning",
  READY: "bg-accent-subtle text-accent",
  FAILED: "bg-danger-subtle text-danger",
};

const LABELS: Record<DocumentStatus, string> = {
  PENDING: "Queued",
  PROCESSING: "Processing",
  READY: "Ready",
  FAILED: "Failed",
};

const ICONS: Record<DocumentStatus, typeof Clock> = {
  PENDING: Clock,
  PROCESSING: Loader2,
  READY: CheckCircle2,
  FAILED: AlertCircle,
};

export function StatusBadge({
  status,
  chunkCount,
  className,
}: {
  status: DocumentStatus;
  /** Chunks stored. When a `ready` document has none, it is indexed but not searchable. */
  chunkCount?: number | null;
  className?: string;
}) {
  const isEmpty = status === "READY" && chunkCount === 0;
  const Icon = isEmpty ? AlertCircle : ICONS[status];

  return (
    <span
      className={cn(
        "inline-flex items-center gap-1 rounded-full px-1.5 py-0.5",
        "text-[0.6875rem] font-medium",
        isEmpty ? "bg-warning-subtle text-warning" : STYLES[status],
        className,
      )}
      title={
        isEmpty
          ? "Processed, but no searchable text was found in this file."
          : undefined
      }
    >
      <Icon
        className={cn("size-2.5", status === "PROCESSING" && "animate-spin")}
        aria-hidden
      />
      {isEmpty ? "No text" : LABELS[status]}
    </span>
  );
}
