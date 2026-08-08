"use client";

/**
 * Source chips under an answer — `refund_policy.pdf · page 4` (CLAUDE.md section 6).
 *
 * Clicking one opens it in the right-hand panel rather than navigating away: the point of
 * a citation is to check it against the sentence it supports, which requires both on
 * screen at once.
 */

import { FileText } from "lucide-react";
import type { Citation } from "@/lib/api";
import { cn } from "@/lib/utils";

export function CitationChips({
  citations,
  activeChunkId,
  onSelect,
  className,
}: {
  citations: Citation[];
  activeChunkId?: string;
  onSelect: (citation: Citation) => void;
  className?: string;
}) {
  if (!citations.length) return null;

  return (
    <div className={cn("mt-3 flex flex-wrap items-center gap-1.5", className)}>
      <span className="mr-0.5 text-xs text-muted">Sources</span>
      {citations.map((citation) => {
        const isActive = citation.chunk_id === activeChunkId;
        return (
          <button
            key={citation.chunk_id}
            type="button"
            onClick={() => onSelect(citation)}
            title={citation.excerpt}
            aria-pressed={isActive}
            className={cn(
              "inline-flex max-w-[16rem] items-center gap-1.5 rounded-full border px-2.5 py-1",
              "text-xs transition-colors",
              "focus-visible:ring-2 focus-visible:ring-accent focus-visible:outline-none",
              isActive
                ? "border-accent bg-accent-subtle text-accent"
                : "border-border bg-surface text-muted hover:border-accent/50 hover:text-foreground",
            )}
          >
            <span
              className={cn(
                "flex size-4 shrink-0 items-center justify-center rounded-full text-[0.625rem] font-semibold",
                isActive
                  ? "bg-accent text-accent-foreground"
                  : "bg-surface-raised text-muted",
              )}
            >
              {citation.number}
            </span>
            <FileText className="size-3 shrink-0" aria-hidden />
            <span className="truncate font-mono text-[0.6875rem]">{citation.label}</span>
          </button>
        );
      })}
    </div>
  );
}
