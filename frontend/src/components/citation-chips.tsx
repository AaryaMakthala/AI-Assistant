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

  // Group cited chunks by document so a single document that several chunks
  // came from appears once, not N times.  This is a presentation-layer
  // collapse only: `citations` keeps every per-chunk entry intact for the
  // Sources panel and for page-level verification; only the chips repeat
  // a document name once.
  const groups = new Map<string, Citation[]>();
  for (const citation of citations) {
    const key = citation.document_id;
    const list = groups.get(key);
    if (list) {
      list.push(citation);
    } else {
      groups.set(key, [citation]);
    }
  }

  return (
    <div className={cn("mt-3 flex flex-wrap items-center gap-1.5", className)}>
      <span className="mr-0.5 text-xs text-muted">Sources</span>
      {[...groups.values()].map((group) => {
        // The representative chip: the first chunk of the document.  Clicking
        // opens the panel at that chunk; all the group's chunks stay selectable
        // there.
        const representative = group[0];
        const isActive = activeChunkId != null && group.some(
          (citation) => citation.chunk_id === activeChunkId,
        );
        const count = group.length;
        const label =
          count > 1
            ? `${representative.filename} · ${count} passages`
            : representative.label;
        return (
          <button
            key={representative.chunk_id}
            type="button"
            onClick={() => onSelect(representative)}
            title={group.map((c) => c.excerpt).join("\n\n")}
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
              {representative.number}
            </span>
            <FileText className="size-3 shrink-0" aria-hidden />
            <span className="truncate font-mono text-[0.6875rem]">{label}</span>
          </button>
        );
      })}
    </div>
  );
}
