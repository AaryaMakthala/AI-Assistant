"use client";

/**
 * The right-hand citations panel (CLAUDE.md section 6).
 *
 * Shows every source consulted for the active answer, with the cited ones marked. The
 * distinction is the point: "six passages were retrieved and the answer used two" is a
 * different and more useful statement than either number alone, and it lets a reader see
 * when an answer leaned on very little of what it was given.
 *
 * Excerpts are rendered as plain text, never as markdown or HTML. This is raw document
 * content — the untrusted material of CLAUDE.md 4.4 — and it is being shown for
 * verification, so it must appear exactly as stored rather than formatted.
 */

import { FileText, X } from "lucide-react";
import type { Source } from "@/lib/api";
import { cn } from "@/lib/utils";

export function SourcesPanel({
  sources,
  citedChunkIds,
  activeChunkId,
  onSelect,
  onClose,
}: {
  sources: Source[];
  citedChunkIds: Set<string>;
  activeChunkId?: string;
  onSelect: (chunkId: string) => void;
  onClose: () => void;
}) {
  return (
    <aside className="flex h-full w-80 shrink-0 flex-col border-l border-border bg-surface">
      <header className="flex items-center justify-between border-b border-border px-4 py-3">
        <div className="min-w-0">
          <h2 className="text-sm font-semibold">Sources</h2>
          <p className="mt-0.5 text-xs text-muted">
            {sources.length === 0
              ? "Nothing was consulted"
              : `${citedChunkIds.size} of ${sources.length} used in the answer`}
          </p>
        </div>
        <button
          type="button"
          onClick={onClose}
          aria-label="Close sources panel"
          className={cn(
            "flex size-7 shrink-0 items-center justify-center rounded-md text-muted",
            "transition-colors hover:bg-surface-raised hover:text-foreground",
            "focus-visible:ring-2 focus-visible:ring-accent focus-visible:outline-none",
          )}
        >
          <X className="size-4" aria-hidden />
        </button>
      </header>

      <div className="flex-1 overflow-y-auto p-3">
        {sources.length === 0 ? (
          <p className="px-1 py-8 text-center text-xs text-muted">
            This answer did not draw on any retrieved documents.
          </p>
        ) : (
          <ul className="space-y-2">
            {sources.map((source) => {
              const isCited = citedChunkIds.has(source.chunk_id);
              const isActive = source.chunk_id === activeChunkId;
              return (
                <li key={source.chunk_id}>
                  <button
                    type="button"
                    onClick={() => onSelect(source.chunk_id)}
                    aria-current={isActive}
                    className={cn(
                      "w-full rounded-lg border p-3 text-left transition-colors",
                      "focus-visible:ring-2 focus-visible:ring-accent focus-visible:outline-none",
                      isActive
                        ? "border-accent bg-accent-subtle"
                        : "border-border bg-background hover:border-accent/40",
                      !isCited && "opacity-70",
                    )}
                  >
                    <div className="flex items-center gap-1.5">
                      <span
                        className={cn(
                          "flex size-4 shrink-0 items-center justify-center rounded-full",
                          "text-[0.625rem] font-semibold",
                          isCited
                            ? "bg-accent text-accent-foreground"
                            : "bg-surface-raised text-muted",
                        )}
                      >
                        {source.number}
                      </span>
                      <FileText className="size-3 shrink-0 text-muted" aria-hidden />
                      <span className="min-w-0 flex-1 truncate text-xs font-medium">
                        {source.label}
                      </span>
                    </div>

                    <p className="mt-2 line-clamp-6 text-xs leading-relaxed whitespace-pre-wrap text-muted">
                      {source.excerpt}
                    </p>

                    <div className="mt-2 flex items-center gap-2 text-[0.6875rem] text-muted">
                      <span>{isCited ? "Cited" : "Retrieved, not cited"}</span>
                      <span aria-hidden>·</span>
                      {/* Retrieval similarity: how close the passage was to the question,
                          not how strongly it supports the sentence citing it. */}
                      <span>match {(source.score * 100).toFixed(0)}%</span>
                    </div>
                  </button>
                </li>
              );
            })}
          </ul>
        )}
      </div>
    </aside>
  );
}
