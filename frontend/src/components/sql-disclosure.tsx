"use client";

/**
 * The executed SQL behind a business-data answer — collapsed by default, expandable
 * (CLAUDE.md section 6: "transparency without clutter").
 *
 * Native `<details>` rather than state-driven show/hide: it is keyboard accessible and
 * findable by the browser's own in-page search without any of that being reimplemented.
 */

import { ChevronRight, Database } from "lucide-react";
import { cn } from "@/lib/utils";

export function SqlDisclosure({ sql }: { sql: string }) {
  if (!sql.trim()) return null;

  return (
    <details className="group mt-3 rounded-lg border border-border bg-surface-raised">
      <summary
        className={cn(
          "flex cursor-pointer list-none items-center gap-1.5 px-3 py-2 text-xs text-muted",
          "hover:text-foreground focus-visible:ring-2 focus-visible:ring-accent focus-visible:outline-none",
        )}
      >
        <ChevronRight
          className="size-3 transition-transform group-open:rotate-90"
          aria-hidden
        />
        <Database className="size-3" aria-hidden />
        Show the query that produced this
      </summary>
      <pre className="overflow-x-auto border-t border-border px-3 py-2 font-mono text-[0.75rem] leading-relaxed text-foreground">
        {sql}
      </pre>
    </details>
  );
}
