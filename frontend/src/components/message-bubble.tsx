"use client";

/**
 * One turn in the transcript.
 *
 * The honesty rules from CLAUDE.md section 6 are enforced here, since this is the only
 * place that knows a turn's full state:
 *
 * - A failed turn shows its error *and* whatever text arrived before the failure. Hiding
 *   the partial would misrepresent what the user saw; showing it unlabelled would present
 *   a fragment as an answer.
 * - A turn that stopped early is marked incomplete, including one reloaded from history.
 * - An answer built on no evidence is marked ungrounded rather than styled identically to
 *   a cited one — "never let the UI imply confidence the answer doesn't have."
 */

import { AlertTriangle, Info, User } from "lucide-react";
import { CitationChips } from "./citation-chips";
import { Markdown } from "./markdown";
import { RoutingIndicator } from "./routing-indicator";
import { SqlDisclosure } from "./sql-disclosure";
import type { Citation } from "@/lib/api";
import type { Turn } from "@/lib/hooks/use-chat";
import { cn } from "@/lib/utils";

export function MessageBubble({
  turn,
  activeChunkId,
  onSelectCitation,
}: {
  turn: Turn;
  activeChunkId?: string;
  onSelectCitation: (citation: Citation) => void;
}) {
  if (turn.role === "user") {
    return (
      <div className="flex justify-end gap-3">
        <div
          className={cn(
            "max-w-[min(42rem,85%)] rounded-2xl rounded-br-sm bg-accent",
            "px-4 py-2.5 text-sm leading-relaxed whitespace-pre-wrap text-accent-foreground",
          )}
        >
          {turn.content}
        </div>
        <div className="mt-0.5 flex size-7 shrink-0 items-center justify-center rounded-full bg-surface-raised text-muted">
          <User className="size-3.5" aria-hidden />
        </div>
      </div>
    );
  }

  const isStreaming = turn.status === "streaming";
  const hasContent = Boolean(turn.content);
  const showUngrounded =
    turn.status === "complete" && turn.grounded === false && hasContent;

  return (
    <div className="flex gap-3">
      <div
        className={cn(
          "mt-0.5 flex size-7 shrink-0 items-center justify-center rounded-full",
          "bg-accent-subtle text-xs font-semibold text-accent",
        )}
        aria-hidden
      >
        AI
      </div>

      <div className="min-w-0 flex-1 space-y-2">
        <RoutingIndicator
          routes={turn.routes}
          reason={turn.routeReason}
          steps={turn.steps}
          isStreaming={isStreaming}
          hasContent={hasContent}
        />

        {hasContent && (
          <div className="max-w-[min(48rem,100%)]">
            <Markdown content={turn.content} />
            {isStreaming && (
              <span
                className="caret-blink ml-0.5 inline-block h-4 w-[2px] translate-y-0.5 bg-foreground"
                aria-hidden
              />
            )}
          </div>
        )}

        {turn.status === "failed" && (
          <div
            role="alert"
            className={cn(
              "flex items-start gap-2 rounded-lg border border-danger/30",
              "bg-danger-subtle px-3 py-2 text-xs text-danger",
            )}
          >
            <AlertTriangle className="mt-px size-3.5 shrink-0" aria-hidden />
            <span>
              {turn.error ?? "Something went wrong generating this answer."}
              {turn.incomplete && hasContent && (
                <> The text above is what arrived before the failure.</>
              )}
            </span>
          </div>
        )}

        {turn.status === "complete" && turn.incomplete && hasContent && (
          <div className="flex items-center gap-1.5 text-xs text-warning">
            <Info className="size-3.5 shrink-0" aria-hidden />
            This answer stopped early and may be incomplete.
          </div>
        )}

        {showUngrounded && (
          <div className="flex items-center gap-1.5 text-xs text-muted">
            <Info className="size-3.5 shrink-0" aria-hidden />
            No sources were consulted for this reply.
          </div>
        )}

        <SqlDisclosure sql={turn.sqlQuery} />

        <CitationChips
          citations={turn.citations}
          activeChunkId={activeChunkId}
          onSelect={onSelectCitation}
        />

        {turn.status === "complete" && turn.provider && (
          <p className="pt-0.5 text-[0.6875rem] text-muted">
            {turn.model || turn.provider}
            {turn.usage ? ` · ${turn.usage.total_tokens} tokens` : ""}
          </p>
        )}
      </div>
    </div>
  );
}
