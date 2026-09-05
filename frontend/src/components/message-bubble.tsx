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
      /* The user's own message: a compact right-aligned speech bubble in the
       * light "own words" color (Claude/ChatGPT pattern), with the corner
       * nearest the avatar tapered (bottom-right) so it reads as a deliberate
       * bubble rather than a flat rectangle. Distinct from the wide frosted
       * assistant cards, and always high-contrast. */
      <div className="flex justify-end gap-3">
        <div
          className={cn(
            "max-w-[min(34rem,80%)] rounded-[18px] rounded-br-[6px] bg-accent",
            "px-[18px] py-3 text-sm leading-relaxed whitespace-pre-wrap",
            "font-medium text-accent-foreground shadow-[0_2px_8px_rgba(0,0,0,0.15)]",
          )}
        >
          {turn.content}
        </div>
        <div className="mt-1 flex size-7 shrink-0 items-center justify-center rounded-full bg-surface-raised text-muted">
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
    /* Assistant answer: one wide, spacious frosted glass card over the chat
     * background. The avatar floats to its left; everything else — routing
     * note, answer text, honesty markers, citations, model line — lives
     * inside the card. */
    <div className="flex gap-3">
      <div
        className={cn(
          "mt-1 flex size-7 shrink-0 items-center justify-center rounded-full",
          "border border-[rgba(255,255,255,0.12)] bg-[#0F1A15] text-xs font-semibold",
          "text-[rgba(245,243,236,0.85)]",
        )}
        aria-hidden
      >
        AI
      </div>

      <div className="min-w-0 flex-1">
        <div className="glass-message p-5">
          <div className="space-y-3">
            <RoutingIndicator
              routes={turn.routes}
              reason={turn.routeReason}
              steps={turn.steps}
              isStreaming={isStreaming}
              hasContent={hasContent}
            />

            {hasContent && (
              <div>
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

            {/* Sources sit as a slightly differentiated sub-section of the same
             * card: hairline top separator. Only rendered when citations exist. */}
            <CitationChips
              citations={turn.citations}
              activeChunkId={activeChunkId}
              onSelect={onSelectCitation}
              className="w-full border-t border-[rgba(255,255,255,0.08)] pt-3"
            />

            {turn.status === "complete" && turn.provider && (
              <p className="text-[0.6875rem] text-muted">
                {turn.model || turn.provider}
                {turn.usage ? ` · ${turn.usage.total_tokens} tokens` : ""}
              </p>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
