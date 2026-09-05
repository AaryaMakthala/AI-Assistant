"use client";

/**
 * The chat pane: transcript, empty state, composer.
 *
 * Autoscroll follows the stream only while the user is already at the bottom. Forcing
 * scroll unconditionally would yank the view away from someone reading back through an
 * earlier answer every time a token arrives.
 */

import { Loader2, MessageSquareText, WifiOff } from "lucide-react";
import Image from "next/image";
import { useLayoutEffect, useRef, useState } from "react";
import { Composer } from "./composer";
import { MessageBubble } from "./message-bubble";
import type { Citation } from "@/lib/api";
import type { Turn } from "@/lib/hooks/use-chat";
import { cn } from "@/lib/utils";

/** How close to the bottom still counts as "at the bottom", in px. Absorbs sub-pixel
 * rounding and the height change of the line currently being written. */
const STICK_THRESHOLD = 120;

const SUGGESTIONS = [
  "What does our travel policy say about international flights?",
  "How many documents were uploaded this month?",
  "Summarize the refund policy and how many refunds we processed.",
];

export function ChatPane({
  turns,
  isStreaming,
  isLoadingHistory,
  transportError,
  activeChunkId,
  onSend,
  onStop,
  onSelectCitation,
  disabled,
}: {
  turns: Turn[];
  isStreaming: boolean;
  isLoadingHistory: boolean;
  transportError?: string;
  activeChunkId?: string;
  onSend: (message: string) => void;
  onStop: () => void;
  onSelectCitation: (citation: Citation) => void;
  disabled?: boolean;
}) {
  const scrollRef = useRef<HTMLDivElement>(null);
  const [isPinned, setIsPinned] = useState(true);

  // Layout effect, not effect: scrolling after paint shows one frame at the old offset,
  // which reads as a jitter on every token.
  useLayoutEffect(() => {
    if (!isPinned) return;
    const element = scrollRef.current;
    if (element) element.scrollTop = element.scrollHeight;
  }, [turns, isPinned]);

  // Re-pin whenever a new turn starts, so sending a message always brings the view back
  // down even if the user had scrolled up to read something.
  const handleSend = (message: string) => {
    setIsPinned(true);
    onSend(message);
  };

  const isEmpty = !turns.length && !isLoadingHistory;

  return (
    /* Layout: chat-bg.png is ONE continuous backdrop layer behind this whole
     * panel — the message list AND the floating composer share it, so there
     * is no visual seam where one ends and the other begins. Messages scroll
     * over it as individual frosted glass cards (message-bubble); the
     * composer floats as its own glass box near the bottom. A soft scrim
     * sits between image and content to keep the bare backdrop calm for
     * text. The sidebar / top bar stay flat solids outside this panel. */
    <div className="relative flex min-h-0 min-w-0 flex-1 flex-col overflow-hidden">
      <Image
        src="/chat-bg.png"
        alt=""
        aria-hidden
        fill
        priority
        sizes="100vw"
        draggable={false}
        style={{ objectFit: "cover" }}
        className="z-0"
      />
      {/* pointer-events-none: the scrim is decorative only and must never
       * sit above interactive content (the composer's textarea/send button
       * are static descendants of this panel and would otherwise paint and
       * hit-test beneath it — which made the input unclickable). */}
      <div
        className="pointer-events-none absolute inset-0 z-[1] bg-[rgba(9,15,12,0.35)]"
        aria-hidden
      />
      <div
        ref={scrollRef}
        onScroll={(event) => {
          const element = event.currentTarget;
          const distance =
            element.scrollHeight - element.scrollTop - element.clientHeight;
          setIsPinned(distance < STICK_THRESHOLD);
        }}
        className="chat-scroll relative z-10 min-h-0 flex-1 overflow-y-auto"
      >
        <div className="mx-auto w-full max-w-[780px] px-4 py-8">
          {isLoadingHistory && (
            <div className="flex items-center justify-center gap-2 py-16 text-sm text-muted">
              <Loader2 className="size-4 animate-spin" aria-hidden />
              Loading conversation…
            </div>
          )}

          {isEmpty && <EmptyState onSend={handleSend} disabled={disabled} />}

          {turns.length > 0 && (
            <div className="space-y-6">
              {turns.map((turn) => (
                <MessageBubble
                  key={turn.id}
                  turn={turn}
                  activeChunkId={activeChunkId}
                  onSelectCitation={onSelectCitation}
                />
              ))}
            </div>
          )}

          {transportError && (
            <div
              role="alert"
              className={cn(
                "mt-6 flex items-start gap-2 rounded-lg border border-danger/30",
                "bg-danger-subtle px-3 py-2.5 text-xs text-danger",
              )}
            >
              <WifiOff className="mt-px size-3.5 shrink-0" aria-hidden />
              <span>{transportError}</span>
            </div>
          )}
        </div>
      </div>

      <Composer
        onSend={handleSend}
        onStop={onStop}
        isStreaming={isStreaming}
        disabled={disabled}
      />
    </div>
  );
}

function EmptyState({
  onSend,
  disabled,
}: {
  onSend: (message: string) => void;
  disabled?: boolean;
}) {
  return (
    <div className="flex flex-col items-center py-16 text-center">
      <div className="flex size-11 items-center justify-center rounded-xl bg-accent-subtle text-accent">
        <MessageSquareText className="size-5" aria-hidden />
      </div>
      <h2 className="font-display mt-4 text-xl tracking-tight">
        Ask about anything your organization knows
      </h2>
      <p className="mt-1.5 max-w-md text-sm text-muted">
        Questions are routed automatically to your documents, your business
        database, or connected repositories — and answers cite where they came
        from.
      </p>

      <div className="mt-6 flex w-full max-w-lg flex-col gap-2">
        {SUGGESTIONS.map((suggestion) => (
          <button
            key={suggestion}
            type="button"
            disabled={disabled}
            onClick={() => onSend(suggestion)}
            className={cn(
              "rounded-xl border border-[rgba(255,255,255,0.12)] bg-[rgba(255,255,255,0.06)]",
              "px-3.5 py-2.5 text-left text-sm text-muted transition-colors",
              "hover:border-[rgba(255,255,255,0.2)] hover:bg-[rgba(255,255,255,0.1)]",
              "hover:text-foreground focus-visible:ring-2 focus-visible:ring-accent",
              "focus-visible:outline-none disabled:cursor-not-allowed disabled:opacity-50",
            )}
          >
            {suggestion}
          </button>
        ))}
      </div>
    </div>
  );
}