"use client";

/** The message composer. Enter sends, Shift+Enter inserts a newline. */

import { ArrowUp, Square } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { cn } from "@/lib/utils";

const MAX_LENGTH = 8000; // Matches the backend's ChatRequest bound.
const MAX_ROWS_PX = 200;

export function Composer({
  onSend,
  onStop,
  isStreaming,
  disabled,
  placeholder = "Ask about your documents, business data, or code…",
}: {
  onSend: (message: string) => void;
  onStop: () => void;
  isStreaming: boolean;
  disabled?: boolean;
  placeholder?: string;
}) {
  const [value, setValue] = useState("");
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // Grow with the content up to a cap. Height is reset to `auto` first because
  // `scrollHeight` never shrinks below the element's current height.
  useEffect(() => {
    const textarea = textareaRef.current;
    if (!textarea) return;
    textarea.style.height = "auto";
    textarea.style.height = `${Math.min(textarea.scrollHeight, MAX_ROWS_PX)}px`;
  }, [value]);

  const submit = () => {
    const message = value.trim();
    if (!message || isStreaming || disabled) return;
    setValue("");
    onSend(message);
  };

  return (
    /* The composer floats as a glass box over the SAME continuous chat
     * backdrop the messages sit on (chat-pane draws one image layer behind
     * this whole panel) — no solid bar, no seam. The wrapper is transparent
     * and margins the box off the panel's bottom edge so it reads as
     * floating. It is relative z-10: the content is static HTML and would
     * otherwise paint beneath the panel's fixed backdrop/scrim layers (both
     * positioned with positive z-index), which would swallow pointer
     * events meant for the textarea and send button. */
    <div className="relative z-10 px-4 pt-1 pb-6">
      <div
        className={cn(
          "glass-composer mx-auto flex max-w-[780px] items-end gap-2 p-2",
        )}
      >
        <textarea
          ref={textareaRef}
          value={value}
          rows={1}
          maxLength={MAX_LENGTH}
          disabled={disabled}
          placeholder={placeholder}
          aria-label="Message"
          onChange={(event) => setValue(event.target.value)}
          onKeyDown={(event) => {
            // Not while composing: an IME uses Enter to accept a candidate, and
            // submitting there would send a half-typed word in CJK input.
            if (
              event.key === "Enter" &&
              !event.shiftKey &&
              !event.nativeEvent.isComposing
            ) {
              event.preventDefault();
              submit();
            }
          }}
          className={cn(
            "max-h-[200px] min-h-[2.25rem] flex-1 resize-none bg-transparent px-2 py-1.5",
            "text-sm leading-relaxed outline-none placeholder:text-muted",
            "disabled:cursor-not-allowed disabled:opacity-60",
          )}
        />

        {isStreaming ? (
          <button
            type="button"
            onClick={onStop}
            aria-label="Stop generating"
            className={cn(
              "flex size-8 shrink-0 items-center justify-center rounded-full",
              "border border-border text-muted transition-colors hover:text-foreground",
              "focus-visible:ring-2 focus-visible:ring-accent focus-visible:outline-none",
            )}
          >
            <Square className="size-3.5 fill-current" aria-hidden />
          </button>
        ) : (
          <button
            type="button"
            onClick={submit}
            disabled={!value.trim() || disabled}
            aria-label="Send message"
            className={cn(
              "flex size-8 shrink-0 items-center justify-center rounded-full",
              "bg-accent text-accent-foreground transition-[filter]",
              "hover:brightness-110 focus-visible:ring-2 focus-visible:ring-accent",
              "focus-visible:outline-none disabled:cursor-not-allowed disabled:opacity-40",
            )}
          >
            <ArrowUp className="size-4" aria-hidden />
          </button>
        )}
      </div>

      <p className="mx-auto mt-2 max-w-[780px] px-2 text-center text-[0.6875rem] text-muted">
        Answers are drawn from your organization&apos;s documents and data. Check
        cited sources for anything consequential.
      </p>
    </div>
  );
}
