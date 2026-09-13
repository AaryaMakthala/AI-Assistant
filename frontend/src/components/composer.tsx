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
  placeholder = "Ask about your workspace...",
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
    <div className="relative z-10 px-4 pt-1 pb-6 max-md:px-3 max-md:pt-1.5 max-md:pb-3">
      {/* Unified composer container — one rounded box with embedded send button */}
      <div
        className={cn(
          "glass-composer mx-auto flex max-w-[780px] items-end gap-2 p-2.5",
          "max-md:mx-auto max-md:w-full max-md:max-w-none max-md:rounded-[20px] max-md:p-2",
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
            "max-h-[200px] min-h-[2.5rem] flex-1 resize-none bg-transparent px-2.5 py-2",
            "text-[14px] leading-relaxed outline-none placeholder:text-muted/60",
            "disabled:cursor-not-allowed disabled:opacity-60",
            "max-md:text-[14px] max-md:leading-[1.5] max-md:py-2 max-md:px-2",
          )}
        />

        {isStreaming ? (
          <button
            type="button"
            onClick={onStop}
            aria-label="Stop generating"
            className={cn(
              "flex h-[38px] w-[38px] shrink-0 items-center justify-center rounded-full",
              "border border-border text-muted transition-colors hover:text-foreground",
              "focus-visible:ring-2 focus-visible:ring-accent focus-visible:outline-none",
              "max-md:h-10 max-md:w-10",
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
              "flex h-[38px] w-[38px] shrink-0 items-center justify-center rounded-full",
              "bg-accent text-accent-foreground transition-[filter]",
              "hover:brightness-110 focus-visible:ring-2 focus-visible:ring-accent",
              "focus-visible:outline-none disabled:cursor-not-allowed disabled:opacity-40",
              "max-md:h-10 max-md:w-10",
            )}
          >
            <ArrowUp className="size-4" aria-hidden />
          </button>
        )}
      </div>

      <p className="mx-auto mt-2 max-w-[780px] px-2 text-center text-[0.6875rem] text-muted max-md:mt-1.5 max-md:text-[10px]">
        Answers are drawn from your organization&apos;s documents and data. Check
        cited sources for anything consequential.
      </p>
    </div>
  );
}
