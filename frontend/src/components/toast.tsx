"use client";

/**
 * A transient error toast for actions whose failure must be seen where the
 * click happened.
 *
 * Why this exists: `useDocuments` publishes every failure through one shared
 * `error` channel, which the chat pane renders as a banner in the transcript —
 * anywhere from directly below the delete button to entirely off-screen. An
 * action like delete has a precise location (the row, the modal, the confirm)
 * and a short audience (the person who just clicked), so a popup that appears
 * near the action and dismisses itself is the right shape. Read the callers:
 * the chat transcript banner keeps other failure kinds (chat transport, list
 * refresh), only action failures route here.
 *
 * Fixed to the viewport's bottom-right so it never shifts layout and never
 * scrolls away. `role="alert"` makes screen readers announce it on mount.
 * Auto-dismiss runs on a timeout that resets whenever the message changes —
 * a second failure re-arms the timer instead of silently expiring early.
 */

import { AlertTriangle, X } from "lucide-react";
import { useEffect, useRef } from "react";
import { cn } from "@/lib/utils";

const DISMISS_MS = 4500;

export function Toast({
  message,
  onDismiss,
}: {
  message: string;
  onDismiss: () => void;
}) {
  const timerRef = useRef<ReturnType<typeof setTimeout> | undefined>(undefined);

  useEffect(() => {
    if (!message) return;
    clearTimeout(timerRef.current);
    timerRef.current = setTimeout(onDismiss, DISMISS_MS);
    return () => clearTimeout(timerRef.current);
  }, [message, onDismiss]);

  if (!message) return null;

  return (
    <div
      role="alert"
      aria-live="assertive"
      className={cn(
        "fixed right-4 bottom-4 z-[100] flex max-w-[min(24rem,calc(100vw-2rem))] items-start gap-2.5 p-3 pr-2",
        "text-xs leading-relaxed text-white",
        "shadow-lg shadow-black/30",
      )}
      style={{
        background: "rgba(20,40,30,0.95)",
        backdropFilter: "blur(12px)",
        WebkitBackdropFilter: "blur(12px)",
        borderRadius: "10px",
        border: "1px solid rgba(255,255,255,0.1)",
        borderLeft: "3px solid var(--danger)",
      }}
    >
      <AlertTriangle className="mt-0.5 size-3.5 shrink-0 text-danger" aria-hidden />
      <span className="min-w-0 flex-1 break-words">{message}</span>
      <button
        type="button"
        onClick={onDismiss}
        aria-label="Dismiss"
        className={cn(
          "flex size-5 shrink-0 items-center justify-center rounded text-white/60",
          "transition-colors hover:bg-white/10 hover:text-white",
          "focus-visible:ring-2 focus-visible:ring-accent focus-visible:outline-none",
        )}
      >
        <X className="size-3" aria-hidden />
      </button>
    </div>
  );
}
