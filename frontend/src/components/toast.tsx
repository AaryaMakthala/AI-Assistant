"use client";

/**
 * A transient error toast for actions whose failure must be seen where the
 * click happened.
 *
 * Why this exists: `useDocuments` publishes every failure through one shared
 * `error` channel, which the chat pane renders as a banner in the transcript —
 * anywhere from directly below the delete button to entirely off-screen. An
 * action like delete has a precise location (the row, the modal, the confirm)
 * and a short audience (the person who just clicked), so a prominent popup
 * that demands attention and dismisses itself is the right shape. Read the
 * callers: the chat transcript banner keeps other failure kinds (chat
 * transport, list refresh), only action failures route here.
 *
 * Rendering: centered in the viewport (fixed, translate(-50%, -50%)) above a
 * semi-transparent backdrop overlay, in the app's dark-green theme, with a
 * short fade/scale entrance. The overlay draws focus so the message cannot
 * get lost against page content; clicking it dismisses the toast early, in
 * addition to the auto-dismiss timer.
 *
 * `role="alert"` makes screen readers announce it on mount. Auto-dismiss runs
 * on a timeout that resets whenever the message changes — a second failure
 * re-arms the timer instead of silently expiring early.
 */

import { AlertTriangle, X } from "lucide-react";
import { useEffect, useRef } from "react";
import { cn } from "@/lib/utils";

const DISMISS_MS = 4500;

/** Entrance animation — the keyframes restate the centering transform. */
const TOAST_KEYFRAMES = `
@keyframes toastIn {
  from { opacity: 0; transform: translate(-50%, -50%) scale(0.95); }
  to { opacity: 1; transform: translate(-50%, -50%) scale(1); }
}
`;

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
    <>
      <style>{TOAST_KEYFRAMES}</style>

      {/* Backdrop: draws focus to the toast; click dismisses early. */}
      <div
        aria-hidden="true"
        onClick={onDismiss}
        style={{
          position: "fixed",
          inset: 0,
          background: "rgba(0, 0, 0, 0.35)",
          zIndex: 99,
        }}
      />

      <div
        role="alert"
        aria-live="assertive"
        style={{
          position: "fixed",
          top: "50%",
          left: "50%",
          transform: "translate(-50%, -50%)",
          zIndex: 100,
          display: "flex",
          alignItems: "flex-start",
          gap: "10px",
          background: "rgba(13, 40, 24, 0.92)",
          backdropFilter: "blur(16px)",
          WebkitBackdropFilter: "blur(16px)",
          border: "1px solid rgba(220, 100, 100, 0.35)",
          borderRadius: "16px",
          padding: "20px 28px",
          boxShadow: "0 12px 40px rgba(0, 0, 0, 0.5)",
          color: "#F2F5EF",
          fontSize: "15px",
          textAlign: "center",
          maxWidth: "420px",
          lineHeight: 1.5,
          animation: "toastIn 0.2s ease-out",
        }}
      >
        <AlertTriangle
          className="mt-0.5 size-4 shrink-0 text-danger"
          aria-hidden
        />
        <span className="min-w-0 flex-1 break-words">{message}</span>
        <button
          type="button"
          onClick={onDismiss}
          aria-label="Dismiss"
          className={cn(
            "flex size-5 shrink-0 items-center justify-center rounded",
            "transition-colors hover:bg-white/10",
            "focus-visible:ring-2 focus-visible:ring-accent focus-visible:outline-none",
          )}
          style={{ color: "rgba(242, 245, 239, 0.6)" }}
        >
          <X className="size-3.5" aria-hidden />
        </button>
      </div>
    </>
  );
}
