"use client";

/**
 * A transient toast for messages whose contents must be seen where the click
 * happened. Default tone is `error` (action failures — red accent, triangle
 * icon). Pass `variant="info"` for informational notices (sage-green accent,
 * info glyph) — same card shape, typography and radius, only the accent
 * changes.
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

import { AlertTriangle, Info, X } from "lucide-react";
import { useEffect, useRef } from "react";
import { cn } from "@/lib/utils";

const DISMISS_MS = 4500;

/**
 * Toast tone.
 *  - `error` (default): red triangle + red hairline border — action failures.
 *  - `info`: sage-green info glyph + sage-green hairline border — informational
 *    notices like the demo tier's "uploads unavailable" message. Same card, same
 *    typography, same radius — only the accent tells them apart.
 */
type ToastVariant = "error" | "info";

const INFO_BORDER = "1px solid rgba(156, 197, 168, 0.4)";
const ERROR_BORDER = "1px solid rgba(220, 100, 100, 0.35)";

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
  variant = "error",
}: {
  message: string;
  onDismiss: () => void;
  variant?: ToastVariant;
}) {
  const isInfo = variant === "info";
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
          border: isInfo ? INFO_BORDER : ERROR_BORDER,
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
        {isInfo ? (
          <Info className="mt-0.5 size-4 shrink-0 text-success" aria-hidden />
        ) : (
          <AlertTriangle className="mt-0.5 size-4 shrink-0 text-danger" aria-hidden />
        )}
        <span className="min-w-0 flex-1 break-words">{message}</span>
        <button
          type="button"
          onClick={onDismiss}
          aria-label="Dismiss"
          className={cn(
            "flex size-5 shrink-0 items-center justify-center rounded transition-colors",
            "hover:bg-white/10 focus-visible:ring-2 focus-visible:ring-accent focus-visible:outline-none",
            isInfo ? "text-success" : "text-muted",
          )}
        >
          <X className="size-3.5" aria-hidden />
        </button>
      </div>
    </>
  );
}
