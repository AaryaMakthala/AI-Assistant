"use client";

/**
 * A transient toast for messages whose contents must be seen where the click
 * happened. Default tone is `error` (action failures — muted red accent).
 * Pass `variant="info"` for informational notices (sage-green accent).
 *
 * Rendering: centered in the viewport (fixed, translate(-50%, -50%)) above a
 * semi-transparent backdrop overlay, in the app's dark-green theme, with a
 * subtle fade/scale entrance. The overlay draws focus so the message cannot
 * get lost against page content; clicking it dismisses the toast early, in
 * addition to the auto-dismiss timer.
 *
 * `role="alert"` makes screen readers announce it on mount. Auto-dismiss runs
 * on a timeout that resets whenever the message changes — a second failure
 * re-arms the timer instead of silently expiring early.
 */

import { AlertCircle, Info, X } from "lucide-react";
import { useEffect, useRef } from "react";
import { cn } from "@/lib/utils";

const DISMISS_MS = 4500;

/**
 * Toast tone.
 *  - `error` (default): muted red indicator — action failures.
 *  - `info`: sage-green indicator — informational notices.
 */
type ToastVariant = "error" | "info";

const TOAST_KEYFRAMES = `
@keyframes toastIn {
  from { opacity: 0; transform: translate(-50%, -50%) scale(0.97); }
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

      {/* Backdrop — click dismisses early */}
      <div
        aria-hidden="true"
        onClick={onDismiss}
        className="fixed inset-0 z-40 bg-black/40"
      />

      {/* Toast card */}
      <div
        role="alert"
        aria-live="assertive"
        className={cn(
          "fixed top-1/2 left-1/2 z-50 -translate-x-1/2 -translate-y-1/2",
          "w-[min(420px,calc(100vw-32px))] max-w-[calc(100vw-32px)]",
          "rounded-2xl border shadow-2xl",
          "backdrop-blur-xl",
          "box-border",
          "animate-[toastIn_0.2s_ease-out]",
          isInfo
            ? "border-[rgba(156,197,168,0.25)] bg-[rgba(14,32,22,0.94)]"
            : "border-[rgba(255,107,107,0.2)] bg-[rgba(14,32,22,0.94)]",
        )}
      >
        <div className="flex items-start gap-3 px-5 py-4">
          {/* Icon */}
          <div
            className={cn(
              "mt-0.5 flex h-7 w-7 shrink-0 items-center justify-center rounded-lg",
              isInfo
                ? "bg-[rgba(156,197,168,0.12)]"
                : "bg-[rgba(255,107,107,0.1)]",
            )}
          >
            {isInfo ? (
              <Info className="h-4 w-4 text-[#9cc5a8]" aria-hidden />
            ) : (
              <AlertCircle className="h-4 w-4 text-[#e07070]" aria-hidden />
            )}
          </div>

          {/* Text block */}
          <div className="min-w-0 flex-1">
            <p className="text-[14px] font-semibold leading-snug text-[#f2f5ef]">
              Something went wrong
            </p>
            <p className="mt-1 text-[13px] leading-[1.45] text-[rgba(201,212,198,0.8)]">
              {message}
            </p>
          </div>

          {/* Close button */}
          <button
            type="button"
            onClick={onDismiss}
            aria-label="Dismiss"
            className={cn(
              "flex h-7 w-7 shrink-0 items-center justify-center rounded-md",
              "text-[rgba(201,212,198,0.5)] transition-colors",
              "hover:bg-[rgba(255,255,255,0.06)] hover:text-[rgba(201,212,198,0.8)]",
              "focus-visible:ring-2 focus-visible:ring-accent focus-visible:outline-none",
              "max-md:h-10 max-md:w-10",
            )}
          >
            <X className="h-4 w-4" aria-hidden />
          </button>
        </div>
      </div>
    </>
  );
}
