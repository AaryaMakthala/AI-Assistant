"use client";

/**
 * A modal confirmation for an irreversible action.
 *
 * Native `<dialog>` rather than a hand-rolled overlay: the browser gives focus trapping,
 * Escape-to-close, inertness of the background, and the top layer for free. Reimplementing
 * those in application code is where accessibility bugs come from.
 *
 * The confirm button is not autofocused. For a destructive action the safe default is that
 * Enter does nothing until the user chooses — autofocusing "Delete" turns a stray keypress
 * into data loss.
 */

import { AlertTriangle, Loader2 } from "lucide-react";
import { useEffect, useRef } from "react";
import { cn } from "@/lib/utils";

export function ConfirmDialog({
  open,
  title,
  description,
  confirmLabel = "Delete",
  isBusy,
  onConfirm,
  onCancel,
}: {
  open: boolean;
  title: string;
  description: React.ReactNode;
  confirmLabel?: string;
  isBusy?: boolean;
  onConfirm: () => void;
  onCancel: () => void;
}) {
  const ref = useRef<HTMLDialogElement>(null);

  useEffect(() => {
    const dialog = ref.current;
    if (!dialog) return;
    // showModal/close are imperative by design; calling them in an effect is how the
    // element's open state is kept in step with React's.
    if (open && !dialog.open) dialog.showModal();
    if (!open && dialog.open) dialog.close();
  }, [open]);

  if (!open) return null;

  return (
    <dialog
      ref={ref}
      // Escape fires `cancel`, not a click on either button — without this the dialog
      // would close visually while the parent still believed it was open.
      onCancel={(event) => {
        event.preventDefault();
        if (!isBusy) onCancel();
      }}
      aria-labelledby="confirm-dialog-title"
      className={cn(
        "m-auto w-[min(26rem,calc(100vw-2rem))] rounded-xl border border-border",
        "bg-surface p-0 text-foreground shadow-xl backdrop:bg-black/40",
      )}
    >
      <div className="flex gap-3 p-4">
        <span
          className="flex size-8 shrink-0 items-center justify-center rounded-full bg-danger-subtle"
          aria-hidden
        >
          <AlertTriangle className="size-4 text-danger" />
        </span>
        <div className="min-w-0 flex-1">
          <h2 id="confirm-dialog-title" className="text-sm font-semibold">
            {title}
          </h2>
          <div className="mt-1 text-xs break-words text-muted">{description}</div>
        </div>
      </div>

      <div className="flex justify-end gap-2 border-t border-border px-4 py-3">
        <button
          type="button"
          onClick={onCancel}
          disabled={isBusy}
          className={cn(
            "rounded-md px-3 py-1.5 text-xs font-medium text-muted transition-colors",
            "hover:text-foreground focus-visible:ring-2 focus-visible:ring-accent",
            "focus-visible:outline-none disabled:opacity-50",
          )}
        >
          Cancel
        </button>
        <button
          type="button"
          onClick={onConfirm}
          disabled={isBusy}
          className={cn(
            "flex items-center gap-1.5 rounded-md bg-danger px-3 py-1.5 text-xs",
            "font-semibold text-[#0c1410] transition-opacity hover:opacity-90",
            "focus-visible:ring-2 focus-visible:ring-danger focus-visible:outline-none",
            "disabled:cursor-not-allowed disabled:opacity-50",
          )}
        >
          {isBusy && <Loader2 className="size-3 animate-spin" aria-hidden />}
          {confirmLabel}
        </button>
      </div>
    </dialog>
  );
}
