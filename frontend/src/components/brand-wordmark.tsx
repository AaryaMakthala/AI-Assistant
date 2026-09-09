"use client";

/**
 * The OFFICE BRAIN wordmark — the app's one top-left brand lockup.
 *
 * Every surface that shows the brand top-left (chat/documents top bar, login
 * nav) renders this component, so caps, letter-spacing, weight and size cannot
 * drift apart between pages. It is deliberately NOT used anywhere else: the
 * wordmark appears only in top-left brand positions.
 */

import { cn } from "@/lib/utils";

/** The wordmark's canonical typographic treatment. */
const WORDMARK_CLASS =
  "font-sans text-[11px] font-semibold uppercase tracking-[0.18em] text-foreground/80";

export function BrandWordmark({
  className,
  withMark = false,
  markClassName,
  textClassName,
}: {
  className?: string;
  /** Render the rounded Building2 mark before the text (login nav lockup). */
  withMark?: boolean;
  /** Extra classes for the mark box, e.g. sizing on the login page. */
  markClassName?: string;
  /** Override the wordmark text classes (color, tracking, etc.). */
  textClassName?: string;
}) {
  return (
    <span className={cn("inline-flex items-center gap-2", className)}>
      {withMark && (
          <img src="/logo.png" alt="" className="h-[40px] w-auto object-contain" aria-hidden="true" />
      )}
      <span className={textClassName || cn(WORDMARK_CLASS, "wordmark-text")}>Office Brain</span>
    </span>
  );
}
