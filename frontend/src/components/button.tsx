"use client";

/**
 * One button system for the whole app.
 *
 * Two variants, used everywhere:
 *   - `primary`: warm off-white fill (`#F5F3EC`), deep-green text (`#0C1410`).
 *     Reserved for the single most important action in a view.
 *   - `secondary`: transparent background, hairline white border, white text;
 *     hover brightens. Used for every other action.
 *   - `ghost`: same as secondary but borderless — lowest-emphasis actions.
 *   - `danger`: destructive actions (delete, reject-org).
 *
 * Shape is consistently `rounded-md` and type buttons default to `button`.
 * Icon-only buttons ("shape=icon") are compact squares with no text label.
 */

import { cn } from "@/lib/utils";

type Variant = "primary" | "secondary" | "ghost" | "danger";
type Shape = "rounded" | "icon";

interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: Variant;
  shape?: Shape;
  size?: "sm" | "md";
}

const BASE =
  "inline-flex items-center justify-center gap-1.5 font-medium " +
  "transition-colors focus-visible:ring-2 focus-visible:outline-none " +
  "disabled:cursor-not-allowed disabled:opacity-50";

const VARIANT_CLASSES: Record<Variant, string> = {
  primary:
    "bg-accent text-accent-foreground hover:opacity-90 " +
    "focus-visible:ring-accent",
  secondary:
    "border border-[rgba(255,255,255,0.2)] text-muted hover:text-foreground " +
    "hover:border-[rgba(255,255,255,0.3)] hover:bg-[rgba(255,255,255,0.06)] " +
    "focus-visible:ring-accent",
  ghost:
    "border border-[rgba(255,255,255,0.08)] bg-transparent " +
    "text-muted hover:bg-surface-raised hover:text-foreground " +
    "focus-visible:ring-accent",
  danger:
    "bg-danger text-[#0c1410] hover:opacity-90 " +
    "focus-visible:ring-danger",
};

const SHAPE_CLASSES: Record<Shape, string> = {
  rounded: "rounded-md",
  icon: "rounded-md",
};

const SIZE_CLASSES: Record<"sm" | "md", string> = {
  sm: "px-3 py-1.5 text-xs",
  md: "px-4 py-2 text-sm",
};

export function Button({
  variant = "secondary",
  shape = "rounded",
  size = "sm",
  className,
  ...props
}: ButtonProps) {
  return (
    <button
      className={cn(
        BASE,
        VARIANT_CLASSES[variant],
        SHAPE_CLASSES[shape],
        SIZE_CLASSES[size],
        className,
      )}
      {...props}
    />
  );
}
