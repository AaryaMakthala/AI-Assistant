import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

/**
 * Merge class names, letting a later conflicting Tailwind utility win.
 *
 * Plain concatenation leaves both `px-2` and `px-4` in the string and the winner is
 * whichever CSS rule the stylesheet happens to emit last — which is not the one the
 * caller passed. `twMerge` resolves the conflict in favour of the override.
 */
export function cn(...inputs: ClassValue[]): string {
  return twMerge(clsx(inputs));
}

/** Bytes as a short human string: `2.4 MB`. */
export function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  const units = ["KB", "MB", "GB"];
  let value = bytes / 1024;
  let unit = 0;
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024;
    unit += 1;
  }
  return `${value < 10 ? value.toFixed(1) : Math.round(value)} ${units[unit]}`;
}

/**
 * A timestamp as a short relative string.
 *
 * Server timestamps may arrive without a timezone designator, which `Date` would read as
 * local time and render as "in 5 hours" for something that just happened. A bare
 * ISO-looking string is therefore treated as UTC, which is what the backend stores.
 */
export function formatRelativeTime(iso: string): string {
  const normalized = /[Z+]|-\d{2}:\d{2}$/.test(iso) ? iso : `${iso}Z`;
  const then = new Date(normalized).getTime();
  if (Number.isNaN(then)) return "";

  const seconds = Math.round((Date.now() - then) / 1000);
  if (seconds < 60) return "just now";
  const minutes = Math.round(seconds / 60);
  if (minutes < 60) return `${minutes}m ago`;
  const hours = Math.round(minutes / 60);
  if (hours < 24) return `${hours}h ago`;
  const days = Math.round(hours / 24);
  if (days < 7) return `${days}d ago`;
  return new Date(normalized).toLocaleDateString(undefined, {
    month: "short",
    day: "numeric",
  });
}
