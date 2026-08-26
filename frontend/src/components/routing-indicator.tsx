"use client";

/**
 * The "thinking / routing" indicator (CLAUDE.md section 6).
 *
 * Shows which agents the supervisor picked and what they are doing, from the `routing` and
 * `step` frames. The wait before the first token is the longest part of a multi-agent
 * answer — a SQL branch generates a query, validates it and executes it before synthesis
 * starts — and an unexplained pause reads as a hang. Naming the work is what distinguishes
 * "consulting two sources" from "broken".
 */

import { Database, FileSearch, GitPullRequest, MessageCircle } from "lucide-react";
import type { AgentRoute } from "@/lib/api";
import { cn } from "@/lib/utils";

const ROUTE_LABELS: Record<AgentRoute, { label: string; Icon: typeof Database }> = {
  documents: { label: "Documents", Icon: FileSearch },
  business_data: { label: "Business data", Icon: Database },
  external: { label: "Code", Icon: GitPullRequest },
  direct: { label: "Direct reply", Icon: MessageCircle },
};

export function RoutingIndicator({
  routes,
  reason,
  steps,
  isStreaming,
  hasContent,
}: {
  routes: AgentRoute[];
  reason: string;
  steps: string[];
  isStreaming: boolean;
  hasContent: boolean;
}) {
  // Once tokens are arriving the answer speaks for itself; the trace would just compete
  // with it. It collapses to the route badges, which stay useful as provenance.
  const showTrace = isStreaming && !hasContent;

  if (!routes.length && !steps.length) {
    return showTrace ? (
      <div className="flex items-center gap-2 text-xs text-muted">
        <PulseDots />
        <span>Deciding where to look…</span>
      </div>
    ) : null;
  }

  const latest = steps[steps.length - 1];

  return (
    <div className="space-y-1.5">
      {routes.length > 0 && (
        <div className="flex flex-wrap items-center gap-1.5">
          {routes.map((route) => {
            const entry = ROUTE_LABELS[route];
            if (!entry) return null;
            const { label, Icon } = entry;
            return (
              <span
                key={route}
                className={cn(
                  "inline-flex items-center gap-1.5 rounded-md border border-border",
                  "bg-surface-raised px-2 py-0.5 text-xs text-muted",
                )}
              >
                <Icon className="size-3" aria-hidden />
                {label}
              </span>
            );
          })}
          {showTrace && reason && (
            <span className="text-xs text-muted italic">{reason}</span>
          )}
        </div>
      )}

      {showTrace && latest && (
        <div className="flex items-center gap-2 text-xs text-muted">
          <PulseDots />
          <span className="truncate">{latest}</span>
        </div>
      )}
    </div>
  );
}

function PulseDots() {
  return (
    <span className="flex gap-0.5" aria-hidden>
      {[0, 150, 300].map((delay) => (
        <span
          key={delay}
          className="size-1 animate-pulse rounded-full bg-muted"
          style={{ animationDelay: `${delay}ms` }}
        />
      ))}
    </span>
  );
}
