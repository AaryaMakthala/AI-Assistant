"use client";

/**
 * Who can read a document, as a badge.
 *
 * Sits beside the ingestion `StatusBadge` and answers a different question: not "is this
 * usable" but "who can see it". Both matter when scanning the owner's view of every
 * document in the organization, where company files and members' personal ones are listed
 * together and are otherwise indistinguishable.
 */

import { Building2, Lock } from "lucide-react";
import type { DocumentVisibility } from "@/lib/api";
import { cn } from "@/lib/utils";

const STYLES: Record<DocumentVisibility, string> = {
  org: "bg-accent-subtle text-accent",
  personal: "bg-surface-raised text-muted",
};

const LABELS: Record<DocumentVisibility, string> = {
  org: "Company",
  personal: "Personal",
};

const TITLES: Record<DocumentVisibility, string> = {
  org: "Everyone in the organization can read this and ask questions about it.",
  personal: "Only its uploader — and owners and admins — can read this.",
};

const ICONS: Record<DocumentVisibility, typeof Lock> = {
  org: Building2,
  personal: Lock,
};

export function VisibilityBadge({
  visibility,
  className,
}: {
  visibility: DocumentVisibility;
  className?: string;
}) {
  const Icon = ICONS[visibility];

  return (
    <span
      className={cn(
        "inline-flex items-center gap-1 rounded-full px-1.5 py-0.5",
        "text-[0.6875rem] font-medium",
        STYLES[visibility],
        className,
      )}
      title={TITLES[visibility]}
    >
      <Icon className="size-2.5" aria-hidden />
      {LABELS[visibility]}
    </span>
  );
}
