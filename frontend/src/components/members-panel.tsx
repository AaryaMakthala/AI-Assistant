"use client";

/**
 * The organization's members. Visible to owners and admins.
 *
 * The hiding is presentation, not protection — `/org/members` returns 403 to anyone else
 * regardless of what the UI chose to render (CLAUDE.md 4.6). It is still worth doing: a
 * control that cannot work should not be offered.
 *
 * A 403 arriving here anyway is treated as a real state rather than an error. It means the
 * role the server reported and the role it enforces disagree, which is worth saying plainly
 * instead of showing an empty list that reads as "no members".
 */

import { ShieldCheck, UserRound } from "lucide-react";
import { useEffect, useState } from "react";
import { ApiError, listOrgMembers, type OrgMember } from "@/lib/api";
import { cn } from "@/lib/utils";

export function MembersPanel({ token }: { token?: string }) {
  const [members, setMembers] = useState<OrgMember[]>([]);
  const [error, setError] = useState<string | undefined>();
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    if (!token) return;
    let active = true;
    const controller = new AbortController();

    listOrgMembers({ token, signal: controller.signal })
      .then((response) => {
        if (!active) return;
        setMembers(response.members);
        setError(undefined);
      })
      .catch((caught: unknown) => {
        if (!active || (caught as DOMException)?.name === "AbortError") return;
        setError(
          caught instanceof ApiError
            ? caught.message
            : "Could not load the member list.",
        );
      })
      .finally(() => {
        if (active) setIsLoading(false);
      });

    return () => {
      active = false;
      controller.abort();
    };
  }, [token]);

  if (isLoading) {
    return <p className="px-1 py-6 text-center text-xs text-muted">Loading members…</p>;
  }

  if (error) {
    return (
      <p role="alert" className="px-1 py-6 text-center text-xs text-danger">
        {error}
      </p>
    );
  }

  return (
    <ul className="space-y-0.5">
      {members.map((member) => (
        <li
          key={member.id}
          className="flex items-center gap-2 rounded-md px-2 py-2 hover:bg-surface-raised"
        >
          {member.role === "member" ? (
            <UserRound className="size-3.5 shrink-0 text-muted" aria-hidden />
          ) : (
            <ShieldCheck className="size-3.5 shrink-0 text-accent" aria-hidden />
          )}
          <span className="min-w-0 flex-1">
            <span className="block truncate text-xs">
              {member.full_name || member.email}
            </span>
            <span className="block truncate text-[0.6875rem] text-muted">
              {member.email}
            </span>
          </span>
          <span
            className={cn(
              "shrink-0 rounded border border-border px-1.5 py-0.5 text-[0.625rem]",
              member.role === "member" ? "text-muted" : "text-accent",
            )}
          >
            {member.role}
          </span>
        </li>
      ))}
    </ul>
  );
}
