"use client";

/**
 * The organization's members and invitation management.
 *
 * Visible to owners and admins. Shows the member list and allows inviting new members
 * by email. The invitation flow (CLAUDE.md section 4):
 * 1. Owner creates an invitation by email.
 * 2. Invitee authenticates with Supabase, then accepts the invitation.
 * 3. Accept creates an ACTIVE MEMBER row.
 *
 * The hiding is presentation, not protection — `/workspaces/{id}/members` returns 403
 * to anyone else regardless of what the UI chose to render (CLAUDE.md 4.6).
 */

import { AlertCircle, Check, Loader2, Send, ShieldCheck, Trash2, UserRound, X } from "lucide-react";
import { useCallback, useEffect, useState } from "react";
import { Button } from "./button";
import {
  ApiError,
  createInvitation,
  deleteWorkspace,
  listInvitations,
  listWorkspaceMembers,
  type Invitation,
  type OrgMember,
} from "@/lib/api";
import { useAuth } from "@/lib/auth";
import { cn } from "@/lib/utils";

export function MembersPanel({
  token,
  workspaceId,
  onDeleteWorkspace,
}: {
  token?: string;
  workspaceId?: string;
  onDeleteWorkspace?: () => void;
}) {
  const { signOut } = useAuth();
  const [members, setMembers] = useState<OrgMember[]>([]);
  const [invitations, setInvitations] = useState<Invitation[]>([]);
  const [error, setError] = useState<string | undefined>();
  const [isLoading, setIsLoading] = useState(true);
  const [inviteEmail, setInviteEmail] = useState("");
  const [isInviting, setIsInviting] = useState(false);
  const [inviteSuccess, setInviteSuccess] = useState<string | undefined>();
  const [activeTab, setActiveTab] = useState<"members" | "invitations">("members");
  const [showDeleteConfirm, setShowDeleteConfirm] = useState(false);
  const [isDeleting, setIsDeleting] = useState(false);
  const [deleteError, setDeleteError] = useState<string | undefined>();

  const fetchData = useCallback(async () => {
    if (!token || !workspaceId) return;
    let active = true;
    const controller = new AbortController();

    try {
      const [membersRes, invitationsRes] = await Promise.all([
        listWorkspaceMembers(workspaceId, { token, signal: controller.signal }),
        listInvitations(workspaceId, { token, signal: controller.signal }).catch(
          () => ({ invitations: [] }),
        ),
      ]);
      if (!active) return;
      setMembers(membersRes.members);
      setInvitations(invitationsRes.invitations);
      setError(undefined);
    } catch (caught: unknown) {
      if (!active || (caught as DOMException)?.name === "AbortError") return;
      setError(
        caught instanceof ApiError
          ? caught.message
          : "Could not load the member list.",
      );
    } finally {
      if (active) setIsLoading(false);
    }

    return () => {
      active = false;
      controller.abort();
    };
  }, [token, workspaceId]);

  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect
    void fetchData();
  }, [fetchData]);

  const handleInvite = useCallback(
    async (e: React.FormEvent) => {
      e.preventDefault();
      if (!token || !workspaceId || !inviteEmail.trim()) return;

      setIsInviting(true);
      setError(undefined);
      setInviteSuccess(undefined);

      try {
        const invitation = await createInvitation(workspaceId, inviteEmail.trim(), {
          token,
        });
        setInvitations((current) => [invitation, ...current]);
        setInviteSuccess(`Invitation sent to ${invitation.email}`);
        setInviteEmail("");
      } catch (caught) {
        setError(
          caught instanceof ApiError
            ? caught.message
            : "Could not send the invitation.",
        );
      } finally {
        setIsInviting(false);
      }
    },
    [token, workspaceId, inviteEmail],
  );

  const handleDeleteOrganization = useCallback(async () => {
    if (!token || !workspaceId) return;
    setIsDeleting(true);
    setDeleteError(undefined);
    try {
      await deleteWorkspace(workspaceId, { token });
      setShowDeleteConfirm(false);
      // The owner's account is deleted by the backend. Clear the session
      // and redirect to the login page — do NOT show the zero-org view.
      await signOut();
    } catch (caught) {
      setDeleteError(
        caught instanceof ApiError
          ? caught.message
          : "Failed to delete organization.",
      );
    } finally {
      setIsDeleting(false);
    }
  }, [token, workspaceId, onDeleteWorkspace]);

  const isOwner = members.some(
    (m) => m.role === "OWNER" && m.status === "ACTIVE",
  );

  if (isLoading) {
    return <p className="px-1 py-6 text-center text-xs text-muted">Loading members…</p>;
  }

  if (error && !members.length) {
    return (
      <p role="alert" className="px-1 py-6 text-center text-xs text-danger">
        {error}
      </p>
    );
  }

  const pendingInvitations = invitations.filter((inv) => inv.status === "PENDING");

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      {/* Tab switch */}
      <div role="tablist" aria-label="Member sections" className="flex gap-1 px-3 pb-2">
        {([
          ["members", `Members (${members.length})`],
          ["invitations", `Invites (${pendingInvitations.length})`],
        ] as const).map(([value, label]) => (
          <button
            key={value}
            type="button"
            role="tab"
            aria-selected={activeTab === value}
            onClick={() => setActiveTab(value)}
            className={cn(
              "flex-1 rounded-md px-2 py-1 text-[0.6875rem] font-medium transition-colors",
              "focus-visible:ring-2 focus-visible:ring-accent focus-visible:outline-none",
              activeTab === value
                ? "bg-surface-raised text-foreground"
                : "text-muted hover:text-foreground",
            )}
          >
            {label}
          </button>
        ))}
      </div>

      {/* Error banner */}
      {error && (
        <div
          role="alert"
          className="mx-3 mb-2 flex items-start gap-2 rounded-md border border-danger/30 bg-danger-subtle px-2.5 py-1.5 text-xs text-danger"
        >
          <AlertCircle className="mt-px size-3 shrink-0" aria-hidden />
          <span className="min-w-0 flex-1 break-words">{error}</span>
          <button
            type="button"
            onClick={() => setError(undefined)}
            className="shrink-0 rounded p-0.5 hover:opacity-70"
          >
            <X className="size-3" aria-hidden />
          </button>
        </div>
      )}

      {activeTab === "members" && (
        <nav className="min-h-0 flex-1 overflow-y-auto px-3 pb-3">
          {members.length === 0 ? (
            <p className="px-1 py-6 text-center text-xs text-muted">
              No members yet.
            </p>
          ) : (
            <ul className="space-y-0.5">
              {members.map((member) => (
                <li
                  key={member.id}
                  className="flex items-center gap-2 rounded-md px-2 py-2 hover:bg-surface-raised"
                >
                  {member.role === "MEMBER" ? (
                    <UserRound className="size-3.5 shrink-0 text-muted" aria-hidden />
                  ) : (
                    <ShieldCheck className="size-3.5 shrink-0 text-accent" aria-hidden />
                  )}
                  <span className="min-w-0 flex-1">
                    <span className="block truncate text-xs">
                      {member.user_id}
                    </span>
                    <span className="block text-[0.6875rem] text-muted">
                      Joined {new Date(member.created_at).toLocaleDateString()}
                    </span>
                  </span>
                  <span
                    className={cn(
                      "shrink-0 rounded border border-border px-1.5 py-0.5 text-[0.625rem]",
                      member.role === "OWNER" ? "text-accent" : "text-muted",
                    )}
                  >
                    {member.role}
                  </span>
                </li>
              ))}
            </ul>
          )}
        </nav>
      )}

      {activeTab === "invitations" && (
        <div className="min-h-0 flex-1 overflow-y-auto px-3 pb-3">
          {/* Invite form */}
          <form onSubmit={handleInvite} className="mb-3">
            <div className="flex gap-1.5">
              <input
                type="email"
                value={inviteEmail}
                onChange={(e) => setInviteEmail(e.target.value)}
                placeholder="Email address"
                required
                className={cn(
                  "min-w-0 flex-1 rounded-md border border-border bg-background px-2.5 py-1.5 text-xs",
                  "placeholder:text-muted/60 transition-colors",
                  "focus:border-accent focus:ring-1 focus:ring-accent focus:outline-none",
                )}
              />
              <Button
                type="submit"
                variant="primary"
                disabled={isInviting || !inviteEmail.trim()}
              >
                {isInviting ? (
                  <Loader2 className="size-3 animate-spin" aria-hidden />
                ) : (
                  <Send className="size-3" aria-hidden />
                )}
                Invite
              </Button>
            </div>
            {inviteSuccess && (
              <p className="mt-1.5 flex items-center gap-1 text-[0.6875rem] text-success">
                <Check className="size-3" aria-hidden />
                {inviteSuccess}
              </p>
            )}
          </form>

          {/* Invitations list */}
          {pendingInvitations.length === 0 ? (
            <p className="px-1 py-4 text-center text-xs text-muted">
              No pending invitations.
            </p>
          ) : (
            <ul className="space-y-0.5">
              {pendingInvitations.map((invitation) => (
                <li
                  key={invitation.id}
                  className="flex items-center gap-2 rounded-md px-2 py-2 hover:bg-surface-raised"
                >
                  <Send className="size-3.5 shrink-0 text-muted" aria-hidden />
                  <span className="min-w-0 flex-1">
                    <span className="block truncate text-xs">{invitation.email}</span>
                    <span className="block text-[0.6875rem] text-muted">
                      Sent {new Date(invitation.created_at).toLocaleDateString()}
                    </span>
                  </span>
                  <span className="shrink-0 rounded border border-border px-1.5 py-0.5 text-[0.625rem] text-warning">
                    Pending
                  </span>
                </li>
              ))}
            </ul>
          )}
        </div>
      )}

      {/* Delete Organization - Owner only */}
      {isOwner && (
        <div className="border-t border-border px-3 py-3">
          {showDeleteConfirm ? (
            <div className="space-y-3">
              <div className="rounded-md bg-danger/10 p-3 text-xs text-danger border border-danger/20">
                <p className="font-medium">Delete organization?</p>
                <p className="mt-1 text-danger/80">
                  This will permanently delete the organization, its associated data, and your account. This action cannot be undone.
                </p>
              </div>
              {deleteError && (
                <p className="text-xs text-danger">{deleteError}</p>
              )}
              <div className="flex gap-2">
                <Button
                  variant="danger"
                  onClick={() => void handleDeleteOrganization()}
                  disabled={isDeleting}
                >
                  {isDeleting ? (
                    <Loader2 className="size-3 animate-spin" aria-hidden />
                  ) : (
                    <Trash2 className="size-3" aria-hidden />
                  )}
                  Delete organization
                </Button>
                <Button
                  variant="ghost"
                  onClick={() => {
                    setShowDeleteConfirm(false);
                    setDeleteError(undefined);
                  }}
                  disabled={isDeleting}
                >
                  Cancel
                </Button>
              </div>
            </div>
          ) : (
            <button
              type="button"
              onClick={() => setShowDeleteConfirm(true)}
              className={cn(
                "flex w-full items-center gap-1.5 rounded-md px-2 py-1.5 text-xs",
                "text-danger/70 transition-colors hover:text-danger hover:bg-danger/5",
              )}
            >
              <Trash2 className="size-3.5" aria-hidden />
              Delete organization
            </button>
          )}
        </div>
      )}
    </div>
  );
}
