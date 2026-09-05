"use client";

/**
 * The workspace: sidebar, chat pane, and the citations panel.
 *
 * State lives here rather than in a store — three hooks and two selections is not enough
 * to justify one, and keeping the wiring visible in a single component is what makes
 * the data flow readable.
 */

import { Building2, LogOut, PanelRightOpen, Upload } from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Button } from "@/components/button";
import { ChatPane } from "@/components/chat-pane";
import { Sidebar } from "@/components/sidebar";
import { SourcesPanel } from "@/components/sources-panel";
import { UploadInterface } from "@/components/upload-interface";
import { useAuth } from "@/lib/auth";
import { listWorkspaces, isWorkspaceNotFound } from "@/lib/api";
import { useChat } from "@/lib/hooks/use-chat";
import { useDocuments } from "@/lib/hooks/use-documents";
import { useSessions } from "@/lib/hooks/use-sessions";

/** Roles allowed to see org administration. Mirrors the backend role model. */
const ADMIN_ROLES = ["OWNER", "owner"];

export function Workspace() {
  const { token, isAuthenticated, isLoading, me, role, signOut } = useAuth();
  const [activeChunkId, setActiveChunkId] = useState<string | undefined>();
  const [isPanelOpen, setIsPanelOpen] = useState(false);
  // Active workspace — starts from the server-confirmed default, but can be
  // overridden (e.g. by stale-workspace recovery).  The X-Workspace-ID header
  // tells the backend which workspace to scope requests to.
  const [activeWorkspaceId, setActiveWorkspaceId] = useState<string | undefined>();
  const workspaceId = activeWorkspaceId ?? me?.workspace_id;

  const canManageOrg = Boolean(role && ADMIN_ROLES.includes(role));

  const sessions = useSessions(token, workspaceId);
  const documents = useDocuments(token, workspaceId);

  const chat = useChat({
    token,
    workspaceId,
    onSessionCreated: () => void sessions.refresh(),
    onTurnComplete: () => void sessions.refresh(),
  });

  // Upload view toggle: when true, the main area shows the upload interface
  // instead of the chat pane.
  const [showUpload, setShowUpload] = useState(false);

  // --- Stale-workspace recovery ---
  // When the active workspace is deleted (or the user's membership removed),
  // every workspace-scoped API call returns 404 "Workspace not found.".
  // Instead of showing a dead-end error, we auto-switch to a valid workspace.
  const recoveryInFlight = useRef(false);
  const [hasNoWorkspaces, setHasNoWorkspaces] = useState(false);

  const combinedError = chat.transportError ?? sessions.error ?? documents.error;
  const attemptRecovery = useCallback(async () => {
    if (recoveryInFlight.current || !token) return;
    recoveryInFlight.current = true;
    try {
      const { workspaces } = await listWorkspaces({ token });
      if (workspaces.length > 0) {
        // Switch to the first available workspace.
        setActiveWorkspaceId(workspaces[0].id);
        setHasNoWorkspaces(false);
        chat.reset();
        setActiveChunkId(undefined);
        setIsPanelOpen(false);
      } else {
        // Zero workspaces — show the create-organization flow.
        setHasNoWorkspaces(true);
      }
    } catch {
      // If listing workspaces also fails (e.g. network), leave the error
      // state as-is — the user can retry manually.
    } finally {
      recoveryInFlight.current = false;
    }
  }, [token, chat]);

  useEffect(() => {
    if (combinedError && isWorkspaceNotFound(combinedError)) {
      void attemptRecovery();
    }
  }, [combinedError, attemptRecovery]);

  // The panel follows the most recent assistant turn: that is the answer the user is
  // reading, and the one whose citations the chips belong to.
  const activeTurn = useMemo(
    () => [...chat.turns].reverse().find((turn) => turn.role === "assistant"),
    [chat.turns],
  );

  const citedChunkIds = useMemo(
    () => new Set(activeTurn?.citations.map((citation) => citation.chunk_id) ?? []),
    [activeTurn],
  );

  const handleSelectCitation = (chunkId: string) => {
    setActiveChunkId(chunkId);
    setIsPanelOpen(true);
  };

  return (
    <div className="flex h-dvh w-full overflow-hidden">
      <Sidebar
        sessions={sessions.sessions}
        activeSessionId={chat.sessionId}
        documents={documents.documents}
        uploads={documents.uploads}
        deletingDocumentIds={documents.deletingIds}
        approvingDocumentIds={documents.approvingIds}
        rejectingDocumentIds={documents.rejectingIds}
        isDisabled={!isAuthenticated}
        canManageOrg={canManageOrg}
        currentUserId={me?.user_id}
        token={token}
        workspaceId={workspaceId}
        onNewChat={() => {
          chat.reset();
          setActiveChunkId(undefined);
          setIsPanelOpen(false);
          setShowUpload(false);
        }}
        onSelectSession={(id) => {
          setActiveChunkId(undefined);
          setIsPanelOpen(false);
          setShowUpload(false);
          void chat.loadSession(id);
        }}
        onDeleteSession={(id) => {
          if (id === chat.sessionId) chat.reset();
          void sessions.remove(id);
        }}
        onOpenUpload={() => setShowUpload(true)}
        onDismissUpload={documents.dismissUpload}
        documentsViewActive={showUpload}
        onSelectChats={() => setShowUpload(false)}
        onSelectDocuments={() => setShowUpload(true)}
        onDeleteDocument={(id) => void documents.remove(id)}
        onReprocessDocument={(id) => void documents.reprocess(id)}
        onApproveDocument={(id) => void documents.approve(id)}
        onRejectDocument={(id) => void documents.reject(id)}
        onDeleteWorkspace={() => {
          // Organization was deleted AND the owner's auth account was removed
          // by the backend. Sign out and redirect to the login page — do NOT
          // show the zero-org view, because the account itself is now gone.
          void signOut();
        }}
      />

      <main className="flex min-h-0 min-w-0 flex-1 flex-col">
        <header className="flex items-center justify-between border-b border-border bg-[#0F1A15] px-4 py-2.5">
          <div className="flex min-w-0 items-baseline gap-2">
            <h1 className="font-sans text-[11px] font-semibold uppercase tracking-[0.18em] text-foreground/80">OFFICE BRAIN</h1>
            {me?.workspace_name && (
              <span className="truncate text-xs text-muted">{me.workspace_name}</span>
            )}
          </div>
          <div className="flex items-center gap-1">
            <Button
              variant={isPanelOpen ? "ghost" : "secondary"}
              onClick={() => setIsPanelOpen((open) => !open)}
              aria-pressed={isPanelOpen}
              className={isPanelOpen ? "text-accent" : undefined}
            >
              <PanelRightOpen className="size-3.5" aria-hidden />
              Sources
              {activeTurn?.sources.length ? ` (${activeTurn.sources.length})` : ""}
            </Button>

            {isAuthenticated && (
              <Button
                variant="secondary"
                onClick={() => void signOut()}
                title={me?.email ?? undefined}
              >
                <LogOut className="size-3.5" aria-hidden />
                Sign out
              </Button>
            )}
          </div>
        </header>

        {!isAuthenticated && !isLoading && <AuthNotice />}

        {hasNoWorkspaces && <NoWorkspacesNotice onCreated={(id) => {
          setActiveWorkspaceId(id);
          setHasNoWorkspaces(false);
          chat.reset();
          setActiveChunkId(undefined);
          setIsPanelOpen(false);
        }} token={token} />}

        {!hasNoWorkspaces && showUpload && (
          <UploadInterface
            onUpload={(file, description) => void documents.upload(file, description)}
            onDismissUpload={documents.dismissUpload}
            uploads={documents.uploads}
            documents={documents.documents}
            onBack={() => setShowUpload(false)}
            disabled={!isAuthenticated}
            token={token}
            workspaceId={workspaceId}
          />
        )}

        {!hasNoWorkspaces && !showUpload && (
          <ChatPane
            turns={chat.turns}
            isStreaming={chat.isStreaming}
            isLoadingHistory={chat.isLoadingHistory}
            transportError={chat.transportError ?? sessions.error ?? documents.error}
            activeChunkId={activeChunkId}
            onSend={(message) => void chat.send(message)}
            onStop={chat.stop}
            onSelectCitation={(citation) => handleSelectCitation(citation.chunk_id)}
            disabled={!isAuthenticated}
          />
        )}
      </main>

      {isPanelOpen && (
        <SourcesPanel
          sources={activeTurn?.sources ?? []}
          citedChunkIds={citedChunkIds}
          activeChunkId={activeChunkId}
          onSelect={setActiveChunkId}
          onClose={() => setIsPanelOpen(false)}
        />
      )}
    </div>
  );
}

/**
 * Shown when no session is available.
 *
 * `proxy.ts` normally redirects an anonymous visitor to /login before this renders, so in
 * practice this covers the narrower case of a session that expired or was signed out in
 * another tab. Every backend endpoint requires a verified JWT (CLAUDE.md 4.6), so without
 * one the app can render but cannot do anything — saying so plainly beats letting every
 * action fail with a 401 the user has to interpret.
 *
 * The button calls `signOut()` — the same proper flow as the header's Sign-out button —
 * which clears Supabase cookies/session, resets local state, and navigates to /login.
 * A plain `<a href="/login">` would leave stale cookies behind, causing proxy.ts to
 * see a stale session and potentially redirect the user away from the login page before
 * they can sign in.
 */
function AuthNotice() {
  const { signOut } = useAuth();

  return (
    <div
      role="status"
      className="flex items-center gap-2 border-b border-warning/30 bg-warning-subtle px-4 py-2 text-xs text-warning"
    >
      <span>Your session has ended.</span>
      <button
        type="button"
        onClick={() => void signOut()}
        className="font-medium underline transition-colors hover:text-foreground focus-visible:ring-2 focus-visible:ring-accent focus-visible:outline-none"
      >
        Sign out
      </button>
      <span>to clear your session, then sign in again.</span>
    </div>
  );
}

/**
 * Shown when the user has no workspaces (all deleted, or membership removed).
 * Prompts them to create their first organization.
 */
function NoWorkspacesNotice({
  onCreated,
  token,
}: {
  onCreated: (workspaceId: string) => void;
  token?: string;
}) {
  const [name, setName] = useState("");
  const [isCreating, setIsCreating] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleCreate = async () => {
    if (!token || !name.trim()) return;
    setIsCreating(true);
    setError(null);
    try {
      const { createWorkspace } = await import("@/lib/api");
      const ws = await createWorkspace(name.trim(), { token });
      onCreated(ws.id);
    } catch (err) {
      setError(
        err instanceof Error ? err.message : "Failed to create organization.",
      );
    } finally {
      setIsCreating(false);
    }
  };

  return (
    <div className="flex flex-1 items-center justify-center p-8">
      <div className="w-full max-w-sm space-y-4 text-center">
        <Building2 className="mx-auto size-10 text-muted" aria-hidden />
        <h2 className="text-lg font-semibold">No organizations yet</h2>
        <p className="text-sm text-muted">
          You don't belong to any organizations. Create one to get started.
        </p>
        <div className="flex gap-2">
          <input
            type="text"
            value={name}
            onChange={(e) => setName(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") void handleCreate();
            }}
            placeholder="Organization name"
            maxLength={200}
            disabled={isCreating}
            className="flex-1 rounded-md border border-border bg-surface-raised px-3 py-2 text-sm text-foreground placeholder:text-muted/60 focus:border-accent focus:ring-1 focus:ring-accent focus:outline-none disabled:opacity-60"
          />
          <Button
            variant="primary"
            size="md"
            onClick={() => void handleCreate()}
            disabled={isCreating || !name.trim()}
          >
            {isCreating ? "Creating..." : "Create"}
          </Button>
        </div>
        {error && <p className="text-xs text-danger">{error}</p>}
      </div>
    </div>
  );
}
