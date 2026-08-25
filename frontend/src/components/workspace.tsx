"use client";

/**
 * The workspace: sidebar, chat pane, and the citations panel.
 *
 * State lives here rather than in a store — three hooks and two selections is not enough
 * to justify one, and keeping the wiring visible in a single component is what makes
 * the data flow readable.
 */

import { LogOut, PanelRightOpen } from "lucide-react";
import { useMemo, useState } from "react";
import { ChatPane } from "@/components/chat-pane";
import { Sidebar } from "@/components/sidebar";
import { SourcesPanel } from "@/components/sources-panel";
import { useAuth } from "@/lib/auth";
import { useChat } from "@/lib/hooks/use-chat";
import { useDocuments } from "@/lib/hooks/use-documents";
import { useSessions } from "@/lib/hooks/use-sessions";
import { cn } from "@/lib/utils";

/** Roles allowed to see org administration. Mirrors the backend role model. */
const ADMIN_ROLES = ["OWNER", "owner"];

export function Workspace() {
  const { token, isAuthenticated, isLoading, me, role, signOut } = useAuth();
  const [activeChunkId, setActiveChunkId] = useState<string | undefined>();
  const [isPanelOpen, setIsPanelOpen] = useState(false);

  const canManageOrg = Boolean(role && ADMIN_ROLES.includes(role));
  const workspaceId = me?.workspace_id;

  const sessions = useSessions(token);
  const documents = useDocuments(token);

  const chat = useChat({
    token,
    onSessionCreated: () => void sessions.refresh(),
    onTurnComplete: () => void sessions.refresh(),
  });

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
        }}
        onSelectSession={(id) => {
          setActiveChunkId(undefined);
          setIsPanelOpen(false);
          void chat.loadSession(id);
        }}
        onDeleteSession={(id) => {
          if (id === chat.sessionId) chat.reset();
          void sessions.remove(id);
        }}
        onUpload={(file) => void documents.upload(file)}
        onDismissUpload={documents.dismissUpload}
        onDeleteDocument={(id) => void documents.remove(id)}
        onReprocessDocument={(id) => void documents.reprocess(id)}
        onApproveDocument={(id) => void documents.approve(id)}
        onRejectDocument={(id) => void documents.reject(id)}
      />

      <main className="flex min-h-0 min-w-0 flex-1 flex-col">
        <header className="flex items-center justify-between border-b border-border bg-surface px-4 py-2.5">
          <div className="flex min-w-0 items-baseline gap-2">
            <h1 className="text-sm font-semibold">Knowledge Assistant</h1>
            {me?.workspace_name && (
              <span className="truncate text-xs text-muted">{me.workspace_name}</span>
            )}
          </div>
          <div className="flex items-center gap-1">
            <button
              type="button"
              onClick={() => setIsPanelOpen((open) => !open)}
              aria-pressed={isPanelOpen}
              className={cn(
                "flex items-center gap-1.5 rounded-md px-2 py-1 text-xs transition-colors",
                "focus-visible:ring-2 focus-visible:ring-accent focus-visible:outline-none",
                isPanelOpen
                  ? "bg-accent-subtle text-accent"
                  : "text-muted hover:text-foreground",
              )}
            >
              <PanelRightOpen className="size-3.5" aria-hidden />
              Sources
              {activeTurn?.sources.length ? ` (${activeTurn.sources.length})` : ""}
            </button>

            {isAuthenticated && (
              <button
                type="button"
                onClick={() => void signOut()}
                title={me?.email ?? undefined}
                className={cn(
                  "flex items-center gap-1.5 rounded-md px-2 py-1 text-xs text-muted",
                  "transition-colors hover:text-foreground",
                  "focus-visible:ring-2 focus-visible:ring-accent focus-visible:outline-none",
                )}
              >
                <LogOut className="size-3.5" aria-hidden />
                Sign out
              </button>
            )}
          </div>
        </header>

        {!isAuthenticated && !isLoading && <AuthNotice />}

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
