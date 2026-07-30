"use client";

/**
 * The workspace: sidebar, chat pane, and the citations panel.
 *
 * State lives here rather than in a store — three hooks and two selections is not enough
 * to justify one, and keeping the wiring visible in a single component is what makes the
 * data flow readable.
 */

import { PanelRightOpen } from "lucide-react";
import { useMemo, useState } from "react";
import { ChatPane } from "@/components/chat-pane";
import { Sidebar } from "@/components/sidebar";
import { SourcesPanel } from "@/components/sources-panel";
import { useAuth } from "@/lib/auth";
import { useChat } from "@/lib/hooks/use-chat";
import { useDocuments } from "@/lib/hooks/use-documents";
import { useSessions } from "@/lib/hooks/use-sessions";
import { cn } from "@/lib/utils";

export function Workspace() {
  const { token, isAuthenticated } = useAuth();
  const [activeChunkId, setActiveChunkId] = useState<string | undefined>();
  const [isPanelOpen, setIsPanelOpen] = useState(false);

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
        isDisabled={!isAuthenticated}
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
      />

      <main className="flex min-w-0 flex-1 flex-col">
        <header className="flex items-center justify-between border-b border-border bg-surface px-4 py-2.5">
          <h1 className="text-sm font-semibold">Knowledge Assistant</h1>
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
        </header>

        {!isAuthenticated && <AuthNotice />}

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
 * Shown when no token is available.
 *
 * Every backend endpoint requires a verified JWT (CLAUDE.md 4.6), so without one the app
 * can render but cannot do anything. Saying so plainly beats letting every action fail
 * with a 401 the user has to interpret.
 */
function AuthNotice() {
  return (
    <div
      role="status"
      className="border-b border-warning/30 bg-warning-subtle px-4 py-2 text-xs text-warning"
    >
      Not signed in — the assistant cannot reach your organization&apos;s data.
      Sign-in arrives in the next phase; set{" "}
      <code className="font-mono">NEXT_PUBLIC_DEV_JWT</code> to a development
      token to try it now.
    </div>
  );
}
