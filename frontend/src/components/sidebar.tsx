"use client";

/**
 * The left sidebar: new chat, conversation history, document library.
 *
 * History and library share the rail through a tab switch rather than stacking, so
 * neither collapses to a few rows on a short screen.
 */

import { FolderOpen, MessageSquare, Plus, Trash2 } from "lucide-react";
import Link from "next/link";
import { useState } from "react";
import { DocumentLibrary } from "./document-library";
import { MembersPanel } from "./members-panel";
import type { ChatSession, DocumentSummary } from "@/lib/api";
import type { UploadState } from "@/lib/hooks/use-documents";
import { cn, formatRelativeTime } from "@/lib/utils";

type Tab = "chats" | "documents" | "members";

export function Sidebar({
  sessions,
  activeSessionId,
  documents,
  uploads,
  deletingDocumentIds,
  approvingDocumentIds,
  rejectingDocumentIds,
  isDisabled,
  canManageOrg,
  currentUserId,
  token,
  workspaceId,
  onNewChat,
  onSelectSession,
  onDeleteSession,
  onUpload,
  onDismissUpload,
  onDeleteDocument,
  onReprocessDocument,
  onApproveDocument,
  onRejectDocument,
}: {
  sessions: ChatSession[];
  activeSessionId?: string;
  documents: DocumentSummary[];
  uploads: UploadState[];
  deletingDocumentIds?: ReadonlySet<string>;
  approvingDocumentIds?: ReadonlySet<string>;
  rejectingDocumentIds?: ReadonlySet<string>;
  isDisabled?: boolean;
  /** Owners and admins only. Presentation; the server gates the data independently. */
  canManageOrg?: boolean;
  /** Whose uploads count as personal in the library's "My Docs" section. */
  currentUserId?: string;
  token?: string;
  workspaceId?: string;
  onNewChat: () => void;
  onSelectSession: (id: string) => void;
  onDeleteSession: (id: string) => void;
  onUpload: (file: File, description?: string) => void;
  onDismissUpload: (id: string) => void;
  onDeleteDocument: (id: string) => void;
  onReprocessDocument: (id: string) => void;
  onApproveDocument: (id: string) => void;
  onRejectDocument: (id: string) => void;
}) {
  const [tab, setTab] = useState<Tab>("chats");

  const activeTab: Tab = tab === "members" && !canManageOrg ? "chats" : tab;

  const tabs: ReadonlyArray<readonly [Tab, string]> = canManageOrg
    ? ([
        ["chats", "Chats"],
        ["documents", "Docs"],
        ["members", "Members"],
      ] as const)
    : ([
        ["chats", "Chats"],
        ["documents", "Documents"],
      ] as const);

  return (
    <aside className="flex h-full w-64 shrink-0 flex-col border-r border-border bg-surface">
      <div className="p-3">
        <button
          type="button"
          onClick={onNewChat}
          className={cn(
            "flex w-full items-center justify-center gap-1.5 rounded-lg bg-accent",
            "px-3 py-2 text-sm font-medium text-accent-foreground transition-opacity",
            "hover:opacity-90 focus-visible:ring-2 focus-visible:ring-accent focus-visible:outline-none",
          )}
        >
          <Plus className="size-4" aria-hidden />
          New chat
        </button>
      </div>

      <div
        role="tablist"
        aria-label="Sidebar sections"
        className="flex gap-1 px-3 pb-2"
      >
        {tabs.map(([value, label]) => (
          <button
            key={value}
            type="button"
            role="tab"
            aria-selected={activeTab === value}
            onClick={() => setTab(value)}
            className={cn(
              "flex-1 rounded-md px-2 py-1.5 text-xs font-medium transition-colors",
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

      {activeTab === "chats" && (
        <nav className="min-h-0 flex-1 overflow-y-auto px-3 pb-3">
          {sessions.length === 0 ? (
            <p className="px-1 py-6 text-center text-xs text-muted">
              No conversations yet.
            </p>
          ) : (
            <ul className="space-y-0.5">
              {sessions.map((session) => (
                <li key={session.id}>
                  <SessionRow
                    session={session}
                    isActive={session.id === activeSessionId}
                    onSelect={onSelectSession}
                    onDelete={onDeleteSession}
                  />
                </li>
              ))}
            </ul>
          )}
        </nav>
      )}

      {activeTab === "documents" && (
        <>
          <DocumentLibrary
            documents={documents}
            uploads={uploads}
            deletingIds={deletingDocumentIds}
            approvingIds={approvingDocumentIds}
            rejectingIds={rejectingDocumentIds}
            currentUserId={currentUserId}
            canManageOrg={canManageOrg}
            onUpload={onUpload}
            onDismissUpload={onDismissUpload}
            onDelete={onDeleteDocument}
            onReprocess={onReprocessDocument}
            onApprove={onApproveDocument}
            onReject={onRejectDocument}
            disabled={isDisabled}
          />
          {canManageOrg && (
            <div className="border-t border-border px-3 py-2">
              <Link
                href="/documents"
                className={cn(
                  "flex items-center gap-1.5 rounded-md px-2 py-1.5 text-xs text-muted",
                  "transition-colors hover:text-foreground focus-visible:ring-2",
                  "focus-visible:ring-accent focus-visible:outline-none",
                )}
              >
                <FolderOpen className="size-3.5" aria-hidden />
                Manage all documents
              </Link>
            </div>
          )}
        </>
      )}

      {activeTab === "members" && (
        <nav className="min-h-0 flex-1 overflow-y-auto px-3 pb-3">
          <MembersPanel token={token} workspaceId={workspaceId} />
        </nav>
      )}
    </aside>
  );
}

function SessionRow({
  session,
  isActive,
  onSelect,
  onDelete,
}: {
  session: ChatSession;
  isActive: boolean;
  onSelect: (id: string) => void;
  onDelete: (id: string) => void;
}) {
  const [isConfirming, setIsConfirming] = useState(false);

  return (
    <div
      className={cn(
        "group flex items-center gap-1 rounded-md pr-1 transition-colors",
        isActive ? "bg-surface-raised" : "hover:bg-surface-raised",
      )}
    >
      <button
        type="button"
        onClick={() => onSelect(session.id)}
        aria-current={isActive}
        className={cn(
          "flex min-w-0 flex-1 items-center gap-2 rounded-md px-2 py-2 text-left",
          "focus-visible:ring-2 focus-visible:ring-accent focus-visible:outline-none",
        )}
      >
        <MessageSquare
          className={cn("size-3.5 shrink-0", isActive ? "text-accent" : "text-muted")}
          aria-hidden
        />
        <span className="min-w-0 flex-1">
          <span className="block truncate text-xs">
            {session.title || "Untitled conversation"}
          </span>
          <span className="block text-[0.6875rem] text-muted">
            {formatRelativeTime(session.updated_at)}
          </span>
        </span>
      </button>

      {isConfirming ? (
        <span className="flex shrink-0 items-center gap-1 pr-0.5">
          <button
            type="button"
            onClick={() => {
              setIsConfirming(false);
              onDelete(session.id);
            }}
            className="rounded px-1 text-[0.6875rem] font-medium text-danger hover:underline"
          >
            Delete
          </button>
          <button
            type="button"
            onClick={() => setIsConfirming(false)}
            className="rounded px-1 text-[0.6875rem] text-muted hover:underline"
          >
            Cancel
          </button>
        </span>
      ) : (
        <button
          type="button"
          onClick={() => setIsConfirming(true)}
          aria-label={`Delete ${session.title || "conversation"}`}
          className={cn(
            "flex size-6 shrink-0 items-center justify-center rounded text-muted opacity-0",
            "transition-opacity group-hover:opacity-100 focus-visible:opacity-100",
            "hover:text-danger focus-visible:ring-2 focus-visible:ring-accent focus-visible:outline-none",
          )}
        >
          <Trash2 className="size-3.5" aria-hidden />
        </button>
      )}
    </div>
  );
}
