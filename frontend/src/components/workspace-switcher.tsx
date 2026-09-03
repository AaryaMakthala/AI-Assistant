"use client";

/**
 * Workspace switcher: shows the current workspace, allows switching between
 * workspaces, and lets the user create new ones (up to a max of 4).
 *
 * Placed at the top of the sidebar, below the "New chat" button.
 */

import { Building2, ChevronDown, Loader2, Plus, Check } from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";
import {
  listWorkspaces,
  createWorkspace,
  ApiError,
  type Workspace,
} from "@/lib/api";
import { cn } from "@/lib/utils";

interface WorkspaceSwitcherProps {
  token?: string;
  currentWorkspaceId?: string;
  onSwitch: (workspaceId: string) => void;
}

export function WorkspaceSwitcher({
  token,
  currentWorkspaceId,
  onSwitch,
}: WorkspaceSwitcherProps) {
  const [workspaces, setWorkspaces] = useState<Workspace[]>([]);
  const [isOpen, setIsOpen] = useState(false);
  const [isCreating, setIsCreating] = useState(false);
  const [newName, setNewName] = useState("");
  const [createError, setCreateError] = useState<string | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const dropdownRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);



  // Load workspaces when dropdown opens
  const loadWorkspaces = useCallback(async () => {
    if (!token) return;
    try {
      const resp = await listWorkspaces({ token });
      setWorkspaces(resp.workspaces);
    } catch {
      // Silent — the switcher just won't show options
    }
  }, [token]);

  useEffect(() => {
    if (isOpen) {
      void loadWorkspaces();
    }
  }, [isOpen, loadWorkspaces]);

  // Close dropdown on outside click
  useEffect(() => {
    if (!isOpen) return;
    const handler = (e: MouseEvent) => {
      if (dropdownRef.current && !dropdownRef.current.contains(e.target as Node)) {
        setIsOpen(false);
        setIsCreating(false);
        setCreateError(null);
        setNewName("");
      }
    };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, [isOpen]);

  // Focus input when creating
  useEffect(() => {
    if (isCreating) {
      inputRef.current?.focus();
    }
  }, [isCreating]);

  const handleCreate = async () => {
    if (!token || !newName.trim()) return;
    setIsSubmitting(true);
    setCreateError(null);
    try {
      const ws = await createWorkspace(newName.trim(), { token });
      setWorkspaces((prev) => [...prev, ws]);
      setNewName("");
      setIsCreating(false);
      onSwitch(ws.id);
      setIsOpen(false);
    } catch (err) {
      if (err instanceof ApiError) {
        setCreateError(err.message);
      } else {
        setCreateError("Failed to create organization.");
      }
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleCreateKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter") {
      e.preventDefault();
      void handleCreate();
    } else if (e.key === "Escape") {
      setIsCreating(false);
      setCreateError(null);
      setNewName("");
    }
  };

  const currentWorkspace = workspaces.find((ws) => ws.id === currentWorkspaceId);
  const displayName = currentWorkspace?.name ?? "Workspace";

  return (
    <div className="relative" ref={dropdownRef}>
      {/* Trigger button */}
      <button
        type="button"
        onClick={() => setIsOpen(!isOpen)}
        className={cn(
          "flex w-full items-center gap-2 rounded-lg border border-border px-3 py-2",
          "text-left text-sm transition-colors",
          "hover:bg-surface-raised focus-visible:ring-2 focus-visible:ring-accent focus-visible:outline-none",
          isOpen && "bg-surface-raised",
        )}
      >
        <Building2 className="size-4 shrink-0 text-muted" aria-hidden />
        <span className="min-w-0 flex-1 truncate font-medium">{displayName}</span>
        <ChevronDown
          className={cn(
            "size-3.5 shrink-0 text-muted transition-transform",
            isOpen && "rotate-180",
          )}
          aria-hidden
        />
      </button>

      {/* Dropdown */}
      {isOpen && (
        <div className="absolute left-0 right-0 top-full z-50 mt-1 overflow-hidden rounded-lg border border-border bg-surface shadow-lg">


          {/* Workspace list */}
          <div className="max-h-48 overflow-y-auto py-1">
            {workspaces.map((ws) => (
              <button
                key={ws.id}
                type="button"
                onClick={() => {
                  onSwitch(ws.id);
                  setIsOpen(false);
                }}
                className={cn(
                  "flex w-full items-center gap-2 px-3 py-2 text-left text-sm transition-colors",
                  "hover:bg-surface-raised focus-visible:bg-surface-raised focus-visible:outline-none",
                  ws.id === currentWorkspaceId && "bg-surface-raised font-medium",
                )}
              >
                {ws.id === currentWorkspaceId && (
                  <Check className="size-3.5 shrink-0 text-accent" aria-hidden />
                )}
                <span className="min-w-0 flex-1 truncate">{ws.name}</span>
              </button>
            ))}
          </div>

          {/* Create new / limit reached */}
          <div className="border-t border-border">
            {isCreating ? (
              <div className="p-2">
                <div className="flex gap-1.5">
                  <input
                    ref={inputRef}
                    type="text"
                    value={newName}
                    onChange={(e) => setNewName(e.target.value)}
                    onKeyDown={handleCreateKeyDown}
                    placeholder="Organization name"
                    maxLength={200}
                    disabled={isSubmitting}
                    className={cn(
                      "flex-1 rounded-md border border-border bg-surface-raised px-2.5 py-1.5",
                      "text-sm text-foreground placeholder:text-muted/60",
                      "focus:border-accent focus:ring-1 focus:ring-accent focus:outline-none",
                      "disabled:opacity-60",
                    )}
                  />
                  <button
                    type="button"
                    onClick={() => void handleCreate()}
                    disabled={isSubmitting || !newName.trim()}
                    className={cn(
                      "rounded-md bg-accent px-2.5 py-1.5 text-xs font-medium text-accent-foreground",
                      "transition-opacity hover:opacity-90 disabled:opacity-60",
                    )}
                  >
                    {isSubmitting ? (
                      <Loader2 className="size-3.5 animate-spin" aria-hidden />
                    ) : (
                      "Create"
                    )}
                  </button>
                  <button
                    type="button"
                    onClick={() => {
                      setIsCreating(false);
                      setCreateError(null);
                      setNewName("");
                    }}
                    className="rounded-md px-2 py-1.5 text-xs text-muted hover:text-foreground"
                  >
                    Cancel
                  </button>
                </div>
                {createError && (
                  <p className="mt-1.5 text-[11px] text-danger">{createError}</p>
                )}
              </div>
            ) : (
              <button
                type="button"
                onClick={() => setIsCreating(true)}
                className={cn(
                  "flex w-full items-center gap-2 px-3 py-2.5 text-left text-sm transition-colors",
                  "hover:bg-surface-raised focus-visible:bg-surface-raised focus-visible:outline-none",
                )}
                title="Create a new organization"
              >
                <Plus className="size-3.5 shrink-0 text-muted" aria-hidden />
                <span className="text-xs text-muted">Create organization</span>
              </button>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
