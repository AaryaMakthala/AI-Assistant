"use client";

/** The chat-history sidebar's data: list, refresh, delete. */

import { useCallback, useEffect, useRef, useState } from "react";
import {
  ApiError,
  deleteSession as deleteSessionRequest,
  listSessions,
  type ChatSession,
} from "@/lib/api";

export function useSessions(token?: string, workspaceId?: string) {
  const [sessions, setSessions] = useState<ChatSession[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | undefined>();
  const mountedRef = useRef(true);

  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
    };
  }, []);

  const refresh = useCallback(async () => {
    if (!token) return;
    setIsLoading(true);
    try {
      const { sessions: rows } = await listSessions({ token, workspaceId, limit: 50 });
      if (mountedRef.current) {
        setSessions(rows);
        setError(undefined);
      }
    } catch (caught) {
      if (mountedRef.current) {
        setError(
          caught instanceof ApiError
            ? caught.message
            : "Could not load your conversations.",
        );
      }
    } finally {
      if (mountedRef.current) setIsLoading(false);
    }
  }, [token, workspaceId]);

  useEffect(() => {
    // Fetch-on-mount: `refresh` flips the loading flag before awaiting, which the rule
    // reads as a cascading render. The cascade is the point here — one extra render to
    // show the spinner, then one to show the rows.
    // eslint-disable-next-line react-hooks/set-state-in-effect
    void refresh();
  }, [refresh]);

  const remove = useCallback(
    async (id: string) => {
      // Removed from the list first: the row is gone from the server either way, and
      // waiting on the round trip leaves a deleted conversation clickable.
      const previous = sessions;
      setSessions((current) => current.filter((session) => session.id !== id));
      try {
        await deleteSessionRequest(id, { token, workspaceId });
      } catch (caught) {
        if (!mountedRef.current) return;
        setSessions(previous);
        setError(
          caught instanceof ApiError
            ? caught.message
            : "Could not delete that conversation.",
        );
      }
    },
    [sessions, token, workspaceId],
  );

  return { sessions, isLoading, error, refresh, remove };
}
