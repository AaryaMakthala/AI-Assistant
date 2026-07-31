"use client";

/**
 * Chat state over the Phase 7 supervisor stream.
 *
 * Hand-written rather than the Vercel AI SDK's `useChat`. That hook expects the AI SDK's
 * own protocol; this backend speaks a typed SSE envelope of its own, carrying frames the
 * SDK has no concept of — `routing`, `step`, `sources`, `citations`, and the executed SQL
 * on `done`. Adapting the stream to fit the SDK would mean discarding exactly the frames
 * the citation and routing UI is built on, so the parsing lives here instead (`lib/api/sse`).
 *
 * The invariant worth stating: **a turn is never presented as a finished answer unless it
 * actually finished.** `done` completes it and `error` marks it failed, each keeping
 * whatever text had streamed. A stream that simply stops — a dropped connection, the user
 * pressing stop, a proxy closing the response — settles the turn as complete *and*
 * incomplete, so the UI can say it ended early rather than either pulsing forever or
 * passing a fragment off as the whole answer.
 */

import { useCallback, useEffect, useRef, useState } from "react";
import {
  ApiError,
  listMessages,
  sendMessage,
  type AgentRoute,
  type ChatStreamEvent,
  type Citation,
  type Source,
  type TokenUsage,
} from "@/lib/api";

export type TurnStatus = "streaming" | "complete" | "failed";

/** One turn as the UI renders it. Wider than the wire `ChatMessage`: it also holds the
 * live streaming state a persisted row has no reason to carry. */
export interface Turn {
  id: string;
  role: "user" | "assistant";
  content: string;
  status: TurnStatus;
  /** Everything retrieved for this turn — the superset the citations index into. */
  sources: Source[];
  citations: Citation[];
  /** Which agents the supervisor consulted. Empty until the `routing` frame arrives. */
  routes: AgentRoute[];
  routeReason: string;
  /** The agent trace, newest last. Drives the routing indicator while streaming. */
  steps: string[];
  /** The validated SELECT behind a business-data answer. Empty when no SQL ran. */
  sqlQuery: string;
  usage?: TokenUsage;
  provider?: string;
  model?: string;
  /** False when no agent supplied evidence — the answer is an honest non-answer. */
  grounded?: boolean;
  /** Set on a failed turn. Safe to show; the backend phrases these for a reader. */
  error?: string;
  /** True when the turn stopped early and the text is partial. */
  incomplete?: boolean;
}

let turnCounter = 0;
/** Stable within a session and never colliding with a server UUID. `Date.now()` alone
 * collides when two turns start in the same millisecond. */
function nextTurnId(prefix: string): string {
  turnCounter += 1;
  return `${prefix}-${turnCounter}`;
}

function emptyAssistantTurn(): Turn {
  return {
    id: nextTurnId("assistant"),
    role: "assistant",
    content: "",
    status: "streaming",
    sources: [],
    citations: [],
    routes: [],
    routeReason: "",
    steps: [],
    sqlQuery: "",
  };
}

export interface UseChatOptions {
  token?: string;
  /** Fires when the backend creates a session, so the sidebar can refresh. */
  onSessionCreated?: (sessionId: string) => void;
  /** Fires when a turn completes, so the session list can re-sort by recency. */
  onTurnComplete?: () => void;
}

export function useChat({
  token,
  onSessionCreated,
  onTurnComplete,
}: UseChatOptions = {}) {
  const [turns, setTurns] = useState<Turn[]>([]);
  const [sessionId, setSessionId] = useState<string | undefined>();
  const [isStreaming, setIsStreaming] = useState(false);
  /** A failure that has no turn to attach to — the request never opened a stream. */
  const [transportError, setTransportError] = useState<string | undefined>();
  const [isLoadingHistory, setIsLoadingHistory] = useState(false);

  const abortRef = useRef<AbortController | null>(null);
  // Read inside the async send loop, which closes over the values it captured at call
  // time. A ref keeps the callbacks current without re-creating `send` on every render.
  const callbacksRef = useRef({ onSessionCreated, onTurnComplete });
  useEffect(() => {
    callbacksRef.current = { onSessionCreated, onTurnComplete };
  }, [onSessionCreated, onTurnComplete]);
  // Mirrors `sessionId` so the stream handler can compare against it without a state
  // updater — updaters must stay pure, and StrictMode calls them twice, which would fire
  // the sidebar refresh twice for one new session.
  const sessionIdRef = useRef<string | undefined>(undefined);

  // Abort an in-flight stream when the component goes away, so a backgrounded tab is not
  // left holding an open response the user will never see.
  useEffect(() => () => abortRef.current?.abort(), []);

  /** Apply an update to the assistant turn currently streaming (always the last one).
   *
   * A no-op when the last turn is not a streaming assistant turn. That matters for the
   * settle-on-exit path in `send`: aborting to switch conversations makes the stream
   * throw *after* `reset`/`loadSession` has already replaced the transcript, and without
   * this guard the old run would mark a turn belonging to the new one. */
  const updateStreamingTurn = useCallback((patch: (turn: Turn) => Turn) => {
    setTurns((current) => {
      const index = current.length - 1;
      const last = current[index];
      if (!last || last.role !== "assistant" || last.status !== "streaming") {
        return current;
      }
      const next = current.slice();
      next[index] = patch(last);
      return next;
    });
  }, []);

  const send = useCallback(
    async (message: string) => {
      const question = message.trim();
      if (!question || abortRef.current) return;

      setTransportError(undefined);
      const controller = new AbortController();
      abortRef.current = controller;
      setIsStreaming(true);

      setTurns((current) => [
        ...current,
        {
          id: nextTurnId("user"),
          role: "user",
          content: question,
          status: "complete",
          sources: [],
          citations: [],
          routes: [],
          routeReason: "",
          steps: [],
          sqlQuery: "",
        },
        emptyAssistantTurn(),
      ]);

      try {
        const stream = sendMessage({
          message: question,
          sessionId,
          token,
          signal: controller.signal,
        });

        for await (const event of stream) {
          applyEvent(event);
        }
      } catch (error) {
        // Only failures *before* the stream opened land here — the backend reports
        // anything later as an `error` frame, since it cannot retract a sent 200.
        if ((error as DOMException)?.name === "AbortError") {
          // Deliberate stop or unmount. The `finally` block below settles the turn as
          // incomplete, keeping whatever text arrived.
        } else {
          const detail =
            error instanceof ApiError
              ? error.message
              : "Could not reach the assistant. Check your connection and try again.";
          setTransportError(detail);
          // Drop the empty assistant turn: there is no partial answer to preserve, and an
          // empty bubble reads as the assistant having said nothing in reply.
          setTurns((current) => {
            const last = current[current.length - 1];
            return last?.role === "assistant" && !last.content
              ? current.slice(0, -1)
              : current;
          });
        }
      } finally {
        abortRef.current = null;
        setIsStreaming(false);
        // A stream that ended without `done` or `error` — a dropped connection, a proxy
        // closing the response — leaves the turn mid-flight. Left alone its indicator
        // would pulse forever, implying work that is not happening. It is settled as
        // incomplete instead: honest about having stopped early, and never presented as
        // a finished answer.
        updateStreamingTurn((turn) =>
          turn.role === "assistant" && turn.status === "streaming"
            ? { ...turn, status: "complete", incomplete: true }
            : turn,
        );
      }

      function applyEvent(event: ChatStreamEvent) {
        switch (event.type) {
          case "session":
            if (sessionIdRef.current !== event.session_id) {
              sessionIdRef.current = event.session_id;
              setSessionId(event.session_id);
              callbacksRef.current.onSessionCreated?.(event.session_id);
            }
            break;

          case "routing":
            updateStreamingTurn((turn) => ({
              ...turn,
              routes: event.routes,
              routeReason: event.reason,
            }));
            break;

          case "step":
            updateStreamingTurn((turn) => ({
              ...turn,
              steps: [...turn.steps, event.text],
            }));
            break;

          case "sources":
            updateStreamingTurn((turn) => ({ ...turn, sources: event.sources }));
            break;

          case "token":
            updateStreamingTurn((turn) => ({
              ...turn,
              content: turn.content + event.text,
            }));
            break;

          case "citations":
            updateStreamingTurn((turn) => ({ ...turn, citations: event.citations }));
            break;

          case "done":
            updateStreamingTurn((turn) => ({
              ...turn,
              status: "complete",
              routes: event.routes.length ? event.routes : turn.routes,
              sqlQuery: event.sql_query,
              usage: event.usage,
              provider: event.provider,
              model: event.model,
              grounded: event.grounded,
            }));
            callbacksRef.current.onTurnComplete?.();
            break;

          case "error":
            updateStreamingTurn((turn) => ({
              ...turn,
              status: "failed",
              error: event.detail,
              incomplete: event.partial,
            }));
            break;
        }
      }
    },
    [sessionId, token, updateStreamingTurn],
  );

  /** Stop generation, keeping whatever text has already arrived.
   *
   * Aborting makes the stream throw, and `send`'s `finally` settles the turn — so this
   * only has to cancel the request. */
  const stop = useCallback(() => {
    abortRef.current?.abort();
  }, []);

  /** Start a new conversation. */
  const reset = useCallback(() => {
    abortRef.current?.abort();
    abortRef.current = null;
    sessionIdRef.current = undefined;
    setTurns([]);
    setSessionId(undefined);
    setTransportError(undefined);
    setIsStreaming(false);
  }, []);

  /** Load an existing conversation's transcript into the pane. */
  const loadSession = useCallback(
    async (id: string) => {
      abortRef.current?.abort();
      abortRef.current = null;
      setIsStreaming(false);
      setTransportError(undefined);
      setIsLoadingHistory(true);
      sessionIdRef.current = id;
      setSessionId(id);
      setTurns([]);

      try {
        const { messages } = await listMessages(id, { token });
        setTurns(
          messages
            .filter((message) => message.role === "user" || message.role === "assistant")
            .map((message) => ({
              id: message.id,
              role: message.role as "user" | "assistant",
              content: message.content,
              // A stored turn is settled by definition; `incomplete` records that its
              // text is partial, which is a different claim from "still streaming".
              status: "complete" as const,
              // Only citations are persisted. The full retrieved set is not, so a
              // reloaded turn shows what was cited rather than inventing a superset.
              sources: message.citations,
              citations: message.citations,
              routes: message.routes,
              routeReason: "",
              steps: [],
              sqlQuery: message.sql_query,
              incomplete: message.incomplete,
            })),
        );
      } catch (error) {
        setTransportError(
          error instanceof ApiError
            ? error.message
            : "Could not load this conversation.",
        );
      } finally {
        setIsLoadingHistory(false);
      }
    },
    [token],
  );

  return {
    turns,
    sessionId,
    isStreaming,
    isLoadingHistory,
    transportError,
    send,
    stop,
    reset,
    loadSession,
  };
}
