/**
 * Backend API client.
 *
 * The auth token is supplied per call rather than read from a module-level store: Phase 9
 * wires up Supabase Auth, and this keeps the two decoupled until then.
 */

import { parseEventStream } from "./sse";
import type {
  ChatMessageListResponse,
  ChatSessionListResponse,
  ChatStreamEvent,
  DocumentListResponse,
  DocumentSummary,
  UploadAcceptedResponse,
} from "./types";

const BASE_URL = (
  process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000"
).replace(/\/$/, "");

/** An error carrying the backend's request_id, which server logs are keyed by. */
export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
    readonly requestId?: string,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

export interface RequestOptions {
  token?: string;
  signal?: AbortSignal;
}

function authHeaders(token?: string): HeadersInit {
  return token ? { Authorization: `Bearer ${token}` } : {};
}

async function failure(response: Response): Promise<ApiError> {
  let detail = `Request failed with status ${response.status}`;
  let requestId = response.headers.get("x-request-id") ?? undefined;
  try {
    const body = (await response.json()) as {
      detail?: string;
      request_id?: string;
    };
    if (body.detail) detail = body.detail;
    if (body.request_id) requestId = body.request_id;
  } catch {
    // A non-JSON error body (a proxy's HTML 502, say) leaves the status-derived message.
  }
  return new ApiError(detail, response.status, requestId);
}

async function getJson<T>(path: string, options: RequestOptions = {}): Promise<T> {
  const response = await fetch(`${BASE_URL}${path}`, {
    headers: authHeaders(options.token),
    signal: options.signal,
  });
  if (!response.ok) throw await failure(response);
  return (await response.json()) as T;
}

export interface SendMessageArgs extends RequestOptions {
  message: string;
  /** Omit to start a new conversation; the `session` frame returns the new id. */
  sessionId?: string;
}

/**
 * Ask a question and yield the answer as it streams.
 *
 * Terminal failures arrive as an `error` event rather than a rejection once the response
 * headers are out — the server cannot change a 200 it has already sent. Only failures
 * *before* the stream opens throw.
 */
export async function* sendMessage({
  message,
  sessionId,
  token,
  signal,
}: SendMessageArgs): AsyncGenerator<ChatStreamEvent> {
  const response = await fetch(`${BASE_URL}/chat`, {
    method: "POST",
    headers: {
      ...authHeaders(token),
      "Content-Type": "application/json",
      Accept: "text/event-stream",
    },
    body: JSON.stringify({
      message,
      ...(sessionId ? { session_id: sessionId } : {}),
    }),
    signal,
  });

  if (!response.ok) throw await failure(response);
  if (!response.body) {
    throw new ApiError("The server returned an empty stream.", response.status);
  }

  yield* parseEventStream(response.body, signal);
}

export function listSessions(
  options: RequestOptions & { limit?: number; offset?: number } = {},
): Promise<ChatSessionListResponse> {
  const query = new URLSearchParams();
  if (options.limit !== undefined) query.set("limit", String(options.limit));
  if (options.offset !== undefined) query.set("offset", String(options.offset));
  const suffix = query.size ? `?${query}` : "";
  return getJson<ChatSessionListResponse>(`/chat/sessions${suffix}`, options);
}

export function listMessages(
  sessionId: string,
  options: RequestOptions & { limit?: number } = {},
): Promise<ChatMessageListResponse> {
  const suffix = options.limit === undefined ? "" : `?limit=${options.limit}`;
  return getJson<ChatMessageListResponse>(
    `/chat/sessions/${encodeURIComponent(sessionId)}/messages${suffix}`,
    options,
  );
}

export async function deleteSession(
  sessionId: string,
  options: RequestOptions = {},
): Promise<void> {
  const response = await fetch(
    `${BASE_URL}/chat/sessions/${encodeURIComponent(sessionId)}`,
    {
      method: "DELETE",
      headers: authHeaders(options.token),
      signal: options.signal,
    },
  );
  if (!response.ok) throw await failure(response);
}

export function listDocuments(
  options: RequestOptions & { limit?: number; offset?: number } = {},
): Promise<DocumentListResponse> {
  const query = new URLSearchParams();
  if (options.limit !== undefined) query.set("limit", String(options.limit));
  if (options.offset !== undefined) query.set("offset", String(options.offset));
  const suffix = query.size ? `?${query}` : "";
  return getJson<DocumentListResponse>(`/documents${suffix}`, options);
}

export function getDocument(
  documentId: string,
  options: RequestOptions = {},
): Promise<DocumentSummary> {
  return getJson<DocumentSummary>(
    `/documents/${encodeURIComponent(documentId)}`,
    options,
  );
}

export async function uploadDocument(
  file: File,
  options: RequestOptions = {},
): Promise<UploadAcceptedResponse> {
  const form = new FormData();
  form.append("file", file);

  // No Content-Type header: the browser must set it so the multipart boundary matches.
  const response = await fetch(`${BASE_URL}/documents`, {
    method: "POST",
    headers: authHeaders(options.token),
    body: form,
    signal: options.signal,
  });
  if (!response.ok) throw await failure(response);
  return (await response.json()) as UploadAcceptedResponse;
}
