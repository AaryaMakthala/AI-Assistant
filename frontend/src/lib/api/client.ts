/**
 * Backend API client.
 *
 * The auth token is supplied per call rather than read from a module-level store. That
 * keeps this module free of any dependency on how a session is obtained: `lib/auth.tsx`
 * owns the Supabase session and hands the current access token to each call, so a refresh
 * is picked up automatically and nothing here caches a token past its expiry.
 */

import { parseEventStream } from "./sse";
import type {
  ChatMessageListResponse,
  ChatSessionListResponse,
  ChatStreamEvent,
  DocumentListResponse,
  DocumentSummary,
  InvitationListResponse,
  MeResponse,
  OrgMemberListResponse,
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
  options: RequestOptions & {
    limit?: number;
    offset?: number;
  } = {},
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

/**
 * Delete a document and everything derived from it.
 *
 * The backend removes the row, its chunks and vectors, its failure trail and the stored
 * bytes. A 404 here means the document is already gone — which the caller may reasonably
 * treat as success, since the desired end state holds either way.
 */
export async function deleteDocument(
  documentId: string,
  options: RequestOptions = {},
): Promise<void> {
  const response = await fetch(
    `${BASE_URL}/documents/${encodeURIComponent(documentId)}`,
    {
      method: "DELETE",
      headers: authHeaders(options.token),
      signal: options.signal,
    },
  );
  if (!response.ok) throw await failure(response);
}

/**
 * Approve a pending document (owner only).
 *
 * PENDING → ingest inline → READY + chunks, atomically on the backend.
 */
export async function approveDocument(
  documentId: string,
  options: RequestOptions = {},
): Promise<UploadAcceptedResponse> {
  const response = await fetch(
    `${BASE_URL}/documents/${encodeURIComponent(documentId)}/approve`,
    {
      method: "POST",
      headers: authHeaders(options.token),
      signal: options.signal,
    },
  );
  if (!response.ok) throw await failure(response);
  return (await response.json()) as UploadAcceptedResponse;
}

/**
 * Reject a pending document (owner only).
 *
 * PENDING → REJECTED, never ingested.
 */
export async function rejectDocument(
  documentId: string,
  options: RequestOptions = {},
): Promise<DocumentSummary> {
  const response = await fetch(
    `${BASE_URL}/documents/${encodeURIComponent(documentId)}/reject`,
    {
      method: "POST",
      headers: authHeaders(options.token),
      signal: options.signal,
    },
  );
  if (!response.ok) throw await failure(response);
  return (await response.json()) as DocumentSummary;
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

export interface UploadProgressOptions extends RequestOptions {
  /** Fraction of bytes sent, 0–1. Called repeatedly while the body uploads. */
  onProgress?: (fraction: number) => void;
}

/**
 * Upload with byte-level progress.
 *
 * XMLHttpRequest rather than `fetch`, because `fetch` still has no upload-progress
 * event — the request-body stream is not observable, so a fetch-based upload can only
 * show an indeterminate spinner. For a 25 MB PDF on a slow connection that is the
 * difference between "working" and "frozen".
 *
 * Note this reports only the *transfer*. Ingestion is synchronous on the backend
 * (Phase 3), and the progress of extraction/embedding is not reported — the user
 * sees a processing spinner after the upload completes.
 */
export function uploadDocumentWithProgress(
  file: File,
  options: UploadProgressOptions = {},
): Promise<UploadAcceptedResponse> {
  return new Promise((resolve, reject) => {
    const form = new FormData();
    form.append("file", file);

    const request = new XMLHttpRequest();
    request.open("POST", `${BASE_URL}/documents`);
    if (options.token) {
      request.setRequestHeader("Authorization", `Bearer ${options.token}`);
    }

    const abort = () => request.abort();
    options.signal?.addEventListener("abort", abort);
    const cleanup = () => options.signal?.removeEventListener("abort", abort);

    request.upload.addEventListener("progress", (event) => {
      if (event.lengthComputable && event.total > 0) {
        options.onProgress?.(event.loaded / event.total);
      }
    });

    request.addEventListener("load", () => {
      cleanup();
      let body: { detail?: string; request_id?: string } = {};
      try {
        body = JSON.parse(request.responseText);
      } catch {
        // A non-JSON body (a proxy's HTML error page) leaves the status-derived message.
      }
      if (request.status >= 200 && request.status < 300) {
        options.onProgress?.(1);
        resolve(body as unknown as UploadAcceptedResponse);
        return;
      }
      reject(
        new ApiError(
          body.detail ?? `Upload failed with status ${request.status}`,
          request.status,
          body.request_id ??
            request.getResponseHeader("x-request-id") ??
            undefined,
        ),
      );
    });

    request.addEventListener("error", () => {
      cleanup();
      reject(new ApiError("The upload could not reach the server.", 0));
    });

    request.addEventListener("abort", () => {
      cleanup();
      reject(new DOMException("Upload aborted", "AbortError"));
    });

    request.send(form);
  });
}

/** The authenticated caller, per the server. The only trustworthy source of the role. */
export function getMe(options: RequestOptions = {}): Promise<MeResponse> {
  return getJson<MeResponse>("/me", options);
}

/**
 * Members of the caller's workspace. Requires workspace_id as a path parameter.
 * 403s for anyone not in the workspace — the gate is server-side.
 */
export function listWorkspaceMembers(
  workspaceId: string,
  options: RequestOptions = {},
): Promise<OrgMemberListResponse> {
  return getJson<OrgMemberListResponse>(
    `/workspaces/${encodeURIComponent(workspaceId)}/members`,
    options,
  );
}

/**
 * Pending invitations for a workspace. Owner only — 403 for anyone else.
 */
export function listInvitations(
  workspaceId: string,
  options: RequestOptions = {},
): Promise<InvitationListResponse> {
  return getJson<InvitationListResponse>(
    `/workspaces/${encodeURIComponent(workspaceId)}/invitations`,
    options,
  );
}

/**
 * Create an invitation. Owner only — 403 for anyone else.
 */
export async function createInvitation(
  workspaceId: string,
  email: string,
  options: RequestOptions = {},
): Promise<InvitationListResponse["invitations"][number]> {
  const response = await fetch(
    `${BASE_URL}/workspaces/${encodeURIComponent(workspaceId)}/invitations`,
    {
      method: "POST",
      headers: {
        ...authHeaders(options.token),
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ email }),
      signal: options.signal,
    },
  );
  if (!response.ok) throw await failure(response);
  return (await response.json()) as InvitationListResponse["invitations"][number];
}

/**
 * Accept an invitation. The user must be authenticated.
 */
export async function acceptInvitation(
  invitationId: string,
  options: RequestOptions = {},
): Promise<OrgMemberListResponse["members"][number]> {
  const response = await fetch(
    `${BASE_URL}/invitations/${encodeURIComponent(invitationId)}/accept`,
    {
      method: "POST",
      headers: authHeaders(options.token),
      signal: options.signal,
    },
  );
  if (!response.ok) throw await failure(response);
  return (await response.json()) as OrgMemberListResponse["members"][number];
}
