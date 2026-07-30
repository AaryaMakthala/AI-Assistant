/**
 * Wire types for the backend API.
 *
 * These mirror the Pydantic models in `backend/app/api/`. They are hand-written rather
 * than generated because the surface is small and a generator would be one more build
 * step to keep working; if the surface grows past this, generate from the OpenAPI schema
 * instead of letting these drift.
 */

/** A source that was retrieved and sent to the model. */
export interface Source {
  /** 1-based, matching the bracketed numbers the model is told to cite. */
  number: number;
  document_id: string;
  chunk_id: string;
  filename: string;
  /** Null for formats without pages (CSV, TXT). */
  page: number | null;
  /** Pre-rendered display label, e.g. "refund_policy.pdf · page 4". */
  label: string;
  excerpt: string;
  /** Retrieval similarity 0–1 — how close the passage was to the question, not how
   * strongly it supports the sentence citing it. */
  score: number;
}

/**
 * A source the answer actually cited. Structurally identical to {@link Source}: a
 * citation is the subset of what was retrieved that the model referenced.
 */
export type Citation = Source;

export interface TokenUsage {
  prompt_tokens: number;
  completion_tokens: number;
  total_tokens: number;
}

export interface ChatSession {
  id: string;
  title: string | null;
  created_at: string;
  updated_at: string;
}

export interface ChatSessionListResponse {
  sessions: ChatSession[];
}

export interface ChatMessage {
  id: string;
  role: "user" | "assistant" | "system" | "tool";
  content: string;
  citations: Citation[];
  created_at: string;
  /** Generation stopped early; the content is a partial answer. */
  incomplete: boolean;
  /** Which sources the supervisor consulted for this turn. Empty on user turns. */
  routes: AgentRoute[];
  /** The validated SELECT behind a business-data answer; empty when no SQL ran. */
  sql_query: string;
}

export interface ChatMessageListResponse {
  messages: ChatMessage[];
}

export interface DocumentSummary {
  id: string;
  filename: string;
  mime_type: string;
  size_bytes: number;
  status: "pending" | "processing" | "ready" | "failed";
  page_count: number | null;
  error_message: string | null;
  created_at: string;
}

export interface DocumentListResponse {
  documents: DocumentSummary[];
}

export interface UploadAcceptedResponse {
  document: DocumentSummary;
  task_id: string | null;
}

/* --- SSE frames emitted by POST /chat --- */

/**
 * A source the supervisor can route a question to. `direct` is not an agent: it means the
 * question needed no lookup at all.
 */
export type AgentRoute =
  | "documents"
  | "business_data"
  | "external"
  | "direct";

/** Sent first, before generation. Carries the id of a newly created session. */
export interface SessionEvent {
  type: "session";
  session_id: string;
}

/**
 * The supervisor's routing decision, sent once as soon as it is made and before any agent
 * runs. Drives the "thinking/routing" indicator.
 */
export interface RoutingEvent {
  type: "routing";
  routes: AgentRoute[];
  /** The router's short justification. May be empty. */
  reason: string;
}

/**
 * One line of the agent trace, emitted as each node completes — "documents: retrieved 6
 * passage(s)". Progress detail, not part of the answer.
 */
export interface StepEvent {
  type: "step";
  text: string;
}

/** Everything retrieved, sent before the first token so the UI can show what is being
 * consulted while it waits. An empty list means the answer rests on nothing. */
export interface SourcesEvent {
  type: "sources";
  sources: Source[];
}

export interface TokenEvent {
  type: "token";
  text: string;
}

/** What the answer actually cited, resolved against the sources above. */
export interface CitationsEvent {
  type: "citations";
  citations: Citation[];
}

/** Terminal success frame. */
export interface DoneEvent {
  type: "done";
  usage: TokenUsage;
  provider: string;
  model: string;
  /** False when nothing was retrieved — the answer is an honest non-answer. */
  grounded: boolean;
  /** Which agents contributed to this answer. */
  routes: AgentRoute[];
  /** The validated SELECT the SQL agent ran; empty string when none did. */
  sql_query: string;
}

/** Terminal failure frame. Arrives after a 200 header, so it is not an HTTP error. */
export interface ErrorEvent {
  type: "error";
  detail: string;
  /** True when some text was already streamed before the failure. */
  partial: boolean;
}

export type ChatStreamEvent =
  | SessionEvent
  | RoutingEvent
  | StepEvent
  | SourcesEvent
  | TokenEvent
  | CitationsEvent
  | DoneEvent
  | ErrorEvent;
