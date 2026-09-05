# System Reference — Multi-Tenant Knowledge Assistant

**Last updated:** 2026-09-04
**Branch:** `main`

---

## 1. Architecture Overview

A multi-tenant company knowledge assistant. A single deployment hosts many independent company workspaces, each fully isolated. Employees ask natural-language questions in a chat interface; the assistant answers only from that workspace's approved documents using hybrid retrieval, local reranking, and backend-verified citations.

### Request Flow (Chat)

```
User Question
     │
     ▼
Authentication (Supabase Auth JWT) + workspace authorization (members table)
     │
     ▼
Intent Classification (regex fast-path → LLM 5-route router)
  ├── GREETING / IDENTITY / PERMISSIONS / APP_HELP / CONVERSATION_HISTORY
  │   └── direct response, no retrieval, no LLM generation call
  ├── WORKSPACE_METADATA / DOCUMENT_LIST
  │   └── DB query → structured response
  ├── DOCUMENT_CONTENT
  │   └── query rewrite (if follow-up) → relevance gate → retrieval → generation
  ├── CLARIFICATION / OUT_OF_SCOPE
  │   └── refusal or clarification prompt
  └── GENERAL_CONVERSATION
      └── lightweight LLM chat response
     │
     ▼  (DOCUMENT_CONTENT path only)
Query Rewrite (pronoun/reference resolution from conversation context)
     │
     ▼
Relevance Gate
  ├── Layer 1 (deterministic): greetings, obviously unrelated, contact/personnel heuristics
  └── Layer 2 (LLM classifier): workspace document titles as context
     │
     ├── not relevant → refuse (no retrieval, no LLM call)
     └── relevant / ambiguous → continue
     │
     ▼
Hybrid Retrieval
  ├── pgvector cosine similarity (semantic, bi-encoder)
  └── PostgreSQL full-text search on content_tsv (keyword)
     │
     ▼
Filename-aware retrieval (Phase B-2: document name matching)
     │
     ▼
RRF Fusion (k=60) → top ~15 candidates
     │
     ▼
Local cross-encoder reranking (ms-marco-MiniLM-L-6-v2) → top 5-8 chunks
     │
     ▼
Grounding Check (query-shape-aware)
  ├── OVERVIEW: absolute-threshold aggregate grounding (top-k mean)
  ├── FACT_LOOKUP: single-chunk top-score threshold
  ├── HIGH_CONFIDENCE_DOC_TARGET: relaxed threshold
  └── FILENAME_MATCH: very permissive threshold
     │
     ├── not grounded → refuse (no LLM call)
     └── grounded → continue
     │
     ▼
LLM Generation (sequential fallback: Groq → OpenRouter → Gemini)
  └── strict "answer only from context" system prompt
     │
     ▼
Answer + backend-constructed citations (built from actual source chunks)
```

### Request Flow (Document Upload → Ingestion)

```
Authenticated user (JWT) → workspace membership + role (members table)
     │
     ▼
File validation (type allowlist + content sniff, size cap, SHA-256)
     │
     ├── OWNER upload:
     │   ├── extract (PyMuPDF / python-docx / CSV) → normalized pages
     │   ├── chunk (RecursiveCharacterTextSplitter)
     │   ├── embed (bge-small-en-v1.5, 384-dim)
     │   └── store document READY + chunks, ONE transaction
     │
     └── MEMBER upload:
         └── store document PENDING only (no extraction, no chunking, no embedding)
              → structurally unsearchable until owner approval
```

### Request Flow (Document Approval)

```
MEMBER upload → PENDING (no chunks)
     │
     ▼
OWNER reviews (GET /documents?status=PENDING)
     ├── approve → extract → chunk → embed (inline) → READY + chunks, atomic
     └── reject  → REJECTED (never ingested, permanent)
```

---

## 2. Authentication & Tenancy

### JWT Verification

File: `backend/app/security/auth.py`

- Verifies Supabase-issued JWTs using JWKS (ES256/RS256), with HS256 fallback.
- Configurable `JWT_LEEWAY_SECONDS` (default 30) to tolerate clock skew between local machine and Supabase.
- Extracts `sub` claim as `user_id`, workspace claim as `workspace_id` from the token.
- Returns a `Principal` dataclass: `{ user_id, workspace_id, email }`.

### Authorization (Members Table)

Every workspace-scoped endpoint performs:

1. Resolve authenticated user's membership in the requested workspace from the `members` table (role + status).
2. A user with no membership row gets **403**.
3. Role enforcement: OWNER-only endpoints call `assert_workspace_role(ws, principal, "OWNER")`.
4. `workspace_id` is always filtered server-side, never trusted from the frontend.

### Row-Level Security (RLS)

File: `backend/alembic/versions/0008_*.py`

- RLS policies on `document_chunks`, `documents`, `chat_sessions`, `chat_messages` enforce workspace isolation at the database level.
- `app_tenant` role (used by the application) is NOBYPASSRLS.
- `postgres` role has BYPASSRLS (used by test infrastructure).
- Application-level `workspace_id` filtering is always present on top of RLS as defense-in-depth.

### Session Claims

File: `backend/app/security/rls.py`

```python
_SET_CLAIMS = "SET LOCAL app.current_user_id = :uid; SET LOCAL app.current_org_id = :org;"
```

Every request sets `app.current_user_id` and `app.current_org_id` as PostgreSQL session variables. RLS policies reference these via `app.current_user_id()` and `app.current_claims()` SQL functions (migration 0001).

---

## 3. Chat Lifecycle

### Entry Point

File: `backend/app/api/chat_v2.py` (2184 lines)

Endpoints:
- `POST /chat` — SSE streaming response
- `POST /chat/grounded` — non-streaming JSON response (test/acceptance only)
- `GET /chat/sessions` — list sessions
- `GET /chat/sessions/{id}` — get session messages
- `DELETE /chat/sessions/{id}` — delete session

### Session Persistence

- Session is resolved **once** at the top of the handler, before intent dispatch.
- All intent branches write to the pre-resolved session.
- The `session` SSE event is emitted only when a new session is created.
- Verified: 4-turn mixed conversation → 1 ChatSession row, 8 messages.

### SSE Events

| Event | Payload | When |
|---|---|---|
| `session` | `{ id, created_at }` | Only on new session creation |
| `sources` | `[{ number, document_id, filename, page, label, excerpt, score }]` | Before generation starts |
| `token` | `{ delta }` | Each streaming token |
| `done` | `{ text, citations }` | Stream complete |

### Think-Tag Filtering

Some models (Qwen3, Gemini) emit `<think>...</think>` reasoning blocks. These are stripped from the streamed output before delivery to the client.

### Conversation Context

- Last 1-2 exchanges are included when forming the retrieval query for follow-up questions.
- Pronouns resolve via `query_rewrite.py` (319 lines): extracts conversational context, rewrites the query.
- Only a small window is sent — full history is never included.

---

## 4. Intent Classification

File: `backend/app/retrieval/intent.py` (1314 lines)

### 15-Category Taxonomy

| Category | Behavior | Retrieval? | LLM Call? |
|---|---|---|---|
| `GREETING` | Hardcoded conversational response | No | No |
| `APP_HELP` | Application usage guidance | No | No |
| `IDENTITY` | Who-am-I from session data | No | No |
| `IDENTITY_ASSISTANT` | Assistant identity | No | No |
| `IDENTITY_USER` | User identity | No | No |
| `PERMISSIONS` | Role/permission info | No | No |
| `WORKSPACE_METADATA` | Doc count, member count, etc. | No (DB query) | No |
| `WORKSPACE_PERMISSION` | Who-can-do-what | No | No |
| `DOCUMENT_LIST` | List/show documents | No (DB query) | No |
| `DOCUMENT_CONTENT` | Answer from document content | **Yes** | **Yes** |
| `DOCUMENT_COMPARISON` | Compare documents | **Yes** | **Yes** |
| `CONVERSATION_HISTORY` | What did we discuss | No (DB query) | No |
| `OUT_OF_SCOPE` | General knowledge | No | No |
| `GENERAL_CONVERSATION` | General chat | No | Yes (lightweight) |
| `AMBIGUOUS` | Needs clarification | No | Clarification prompt |

### Classification Flow

1. **Regex fast-path** (`classify_intent_regex()`): handles ~80% of queries deterministically. Checks greetings, identity, permissions, document list, metadata, conversation history, out-of-scope, etc.
2. **LLM 5-route router** (`llm_router.py`, 330 lines): for genuinely ambiguous queries. Routes to one of: `direct`, `metadata`, `retrieval`, `clarification`, `out_of_scope`.
3. Route name → `IntentCategory` mapping in the dispatch logic.

### Query Shape Classification

For `DOCUMENT_CONTENT` intents, a `QueryShape` is computed:

| Shape | Description | Grounding Strategy |
|---|---|---|
| `OVERVIEW` | "tell me about X" | Aggregate threshold (top-k mean) |
| `FACT_LOOKUP` | "what is X's value for Y" | Single-chunk top-score threshold |
| `HIGH_CONFIDENCE_DOC_TARGET` | "in the handbook, what does it say about X" | Relaxed threshold |
| `FILENAME_MATCH` | "what's in expenses.pdf" | Very permissive threshold |

---

## 5. Retrieval Pipeline

### Hybrid Search

File: `backend/app/rag/retrieval.py` (156 lines)

Two parallel searches run, then merge:

1. **Semantic search**: pgvector cosine similarity on `embedding vector(384)`. Uses `embed_query()` with the bge instruction prefix (`"Represent this sentence for searching relevant passages: "`).
2. **Keyword search**: PostgreSQL full-text search on `content_tsv` (GENERATED ALWAYS AS `to_tsvector('english', content)`). Uses `ts_rank` for scoring.

### RRF Fusion

```python
score = 1/(k + rank_semantic) + 1/(k + rank_keyword)  # k=60
```

Standard Reciprocal Rank Fusion. Chosen because cosine similarity and `ts_rank` are on incomparable scales — rank-based fusion sidesteps normalization. Top ~15 candidates by combined score.

### Filename-Aware Retrieval (Phase B-2)

Additional retrieval pass for queries that mention document filenames. Supplements the hybrid search results with chunks from the specifically-named document.

### Local Cross-Encoder Reranking

File: `backend/app/retrieval/rerank.py` (80 lines)

- Model: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- Lazy-loaded (first invocation loads the model; subsequent calls reuse it)
- Thread-safe via `threading.Lock`
- Scores `(query, chunk_text)` pairs directly — no separate embedding step
- Top 5-8 chunks by rerank score pass to generation

### Relevance Gate

File: `backend/app/retrieval/relevance.py` (349 lines)

Two layers:

1. **Deterministic gate** (Layer 1): catches greetings, obviously unrelated queries, contact/personnel heuristics. No retrieval, no LLM call.
2. **LLM classifier** (Layer 2): sends query + workspace document titles to a small LLM call. Classifies as relevant, ambiguous, or not-relevant.

### Grounding Check

File: `backend/app/retrieval/grounding.py`

Query-shape-aware thresholds applied to reranked chunks:

- **OVERVIEW**: top-k mean score must exceed threshold
- **FACT_LOOKUP**: single top-score must exceed threshold
- **HIGH_CONFIDENCE_DOC_TARGET**: relaxed threshold
- **FILENAME_MATCH**: very permissive

If grounding fails → refuse honestly (no LLM call).

---

## 6. LLM Fallback Chain

File: `backend/app/llm/fallback.py` (291 lines)

### Sequential Failover

```
Primary (Groq) → Fallback (OpenRouter) → Secondary Fallback (Gemini)
```

- Providers tried **strictly sequentially**, never in parallel.
- Only providers whose API key is present are included in the chain.
- Configuration via env vars: `GROQ_API_KEY`, `OPENROUTER_API_KEY`, `GEMINI_API_KEY`, or generic `LLM_PROVIDER/LLM_MODEL/LLM_API_KEY/LLM_BASE_URL`.

### Failure Triggers

Fallback on: HTTP 429, 500, 502, 503, 504, timeout, connection error.
Does **not** trigger on: invalid requests from our own code (those surface clearly).

### Streaming Semantics

1. First provider tried with `stream=True`.
2. If it fails **before** any token is yielded → next provider attempted.
3. If it fails **after** tokens have been yielded → error surfaced (cannot un-send tokens).

### Timeout Budget

- Per-provider timeout: `LLM_TIMEOUT_SECONDS` setting.
- Overall timeout: tracked across all attempts. Prevents three sequential timeouts from summing to minutes.

### Provider Protocol

`FallbackChainProvider` implements `LLMRouterProtocol` — callers cannot tell whether they hold one model or a failover chain.

---

## 7. Data Model

File: `backend/app/db/models.py` (281 lines)

### Tables

```
workspaces ──┬── members (workspace_id, user_id, role OWNER/MEMBER, status)
             ├── documents (workspace_id, status PENDING/READY/REJECTED/FAILED, BYTEA file_data)
             ├── document_chunks (workspace_id, document_id, content_tsv, embedding vector(384))
             ├── chat_sessions (workspace_id, user_id)
             └── chat_messages (session_id, role, content, sources jsonb)

invitations (workspace_id, email, status PENDING/ACCEPTED/EXPIRED)
```

### Key Models

**Workspace**: `id` (UUID), `name`, `owner_id`, `created_at`

**Member**: `id`, `workspace_id`, `user_id`, `role` (OWNER/MEMBER), `status` (INVITED/ACTIVE/REMOVED), `created_at`. Unique on `(workspace_id, user_id)`.

**Document**: `id`, `workspace_id`, `uploaded_by`, `filename`, `mime_type`, `file_size`, `checksum` (SHA-256), `file_data` (BYTEA), `status` (PENDING/READY/REJECTED/FAILED), `error_message`, `created_at`, `approved_at`. Unique on `(workspace_id, checksum)`.

**DocumentChunk**: `id`, `document_id`, `workspace_id` (denormalized), `chunk_index`, `content`, `content_tsv` (tsvector, generated), `embedding` (vector(384)), `page_number`, `section_title`, `metadata` (JSONB). Unique on `(document_id, chunk_index)`.

**ChatSession**: `id`, `workspace_id`, `user_id`, `created_at`

**ChatMessage**: `id`, `session_id`, `role` (user/assistant), `content`, `sources` (JSONB), `created_at`

**Invitation**: `id`, `workspace_id`, `email`, `status` (PENDING/ACCEPTED/EXPIRED), `invited_by`, `created_at`. Unique on `(workspace_id, email)`.

### Alembic Migrations

16 migrations total (`backend/alembic/versions/`). Key ones:

| Migration | Purpose |
|---|---|
| 0001 | Initial schema: workspaces, members, documents, document_chunks, chat_sessions, chat_messages. `app.current_user_id()`, `app.current_claims()` SQL functions. |
| 0007 | Invitations table. |
| 0008 | RLS policies: `document_chunks_write`, `document_chunks_read`, workspace-scoped read/write for all tables. |
| 0015 | `create_workspace` SQL function with one-org-per-email enforcement. |
| 0016 | Document deletion support. |

---

## 8. Document Ingestion

### Pipeline

File: `backend/app/rag/pipeline.py` (198 lines) + `backend/app/ingestion/pipeline.py`

Steps: extract → normalize → chunk → embed → store

### Extraction

| Format | Library | Notes |
|---|---|---|
| PDF | PyMuPDF | Page numbers preserved |
| DOCX | python-docx | Paragraphs + tables |
| CSV | pandas-style parsing | Header repeated per page block |
| XLSX | openpyxl | Multi-sheet support |

### Chunking

- `RecursiveCharacterTextSplitter` from LangChain
- Configurable chunk size and overlap via `config.py`
- Each chunk carries: `chunk_index`, `content`, `page_number`, `section_title`, `metadata`

### Embedding

- Model: `BAAI/bge-small-en-v1.5` (384 dimensions), pinned
- Query-side instruction prefix: `"Represent this sentence for searching relevant passages: "`
- Stored as `vector(384)` in PostgreSQL via pgvector
- Local, free, no GPU required

### Deduplication

SHA-256 checksum + `UNIQUE (workspace_id, checksum)` — a second identical upload within the same workspace is rejected with 409. Cross-workspace deduplication is intentionally not performed.

---

## 9. Configuration

File: `backend/app/config.py` (504 lines)

All settings from `.env` via Pydantic `BaseSettings`. Key variables:

| Variable | Default | Purpose |
|---|---|---|
| `DATABASE_URL` | — | PostgreSQL connection string |
| `SUPABASE_URL` | — | Supabase project URL |
| `SUPABASE_ANON_KEY` | — | Supabase anon key |
| `SUPABASE_SERVICE_ROLE_KEY` | — | Supabase service role key |
| `GROQ_API_KEY` | — | Primary LLM provider |
| `OPENROUTER_API_KEY` | — | Fallback LLM provider |
| `GEMINI_API_KEY` | — | Secondary fallback LLM provider |
| `EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | Local embedding model |
| `EMBEDDING_DIMENSION` | `384` | Must match model output |
| `RERANKER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Local reranker |
| `RETRIEVAL_CANDIDATE_COUNT` | `15` | Pre-rerank candidates |
| `RETRIEVAL_FINAL_COUNT` | `8` | Post-rerank chunks |
| `RETRIEVAL_RELEVANCE_THRESHOLD` | `0.3` | Grounding gate threshold |
| `MAX_UPLOAD_SIZE_MB` | `10` | Upload size cap |
| `JWT_LEEWAY_SECONDS` | `30` | Clock-skew tolerance |
| `LLM_TIMEOUT_SECONDS` | — | Per-provider + overall timeout |

---

## 10. Key Files

| File | Lines | Role |
|---|---|---|
| `backend/app/api/chat_v2.py` | 2184 | Chat endpoint, session management, streaming, intent dispatch |
| `backend/app/retrieval/intent.py` | 1314 | Intent taxonomy (15 categories), regex classification, query shape |
| `backend/app/retrieval/pipeline.py` | 415 | Retrieval orchestration: relevance gate → hybrid search → RRF → rerank → grounding |
| `backend/app/retrieval/llm_router.py` | 330 | LLM 5-route intent router |
| `backend/app/retrieval/relevance.py` | 349 | Two-layer relevance gate |
| `backend/app/retrieval/grounding.py` | 150 | Query-shape-aware grounding thresholds |
| `backend/app/retrieval/query_rewrite.py` | 319 | Conversational query rewriting for follow-ups |
| `backend/app/retrieval/rerank.py` | 80 | Local cross-encoder reranking |
| `backend/app/retrieval/hybrid.py` | — | Semantic + keyword FTS, RRF merge, filename search |
| `backend/app/retrieval/doc_targeting.py` | — | Document-level targeting (Phase B-2) |
| `backend/app/rag/pipeline.py` | 198 | RAG answer pipeline: retrieve → prompt → stream → cite |
| `backend/app/rag/retrieval.py` | 156 | Vector retrieval over ingested chunks |
| `backend/app/rag/prompts.py` | — | System prompt construction |
| `backend/app/rag/embeddings.py` | — | Embedding model setup (lazy-loaded) |
| `backend/app/config.py` | 504 | All settings from .env, LLM provider auto-derivation |
| `backend/app/main.py` | 178 | App factory, lifespan (migration auto-apply, demo seed, cleanup loop) |
| `backend/app/llm/fallback.py` | 291 | Sequential LLM fallback chain with 429 retry and timeout budget |
| `backend/app/security/auth.py` | 355 | JWT verification, Principal resolution, clock-skew leeway |
| `backend/app/security/rls.py` | — | Session claims, tenant session context |
| `backend/app/demo/seed.py` | — | Demo workspace seeding (documents, chunks, members) |
| `backend/app/demo/cleanup.py` | 253 | Guest cleanup: footprint cascade, workspace reassignment, auth user deletion |
| `backend/app/api/demo.py` | 225 | Demo entry endpoint (ephemeral guest user creation) |
| `backend/app/db/models.py` | 281 | All SQLAlchemy models |
| `backend/app/ingestion/pipeline.py` | — | Document upload → chunking → embedding pipeline |
| `frontend/src/app/page.tsx` | 315 | Main workspace shell with sidebar, document panel, chat |
| `frontend/src/app/login/page.tsx` | 397 | Auth UI (sign-in, sign-up, demo entry) |
| `frontend/src/lib/hooks/use-chat.ts` | 389 | Chat streaming hook: SSE parsing, turn management, session persistence |

---

## 11. Known-Incomplete Items

| Item | Status | Notes |
|---|---|---|
| **NTP sync** | BLOCKED | Clock ~0.9ms behind Supabase. JWT leeway (30s) masks it. Requires admin to enable Windows Time service. |
| **Invitation flow** | Incomplete | Backend accept endpoint exists; no frontend landing page, no email delivery. |
| **POST /chat/grounded** | Test-only | Non-persistent endpoint for acceptance testing. Should only run under developer's own token. |
| **Legacy `organizations` table** | Stale | Some integration tests reference this dropped table. Tests skip or fail. Schema rewrite deferred. |
| **5 integration test failures** | Pre-existing | Legacy `organizations` table issue. `_seed_document`/`_seed` helpers need rewrite to use `workspace_id` schema. |
| **Phase 10 full test pass** | Not run | Complete pytest + vitest/playwright pass has not been executed. |
| **Uncommitted work** | Pending | 14 modified + 2 untracked files from routing refactor and bug fixes. |

---

## 12. Test Suites

| Suite | Location | Scope | Status |
|---|---|---|---|
| Unit tests | `backend/tests/unit/` (38 files) | Intent routing, chat persistence, session management, demo cleanup, auth | **674/674 pass** |
| Security tests | `backend/tests/security/` (6 files) | JWT verification, RLS policies, member 403 authorization | **60/60 pass** (not run this session) |
| Integration tests | `backend/tests/integration/` (10 files) | Seed, cache staleness, document deletion, ingestion | **33 pass, 5 fail (legacy table), 33 skip** |

Run command: `python -m pytest backend/tests/unit -v` (unit) or `python -m pytest backend/tests/integration -v` (integration).
Interpreter: `backend\.venv-py311\Scripts\python.exe` (Python 3.11.15).
