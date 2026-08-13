# CLAUDE.md — Multi-Tenant Company Knowledge Assistant (Free-Tier RAG Portfolio Project)

This file is the single source of truth. Read it fully before writing any code. Work
**one phase at a time, in order**. After finishing a phase: summarize what was built,
list modified files, list checks run, list known issues, and **stop and wait for
explicit go-ahead**. Never start the next phase automatically.

This project is built and run entirely on **free-tier / locally runnable
infrastructure** (Claude Code itself is being run here through OpenRouter's free model
routing — see Section 12, which governs how work gets done, not just what gets built).
Every architectural decision below was made specifically to keep the project buildable
by a fresher, on a normal laptop, at zero cost, while still being technically defensible
in a backend/RAG interview. If a change would require paid infrastructure, it doesn't
belong in this project — flag it instead of adding it.

If any instruction here conflicts with speed or "how a big company would do it," this
file wins. Boring, correct, and explainable beats impressive and unverified.

---

## 0. Ground Rules

1. **One phase per work session.** Do not start Phase N+1 until Phase N's deliverable is
   in place and confirmed by the user.
2. **No hardcoded secrets, ever.** All keys/URLs come from `.env` via Pydantic
   `BaseSettings`. `.env` is gitignored from commit #1; ship `.env.example` with empty
   placeholders.
3. **Every external input is hostile until proven otherwise**: file uploads, chat
   messages, retrieved document text.
4. **The LLM never gets write access to anything.** It answers questions from retrieved
   context. It does not run queries against arbitrary tables, does not call external
   tools, does not take actions on the user's behalf.
5. **Testing happens once, at the end** (Section 9's final phase). Static checks
   (lint/typecheck) run every phase; the automated test suite and evaluation set are
   written once the system works end-to-end.
6. **Prefer boring, well-understood, free tools.** One LLM (configured, not hardcoded),
   one local embedding model, one local reranker, one database. No fallback chains, no
   framework-of-the-week.
7. **Small, reviewable commits.** The user commits manually — never run `git commit`
   unless explicitly asked.
8. **Before touching anything**, inspect the actual repository state — existing files,
   models, migrations, routes, env vars, dependencies. Do not assume the repo is empty
   and do not assume a described feature exists until you've verified it in code.
9. **Minimal diffs.** Make the smallest correct change that satisfies the current
   phase. Do not rewrite working code to make it "cleaner" unless the phase asked for
   that specifically.

---

## 1. Project Overview

A **multi-tenant** company knowledge assistant. A single deployment hosts many
independent companies (workspaces), each fully isolated from the others. Within a
workspace: an **owner** creates it, invites **members**, and uploads official documents
that publish immediately. Members can upload their own documents, which stay
**pending** until the owner approves them — only approved documents ever become
searchable.

Employees ask natural-language questions in a chat interface. The assistant answers
**only** from that workspace's approved documents, using hybrid (semantic + keyword)
retrieval with local reranking, and always returns backend-verified citations. If the
retrieved evidence is insufficient, it refuses honestly instead of guessing.

That is the entire product. Do not turn it into a general agent, a SQL chatbot, an
automation platform, or an MCP host — see Section 11 for the exact list of things this
project deliberately does not do.

---

## 2. Free-First Tech Stack

```
Frontend
  Next.js (App Router) + TypeScript
  Tailwind CSS + shadcn/ui

Backend
  Python 3.11, FastAPI, Pydantic v2, SQLAlchemy 2.0 (async), Alembic

Database (the ONLY infrastructure component besides the LLM)
  PostgreSQL via Supabase (free tier) + pgvector + PostgreSQL full-text search + JSONB
  One database for everything: workspaces, members, documents (raw bytes included),
  chunks, chat history. No second datastore, no cache, no queue.

Auth
  Supabase Auth for identity ("who is this user"). Application tables own authorization
  ("what can this user access") — workspace membership, OWNER/MEMBER role, invitation
  state. No custom JWT rotation, no custom refresh logic — Supabase issues the token,
  the backend verifies it and looks up membership from the database.

LLM
  ONE model, configured entirely through environment variables (LLM_PROVIDER,
  LLM_MODEL, LLM_API_KEY, LLM_BASE_URL). No hardcoded provider, no fallback chain.
  Must run against free-tier or free-routed models (e.g. OpenRouter's free model
  routing). The code must not assume any specific provider's quirks — treat it as a
  generic chat-completions endpoint.

Embeddings
  Exactly ONE local, free embedding model via sentence-transformers
  (e.g. BAAI/bge-small-en-v1.5 — 384 dimensions; pick one and document the dimension
  in EMBEDDING_DIMENSION). Never call a paid embedding API. Never mix embeddings from
  two different models in the same index — if the model ever changes, re-embed
  everything, don't patch in place.

Reranking
  ONE local, free cross-encoder (e.g. cross-encoder/ms-marco-MiniLM-L-6-v2), run via
  sentence-transformers. No LLM-as-judge reranking unless a specific, documented
  technical reason emerges — an LLM call per candidate is slow and burns free-tier
  quota for no real benefit at this scale.

Deployment
  Docker Compose locally (app + Postgres only). Vercel (frontend) + Railway/Render free
  tier (backend) + Supabase (database) for a hosted demo.
```

**Explicitly excluded, and why it stays excluded:** Redis, Celery, Kafka, RabbitMQ,
MCP, Kubernetes, AWS S3, a separate vector database, paid observability platforms,
multi-agent frameworks (LangGraph et al.), paid LLM/embedding/reranking APIs. None of
these are needed for the product described in Section 1, and every one of them would
either cost money, add a service that can fail independently, or add complexity this
project has no real requirement for. See Section 11 for the full "don't add this"
table with reasoning per item.

---

## 3. Repository Structure

```
knowledge-assistant/
├── CLAUDE.md
├── .env.example
├── .gitignore
├── docker-compose.yml           # app + postgres only
├── backend/
│   ├── pyproject.toml
│   ├── app/
│   │   ├── main.py
│   │   ├── config.py             # Pydantic BaseSettings — every env var in Section 13
│   │   ├── api/                  # routers: auth, workspaces, members, documents, chat
│   │   ├── ingestion/             # extract -> chunk -> embed (inline, synchronous)
│   │   ├── retrieval/              # hybrid search, merge/fusion, reranking, grounding
│   │   ├── db/                      # SQLAlchemy models, Alembic migrations
│   │   └── security/                 # session/auth verification, workspace authz
│   ├── eval/                          # evaluation dataset + runner (Section 15)
│   └── tests/                          # written once, at the final phase
└── frontend/
    ├── package.json
    └── src/
        ├── app/
        ├── components/
        └── lib/
```

---

## 4. Multi-Tenancy (Hard Requirement)

A single deployment hosts **many independent workspaces** (companies). Every
workspace-owned table — `documents`, `document_chunks`, `chat_sessions`,
`chat_messages` — carries `workspace_id`, and **every query that touches these tables
filters on it, server-side, without exception.** The frontend must never be trusted to
enforce this boundary; it's a convenience layer, not a security layer.

Concretely: before any workspace-scoped endpoint does anything, it must resolve the
authenticated user's membership in the requested workspace (role + status) from the
`members` table. A user with no membership row for that workspace gets a 403, full
stop — there is no implicit access.

Two roles only: **OWNER** and **MEMBER**. Do not add ADMIN, SUPER_ADMIN, MODERATOR,
EDITOR, or VIEWER — the product described in Section 1 has no task that needs a third
role, and adding one "for flexibility" is exactly the kind of unnecessary complexity
this project avoids.

| | Owner | Member |
|---|---|---|
| Create workspace | ✅ (becomes owner) | — |
| Invite members | ✅ | ❌ |
| Upload → published immediately | ✅ | ❌ |
| Upload → pending approval | — | ✅ |
| Approve/reject pending documents | ✅ | ❌ |
| Delete any workspace document | ✅ | own uploads only |
| Search approved knowledge / chat | ✅ | ✅ |

---

## 5. Document Lifecycle (Enforced Structurally, Not by Prompting)

```
OWNER upload  → validate → extract → chunk → embed → store  → READY  (immediate)

MEMBER upload → validate → store document only               → PENDING
                (no extraction, no chunking, no embedding yet)

OWNER approves a PENDING doc → extract → chunk → embed → store → READY
OWNER rejects a PENDING doc  → REJECTED (never ingested, permanent)

Any ingestion failure → FAILED, with the error persisted on the row
```

**The rule that keeps this simple:** only `READY` documents are ever chunked or
embedded. Retrieval doesn't need to filter by status at query time, because a
`PENDING` or `REJECTED` document structurally has zero rows in `document_chunks` —
there is nothing to accidentally leak. This must hold as an invariant, not an
assumption: a document's status and the existence of its chunks are never allowed to
disagree (see Section 7's transaction notes).

---

## 6. Document Storage

`documents.file_data` is `BYTEA` — the raw file lives in Postgres, not on disk and not
in S3. This is intentional: it demonstrates a genuinely single-database architecture,
which is the whole point of the "database is the source of truth" requirement.

**Honest limitation:** `BYTEA` is not built for large binary blobs at scale — every
read pulls the full bytes through the connection, and Postgres row/page overhead
matters more than with dedicated blob storage. For a portfolio project with a
sensible per-file size cap this is a non-issue. **Enforce `MAX_UPLOAD_SIZE_MB` (default
10MB) at the API layer, checked from `Content-Length` before the file is read into
memory** — reject oversized uploads before any bytes are pulled off the wire. If this
project ever needed to hold hundreds of large files per workspace in production,
Supabase Storage (object storage, not a new service) would be the natural next step —
noted here as a documented trade-off, not something to build now.

`documents` also stores: filename, MIME type, file size, a SHA-256 checksum (used to
detect duplicate uploads **within a workspace** — do not attempt cross-workspace
deduplication, tenant boundaries matter more than saving a few embeddings), uploader,
workspace, status, timestamps, and ingestion error text where relevant.

---

## 7. Database Schema

```sql
workspaces
  id            uuid PK
  name          text NOT NULL
  owner_id      uuid NOT NULL REFERENCES auth.users(id)
  created_at    timestamptz NOT NULL DEFAULT now()

members
  id            uuid PK
  workspace_id  uuid NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE
  user_id       uuid NOT NULL REFERENCES auth.users(id)
  role          text NOT NULL CHECK (role IN ('OWNER','MEMBER'))
  status        text NOT NULL CHECK (status IN ('INVITED','ACTIVE','REMOVED'))
  created_at    timestamptz NOT NULL DEFAULT now()
  UNIQUE (workspace_id, user_id)
  INDEX (workspace_id, user_id)          -- membership lookup on every request

documents
  id            uuid PK
  workspace_id  uuid NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE
  uploaded_by   uuid NOT NULL REFERENCES auth.users(id)
  filename      text NOT NULL
  mime_type     text NOT NULL
  file_size     integer NOT NULL
  checksum      text NOT NULL            -- sha256, for in-workspace dedupe
  file_data     bytea NOT NULL
  status        text NOT NULL CHECK (status IN ('PENDING','READY','REJECTED','FAILED'))
  error_message text
  created_at    timestamptz NOT NULL DEFAULT now()
  approved_at   timestamptz
  INDEX (workspace_id, status)            -- "give me all READY docs" / approval queue
  INDEX (workspace_id, uploaded_by)       -- "my documents" view
  UNIQUE (workspace_id, checksum)         -- duplicate detection

document_chunks
  id              uuid PK
  document_id     uuid NOT NULL REFERENCES documents(id) ON DELETE CASCADE
  workspace_id    uuid NOT NULL            -- denormalized on purpose: every retrieval
                                            -- query filters here directly, no join
                                            -- required to enforce tenant isolation
  chunk_index     integer NOT NULL
  content         text NOT NULL
  content_tsv     tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED
  embedding       vector(384) NOT NULL     -- MUST match EMBEDDING_DIMENSION exactly
  page_number     integer
  section_title   text
  metadata        jsonb
  created_at      timestamptz NOT NULL DEFAULT now()
  INDEX (workspace_id)
  INDEX (document_id)
  INDEX USING GIN (content_tsv)            -- full-text search
  INDEX USING ivfflat (embedding vector_cosine_ops)  -- pgvector ANN index

chat_sessions
  id            uuid PK
  workspace_id  uuid NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE
  user_id       uuid NOT NULL REFERENCES auth.users(id)
  created_at    timestamptz NOT NULL DEFAULT now()
  INDEX (workspace_id, user_id)

chat_messages
  id            uuid PK
  session_id    uuid NOT NULL REFERENCES chat_sessions(id) ON DELETE CASCADE
  role          text NOT NULL CHECK (role IN ('user','assistant'))
  content       text NOT NULL
  sources       jsonb                      -- structured citations, see Section 8.4
  created_at    timestamptz NOT NULL DEFAULT now()
  INDEX (session_id)

invitations                                 -- the one supporting table this needs
  id            uuid PK
  workspace_id  uuid NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE
  email         text NOT NULL
  status        text NOT NULL CHECK (status IN ('PENDING','ACCEPTED','EXPIRED'))
  invited_by    uuid NOT NULL REFERENCES auth.users(id)
  created_at    timestamptz NOT NULL DEFAULT now()
  UNIQUE (workspace_id, email)
```

That's six core tables plus `invitations`. Do not add tables that "look enterprise" —
every table above maps to a specific requirement in Section 1.

**Transaction boundary that matters most:** approval (Section 5) must be atomic —
`documents.status = READY` and its `document_chunks` rows are written in the same
transaction, or ingestion fails and the row goes to `FAILED` with nothing partially
searchable. Never allow `status = READY` with zero chunks, or `status = PENDING/FAILED`
with chunks present. On deletion, rely on the `ON DELETE CASCADE` from
`document_chunks.document_id` — there must be no orphaned chunk left searchable after
its parent document is gone.

---

## 8. RAG Pipeline

```
User question
     │
     ▼
Authentication (Supabase Auth) + workspace authorization (membership check)
     │
     ▼
Conversation context resolution (last 1–2 exchanges only, not full history)
     │
     ▼
Hybrid retrieval
  ├── pgvector cosine similarity search (semantic)
  └── PostgreSQL full-text search on content_tsv (keyword)
     │
     ▼
Merge + deduplicate → top ~10–15 candidates
     │
     ▼
Local cross-encoder reranking (query, candidate) → relevance scores
     │
     ▼
Top ~5–8 chunks
     │
     ▼
Relevance/grounding threshold check
     │
     ├── below threshold → return honest refusal, DO NOT call the LLM
     └── above threshold → continue
     │
     ▼
LLM (strict "answer only from context" system prompt)
     │
     ▼
Answer + backend-constructed citations
```

Deterministic, single pipeline. No LangGraph, no supervisor, no routing between
specialized agents — there is exactly one kind of question this system answers.

### 8.1 Hybrid retrieval and merge strategy
Semantic-only search misses exact terms (`HR-004`, `POL-17`, `20 days`) because
embeddings represent meaning, not tokens. Keyword-only search misses paraphrases
(`vacation` vs. `annual leave`). Run both, then merge with **Reciprocal Rank Fusion
(RRF)**: for each candidate, `score = 1/(k + rank_semantic) + 1/(k + rank_keyword)`
with `k = 60` (a standard, well-documented default), and take the top ~15 by combined
score before reranking. RRF is chosen over normalized-score blending because cosine
similarity and `ts_rank` are on incomparable scales, and rank-based fusion sidesteps
needing to normalize two dissimilar scoring functions correctly.

### 8.2 Reranking
The local cross-encoder scores `(query, chunk_text)` pairs directly — no separate
embedding step needed for this stage. Cap candidates into the reranker at ~15; never
rerank hundreds of chunks, both because it's unnecessary and because it's the slowest
step in the pipeline on CPU. Final context passed to the LLM: top 5–8 by rerank score.

### 8.3 Two-layer grounding
Layer 1 — **retrieval-level**: if the top reranked chunk's score is below a configured
threshold (`RETRIEVAL_RELEVANCE_THRESHOLD`), skip the LLM call entirely and return
"I couldn't find that information in the approved company knowledge base." This is
not just an optimization — it's what prevents the LLM from ever seeing a question with
no real supporting evidence and being tempted to fill the gap.
Layer 2 — **generation-level**: the system prompt requires the LLM to answer only from
the supplied context and to say so explicitly if the context doesn't cover the
question, as a backstop for cases that pass the threshold but are still a partial
match. Neither layer alone is sufficient; both apply on every request.

An out-of-scope question ("who won the World Cup", "write me a Python game") is not a
special case needing its own agent — it's simply a question with no relevant
retrieval results, and layer 1 handles it the same way it handles any ungrounded
question.

### 8.4 Citations
The LLM is never trusted to invent citation metadata. The backend builds the citation
list directly from the chunks that were actually sent to the LLM — `document_id`,
`filename`, `page_number` (if available), `chunk_id` — and returns it as structured
data alongside the generated text. The frontend renders it (`Employee Handbook — Page
14`); the LLM's job is text generation only.

### 8.5 Conversation context
Follow-ups ("can I carry it over?") need the last 1–2 exchanges included when forming
the retrieval query, so pronouns resolve. Send only that small window into the
retrieval + generation step — never the full session history, both to keep prompts
small (Section 10 free-tier constraint) and because older turns rarely help resolve a
current pronoun.

---

## 9. Phased Build Plan

### Phase 0 — Scaffolding
FastAPI `/health`, Next.js blank app, docker-compose (app + postgres only),
`.env.example`, `.gitignore`.
Verify: `docker compose up` starts postgres; `GET /health` returns 200.

### Phase 1 — Database & Config
`config.py` (Pydantic BaseSettings, Section 13), async SQLAlchemy engine, Alembic, all
tables from Section 7, pgvector extension enabled, `EMBEDDING_DIMENSION` wired through
consistently.
Verify: migration applies cleanly; app fails loudly on missing required env vars.

### Phase 2 — Auth & Multi-Tenant Workspaces
Supabase Auth integration, workspace creation (creator becomes OWNER), invitation
flow (`invitations` table), member accept → `ACTIVE` membership row.
Verify: create two separate workspaces as two different owners; confirm a member of
Workspace A gets 403 on any Workspace B endpoint.

### Phase 3 — Document Upload & Synchronous Ingestion
Upload endpoint (size check before read, type allowlist, checksum computed),
extraction (PyMuPDF/python-docx/pandas), chunking, local embedding model, inline
storage. Owner upload → `READY`. Member upload → `PENDING`, not ingested.
Verify: owner upload produces chunks with correct `workspace_id`, correct embedding
dimension; member upload leaves `document_chunks` empty for that document.

### Phase 4 — Approval Flow
Owner's pending queue, approve (→ ingest → `READY`, atomic per Section 7) / reject
(→ `REJECTED`, never ingested) endpoints, restricted to OWNER role server-side.
Verify: a member calling the approve endpoint gets 403; approving produces chunks
identical in shape to an owner upload; a `FAILED` ingestion leaves zero chunks and a
persisted error message.

### Phase 5 — Hybrid Retrieval + Reranking
pgvector search, full-text search, RRF merge (Section 8.1), local cross-encoder
rerank, relevance threshold (Section 8.3).
Verify: a paraphrased query and an exact-code query both return the right chunk; a
query about content only in a `PENDING` document returns nothing.

### Phase 6 — Grounded Chat Endpoint
LLM call (configured via env, Section 2) with strict system prompt, citations
(Section 8.4), conversation context window (Section 8.5), streaming response.
Verify: an in-scope question gets a correctly cited answer; an out-of-scope question
is refused **without an LLM call** (check logs for retrieval-threshold short-circuit);
a follow-up question resolves a pronoun correctly.

### Phase 7 — Frontend
Chat pane with streaming + citation chips, Company Docs / My Docs views, upload with
status, owner approval queue, member invite UI, role-aware navigation.
Verify: full browser round trip — owner uploads, asks a question, sees cited answer;
member uploads, sees it pending, owner approves, member can now get an answer sourced
from it.

### Phase 8 — Deploy
Dockerfiles, deploy backend to Railway/Render free tier, frontend to Vercel, Supabase
free tier for the database, secrets via each platform's secret manager.
Verify: fresh clone + documented setup produces a working deployed system on entirely
free infrastructure.

### Phase 9 — Evaluation
Build the evaluation set and runner described in Section 15.
Verify: metrics are computed and recorded, not just eyeballed.

### Phase 10 — Full Test Pass (once, for everything)
`pytest`: ingestion correctness, hybrid retrieval + fusion correctness, grounding/
refusal behavior, approval-flow state transitions, workspace isolation (adversarial:
member reads another workspace's doc; member calls approve; member fetches a
`PENDING` document's chunks; user opens another user's chat session — all must fail).
`vitest`/`playwright`: chat round trip, upload flow, approval flow.
Verify: every check passes, or is triaged, fixed, and re-run — this is the real
acceptance bar for the project, not "verification passes" at each earlier phase.

---

## 10. Free-Model Resource Constraints

Assume limited hardware and API quota throughout:
- Local embedding and reranker models must stay "small" class (e.g. MiniLM/bge-small
  tier) — runnable on a normal laptop CPU without a GPU.
- No simultaneous/parallel model calls per request — retrieval, rerank, and generation
  run sequentially, one at a time.
- Retrieval candidates capped at ~15 pre-rerank, ~5–8 post-rerank (Section 8.2) — this
  isn't just quality, it's also what keeps prompts small enough for a free/rate-limited
  LLM tier.
- Use streaming only if the configured provider reliably supports it; if not,
  synchronous responses are fine — don't build a fallback streaming shim.
- No LLM fallback chain. One provider, one model, configured through env vars. If the
  free tier rate-limits, that's a documented limitation (Section 14 risk register), not
  a reason to add a second provider.

---

## 11. What NOT to Add (and why it's tempting)

| Thing | Why it's tempting | Why it's still wrong here |
|---|---|---|
| Redis + Celery | "Production systems use async queues" | No document volume here justifies a background worker; synchronous ingestion with a size cap is simpler and just as correct |
| MCP servers | Trendy protocol for tool-calling agents | There are no external tools to wrap — nothing for MCP to do |
| LangGraph / multi-agent supervisor | Looks more sophisticated | One retrieval pipeline handles every question type this app has; a router adds a failure mode (misrouting) without adding capability |
| SQL agent over structured data | Could answer "how many refunds last month" | Out of scope for a document knowledge base, and a much higher-risk feature (LLM-generated SQL against real data) this project doesn't need to demonstrate |
| Paid LLM/embedding/reranker APIs | Slightly better quality | Violates the hard free-tier constraint; a free local model plus a good retrieval pipeline is more than enough to prove the architecture works |
| Full JWT/RLS stack | "Enterprise auth is more impressive" | Supabase Auth + application-level membership checks demonstrate the same authorization understanding; RLS is a real, valid trade-off but is deliberately **not** auto-added here (Section 7) — the project intentionally shows application-level tenant isolation done correctly |
| S3 / external file storage | Conventional for "real" apps | The explicit requirement is DB-as-source-of-truth; `BYTEA` with a size cap satisfies it directly |
| Retrieving 30–50 chunks "to be safe" | Feels like it reduces missed answers | Slower, dilutes LLM attention, burns free-tier token quota — a good hybrid + rerank pipeline reliably finds the right 5–8 |
| A third role (ADMIN, VIEWER, etc.) | "More flexible permission model" | Nothing in this product needs more than OWNER/MEMBER; adding one is unused surface area |

---

## 12. FREE MODEL / CLAUDE CODE OPERATING RULES

This project is worked on by Claude Code running on **free/variable-quality models**
(OpenRouter free routing). These rules govern *how work gets done*, independent of
what gets built, and apply for the whole lifetime of the project — read them before
every session, not just once.

**Always:**
- Work in small, clearly scoped steps within the current phase only.
- Inspect the actual repository before proposing or making any change — read the
  relevant files, don't assume from memory what they contain.
- Prefer the simplest deterministic implementation that satisfies the requirement.
- Make minimal diffs. Touch only the files the current task requires.
- Explain the plan before a substantial change and wait for approval unless the user
  has explicitly said "implement this."
- Run the relevant static checks / tests after a change and report exact results.
- Keep context usage efficient — read targeted sections of large files instead of
  whole files when a targeted read is sufficient.
- Break a large task into smaller steps rather than attempting it in one pass.
- If a task is too large or ambiguous for reliable execution at the current model
  quality, say so explicitly and propose how to split it, rather than attempting it
  anyway and guessing.

**Never:**
- Never claim something works without having actually run it and observed the result.
- Never fabricate test results, API responses, database state, or model capabilities.
- Never assume a feature exists just because this file describes it — verify against
  the actual code first.
- Never make broad, uncontrolled changes across many files for one small task.
- Never rewrite working code without a concrete, stated reason tied to the current
  phase.
- Never change architecture (add/remove a major component) without first explaining:
  current behavior → the problem → the proposed change → why it's necessary → which
  files change — and getting confirmation.
- Never silently skip a failing check — report it as failing.
- Never use a paid model or paid API "because it would be easier" — the free-tier
  constraint in Section 2 is absolute.
- Never install a new dependency without stating why the existing stack can't do it.
- Never touch `.env`, secrets, or committed credentials casually.
- Never move to the next phase automatically, and never run `git commit`.

### Phase Workflow (for every requested task)
1. Read the relevant section(s) of this file.
2. Inspect the existing implementation for the files involved.
3. State what is currently implemented (verified from code, not assumed).
4. State what is actually missing or broken.
5. Produce a short implementation plan.
6. Wait for approval before large changes, unless the user explicitly asked for
   implementation directly.
7. Implement only the agreed scope — nothing adjacent, no "while I'm here" cleanups.
8. Run the relevant static checks/tests.
9. Review the diff for anything outside the intended scope.
10. Report: what changed, files changed, checks run, results, remaining issues.
11. Stop and wait for the next instruction.

---

## 13. Configuration — `.env.example`

```
# Database (Supabase / Postgres — the only datastore)
DATABASE_URL=
SUPABASE_URL=
SUPABASE_ANON_KEY=
SUPABASE_SERVICE_ROLE_KEY=

# LLM — one provider, configured, never hardcoded
LLM_PROVIDER=
LLM_MODEL=
LLM_API_KEY=
LLM_BASE_URL=

# Embeddings — local, free, pinned
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
EMBEDDING_DIMENSION=384

# Reranker — local, free
RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2

# Retrieval tuning
RETRIEVAL_CANDIDATE_COUNT=15
RETRIEVAL_FINAL_COUNT=8
RETRIEVAL_RELEVANCE_THRESHOLD=0.3

# Uploads
MAX_UPLOAD_SIZE_MB=10
```

Never commit `.env`. `.env.example` documents every variable with no real values.

---

## 14. Risk Register

| Risk | Mitigation |
|---|---|
| Embedding model swapped mid-project, old vectors incomparable to new queries | Pin `EMBEDDING_MODEL`/`EMBEDDING_DIMENSION` from Phase 1; changing either requires re-embedding the whole index, never mixing |
| Free-tier LLM rate limits stall the demo | Documented, expected limitation of a single free-tier provider — not a reason to add a fallback chain; log and surface a clear error to the user |
| Large upload makes the request slow | `MAX_UPLOAD_SIZE_MB` enforced before the file is read into memory (Section 6) |
| A member's pending document leaks into search results | Structurally impossible while the Section 5 invariant holds — `PENDING`/`REJECTED` documents have zero chunks; treat any violation as a Section 7 transaction bug |
| Retrieval returns irrelevant/too many chunks | Bounded pre-rerank (~15) and post-rerank (~5–8) context, per Section 8 |
| Assistant answers general trivia instead of refusing | Two-layer grounding (Section 8.3) — retrieval threshold plus an explicit, separately-stated scope boundary in the prompt |
| Cross-workspace data leak | Every workspace-owned table filters on `workspace_id` server-side (Section 4); this is the #1 thing to check on any new endpoint |
| A colleague's personal (pending) document text leaks through chunks | Enforced structurally by ingestion timing (Section 5), not by a query-time filter alone — verify both hold together |
| Free/small local reranker gives noisier scores than a hosted one | Acceptable, documented trade-off for a free-tier project; the evaluation set (Section 15) quantifies this rather than assuming it's fine |
| Secrets committed to git | `.gitignore` from commit #1, `.env.example` only |

---

## 15. Evaluation (Phase 9)

Build a small, reproducible evaluation set of **30–50 questions** in `backend/eval/`,
covering: semantic-paraphrase questions, exact-keyword/identifier questions, numeric
facts, questions needing multiple chunks, genuinely out-of-scope questions,
in-scope-but-unsupported questions, and follow-up questions needing conversation
context.

Compute and record, not just assert:
- **Retrieval Recall@K** — did the right chunk make it into the candidate set.
- **MRR** (or similar) — how highly the right chunk ranked after fusion + rerank.
- **Citation correctness** — does the returned citation actually match the source of
  the answer.
- **Grounded-answer correctness** — is the answer actually supported by the cited
  chunk(s).
- **Refusal correctness** — does an out-of-scope/unsupported question get refused,
  and does an in-scope question *not* get incorrectly refused.
- **Workspace isolation** — a question scoped to Workspace A never returns a chunk from
  Workspace B, run against the eval harness itself as a sanity check.

This is what turns "the chatbot generates plausible answers" into "the RAG system
measurably works" — the second is what should go on a resume and be defensible in an
interview.

---

## 16. Coding Standards

- **Python**: `ruff`, full type hints, every API input/output is a Pydantic model — no
  raw `dict` in request/response signatures.
- **TypeScript**: strict mode, `eslint` + `prettier`, no `any`.
- **Commits**: one logical change per commit, message states what and why. User commits
  manually.
- **No silent TODOs**: anything deferred goes in Section 14's risk register or a
  tracked issue, not a comment that gets forgotten.
- **Logging**: structured logs with request ID, workspace ID, user ID (where safe),
  document ID, and durations for ingestion/retrieval/rerank/LLM steps. Never log
  passwords, API keys, raw file bytes, or full chat content unnecessarily. No paid
  observability platform — plain structured logs are sufficient here.

---

## 17. Resume Pitch

> I built a multi-tenant company knowledge assistant — many isolated company
> workspaces in one deployment — where an owner invites members, uploads official
> documents that publish immediately, and members contribute their own documents
> pending owner approval. Documents are stored directly in Postgres with pgvector;
> retrieval is hybrid (semantic + full-text, fused with RRF) with a local cross-encoder
> reranker, two-layer grounding, and backend-verified citations. The entire stack runs
> on free-tier and locally-hosted models, and I evaluated retrieval and grounding
> quality against a hand-built question set rather than assuming it worked.
