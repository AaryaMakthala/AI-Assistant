# CLAUDE.md — Enterprise AI Knowledge Intelligence Agent

This file is the single source of truth for building this project. Read it fully before
writing any code. Work **one phase at a time, in order**. After finishing a phase:
summarize what was built, list modified files, and **stop and wait for explicit
go-ahead** before starting the next phase. Do not skip phases or merge them to "save
time" — most production incidents in agentic-AI systems come from skipping the boring
steps (auth, input validation, sandboxing) to get to the fun part (agents).

**Testing is deferred.** Per-phase automated test suites are NOT written or run during
Phases 0–13. Static verification only at each phase (compiles/typechecks, `ruff`,
`eslint`, `tsc --noEmit`, and quick manual/stubbed sanity checks where useful). The full
`pytest` / `vitest` suite — including the security suite — is written and run once, in
Phase 14, after the whole system is running end-to-end. This is a deliberate speed
tradeoff: it means bugs from an earlier phase may not surface until Phase 14. Flag
anything you're uncertain about in your phase summary rather than silently assuming it
works.

If any instruction here conflicts with speed or convenience, this file wins — with the
above exception for testing.

---

## 0. Ground Rules for Claude Code

1. **One phase per work session.** Do not start Phase N+1 until Phase N's deliverables
   are in place and the user has confirmed.
2. **No hardcoded secrets, ever.** All keys/URLs come from `.env`, loaded via Pydantic
   `BaseSettings`. `.env` is gitignored from commit #1. Ship `.env.example` with empty
   placeholders instead.
3. **Every external input is hostile until proven otherwise**: file uploads, chat
   messages, LLM-generated SQL, retrieved document text, MCP tool arguments. Validate
   before use, not after something breaks.
4. **The LLM never gets raw write access to anything.** No LLM-generated SQL runs without
   passing through the guardrails in Section 4.3. No MCP tool executes an action (write,
   delete, external API call with side effects) without an explicit allowlist.
5. **Testing happens once, at the end (Phase 14).** Do not write test files during
   Phases 0–13 unless explicitly asked. Do not run `pytest`/`vitest` during Phases 0–13
   unless explicitly asked. Static checks (lint/typecheck/compile) still run every phase.
6. **Prefer boring, well-understood tools over clever ones.** This project already has
   enough novel surface area (agents, MCP, RAG) — don't also make the plumbing exotic.
7. **Small, reviewable commits.** One logical change per commit, message says what and
   why. The user commits manually — do not run `git commit` unless explicitly asked to.

---

## 1. Project Overview

An internal AI assistant that lets employees ask natural-language questions and get
answers sourced from three places, automatically routed:

- **Unstructured knowledge** (PDFs, DOCX, CSV, policies, manuals) → RAG
- **Structured business data** (customers, orders, employees) → a guarded SQL agent
- **External systems** (GitHub, later Slack/Drive) → MCP tool servers

The system must never let a user query see data they're not authorized to see, never let
the LLM run arbitrary destructive SQL, and never let a malicious document "instruct" the
agent to do something the user didn't ask for (prompt injection via retrieved content).

---

## 2. Finalized Tech Stack

```
Frontend
  Next.js 15 (App Router) + TypeScript
  Tailwind CSS + shadcn/ui
  Vercel AI SDK (for streaming — don't hand-roll SSE parsing)

Backend
  Python 3.11, FastAPI, Pydantic v2, SQLAlchemy 2.0 (async)
  uv or poetry for dependency management (not bare pip)

Agent Orchestration
  LangGraph (supervisor + RAG/SQL/MCP sub-agents)
  LangChain (loaders, retrievers, prompt templates only — not the agent runtime)

LLMs
  Primary:  Google Gemini Flash
  Fallback: Groq (Llama 3.3 70B)

Embeddings
  BAAI/bge-small-en-v1.5 (HuggingFace, local inference via sentence-transformers)
  Pin the exact model — never swap embedding models on a live index (Section 7, Risk 1)

Vector Storage
  Postgres + pgvector (via Supabase) — one database for structured AND vector data.

Relational Database
  PostgreSQL via Supabase (also holds users, orgs, chat history, document metadata,
  and the structured business tables the SQL agent queries — see Section 4.3, this
  project runs directly against the real application database, not a separate demo copy)

MCP
  Official MCP Python SDK
  Servers: Document MCP, Database MCP (read-only), GitHub MCP (read-only initially)

Document Processing
  PyMuPDF (PDF), python-docx (DOCX), pandas + openpyxl (XLSX/CSV)
  Tesseract OCR — only if you actually encounter scanned PDFs; don't build it day one

Auth & Authorization
  Supabase Auth (JWT) + Postgres Row-Level Security (RLS) for data access control.
  Do NOT rely on backend-only checks — RLS is your last line of defense if a backend
  check is ever missed.

Background Jobs
  Celery + Redis for document ingestion (embedding generation must never block a
  request thread)

Observability
  Loguru (structured logs), Sentry (errors), LangSmith (agent traces — dev/staging only,
  never pipe production user data to a third-party trace tool without checking your
  data-handling policy first)

Deployment
  Docker + Docker Compose locally; Vercel (frontend) + Railway/Render (backend) + Supabase
```

---

## 3. Repository Structure

```
enterprise-ai-agent/
├── CLAUDE.md                  ← this file, lives at repo root
├── .env.example
├── .gitignore
├── docker-compose.yml
├── backend/
│   ├── pyproject.toml
│   ├── app/
│   │   ├── main.py
│   │   ├── config.py           # Pydantic BaseSettings, reads .env
│   │   ├── api/                # FastAPI routers
│   │   ├── agents/              # LangGraph supervisor + sub-agents
│   │   ├── rag/                 # ingestion, chunking, retrieval
│   │   ├── sql_agent/            # guarded SQL generation + execution
│   │   ├── mcp_servers/          # document / database / github MCP servers
│   │   ├── db/                   # SQLAlchemy models, migrations (alembic)
│   │   ├── security/              # auth, RLS helpers, input validation, rate limiting
│   │   └── workers/                # Celery tasks
│   └── tests/                       # written once, in Phase 14
│       ├── unit/
│       ├── integration/
│       └── security/            # SQLi attempts, prompt-injection attempts, auth bypass
└── frontend/
    ├── package.json
    └── src/
        ├── app/
        ├── components/
        └── lib/
```

---

## 4. Security Principles (non-negotiable, apply from Phase 1 onward)

Deferred testing does not mean deferred security *design*. Every item below is built in
from the start — only the automated proof that it works is deferred to Phase 14.

### 4.1 Secrets
- All API keys, DB URLs, JWT secrets live in `.env`, never in code or commit history.
- Use a `.env.example` with empty values so the repo documents what's needed.
- If a secret ever gets committed, treat it as compromised — rotate it, don't just delete
  the commit.

### 4.2 File Upload Safety
- Enforce file type allowlist (pdf, docx, csv, xlsx, txt) and a max size limit at the API
  layer, before the file touches disk.
- Process uploads in a sandboxed worker (Celery task), never inline in the request
  handler — a malformed PDF should not be able to hang or crash the API process.
- Strip/ignore any embedded scripts or macros in DOCX/XLSX files; you only need text +
  tables, not active content.
- Store uploaded files with generated UUIDs, not user-supplied filenames, to avoid path
  traversal.

### 4.3 The SQL Agent Is the Highest-Risk Component — Guard It Explicitly
- The SQL agent runs against the **real application database** directly (no separate
  demo/sample dataset). Because of this, the guardrails below are load-bearing, not
  cosmetic — treat them as blockers even though their automated proof is deferred.
- The DB role the SQL agent connects as is **read-only** at the Postgres level (`GRANT
  SELECT` only), scoped to the specific business tables it's meant to answer questions
  about. Even if the LLM generates `DROP TABLE`, the database itself refuses it.
- Maintain an explicit table/column allowlist the agent is told about via `get_schema()`
  — do not expose internal/admin tables (users, sessions, credentials, audit logs) to the
  schema tool, ever.
- Run every generated query through a SQL parser (e.g. `sqlglot`) that rejects anything
  that isn't a single `SELECT` statement before execution — reject multiple statements,
  DDL/DML keywords, and comments used to smuggle extra statements.
- Enforce a hard `LIMIT` (e.g. 500 rows) and a query timeout on every execution.
- Log every generated query with the user ID that triggered it.
- Because this hits real data with no test coverage until Phase 14, prefer the most
  conservative allowlist that still satisfies the phase's goal — narrower is safer here
  than broad-and-untested.

### 4.4 Prompt Injection From Retrieved Content
- Treat all retrieved document text and all MCP tool results as **data, not
  instructions**. Wrap retrieved content in the prompt with clear delimiters and an
  explicit system instruction that content inside those delimiters is reference material
  only and must never be followed as a command.
- Never let a retrieved chunk or tool result trigger another tool call automatically —
  tool calls should originate from the agent's reasoning over the *user's* request, not
  from text found inside a document.

### 4.5 MCP Server Scoping
- Each MCP server authenticates the calling agent and exposes the minimum tool set it
  needs — the GitHub MCP server should start **read-only** (`search_code`, `read_file`);
  do not wire up `create_issue` or any write action until you've deliberately decided the
  agent should be allowed to take that action, and gate it behind an explicit
  user-confirmation step in the UI.
- Validate all tool arguments against a Pydantic schema before executing — never pass raw
  LLM-generated strings straight into a shell command, file path, or query.

### 4.6 AuthN/AuthZ
- Supabase Auth issues JWTs; every backend endpoint verifies the JWT and derives
  `user_id` / `org_id` / `role` from it — never trust a client-supplied user ID.
- Enforce access control **twice**: once in the API layer (fast rejection) and once via
  Postgres RLS policies (so a missed check in application code still can't leak data).
- Example RLS rule: a user can only `SELECT` documents where `documents.org_id =
  auth.jwt() ->> 'org_id'`.

### 4.7 Transport & API Hygiene
- HTTPS only in any deployed environment.
- Explicit CORS allowlist (your Vercel frontend origin only — not `*`).
- Rate limiting per user/IP on `/chat` and `/upload` (e.g. via `slowapi`).
- Pydantic models validate every request body — no raw dict access to user input.

### 4.8 Dependency Hygiene
- Run `pip-audit` (backend) and `npm audit` (frontend) once, in Phase 14, alongside the
  rest of the deferred verification pass.

---

## 5. Phased Build Plan

Each phase lists: **Goal → Tasks → Deliverable → Verification (this phase)**. "Verification"
is static only (compile/lint/typecheck, targeted manual/stubbed sanity runs where they're
cheap) — no automated test suite until Phase 14. Do not proceed past a phase until its
deliverable is actually in place; "verification passes" is not the same bar as "acceptance
criteria met," and that gap is intentional and closes in Phase 14.

### Phase 0 — Scaffolding
- Goal: an empty but correctly wired monorepo.
- Tasks: create folder structure above; init FastAPI app with `/health`; init Next.js
  app; docker-compose with postgres+redis; `.env.example`; `.gitignore`.
- Verification: `docker compose up` starts postgres+redis; `GET /health` returns 200;
  Next.js dev server renders a blank page.

### Phase 1 — Backend Core & Config
- Goal: settings, DB connection, base error handling, logging.
- Tasks: `config.py` (Pydantic BaseSettings), async SQLAlchemy engine, Alembic setup,
  Loguru structured logging, global exception handler that never leaks stack traces to
  the client.
- Verification: app boots with missing `.env` failing loudly and clearly, not silently.

### Phase 2 — Database Schema & RLS
- Goal: core tables + row-level security from day one, not bolted on later.
- Tasks: `organizations`, `users`, `documents`, `document_chunks` (with vector column),
  `chat_sessions`, `chat_messages`. Write RLS policies for org-scoped access. Alembic
  migration.
- Verification: migration applies cleanly; manually confirm (e.g. via `psql`) that a
  cross-org `SELECT` under the app's DB role returns nothing. Full automated proof of
  this is part of the Phase 14 security suite.

### Phase 3 — Document Ingestion Pipeline
- Goal: upload → extract → chunk → embed → store, running as a background job.
- Tasks: upload endpoint (validated, sandboxed), Celery task for extraction (PyMuPDF /
  python-docx / pandas), `RecursiveCharacterTextSplitter`, bge-small embeddings, store
  vectors + metadata in pgvector.
- Verification: uploading a sample PDF results in queryable chunks in `document_chunks`
  with correct `org_id` and `source`/`page` metadata (manual check).

### Phase 4 — RAG Retrieval + Basic Chat
- Goal: a working single-purpose RAG chat endpoint (no SQL/MCP yet), streaming to the
  frontend.
- Tasks: retrieval function (top-k vector search, org-scoped), prompt template with
  injection-safe delimiters (Section 4.4), streaming response via SSE, source citations
  returned alongside the answer.
- Verification: asking about content from an uploaded doc returns a correct answer with
  a citation; asking something not in any doc returns an honest "I don't have that
  information" (manual/spot check — automated proof deferred to Phase 14).

### Phase 5 — Guarded SQL Agent
- Goal: natural-language → safe SQL → result, running directly against the real
  application's structured business tables (see Section 4.3 — no separate demo dataset).
- Tasks: `get_schema()`/`describe_table()`/`execute_query()` tools, sqlglot validation
  layer, row limit + timeout, read-only DB role scoped to an explicit table allowlist,
  full audit logging.
- Verification: manually attempt a `DROP TABLE`/stacked-query prompt and confirm it's
  rejected before reaching the database. This one check is worth doing live even though
  the formal test is deferred — it's the highest-risk component in the system.

### Phase 6 — MCP Servers
- Goal: Document MCP and Database MCP wrapping Phases 3–5 behind the MCP protocol;
  GitHub MCP (read-only) as the external-system example.
- Tasks: implement MCP servers per Section 4.5 scoping rules; argument validation on
  every tool.
- Verification: an MCP client can discover and call each tool; manually confirm malformed
  arguments are rejected with a clear error, not a crash.

### Phase 7 — LangGraph Multi-Agent Orchestration
- Goal: a supervisor agent that routes a question to RAG agent, SQL agent, or MCP agent
  (or a combination), then synthesizes a final answer.
- Tasks: intent-routing node, sub-agent nodes wrapping Phases 4–6, response synthesis
  node, max-iteration guard to prevent infinite agent loops.
- Verification: a mixed question ("what's our refund policy, and how many refunds did we
  process last month") correctly triggers both RAG and SQL agents and merges the answer
  (manual run).

### Phase 8 — Frontend Chat UI
- Goal: a polished chat interface (see Section 6 spec).
- Tasks: streaming chat pane, file upload with progress, source citation display, chat
  history sidebar, markdown rendering for answers.
- Verification: a full round trip works end-to-end from the browser against the real
  backend, with visible sources (manual click-through).

### Phase 9 — Auth & RBAC Integration (Frontend)
- Goal: login/signup via Supabase Auth, JWT attached to every API call, role-aware UI
  (e.g. hide admin-only views).
- Verification: manually confirm an unauthenticated request to a protected endpoint is
  rejected, and a logged-in user only sees their org's data in the UI.

### Phase 10 — Background Jobs Hardening
- Goal: Celery/Redis handles all slow work; retries and dead-letter handling for failed
  ingestion jobs.
- Verification: a large (e.g. 100-page) PDF upload doesn't block the API and completes
  asynchronously with a visible status in the UI.

### Phase 11 — Observability
- Goal: Sentry for errors, LangSmith for agent traces (non-prod only unless
  data-handling is explicitly reviewed), structured logs correlated by request ID.
- Verification: a deliberately triggered error shows up in Sentry with useful context and
  no leaked secrets.

### Phase 12 — MCP/Agent Hardening Review
- Goal: a design-level pass over everything built so far against Section 4, before the
  formal test suite is written.
- Tasks: re-read Sections 4.3–4.6 against the actual code; fix anything found; note
  anything ambiguous for the Phase 14 test suite to specifically target.
- Verification: written summary of the review; no automated tests yet.

### Phase 13 — Dockerize & Deploy
- Goal: reproducible deployment.
- Tasks: production Dockerfiles for frontend/backend, docker-compose for local
  full-stack, deploy backend to Railway/Render, frontend to Vercel, confirm env vars are
  set via each platform's secret manager (not committed anywhere).
- Verification: a fresh clone + documented setup steps produces a working deployed
  system.

### Phase 14 — Full Test Pass (Testing happens here, once, for everything)
- Goal: the single point where every phase's acceptance criteria gets a real automated
  test, run against the fully assembled system.
- Tasks:
  - Backend: `pytest` covering unit + integration for Phases 1–11 (RLS cross-org
    isolation, ingestion correctness, RAG citation/honesty behavior, SQL agent
    injection rejection, MCP argument validation, agent routing/max-iteration guard).
  - `tests/security/`: SQL injection attempts against the SQL agent, prompt-injection
    documents fed into RAG, auth-bypass attempts (missing JWT, wrong org_id, expired
    token), file-upload fuzzing (oversized files, disallowed types, zip bombs).
  - Frontend: `vitest`/`playwright` for the chat UI round trip and auth-gated routes.
  - `pip-audit` and `npm audit`.
- Verification (this is the real acceptance bar for the whole project): every check
  above passes, or every failure is triaged, fixed, and re-run. Anything not fixable
  immediately goes into the Risk Register (Section 7) as a tracked, explicit gap — not a
  silent TODO.

---

## 6. Chat UI Spec (Phase 8 detail)

- **Layout**: sidebar (chat history, new chat, document library) + main chat pane +
  optional right panel for citations/sources on the active answer.
- **Streaming**: tokens appear incrementally (Vercel AI SDK `useChat` or equivalent),
  with a visible "thinking/routing" indicator when the agent is deciding which sub-agent
  to use.
- **Citations**: every RAG-sourced answer shows clickable source chips
  (`refund_policy.pdf · page 4`); SQL-sourced answers show the executed query on request
  (collapsed by default, expandable — transparency without clutter).
- **Upload**: drag-and-drop, progress bar tied to the Celery job status, clear
  error states for rejected file types/oversized files.
- **Empty/error states**: honest "I don't know" is a first-class UI state, not an
  afterthought — never let the UI imply confidence the answer doesn't have.

---

## 7. Risk Register — Things That Commonly Go Wrong

| Risk | Why it happens | Mitigation |
|---|---|---|
| Vector search returns garbage after a model change | Query embedded with a different model than the stored chunks | Pin embedding model version in config; re-embed the whole index on any change, never mix |
| SQL agent touches unintended tables | Schema tool exposes more than it should | Explicit table allowlist in `get_schema()`, never introspect the full DB automatically |
| SQL agent runs against real data with no automated proof until Phase 14 | Testing deferred by design | Keep the allowlist as narrow as possible per phase; do the one manual injection check in Phase 5 even though the suite is deferred |
| Free-tier LLM rate limits stall production | Single provider dependency | Gemini → Groq fallback wired from Phase 4, not bolted on later |
| Large upload blocks the API | Synchronous processing in the request handler | All ingestion is a Celery task from Phase 3 onward |
| Prompt injection via a malicious uploaded doc | Retrieved text treated as instructions | Delimiter + system-instruction pattern (4.4), formally tested in Phase 14 |
| Agent loops forever | No termination condition in the LangGraph graph | Explicit max-iteration guard on the supervisor node |
| Data leak across organizations | Missing or bypassed authz check | Enforce both API-layer checks and Postgres RLS (4.6); formally tested in Phase 14 |
| Secrets committed to git | `.env` accidentally staged | `.gitignore` from commit #1, `.env.example` only, pre-commit secret scan |
| Cost/quota overrun | Free-tier limits hit silently | Log token usage per request from Phase 4; alert threshold in Phase 11 |
| Zip bomb / malformed file crashes a worker | No size/type validation before processing | Enforce limits in Phase 3 before the file reaches the extraction library |
| A bug from an early phase isn't caught until Phase 14 | Per-phase testing deferred by design | Phase 12 hardening review exists specifically to catch design-level issues before the formal suite; keep phase summaries honest about uncertainty |

---

## 8. Coding Standards

- **Python**: `black` + `ruff`, full type hints, every API input/output is a Pydantic
  model — no raw `dict` in request/response signatures.
- **TypeScript**: strict mode on, `eslint` + `prettier`, no `any`.
- **Tests**: written once, in Phase 14 — `pytest` (backend), `vitest`/`playwright`
  (frontend). Each phase's deliverable maps to at least one Phase 14 test.
- **Commits**: one logical change per commit; message states what changed and why. User
  commits manually.
- **No silent TODOs**: if something is deferred, it goes in this file's risk register or
  a tracked issue, not a comment that gets forgotten.

---

## 9. `.env.example`

```
# Database
DATABASE_URL=
SUPABASE_URL=
SUPABASE_ANON_KEY=
SUPABASE_SERVICE_ROLE_KEY=

# LLMs
GEMINI_API_KEY=
GROQ_API_KEY=

# Redis / Celery
REDIS_URL=

# Auth
JWT_SECRET=

# Observability
SENTRY_DSN=
LANGSMITH_API_KEY=

# GitHub MCP
GITHUB_TOKEN=
```

---
