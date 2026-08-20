<div align="center">

# Knowledge Assistant

### A Multi-Tenant Company Knowledge Assistant

A multi-tenant company knowledge assistant where employees ask natural-language questions and get answers sourced from approved company documents, using hybrid retrieval with local reranking and backend-verified citations.

[Documentation](#overview) &nbsp;•&nbsp; [Architecture](#architecture) &nbsp;•&nbsp; [Getting Started](#getting-started) &nbsp;•&nbsp; [Deployment](#deployment)

<br/>

![Next.js](https://img.shields.io/badge/Next.js-000000?style=for-the-badge&logo=nextdotjs&logoColor=white)
![TypeScript](https://img.shields.io/badge/TypeScript-3178C6?style=for-the-badge&logo=typescript&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-4169E1?style=for-the-badge&logo=postgresql&logoColor=white)
![Supabase](https://img.shields.io/badge/Supabase-3FCF8E?style=for-the-badge&logo=supabase&logoColor=white)

</div>

---

## Overview

Employees ask questions in natural language and the system answers from approved company documents using:

- **Hybrid retrieval** — semantic (pgvector) + full-text search, fused with Reciprocal Rank Fusion
- **Local reranking** — cross-encoder model scores relevance without external APIs
- **Two-layer grounding** — retrieval threshold + LLM prompt to prevent hallucination
- **Backend-verified citations** — structured citations built from actual source chunks

The system supports multiple independent company workspaces, each fully isolated. Owners upload documents that publish immediately; members contribute documents pending owner approval.

---

## Architecture

```
User Question
     │
     ▼
Authentication (Supabase Auth) + workspace authorization
     │
     ▼
Hybrid Retrieval
  ├── pgvector cosine similarity (semantic)
  └── PostgreSQL full-text search (keyword)
     │
     ▼
Reciprocal Rank Fusion → top ~15 candidates
     │
     ▼
Local cross-encoder reranking → top 5-8 chunks
     │
     ▼
Relevance threshold check
  ├── below threshold → honest refusal (no LLM call)
  └── above threshold → LLM generation
     │
     ▼
Answer + backend-constructed citations
```

### Multi-Tenancy

A single deployment hosts many independent company workspaces. Every workspace-owned table carries `workspace_id` and every query filters on it server-side. Two roles only: **OWNER** and **MEMBER**.

| Action | Owner | Member |
|---|---|---|
| Create workspace | ✅ (becomes owner) | — |
| Invite members | ✅ | ❌ |
| Upload → published immediately | ✅ | ❌ |
| Upload → pending approval | — | ✅ |
| Approve/reject pending documents | ✅ | ❌ |
| Search approved knowledge / chat | ✅ | ✅ |

### Document Lifecycle

```
OWNER upload  → validate → extract → chunk → embed → store  → READY  (immediate)
MEMBER upload → validate → store document only               → PENDING
OWNER approves a PENDING doc → extract → chunk → embed → store → READY
OWNER rejects a PENDING doc  → REJECTED (never ingested, permanent)
Any ingestion failure → FAILED, with the error persisted on the row
```

---

## Technology Stack

| Component | Technology |
|---|---|
| Frontend | Next.js (App Router) + TypeScript + Tailwind CSS + shadcn/ui |
| Backend | Python 3.11, FastAPI, Pydantic v2, SQLAlchemy 2.0 (async), Alembic |
| Database | PostgreSQL via Supabase (free tier) + pgvector + full-text search |
| Auth | Supabase Auth for identity; application tables for authorization |
| LLM | One model, configured via environment variables (no hardcoded provider) |
| Embeddings | Local, free model via sentence-transformers (BAAI/bge-small-en-v1.5) |
| Reranking | Local, free cross-encoder (cross-encoder/ms-marco-MiniLM-L-6-v2) |

**Infrastructure:** Vercel (frontend) + Railway/Render free tier (backend) + Supabase (database). No Docker, no Redis, no Celery.

---

## Getting Started

### Prerequisites

- Python 3.11 or later
- Node.js 18 or later
- A Supabase project (free tier)
- `uv` for backend dependency management

### Supabase Setup

1. Create a new Supabase project at [supabase.com](https://supabase.com)
2. Enable the `vector` extension in the SQL Editor: `CREATE EXTENSION IF NOT EXISTS vector;`
3. Run the database migrations (see `backend/alembic/versions/`)
4. Note your project URL and keys from Settings → API

### Backend Setup

```bash
cd backend
uv sync
cp ../.env.example ../.env   # fill in real values
```

Start the backend:

```bash
uvicorn app.main:app --reload
```

The API runs at `http://localhost:8000` and exposes:
- `GET /health` — health check
- All other endpoints under `/api/v1/`

### Frontend Setup

```bash
cd frontend
npm install
```

Create `.env.local` in the frontend directory:

```
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_SUPABASE_URL=your-supabase-url
NEXT_PUBLIC_SUPABASE_ANON_KEY=your-supabase-anon-key
```

Start the frontend:

```bash
npm run dev
```

The app runs at `http://localhost:3000`.

---

## Deployment

### Backend (Railway or Render)

1. **Create a Railway or Render account** (free tier)
2. **Connect your GitHub repository**
3. **Configure the service:**
   - Root directory: `backend`
   - Build command: `pip install -e .`
   - Start command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
4. **Set environment variables** (see `.env.example` for the full list):
   - `ENVIRONMENT=production`
   - `DATABASE_URL=your-supabase-database-url`
   - `SUPABASE_URL=your-supabase-url`
   - `SUPABASE_ANON_KEY=your-supabase-anon-key`
   - `SUPABASE_SERVICE_ROLE_KEY=your-supabase-service-role-key`
   - `JWT_SECRET=your-jwt-secret`
   - `LLM_PROVIDER=your-provider`
   - `LLM_MODEL=your-model`
   - `LLM_API_KEY=your-api-key`
   - `CORS_ALLOW_ORIGINS=https://your-frontend-domain.vercel.app`
5. **Deploy** — Railway/Render will build and start the service

### Frontend (Vercel)

1. **Create a Vercel account** (free tier)
2. **Import your GitHub repository**
3. **Configure the project:**
   - Framework: Next.js
   - Root directory: `frontend`
4. **Set environment variables:**
   - `NEXT_PUBLIC_API_URL=https://your-backend-domain.onrender.com`
   - `NEXT_PUBLIC_SUPABASE_URL=your-supabase-url`
   - `NEXT_PUBLIC_SUPABASE_ANON_KEY=your-supabase-anon-key`
5. **Deploy** — Vercel will build and deploy automatically

### Connecting Frontend to Backend

After both services are deployed:
1. Set `CORS_ALLOW_ORIGINS` on the backend to include your Vercel frontend URL
2. Set `NEXT_PUBLIC_API_URL` on the frontend to your backend URL
3. Redeploy both services

---

## Environment Variables

See `.env.example` for the full list. Key variables:

### Backend

| Variable | Required | Description |
|---|---|---|
| `DATABASE_URL` | Yes | PostgreSQL connection string (Supabase) |
| `SUPABASE_URL` | Yes | Supabase project URL |
| `SUPABASE_ANON_KEY` | Yes | Supabase anonymous key |
| `SUPABASE_SERVICE_ROLE_KEY` | Yes | Supabase service role key |
| `JWT_SECRET` | Yes | JWT secret for token verification |
| `LLM_PROVIDER` | Yes | LLM provider name |
| `LLM_MODEL` | Yes | LLM model name |
| `LLM_API_KEY` | Yes | LLM API key |
| `LLM_BASE_URL` | No | LLM base URL (optional) |
| `CORS_ALLOW_ORIGINS` | No | Comma-separated list of allowed origins (default: `http://localhost:3000`) |
| `ENVIRONMENT` | No | `development`, `staging`, or `production` (default: `development`) |

### Frontend

| Variable | Required | Description |
|---|---|---|
| `NEXT_PUBLIC_API_URL` | Yes | Backend API URL (default: `http://localhost:8000`) |
| `NEXT_PUBLIC_SUPABASE_URL` | Yes | Supabase project URL |
| `NEXT_PUBLIC_SUPABASE_ANON_KEY` | Yes | Supabase anonymous key |

---

## Repository Structure

```
knowledge-assistant/
├── CLAUDE.md                    # Source of truth
├── .env.example
├── .gitignore
├── docker-compose.yml           # Legacy / unused (see CLAUDE.md Infrastructure Constraint)
├── backend/
│   ├── pyproject.toml
│   ├── render.yaml              # Render deployment config
│   ├── alembic/                 # Database migrations
│   ├── app/
│   │   ├── main.py              # FastAPI application
│   │   ├── config.py            # Pydantic BaseSettings
│   │   ├── api/                 # Routers: auth, workspaces, documents, chat
│   │   ├── ingestion/           # Extract → chunk → embed (synchronous)
│   │   ├── retrieval/           # Hybrid search, reranking, grounding
│   │   ├── db/                  # SQLAlchemy models, Alembic migrations
│   │   └── security/            # Auth verification, workspace authorization
│   └── tests/                   # Unit, integration, security tests
└── frontend/
    ├── package.json
    └── src/
        ├── app/                 # Next.js App Router pages
        ├── components/          # UI components
        └── lib/                 # API client, auth, hooks
```

---

## License

This project is currently unlicensed. Add a license file if you intend to distribute or open-source this project.

---

<div align="center">

Built with FastAPI, Next.js, and Supabase

</div>
