"""Phase 1C migration integration tests: legacy -> canonical (0001..0009).

These tests are the Phase 1C gate. They apply the *real* Alembic migrations to a
disposable Postgres, seed representative legacy data (orgs, owners/members, org-wide
and personal documents with real files on disk, chunks, chats with all four legacy
roles), run the host-side byte backfill between 0008 and 0009, and then assert every
locked Phase 1C decision:

* workspace/member migration (roles OWNER/MEMBER, status ACTIVE, ids preserved)
* document status mapping (incl. the READY-with-zero-chunks -> FAILED rule)
* the section 5 chunk invariant (no chunks on non-READY documents)
* byte backfill with SHA-256, missing-file and duplicate-checksum handling
* generated content_tsv, composite workspace/document integrity
* chat role coercion and citations -> sources
* workspace-scoped RLS isolation
* auth provisioning (workspace + OWNER membership + workspace_id claim)
* HNSW vector index, exactly the seven canonical tables, no legacy tables left

**Safety.** The database is chosen via PHASE1C_TEST_DATABASE_URL, read from the
*environment only* — never from the repo .env, which may point at a real deployment.
The URL must be localhost and its database name must contain "test", or the module
skips. The test resets that database (DROP SCHEMA public/auth CASCADE) and rebuilds
it from scratch; point it at anything but a disposable local database and it will
not run.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import uuid
from datetime import datetime
from pathlib import Path

import pytest
from alembic.config import Config
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine, create_async_engine

from alembic import command
from scripts.backfill_file_data import BackfillReport, backfill_file_data

_TEST_DATABASE_URL = "PHASE1C_TEST_DATABASE_URL"
_REPO_ROOT = Path(__file__).resolve().parents[3]
_BACKEND = _REPO_ROOT / "backend"
_ALEMBIC_INI = _BACKEND / "alembic.ini"
_AUTH_BOOTSTRAP = _BACKEND / "scripts" / "dev_auth_schema.sql"

#: A complete, valid environment for app.config.Settings — the conftest strips .env,
#: so these must be provided explicitly whenever anything calls get_settings().
_REQUIRED_ENV = {
    "SUPABASE_URL": "https://test.supabase.co",
    "SUPABASE_ANON_KEY": "test-anon-key",
    "SUPABASE_SERVICE_ROLE_KEY": "test-service-role-key",
    "LLM_PROVIDER": "test-provider",
    "LLM_MODEL": "test-model",
    "LLM_API_KEY": "test-llm-api-key",
    "JWT_SECRET": "test-jwt-secret-that-is-long-enough-to-pass-validation",
}

# ---------------------------------------------------------------------------
# Seed data
# ---------------------------------------------------------------------------

B1 = b"Annual leave policy: 20 days per year, carryover allowed."
B2 = b"Personal notes on the HR portal."
B4 = b"Processing draft document."
B5 = b"Broken document bytes."
B6 = b"Ready but with no chunks."
BDUP = b"Duplicate content bytes."

VECTOR_LITERAL = "('[' || repeat('0.1,', 383) || '0.1]')::vector"


def _test_database_url() -> str | None:
    url = os.environ.get(_TEST_DATABASE_URL)
    if not url:
        return None
    if "localhost" not in url and "127.0.0.1" not in url:
        pytest.skip(f"{_TEST_DATABASE_URL} must point at a local disposable database")
    if "test" not in url.rsplit("/", 1)[-1]:
        pytest.skip("the database name in PHASE1C_TEST_DATABASE_URL must contain 'test'")
    return url


def _split_sql(script: str) -> list[str]:
    """Split a SQL script on top-level ';', respecting $$ dollar-quoted bodies and
    -- line comments (a comment may legitimately contain a ';')."""
    statements: list[str] = []
    current: list[str] = []
    in_dollar = False
    i = 0
    n = len(script)
    while i < n:
        ch = script[i]
        if not in_dollar and script.startswith("$$", i):
            in_dollar = True
            current.append("$$")
            i += 2
            continue
        if in_dollar and script.startswith("$$", i):
            in_dollar = False
            current.append("$$")
            i += 2
            continue
        if not in_dollar and script.startswith("--", i):
            while i < n and script[i] != "\n":
                i += 1
            current.append("\n")
            continue
        if not in_dollar and ch == ";":
            statement = "".join(current).strip()
            if statement:
                statements.append(statement)
            current = []
            i += 1
            continue
        current.append(ch)
        i += 1
    statement = "".join(current).strip()
    if statement:
        statements.append(statement)
    return statements


def _upgrade(rev: str) -> None:
    config = Config(str(_ALEMBIC_INI))
    command.upgrade(config, rev)


def _reset_and_bootstrap(url: str) -> None:
    async def _run() -> None:
        engine = create_async_engine(url)
        try:
            async with engine.begin() as conn:
                await conn.execute(text("DROP SCHEMA IF EXISTS public CASCADE"))
                await conn.execute(text("CREATE SCHEMA public"))
                await conn.execute(text("DROP SCHEMA IF EXISTS auth CASCADE"))
            for statement in _split_sql(_AUTH_BOOTSTRAP.read_text(encoding="utf-8")):
                async with engine.begin() as conn:
                    await conn.execute(text(statement))
        finally:
            await engine.dispose()

    asyncio.run(_run())


async def _seed_legacy(conn: AsyncConnection, upload_dir: Path) -> dict[str, uuid.UUID]:
    """Representative legacy data, inserted bypassing RLS as the superuser role.

    Returns the ids the assertions need. ``upload_dir`` receives a real file per
    storage_key (except the deliberately-missing one).
    """
    ids = {
        "user_a": uuid.uuid4(),
        "user_b": uuid.uuid4(),
    }

    # Signups: the 0004/0006 trigger provisions an org, an owner user, a default
    # workspace and an owner membership for each.
    for user, meta in (
        (ids["user_a"], {"full_name": "Alice", "org_name": "Acme"}),
        (ids["user_b"], {"full_name": "Bob"}),
    ):
        await conn.execute(
            text(
                "INSERT INTO auth.users (id, email, raw_user_meta_data) "
                "VALUES (:id, :email, :meta)"
            ),
            {"id": user, "email": f"{user.hex[:8]}@example.test", "meta": json.dumps(meta)},
        )

    org_a = (
        await conn.execute(text("SELECT id FROM organizations WHERE name = 'Acme'"))
    ).scalar_one()

    # Fold user_b into org_a (provisioning gives everyone their own org) and retire
    # org_b so the test data is a single coherent tenant.
    org_b = (
        await conn.execute(text("SELECT id FROM organizations WHERE id <> :org"), {"org": org_a})
    ).scalar_one()
    await conn.execute(
        text("UPDATE users SET org_id = :org WHERE id = :user"),
        {"org": org_a, "user": ids["user_b"]},
    )
    await conn.execute(
        text(
            "DELETE FROM workspace_members WHERE org_id = :org "
            "AND workspace_id IN (SELECT id FROM workspaces WHERE org_id = :org)"
        ),
        {"org": org_b},
    )
    await conn.execute(text("DELETE FROM workspaces WHERE org_id = :org"), {"org": org_b})
    await conn.execute(text("DELETE FROM organizations WHERE id = :org"), {"org": org_b})

    # A second workspace under org_a.
    ws_default = (
        await conn.execute(
            text("SELECT id FROM workspaces WHERE org_id = :org AND slug = 'default'"),
            {"org": org_a},
        )
    ).scalar_one()
    ws_team = uuid.uuid4()
    await conn.execute(
        text(
            "INSERT INTO workspaces (id, org_id, name, slug, owner_id) "
            "VALUES (:id, :org, 'Team', 'team', :owner)"
        ),
        {"id": ws_team, "org": org_a, "owner": ids["user_a"]},
    )
    # The provisioning trigger already created the default-workspace OWNER membership;
    # ON CONFLICT keeps this idempotent rather than assuming what the trigger did.
    for ws, user, role in (
        (ws_default, ids["user_a"], "owner"),
        (ws_default, ids["user_b"], "editor"),
        (ws_team, ids["user_a"], "owner"),
        (ws_team, ids["user_b"], "editor"),
    ):
        await conn.execute(
            text(
                "INSERT INTO workspace_members (workspace_id, user_id, org_id, role) "
                "VALUES (:ws, :user, :org, :role) "
                "ON CONFLICT (workspace_id, user_id) DO NOTHING"
            ),
            {"ws": ws, "user": user, "org": org_a, "role": role},
        )

    # Files on disk for every storage_key except the deliberately missing one.
    files = {
        "k1": B1,
        "k2": B2,
        "k4": B4,
        "k5": B5,
        "k6": B6,
        "k7": BDUP,
        "k8": BDUP,
    }
    keys = {}
    for name, data in files.items():
        key = uuid.uuid4()
        (upload_dir / str(key)).write_bytes(data)
        keys[name] = key

    # Documents. created_at ascending so the duplicate-checksum order is deterministic.
    docs = [
        # (name, ws, uploader, filename, mime, storage_key, size, status, visibility,
        #  created_at, error)
        ("doc_org_ready", ws_default, ids["user_a"], "policy.txt", "text/plain",
         keys["k1"], len(B1), "ready", "org", "2026-01-01 00:00:01+00", None),
        ("doc_personal_ready", ws_default, ids["user_b"], "notes.md", "text/markdown",
         keys["k2"], len(B2), "ready", "personal", "2026-01-01 00:00:02+00", None),
        ("doc_pending_missing", ws_default, ids["user_b"], "gone.txt", "text/plain",
         uuid.uuid4(), 10, "pending", "personal", "2026-01-01 00:00:03+00", None),
        ("doc_processing", ws_default, ids["user_b"], "draft.txt", "text/plain",
         keys["k4"], len(B4), "processing", "personal", "2026-01-01 00:00:04+00", None),
        ("doc_failed", ws_default, ids["user_a"], "broken.pdf", "application/pdf",
         keys["k5"], len(B5), "failed", "org", "2026-01-01 00:00:05+00", "legacy boom"),
        ("doc_ready_no_chunks", ws_team, ids["user_a"], "empty.txt", "text/plain",
         keys["k6"], len(B6), "ready", "org", "2026-01-01 00:00:06+00", None),
        ("doc_dup_a", ws_default, ids["user_a"], "dup1.txt", "text/plain",
         keys["k7"], len(BDUP), "ready", "org", "2026-01-01 00:00:07+00", None),
        ("doc_dup_b", ws_default, ids["user_a"], "dup2.txt", "text/plain",
         keys["k8"], len(BDUP), "ready", "org", "2026-01-01 00:00:08+00", None),
    ]
    doc_ids: dict[str, uuid.UUID] = {}
    for name, ws, uploader, filename, mime, key, size, status, vis, created, error in docs:
        doc_id = uuid.uuid4()
        doc_ids[name] = doc_id
        await conn.execute(
            text(
                "INSERT INTO documents (id, org_id, workspace_id, uploaded_by, filename, "
                "mime_type, size_bytes, storage_key, status, visibility, error_message, "
                "created_at) VALUES (:id, :org, :ws, :uploader, :filename, :mime, :size, "
                ":key, :status, :vis, :error, :created)"
            ),
            {
                "id": doc_id, "org": org_a, "ws": ws, "uploader": uploader,
                "filename": filename, "mime": mime, "size": size, "key": key,
                "status": status, "vis": vis, "error": error,
                "created": datetime.fromisoformat(created),
            },
        )

    # Chunks — only for documents that were searchable in legacy. Personal 'ready'
    # chunks exist here so the migration can be proven to delete them.
    chunk_specs = [
        (doc_ids["doc_org_ready"], 0, "Annual leave policy: 20 days per year.", 1),
        (doc_ids["doc_org_ready"], 1, "Carryover allowed for unused days.", 2),
        (doc_ids["doc_personal_ready"], 0, "Personal HR portal notes.", 1),
        (doc_ids["doc_dup_a"], 0, "Duplicate content policy.", 1),
        (doc_ids["doc_dup_b"], 0, "Duplicate content policy.", 1),
    ]
    for doc_id, index, content, page in chunk_specs:
        await conn.execute(
            text(
                "INSERT INTO document_chunks (id, org_id, document_id, chunk_index, "
                f"content, page, embedding, chunk_metadata) VALUES (:id, :org, :doc, "
                f":index, :content, :page, {VECTOR_LITERAL}, '{{}}'::jsonb)"
            ),
            {
                "id": uuid.uuid4(), "org": org_a, "doc": doc_id, "index": index,
                "content": content, "page": page,
            },
        )

    # Chats. session2 has a NULL workspace_id to exercise the backfill rule.
    session1, session2 = uuid.uuid4(), uuid.uuid4()
    await conn.execute(
        text(
            "INSERT INTO chat_sessions (id, org_id, user_id, workspace_id, title) "
            "VALUES (:id, :org, :user, :ws, 'Onboarding')"
        ),
        {"id": session1, "org": org_a, "user": ids["user_a"], "ws": ws_default},
    )
    await conn.execute(
        text(
            "INSERT INTO chat_sessions (id, org_id, user_id, workspace_id, title) "
            "VALUES (:id, :org, :user, NULL, 'Help')"
        ),
        {"id": session2, "org": org_a, "user": ids["user_b"]},
    )
    # The legacy composite FK is (session_id, org_id, user_id) -> chat_sessions, so each
    # message must carry its session's owner.
    messages = [
        (session1, ids["user_a"], "user", "What is the leave policy?", None),
        (session1, ids["user_a"], "assistant", "The policy is 20 days.", json.dumps(
            [{"document_id": str(doc_ids["doc_org_ready"]), "title": "policy.txt"}]
        )),
        (session1, ids["user_a"], "system", "system context", None),
        (session1, ids["user_a"], "tool", "tool output", None),
        (session2, ids["user_b"], "user", "Hi", None),
    ]
    for session_id, user_id, role, content, citations in messages:
        await conn.execute(
            text(
                "INSERT INTO chat_messages (id, org_id, user_id, session_id, role, "
                "content, citations) VALUES (:id, :org, :user, :session, :role, "
                ":content, :citations)"
            ),
            {
                "id": uuid.uuid4(), "org": org_a, "user": user_id,
                "session": session_id, "role": role, "content": content,
                # Legacy citations is NOT NULL; user/system/tool turns carry an empty list.
                "citations": citations or "[]",
            },
        )

    return {
        "org_a": org_a,
        "user_a": ids["user_a"],
        "user_b": ids["user_b"],
        "ws_default": ws_default,
        "ws_team": ws_team,
        **doc_ids,
        "session1": session1,
        "session2": session2,
    }


@pytest.fixture(scope="module")
def migrated_db(tmp_path_factory) -> dict:
    """Fresh 0001..0009 on a disposable database, with seeded legacy data moved."""
    url = _test_database_url()  # may raise Skipped
    if url is None:
        pytest.skip(f"{_TEST_DATABASE_URL} is not set — no disposable database")

    import app.config as config_module

    os.environ["DATABASE_URL"] = url
    for key, value in _REQUIRED_ENV.items():
        os.environ[key] = value
    config_module.get_settings.cache_clear()

    upload_dir = tmp_path_factory.mktemp("uploads")

    _reset_and_bootstrap(url)
    _upgrade("0007_document_visibility")  # the last legacy migration

    async def _seed() -> dict[str, uuid.UUID]:
        engine = create_async_engine(url)
        try:
            async with engine.begin() as conn:
                return await _seed_legacy(conn, upload_dir)
        finally:
            await engine.dispose()

    ids = asyncio.run(_seed())

    _upgrade("0008_phase1c_canonical_schema")

    report = asyncio.run(backfill_file_data(upload_dir))

    # 0009 refuses to apply while any file_data/checksum is NULL (decision 12), and the
    # backfill keeps byte-less rows FAILED with the persisted reason (decisions 8 and
    # 10) — so the operator must resolve them explicitly before finalizing. Capture what
    # the backfill did to those rows FIRST, then play the operator: delete the
    # irrecoverable ones, exactly as 0009's error directs.
    async def _capture_and_resolve_unfilled() -> dict[uuid.UUID, dict]:
        engine = create_async_engine(url)
        try:
            async with engine.begin() as conn:
                rows = (
                    await conn.execute(
                        text(
                            "SELECT id, status, error_message FROM documents "
                            "WHERE file_data IS NULL"
                        )
                    )
                ).all()
                captured = {
                    row.id: {"status": row.status, "error_message": row.error_message}
                    for row in rows
                }
                await conn.execute(text("DELETE FROM documents WHERE file_data IS NULL"))
                return captured
        finally:
            await engine.dispose()

    unfilled_before_cleanup = asyncio.run(_capture_and_resolve_unfilled())

    _upgrade("head")  # 0009

    return {
        "url": url,
        "upload_dir": upload_dir,
        "report": report,
        "unfilled_before_cleanup": unfilled_before_cleanup,
        **ids,
    }


async def _engine(url: str) -> AsyncEngine:
    return create_async_engine(url, connect_args={"statement_cache_size": 0})


async def _tenant_conn(
    url: str,
    *,
    workspace_id: uuid.UUID | None,
    user_id: uuid.UUID | None,
    svc: str | None = None,
) -> tuple[AsyncConnection, AsyncEngine]:
    """A connection acting as app_tenant with the given claims, in a transaction."""
    engine = await _engine(url)
    conn = await engine.connect()
    await conn.begin()
    claims = {}
    if workspace_id is not None:
        claims["workspace_id"] = str(workspace_id)
    if user_id is not None:
        claims["sub"] = str(user_id)
    if svc is not None:
        claims["svc"] = svc
    await conn.execute(
        text("SELECT set_config('request.jwt.claims', :claims, true)"),
        {"claims": json.dumps(claims)},
    )
    await conn.execute(text("SET LOCAL ROLE app_tenant"))
    return conn, engine


async def _visible_doc_ids(conn: AsyncConnection) -> set[uuid.UUID]:
    result = await conn.execute(text("SELECT id FROM documents"))
    return {row[0] for row in result}


async def _close(conn: AsyncConnection, engine: AsyncEngine) -> None:
    await conn.rollback()
    await conn.close()
    await engine.dispose()


# ---------------------------------------------------------------------------
# Schema shape
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_only_canonical_tables_remain(migrated_db) -> None:
    engine = await _engine(migrated_db["url"])
    try:
        async with engine.connect() as conn:
            rows = (
                await conn.execute(
                    text(
                        "SELECT table_name FROM information_schema.tables "
                        "WHERE table_schema = 'public' AND table_type = 'BASE TABLE' "
                        "AND table_name <> 'alembic_version'"
                    )
                )
            ).scalars().all()
    finally:
        await engine.dispose()
    assert set(rows) == {
        "workspaces",
        "members",
        "documents",
        "document_chunks",
        "chat_sessions",
        "chat_messages",
        "invitations",
    }


@pytest.mark.asyncio
async def test_rls_enabled_and_forced_on_all_canonical_tables(migrated_db) -> None:
    engine = await _engine(migrated_db["url"])
    try:
        async with engine.connect() as conn:
            result = await conn.execute(
                text(
                    "SELECT relname, relrowsecurity, relforcerowsecurity FROM pg_class "
                    "WHERE relname = ANY(:names)"
                ),
                {"names": list(_CANONICAL_TABLES_NAMES)},
            )
            rows = {r.relname: (r.relrowsecurity, r.relforcerowsecurity) for r in result}
    finally:
        await engine.dispose()
    assert set(rows) == set(_CANONICAL_TABLES_NAMES)
    for table, (enabled, forced) in rows.items():
        assert enabled, f"{table} has RLS disabled"
        assert forced, f"{table} does not FORCE RLS"


_CANONICAL_TABLES_NAMES = (
    "workspaces",
    "members",
    "documents",
    "document_chunks",
    "chat_sessions",
    "chat_messages",
    "invitations",
)


@pytest.mark.asyncio
async def test_hnsw_vector_index_exists(migrated_db) -> None:
    engine = await _engine(migrated_db["url"])
    try:
        async with engine.connect() as conn:
            indexdef = (
                await conn.execute(
                    text(
                        "SELECT indexdef FROM pg_indexes "
                        "WHERE indexname = 'ix_document_chunks_embedding_hnsw'"
                    )
                )
            ).scalar_one()
    finally:
        await engine.dispose()
    assert "USING hnsw" in indexdef
    assert "vector_cosine_ops" in indexdef


# ---------------------------------------------------------------------------
# Workspaces and members
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_workspaces_migrated(migrated_db) -> None:
    engine = await _engine(migrated_db["url"])
    try:
        async with engine.connect() as conn:
            rows = (
                await conn.execute(
                    text(
                        "SELECT id, name, owner_id FROM workspaces "
                        "WHERE id IN (:a, :b) ORDER BY created_at"
                    ),
                    {"a": migrated_db["ws_default"], "b": migrated_db["ws_team"]},
                )
            ).all()
    finally:
        await engine.dispose()
    assert {r.id for r in rows} == {migrated_db["ws_default"], migrated_db["ws_team"]}
    for row in rows:
        assert row.owner_id == migrated_db["user_a"]


@pytest.mark.asyncio
async def test_members_migrated(migrated_db) -> None:
    engine = await _engine(migrated_db["url"])
    try:
        async with engine.connect() as conn:
            rows = (
                await conn.execute(
                    text(
                        "SELECT workspace_id, user_id, role, status FROM members "
                        "WHERE workspace_id IN (:a, :b)"
                    ),
                    {"a": migrated_db["ws_default"], "b": migrated_db["ws_team"]},
                )
            ).all()
    finally:
        await engine.dispose()
    expected = {
        (migrated_db["ws_default"], migrated_db["user_a"], "OWNER", "ACTIVE"),
        (migrated_db["ws_default"], migrated_db["user_b"], "MEMBER", "ACTIVE"),
        (migrated_db["ws_team"], migrated_db["user_a"], "OWNER", "ACTIVE"),
        (migrated_db["ws_team"], migrated_db["user_b"], "MEMBER", "ACTIVE"),
    }
    assert {(r.workspace_id, r.user_id, r.role, r.status) for r in rows} == expected


# ---------------------------------------------------------------------------
# Documents: status mapping, bytes, checksums
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_document_status_mapping(migrated_db) -> None:
    engine = await _engine(migrated_db["url"])
    try:
        async with engine.connect() as conn:
            result = await conn.execute(
                text("SELECT id, status, approved_at, error_message FROM documents")
            )
            rows = {r.id: r for r in result}
    finally:
        await engine.dispose()

    by_name = {
        name: rows[migrated_db[name]] for name in _DOC_NAMES if migrated_db[name] in rows
    }

    assert by_name["doc_org_ready"].status == "READY"
    assert by_name["doc_org_ready"].approved_at is not None
    assert by_name["doc_org_ready"].error_message is None

    assert by_name["doc_personal_ready"].status == "PENDING"
    assert by_name["doc_personal_ready"].approved_at is None

    # Decisions 8/10: the backfill marked the byte-less rows FAILED with the persisted
    # reason; the operator then resolved them (deleted) so 0009 could enforce NOT NULL.
    assert migrated_db["unfilled_before_cleanup"][migrated_db["doc_pending_missing"]] == {
        "status": "FAILED",
        "error_message": "migration: raw bytes missing",
    }
    assert migrated_db["unfilled_before_cleanup"][migrated_db["doc_dup_b"]] == {
        "status": "FAILED",
        "error_message": "migration: duplicate checksum",
    }
    assert "doc_pending_missing" not in by_name

    assert by_name["doc_processing"].status == "PENDING"

    assert by_name["doc_failed"].status == "FAILED"
    assert by_name["doc_failed"].error_message == "legacy boom"  # preserved

    assert by_name["doc_ready_no_chunks"].status == "FAILED"
    assert by_name["doc_ready_no_chunks"].error_message == (
        "migration: ready document had no chunks"
    )

    assert by_name["doc_dup_a"].status == "READY"
    assert "doc_dup_b" not in by_name  # duplicate-losing row resolved (deleted)


_DOC_NAMES = (
    "doc_org_ready",
    "doc_personal_ready",
    "doc_pending_missing",
    "doc_processing",
    "doc_failed",
    "doc_ready_no_chunks",
    "doc_dup_a",
    "doc_dup_b",
)


@pytest.mark.asyncio
async def test_byte_backfill_and_checksum(migrated_db) -> None:
    report: BackfillReport = migrated_db["report"]
    assert report.filled == 6
    assert report.missing == 1
    assert report.duplicates == 1

    engine = await _engine(migrated_db["url"])
    try:
        async with engine.connect() as conn:
            result = await conn.execute(
                text("SELECT id, file_data, checksum, file_size FROM documents")
            )
            rows = {r.id: r for r in result}
    finally:
        await engine.dispose()

    by_name = {
        name: rows[migrated_db[name]] for name in _DOC_NAMES if migrated_db[name] in rows
    }

    assert by_name["doc_org_ready"].file_data == B1
    assert by_name["doc_org_ready"].checksum == hashlib.sha256(B1).hexdigest()
    assert by_name["doc_org_ready"].file_size == len(B1)

    assert by_name["doc_dup_a"].file_data == BDUP
    assert by_name["doc_dup_a"].checksum == hashlib.sha256(BDUP).hexdigest()
    assert "doc_dup_b" not in by_name  # resolved (deleted) by the operator

    assert by_name["doc_processing"].file_data == B4
    assert "doc_pending_missing" not in by_name  # resolved (deleted) by the operator


    # UNIQUE(workspace_id, checksum) holds: no two non-null checksums per workspace.
    engine2 = await _engine(migrated_db["url"])
    try:
        async with engine2.connect() as conn:
            dup = (
                await conn.execute(
                    text(
                        "SELECT count(*) FROM documents a JOIN documents b "
                        "ON a.workspace_id = b.workspace_id "
                        "AND a.checksum = b.checksum AND a.id <> b.id"
                    )
                )
            ).scalar()
    finally:
        await engine2.dispose()
    assert dup == 0


# ---------------------------------------------------------------------------
# Chunks: invariant, content_tsv, composite integrity
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_chunk_invariant_and_content_tsv(migrated_db) -> None:
    engine = await _engine(migrated_db["url"])
    try:
        async with engine.connect() as conn:
            counts = dict(
                (
                    await conn.execute(
                        text(
                            "SELECT document_id, count(*) FROM document_chunks "
                            "GROUP BY document_id"
                        )
                    )
                ).all()
            )
            non_ready_chunks = (
                await conn.execute(
                    text(
                        "SELECT count(*) FROM document_chunks c JOIN documents d "
                        "ON d.id = c.document_id WHERE d.status <> 'READY'"
                    )
                )
            ).scalar()
            tsv_bad = (
                await conn.execute(
                    text(
                        "SELECT count(*) FROM document_chunks WHERE content_tsv "
                        "IS DISTINCT FROM to_tsvector('english', content)"
                    )
                )
            ).scalar()
            mismatch = (
                await conn.execute(
                    text(
                        "SELECT count(*) FROM document_chunks c JOIN documents d "
                        "ON d.id = c.document_id WHERE c.workspace_id <> d.workspace_id"
                    )
                )
            ).scalar()
    finally:
        await engine.dispose()

    assert counts.get(migrated_db["doc_org_ready"]) == 2
    assert counts.get(migrated_db["doc_dup_a"]) == 1
    for name in (
        "doc_personal_ready",
        "doc_pending_missing",
        "doc_processing",
        "doc_failed",
        "doc_ready_no_chunks",
        "doc_dup_b",
    ):
        assert counts.get(migrated_db[name]) is None, f"{name} must have zero chunks"
    assert non_ready_chunks == 0
    assert tsv_bad == 0
    assert mismatch == 0


@pytest.mark.asyncio
async def test_composite_fk_and_unique_targets_exist(migrated_db) -> None:
    engine = await _engine(migrated_db["url"])
    try:
        async with engine.connect() as conn:
            constraints = (
                await conn.execute(
                    text(
                        "SELECT conname, contype FROM pg_constraint "
                        "WHERE conname IN ('fk_document_chunks_document_workspace', "
                        "'uq_documents_id_workspace', 'uq_documents_workspace_checksum')"
                    )
                )
            ).all()
    finally:
        await engine.dispose()
    assert {c.conname for c in constraints} == {
        "fk_document_chunks_document_workspace",
        "uq_documents_id_workspace",
        "uq_documents_workspace_checksum",
    }


# ---------------------------------------------------------------------------
# Chats
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_chat_migration_and_role_coercion(migrated_db) -> None:
    engine = await _engine(migrated_db["url"])
    try:
        async with engine.connect() as conn:
            sessions = (
                await conn.execute(
                    text("SELECT id, workspace_id, user_id FROM chat_sessions")
                )
            ).all()
            messages = (
                await conn.execute(
                    text("SELECT role, content, sources FROM chat_messages")
                )
            ).all()
    finally:
        await engine.dispose()

    assert {s.id for s in sessions} == {migrated_db["session1"], migrated_db["session2"]}
    for session in sessions:
        assert session.workspace_id == migrated_db["ws_default"]  # NULL backfilled
    assert {s.user_id for s in sessions} == {migrated_db["user_a"], migrated_db["user_b"]}

    roles = {m.role for m in messages}
    assert roles <= {"user", "assistant"}
    contents = {m.content for m in messages}
    assert "system context" in contents and "tool output" in contents  # content preserved
    assistant = next(m for m in messages if m.content == "The policy is 20 days.")
    assert assistant.sources == json.loads(
        json.dumps([{"document_id": str(migrated_db["doc_org_ready"]), "title": "policy.txt"}])
    )


# ---------------------------------------------------------------------------
# RLS isolation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_rls_member_sees_only_readable_documents(migrated_db) -> None:
    conn, engine = await _tenant_conn(
        migrated_db["url"],
        workspace_id=migrated_db["ws_default"],
        user_id=migrated_db["user_b"],
    )
    try:
        visible = await _visible_doc_ids(conn)
    finally:
        await _close(conn, engine)

    assert visible == {
        migrated_db["doc_org_ready"],
        migrated_db["doc_dup_a"],
        # Own PENDING/FAILED rows are visible to their uploader.
        migrated_db["doc_personal_ready"],
        migrated_db["doc_processing"],
    }


@pytest.mark.asyncio
async def test_rls_owner_sees_everything_in_workspace(migrated_db) -> None:
    conn, engine = await _tenant_conn(
        migrated_db["url"],
        workspace_id=migrated_db["ws_default"],
        user_id=migrated_db["user_a"],
    )
    try:
        visible = await _visible_doc_ids(conn)
    finally:
        await _close(conn, engine)

    assert visible == {
        migrated_db[name]
        for name in _DOC_NAMES
        if name not in ("doc_ready_no_chunks", "doc_pending_missing", "doc_dup_b")
    }


@pytest.mark.asyncio
async def test_rls_no_claims_sees_nothing(migrated_db) -> None:
    conn, engine = await _tenant_conn(
        migrated_db["url"], workspace_id=None, user_id=None
    )
    try:
        visible = await _visible_doc_ids(conn)
    finally:
        await _close(conn, engine)
    assert visible == set()


@pytest.mark.asyncio
async def test_rls_member_cannot_update_someone_elses_document(migrated_db) -> None:
    conn, engine = await _tenant_conn(
        migrated_db["url"],
        workspace_id=migrated_db["ws_default"],
        user_id=migrated_db["user_b"],
    )
    try:
        # RLS denies by filtering the row out of the UPDATE: zero rows affected, and the
        # document's name must be unchanged.
        result = await conn.execute(
            text("UPDATE documents SET filename = 'pwned.txt' WHERE id = :id"),
            {"id": migrated_db["doc_org_ready"]},
        )
        assert result.rowcount == 0
        still = (
            await conn.execute(
                text("SELECT filename FROM documents WHERE id = :id"),
                {"id": migrated_db["doc_org_ready"]},
            )
        ).scalar_one()
        assert still == "policy.txt"
    finally:
        await _close(conn, engine)


@pytest.mark.asyncio
async def test_rls_member_cannot_self_publish(migrated_db) -> None:
    """A MEMBER can upload PENDING documents but can never publish them or write
    chunks — section 5's lifecycle is enforced structurally by the RLS policies."""
    from sqlalchemy.exc import DBAPIError

    url = migrated_db["url"]
    ws = migrated_db["ws_default"]
    member = migrated_db["user_b"]
    owner = migrated_db["user_a"]
    ready_doc = migrated_db["doc_org_ready"]

    conn, engine = await _tenant_conn(url, workspace_id=ws, user_id=member)
    try:
        # The member can upload their own document as PENDING (Phase 3 member flow).
        mine_bytes = b"mine"
        mine_id = (
            await conn.execute(
                text(
                    "INSERT INTO documents (workspace_id, uploaded_by, filename, "
                    "mime_type, file_size, checksum, file_data, status) VALUES "
                    "(:ws, :user, 'mine.txt', 'text/plain', 4, :checksum, :bytes, "
                    "'PENDING') RETURNING id"
                ),
                {
                    "ws": ws,
                    "user": member,
                    "checksum": hashlib.sha256(mine_bytes).hexdigest(),
                    "bytes": mine_bytes,
                },
            )
        ).scalar_one()

        # ...but flipping it to READY is refused by the WITH CHECK (self-publish).
        with pytest.raises(DBAPIError):
            async with conn.begin_nested():
                await conn.execute(
                    text("UPDATE documents SET status = 'READY' WHERE id = :id"),
                    {"id": mine_id},
                )

        # Writing chunks under their own PENDING document is refused — a PENDING
        # document must structurally have zero chunks.
        with pytest.raises(DBAPIError):
            async with conn.begin_nested():
                await conn.execute(
                    text(
                        "INSERT INTO document_chunks (document_id, workspace_id, "
                        "chunk_index, content, embedding) VALUES (:doc, :ws, 0, "
                        f"'leak', {VECTOR_LITERAL})"
                    ),
                    {"doc": mine_id, "ws": ws},
                )

        # Writing chunks into the owner's READY document is also refused — a member
        # never writes chunks, not even for readable content.
        with pytest.raises(DBAPIError):
            async with conn.begin_nested():
                await conn.execute(
                    text(
                        "INSERT INTO document_chunks (document_id, workspace_id, "
                        "chunk_index, content, embedding) VALUES (:doc, :ws, 99, "
                        f"'pollute', {VECTOR_LITERAL})"
                    ),
                    {"doc": ready_doc, "ws": ws},
                )

        # The member can still READ the owner's READY chunks.
        count = (
            await conn.execute(
                text(
                    "SELECT count(*) FROM document_chunks WHERE document_id = :doc"
                ),
                {"doc": ready_doc},
            )
        ).scalar()
        assert count == 2

        # A member may not DELETE someone else's READY document (documents_delete is
        # narrower than SELECT: own uploads or OWNER only).
        deleted = await conn.execute(
            text("DELETE FROM documents WHERE id = :id"), {"id": ready_doc}
        )
        assert deleted.rowcount == 0
    finally:
        await _close(conn, engine)

    # Positive control: the OWNER may write chunks for a READY document.
    conn, engine = await _tenant_conn(url, workspace_id=ws, user_id=owner)
    try:
        async with conn.begin_nested():
            await conn.execute(
                text(
                    "INSERT INTO document_chunks (id, document_id, workspace_id, "
                    "chunk_index, content, embedding) VALUES (gen_random_uuid(), :doc, "
                    f":ws, 98, 'owner', {VECTOR_LITERAL})"
                ),
                {"doc": ready_doc, "ws": ws},
            )
    finally:
        await _close(conn, engine)

    # Positive control: the ingestion worker (svc claim, no sub) may write chunks for a
    # READY document — this is the Phase 3 ingestion path.
    conn, engine = await _tenant_conn(url, workspace_id=ws, user_id=None, svc="ingestion")
    try:
        async with conn.begin_nested():
            await conn.execute(
                text(
                    "INSERT INTO document_chunks (id, document_id, workspace_id, "
                    "chunk_index, content, embedding) VALUES (gen_random_uuid(), :doc, "
                    f":ws, 97, 'worker', {VECTOR_LITERAL})"
                ),
                {"doc": ready_doc, "ws": ws},
            )
    finally:
        await _close(conn, engine)


@pytest.mark.asyncio
async def test_rls_owner_can_approve_pending_document(migrated_db) -> None:
    """The approval path: an OWNER may flip a member's PENDING document to READY
    (documents_update WITH CHECK passes for the owner branch)."""
    url = migrated_db["url"]
    ws = migrated_db["ws_default"]
    owner = migrated_db["user_a"]
    pending_doc = migrated_db["doc_personal_ready"]  # PENDING, uploaded by user_b

    conn, engine = await _tenant_conn(url, workspace_id=ws, user_id=owner)
    try:
        async with conn.begin_nested():
            result = await conn.execute(
                text("UPDATE documents SET status = 'READY' WHERE id = :id"),
                {"id": pending_doc},
            )
            assert result.rowcount == 1
            approved = (
                await conn.execute(
                    text("SELECT status FROM documents WHERE id = :id"),
                    {"id": pending_doc},
                )
            ).scalar_one()
            assert approved == "READY"
    finally:
        await _close(conn, engine)


@pytest.mark.asyncio
async def test_workspace_creation_via_helper_function(migrated_db) -> None:
    """Workspace creation goes through app.create_workspace() only: a raw INSERT is
    denied by RLS (no INSERT policy exists), and the helper atomically creates the
    workspace plus its OWNER/ACTIVE membership (section 4 — creator becomes owner)."""
    from sqlalchemy.exc import DBAPIError

    url = migrated_db["url"]
    member = migrated_db["user_b"]

    conn, engine = await _tenant_conn(url, workspace_id=None, user_id=member)
    try:
        # No table-level INSERT policy: a raw INSERT is refused for anyone.
        with pytest.raises(DBAPIError):
            async with conn.begin_nested():
                await conn.execute(
                    text("INSERT INTO workspaces (name, owner_id) VALUES ('Raw', :owner)"),
                    {"owner": member},
                )

        # The helper creates a workspace the caller owns, plus the OWNER membership.
        ws_id = (
            await conn.execute(
                text("SELECT app.create_workspace('Mine')")
            )
        ).scalar_one()
        ws = (
            await conn.execute(
                text("SELECT name, owner_id FROM workspaces WHERE id = :id"),
                {"id": ws_id},
            )
        ).one()
        member_row = (
            await conn.execute(
                text(
                    "SELECT role, status FROM members "
                    "WHERE workspace_id = :ws AND user_id = :user"
                ),
                {"ws": ws_id, "user": member},
            )
        ).one()

        # The new workspace is immediately visible to its owner (workspaces_select).
        visible = (
            await conn.execute(
                text("SELECT count(*) FROM workspaces WHERE id = :id"), {"id": ws_id}
            )
        ).scalar()
        assert visible == 1
    finally:
        await _close(conn, engine)

    assert ws.name == "Mine"
    assert ws.owner_id == member
    assert (member_row.role, member_row.status) == ("OWNER", "ACTIVE")


@pytest.mark.asyncio
async def test_rls_chat_private_to_author(migrated_db) -> None:
    conn, engine = await _tenant_conn(
        migrated_db["url"],
        workspace_id=migrated_db["ws_default"],
        user_id=migrated_db["user_b"],
    )
    try:
        result = await conn.execute(
            text("SELECT id FROM chat_sessions WHERE id = :id"),
            {"id": migrated_db["session1"]},
        )
        assert result.first() is None
    finally:
        await _close(conn, engine)


# ---------------------------------------------------------------------------
# Auth provisioning
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_new_signup_provisions_workspace_and_membership(migrated_db) -> None:
    carol = uuid.uuid4()
    engine = await _engine(migrated_db["url"])
    try:
        async with engine.begin() as conn:
            await conn.execute(
                text(
                    "INSERT INTO auth.users (id, email, raw_user_meta_data) "
                    "VALUES (:id, :email, :meta)"
                ),
                {
                    "id": carol,
                    "email": "carol@example.test",
                    "meta": json.dumps({"org_name": "Carol Co"}),
                },
            )
        async with engine.connect() as conn:
            ws_id = (
                await conn.execute(
                    text(
                        "SELECT w.id FROM workspaces w WHERE w.name = 'Carol Co' "
                        "AND w.owner_id = :owner"
                    ),
                    {"owner": carol},
                )
            ).scalar_one()
            member = (
                await conn.execute(
                    text(
                        "SELECT role, status FROM members "
                        "WHERE workspace_id = :ws AND user_id = :user"
                    ),
                    {"ws": ws_id, "user": carol},
                )
            ).one()
            claim = (
                await conn.execute(
                    text(
                        "SELECT raw_app_meta_data ->> 'workspace_id' "
                        "FROM auth.users WHERE id = :id"
                    ),
                    {"id": carol},
                )
            ).scalar_one()
    finally:
        await engine.dispose()

    assert (member.role, member.status) == ("OWNER", "ACTIVE")
    assert claim == str(ws_id)
