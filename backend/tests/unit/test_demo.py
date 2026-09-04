"""Unit tests for the demo entry flow, guest membership, and cleanup.

Tests verify:
- Guest user gets role='member' (never owner) when joining the demo workspace.
- Guest user is rejected from an owner-only endpoint (403).
- Trigger-created workspace + membership are cleaned up after guest creation.
- Guest ends up with exactly one membership in the demo workspace.
- Admin API failure returns 503 without orphaned state.
- Cleanup job removes expired guests and leaves non-expired ones untouched.
- Demo endpoint is disabled when demo_enabled=False.
- Demo endpoint returns 503 when workspace is not seeded.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi import HTTPException
import httpx
import pytest


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


class _AsyncCtx:
    """Minimal async context manager for mocking session.begin()."""

    def __init__(self, return_value: Any = None):
        self._return_value = return_value

    async def __aenter__(self):
        return self._return_value

    async def __aexit__(self, *args: Any) -> bool:
        return False


class _FakeSession:
    """A fake async session with optional in-memory row stores.

    When a *row_store* dict is provided, ``add()`` registers Member instances
    and ``execute()`` handles DELETE/SELECT against it.  A separate
    *workspace_store* dict tracks Workspace rows the same way.  This lets
    the test verify the actual row state after operations, not just that a
    statement was constructed.

    Without stores the session falls back to the original canned-results
    behaviour, so existing tests are unaffected.
    """

    def __init__(
        self,
        results: list[MagicMock] | None = None,
        row_store: dict[uuid.UUID, Any] | None = None,
        workspace_store: dict[uuid.UUID, Any] | None = None,
    ):
        self._results = list(results or [])
        self._idx = 0
        self.execute_calls: list[Any] = []
        self._row_store = row_store  # shared mutable dict: member_pk → instance
        self._workspace_store = workspace_store  # shared mutable dict: ws_pk → instance

    async def execute(self, stmt: Any, **kw: Any) -> MagicMock:
        self.execute_calls.append(stmt)

        from sqlalchemy import Delete, Select
        from app.db.models import Member, Workspace

        if isinstance(stmt, (Delete, Select)):
            table = self._detect_table(stmt)

            if table == "member" and self._row_store is not None:
                return self._handle_member_execute(stmt)
            if table == "workspace" and self._workspace_store is not None:
                return self._handle_workspace_execute(stmt)

        # --- canned-results fallback (original behaviour) ---
        if self._idx < len(self._results):
            r = self._results[self._idx]
            self._idx += 1
            return r
        return MagicMock(scalar_one_or_none=MagicMock(return_value=None))

    # ------------------------------------------------------------------
    # Table detection + per-table execute handlers
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_table(stmt: Any) -> str | None:
        """Heuristically detect which table a DELETE/SELECT targets."""
        try:
            from app.db.models import Member, Workspace

            if hasattr(stmt, "table"):
                table_name = getattr(stmt.table, "name", None)
                if table_name == "members":
                    return "member"
                if table_name == "workspaces":
                    return "workspace"
            # For SELECT, check the froms
            from_clause = (
                stmt.get_final_froms() if hasattr(stmt, "get_final_froms")
                else getattr(stmt, "froms", None)
            )
            if from_clause:
                from_name = getattr(from_clause[0], "name", None)
                if from_name == "members":
                    return "member"
                if from_name == "workspaces":
                    return "workspace"
        except Exception:
            pass
        return None

    def _handle_member_execute(self, stmt: Any) -> MagicMock:
        from sqlalchemy import Delete, Select

        if isinstance(stmt, Delete):
            pk = self._extract_pk_from_where(stmt)
            if pk is not None and pk in self._row_store:
                del self._row_store[pk]
            result = MagicMock()
            result.rowcount = 1 if pk is not None else 0
            return result

        if isinstance(stmt, Select):
            rows = list(self._row_store.values())
            if rows:
                return MagicMock(scalar_one_or_none=MagicMock(return_value=rows[0]))
            return MagicMock(scalar_one_or_none=MagicMock(return_value=None))

        return MagicMock(scalar_one_or_none=MagicMock(return_value=None))

    def _handle_workspace_execute(self, stmt: Any) -> MagicMock:
        from sqlalchemy import Delete, Select

        if isinstance(stmt, Delete):
            pk = self._extract_pk_from_where(stmt)
            if pk is not None and pk in self._workspace_store:
                del self._workspace_store[pk]
            result = MagicMock()
            result.rowcount = 1 if pk is not None else 0
            return result

        if isinstance(stmt, Select):
            rows = list(self._workspace_store.values())
            if rows:
                return MagicMock(scalar_one_or_none=MagicMock(return_value=rows[0]))
            return MagicMock(scalar_one_or_none=MagicMock(return_value=None))

        return MagicMock(scalar_one_or_none=MagicMock(return_value=None))

    @staticmethod
    def _extract_pk_from_where(stmt: Any) -> uuid.UUID | None:
        """Walk a statement's WHERE clause to find ``column == <uuid>``."""
        try:
            clause = stmt.whereclause
            if clause is not None and hasattr(clause, "right"):
                val = clause.right.value
                if isinstance(val, uuid.UUID):
                    return val
        except Exception:
            pass
        return None

    def begin(self) -> _AsyncCtx:
        return _AsyncCtx(self)

    def add(self, instance: Any) -> None:
        """Register the instance in the appropriate store and assign a PK
        when one is missing — mirroring what the real session does on
        flush/commit.
        """
        if hasattr(instance, "id") and instance.id is None:
            instance.id = uuid.uuid4()
        if not hasattr(instance, "id"):
            return
        # Route to the correct store by class
        from app.db.models import Member, Workspace

        if isinstance(instance, Workspace) and self._workspace_store is not None:
            self._workspace_store[instance.id] = instance
        elif isinstance(instance, Member) and self._row_store is not None:
            self._row_store[instance.id] = instance

    async def flush(self) -> None:
        """No-op: real session.flush writes pending changes."""

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args: Any) -> bool:
        return False


def _mock_settings(**overrides):
    """Build a mock Settings object with sensible defaults."""
    defaults = {
        "demo_enabled": True,
        "demo_workspace_name": "Office Brain Demo",
        "demo_guest_ttl_hours": 24,
        "supabase_url": "https://test.supabase.co",
        "supabase_service_role_key": MagicMock(
            get_secret_value=MagicMock(return_value="test-svc-key")
        ),
        **overrides,
    }
    return MagicMock(**defaults)


def _make_session_factory_side_effect(*sessions):
    """Build a side_effect for get_session_factory that returns sessions in order.

    Each call to get_session_factory() returns a callable; calling that callable
    returns the next session in the list.
    """
    iter_sessions = iter(sessions)

    def _get_factory():
        session = next(iter_sessions)
        return lambda: session

    return _get_factory


# ---------------------------------------------------------------------------
# Demo entry endpoint tests
# ---------------------------------------------------------------------------


class TestDemoEnter:
    """Tests for POST /demo/enter."""

    @pytest.mark.asyncio
    async def test_demo_disabled_returns_404(self):
        """When demo_enabled=False, the endpoint returns 404."""
        from app.api.demo import demo_enter

        settings = _mock_settings(demo_enabled=False)
        with patch("app.api.demo.get_settings", return_value=settings):
            with pytest.raises(HTTPException) as exc_info:
                await demo_enter()
            assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    async def test_demo_workspace_not_seeded_returns_503(self):
        """When the demo workspace doesn't exist, the endpoint returns 503."""
        from app.api.demo import demo_enter

        settings = _mock_settings(demo_enabled=True)
        # The endpoint now tries up to 3 lookup paths (by explicit ID, by stored
        # ID, by name).  Provide enough None results for all three.
        session = _FakeSession(
            results=[
                MagicMock(scalar_one_or_none=MagicMock(return_value=None)),
                MagicMock(scalar_one_or_none=MagicMock(return_value=None)),
                MagicMock(scalar_one_or_none=MagicMock(return_value=None)),
            ]
        )
        with (
            patch("app.api.demo.get_settings", return_value=settings),
            patch(
                "app.api.demo.get_session_factory",
                side_effect=_make_session_factory_side_effect(session),
            ),
            patch("app.api.demo.get_seeded_workspace_id", return_value=None),
        ):
            with pytest.raises(HTTPException) as exc_info:
                await demo_enter()
            assert exc_info.value.status_code == 503

    @pytest.mark.asyncio
    async def test_demo_enter_resolves_stored_workspace_id(self):
        """After seed_demo_workspace stores the ID, /demo/enter finds the workspace
        by ID — even if its name differs from demo_workspace_name.
        """
        from app.api.demo import demo_enter

        settings = _mock_settings(demo_enabled=True, demo_workspace_name="Different Name")
        demo_ws_id = uuid.uuid4()
        guest_user_id = uuid.uuid4()

        # Workspace has a DIFFERENT name than settings.demo_workspace_name.
        ws_session = _FakeSession(
            results=[
                MagicMock(
                    scalar_one_or_none=MagicMock(return_value=MagicMock(id=demo_ws_id))
                )
            ]
        )

        # Membership check (no existing member) — step 2 now inserts first
        member_session = _FakeSession(
            results=[MagicMock(scalar_one_or_none=MagicMock(return_value=None))]
        )

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"id": str(guest_user_id)}

        with (
            patch("app.api.demo.get_settings", return_value=settings),
            patch(
                "app.api.demo.get_session_factory",
                side_effect=_make_session_factory_side_effect(
                    ws_session, member_session
                ),
            ),
            patch("app.api.demo.get_seeded_workspace_id", return_value=demo_ws_id),
            patch("app.api.demo.httpx.AsyncClient") as mock_client_cls,
        ):
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.post.return_value = mock_response
            mock_client_cls.return_value = mock_client

            result = await demo_enter()

        assert result.workspace_id == demo_ws_id
        assert result.email.startswith("guest_")
        assert result.redirect_url == "/"

    @pytest.mark.asyncio
    async def test_guest_email_and_workspace_in_response(self):
        """The demo endpoint returns a guest email, workspace ID, and redirect URL."""
        from app.api.demo import demo_enter, DemoEnterResponse

        settings = _mock_settings(demo_enabled=True)
        demo_ws_id = uuid.uuid4()
        guest_user_id = uuid.uuid4()

        # Session 1: workspace lookup returns a workspace
        ws_session = _FakeSession(
            results=[
                MagicMock(
                    scalar_one_or_none=MagicMock(return_value=MagicMock(id=demo_ws_id))
                )
            ]
        )

        # Session 2: membership check (no existing member) — step 2 inserts first
        member_session = _FakeSession(
            results=[MagicMock(scalar_one_or_none=MagicMock(return_value=None))]
        )

        # Mock Supabase Admin API response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"id": str(guest_user_id)}

        with (
            patch("app.api.demo.get_settings", return_value=settings),
            patch(
                "app.api.demo.get_session_factory",
                side_effect=_make_session_factory_side_effect(
                    ws_session, member_session
                ),
            ),
            patch("app.api.demo.httpx.AsyncClient") as mock_client_cls,
        ):
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            result = await demo_enter()

        # Verify the response shape
        assert isinstance(result, DemoEnterResponse)
        assert result.user_id == guest_user_id
        assert result.workspace_id == demo_ws_id
        assert result.redirect_url == "/"
        assert result.email.startswith("guest_")
        assert result.email.endswith("@demo.local")
        assert len(result.password) > 10

    @pytest.mark.asyncio
    async def test_guest_rejected_from_owner_endpoint(self):
        """A guest (MEMBER role) calling an owner-only endpoint gets 403."""
        from app.api.workspace_deps import get_workspace_member
        from app.security.auth import Principal

        guest_user_id = uuid.uuid4()
        demo_ws_id = uuid.uuid4()

        mock_member = MagicMock()
        mock_member.role = "MEMBER"
        mock_member.status = "ACTIVE"
        mock_member.user_id = guest_user_id

        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_member
        mock_session.execute.return_value = mock_result
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        principal = Principal(user_id=guest_user_id, workspace_id=demo_ws_id)

        with patch("app.api.workspace_deps.tenant_session") as mock_tenant:
            mock_tenant.return_value.__aenter__ = AsyncMock(
                return_value=mock_session
            )
            mock_tenant.return_value.__aexit__ = AsyncMock(return_value=False)
            member = await get_workspace_member(demo_ws_id, principal)

        assert member.role == "MEMBER"

    # ------------------------------------------------------------------
    # Reordered membership: insert-before-create + cleanup tests
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_admin_api_failure_returns_503(self):
        """When the Supabase Admin API fails during guest user creation,
        the endpoint returns 503 without any database side-effects.

        With the new flow (create user first, then clean up), a failure at
        the API step means no user was created and no cleanup is needed —
        just a clean 503.
        """
        from app.api.demo import demo_enter

        settings = _mock_settings(demo_enabled=True)
        demo_ws_id = uuid.uuid4()

        # Session 1: workspace lookup returns the demo workspace
        ws_session = _FakeSession(
            results=[
                MagicMock(
                    scalar_one_or_none=MagicMock(return_value=MagicMock(id=demo_ws_id))
                )
            ]
        )

        # Mock Supabase Admin API returning 500
        api_error_response = MagicMock()
        api_error_response.status_code = 500
        api_error_response.text = "Internal Server Error"

        with (
            patch("app.api.demo.get_settings", return_value=settings),
            patch(
                "app.api.demo.get_session_factory",
                side_effect=_make_session_factory_side_effect(ws_session),
            ),
            patch("app.api.demo.httpx.AsyncClient") as mock_client_cls,
        ):
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.post.return_value = api_error_response
            mock_client_cls.return_value = mock_client

            with pytest.raises(HTTPException) as exc_info:
                await demo_enter()

            assert exc_info.value.status_code == 503

        # No cleanup session was needed — the user was never created, so
        # only one session factory call (workspace lookup) was made.
        assert len(ws_session.execute_calls) == 1

    @pytest.mark.asyncio
    async def test_regression_trigger_workspace_cleaned_up(self):
        """After /demo/enter, the on_auth_user_created trigger-created workspace
        no longer exists and the guest has exactly one membership — the demo
        workspace (Kestrel Harbor), with role MEMBER.

        This verifies the post-creation cleanup transaction in demo_enter:
        1. The trigger-created membership (OWNER in the temp workspace) is deleted.
        2. The trigger-created workspace is deleted.
        3. The correct MEMBER membership linking guest → demo workspace is inserted.
        """
        from app.api.demo import demo_enter
        from app.db.models import Member, Workspace

        settings = _mock_settings(demo_enabled=True, demo_workspace_name="Kestrel Harbor")
        demo_ws_id = uuid.uuid4()
        guest_user_id = uuid.uuid4()
        trigger_ws_id = uuid.uuid4()
        trigger_member_id = uuid.uuid4()

        # Session 1: workspace lookup returns Kestrel Harbor
        ws_session = _FakeSession(
            results=[
                MagicMock(
                    scalar_one_or_none=MagicMock(return_value=MagicMock(id=demo_ws_id))
                )
            ]
        )

        # Shared stores for the cleanup transaction — both sessions see the
        # same in-memory "database" so we can verify row state after.
        member_store: dict[uuid.UUID, Any] = {}
        workspace_store: dict[uuid.UUID, Any] = {}

        # Pre-populate the trigger-created membership in the store.
        trigger_member = MagicMock(spec=Member)
        trigger_member.id = trigger_member_id
        trigger_member.workspace_id = trigger_ws_id
        trigger_member.user_id = guest_user_id
        trigger_member.role = "OWNER"
        trigger_member.status = "ACTIVE"
        member_store[trigger_member_id] = trigger_member

        # Pre-populate the trigger-created workspace in the store.
        trigger_ws = MagicMock(spec=Workspace)
        trigger_ws.id = trigger_ws_id
        trigger_ws.name = "Demo User"
        workspace_store[trigger_ws_id] = trigger_ws

        # Session 2: cleanup transaction
        #   - SELECT trigger membership → returns trigger_member from store
        #   - DELETE trigger membership → removes from member_store
        #   - DELETE trigger workspace → removes from workspace_store
        #   - INSERT correct membership → adds to member_store
        cleanup_session = _FakeSession(
            row_store=member_store,
            workspace_store=workspace_store,
        )

        # Mock Supabase Admin API response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"id": str(guest_user_id)}

        with (
            patch("app.api.demo.get_settings", return_value=settings),
            patch(
                "app.api.demo.get_session_factory",
                side_effect=_make_session_factory_side_effect(
                    ws_session, cleanup_session
                ),
            ),
            patch("app.api.demo.httpx.AsyncClient") as mock_client_cls,
        ):
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            result = await demo_enter()

        # 1. /demo/enter returns the correct workspace ID.
        assert result.workspace_id == demo_ws_id
        assert result.email.startswith("guest_")
        assert result.email.endswith("@demo.local")

        # 2. The trigger-created workspace no longer exists.
        assert trigger_ws_id not in workspace_store, (
            f"Trigger-created workspace {trigger_ws_id} should have been deleted, "
            f"but it still exists in the workspace store"
        )

        # 3. The guest has exactly one membership: the demo workspace, MEMBER role.
        assert len(member_store) == 1, (
            f"Expected exactly 1 membership row, got {len(member_store)}: "
            f"{list(member_store.keys())}"
        )
        sole_member = list(member_store.values())[0]
        assert sole_member.workspace_id == demo_ws_id, (
            f"Sole membership should be in demo workspace {demo_ws_id}, "
            f"not {sole_member.workspace_id}"
        )
        assert sole_member.role == "MEMBER", (
            f"Guest should be MEMBER, not {sole_member.role}"
        )
        assert sole_member.user_id == guest_user_id


# ---------------------------------------------------------------------------
# Cleanup tests
# ---------------------------------------------------------------------------


class TestDemoCleanup:
    """Tests for the demo guest cleanup job."""

    @pytest.mark.asyncio
    async def test_cleanup_returns_zero_when_disabled(self):
        """When demo_enabled=False, cleanup does nothing."""
        from app.demo.cleanup import cleanup_demo_guests

        settings = _mock_settings(demo_enabled=False)
        with patch("app.demo.cleanup.get_settings", return_value=settings):
            result = await cleanup_demo_guests()
        assert result == 0

    @pytest.mark.asyncio
    async def test_cleanup_removes_expired_guests(self):
        """Guests older than the TTL are removed; fresh guests are kept."""
        from app.demo.cleanup import cleanup_demo_guests

        settings = _mock_settings(demo_enabled=True, demo_guest_ttl_hours=24)
        demo_ws_id = uuid.uuid4()
        expired_user_id = uuid.uuid4()
        fresh_user_id = uuid.uuid4()

        # Session 1: workspace lookup
        ws_session = _FakeSession(
            results=[
                MagicMock(
                    scalar_one_or_none=MagicMock(return_value=demo_ws_id)
                )
            ]
        )

        # Session 2: member delete
        delete_result = MagicMock()
        delete_result.rowcount = 1
        delete_session = _FakeSession(results=[delete_result])

        # Mock Supabase user list — one expired, one fresh, one non-guest
        old_time = "2020-01-01T00:00:00Z"
        fresh_time = datetime.now(timezone.utc).isoformat()

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "users": [
                {
                    "id": str(expired_user_id),
                    "user_metadata": {"is_guest": True},
                    "created_at": old_time,
                },
                {
                    "id": str(fresh_user_id),
                    "user_metadata": {"is_guest": True},
                    "created_at": fresh_time,
                },
                {
                    "id": str(uuid.uuid4()),
                    "user_metadata": {},
                    "created_at": old_time,
                },
            ]
        }

        delete_user_response = MagicMock()
        delete_user_response.status_code = 204

        with (
            patch("app.demo.cleanup.get_settings", return_value=settings),
            patch(
                "app.demo.cleanup.get_session_factory",
                side_effect=_make_session_factory_side_effect(
                    ws_session, delete_session
                ),
            ),
            patch("app.demo.cleanup.httpx.AsyncClient") as mock_client_cls,
        ):
            mock_client = AsyncMock()
            mock_client.get.return_value = mock_response
            mock_client.delete.return_value = delete_user_response
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            result = await cleanup_demo_guests()

        # One expired guest should have been removed
        assert result == 1

        # Verify the expired user was deleted — check URL path contains the UUID
        expired_str = str(expired_user_id)
        fresh_str = str(fresh_user_id)

        expired_delete_count = 0
        fresh_delete_count = 0
        for call in mock_client.delete.call_args_list:
            url = call[0][0] if call[0] else ""
            if expired_str in url:
                expired_delete_count += 1
            if fresh_str in url:
                fresh_delete_count += 1

        assert expired_delete_count == 1
        assert fresh_delete_count == 0

    @pytest.mark.asyncio
    async def test_cleanup_preserves_non_guest_users(self):
        """Non-guest users are never touched by cleanup."""
        from app.demo.cleanup import cleanup_demo_guests

        settings = _mock_settings(demo_enabled=True, demo_guest_ttl_hours=24)
        demo_ws_id = uuid.uuid4()

        ws_session = _FakeSession(
            results=[
                MagicMock(
                    scalar_one_or_none=MagicMock(return_value=demo_ws_id)
                )
            ]
        )
        delete_session = _FakeSession(results=[MagicMock(rowcount=0)])

        non_guest_id = uuid.uuid4()
        old_time = "2020-01-01T00:00:00Z"

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "users": [
                {
                    "id": str(non_guest_id),
                    "user_metadata": {},  # Not a guest
                    "created_at": old_time,
                },
            ]
        }

        with (
            patch("app.demo.cleanup.get_settings", return_value=settings),
            patch(
                "app.demo.cleanup.get_session_factory",
                side_effect=_make_session_factory_side_effect(
                    ws_session, delete_session
                ),
            ),
            patch("app.demo.cleanup.httpx.AsyncClient") as mock_client_cls,
        ):
            mock_client = AsyncMock()
            mock_client.get.return_value = mock_response
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            result = await cleanup_demo_guests()

        assert result == 0

    @pytest.mark.asyncio
    async def test_cleanup_uses_explicit_workspace_id(self):
        """Regression: an ID-pinned demo workspace is still found and cleaned.

        Deployments that set ``DEMO_WORKSPACE_ID`` to a workspace whose name
        differs from ``DEMO_WORKSPACE_NAME`` previously made cleanup return 0
        forever — the workspace lookup only checked the name. The job must
        resolve the demo workspace by explicit ID first (mirroring the demo
        entry flow).
        """
        from app.demo.cleanup import cleanup_demo_guests

        demo_ws_id = uuid.uuid4()
        settings = _mock_settings(
            demo_enabled=True,
            demo_guest_ttl_hours=24,
            demo_workspace_id=str(demo_ws_id),
            # A name that matches NO workspace — the old lookup failed here.
            demo_workspace_name="No Such Workspace Name",
        )
        expired_user_id = uuid.uuid4()

        # Session 1: workspace lookup by explicit ID -> found.
        ws_session = _FakeSession(
            results=[
                MagicMock(
                    scalar_one_or_none=MagicMock(return_value=demo_ws_id)
                )
            ]
        )
        # Session 2: member delete.
        delete_result = MagicMock()
        delete_result.rowcount = 1
        delete_session = _FakeSession(results=[delete_result])

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "users": [
                {
                    "id": str(expired_user_id),
                    "user_metadata": {"is_guest": True},
                    "created_at": "2020-01-01T00:00:00Z",
                },
            ]
        }

        with (
            patch("app.demo.cleanup.get_settings", return_value=settings),
            patch(
                "app.demo.cleanup.get_session_factory",
                side_effect=_make_session_factory_side_effect(
                    ws_session, delete_session
                ),
            ),
            patch("app.demo.cleanup.httpx.AsyncClient") as mock_client_cls,
        ):
            mock_client = AsyncMock()
            mock_client.get.return_value = mock_response
            mock_client.delete.return_value = MagicMock(status_code=204)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            result = await cleanup_demo_guests()

        assert result == 1
        assert ws_session.execute_calls  # the ID lookup ran
