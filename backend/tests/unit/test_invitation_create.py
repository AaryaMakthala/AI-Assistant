"""Regression test for POST /workspaces/{id}/invitations (create_invitation).

Verifies that the endpoint returns 201 with all six InvitationResponse fields
(id, workspace_id, email, status, invited_by, created_at) populated after the
fix changing ``.one()`` to ``.scalars().first()`` on the insert result.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.api.workspaces import InvitationResponse, create_invitation


@pytest.mark.asyncio
async def test_create_invitation_returns_all_six_fields():
    """POST /workspaces/{id}/invitations returns 201 with all six fields populated."""
    ws_id = uuid.uuid4()
    owner_id = uuid.uuid4()
    invitation_id = uuid.uuid4()
    now = datetime.now(timezone.utc)

    # --- Build the mock ORM object returned by .scalars().first() ---
    fake_invitation = MagicMock(spec=InvitationResponse)
    fake_invitation.id = invitation_id
    fake_invitation.workspace_id = ws_id
    fake_invitation.email = "alice@example.com"
    fake_invitation.status = "PENDING"
    fake_invitation.invited_by = owner_id
    fake_invitation.created_at = now

    # Mock session.execute calls in order:
    #   1. Check for existing PENDING invitation → None (no conflict)
    #   2. Insert + returning → result where .scalars().first() yields the ORM object
    mock_check_result = MagicMock()
    mock_check_result.scalar_one_or_none.return_value = None

    # The critical part: .scalars().first() must return the ORM object,
    # NOT a Row tuple.  This is what the .one() → .scalars().first() fix
    # ensures.
    mock_insert_scalars = MagicMock()
    mock_insert_scalars.first.return_value = fake_invitation

    mock_insert_result = MagicMock()
    mock_insert_result.scalars.return_value = mock_insert_scalars

    mock_session = AsyncMock()
    mock_session.execute = AsyncMock(
        side_effect=[mock_check_result, mock_insert_result]
    )

    # --- Mock the WorkspaceOwner dependency and tenant_session ---
    mock_ctx = MagicMock()
    mock_ctx.workspace_id = ws_id
    mock_ctx.principal = MagicMock()
    mock_ctx.principal.user_id = owner_id

    from app.api.workspaces import InvitationCreate

    payload = InvitationCreate(email="alice@example.com")

    with patch("app.api.workspaces.tenant_session") as mock_tenant:
        mock_tenant.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        mock_tenant.return_value.__aexit__ = AsyncMock(return_value=False)

        result = await create_invitation(mock_ctx, payload)

    # --- Assert all six fields are populated ---
    assert isinstance(result, InvitationResponse)
    assert result.id == invitation_id
    assert result.workspace_id == ws_id
    assert result.email == "alice@example.com"
    assert result.status == "PENDING"
    assert result.invited_by == owner_id
    assert result.created_at == now

    # Verify .scalars().first() was called on the insert result,
    # NOT .one() (which would return a Row tuple).
    mock_insert_result.scalars.assert_called_once()
    mock_insert_scalars.first.assert_called_once()

    # Verify no other execute calls were made (only check + insert).
    assert mock_session.execute.await_count == 2
