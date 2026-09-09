"""One-off read-only lookup: demo workspace owner + active members."""

import asyncio

from sqlalchemy import select

from app.db.models import Member, Workspace
from app.db.session import get_session_factory

WS_ID = "040a479f-715b-4a7d-9555-6ca710f5f406"


async def main() -> None:
    async with get_session_factory()() as s:
        ws = (
            await s.execute(select(Workspace).where(Workspace.id == WS_ID))
        ).scalar_one_or_none()
        if ws is None:
            print(f"Workspace {WS_ID} NOT FOUND")
            return
        print(f"workspace: {ws.id} name={ws.name!r} owner_id={ws.owner_id}")

        rows = (
            await s.execute(
                select(Member.user_id, Member.role, Member.status).where(
                    Member.workspace_id == WS_ID
                )
            )
        ).all()
        print(f"members: {len(rows)}")
        for r in rows:
            print(f"  user_id={r.user_id} role={r.role} status={r.status}")


if __name__ == "__main__":
    asyncio.run(main())