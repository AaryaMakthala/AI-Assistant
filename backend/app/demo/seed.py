"""Idempotent seed script for the demo workspace.

Creates a demo owner account, workspace, and pre-loaded sample documents.
Running twice will not duplicate anything — the owner email is checked before
creating, and document checksums prevent re-ingestion.

This script uses the Supabase Admin API for user creation and a BYPASSRLS
database session for workspace/member/document writes.  It is designed to be
called once at app startup (see ``app.main.lifespan``).
"""

from __future__ import annotations

import hashlib
import uuid

import httpx
import pymupdf
from loguru import logger
from sqlalchemy import select

from app.config import get_settings
from app.db.models import Document, DocumentChunk, Member, Workspace
from app.db.session import get_session_factory
from app.ingestion.pipeline import prepare_document
from app.security.rls import set_tenant_claims

DEMO_OWNER_EMAIL = "demo-owner@officebrain.app"
DEMO_OWNER_PASSWORD = "demo-owner-change-me-in-prod"  # noqa: S105 — intentional demo seed credential

# Module-level storage for the resolved demo workspace ID.
# Set by seed_demo_workspace() at startup so the /demo/enter endpoint can
# read it directly instead of re-deriving "is it seeded" by name.
_seeded_workspace_id: uuid.UUID | None = None

# ---------------------------------------------------------------------------
# Sample documents — fictional company content
# ---------------------------------------------------------------------------

_LEAVE_POLICY = """\
Acme Innovations — Annual Leave Policy

1. Overview
Acme Innovations provides all full-time employees with a generous annual leave
entitlement to support work-life balance and well-being.

2. Entitlement
- Employees with 0–2 years of service: 15 working days per year.
- Employees with 3–5 years of service: 20 working days per year.
- Employees with 6+ years of service: 25 working days per year.

3. Carry-Over
Up to 5 unused days may be carried over into the next calendar year, provided
they are used before the end of Q1 (31 March).  Unused carried-over days
expire automatically.

4. Public Holidays
The company observes 8 public holidays per year.  These do not count against
the annual leave entitlement.

5. Sick Leave
Sick leave is separate from annual leave.  Employees may take up to 10 paid
sick days per year with a medical certificate.  Extended illness beyond 10
days is handled on a case-by-case basis.

6. Requesting Leave
Leave requests must be submitted via the HR portal at least 2 weeks in advance
for leave of 1–3 days, and 4 weeks for leave of 4+ days.  Manager approval is
required.  The team lead must ensure adequate coverage before approving.

7. Unpaid Leave
Unpaid leave may be granted at the discretion of the manager and HR for
personal circumstances not covered by other leave types.  Requests should be
made at least 4 weeks in advance.

8. Termination
Upon termination, accrued but unused leave is paid out in the final paycheck
per applicable labor law.

Policy effective: 1 January 2024
Last reviewed: 15 June 2024
"""

_ONBOARDING_GUIDE = """\
Acme Innovations — New Employee Onboarding Guide

Welcome to Acme Innovations! This guide walks you through your first two weeks.

Day 1 — Your First Day
- Arrive at reception by 9:00 AM.  You will be greeted by your buddy.
- Collect your laptop, badge, and welcome kit from IT (Building A, Floor 2).
- Complete the HR paperwork: tax forms, direct deposit, emergency contact.
- Set up your workstation and install required software (see IT Checklist below).
- Lunch with your team at 12:30 PM (company-sponsored).

Week 1 — Getting Oriented
- Monday: Company overview presentation by the CEO (10:00 AM, All-Hands Room).
- Tuesday: Department introduction with your manager — meet the team, review
  goals, and discuss your first project.
- Wednesday: Security and compliance training (mandatory, 2-hour online module).
- Thursday: Product deep-dive with the product team.
- Friday: 1-on-1 with your manager to review your first-week experience.

Week 2 — Starting Work
- Begin your first project assignment.
- Complete remaining onboarding modules (code of conduct, data privacy).
- Set up development environment (see Engineering Setup Guide).
- Schedule skip-level meeting with your director.

IT Checklist
- [ ] Laptop received and configured
- [ ] Email and calendar set up
- [ ] Slack workspace joined (#introductions channel)
- [ ] VPN configured
- [ ] GitHub/GitLab access granted
- [ ] Jira/Linear board access

Benefits Enrollment
Complete benefits enrollment within 30 days of start date via the HR portal.
Health, dental, and vision insurance options are available.

Contact
- IT Help Desk: it-help@acme.example.com or ext. 4400
- HR General: hr@acme.example.com or ext. 4100
- Your buddy: [assigned on Day 1]

Policy effective: 1 January 2024
"""

_EXPENSE_POLICY = """\
Acme Innovations — Travel & Expense Policy

1. Purpose
This policy outlines the rules for business travel and expense reimbursement.
All employees must follow these guidelines to ensure timely reimbursement.

2. Pre-Approval
Travel exceeding $500 in total estimated cost requires pre-approval from your
department head.  Submit the Travel Request Form at least 2 weeks before the
planned trip.

3. Booking Guidelines
- Flights: Book economy class for domestic flights.  Premium economy is
  permitted for international flights over 6 hours.
- Hotels: Use the company's preferred hotel partners when available.  Maximum
  nightly rate: $200 (domestic), $300 (international).
- Ground transportation: Use ride-share or public transit when practical.
  Rental cars require pre-approval.

4. Per Diem
- Domestic travel: $75 per day for meals and incidentals.
- International travel: $100 per day for meals and incidentals.
- Per diem covers breakfast, lunch, dinner, and tips.  Receipts are not
  required for per diem claims under the daily limit.

5. Expense Submission
- Submit expenses within 30 days of the trip via the Expense Portal.
- Attach receipts for all individual expenses over $25.
- Include business purpose, attendees, and project code for each expense.

6. Reimbursable Expenses
- Airfare and ground transportation
- Hotel accommodations
- Meals (within per diem or with receipts)
- Conference and event fees
- Business-related phone and internet charges

7. Non-Reimbursable Expenses
- Mini-bar charges
- In-room entertainment
- Personal travel extensions
- Alcohol (except when part of a client entertainment event with approval)
- First-class airfare (without VP approval)

8. Client Entertainment
Client entertainment expenses (meals, events) require a separate Client
Entertainment Form with the names and titles of all attendees.

Policy effective: 1 January 2024
Last reviewed: 15 June 2024
"""


def _make_text_pdf(text: str) -> bytes:
    """Create a simple PDF from plain text using PyMuPDF."""
    doc = pymupdf.open()
    page = doc.new_page(width=612, height=792)  # US Letter

    # Write text in a simple layout
    rect = pymupdf.Rect(54, 54, 558, 738)  # 0.75 inch margins
    page.insert_textbox(
        rect,
        text,
        fontsize=10,
        fontname="helv",
        color=(0.1, 0.1, 0.1),
    )

    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


async def seed_demo_workspace() -> uuid.UUID | None:
    """Create the demo workspace if it does not already exist.

    Returns the workspace ID, or None if seeding was skipped (already exists).
    Idempotent: safe to call multiple times.

    When ``DEMO_WORKSPACE_ID`` is set in the environment, the seed flow looks
    up that workspace directly (must exist — raises on startup if not found)
    and skips owner creation / workspace provisioning.  Sample documents are
    still attached (idempotently) if not already present.
    """
    settings = get_settings()

    # --- Branch: explicit demo_workspace_id ---
    if settings.demo_workspace_id:
        try:
            target_ws_id = uuid.UUID(settings.demo_workspace_id)
        except ValueError as exc:
            raise ValueError(
                f"DEMO_WORKSPACE_ID is not a valid UUID: {settings.demo_workspace_id!r}"
            ) from exc

        async with get_session_factory()() as session:
            existing_ws = (
                await session.execute(
                    select(Workspace).where(Workspace.id == target_ws_id)
                )
            ).scalar_one_or_none()

            if existing_ws is None:
                raise RuntimeError(
                    f"DEMO_WORKSPACE_ID is set ({target_ws_id}) but no workspace with "
                    "that ID exists"
                )

            ws_id = existing_ws.id
            owner_id = existing_ws.owner_id

            # Check if documents already present (fully seeded).
            doc_count = (
                await session.execute(
                    select(Document).where(Document.workspace_id == ws_id)
                )
            ).scalars().all()

            if doc_count:
                logger.info(
                    "Demo workspace already seeded (explicit ID): {ws} with {n} documents",
                    ws=ws_id,
                    n=len(doc_count),
                )
                return ws_id

        # Documents not yet present — ingest sample docs using the workspace owner.
        logger.info(
            "Seeding sample documents into existing workspace {ws}", ws=ws_id,
        )
    else:
        # --- Branch: fall back to create-or-reuse-by-name ---

        supabase_url = str(settings.supabase_url).rstrip("/")
        service_key = settings.supabase_service_role_key.get_secret_value()

        # Check if the demo workspace already exists.
        async with get_session_factory()() as session:
            existing_ws = (
                await session.execute(
                    select(Workspace).where(Workspace.name == settings.demo_workspace_name)
                )
            ).scalar_one_or_none()

            if existing_ws is not None:
                # Check if it has documents (fully seeded).
                doc_count = (
                    await session.execute(
                        select(Document).where(Document.workspace_id == existing_ws.id)
                    )
                ).scalars().all()

                if doc_count:
                    logger.info(
                        "Demo workspace already seeded: {ws} with {n} documents",
                        ws=existing_ws.id,
                        n=len(doc_count),
                    )
                    return existing_ws.id

        # Step 1: Create or find the demo owner user.
        owner_id = await _ensure_demo_owner(supabase_url, service_key)
        if owner_id is None:
            logger.error("Could not create or find demo owner user")
            return None

        # Step 2: Create the workspace using the SQL function (atomic with membership).
        async with get_session_factory()() as session:
            async with session.begin():
                # Check again in case of concurrent seeding.
                ws = (
                    await session.execute(
                        select(Workspace).where(Workspace.name == settings.demo_workspace_name)
                    )
                ).scalar_one_or_none()

                if ws is not None:
                    ws_id = ws.id
                else:
                    from sqlalchemy import func

                    # Set RLS claims so that app.current_user_id() resolves the demo
                    # owner inside the SECURITY DEFINER create_workspace function.
                    # Must be set before the call — it reads the sub claim from
                    # request.jwt.claims (set via set_config(... true), transaction-scoped).
                    await set_tenant_claims(
                        session,
                        workspace_id=uuid.uuid4(),  # not read by create_workspace
                        user_id=owner_id,
                    )

                    ws_id = (
                        await session.execute(
                            select(func.app.create_workspace(settings.demo_workspace_name))
                        )
                    ).scalar_one()

            logger.info("Demo workspace created: {ws}", ws=ws_id)

        # Step 3: Ensure the owner is a member with OWNER role.
        async with get_session_factory()() as session:
            async with session.begin():
                member = (
                    await session.execute(
                        select(Member).where(
                            Member.workspace_id == ws_id,
                            Member.user_id == owner_id,
                        )
                    )
                ).scalar_one_or_none()

                if member is None:
                    # The create_workspace function should have created this, but
                    # in case it didn't (e.g. the owner was created separately),
                    # ensure the membership exists.
                    m = Member(
                        workspace_id=ws_id,
                        user_id=owner_id,
                        role="OWNER",
                        status="ACTIVE",
                    )
                    session.add(m)
                elif member.role != "OWNER":
                    from sqlalchemy import update

                    await session.execute(
                        update(Member)
                        .where(Member.id == member.id)
                        .values(role="OWNER")
                    )

    # Ingest sample documents (both branches converge here).
    sample_docs = [
        ("Annual Leave Policy", "leave_policy.pdf", _LEAVE_POLICY),
        ("New Employee Onboarding Guide", "onboarding_guide.pdf", _ONBOARDING_GUIDE),
        ("Travel & Expense Policy", "expense_policy.pdf", _EXPENSE_POLICY),
    ]

    for doc_name, filename, content in sample_docs:
        await _ingest_sample_doc(ws_id, owner_id, doc_name, filename, content)

    logger.info(
        "Demo workspace seeded successfully: {ws} with {n} documents",
        ws=ws_id,
        n=len(sample_docs),
    )
    global _seeded_workspace_id
    _seeded_workspace_id = ws_id
    return ws_id


async def _ensure_demo_owner(supabase_url: str, service_key: str) -> uuid.UUID | None:
    """Find or create the demo owner user in Supabase Auth.

    Returns the user ID, or None on failure.
    """
    # Check if the owner already exists.
    async with httpx.AsyncClient(timeout=15.0) as client:
        response = await client.get(
            f"{supabase_url}/auth/v1/admin/users",
            headers={
                "apikey": service_key,
                "Authorization": f"Bearer {service_key}",
            },
            params={"email": DEMO_OWNER_EMAIL},
        )

        if response.status_code == 200:
            users = response.json().get("users", [])
            if users:
                owner_id = uuid.UUID(users[0]["id"])
                logger.info("Demo owner already exists: {id}", id=owner_id)
                return owner_id

        # Create the owner user.
        response = await client.post(
            f"{supabase_url}/auth/v1/admin/users",
            headers={
                "apikey": service_key,
                "Authorization": f"Bearer {service_key}",
            },
            json={
                "email": DEMO_OWNER_EMAIL,
                "password": DEMO_OWNER_PASSWORD,
                "email_confirm": True,
                "user_metadata": {
                    "full_name": "Demo Admin",
                },
            },
        )

        if response.status_code >= 400:
            logger.error(
                "Failed to create demo owner: {status} {body}",
                status=response.status_code,
                body=response.text[:300],
            )
            return None

        owner_id = uuid.UUID(response.json()["id"])
        logger.info("Demo owner created: {id}", id=owner_id)
        return owner_id


async def _ingest_sample_doc(
    ws_id: uuid.UUID,
    owner_id: uuid.UUID,
    doc_name: str,
    filename: str,
    content: str,
) -> None:
    """Create and ingest a sample document into the demo workspace."""
    pdf_bytes = _make_text_pdf(content)
    checksum = hashlib.sha256(pdf_bytes).hexdigest()

    async with get_session_factory()() as session:
        # Check for duplicate by checksum.
        existing = (
            await session.execute(
                select(Document.id).where(
                    Document.workspace_id == ws_id,
                    Document.checksum == checksum,
                )
            )
        ).scalar_one_or_none()

        if existing is not None:
            logger.debug("Sample doc {name} already exists, skipping", name=filename)
            return

    # Run the ingestion pipeline (extract → chunk → embed).
    try:
        prepared = await _run_ingestion(pdf_bytes, "application/pdf", filename)
    except Exception as exc:
        logger.error(
            "Failed to ingest sample doc {name}: {error}",
            name=filename,
            error=str(exc)[:200],
        )
        # Store as FAILED so it's visible in the admin UI.
        async with get_session_factory()() as session:
            async with session.begin():
                doc = Document(
                    workspace_id=ws_id,
                    uploaded_by=owner_id,
                    filename=filename,
                    mime_type="application/pdf",
                    file_size=len(pdf_bytes),
                    checksum=checksum,
                    file_data=pdf_bytes,
                    status="FAILED",
                    error_message=f"Seed ingestion failed: {str(exc)[:500]}",
                )
                session.add(doc)
        return

    # Persist document + chunks in one transaction.
    async with get_session_factory()() as session:
        async with session.begin():
            doc = Document(
                workspace_id=ws_id,
                uploaded_by=owner_id,
                filename=filename,
                mime_type="application/pdf",
                file_size=len(pdf_bytes),
                checksum=checksum,
                file_data=pdf_bytes,
                status="READY",
            )
            session.add(doc)
            await session.flush()  # Get the document ID.

            for chunk in prepared.chunks:
                db_chunk = DocumentChunk(
                    document_id=doc.id,
                    workspace_id=ws_id,
                    chunk_index=chunk.chunk_index,
                    content=chunk.content,
                    embedding=chunk.embedding,
                    page_number=chunk.page_number,
                    section_title=chunk.section_title,
                    chunk_metadata=chunk.chunk_metadata,
                )
                session.add(db_chunk)

    logger.info(
        "Sample doc ingested: {name} ({chunks} chunks)",
        name=filename,
        chunks=len(prepared.chunks),
    )


async def _run_ingestion(pdf_bytes: bytes, mime_type: str, filename: str):
    """Run the ingestion pipeline in a thread (CPU-bound embedding work)."""
    import asyncio

    return await asyncio.to_thread(
        prepare_document,
        pdf_bytes,
        mime_type=mime_type,
        filename=filename,
    )


def get_seeded_workspace_id() -> uuid.UUID | None:
    """Return the workspace ID resolved by seed_demo_workspace(), or None if
    seeding hasn't run or hasn't resolved a workspace yet.
    """
    return _seeded_workspace_id


__all__ = ["seed_demo_workspace", "get_seeded_workspace_id"]
