"""Run one MCP server over stdio: `python -m app.mcp_servers <name>`.

For local development and for MCP client discovery — the Phase 6 verification is that a
client can connect, list the tools, and call them. In Phase 7 the LangGraph supervisor calls
the same tool functions in-process instead, so this entrypoint is a development and
inspection surface, not the production path.

**Why this refuses to run in production.** A stdio server has no HTTP request and therefore
no JWT, so there is nothing to derive identity from — yet the document and database tools
need an org to scope to. The only way to supply one here is `--org-id` on the command line,
which is an *asserted* identity rather than a *verified* one. That is acceptable for a
developer inspecting their own tenant and unacceptable anywhere real, so:

- `--org-id` is required for the `documents` and `database` servers, and
- the whole entrypoint exits if `ENVIRONMENT=production`.

The `github` server needs no identity: its scope comes from the token's own permissions, not
from the caller's organization.
"""

from __future__ import annotations

import argparse
import importlib
import sys
import uuid

from loguru import logger

from app.config import get_settings
from app.logging_config import configure_logging
from app.mcp_servers import SERVERS
from app.mcp_servers.identity import acting_as
from app.security.auth import Principal

#: Servers whose tools are org-scoped and therefore require an identity to act as.
_NEEDS_IDENTITY = frozenset({"documents", "database"})


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m app.mcp_servers",
        description="Run one MCP server over stdio (development only).",
    )
    parser.add_argument("server", choices=sorted(SERVERS), help="Which server to run.")
    parser.add_argument(
        "--org-id",
        type=uuid.UUID,
        default=None,
        help="Organization to act as. Required for the documents and database servers.",
    )
    parser.add_argument(
        "--user-id",
        type=uuid.UUID,
        default=None,
        help="User to attribute calls to in the audit log. Defaults to --org-id.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    settings = get_settings()

    # stdout is the JSON-RPC channel: a stray log line there corrupts the protocol stream.
    configure_logging(level="DEBUG" if settings.debug else "INFO", serialize=False)

    if settings.is_production:
        logger.error(
            "The stdio MCP entrypoint is disabled in production: it takes identity from the "
            "command line rather than from a verified JWT."
        )
        return 2

    needs_identity = args.server in _NEEDS_IDENTITY
    if needs_identity and args.org_id is None:
        logger.error("--org-id is required for the {server} server.", server=args.server)
        return 2

    module = importlib.import_module(SERVERS[args.server])
    server = module.build_server()

    logger.info("Serving MCP server {name} over stdio", name=args.server)

    if not needs_identity:
        server.run("stdio")
        return 0

    principal = Principal(
        user_id=args.user_id or args.org_id,
        org_id=args.org_id,
        role="member",
    )
    # The context wraps the whole server lifetime, so every tool call this process serves
    # acts as this one asserted principal — which is exactly why it is dev-only.
    with acting_as(principal):
        server.run("stdio")
    return 0


if __name__ == "__main__":
    sys.exit(main())
