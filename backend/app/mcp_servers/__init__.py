"""MCP servers exposing this system's capabilities over the protocol (Phase 6).

Three servers, all read-only:

- ``documents``  — vector search and reads over ingested files (Phases 3–4)
- ``database``   — the guarded SQL agent (Phase 5), unchanged and unweakened
- ``github``     — the external-system example, exactly ``search_code`` and ``read_file``

Two rules hold across all three, and both are structural rather than advisory:

**Identity is never an argument.** The caller's org and user come from
:mod:`app.mcp_servers.identity`, bound out of band from a verified JWT. No tool schema has
an ``org_id`` field, so a prompt-injected "query the other organization" has nowhere to put
the value (CLAUDE.md 4.6).

**Results are data, never instructions.** Everything returned from a document, a database
row, or a repository is fenced by :mod:`app.security.untrusted` with a per-call nonce, and
no tool result may trigger another tool call (CLAUDE.md 4.4).

Import cost is why the servers are not re-exported eagerly: the document server pulls in
sentence-transformers, and a client that only wants GitHub should not pay for torch. Import
the specific module instead.
"""

from app.mcp_servers.errors import internal_error, invalid_params, refusal, validate_args
from app.mcp_servers.identity import (
    NotAuthenticated,
    acting_as,
    current_org_and_user,
    current_principal,
)

#: Server name → import path of the module exposing `build_server()`.
SERVERS: dict[str, str] = {
    "documents": "app.mcp_servers.document_server",
    "database": "app.mcp_servers.database_server",
    "github": "app.mcp_servers.github_server",
}

__all__ = [
    "SERVERS",
    "NotAuthenticated",
    "acting_as",
    "current_org_and_user",
    "current_principal",
    "internal_error",
    "invalid_params",
    "refusal",
    "validate_args",
]
