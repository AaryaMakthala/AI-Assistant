"""Verify the process no longer imports any local ML/torch/sentence-transformers code.

Part of the memory fix's hard boundary: the render worker must never import torch,
so it can never be pulled back into the lazy-loaded modules the fast path skips.

The verify step:

1. Imports the whole ``app`` package the way the server does (with test-safe env),
   then asserts none of the forbidden top-level modules are in ``sys.modules``.
2. AST-scans ``app/`` for imports/attribute access of the forbidden names and for
   the removed symbols (``get_model``, ``CrossEncoder``, ``get_reranker``,
   ``rerank_scores``, ``apply_torch_runtime_config``).
3. Asserts the running embedding behind ``app.rag.embeddings`` is the hosted Voyage
   provider with dimension 1024 (the stored schema dimension).

Usage (from ``backend/``):

    python scripts/verify_no_torch.py
"""

from __future__ import annotations

import ast
import os
import sys
import sysconfig
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
APP_DIR = BACKEND_DIR / "app"

FORBIDDEN_MODULES = {
    "torch",
    "torchvision",
    "jt",
    "sentence_transformers",
    "transformers",
    "onnxruntime",
    "triton",
    "numba",
    "sklearn",
    "scipy",
}

REMOVED_SYMBOLS = {
    "get_model",
    "CrossEncoder",
    "get_reranker",
    "rerank_scores",
    "apply_torch_runtime_config",
}

# Two other obvious pre-torch imports: `uvicorn[standard]` pulls `uvloop`,
# `websockets`, `watchfiles`, and `httptools` — none of which is torch.  Keep the
# scan scoped to actual ML imports so the result is legible.

# Import-time env so `app` can be imported without a real database/keys.  Embedding
# provider construction is lazy, so a fake key here never makes a network call.
VERIFY_ENV = {
    "DATABASE_URL": "postgresql+asyncpg://verify:verify@localhost:5432/verify",
    "SUPABASE_URL": "https://verify.supabase.co",
    "SUPABASE_ANON_KEY": "verify-anon",
    "SUPABASE_SERVICE_ROLE_KEY": "verify-service",
    "JWT_SECRET": "verify-jwt-secret-that-is-long-enough-to-pass-validation",
    "LLM_API_KEY": "verify-llm-key",
    "LLM_MODEL": "test-model",
    "LLM_PROVIDER": "test-provider",
    "GEMINI_API_KEY": "verify-gemini-key",
}


def _walk_ast(path: Path):
    for child in sorted(path.rglob("*.py")):
        if "__pycache__" in str(child):
            continue
        yield child, ast.parse(child.read_text(encoding="utf-8"), filename=str(child))


def _forbidden_imports() -> list[str]:
    hits: list[str] = []
    for path, tree in _walk_ast(APP_DIR):
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root = alias.name.split(".")[0]
                    if root in FORBIDDEN_MODULES:
                        hits.append(f"{path}:{node.lineno}: import {alias.name}")
                    if alias.asname in REMOVED_SYMBOLS:
                        hits.append(f"{path}:{node.lineno}: import {alias.asname}")
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    root = node.module.split(".")[0]
                    if root in FORBIDDEN_MODULES:
                        hits.append(f"{path}:{node.lineno}: from {node.module} import ...")
                for alias in node.names:
                    if alias.name in REMOVED_SYMBOLS:
                        hits.append(f"{path}:{node.lineno}: {alias.name}")
            elif isinstance(node, ast.Attribute):
                if node.attr in REMOVED_SYMBOLS:
                    hits.append(f"{path}:{node.lineno}: .{node.attr}")
            elif isinstance(node, ast.Name):
                if node.id in REMOVED_SYMBOLS:
                    hits.append(f"{path}:{node.lineno}: {node.id}")
    return hits


def main() -> int:
    sys.path.insert(0, str(BACKEND_DIR))
    for key, value in VERIFY_ENV.items():
        os.environ[key] = value

    problems: list[str] = []

    ast_hits = _forbidden_imports()
    if ast_hits:
        problems.append("AST scan found local-ML references:\n" + "\n".join(ast_hits))

    # Import the app the way the server does.
    from app.main import app  # noqa: F401  (import smoke test)

    imported = {name for name in sys.modules if name.split(".")[0] in FORBIDDEN_MODULES}
    if imported:
        problems.append("forbidden modules present at import time: " + ", ".join(sorted(imported)))

    # Dimension pinned by the model, the schema, and the running provider alike.
    from app.config import get_settings
    from app.db.models import EMBEDDING_DIM

    settings = get_settings()
    if settings.embedding_provider != "voyage":
        problems.append(f"EMBEDDING_PROVIDER={settings.embedding_provider} (want voyage)")
    if settings.embedding_dim != 1024:
        problems.append(f"EMBEDDING_DIMENSION={settings.embedding_dim} (want 1024)")
    if EMBEDDING_DIM != 1024:
        problems.append(f"models.EMBEDDING_DIM={EMBEDDING_DIM} (want 1024)")

    if problems:
        print("\n".join(problems), file=sys.stderr)
        print(f"verify_no_torch: FAILED in {sysconfig.get_platform()}", file=sys.stderr)
        return 1

    print("verify_no_torch: OK — no local ML imports, hosted voyage embeddings at 1024 dims")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())