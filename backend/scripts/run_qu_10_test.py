"""Run the 10 QU test queries against the live backend (port 8001).

For each query: POST /chat/grounded, print the actual response. One retry per
HTTP/transport failure, then move on. Intent/handler info comes from the
server log (qu_test_run.log), not from this script.
"""

import time
import uuid
import httpx
import jwt

BASE_URL = "http://localhost:8001"

# Database values from the live Supabase instance (looked up via scripts/lookup_demo_ws.py).
WORKSPACE_ID = "040a479f-715b-4a7d-9555-6ca710f5f406"
USER_ID = "3fd6bacf-da33-4605-b8bf-6669eb56acd6"

# The JWT secret from .env (the dev-only fallback).
JWT_SECRET = "dev-only-jwt-secret-replace-in-production-if-using-hs256"


def make_token(user_id: str, workspace_id: str) -> str:
    """Create a HS256 JWT matching the backend's jwt_secret."""
    now = int(time.time())
    payload = {
        "sub": user_id,
        "workspace_id": workspace_id,
        "email": "test@example.com",
        "aud": "authenticated",
        "iat": now,
        "exp": now + 3600,
    }
    return jwt.encode(payload, JWT_SECRET, algorithm="HS256")


QUERIES = [
    "hi",
    "im aarya",
    "what is you name",
    "who is ceo",
    "what is this about",
    "can you make a file",
    "can you design an poster",
    "write an python code",
    "how many files are there",
    "what is the name of company",
]


def run_query(token: str, query: str, attempt: int) -> dict:
    """Run one query against /chat/grounded and return the result."""
    headers = {"Authorization": f"Bearer {token}"}
    payload = {"message": query}
    try:
        with httpx.Client(timeout=180.0) as client:
            resp = client.post(f"{BASE_URL}/chat/grounded", json=payload, headers=headers)
            if resp.status_code != 200:
                return {"query": query, "attempt": attempt, "http_error": f"HTTP {resp.status_code}: {resp.text[:300]}"}
            data = resp.json()
            return {
                "query": query,
                "attempt": attempt,
                "answer": data.get("answer", ""),
                "grounded": data.get("grounded", None),
                "insufficient_evidence": data.get("insufficient_evidence", None),
                "sources": [
                    {"filename": s.get("filename"), "page": s.get("page"), "score": s.get("score")}
                    for s in data.get("sources", [])
                ],
                "provider": data.get("provider", ""),
            }
    except Exception as e:  # noqa: BLE001
        return {"query": query, "attempt": attempt, "transport_error": str(e)}


def main() -> None:
    token = make_token(USER_ID, WORKSPACE_ID)
    print(f"Token generated for user={USER_ID[:8]}... workspace={WORKSPACE_ID[:8]}...")
    print()

    results = []
    for i, query in enumerate(QUERIES, 1):
        print(f"--- Query {i}/{len(QUERIES)}: {query!r}")
        result = run_query(token, query, attempt=1)
        if "http_error" in result or "transport_error" in result:
            print(f"  ATTEMPT 1 FAILED: {result.get('http_error') or result.get('transport_error')}")
            print("  Retrying once...")
            result = run_query(token, query, attempt=2)
        results.append(result)

        if "http_error" in result or "transport_error" in result:
            print(f"  ERROR after retry: {result.get('http_error') or result.get('transport_error')}")
        else:
            print(f"  grounded={result['grounded']} insufficient_evidence={result['insufficient_evidence']}")
            print(f"  sources={len(result['sources'])}")
            for s in result["sources"]:
                print(f"    - {s['filename']} (page={s['page']}, score={s['score']})")
            print(f"  answer={result['answer']!r}")
        print()

    # Summary line for the QU degradation count.
    degraded = sum(1 for r in results if "http_error" in r or "transport_error" in r)
    print(f"SUMMARY: {len(results)} queries, {degraded} with transport/HTTP failure after retry")


if __name__ == "__main__":
    main()