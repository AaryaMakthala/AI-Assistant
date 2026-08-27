"""Run all 15 Phase B-2 acceptance queries against the live backend."""

import json
import time
import uuid
import jwt
import httpx

BASE_URL = "http://localhost:8000"

# Database values from the live Supabase instance.
WORKSPACE_ID = "abb1bb8b-84e7-4581-9138-df7842d7e3b2"
USER_ID = "56eec760-3d54-4f04-a66e-ad180688f295"

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
    "What is Kanban?",
    "Which section discusses Kanban?",
    "What does the DevOps document say about Kanban?",
    "Compare Kanban and Scrum.",
    "Compare information from two documents.",
    "do you have any resume",
    "aarya document you have",
    "what does data mining have",
    "How many documents were uploaded this month?",
    "Tell me about the DevOps document.",
    "What questions about it mention Kanban?",
    "What is my name",
    "How many members are in the workspace?",
    "How many are invited?",
    "what is 2+2",
]


def run_query(token: str, query: str, idx: int) -> dict:
    """Run one query against /chat/grounded and return the result."""
    headers = {"Authorization": f"Bearer {token}"}
    payload = {"message": query}

    try:
        with httpx.Client(timeout=120.0) as client:
            resp = client.post(
                f"{BASE_URL}/chat/grounded",
                json=payload,
                headers=headers,
            )
            if resp.status_code != 200:
                return {
                    "query": query,
                    "error": f"HTTP {resp.status_code}: {resp.text[:200]}",
                }
            data = resp.json()
            return {
                "query": query,
                "answer": data.get("answer", ""),
                "grounded": data.get("grounded", None),
                "insufficient_evidence": data.get("insufficient_evidence", None),
                "sources": data.get("sources", []),
                "provider": data.get("provider", ""),
                "model": data.get("model", ""),
            }
    except Exception as e:
        return {"query": query, "error": str(e)}


def main():
    token = make_token(USER_ID, WORKSPACE_ID)
    print(f"Token generated for user={USER_ID[:8]}... workspace={WORKSPACE_ID[:8]}...")
    print()

    results = []
    for i, query in enumerate(QUERIES, 1):
        print(f"--- Query {i}/{len(QUERIES)}: {query}")
        result = run_query(token, query, i)
        results.append(result)

        # Print result summary.
        if "error" in result:
            print(f"  ERROR: {result['error']}")
        else:
            sources = result.get("sources", [])
            source_info = ""
            if sources:
                source_info = f"\n  Sources: {len(sources)} source(s)"
                for s in sources:
                    score = s.get("score", "?")
                    fname = s.get("filename", "?")
                    label = s.get("label", "?")
                    source_info += f"\n    - {fname} (score={score}) [{label}]"
            print(f"  Grounded: {result.get('grounded')}")
            print(f"  Answer: {result.get('answer', '')[:300]}")
            if source_info:
                print(source_info)
        print()

    # Save results for analysis.
    with open("/tmp/acceptance_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print("Results saved to /tmp/acceptance_results.json")


if __name__ == "__main__":
    main()
