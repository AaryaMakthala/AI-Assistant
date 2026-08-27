"""Run all 15 Phase B-2 acceptance queries — captures provider info and full diagnostics."""

import json
import time
import jwt
import httpx

BASE_URL = "http://localhost:8000"
WORKSPACE_ID = "abb1bb8b-84e7-4581-9138-df7842d7e3b2"
USER_ID = "56eec760-3d54-4f04-a66e-ad180688f295"
JWT_SECRET = "dev-only-jwt-secret-replace-in-production-if-using-hs256"


def make_token(user_id, workspace_id):
    now = int(time.time())
    payload = {
        "sub": user_id, "workspace_id": workspace_id,
        "email": "test@example.com", "aud": "authenticated",
        "iat": now, "exp": now + 3600,
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


def run_query(token, query):
    headers = {"Authorization": f"Bearer {token}"}
    try:
        with httpx.Client(timeout=120.0) as client:
            resp = client.post(f"{BASE_URL}/chat/grounded", json={"message": query}, headers=headers)
            if resp.status_code != 200:
                return {"query": query, "error": f"HTTP {resp.status_code}: {resp.text[:300]}"}
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
    results = []
    for i, query in enumerate(QUERIES, 1):
        print(f"--- Query {i}/{len(QUERIES)}: {query}")
        result = run_query(token, query)
        results.append(result)
        if "error" in result:
            print(f"  ERROR: {result['error']}")
        else:
            sources = result.get("sources", [])
            provider = result.get("provider", "")
            model = result.get("model", "")
            print(f"  Grounded: {result.get('grounded')}")
            print(f"  Provider: {provider or '(none - no LLM call)'}")
            print(f"  Model: {model or '(none)'}")
            print(f"  Answer: {result.get('answer', '')[:400]}")
            if sources:
                print(f"  Sources: {len(sources)}")
                for s in sources[:3]:
                    print(f"    - {s.get('filename', '?')} (score={s.get('score', '?')})")
            else:
                print(f"  Sources: 0 (bypassed retrieval)")
        print()
        if i < len(QUERIES):
            time.sleep(5)

    with open("/tmp/acceptance_results3.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print("Results saved to /tmp/acceptance_results3.json")


if __name__ == "__main__":
    main()
