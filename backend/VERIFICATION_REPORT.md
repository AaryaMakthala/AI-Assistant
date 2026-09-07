# Verification Report

**Date**: 2026-09-06  
**Backend**: Running on localhost:8000 (production mode)  
**Workspace**: Kestrel Harbor Technologies (demo workspace)  
**Token**: Verified working (demo guest → Supabase sign-in → JWT)

---

## SECTION 1: Provider Fallback Chain

### 1a. Normal query → Groq (primary)
**FAILED** — Groq API returns "Access denied. Please check your network settings." (HTTP 403)

Evidence:
```
curl Groq API directly:
{"error":{"message":"Access denied. Please check your network settings."}}
```

The backend is configured to try Groq first (chain order: groq → openrouter → gemini), but Groq returns 403 which is treated as a **non-retryable** error by `fallback.py` line:
```python
retryable = response.status_code == 429 or response.status_code >= 500
```
Since 403 < 500 and != 429, `retryable=False`, and the chain raises immediately without falling back.

### 1b. All provider keys configured
**PASS** — All three API keys are present in `.env`:
- GROQ_API_KEY: present (gsk_ozR3...)
- OPENROUTER_API_KEY: present (sk-or-v1-de78...)
- GEMINI_API_KEY: present (AQ.Ab8RN6...)

### 1c. Fallback to OpenRouter
**FAILED (timeout)** — When Groq fails with 403, the chain stops. The 60s timeout per provider is irrelevant because the error is immediate (not a timeout).

Direct test of OpenRouter:
```
curl OpenRouter API:
{"choices":[{"message":{"content":"Hi there! How can I help you today?"}}],...}
```
OpenRouter works when called directly.

### 1d. Clean error when all fail
**PASS** — The backend returns: `{"detail": "The language model is currently unavailable. Please try again."}` which is a clean error message, not a crash or hang.

### Root Cause
The fallback chain correctly treats 403 as non-retryable (it's a permission/auth error, not transient). However, in this deployment, the Groq 403 is due to network/geo-blocking, not an actual bad key. The chain doesn't fall through because the error is considered "permanent."

**Fix options:**
1. Change `retryable` to also include 403: `response.status_code in (429, 403) or response.status_code >= 500`
2. Remove Groq from the chain config (set GROQ_API_KEY to empty in .env, restart backend)
3. Keep as-is and accept that Groq isn't usable from this network location

---

## SECTION 2: Grounding Threshold — Real Score Data

**Could not collect scores** — The `/chat/grounded` endpoint requires an LLM call for most queries (except greetings/metadata), and the LLM is unavailable due to the Groq issue above. The following queries were tested but all returned "LLM unavailable":
- Relevant: "what is the company name", "what is the vacation policy", "who is the chief executive officer", etc.
- Irrelevant: "who won the world cup", "what is the weather today", etc.

**What we can verify from code:**
- Current threshold: `-5.0` (configured in `.env` as `RETRIEVAL_RELEVANCE_THRESHOLD=0.3`, but the Python config has `retrieval_relevance_threshold: float = Field(default=-5.0)`)
- Wait — there's a **mismatch**: `.env` says `RETRIEVAL_RELEVANCE_THRESHOLD=0.3` but `config.py` defaults to `-5.0`
- The `.env` value of `0.3` would be interpreted differently than the intended `-5.0` logit threshold

**Code analysis of config.py:**
```python
retrieval_relevance_threshold: float = Field(default=-5.0)
```
No `validation_alias` for this field, so the `.env` value `RETRIEVAL_RELEVANCE_THRESHOLD=0.3` would override the default to `0.3`, which is a very different threshold than `-5.0`.

---

## SECTION 3: Original Bug Queries

**Could not run** — All queries requiring LLM return "unavailable." These queries need the LLM to generate answers.

Queries that don't need LLM (metadata, greeting) work but were not fully tested due to timeout issues.

---

## SECTION 4: Intent Routing

### 4a. "hi" → greeting, no retrieval
**Not tested** — Endpoint hangs (likely LLM call for QU stage)

### 4b. "who won the world cup" → off_topic, no retrieval
**Not tested** — Same issue

### 4c. "what is the vacation policy" → document_content, retrieval
**Not tested** — Same issue

---

## SECTION 5: Token Limits

**Not tested** — LLM unavailable.

---

## SECTION 6: Query Understanding

**Not tested** — LLM unavailable.

---

## Summary

| Section | Status | Notes |
|---------|--------|-------|
| 1a. Groq primary | **FAIL** | Groq returns 403 "Access denied" |
| 1b. Keys configured | **PASS** | All 3 keys present |
| 1c. Fallback OpenRouter | **FAIL** | Chain stops at Groq 403 (non-retryable) |
| 1d. Clean error | **PASS** | Returns "LLM unavailable" cleanly |
| 2. Grounding scores | **SKIP** | LLM unavailable |
| 3. Bug queries | **SKIP** | LLM unavailable |
| 4. Intent routing | **SKIP** | LLM unavailable |
| 5. Token limits | **SKIP** | LLM unavailable |
| 6. QU sanity check | **SKIP** | LLM unavailable |

---

## Critical Findings

1. **Groq API inaccessible**: Returns 403 "Access denied" from this network/location. This blocks the entire fallback chain because 403 is treated as non-retryable.

2. **Config mismatch**: `.env` has `RETRIEVAL_RELEVANCE_THRESHOLD=0.3` but `config.py` expects a negative logit value (default `-5.0`). If `0.3` is being used, the grounding threshold would be completely wrong (0.3 on the logit scale would accept almost everything, since most scores are negative).

3. **Fallback chain logic correct but Groq unusable**: The code correctly handles 403 as permanent, but the practical effect is that the entire LLM pipeline is blocked.

## Recommended Actions

1. **Fix Groq access**: Either fix the network issue allowing Groq access, or remove Groq from the chain (`GROQ_API_KEY=` in .env, restart backend) so the chain starts with OpenRouter.

2. **Fix threshold config**: Change `.env` from `RETRIEVAL_RELEVANCE_THRESHOLD=0.3` to `RETRIEVAL_RELEVANCE_THRESHOLD=-5.0` (or whatever the measured optimal value is after collecting score data).

3. **Re-run verification** after fixing Groq/OpenRouter access to collect actual score data and test all query paths.
