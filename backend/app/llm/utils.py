"""Shared LLM utilities — think-tag stripping (used by chat_v2, relevance, QU, router).

Models like Qwen3 (on Groq) and Gemini-2.5 (on OpenRouter)
emit reasoning wrapped in tags like ``<|start_of_thought|>...<|end_of_thought|>`` or
``<think>...</think>`` before the final answer.  These tags must be stripped from:

* **Answer text** — the user must see only the final answer, not the reasoning preamble.
* **Structured LLM responses** — JSON classification responses (relevance gate, query
  understanding, intent router) break if the thinking tags are embedded in the output.

This module provides the canonical ``strip_think_tags`` function used everywhere.
"""

from __future__ import annotations

import re

# Real-world tag shapes seen in this project's configured providers:
# * Qwen3 (Groq) emits a *bare-word* opener on its own line: a line whose
#   content is just "thinking" (optionally leading whitespace), followed by the
#   reasoning, then a matching closer line.  No angle brackets.
# * Qwen3 / vLLM-style endpoints use ``<|start_of_thought|>...<|end_of_thought|>``
#   angle-bracket markers.
# * ``<think>...</think>`` (with or without the ``|`` wrapper) is also common.
#
# The regexes below accept BOTH the bare-word line form AND the bracketed form,
# so the same code handles every configured provider's actual output.

_THINK_OPENERS = re.compile(
    r"<\|?(?:start_of_thought|thinking_start|thinking|think)\|?>"
    r"|^\s*thinking\s*$\r?\n",
    re.MULTILINE | re.IGNORECASE,
)

_THINK_CLOSERS = re.compile(
    r"<\|?(?:end_of_thought|thinking_end|/think|/thinking)\s*\|?>"
    r"|^\s*(?:thinks?|done|stop|end_of_thought|thinking_end)\s*$\r?\n",
    re.MULTILINE | re.IGNORECASE,
)


def detect_unclosed_think_block(text: str) -> bool:
    """Return True when ``text`` contains a think opener that was never closed.

    This is the signature of a max-tokens cutoff mid-reasoning: the model was
    still thinking when the token budget ran out, so its actual answer (e.g.
    the QU JSON object) was never emitted.  Callers must treat this as a
    distinct failure mode — not as an empty response — so metrics and logs can
    tell "the model said nothing" apart from "the model was cut off".
    """
    if not text:
        return False
    openers = list(_THINK_OPENERS.finditer(text))
    if not openers:
        return False
    closers = list(_THINK_CLOSERS.finditer(text))
    # An opener is unclosed when it has no closer after it.
    last_opener = openers[-1].start()
    return not any(c.start() > last_opener for c in closers)


def strip_think_tags(text: str) -> str:
    """Remove model-injected thinking/reasoning blocks from LLM output.

    Some models (e.g. Qwen3 on Groq, Gemini-2.5 on OpenRouter) emit reasoning wrapped in tags like
    ``<|start_of_thought|>...<|end_of_thought|>``.  The user must see only the
    final answer.  This removes the entire block (tags and the reasoning text
    between them), and also drops anything after an opener that was never closed
    (a max-tokens cutoff mid-reasoning).
    """
    parts = _THINK_OPENERS.split(text)
    if len(parts) == 1:
        # No opener — nothing to strip.
        return text.strip()

    # The splitter includes the matched opener text as the *separator*, which
    # means parts[1], parts[2], ... begin with whatever followed each opener.
    # Reassemble: keep parts[0] (text before the first opener) and, for each
    # remaining segment, keep only the text that comes after a closing marker.
    # Content inside an unclosed block (no closer before the next opener or the
    # end of the string) is dropped.
    kept = [parts[0]]
    for segment in parts[1:]:
        # segment is everything after one opener up to (but not including) the
        # next opener, OR the tail after the last opener.
        # Split off the FIRST closing marker (if any) to recover what follows it.
        closed = _THINK_CLOSERS.split(segment, maxsplit=1)
        if len(closed) == 2:
            # There was a closer: drop the reasoning between opener and closer,
            # keep what comes after the closer.
            kept.append(closed[1])
        # else: unclosed block — drop it entirely (max_tokens cutoff mid-reasoning).
    return "".join(kept).strip()


# ---------------------------------------------------------------------------
# Streaming variant
# ---------------------------------------------------------------------------

#: Length of the rolling lookahead buffer used by the streaming think filter.
_THINK_TAIL = 40


async def stream_think_filtered(source):  # type: ignore[type-arg]
    """Wrap an LLM token stream, suppressing think-block reasoning text.

    The streaming path must not send reasoning to the client, so this filters
    tokens as they arrive rather than relying on a post-hoc strip.  It keeps a
    rolling buffer and only yields text once it is known to be outside a
    reasoning block (``<|start_of_thought|>...<|end_of_thought|>`` or
    ``<think>...</think>``).  Markers that arrive split across stream
    tokens are still caught because we scan the whole buffered window.  An
    unclosed block (max-tokens cutoff mid-reasoning) is discarded entirely.
    """
    from collections.abc import AsyncIterator  # noqa: F811

    buf = ""
    in_think = False
    async for token in source:
        buf += token
        while True:
            if in_think:
                close = _THINK_CLOSERS.search(buf)
                if close is not None:
                    buf = buf[close.end():]
                    in_think = False
                else:
                    # Still inside reasoning — hold.  Keep only the tail so an
                    # arbitrarily long reasoning block cannot grow memory.
                    if len(buf) > _THINK_TAIL:
                        buf = buf[-_THINK_TAIL:]
                    break
            else:
                open_m = _THINK_OPENERS.search(buf)
                close_m = _THINK_CLOSERS.search(buf)
                if open_m is not None and (close_m is None or open_m.start() < close_m.start()):
                    # Emit everything before the opener, drop the opener, hide.
                    if open_m.start():
                        yield buf[:open_m.start()]
                    buf = buf[open_m.end():]
                    in_think = True
                elif close_m is not None:
                    # Stray closer with no preceding opener — drop just the closer.
                    if close_m.start():
                        yield buf[:close_m.start()]
                    buf = buf[close_m.end():]
                else:
                    # No markers in the buffered window: emit all but the tail,
                    # which we hold back in case a marker starts at the boundary.
                    if len(buf) > _THINK_TAIL:
                        yield buf[:-_THINK_TAIL]
                        buf = buf[-_THINK_TAIL:]
                    break
    # End of stream.
    if in_think:
        return  # unclosed thinking block — discard the remainder
    if buf:
        yield buf


__all__ = [
    "detect_unclosed_think_block",
    "strip_think_tags",
    "stream_think_filtered",
]
