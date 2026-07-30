"""LLM provider abstraction with configurable failover (default: Groq → Gemini)."""

from app.llm.base import (
    Completion,
    LLMError,
    LLMProvider,
    LLMRouterProtocol,
    Message,
    Role,
    TokenUsage,
)
from app.llm.router import LLMRouter, build_default_router

__all__ = [
    "Completion",
    "LLMError",
    "LLMProvider",
    "LLMRouter",
    "LLMRouterProtocol",
    "Message",
    "Role",
    "TokenUsage",
    "build_default_router",
]
