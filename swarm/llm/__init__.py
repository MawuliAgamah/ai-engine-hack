"""
LLM integration layer for SWARM.

Provides the protocol (prompt building + response parsing) and abstract client
interface for delegating agent decisions to a Large Language Model.
"""

from swarm.llm.client import LLMClient, MockClient, OpenAIClient
from swarm.llm.prompt import build_system_prompt, build_user_message, get_available_moves
from swarm.llm.parser import LLMDecision, parse_llm_response

__all__ = [
    "LLMClient",
    "MockClient",
    "OpenAIClient",
    "build_system_prompt",
    "build_user_message",
    "get_available_moves",
    "LLMDecision",
    "parse_llm_response",
]
