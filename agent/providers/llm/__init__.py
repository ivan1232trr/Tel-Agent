"""LLM providers: stream(messages, tools) -> token stream + tool calls."""

from __future__ import annotations

from pathlib import Path

from agent.config import LlmSettings, llm_settings
from agent.providers.llm.base import LLMProvider, Message
from agent.providers.llm.openai_compatible import OpenAICompatibleLLM

__all__ = [
    "LLMProvider",
    "Message",
    "OpenAICompatibleLLM",
    "configured_provider",
    "provider_for",
]


def provider_for(settings: LlmSettings) -> LLMProvider:
    """The implementation named by a configuration.

    One `if` today and a dictionary the day there are three. It stays a function so the
    choice lives in one place: a second implementation added at the call sites is a
    second implementation that some call site does not know about.
    """
    if settings.provider == "openai":
        return OpenAICompatibleLLM(settings)
    if settings.provider == "chatgpt_plan":
        from agent.chatgpt_auth import ChatGPTAuthStore
        from agent.config import chatgpt_auth_directory
        from agent.providers.llm.chatgpt_plan import ChatGPTPlanLLM

        directory = (
            Path(settings.chatgpt_auth_dir)
            if settings.chatgpt_auth_dir
            else chatgpt_auth_directory()
        )
        store = ChatGPTAuthStore(directory)
        # Capture the selected registration for this turn. A concurrent account
        # switch must not change who pays for an in-flight conversation turn.
        selected = settings.chatgpt_client_id or store.status().get("selected_client_id")

        async def token() -> str:
            if not isinstance(selected, str) or not selected:
                raise ValueError("No ChatGPT account is selected.")
            return await store.access_token(client_id=selected)

        return ChatGPTPlanLLM(settings.model, token)
    # `agent.config` refuses an unsupported name before this is reached, so arriving
    # here means the two lists have drifted apart - which is a bug in this file.
    raise ValueError(f"no implementation for LLM provider {settings.provider!r}")


def configured_provider() -> LLMProvider | None:
    """What the environment says should answer, or `None` when nothing is configured."""
    settings = llm_settings()
    return None if settings is None else provider_for(settings)
