"""LLM configuration and instance management."""

from typing import Literal

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

from llm_debate_assistant.config import app_config

# Default models for each provider
# TODO: better model management strategy
DEFAULT_GEMINI_MODEL = "gemini-3-pro-preview"
DEFAULT_OPENAI_MODEL = "gpt-5.1-2025-11-13"


def get_llm(
    temperature: float = 0.0,
    provider: Literal["gemini", "openai"] = "gemini",
    model: str | None = None,
) -> BaseChatModel:
    """Get a configured LLM instance.

    Args:
        temperature (float, optional): Sampling temperature. Defaults to 0.0.
        provider (str, optional): Model provider ("gemini" or "openai").
            Defaults to "gemini".
        model (str, optional): Model name. If None, uses default for provider.

    Returns:
        BaseChatModel: Configured LLM instance.

    Raises:
        ValueError: If an unsupported provider is specified.
    """
    if provider == "gemini":
        return ChatGoogleGenerativeAI(
            model=model or DEFAULT_GEMINI_MODEL,
            api_key=app_config.api_keys.gemini_api_key,
            temperature=temperature,
        )
    elif provider == "openai":
        return ChatOpenAI(
            model=model or DEFAULT_OPENAI_MODEL,
            api_key=app_config.api_keys.openai_api_key,
            temperature=temperature,
        )
    else:
        raise ValueError(f"Unsupported provider: {provider}")
