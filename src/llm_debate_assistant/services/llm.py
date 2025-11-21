"""LLM configuration and instance management."""

from langchain_google_genai import ChatGoogleGenerativeAI

from llm_debate_assistant.config import app_config

# Default model for all LLM calls
DEFAULT_MODEL = "gemini-2.5-pro"


def get_llm(temperature: float = 0.0) -> ChatGoogleGenerativeAI:
    """Get a configured Gemini LLM instance.

    Args:
        temperature (float, optional): Sampling temperature.
        Defaults to 0.0.

    Returns:
        ChatGoogleGenerativeAI: Configured Gemini LLM instance.
    """
    return ChatGoogleGenerativeAI(
        model=DEFAULT_MODEL,
        api_key=app_config.api_keys.gemini_api_key,
        temperature=temperature,
    )
