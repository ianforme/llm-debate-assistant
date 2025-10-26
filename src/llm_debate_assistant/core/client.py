from google import genai
from google.genai import types
from openai import OpenAI
from opik.integrations.openai import track_openai

from llm_debate_assistant.config import app_config

# Centralized LLM client instances - lazy initialization
_openai_client = None
_gemini_client = None
_gemini_async_client = None


def get_client() -> OpenAI:
    """Get or create the OpenAI client instance with Opik tracking."""
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAI(
            api_key=app_config.api_keys.openai_api_key,
            organization=app_config.api_keys.org_key,
            project=app_config.api_keys.project_key,
        )
        # Wrap with Opik tracking after initialization
        _openai_client = track_openai(
            _openai_client, project_name="llm-debate-assistant"
        )
    return _openai_client


def get_gemini_client() -> genai.Client:
    """Get or create the Gemini client instance.

    Returns:
        genai.Client: The Gemini client instance.
    """
    global _gemini_client
    if _gemini_client is None:
        _gemini_client = genai.Client(
            api_key=app_config.api_keys.gemini_api_key,
        )
    return _gemini_client


def get_async_gemini_client() -> genai.Client:
    """Get or create the shared async Gemini client.

    Returns:
        genai.Client: The shared async client instance.
    """
    global _gemini_async_client
    if _gemini_async_client is None:
        # Create async client with custom HTTP options
        # See: https://github.com/aio-libs/aiohttp/blob/v3.12.13/aiohttp/client.py#L170
        http_options = types.HttpOptions(async_client_args={})
        _gemini_async_client = genai.Client(
            api_key=app_config.api_keys.gemini_api_key,
            http_options=http_options,
        )
    return _gemini_async_client


# For backward compatibility
client = None  # Will be initialized on first use
