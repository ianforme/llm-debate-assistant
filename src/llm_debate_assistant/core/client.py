from openai import OpenAI

from llm_debate_assistant.config import app_config

# Centralized LLM client instance - lazy initialization
_client = None


def get_client() -> OpenAI:
    """Get or create the OpenAI client instance."""
    global _client
    if _client is None:
        _client = OpenAI(
            api_key=app_config.api_keys.openai_api_key,
            organization=app_config.api_keys.org_key,
            project=app_config.api_keys.project_key,
        )
    return _client


# For backward compatibility
client = None  # Will be initialized on first use
