from openai import OpenAI
from llm_debate_assistant.config import app_config

# Centralized LLM client instance
client = OpenAI(
    api_key=app_config.api_keys.openai_api_key,
    organization=app_config.api_keys.org_key,
    project=app_config.api_keys.project_key,
)
