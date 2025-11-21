"""Tests for LLM instance creation."""

import pytest
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

from llm_debate_assistant.services.llm import (
    DEFAULT_GEMINI_MODEL,
    DEFAULT_OPENAI_MODEL,
    get_llm,
)


class TestGetLLM:
    """Tests for get_llm function."""

    def test_default_returns_gemini(self):
        """Default call should return Gemini model."""
        llm = get_llm()
        assert isinstance(llm, ChatGoogleGenerativeAI)
        # Gemini adds 'models/' prefix to model names
        assert llm.model.endswith(DEFAULT_GEMINI_MODEL)
        assert llm.temperature == 0.0

    def test_gemini_provider_explicit(self):
        """Explicit gemini provider should return Gemini model."""
        llm = get_llm(provider="gemini")
        assert isinstance(llm, ChatGoogleGenerativeAI)
        assert llm.model.endswith(DEFAULT_GEMINI_MODEL)

    def test_openai_provider(self):
        """OpenAI provider should return ChatOpenAI model."""
        llm = get_llm(provider="openai")
        assert isinstance(llm, ChatOpenAI)
        assert llm.model_name == DEFAULT_OPENAI_MODEL

    def test_custom_temperature_gemini(self):
        """Custom temperature should be applied for Gemini."""
        llm = get_llm(temperature=0.7)
        assert llm.temperature == 0.7

    def test_custom_model_gemini(self):
        """Custom model name should override default for Gemini."""
        custom_model = "gemini-2.5-pro"
        llm = get_llm(model=custom_model)
        assert llm.model.endswith(custom_model)

    def test_custom_model_openai(self):
        """Custom model name should override default for OpenAI."""
        custom_model = "gpt-5-nano-2025-08-07"
        llm = get_llm(provider="openai", model=custom_model)
        assert llm.model_name == custom_model

    def test_invalid_provider_raises_error(self):
        """Invalid provider should raise ValueError."""
        with pytest.raises(ValueError, match="Unsupported provider"):
            get_llm(provider="anthropic")  # type: ignore

    def test_all_parameters_combined(self):
        """All parameters should work together."""
        llm = get_llm(
            temperature=0.8,
            provider="openai",
            model="gpt-5-nano-2025-08-07",
        )
        assert isinstance(llm, ChatOpenAI)
        assert llm.model_name == "gpt-5-nano-2025-08-07"
