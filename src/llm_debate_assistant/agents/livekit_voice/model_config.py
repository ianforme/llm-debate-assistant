from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class ModelType(str, Enum):
    """Available model architectures for the voice agent."""

    GOOGLE_REALTIME = "google_realtime"
    OPENAI_REALTIME = "openai_realtime"
    STANDARD = "standard"


class StandardPipelineConfig(BaseModel):
    """Configuration for standard pipeline (STT + LLM + TTS + VAD)."""

    stt_model: str = Field(
        default="assemblyai/universal-streaming",
        description="Speech-to-text model identifier",
    )
    stt_language: str = Field(
        default="zh",
        description="Language code for STT (e.g., 'zh' for Chinese, 'en' for English)",
    )

    llm_model: str = Field(
        default="openai/gpt-4o-mini",
        description="Language model identifier",
    )

    tts_model: str = Field(
        default="cartesia/sonic-3",
        description="Text-to-speech model identifier",
    )
    tts_voice: str = Field(
        default="9626c31c-bec5-4cca-baa8-f8ba9e84c8bc",
        description="Voice ID for TTS model",
    )
    tts_language: str = Field(
        default="zh",
        description="Language code for TTS (e.g., 'zh' for Chinese, 'en' for English)",
    )


class GoogleRealtimeConfig(BaseModel):
    """Configuration for Google Realtime Model."""

    model: str = Field(
        default="gemini-2.5-flash-native-audio-preview-09-2025",
        description="Google Realtime model identifier",
    )
    enable_google_search: bool = Field(
        default=False,
        description="Enable Google Search tools for the model",
    )


class OpenAIRealtimeConfig(BaseModel):
    """Configuration for OpenAI Realtime Model."""

    model: str = Field(
        default="gpt-realtime-2025-08-28",
        description="OpenAI Realtime model identifier",
    )
    voice: str = Field(
        default="alloy",
        description="Voice for OpenAI Realtime API (alloy, echo, shimmer, etc.)",
    )


class DebateConfig(BaseModel):
    """Configuration for debate-specific parameters.

    These are provided by the user at runtime, not hardcoded.
    """

    topic: str = Field(
        description="Debate topic (provided by user)",
    )
    assistant_side: str = Field(
        description="Which side the assistant is arguing for (provided by user)",
    )
    debate_baseline: str = Field(
        description="Baseline debate guidelines (from prompts module)",
    )
    match_history: str = Field(
        description="Match history context (from prompts module)",
    )
    assistant_examples: str = Field(
        description="Example responses for the assistant (from prompts module)",
    )


class AgentConfig(BaseModel):
    """Main configuration for the voice agent."""

    # Model selection
    model_type: ModelType = Field(
        default=ModelType.OPENAI_REALTIME,
        description="Which model architecture to use",
    )

    # Model-specific configurations
    google_realtime: GoogleRealtimeConfig = Field(default_factory=GoogleRealtimeConfig)
    openai_realtime: OpenAIRealtimeConfig = Field(default_factory=OpenAIRealtimeConfig)
    standard_pipeline: StandardPipelineConfig = Field(default_factory=StandardPipelineConfig)

    # Debate-specific configuration (optional, set at runtime)
    debate: Optional[DebateConfig] = Field(
        default=None,
        description="Debate-specific parameters (provided at runtime)",
    )

    # General settings
    enable_noise_cancellation: bool = Field(
        default=True,
        description="Enable BVC noise cancellation",
    )


# Default configuration instance (debate config is set at runtime)
DEFAULT_CONFIG = AgentConfig()


# You can override with custom configurations
CUSTOM_CONFIGS = {
    "google_zh": AgentConfig(
        model_type=ModelType.GOOGLE_REALTIME,
        google_realtime=GoogleRealtimeConfig(enable_google_search=False),
    ),
    "openai_zh": AgentConfig(
        model_type=ModelType.OPENAI_REALTIME,
        openai_realtime=OpenAIRealtimeConfig(voice="alloy"),
    ),
    "standard_zh": AgentConfig(
        model_type=ModelType.STANDARD,
        standard_pipeline=StandardPipelineConfig(
            stt_language="zh",
            llm_model="openai/gpt-4o-mini",
            tts_language="zh",
        ),
    ),
}


def get_config(config_name: Optional[str] = None) -> AgentConfig:
    """Get agent configuration by name.

    Args:
        config_name (Optional[str], optional): Name of the configuration to load.
            If None, returns the default configuration. Defaults to None.

    Raises:
        KeyError: If the specified configuration name does not exist.

    Returns:
        AgentConfig: The agent configuration instance.
    """
    if config_name is None:
        return DEFAULT_CONFIG

    if config_name not in CUSTOM_CONFIGS:
        raise KeyError(
            f"Configuration '{config_name}' not found. "
            f"Available configs: {list(CUSTOM_CONFIGS.keys())}"
        )

    return CUSTOM_CONFIGS[config_name]
