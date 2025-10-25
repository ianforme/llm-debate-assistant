import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

# Try to import pyaudio, but don't fail if it's not available
# (it's an optional dependency)
try:
    import pyaudio

    PYAUDIO_FORMAT = pyaudio.paInt16
except ImportError:
    # Fallback value if pyaudio is not installed (16-bit signed integer)
    PYAUDIO_FORMAT = 8  # paInt16 = 8

load_dotenv()


@dataclass
class ApiKeys:
    openai_api_key: str = os.getenv("OPENAI_API_KEY")
    org_key: str = os.getenv("ORG_KEY")
    project_key: str = os.getenv("PROJECT_KEY")


@dataclass
class RunConfig:
    concurrency: int = 8
    retries: int = 3
    jitter: float = 0.25


@dataclass
class RealtimeConfig:
    CHUNK_SIZE: int = 1024
    RATE: int = 24000
    FORMAT = PYAUDIO_FORMAT
    temperature: float = 0.6
    max_response_token: int = 4096


@dataclass
class AppConfig:
    api_keys: ApiKeys = field(default_factory=ApiKeys)
    run_config: RunConfig = field(default_factory=RunConfig)
    realtime_config: RealtimeConfig = field(default_factory=RealtimeConfig)


app_config = AppConfig()
