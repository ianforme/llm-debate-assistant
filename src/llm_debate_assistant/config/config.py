import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

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
class AppConfig:
    api_keys: ApiKeys = field(default_factory=ApiKeys)
    run_config: RunConfig = field(default_factory=RunConfig)
    is_demo: bool = True


app_config = AppConfig()
