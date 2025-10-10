from llm_debate_assistant.prompts.judge_prompts import judge_comment_prompts

from ..config.schemas import JudgeComment
from openai import OpenAI
from llm_debate_assistant.config import app_config
import json

client = OpenAI(
    api_key=app_config.api_keys.openai_api_key,
    organization=app_config.api_keys.org_key,
    project=app_config.api_keys.project_key,
)


def generate_judge_feedback(debate_history, topic):

    res = client.responses.parse(
        model="gpt-5-mini-2025-08-07",
        input=judge_comment_prompts(debate_history, topic),
        reasoning={"effort": "low"},
        text={"verbosity": "high"},
        text_format=JudgeComment,
    )

    return json.loads(res.output_text)
