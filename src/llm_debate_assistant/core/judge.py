from prompts.judge_prompts import judge_comment_prompts

from ..config.schemas import JudgeComment
from openai import OpenAI
from config import config
import json

client = OpenAI(
    api_key=config.api_keys.openai_api_key,
    organization=config.api_keys.org_key,
    project=config.api_keys.project_key,
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
