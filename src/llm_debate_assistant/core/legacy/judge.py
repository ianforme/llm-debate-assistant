from llm_debate_assistant.prompts.judge_prompts import judge_comment_prompts

from ..config.schemas import JudgeComment
from .client import client
import json


def generate_judge_feedback(debate_history, topic):

    res = client.responses.parse(
        model="gpt-5-mini-2025-08-07",
        input=judge_comment_prompts(debate_history, topic),
        reasoning={"effort": "low"},
        text={"verbosity": "high"},
        text_format=JudgeComment,
    )

    return json.loads(res.output_text)
