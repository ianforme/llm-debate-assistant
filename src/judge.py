from prompts.judge_prompts import judge_comment_prompts

from output_structure import JudgeComment
from openai import OpenAI
from config import OPENAI_API_KEY, ORG_KEY, PROJECT_KEY
import json

client = OpenAI(
    api_key = OPENAI_API_KEY,
    organization=ORG_KEY,
    project=PROJECT_KEY
)


def generate_judge_feedback(debate_history, topic):

    res = client.responses.parse(
        model = 'gpt-5-mini-2025-08-07',
        input = judge_comment_prompts(debate_history, topic),
        reasoning={
            "effort": "low"
        },
        text={"verbosity": "high"},
        text_format = JudgeComment
    )


    return json.loads(res.output_text)