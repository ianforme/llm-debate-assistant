from prompts.conclusion_prompts import conclusion_prompts
from llm_debate_assistant.templates.opening_statement import (
    opening_statement_style_example,
)
from src.utils import rewrite_style
from openai import OpenAI
from config import config

client = OpenAI(
    api_key=config.api_keys.openai_api_key,
    organization=config.api_keys.org_key,
    project=config.api_keys.project_key,
)


def generate_conclusion(debate_history, topic, side, debate_outline):

    res = client.responses.create(
        model="gpt-5-mini-2025-08-07",
        input=conclusion_prompts(debate_history, topic, side, debate_outline),
        reasoning={"effort": "low"},
        text={"verbosity": "high"},
    )

    rewritten_res = rewrite_style(res.output_text, opening_statement_style_example)

    return rewritten_res
