from llm_debate_assistant.prompts.conclusion_prompts import conclusion_prompts
from llm_debate_assistant.templates.opening_statement import (
    opening_statement_style_example,
)
from llm_debate_assistant.utils.helpers import rewrite_style
from .client import client


def generate_conclusion(debate_history, topic, side, debate_outline):

    res = client.responses.create(
        model="gpt-5-mini-2025-08-07",
        input=conclusion_prompts(debate_history, topic, side, debate_outline),
        reasoning={"effort": "low"},
        text={"verbosity": "high"},
    )

    rewritten_res = rewrite_style(res.output_text, opening_statement_style_example)

    return rewritten_res
