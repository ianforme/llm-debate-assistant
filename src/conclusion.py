from prompts.conclusion_prompts import conclusion_prompts
from style_examples.style_examples import opening_statement_style_example
from src.utils import rewrite_style
from openai import OpenAI
from config import OPENAI_API_KEY, ORG_KEY, PROJECT_KEY

client = OpenAI(
    api_key = OPENAI_API_KEY,
    organization=ORG_KEY,
    project=PROJECT_KEY
)

def generate_conclusion(debate_history, topic, side, debate_outline):

    res = client.responses.create(
        model = 'gpt-5-mini-2025-08-07',
        input = conclusion_prompts(debate_history, topic, side, debate_outline),
        reasoning={
            "effort": "low"
        },
        text={"verbosity": "high"},
    )

    rewritten_res = rewrite_style(res.output_text, opening_statement_style_example)

    return rewritten_res 