from llm_debate_assistant.prompts.rebuttal_prompts import (
    definition_rebuttal_prompt,
    weighing_criterion_rebuttal_prompt,
    argument_rebuttal_prompt,
    statement_rebuttal_prompt,
    further_rebuttal_prompt,
)
from llm_debate_assistant.templates.opening_statement import (
    opening_statement_style_example,
)
from llm_debate_assistant.utils.helpers import rewrite_style
from ..config.schemas import Rebuttal

from ..config.schemas import Rebuttal
from .client import client

import json
import asyncio


def generate_definition_rebuttal(
    oppo_statement, own_statement, topic, side, debate_outline
):
    res = client.responses.parse(
        model="gpt-5-mini-2025-08-07",
        input=definition_rebuttal_prompt(
            oppo_statement, own_statement, topic, side, debate_outline
        ),
        reasoning={"effort": "low"},
        text={"verbosity": "high"},
        text_format=Rebuttal,
    )

    return {"definition": json.loads(res.output_text)["rebuttals"]}


def generate_criterion_rebuttal(
    oppo_statement, own_statement, topic, side, debate_outline
):
    res = client.responses.parse(
        model="gpt-5-mini-2025-08-07",
        input=weighing_criterion_rebuttal_prompt(
            oppo_statement, own_statement, topic, side, debate_outline
        ),
        reasoning={"effort": "low"},
        text={"verbosity": "high"},
        text_format=Rebuttal,
    )

    return {"weighing_criterion": json.loads(res.output_text)["rebuttals"]}


def generate_argument_rebuttal(
    oppo_statement, own_statement, topic, side, debate_outline
):
    res = client.responses.parse(
        model="gpt-5-mini-2025-08-07",
        input=argument_rebuttal_prompt(
            oppo_statement,
            own_statement,
            topic,
            side,
            debate_outline,
        ),
        reasoning={"effort": "low"},
        text={"verbosity": "high"},
        text_format=Rebuttal,
    )
    return {"arguments": json.loads(res.output_text)["rebuttals"]}


async def generate_rebuttals_outline(
    oppo_statement, own_statement, topic, side, debate_outline
):
    sema = asyncio.Semaphore(3)

    async def run_with_sema(func, *args):
        async with sema:
            return await asyncio.to_thread(func, *args)

    tasks = [
        asyncio.create_task(
            run_with_sema(
                generate_definition_rebuttal,
                oppo_statement,
                own_statement,
                topic,
                side,
                debate_outline,
            )
        ),
        asyncio.create_task(
            run_with_sema(
                generate_criterion_rebuttal,
                oppo_statement,
                own_statement,
                topic,
                side,
                debate_outline,
            )
        ),
        asyncio.create_task(
            run_with_sema(
                generate_argument_rebuttal,
                oppo_statement,
                own_statement,
                topic,
                side,
                debate_outline,
            )
        ),
    ]

    results = await asyncio.gather(*tasks)
    return results


async def generate_statement_rebuttal(
    oppo_statement, own_statement, topic, side, debate_outline
):
    retbutal_outline = await generate_rebuttals_outline(
        oppo_statement, own_statement, topic, side, debate_outline
    )

    res = client.responses.create(
        model="gpt-5-mini-2025-08-07",
        input=statement_rebuttal_prompt(
            oppo_statement, own_statement, retbutal_outline, topic, side
        ),
        reasoning={"effort": "low"},
        text={"verbosity": "high"},
    )

    rewritten_res = rewrite_style(res.output_text, opening_statement_style_example)

    return rewritten_res


def generate_further_rebuttal(debate_history, topic, side, debate_outline):

    res = client.responses.create(
        model="gpt-5-mini-2025-08-07",
        input=further_rebuttal_prompt(debate_history, topic, side, debate_outline),
        reasoning={"effort": "low"},
        text={"verbosity": "high"},
    )

    rewritten_res = rewrite_style(res.output_text, opening_statement_style_example)

    return rewritten_res
