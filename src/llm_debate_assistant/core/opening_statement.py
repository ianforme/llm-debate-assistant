from openai import OpenAI
from llm_debate_assistant.config import app_config

import os
from llm_debate_assistant.utils.helpers import rewrite_style, mk_notify
from llm_debate_assistant.templates.opening_statement import (
    opening_statement_style_example,
)
from llm_debate_assistant.prompts.opening_statement_prompts import (
    debate_outline_prompt,
    example_card_prompt,
    opening_statement_prompt,
    opening_statement_improver_prompt,
    opening_statement_evaluator_prompt,
)
from ..config.schemas import (
    OpeningStatementOutline,
    Examples,
    OpeningStatement,
    OpeningStatementEvaluationFeedback,
)

import json
import asyncio
import random

from agents import Agent, Runner, trace, ModelSettings, WebSearchTool
from openai.types.shared import Reasoning

import datetime

client = OpenAI(
    api_key=app_config.api_keys.openai_api_key,
    organization=app_config.api_keys.org_key,
    project=app_config.api_keys.project_key,
)

os.environ["OPENAI_API_KEY"] = app_config.api_keys.openai_api_key


def generate_debate_outline(topic, side):
    print(f"规划立论框架 - {topic}： {side}")
    print("*" * 50)
    debate_outline = client.responses.parse(
        model="gpt-5-mini-2025-08-07",
        input=debate_outline_prompt(topic, side),
        reasoning={"effort": "high"},
        text_format=OpeningStatementOutline,
    )

    debate_outline = json.loads(debate_outline.output_text)

    return debate_outline


def fetch_one_sync(argument, warrant, evidence_needed, topic, side):
    print(f"资料搜寻 - 论点: {argument}\n论证: {warrant}\n所需资料: {evidence_needed}")
    print("~" * 50)
    resp = client.responses.parse(
        model="gpt-5-mini-2025-08-07",
        input=example_card_prompt(argument, warrant, evidence_needed, topic, side),
        tools=[{"type": "web_search"}],
        reasoning={"effort": "low"},
        text_format=Examples,
    )
    data = json.loads(resp.output_text)
    return data["evidences"]


async def fetch_one_wrapped(sema, argument, warrant, evidence_needed, topic, side):
    backoff = 0.5
    for attempt in range(1, app_config.run_config.retries + 1):
        try:
            async with sema:
                evidences = await asyncio.to_thread(
                    fetch_one_sync, argument, warrant, evidence_needed, topic, side
                )
            return argument, warrant, evidences
        except Exception as e:
            print(e)
            if attempt == app_config.run_config.retries:
                return argument, warrant, []
            await asyncio.sleep(
                backoff + random.random() * app_config.run_config.jitter
            )
            backoff *= 2


async def parallel_fetch_evidences(debate_outline, topic, side):
    sema = asyncio.Semaphore(app_config.run_config.concurrency)
    tasks = []

    for arg_card in debate_outline["arguments"]:
        for evidence_needed in arg_card["evidence_needed"]:
            tasks.append(
                fetch_one_wrapped(
                    sema,
                    arg_card["argument"],
                    arg_card["warrant"],
                    evidence_needed,
                    topic,
                    side,
                )
            )
    # this is much faster
    results = await asyncio.gather(*tasks)

    arguments = []
    for argument, warrant, evidences in results:
        if (len(arguments) == 0) or (
            argument not in [arg["argument"] for arg in arguments]
        ):
            new_arg = {"argument": argument, "warrant": warrant, "evidences": evidences}
            arguments.append(new_arg)
        else:
            for exist_arg in arguments:
                if exist_arg["argument"] == argument:
                    exist_arg["evidences"].extend(evidences)

    debate_outline["arguments"] = arguments

    return debate_outline


def initial_opening_statement(debate_outline, topic, side):
    print(f"生成立论初稿 - {topic}： {side}")
    print("*" * 50)
    opening_statement = client.responses.parse(
        model="gpt-5-mini-2025-08-07",
        input=opening_statement_prompt(debate_outline, topic, side),
        reasoning={"effort": "medium"},
        text_format=OpeningStatement,
    )

    opening_statement = json.loads(opening_statement.output_text)

    return opening_statement


async def opening_statement_improvement(opening_statement, debate_outline, topic, side):

    opening_statement_writer = Agent(
        name="opening_statement_writer",
        instructions=opening_statement_improver_prompt(debate_outline, topic, side),
        output_type=OpeningStatement,
        model="gpt-5-mini-2025-08-07",
        model_settings=ModelSettings(
            reasoning=Reasoning(effort="medium"), verbosity="high"
        ),
        tools=[WebSearchTool()],
    )

    opening_statement_evaluator = Agent(
        name="opening_statement_evaluator",
        instructions=opening_statement_evaluator_prompt(debate_outline, topic, side),
        output_type=OpeningStatementEvaluationFeedback,
        model="gpt-5-mini-2025-08-07",
        model_settings=ModelSettings(
            reasoning=Reasoning(effort="medium"), verbosity="low"
        ),
    )

    input_items = [
        {
            "content": f"opening statement to be evaluated: {opening_statement['opening_statement']}",
            "role": "user",
        }
    ]
    counter = 1

    with trace(f"LLM as a judge-{datetime.datetime.now().strftime("%Y%m%d%H%M%S")}"):
        while True:

            if counter == 5:
                break

            print("=" * 20 + f"Iter {counter}" + "=" * 20)
            evaluator_result = await Runner.run(
                opening_statement_evaluator, input_items
            )
            result = evaluator_result.final_output

            print(f"立论审核结果: {result.evaluation_result}")
            print("\n")

            if result.evaluation_result[0] == "pass":
                print("立论已足够优秀，退出审核")
                break

            print(f"审核反馈: \n{result.feedback}")
            print("\n")

            print("基于反馈重新生成立论...")
            input_items.append(
                {"content": f"Feedback: {result.feedback}", "role": "user"}
            )
            print("\n")

            enhanced_opening_statement = await Runner.run(
                opening_statement_writer, input_items
            )
            print(
                f"最新立论:\n\n {enhanced_opening_statement.final_output.opening_statement}"
            )
            input_items.append(
                {
                    "content": f"opening statement to be evaluated: {enhanced_opening_statement.final_output.opening_statement}",
                    "role": "user",
                }
            )

            counter += 1
            print("\n\n")

    return enhanced_opening_statement.final_output, input_items


async def generate_opening_statement(topic, side, llm_as_judge=True, status_cb=None):
    notify = mk_notify(status_cb)

    notify("=" * 10 + "Stage 1: 生成立论框架..." + "=" * 10 + "\n")
    debate_outline = generate_debate_outline(topic, side)

    notify("=" * 10 + "Stage 2: 基于立论框架搜寻所需资料..." + "=" * 10 + "\n")
    debate_outline = await parallel_fetch_evidences(debate_outline, topic, side)

    notify("=" * 10 + "Stage 3: 生成立论初稿..." + "=" * 10 + "\n")
    opening_statement = initial_opening_statement(debate_outline, topic, side)

    if llm_as_judge:
        notify("=" * 10 + "Stage 4: 基于教练审核意见修改立论..." + "=" * 10 + "\n")
        enhanced_opening_statement, _ = await opening_statement_improvement(
            opening_statement, debate_outline, topic, side
        )

        notify("=" * 10 + "Stage 5: 基于示例优化写作风格..." + "=" * 10 + "\n")
        final_opening_statement = rewrite_style(
            enhanced_opening_statement.opening_statement,
            opening_statement_style_example,
        )

    else:
        notify("=" * 10 + "Stage 4: 基于示例优化写作风格..." + "=" * 10 + "\n")
        final_opening_statement = rewrite_style(
            opening_statement["opening_statement"], opening_statement_style_example
        )

    return final_opening_statement, debate_outline
