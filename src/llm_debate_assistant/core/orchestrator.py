from llm_debate_assistant.core.assistant import DebateAssistant
import asyncio
from llm_debate_assistant.utils.helpers import mk_notify
from llm_debate_assistant.config import app_config
from agents import Agent, Runner, trace, ModelSettings, WebSearchTool
from openai.types.shared import Reasoning
import datetime
import random
from typing import Dict, Any
from llm_debate_assistant.config.schemas import (
    OpeningStatement,
    OpeningStatementEvaluationFeedback,
)
from llm_debate_assistant.prompts.opening_statement_prompts import (
    opening_statement_improver_prompt,
    opening_statement_evaluator_prompt,
)
from llm_debate_assistant.utils.helpers import rewrite_style
from llm_debate_assistant.templates.opening_statement import (
    opening_statement_style_example,
)


class DebateOrchestrator:
    def __init__(self, assistant: DebateAssistant):
        self.assistant = assistant

    async def generate_rebuttals_outline(
        self, oppo_statement, own_statement, topic, side, debate_outline
    ):
        sema = asyncio.Semaphore(3)

        async def run_with_sema(func, *args):
            async with sema:
                return await asyncio.to_thread(func, *args)

        tasks = [
            asyncio.create_task(
                run_with_sema(
                    self.assistant.generate_definition_rebuttal,
                    oppo_statement,
                    own_statement,
                    topic,
                    side,
                    debate_outline,
                )
            ),
            asyncio.create_task(
                run_with_sema(
                    self.assistant.generate_criterion_rebuttal,
                    oppo_statement,
                    own_statement,
                    topic,
                    side,
                    debate_outline,
                )
            ),
            asyncio.create_task(
                run_with_sema(
                    self.assistant.generate_argument_rebuttal,
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
        self, oppo_statement, own_statement, topic, side, debate_outline
    ):
        retbutal_outline = await self.generate_rebuttals_outline(
            oppo_statement, own_statement, topic, side, debate_outline
        )

        res = self.assistant._generate(
            statement_rebuttal_prompt(
                oppo_statement, own_statement, retbutal_outline, topic, side
            ),
            reasoning={"effort": "low"},
            text={"verbosity": "high"},
        )

        rewritten_res = rewrite_style(res, opening_statement_style_example)
        return rewritten_res

    async def fetch_one_wrapped(
        self,
        sema,
        argument: str,
        warrant: str,
        evidence_needed: str,
        topic: str,
        side: str,
    ):
        backoff = 0.5
        for attempt in range(1, app_config.run_config.retries + 1):
            try:
                async with sema:
                    evidences = await asyncio.to_thread(
                        self.assistant.fetch_one_sync,
                        argument,
                        warrant,
                        evidence_needed,
                        topic,
                        side,
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

    async def parallel_fetch_evidences(
        self, debate_outline: Dict[str, Any], topic: str, side: str
    ):
        sema = asyncio.Semaphore(app_config.run_config.concurrency)
        tasks = []

        for arg_card in debate_outline["arguments"]:
            for evidence_needed in arg_card["evidence_needed"]:
                tasks.append(
                    self.fetch_one_wrapped(
                        sema,
                        arg_card["argument"],
                        arg_card["warrant"],
                        evidence_needed,
                        topic,
                        side,
                    )
                )
        results = await asyncio.gather(*tasks)

        arguments = []
        for argument, warrant, evidences in results:
            if (len(arguments) == 0) or (
                argument not in [arg["argument"] for arg in arguments]
            ):
                new_arg = {
                    "argument": argument,
                    "warrant": warrant,
                    "evidences": evidences,
                }
                arguments.append(new_arg)
            else:
                for exist_arg in arguments:
                    if exist_arg["argument"] == argument:
                        exist_arg["evidences"].extend(evidences)

        debate_outline["arguments"] = arguments
        return debate_outline

    async def opening_statement_improvement(
        self,
        opening_statement: Dict[str, Any],
        debate_outline: Dict[str, Any],
        topic: str,
        side: str,
    ):
        opening_statement_writer = Agent(
            name="opening_statement_writer",
            instructions=opening_statement_improver_prompt(debate_outline, topic, side),
            output_type=OpeningStatement,
            model=self.assistant.model,
            model_settings=ModelSettings(
                reasoning=Reasoning(effort="medium"), verbosity="high"
            ),
            tools=[WebSearchTool()],
        )

        opening_statement_evaluator = Agent(
            name="opening_statement_evaluator",
            instructions=opening_statement_evaluator_prompt(
                debate_outline, topic, side
            ),
            output_type=OpeningStatementEvaluationFeedback,
            model=self.assistant.model,
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

        with trace(
            f"LLM as a judge-{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
        ):
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

    async def generate_opening_statement(
        self, topic: str, side: str, llm_as_judge: bool = True, status_cb=None
    ):
        notify = mk_notify(status_cb)

        notify("=" * 10 + "Stage 1: 生成立论框架..." + "=" * 10 + "\n")
        debate_outline = self.assistant.generate_debate_outline(topic, side)

        notify("=" * 10 + "Stage 2: 基于立论框架搜寻所需资料..." + "=" * 10 + "\n")
        debate_outline = await self.parallel_fetch_evidences(
            debate_outline, topic, side
        )

        notify("=" * 10 + "Stage 3: 生成立论初稿..." + "=" * 10 + "\n")
        opening_statement = self.assistant.initial_opening_statement(
            debate_outline, topic, side
        )

        if llm_as_judge:
            notify("=" * 10 + "Stage 4: 基于教练审核意见修改立论..." + "=" * 10 + "\n")
            enhanced_opening_statement, _ = await self.opening_statement_improvement(
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
                opening_statement["opening_statement"],
                opening_statement_style_example,
            )

        return final_opening_statement, debate_outline


if __name__ == "__main__":
    assistant = DebateAssistant()
    orchestrator = DebateOrchestrator(assistant)
    final_opening_statement, debate_outline = asyncio.run(
        orchestrator.generate_opening_statement(
            topic="人工智能是否应该被严格监管？", side="正方", llm_as_judge=True
        )
    )
    print(final_opening_statement)
    print(debate_outline)
