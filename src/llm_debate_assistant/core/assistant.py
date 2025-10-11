from llm_debate_assistant.core.client import client
import json
from typing import Optional, Any, Dict
from llm_debate_assistant.prompts.conclusion_prompts import conclusion_prompts
from llm_debate_assistant.utils.helpers import rewrite_style
from llm_debate_assistant.prompts.judge_prompts import judge_comment_prompts
from llm_debate_assistant.config.schemas import JudgeComment
from llm_debate_assistant.prompts.rebuttal_prompts import (
    rebuttal_statement_prompt
)
from llm_debate_assistant.config.schemas import Rebuttal
from llm_debate_assistant.prompts.opening_statement_prompts import (
    debate_outline_prompt,
    example_card_prompt,
    opening_statement_prompt,
    opening_statement_improver_prompt,
    opening_statement_evaluator_prompt

)
from llm_debate_assistant.prompts.summary_prompt import match_summary_prompt
from llm_debate_assistant.config.schemas import (
    OpeningStatementOutline,
    Examples,
    OpeningStatement,
    MatchTurnSummary,
    OpeningStatementEvaluationFeedback
)

from agents import Agent, Runner, trace, ModelSettings, WebSearchTool
from openai.types.shared import Reasoning

import datetime
import asyncio


class DebateAssistant:
    def __init__(self, model: str = "gpt-5-mini-2025-08-07"):
        self.model = model
        self.client = client

    def _generate(
        self,
        input_prompt: str,
        structured_output: Optional[Any] = None,
        **kwargs: Dict[str, Any],
    ):
        """
        A generic method to interact with the LLM client.
        It can handle both structured (parse) and unstructured (create) responses.
        """
        params = {
            "model": self.model,
            "input": input_prompt,
        }
        params.update(kwargs)

        if structured_output:
            response = self.client.responses.parse(
                **params,
                text_format=structured_output,
            )
            return json.loads(response.output_text)
        else:
            response = self.client.responses.create(
                **params,
            )
            return response.output_text

    def generate_conclusion(self, debate_history, topic, side, debate_outline, style_example):
        res = self._generate(
            conclusion_prompts(debate_history, topic, side, debate_outline),
            reasoning={"effort": "low"},
            text={"verbosity": "high"},
        )

        rewritten_res = rewrite_style(res.output_text, style_example)
        return rewritten_res

    def generate_debate_outline(self, topic: str, side: str):
        print(f"规划立论框架 - {topic}： {side}")
        print("*" * 50)
        return self._generate(
            debate_outline_prompt(topic, side),
            structured_output=OpeningStatementOutline,
            reasoning={"effort": "high"},
        )

    def search_for_evidence(
        self, argument: str, warrant: str, evidence_needed: str, topic: str, side: str
    ):
        print(
            f"资料搜寻 - 论点: {argument}\n论证: {warrant}\n所需资料: {evidence_needed}"
        )
        print("~" * 50)
        data = self._generate(
            example_card_prompt(argument, warrant, evidence_needed, topic, side),
            structured_output=Examples,
            tools=[{"type": "web_search"}],
            reasoning={"effort": "low"},
        )
        return data["evidences"]

    def initial_opening_statement(
        self, debate_outline: Dict[str, Any], topic: str, side: str
    ):
        print(f"生成立论初稿 - {topic}： {side}")
        print("*" * 50)
        return self._generate(
            opening_statement_prompt(debate_outline, topic, side),
            structured_output=OpeningStatement,
            reasoning={"effort": "medium"},
        )
    
    def generate_match_summary(
            self, topic: str, speech: str
    ):
        print("总结辩手发言...")
        return self._generate(
            match_summary_prompt(topic, speech),
            MatchTurnSummary,
            reasoning={"effort": "minimal"},
            text={"verbosity": "low"},
        )

    def generate_judge_feedback(self, debate_history, topic):
        return self._generate(
            judge_comment_prompts(debate_history, topic),
            structured_output=JudgeComment,
            reasoning={"effort": "low"},
            text={"verbosity": "high"},
        )
    
    def generate_rebuttal_statement(self, topic, side, match_history, debate_outline, minutes):
        return self._generate(
            rebuttal_statement_prompt(topic, side, match_history, debate_outline, minutes),
            reasoning = {'effort': 'medium'},
        )
    
    async def parallel_fetch_evidences(
        self, debate_outline: Dict[str, Any], topic: str, side: str
    ):  
        # default we have 3 arguments per opening statement, each argument will take one thread
        sema = asyncio.Semaphore(3)
        tasks = []

        async def run_with_sema(func, argument, warrant, evidence_needed, topic, side):
            async with sema:
                return argument, warrant, await asyncio.to_thread(func, argument, warrant, evidence_needed, topic, side)

        for arg_card in debate_outline["arguments"]:
            argument = arg_card['argument']
            warrant = arg_card['warrant']
            evidence_needed = ';'.join(arg_card['evidence_needed'])
            tasks.append(
                asyncio.create_task(
                    run_with_sema(
                        self.search_for_evidence,
                        argument,
                        warrant,
                        evidence_needed,
                        topic,
                        side,
                    )
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
            model=self.model,
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
            model=self.model,
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


if __name__ == "__main__":
    assistant = DebateAssistant()
    response = assistant.generate_debate_outline("人工智能是否应该被严格监管？", "正方")
    print(response)
