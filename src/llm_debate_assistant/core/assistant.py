from llm_debate_assistant.core.client import client
import json
from typing import Optional, Any, Dict
from llm_debate_assistant.prompts.conclusion_prompts import conclusion_prompts
from llm_debate_assistant.utils.helpers import rewrite_style
from llm_debate_assistant.prompts.judge_prompts import judge_comment_prompts
from llm_debate_assistant.config.schemas import JudgeComment
from llm_debate_assistant.prompts.rebuttal_prompts import (
    definition_rebuttal_prompt,
    weighing_criterion_rebuttal_prompt,
    argument_rebuttal_prompt,
    further_rebuttal_prompt,
)
from llm_debate_assistant.config.schemas import Rebuttal
from llm_debate_assistant.prompts.opening_statement_prompts import (
    debate_outline_prompt,
    example_card_prompt,
    opening_statement_prompt,
)
from llm_debate_assistant.config.schemas import (
    OpeningStatementOutline,
    Examples,
    OpeningStatement,
)


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

    def generate_definition_rebuttal(
        self, oppo_statement, own_statement, topic, side, debate_outline
    ):
        res = self._generate(
            definition_rebuttal_prompt(
                oppo_statement, own_statement, topic, side, debate_outline
            ),
            structured_output=Rebuttal,
            reasoning={"effort": "low"},
            text={"verbosity": "high"},
        )
        return {"definition": res["rebuttals"]}

    def generate_criterion_rebuttal(
        self, oppo_statement, own_statement, topic, side, debate_outline
    ):
        res = self._generate(
            weighing_criterion_rebuttal_prompt(
                oppo_statement, own_statement, topic, side, debate_outline
            ),
            structured_output=Rebuttal,
            reasoning={"effort": "low"},
            text={"verbosity": "high"},
        )
        return {"weighing_criterion": res["rebuttals"]}

    def generate_argument_rebuttal(
        self, oppo_statement, own_statement, topic, side, debate_outline
    ):
        res = self._generate(
            argument_rebuttal_prompt(
                oppo_statement,
                own_statement,
                topic,
                side,
                debate_outline,
            ),
            structured_output=Rebuttal,
            reasoning={"effort": "low"},
            text={"verbosity": "high"},
        )
        return {"arguments": res["rebuttals"]}

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

    def generate_further_rebuttal(self, debate_history, topic, side, debate_outline, style_example):
        res = self._generate(
            further_rebuttal_prompt(debate_history, topic, side, debate_outline),
            reasoning={"effort": "low"},
            text={"verbosity": "high"},
        )
        rewritten_res = rewrite_style(res, style_example)
        return rewritten_res

    def generate_judge_feedback(self, debate_history, topic):
        return self._generate(
            judge_comment_prompts(debate_history, topic),
            structured_output=JudgeComment,
            reasoning={"effort": "low"},
            text={"verbosity": "high"},
        )


if __name__ == "__main__":
    assistant = DebateAssistant()
    response = assistant.generate_debate_outline("人工智能是否应该被严格监管？", "正方")
    print(response)
