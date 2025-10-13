from llm_debate_assistant.core.assistant import DebateAssistant
from llm_debate_assistant.core.realtime_assistant import RealtimeAssistant
import asyncio
from llm_debate_assistant.utils.helpers import mk_notify, rewrite_style
from llm_debate_assistant.config import app_config

from llm_debate_assistant.prompts.rebuttal_prompts import rebuttal_crossfire_prompt


class DebateOrchestrator:
    def __init__(self, assistant: DebateAssistant, realtime_assistant: RealtimeAssistant):
        self.assistant = assistant
        self.realtime_assistant = realtime_assistant

    async def generate_opening_statement(
        self, topic: str, side: str, style_example: str, llm_as_judge: bool = True, status_cb=None
    ):
        notify = mk_notify(status_cb)

        notify("=" * 10 + "Stage 1: 生成立论框架..." + "=" * 10 + "\n")
        debate_outline = self.assistant.generate_debate_outline(topic, side)

        notify("=" * 10 + "Stage 2: 基于立论框架搜寻所需资料..." + "=" * 10 + "\n")
        debate_outline = await self.assistant.parallel_fetch_evidences(
            debate_outline, topic, side
        )

        notify("=" * 10 + "Stage 3: 生成立论初稿..." + "=" * 10 + "\n")
        opening_statement = self.assistant.initial_opening_statement(
            debate_outline, topic, side
        )

        if llm_as_judge:
            notify("=" * 10 + "Stage 4: 基于教练审核意见修改立论..." + "=" * 10 + "\n")
            enhanced_opening_statement, _ = await self.assistant.opening_statement_improvement(
                opening_statement, debate_outline, topic, side
            )

            notify("=" * 10 + "Stage 5: 基于示例优化写作风格..." + "=" * 10 + "\n")
            final_opening_statement = rewrite_style(
                enhanced_opening_statement.opening_statement,
                style_example,
            )

        else:
            notify("=" * 10 + "Stage 4: 基于示例优化写作风格..." + "=" * 10 + "\n")
            final_opening_statement = rewrite_style(
                opening_statement["opening_statement"],
                style_example,
            )

        return final_opening_statement, debate_outline
    
    async def rebuttal_crossfire_practice(
        self, topic: str, assistant_side: str, assistant_statement: str, human_statement: str, proposed_attacks: str = None
    ):
        
        if assistant_side == "正方":
            human_side = "反方"
        else:
            human_side = "正方"

        match_history = f"{human_side}:\n{self.assistant.generate_match_summary(topic, human_statement)}"
        crossfire_context = rebuttal_crossfire_prompt(topic, assistant_side, match_history, assistant_statement, proposed_attacks)
        self.realtime_assistant.run(crossfire_context)

if __name__ == "__main__":
    assistant = DebateAssistant()
    realtime_assistant = RealtimeAssistant()
    orchestrator = DebateOrchestrator(assistant, realtime_assistant)
    final_opening_statement, debate_outline = asyncio.run(
        orchestrator.generate_opening_statement(
            topic="人工智能是否应该被严格监管？", side="正方", llm_as_judge=True
        )
    )
    print(final_opening_statement)
    print(debate_outline)
