from llm_debate_assistant.core.assistant import DebateAssistant
import asyncio
from llm_debate_assistant.utils.helpers import mk_notify
from llm_debate_assistant.config import app_config
from llm_debate_assistant.utils.helpers import rewrite_style


class DebateOrchestrator:
    def __init__(self, assistant: DebateAssistant):
        self.assistant = assistant

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
