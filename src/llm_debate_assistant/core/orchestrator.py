from llm_debate_assistant.core.assistant import DebateAssistant
from llm_debate_assistant.core.realtime_assistant import RealtimeAssistant
import asyncio
from llm_debate_assistant.utils.helpers import mk_notify, rewrite_style
from llm_debate_assistant.config import app_config

from llm_debate_assistant.prompts.rebuttal_prompts import (
    rebuttal_crossfire_or_interrogation_prompt, 
    rebuttal_interrogated_prompt
)

from llm_debate_assistant.prompts.oregon_oxford_prompts import (
    oregon_interrogated_prompts
)

class DebateOrchestrator:
    def __init__(self, assistant: DebateAssistant, realtime_assistant: RealtimeAssistant):
        self.assistant = assistant
        self.realtime_assistant = realtime_assistant


    def generate_opening_statement_sync(
        self, topic: str, side: str, style_example: str, status_cb=None
    ):
        notify = mk_notify(status_cb)

        notify("=" * 10 + "Stage 1: 生成立论框架..." + "=" * 10 + "\n")
        debate_outline = self.assistant.generate_debate_outline(topic, side)

        notify("=" * 10 + "Stage 2: 基于立论框架搜寻所需资料..." + "=" * 10 + "\n")
        debate_outline = self.assistant.sequential_fetch_evidences(
            debate_outline, topic, side
        )

        notify("=" * 10 + "Stage 3: 生成立论初稿..." + "=" * 10 + "\n")
        opening_statement = self.assistant.initial_opening_statement(
            debate_outline, topic, side
        )

        notify("=" * 10 + "Stage 4: 基于示例优化写作风格..." + "=" * 10 + "\n")
        final_opening_statement = rewrite_style(
            opening_statement["opening_statement"],
            style_example,
        )

        return final_opening_statement, debate_outline
        

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
    
    def rebuttal_crossfire_practice(
        self, topic: str, assistant_side: str, assistant_statement: str, 
        human_statement: str, user_time_in_seconds: int, 
        proposed_attacks: str = None,
        status_cb=None
    ):
        notify = mk_notify(status_cb)
        notify("初始化对辩环节中...\n")

        match_history = f"用户:\n{self.assistant.generate_match_summary(topic, human_statement)}"
        exchange_context = rebuttal_crossfire_or_interrogation_prompt(topic, assistant_side, match_history, assistant_statement, proposed_attacks)
        notify("开始对辩环节...\n")
        speech_history = self.realtime_assistant.run(exchange_context, user_time_in_seconds)

        speech_history_text = "\n".join(speech_history)
        match_history += "\n" + speech_history_text
        match_history = "【练习背景】\n用户与AI进行对辩练习\n" + match_history

        notify("教练打分中...\n")
        judge_feedback = self.assistant.generate_exchange_practice_feedback(match_history, topic)

        return judge_feedback, speech_history
    
    def rebuttal_interrogated_practice(
        self, topic: str, assistant_side: str, assistant_statement: str, 
        human_statement: str, user_time_in_seconds: int, 
        proposed_attacks: str = None, status_cb=None
    ):
        notify = mk_notify(status_cb)
        notify("初始化质询环节中...\n")
        match_history = f"用户:\n{self.assistant.generate_match_summary(topic, human_statement)}"
        exchange_context = rebuttal_crossfire_or_interrogation_prompt(topic, assistant_side, match_history, assistant_statement, proposed_attacks, is_interrogation=True)
        notify("开始质询环节...\n")
        speech_history = self.realtime_assistant.run(exchange_context, user_time_in_seconds)

        speech_history_text = "\n".join(speech_history)
        match_history += "\n" + speech_history_text
        match_history = "【练习背景】\n用户与AI进行质询练习，AI为质询方，用户为被质询方\n" + match_history

        notify("教练打分中...\n")
        judge_feedback = self.assistant.generate_exchange_practice_feedback(match_history, topic)

        return judge_feedback, speech_history
    

    def rebuttal_interrogation_practice(
        self, topic: str, assistant_side: str, assistant_statement: str, user_time_in_seconds: int, status_cb=None
    ):
        
        notify = mk_notify(status_cb)
        notify("初始化质询环节中...\n")
        match_history = f"AI助手:\n{self.assistant.generate_match_summary(topic, assistant_statement)}"
        exchange_context = rebuttal_interrogated_prompt(topic, assistant_side, match_history, assistant_statement)
        notify("开始质询环节...\n")
        speech_history = self.realtime_assistant.run(exchange_context, user_time_in_seconds, ai_start_first=False)

        speech_history_text = "\n".join(speech_history)
        match_history += "\n" + speech_history_text
        match_history = "【练习背景】\n用户与AI进行质询练习，用户为质询方，AI为被质询方\n" + match_history

        notify("教练打分中...\n")
        judge_feedback = self.assistant.generate_exchange_practice_feedback(match_history, topic)

        return judge_feedback, speech_history
    
    def oregon_interrogation_practice(
        self, topic: str, assistant_side: str, 
        assistant_statement: str, assistant_baseline:str,
        assistant_examples: str, 
        user_time_in_seconds: int, status_cb=None
    ):
        notify = mk_notify(status_cb)
        notify("初始化质询环节中...\n")
        exchange_context = oregon_interrogated_prompts(topic, assistant_side, assistant_baseline, assistant_statement, assistant_examples)
        notify("开始质询环节...\n")
        speech_history = self.realtime_assistant.run(exchange_context, user_time_in_seconds, ai_start_first=False)

        match_history = f"AI助手:\n{assistant_statement}"
        speech_history_text = "\n".join(speech_history)
        match_history += "\n" + speech_history_text
        match_history = "【练习背景】\n用户与AI进行质询练习，用户为质询方，AI为被质询方\n" + match_history

        notify("教练打分中...\n")
        judge_feedback = self.assistant.generate_exchange_practice_feedback(match_history, topic)

        return judge_feedback, speech_history

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
