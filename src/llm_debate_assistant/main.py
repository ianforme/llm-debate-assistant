from llm_debate_assistant.core.opening_statement import generate_opening_statement
from llm_debate_assistant.core.rebuttals import (
    generate_statement_rebuttal,
    generate_further_rebuttal,
)
from llm_debate_assistant.core.conclusion import generate_conclusion
from llm_debate_assistant.core.judge import generate_judge_feedback
from llm_debate_assistant.utils.helpers import mk_notify
import pickle
import time
import os
from llm_debate_assistant.config import app_config

# go 2 levels up from current file and then into demo_outputs
_main_py_dir = os.path.dirname(os.path.abspath(__file__))
_demo_outputs_dir = os.path.join(_main_py_dir, "..", "..", "demo_outputs")


async def match_preparation(
    topic, llm_as_judge=True, status_cb=None, demo=app_config.is_demo
):
    notify = mk_notify(status_cb)
    notify(f"开始赛前准备：{topic}（LLM教练={llm_as_judge}）\n")

    if demo:
        notify("正在准备正方框架及立论...\n")
        notify("=" * 10 + "Stage 1: 生成立论框架..." + "=" * 10 + "\n")
        time.sleep(0.5)
        notify("=" * 10 + "Stage 2: 基于立论框架搜寻所需资料..." + "=" * 10 + "\n")
        time.sleep(0.5)
        notify("=" * 10 + "Stage 3: 生成立论初稿..." + "=" * 10 + "\n")
        time.sleep(0.5)
        notify("=" * 10 + "Stage 4: 基于教练审核意见修改立论..." + "=" * 10 + "\n")
        time.sleep(0.5)
        notify("=" * 10 + "Stage 5: 基于示例优化写作风格..." + "=" * 10 + "\n")
        time.sleep(0.5)

        notify("正在准备反方框架及立论...\n")
        notify("=" * 10 + "Stage 1: 生成立论框架..." + "=" * 10 + "\n")
        time.sleep(0.5)
        notify("=" * 10 + "Stage 2: 基于立论框架搜寻所需资料..." + "=" * 10 + "\n")
        time.sleep(0.5)
        notify("=" * 10 + "Stage 3: 生成立论初稿..." + "=" * 10 + "\n")
        time.sleep(0.5)
        notify("=" * 10 + "Stage 4: 基于教练审核意见修改立论..." + "=" * 10 + "\n")
        time.sleep(0.5)
        notify("=" * 10 + "Stage 5: 基于示例优化写作风格..." + "=" * 10 + "\n")

        pro_statement = pickle.load(
            open(os.path.join(_demo_outputs_dir, "pro_statement.pkl"), "rb")
        )
        con_statement = pickle.load(
            open(os.path.join(_demo_outputs_dir, "con_statement.pkl"), "rb")
        )
        pro_outline = pickle.load(
            open(os.path.join(_demo_outputs_dir, "pro_outline.pkl"), "rb")
        )
        con_outline = pickle.load(
            open(os.path.join(_demo_outputs_dir, "con_outline.pkl"), "rb")
        )

    else:
        notify("正在准备正方框架及立论...\n")
        pro_statement, pro_outline = await generate_opening_statement(
            topic, "正方", llm_as_judge, status_cb
        )
        notify("正在准备反方框架及立论...\n")
        con_statement, con_outline = await generate_opening_statement(
            topic, "反方", llm_as_judge, status_cb
        )

    return {
        "pro_statement": pro_statement,
        "con_statement": con_statement,
        "pro_outline": pro_outline,
        "con_outline": con_outline,
    }


async def simulate_match(
    topic,
    pro_statement,
    pro_outline,
    con_statement,
    con_outline,
    status_cb=None,
    demo=app_config.is_demo,
):
    notify = mk_notify(status_cb)
    notify("开始比赛模拟\n")

    if demo:
        notify("正方二辩正在思考...\n")
        time.sleep(0.5)
        notify("反方二辩正在思考...\n")
        time.sleep(0.5)
        notify("正方三辩正在思考...\n")
        time.sleep(0.5)
        notify("反方三辩正在思考...\n")
        time.sleep(0.5)
        notify("反方四辩正在思考...\n")
        time.sleep(0.5)
        notify("正方四辩正在思考...\n")
        time.sleep(0.5)
        notify("评审正在打分...\n")
        time.sleep(0.5)

        pro_rebuttals = pickle.load(
            open(os.path.join(_demo_outputs_dir, "pro_rebuttals.pkl"), "rb")
        )
        con_rebuttals = pickle.load(
            open(os.path.join(_demo_outputs_dir, "con_rebuttals.pkl"), "rb")
        )
        pro_further_rebuttals = pickle.load(
            open(os.path.join(_demo_outputs_dir, "pro_further_rebuttals.pkl"), "rb")
        )
        con_further_rebuttals = pickle.load(
            open(os.path.join(_demo_outputs_dir, "con_further_rebuttals.pkl"), "rb")
        )
        con_conclusions = pickle.load(
            open(os.path.join(_demo_outputs_dir, "con_conclusions.pkl"), "rb")
        )
        pro_conclusions = pickle.load(
            open(os.path.join(_demo_outputs_dir, "pro_conclusions.pkl"), "rb")
        )
        judge_feedback = pickle.load(
            open(os.path.join(_demo_outputs_dir, "judge_feedback.pkl"), "rb")
        )

    else:
        notify("正方二辩正在思考...\n")
        pro_rebuttals = await generate_statement_rebuttal(
            con_statement, pro_statement, topic, "正方", pro_outline
        )
        notify("反方二辩正在思考...\n")
        con_rebuttals = await generate_statement_rebuttal(
            pro_statement, con_statement, topic, "反方", con_outline
        )

        debate_history = f"""
        正方一辩立论：
        {pro_statement}

        反方一辩立论：
        {con_statement}

        正方二辩反驳：
        {pro_rebuttals}

        反方二辩反驳：
        {con_rebuttals}
        """

        notify("正方三辩正在思考...\n")
        pro_further_rebuttals = generate_further_rebuttal(
            debate_history, topic, "正方", pro_outline
        )

        debate_history += f"""
        正方三辩陈词:
        {pro_further_rebuttals}
        """

        notify("反方三辩正在思考...\n")
        con_further_rebuttals = generate_further_rebuttal(
            debate_history, topic, "反方", con_outline
        )

        debate_history += f"""
        反方三辩陈词:
        {con_further_rebuttals}
        """

        notify("反方四辩正在思考...\n")
        con_conclusions = generate_conclusion(
            debate_history, topic, "反方", con_outline
        )

        debate_history += f"""
        反方四辩结辩:
        {con_conclusions}
        """

        notify("正方四辩正在思考...\n")
        pro_conclusions = generate_conclusion(
            debate_history, topic, "正方", pro_outline
        )

        debate_history += f"""
        正方四辩结辩:
        {pro_conclusions}
        """

        notify("评审正在打分...\n")
        judge_feedback = generate_judge_feedback(debate_history, topic)

    return {
        "正方一辩立论": pro_statement,
        "反方一辩立论": con_statement,
        "正方二辩反驳": pro_rebuttals,
        "反方二辩反驳": con_rebuttals,
        "正方三辩陈词": pro_further_rebuttals,
        "反方三辩陈词": con_further_rebuttals,
        "反方四辩结辩": con_conclusions,
        "正方四辩结辩": pro_conclusions,
        "评审意见": judge_feedback,
    }
