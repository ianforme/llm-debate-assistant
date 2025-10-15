# app.py
import asyncio
import streamlit as st
import math
import html

from llm_debate_assistant.utils.style_cards import gemini_xbw_style, gemini_ys_style
from llm_debate_assistant.core.assistant import DebateAssistant
from llm_debate_assistant.core.realtime_assistant import RealtimeAssistant
from llm_debate_assistant.core.orchestrator import DebateOrchestrator

orchestrator = DebateOrchestrator(
    assistant=DebateAssistant(), 
    realtime_assistant=RealtimeAssistant()
)

st.set_page_config(page_title="AI辩论备赛与训练助手", page_icon="🤖", layout="wide")
st.title("🤖 AI辩论备赛与训练助手")
st.caption("*Powered by OpenAI GPT-5 & Realtime API*")

st.markdown("""
<style>
.chat-container {
    max-height: 600px;
    overflow-y: auto;
    padding: 12px;
    background-color: #f8f9fa;
    border-radius: 10px;
    border: 1px solid #ddd;
}

/* 行布局 */
.chat-row {
    display: flex;
    margin: 8px 0;
}

/* 左右对齐 */
.chat-left  { justify-content: flex-start; }
.chat-right { justify-content: flex-end; }

/* 气泡样式 */
.bubble {
    max-width: 75%;
    padding: 10px 14px;
    border-radius: 14px;
    line-height: 1.6;
    word-wrap: break-word;
    white-space: pre-wrap;
    box-shadow: 1px 1px 4px rgba(0,0,0,0.1);
    font-size: 15px;
}

/* AI（左侧） */
.bubble-left {
    background: #eef5ff;
    border: 1px solid #cfe0ff;
    border-top-left-radius: 4px;
}

/* 用户（右侧） */
.bubble-right {
    background: #fff0f0;
    border: 1px solid #ffd1d1;
    border-top-right-radius: 4px;
}
            
.comment-card {
    background-color: #ffffff;
    border-radius: 14px;
    box-shadow: 0px 2px 6px rgba(0,0,0,0.1);
    padding: 18px 20px;
    margin-bottom: 15px;
    transition: 0.3s;
    border-left: 6px solid #4a90e2;
}
.comment-card:hover {
    box-shadow: 0px 4px 12px rgba(0,0,0,0.15);
}
.score-badge {
    background-color: #4a90e2;
    color: white;
    font-weight: bold;
    padding: 6px 12px;
    border-radius: 20px;
    display: inline-block;
    font-size: 14px;
    margin-bottom: 8px;
}
.feedback-text {
    font-size: 15px;
    line-height: 1.6;
    color: #333;
}
</style>
""", unsafe_allow_html=True)


def _get(obj, key, default = ""):
    """同时兼容 pydantic model 与 dict 的字段访问。"""
    if obj is None:
        return default
    if hasattr(obj, key):
        return getattr(obj, key, default)
    if isinstance(obj, dict):
        return obj.get(key, default)
    return default

def _chunk_list(items, n_cols):
    if not items:
        return []
    n_rows = math.ceil(len(items) / n_cols)
    return [items[i * n_cols : (i + 1) * n_cols] for i in range(n_rows)]

def _esc(s):
    return html.escape("" if s is None else str(s))

def _status_box(title, key_):
    box = st.container(border=True)
    with box:
        st.markdown(f"**{title}**")
        area = st.empty()
        logs = st.session_state.get(key_, [])
        if logs:
            area.code("".join(logs[-200:]), language="text")
    return area

def _render_evidence_card(card):
    title = _get(card, "title", "未命名证据")
    url = _get(card, "url", "")
    keypoints = _get(card, "keypoints", [])
    original = _get(card, "original_text", [])

    # 用 container 包裹，给它加上 ev-card 的 class
    with st.container(border=True):

        if url:
            st.markdown(f"**[{_esc(title)}]({_esc(url)})**", unsafe_allow_html=True)
        else:
            st.markdown(f"**{_esc(title)}**", unsafe_allow_html=True)

        if keypoints:
            if isinstance(keypoints, str): 
                keypoints = [keypoints]
            st.markdown("\n".join([f"- {_esc(k)}" for k in keypoints if str(k).strip()]))

        if original:
            if isinstance(original, str): 
                original = [original]
            with st.expander("查看原文摘录"):
                st.markdown("\n".join([f"- {_esc(og)}" for og in original if str(og).strip()]))


def _render_outline_side(opening_statement_obj, outline_obj):
    with st.container(border=True):

        # # 准备大纲
        st.markdown("## 1. 框架")

        # ## 定义
        st.markdown("### 1.1 定义")
        defs = _get(outline_obj, "keywords_definition", [])
        if not defs:
            st.markdown("无定义")
        else:
            if isinstance(defs, dict): defs = [defs]
            for d in defs:
                st.markdown(f"- **{_esc(_get(d,'keyword'))}** — {_esc(_get(d,'definition'))}")

        # ## 比较标准
        st.markdown("### 1.2 比较标准")
        criterion = _get(outline_obj, "weighing_criterion", "")
        st.write(criterion if criterion else "无比较标准")

        # ## 论点
        st.markdown("### 1.3 论点")
        args = _get(outline_obj, "arguments", [])
        if not args:
            st.markdown("无论点")
        else:
            for idx, a in enumerate(args, 1):
                arg_title = _get(a, "argument", f"分论点 {idx}")
                with st.expander(f"论点{idx} · {arg_title}", expanded=False):
                    # 论证
                    warrant = _get(a, "warrant", "")
                    st.write(warrant if warrant else "无论证")

                    # 论据
                    with st.expander("#### 论据", expanded=False):
                        evds = _get(a, "evidences", [])
                        if not evds:
                            st.write("无论据")
                        else:
                            rows = _chunk_list(evds, 2)
                            for row in rows:
                                cols = st.columns(2)
                                for col, card in zip(cols, row):
                                    with col:
                                        _render_evidence_card(card)

    with st.container(border=True):    
        # # 立论
        st.markdown("## 2. 立论")
        opening_text = _get(opening_statement_obj, "opening_statement") or str(opening_statement_obj or "")
        st.write(opening_text if opening_text else "（无内容）")

def _render_chat_history(history):
    with st.container(border=True):  
        st.markdown("**对话历史**")
        # ===== 展示部分 =====
        for msg in history:
            msg = msg.strip()
            if msg.startswith("AI助手:"):
                text = html.escape(msg[6:])
                st.markdown(f"""
                <div class="chat-row chat-left">
                    <div class="bubble bubble-left">
                        🤖 <b>AI</b><br>{text}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            elif msg.startswith("用户:"):
                text = html.escape(msg[4:])
                st.markdown(f"""
                <div class="chat-row chat-right">
                    <div class="bubble bubble-right">
                        👤 <b>你</b><br>{text}
                    </div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

def _render_comments(comment):
    with st.container(border=True):  
        st.markdown("**AI教练打分**")
        st.markdown(f"""
        <div class="comment-card">
            <div class="score-badge">评分：{comment["score"]} / 100</div><br>
            <div class="feedback-text">{comment["feedback"]}</div>
        </div>
        """, unsafe_allow_html=True)
    
# Session State 初始化（保持原键名与逻辑）
if "opening_statement_logs" not in st.session_state:
    st.session_state.opening_statement_logs = []
if "oregon_interrogation_logs" not in st.session_state:
    st.session_state.oregon_interrogation_logs = []
if "interrogation_logs" not in st.session_state:
    st.session_state.interrogation_logs = []
if "interrogated_logs" not in st.session_state:
    st.session_state.interrogated_logs = []
if "crossfire_logs" not in st.session_state:
    st.session_state.crossfire_logs = []

# 页面结构：6个 Tab
prep_tab, evidence_search_tab, oregon_interrogation_tab, crossfire_table, interrogation_tab, interrogated_tab = st.tabs(["立论准备", "论据搜索", "奥瑞冈质询练习", "对辩练习", "质询练习", "被质询练习"])

with prep_tab:
    st.subheader("立论准备")
    
    with st.form("prep_form"):
        topic = st.text_input(
            '**辩题**',
            value="人工智能的广泛应用是/否会加剧教育不平等",
            placeholder="人工智能的广泛应用是/否会加剧教育不平等",
        )
        side = st.pills("持方", ['正方', '反方'], selection_mode="single")
        style_card = st.pills("语言风格", ['六侠-gemini', '小霸王-gemini'], selection_mode="single")
        if style_card == '六侠-gemini':
            style = gemini_ys_style
        else:
            style = gemini_xbw_style
        submitted = st.form_submit_button(f"搜寻相关论据，生成立论", type="primary")

    # 状态区（保持原逻辑）
    os_status_area = _status_box("立论准备状态", "opening_statement_logs")

    # 触发准备（保持原逻辑）
    if submitted:
        st.session_state.opening_statement_logs = []
        def _os_status_cb(msg):
            st.session_state.opening_statement_logs.append(str(msg))
            os_status_area.code("".join(st.session_state.opening_statement_logs.prep_logs[-200:]), language="text")

        with st.spinner(f"正在为{side}生成立论并检索证据……"):
            try:
                result = orchestrator.generate_opening_statement_sync(
                    topic=topic, 
                    side=side,
                    style_example=style,
                    status_cb=_os_status_cb
                )
                
                st.success("立论准备完成。")
                statement, outline = result["opening_statement"], result["outline"]
                _render_outline_side(statement, outline)

            except Exception as e:
                st.error("立论准备失败：" + str(e))

with evidence_search_tab:
    st.subheader("论据搜索")

    with st.container(border=True):  
        st.markdown("**所需论据背景**")
        with st.form("prep_form_2"):
            topic = st.text_input(
                '**辩题**',
                value="台湾应废除私人移工中介制度",
                placeholder="台湾应废除私人移工中介制度",
            )
            side = st.pills("**持方**", ['正方', '反方'], selection_mode="single")
            evidence_needed = st.text_area("**所需资料描述**")
            argument = st.text_area("**论点（选填）**")
            warrant = st.text_area("**论证（选填）**")            
            submitted = st.form_submit_button(f"**开始搜寻论据**", type="primary")

    if submitted:
        with st.spinner():
            result = orchestrator.assistant.search_for_evidence(
                topic=topic, 
                side=side,
                argument=argument,
                warrant=warrant,
                evidence_needed=evidence_needed,
            )
        
        rows = _chunk_list(result, 3)
        for row in rows:
            cols = st.columns(3)
            for col, card in zip(cols, row):
                with col:
                    _render_evidence_card(card)
        
with oregon_interrogation_tab:
    st.subheader("奥瑞冈质询练习")
    with st.container(border=True):  
        st.markdown("**质询环节设定**")
        with st.form("prep_form_3"):
            topic = st.text_input(
                '**辩题**',
                value="台湾应废除私人移工中介制度",
                placeholder="台湾应废除私人移工中介制度",
            )
            assistant_side = st.pills("**AI持方**", ['正方', '反方'], selection_mode="single")
            assistant_statement = st.text_area("**AI立论**")
            assistant_examples = st.text_area("**AI使用的论据**")
            user_time_in_seconds = st.number_input("**用户发言时间（30-240秒)**", min_value=30, max_value=240)
            submitted = st.form_submit_button(f"**开始质询AI**", type="primary")

    if submitted:
        st.session_state.oregon_interrogation_logs = []
        def _ore_interrogation_status_cb(msg):
            st.session_state.oregon_interrogation_logs.append(str(msg))
            ore_interrogation_status_area.code("".join(st.session_state.oregon_interrogation_logs[-200:]), language="text")
        ore_interrogation_status_area = _status_box("奥瑞冈质询状态", "oregon_interrogation_logs")
        
        with st.spinner():
            result = orchestrator.oregon_interrogation_practice(
                topic=topic, 
                assistant_side=assistant_side,
                assistant_statement=assistant_statement,
                assistant_examples=assistant_examples,
                user_time_in_seconds=user_time_in_seconds,
                status_cb=_ore_interrogation_status_cb
            )

        judge_feedback, speech_history = result["judge_feedback"], result["speech_history"]

        _render_chat_history(speech_history)
        _render_comments(judge_feedback)


with crossfire_table:
    st.subheader("对辩练习")
    with st.container(border=True):  
        st.markdown("**对辩环节设定**")
        with st.form("prep_form_4"):
            topic = st.text_input(
                '**辩题**',
                value="台湾应废除私人移工中介制度",
                placeholder="台湾应废除私人移工中介制度",
            )
            assistant_side = st.pills("**AI持方**", ['正方', '反方'], selection_mode="single")
            assistant_statement = st.text_area("**AI立论**")
            proposed_attacks = st.text_area("**AI对辩战场 - 选填，设定过后AI将大概率使用这些战场/例子进行对辩**")
            human_statement = st.text_area("**用户立论**")
            user_time_in_seconds = st.number_input("**用户发言时间（30-240秒)**", min_value=30, max_value=240)
            submitted = st.form_submit_button(f"**开始对辩**", type="primary")

    if submitted:
        st.session_state.crossfire_logs = []
        def _crossfire_status_cb(msg):
            st.session_state.crossfire_logs.append(str(msg))
            crossfire_status_area.code("".join(st.session_state.crossfire_logs[-200:]), language="text")
        crossfire_status_area = _status_box("对辩状态", "crossfire_logs")
        
        with st.spinner():
            result = orchestrator.rebuttal_crossfire_practice(
                topic=topic, 
                assistant_side=assistant_side,
                assistant_statement=assistant_statement,
                human_statement=human_statement,
                user_time_in_seconds=user_time_in_seconds,
                status_cb=_crossfire_status_cb,
                proposed_attacks=proposed_attacks
            )

        judge_feedback, speech_history = result["judge_feedback"], result["speech_history"]

        _render_chat_history(speech_history)
        _render_comments(judge_feedback)


with interrogation_tab:
    st.subheader("质询练习")
    with st.container(border=True):  
        st.markdown("**质询环节设定**")
        with st.form("prep_form_5"):
            topic = st.text_input(
                '**辩题**',
                value="台湾应废除私人移工中介制度",
                placeholder="台湾应废除私人移工中介制度",
            )
            assistant_side = st.pills("**AI持方**", ['正方', '反方'], selection_mode="single")
            assistant_statement = st.text_area("**AI立论**")
            user_time_in_seconds = st.number_input("**用户发言时间（30-240秒)**", min_value=30, max_value=240)
            submitted = st.form_submit_button(f"**开始质询AI**", type="primary")

    if submitted:
        st.session_state.interrogation_logs = []
        def _interrogation_status_cb(msg):
            st.session_state.interrogation_logs.append(str(msg))
            interrogation_status_area.code("".join(st.session_state.interrogation_logs[-200:]), language="text")
        interrogation_status_area = _status_box("质询状态", "interrogation_logs")
        
        with st.spinner():
            result = orchestrator.rebuttal_interrogation_practice(
                topic=topic, 
                assistant_side=assistant_side,
                assistant_statement=assistant_statement,
                user_time_in_seconds=user_time_in_seconds,
                status_cb=_interrogation_status_cb
            )

        judge_feedback, speech_history = result["judge_feedback"], result["speech_history"]

        _render_chat_history(speech_history)
        _render_comments(judge_feedback)


with interrogated_tab:
    st.subheader("接质询练习")
    with st.container(border=True):  
        st.markdown("**质询环节设定**")
        with st.form("prep_form_6"):
            topic = st.text_input(
                '**辩题**',
                value="台湾应废除私人移工中介制度",
                placeholder="台湾应废除私人移工中介制度",
            )
            assistant_side = st.pills("**AI持方**", ['正方', '反方'], selection_mode="single")
            assistant_statement = st.text_area("**AI立论**")
            proposed_attacks = st.text_area("**AI质询战场 - 选填，设定过后AI将大概率使用这些战场/例子进行质询**")
            human_statement = st.text_area("**用户立论**")
            user_time_in_seconds = st.number_input("**用户发言时间（30-240秒)**", min_value=30, max_value=240)
            submitted = st.form_submit_button(f"**开始接AI质询**", type="primary")

    if submitted:
        st.session_state.interrogated_logs = []
        def _interrogated_status_cb(msg):
            st.session_state.interrogated_logs.append(str(msg))
            interrogated_status_area.code("".join(st.session_state.interrogated_logs[-200:]), language="text")
        interrogated_status_area = _status_box("接质询状态", "interrogated_logs")
        
        with st.spinner():
            result = orchestrator.rebuttal_interrogated_practice(
                topic=topic, 
                assistant_side=assistant_side,
                assistant_statement=assistant_statement,
                human_statement=human_statement,
                user_time_in_seconds=user_time_in_seconds,
                status_cb=_interrogated_status_cb,
                proposed_attacks=proposed_attacks
            )

        judge_feedback, speech_history = result["judge_feedback"], result["speech_history"]

        _render_chat_history(speech_history)
        _render_comments(judge_feedback)
        