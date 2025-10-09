# app.py
import asyncio
from typing import Any, Dict, List, Tuple
import math
import html
import streamlit as st

# =====================
# 基础设置
# =====================
st.set_page_config(page_title="竞技辩论助手", page_icon="🤖", layout="wide")
st.title("🤖 大语言模型辩论备赛助手")
st.caption("准备正反双方的框架，立论，自动搜寻论据，并一键模拟整场比赛（含评审裁决）。")
st.caption("*Powered by OpenAI GPT 5*")

st.markdown("""
<style>
  /* 聊天气泡布局 */
  .chat-row { display: flex; margin: 10px 0; }
  .chat-left  { justify-content: flex-start; }
  .chat-right { justify-content: flex-end; }

  .bubble {
    max-width: 80%;
    padding: 10px 14px;
    border-radius: 14px;
    line-height: 1.6;
    word-wrap: break-word;
    white-space: pre-wrap;
  }
  /* 左(正方) / 右(反方) 颜色与边框 */
  .bubble-left  { background:#eef5ff; border:1px solid #cfe0ff; border-top-left-radius: 4px; }
  .bubble-right { background:#ffeef0; border:1px solid #ffd0d6; border-top-right-radius: 4px; }

  /* 环节名 */
  .stage {
    font-size: 12px;
    opacity: .75;
    margin-bottom: 4px;
  }
</style>
""", unsafe_allow_html=True)


# =====================
# 业务函数导入（保持原逻辑）
# =====================
try:
    from main import match_preparation, simulate_match  # async functions
except Exception as e:
    st.error(
        "Failed to import functions from main.py. Make sure app.py is in the same folder as main.py.\n\nError: "
        + str(e)
    )
    st.stop()

# =====================
# 工具函数（仅UI/健壮性相关）
# =====================
def _get(obj: Any, key: str, default: Any = ""):
    """同时兼容 pydantic model 与 dict 的字段访问。"""
    if obj is None:
        return default
    if hasattr(obj, key):
        return getattr(obj, key, default)
    if isinstance(obj, dict):
        return obj.get(key, default)
    return default

def _chunk_list(items: List[Any], n_cols: int) -> List[List[Any]]:
    if not items:
        return []
    n_rows = math.ceil(len(items) / n_cols)
    return [items[i * n_cols : (i + 1) * n_cols] for i in range(n_rows)]

def _esc(s: Any) -> str:
    return html.escape("" if s is None else str(s))

def _p(s: Any) -> str:
    """段落HTML（保留换行）"""
    text = _esc(s)
    return f"<p>{text.replace('\\n', '<br/>')}</p>" if text else "<span class='tiny-muted'>（无内容）</span>"

def _ul(items: List[str]) -> str:
    safe = [f"<li>{_esc(x)}</li>" for x in items if str(x).strip()]
    return f"<ul>{''.join(safe)}</ul>" if safe else "<span class='tiny-muted'>（无数据）</span>"

def _status_box(title: str, key_: str):
    box = st.container(border=True)
    with box:
        st.markdown(f"**{title}**")
        area = st.empty()
        logs = st.session_state.get(key_, [])
        if logs:
            area.code("".join(logs[-200:]), language="text")
    return area

def _render_evidence_card(card: Any):
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
            st.markdown("\n".join([f"- {k}" for k in keypoints if str(k).strip()]))

        if original:
            if isinstance(original, str): 
                original = [original]
            with st.expander("查看原文摘录"):
                st.markdown("\n".join([f"- {og}" for og in original if str(og).strip()]))


def _render_outline_side(side_name: str, opening_statement_obj: Any, outline_obj: Any):
    """整体区域用统一背景色，内部保持层级结构"""
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


# =====================
# Session State 初始化（保持原键名与逻辑）
# =====================
if "prep" not in st.session_state:
    st.session_state.prep = None
if "topic" not in st.session_state:
    st.session_state.topic = ""
if "llm_as_judge" not in st.session_state:
    st.session_state.llm_as_judge = True
if "match_result" not in st.session_state:
    st.session_state.match_result = None
if "prep_logs" not in st.session_state:
    st.session_state.prep_logs = []
if "sim_logs" not in st.session_state:
    st.session_state.sim_logs = []
if "stage_ptr" not in st.session_state:
    st.session_state.stage_ptr = -1  # -1 表示尚未开始展示

# =====================
# 页面结构：两个 Tab
# =====================
prep_tab, simulate_tab = st.tabs(["赛前准备", "比赛模拟"])

# =====================
# Tab1：赛前准备（严格对齐）
# =====================
with prep_tab:
    st.subheader("赛前准备")

    with st.form("prep_form"):
        topic = st.text_input(
            '**辩题**',
            value=st.session_state.topic,
            placeholder="应该/不应该废除死刑",
        )
        llm_as_judge = st.checkbox("赛前阶段启用LLM教练修改立论", value=st.session_state.llm_as_judge)
        submitted = st.form_submit_button("生成双方立论与证据", type="primary")

    # 状态区（保持原逻辑）
    prep_status_area = _status_box("赛前准备状态", "prep_logs")

    def _prep_status_cb(msg: str):
        st.session_state.prep_logs.append(str(msg))
        prep_status_area.code("".join(st.session_state.prep_logs[-200:]), language="text")

    # 触发准备（保持原逻辑）
    if submitted:
        if not topic.strip():
            st.warning("请先填写辩题。")
        else:
            st.session_state.topic = topic.strip()
            st.session_state.llm_as_judge = bool(llm_as_judge)
            st.session_state.prep = None
            st.session_state.match_result = None
            st.session_state.prep_logs = []
            st.session_state.stage_ptr = -1  # 重置第二页进度

            with st.spinner("正在为正反双方生成立论并检索证据……"):
                try:
                    result = asyncio.run(
                        match_preparation(
                            st.session_state.topic,
                            st.session_state.llm_as_judge,
                            status_cb=_prep_status_cb,
                        )
                    )
                except TypeError:
                    result = asyncio.run(
                        match_preparation(st.session_state.topic, st.session_state.llm_as_judge)
                    )
                except RuntimeError:
                    try:
                        result = asyncio.get_event_loop().run_until_complete(
                            match_preparation(
                                st.session_state.topic,
                                st.session_state.llm_as_judge,
                                status_cb=_prep_status_cb,  # type: ignore
                            )
                        )
                    except TypeError:
                        result = asyncio.get_event_loop().run_until_complete(
                            match_preparation(
                                st.session_state.topic,
                                st.session_state.llm_as_judge,
                            )
                        )
                except Exception as e:
                    st.error("match_preparation 失败：" + str(e))
                    result = None

            st.session_state.prep = result

    # ===== 展示准备结果（对齐结构） =====
    if st.session_state.prep:
        st.success("赛前准备完成。")

        prep_res: Dict[str, Any] = st.session_state.prep
        pro_statement = prep_res.get("正方一辩立论")
        con_statement = prep_res.get("反方一辩立论")
        pro_outline = prep_res.get("正方立论框架")
        con_outline = prep_res.get("反方立论框架")

        # 改成子 tabs
        subtab_pro, subtab_con = st.tabs(["正方", "反方"])
        with subtab_pro:
            _render_outline_side("正方", pro_statement, pro_outline)
        with subtab_con:
            _render_outline_side("反方", con_statement, con_outline)

# =====================
# Tab2：比赛模拟（保留你已有的“对话气泡 + 下一环节”逻辑与UI）
# =====================
with simulate_tab:
    st.subheader("比赛模拟")

    if not st.session_state.prep:
        st.info("请先在“赛前准备”页完成准备后再进行模拟。")
        st.stop()

    st.markdown(f"**辩题：** {st.session_state.topic}")

    # 状态区（保持原逻辑）
    sim_status_area = _status_box("比赛模拟状态", "sim_logs")

    def _sim_status_cb(msg: str):
        st.session_state.sim_logs.append(str(msg))
        sim_status_area.code("".join(st.session_state.sim_logs[-200:]), language="text")

    # 触发模拟（保持原逻辑）
    if st.button("开始模拟", type="primary"):
        st.session_state.match_result = None
        st.session_state.sim_logs = []
        st.session_state.stage_ptr = -1

        prep_res: Dict[str, Any] = st.session_state.prep
        topic = st.session_state.topic
        pro_statement = prep_res.get("正方一辩立论")
        con_statement = prep_res.get("反方一辩立论")
        pro_outline = prep_res.get("正方立论框架")
        con_outline = prep_res.get("反方立论框架")

        with st.spinner("正在模拟赛程：立论、反驳、陈词与评审裁决……"):
            try:
                sim = asyncio.run(
                    simulate_match(
                        topic,
                        pro_statement,
                        pro_outline,
                        con_statement,
                        con_outline,
                        status_cb=_sim_status_cb,
                    )
                )
            except TypeError:
                sim = asyncio.run(
                    simulate_match(
                        topic,
                        pro_statement,
                        pro_outline,
                        con_statement,
                        con_outline,
                    )
                )
            except RuntimeError:
                try:
                    sim = asyncio.get_event_loop().run_until_complete(
                        simulate_match(
                            topic,
                            pro_statement,
                            pro_outline,
                            con_statement,
                            con_outline,
                            status_cb=_sim_status_cb,
                        )
                    )
                except TypeError:
                    sim = asyncio.get_event_loop().run_until_complete(
                        simulate_match(
                            topic,
                            pro_statement,
                            pro_outline,
                            con_statement,
                            con_outline,
                        )
                    )
            except Exception as e:
                st.error("simulate_match 失败：" + str(e))
                sim = None

        st.session_state.match_result = sim

    # 对话气泡 + 「下一环节」
    if st.session_state.match_result:
        sim: Dict[str, Any] = st.session_state.match_result
        st.success("模拟完成。以下为比赛记录：")

        stage_order: List[Tuple[str, str]] = [
            ("正方一辩立论", str(_get(sim, "正方一辩立论", sim.get("正方一辩立论", "")))),
            ("反方一辩立论", str(_get(sim, "反方一辩立论", sim.get("反方一辩立论", "")))),
            ("正方二辩反驳", str(sim.get("正方二辩反驳", ""))),
            ("反方二辩反驳", str(sim.get("反方二辩反驳", ""))),
            ("正方三辩陈词", str(sim.get("正方三辩陈词", ""))),
            ("反方三辩陈词", str(sim.get("反方三辩陈词", ""))),
            ("反方四辩结辩", str(sim.get("反方四辩结辩", ""))),
            ("正方四辩结辩", str(sim.get("正方四辩结辩", ""))),
            ("评审意见", ""),
        ]

        # 初始化指针
        if st.session_state.stage_ptr == -1 and stage_order:
            st.session_state.stage_ptr = 0

        max_idx = min(st.session_state.stage_ptr, len(stage_order) - 1)

        # --- 聊天容器在上，按钮容器在最下 ---
        chat_container = st.container()
        with chat_container:

            def chat(role: str, stage_name: str, text: str):
                # 判断左右
                is_pro = role.startswith("正方")
                is_con = role.startswith("反方")

                if role == "评审意见":
                    # 裁判：居中用系统消息风格（沿用原有 st.chat_message 以保留你的样式）
                    with st.chat_message(name="评审", avatar="⚖️"):
                        st.subheader("最终裁决与评语")
                        st.write(text if text else "（无可显示内容）")
                    return

                row_class = "chat-row " + ("chat-left" if is_pro else "chat-right")
                bubble_class = "bubble " + ("bubble-left" if is_pro else "bubble-right")
                # 逃逸文本
                safe_text = _esc(text) if text else "（无可显示内容）"

                st.markdown(
                    f"""
                    <div class="{row_class}">
                      <div class="{bubble_class}">
                        <div><b>【{_esc(stage_name)}】</b></div>
                        <div>{safe_text}</div>
                      </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # 逐条渲染到当前指针
            for i in range(0, max_idx + 1):
                stage_name, text = stage_order[i]
                if stage_name != "评审意见":
                    # 用环节名称判断角色（用于左右对齐）
                    role = "正方" if stage_name.startswith("正方") else ("反方" if stage_name.startswith("反方") else "其他")
                    chat(role, stage_name, text)
                else:
                    judge_fb = sim.get("评审意见")
                    # 评审内容继续保留你原先的结构
                    with st.chat_message(name="评审", avatar="⚖️"):
                        st.subheader("最终裁决与评语")
                        st.write(str(_get(judge_fb, "feedback", judge_fb)))
                        pro_score = _get(judge_fb, "pro_score", None)
                        con_score = _get(judge_fb, "con_score", None)
                        m1, m2 = st.columns(2)
                        with m1:
                            st.metric("正方得分", pro_score if pro_score is not None else "-")
                        with m2:
                            st.metric("反方得分", con_score if con_score is not None else "-")

        # --- 控制按钮固定在最下方 ---
        btn_container = st.container()
        with btn_container:
            st.divider()
            bc1, bc2, bc3 = st.columns(3)
            with bc1:
                if st.button("🔁 重置到开头"):
                    st.session_state.stage_ptr = -1
            with bc2:
                if st.button("📜 一次展示全部"):
                    st.session_state.stage_ptr = len(stage_order) - 1
            with bc3:
                if st.button("➡️ 下一环节"):
                    if st.session_state.stage_ptr < len(stage_order) - 1:
                        st.session_state.stage_ptr += 1

