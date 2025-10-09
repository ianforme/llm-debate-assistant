from src.llm_debate_assistant.config import config
from openai import OpenAI
from typing import Callable, Optional


client = OpenAI(
    api_key=config.api_keys.openai_api_key,
    organization=config.api_keys.org_key,
    project=config.api_keys.project_key,
)


def rewrite_style(input_text, style_example):
    prompt = f"""
你是严格遵循风格示例写作的助理。你需要将输入内容按照风格示例的写作风格进行改写。

[写作任务]
- 严格模仿风格实例的写作风格
- 绝对不能对实际内容进行改变，你的任务仅仅是改写写作风格。维持原文的结构，内容，例子，论证等。
- 改写后的文章字数必须与原文相似
- 输出语言：中文（简体）

[风格示例]
{style_example}

[待改写的文章]
{input_text}
    """

    res = client.responses.create(
        model="gpt-5-mini-2025-08-07",
        input=prompt,
    )

    return res.output_text


def mk_notify(status_cb: Optional[Callable[[str], None]]):
    def _notify(msg: str):
        # 原样打印到控制台，便于本地调试
        print(msg)
        # 同步回 Streamlit UI
        if status_cb:
            try:
                status_cb(msg)
            except Exception:
                # 不让 UI 回调影响主流程
                pass

    return _notify
