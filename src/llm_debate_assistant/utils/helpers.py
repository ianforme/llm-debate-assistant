from typing import Callable, Optional

from llm_debate_assistant.core.client import get_client


def rewrite_style(input_text, style_example):
    prompt = f"""
你是严格遵循风格示例写作的助理。你需要将输入内容按照提供的语言风格和发言示例的写作风格进行改写。

【写作任务】
- 严格模仿发言示例的写作风格
- 绝对不能对原文中使用的定义，比较标准，论点，论证与论据进行改变，你的任务仅仅是改写写作风格，使这些内容的呈现方式更接近于提供的语言风格和发言示例
- 改写后的文章字数必须与原文相似
- 输出语言：中文（简体）

{style_example}

【待改写的文章】
{input_text}
    """

    client = get_client()
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
