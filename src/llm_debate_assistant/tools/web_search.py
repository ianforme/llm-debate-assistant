import asyncio
from dataclasses import dataclass
from typing import Any

from google.genai import types

from llm_debate_assistant.core.client import get_async_gemini_client, get_gemini_client

# ==============================================================
# Constants and Prompts
# ==============================================================

EVIDENCE_SEARCH_PROMPT_TEMPLATE = """
你的任务是根据论点、论证和论据，寻找相关的支持性证据。

辩题：{topic}
立场：{side}

论点：{argument}
论证：{warrant}
所需证据类型：
{evidence_list}

请找到：
1. 优先使用一手来源：政府、国际组织、同行评审论文、权威数据集、可信媒体等
2. 优先选择具体证据：真实案例、统计数据、研究报告、法律法规等
3. 所有证据必须支持论点和论证理由

对于每个找到的证据，请提供：
- 标题
- 来源链接
- 关键要点（支持论点的具体内容）
- 相关原文摘录

请使用证据的原始语言输出，不要翻译。
"""


# ==============================================================
# Data Models
# ==============================================================


@dataclass(frozen=True)
class ArgumentEvidence:
    """Container for argument and its evidence search results."""

    argument: str
    warrant: str
    results: dict
    error: str | None = None


# ==============================================================
# Core Search Functions
# ==============================================================


def _extract_search_metadata(response: Any) -> dict[str, Any]:
    """private method to extract search metadata from Gemini response.

    Args:
        response (Any): The Gemini API response object

    Returns:
        dict[str, Any]: A dictionary containing extracted metadata and search results
    """
    result = {
        "text": response.text if response.text else "",
        "grounding_metadata": None,
        "search_results": [],
    }

    if not (hasattr(response, "candidates") and response.candidates):
        return result

    candidate = response.candidates[0]
    if not hasattr(candidate, "grounding_metadata"):
        return result

    grounding_metadata = candidate.grounding_metadata
    result["grounding_metadata"] = grounding_metadata

    # Extract search entry point
    if (
        hasattr(grounding_metadata, "search_entry_point")
        and grounding_metadata.search_entry_point is not None
    ):
        result["search_entry_point"] = (
            grounding_metadata.search_entry_point.rendered_content
        )

    # Extract grounding chunks (search results)
    if hasattr(grounding_metadata, "grounding_chunks"):
        for chunk in grounding_metadata.grounding_chunks:
            if hasattr(chunk, "web"):
                result["search_results"].append(
                    {"uri": chunk.web.uri, "title": chunk.web.title or ""}
                )

    return result


def search_web(query: str, model: str = "gemini-2.5-pro") -> dict[str, Any]:
    """Search the web using Gemini with Google Search grounding.

    Args:
        query (str): The search query
        model (str, optional): The Gemini model to use. Defaults to "gemini-2.5-pro".

    Returns:
        dict[str, Any]: A dictionary containing search results and metadata
    """
    # lazy load client
    # Use shared client from centralized client module
    # otherwise, it might create multiple instances that
    # could lead to rate limiting issues or connecting limit issues
    client = get_gemini_client()

    # check out https://googleapis.github.io/python-genai/genai.html#genai.types.Tool
    google_search_tool = types.Tool(google_search=types.GoogleSearch())

    response = client.models.generate_content(
        model=model,
        contents=query,
        config=types.GenerateContentConfig(
            tools=[google_search_tool],
            response_modalities=["TEXT"],
            temperature=0.0,
            thinking_config=types.ThinkingConfig(
                include_thoughts=False,
                thinking_budget=-1,
            ),
        ),
    )

    return _extract_search_metadata(response)


async def async_search_web(query: str, model: str = "gemini-2.5-pro") -> dict[str, Any]:
    """Search the web using Gemini with Google Search grounding.

    Args:
        query (str): The search query
        model (str, optional): The Gemini model to use. Defaults to "gemini-2.5-pro".

    Returns:
        dict[str, Any]: A dictionary containing search results and metadata
    """
    # Use shared async client from centralized client module
    async_client = get_async_gemini_client()

    google_search_tool = types.Tool(google_search=types.GoogleSearch())

    response = await async_client.aio.models.generate_content(
        model=model,
        contents=query,
        config=types.GenerateContentConfig(
            tools=[google_search_tool],
            response_modalities=["TEXT"],
            temperature=0.0,
            thinking_config=types.ThinkingConfig(
                include_thoughts=False,
                thinking_budget=-1,
            ),
        ),
    )

    return _extract_search_metadata(response)


# ==============================================================
# Evidence Search Functions
# ==============================================================


def _build_evidence_query(
    argument: str,
    warrant: str,
    evidence_needed: list[str],
    topic: str,
    side: str,
) -> str:
    """private method for building evidence search queries.

    Args:
        argument (str): The main argument/claim
        warrant (str): The reasoning supporting the argument
        evidence_needed (list[str]): List of specific evidence types needed
        topic (str): The debate topic
        side (str): Which side of the debate

    Returns:
        str: The formatted search query string
    """
    evidence_list = "\n".join(f"- {e}" for e in evidence_needed)
    return EVIDENCE_SEARCH_PROMPT_TEMPLATE.format(
        topic=topic,
        side=side,
        argument=argument,
        warrant=warrant,
        evidence_list=evidence_list,
    )


# ==============================================================
# Function tool for evidence search in debates
# ==============================================================


# Synchronous version
def search_for_evidence(
    argument: str,
    warrant: str,
    evidence_needed: list[str],
    topic: str,
    side: str,
    model: str = "gemini-2.5-pro",
) -> dict:
    """Search for evidence to support a debate argument.

    Args:
        argument (str): The main argument/claim
        warrant (str): The reasoning supporting the argument
        evidence_needed (list[str]): List of specific evidence types needed
        topic (str): The debate topic
        side (str): Which side of the debate
        model (str, optional): The Gemini model to use. Defaults to "gemini-2.5-pro".

    Returns:
        dict: A dictionary containing search results and generated evidence analysis
    """
    query = _build_evidence_query(argument, warrant, evidence_needed, topic, side)
    return search_web(query, model=model)


# Asynchronous version
async def async_search_for_evidence(
    argument: str,
    warrant: str,
    evidence_needed: list[str],
    topic: str,
    side: str,
    model: str = "gemini-2.5-pro",
) -> dict:
    """Search for evidence to support a debate argument asynchronously.

    Args:
        argument (str): The main argument/claim
        warrant (str): The reasoning supporting the argument
        evidence_needed (list[str]): List of specific evidence types needed
        topic (str): The debate topic
        side (str): Which side of the debate
        model (str, optional): The Gemini model to use. Defaults to "gemini-2.5-pro".

    Returns:
        dict: A dictionary containing search results and generated evidence analysis
    """
    query = _build_evidence_query(argument, warrant, evidence_needed, topic, side)
    return await async_search_web(query, model=model)


# Thread-based async version (more reliable for concurrent requests)
async def async_search_for_evidence_threaded(
    argument: str,
    warrant: str,
    evidence_needed: list[str],
    topic: str,
    side: str,
    model: str = "gemini-2.5-pro",
) -> dict:
    """Search for evidence using thread-based async.

    More reliable for concurrent requests than native async.
    This version runs the synchronous search in a thread pool to avoid
    blocking.
    Use this if async_search_for_evidence has issues with concurrent requests.

    Args:
        argument (str): The main argument/claim
        warrant (str): The reasoning supporting the argument
        evidence_needed (list[str]): List of specific evidence types needed
        topic (str): The debate topic
        side (str): Which side of the debate
        model (str, optional): The Gemini model to use. Defaults to "gemini-2.5-pro".

    Returns:
        dict: A dictionary containing search results and generated evidence analysis
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        search_for_evidence,
        argument,
        warrant,
        evidence_needed,
        topic,
        side,
        model,
    )


# ==============================================================
# Multiple Argument Search Functions
# ==============================================================


async def search_multiple_arguments(
    arguments: list[tuple[str, str, list[str]]],
    topic: str,
    side: str,
    model: str = "gemini-2.5-pro",
    use_threaded: bool = False,
) -> list[ArgumentEvidence]:
    """Search for evidence for multiple arguments concurrently.

    Args:
        arguments (list[tuple[str, str, list[str]]]): A list of tuples
            containing the argument, warrant, and evidence needed.
        topic (str): The debate topic.
        side (str): Which side of the debate.
        model (str, optional): The Gemini model to use.
            Defaults to "gemini-2.5-pro".
        use_threaded (bool, optional): If True, uses thread-based async.
            Defaults to False.

    Returns:
        list[ArgumentEvidence]: A list of ArgumentEvidence objects containing
            results for each argument.
    """
    # Choose async implementation based on use_threaded flag
    search_func = (
        async_search_for_evidence_threaded
        if use_threaded
        else async_search_for_evidence
    )

    # Create all search tasks
    tasks = [
        search_func(
            argument=argument,
            warrant=warrant,
            evidence_needed=evidence_needed,
            topic=topic,
            side=side,
            model=model,
        )
        for argument, warrant, evidence_needed in arguments
    ]

    # Execute all searches concurrently
    gathered = await asyncio.gather(*tasks, return_exceptions=True)

    # Package results
    results = []
    for (argument, warrant, _), result in zip(arguments, gathered):
        if isinstance(result, Exception):
            results.append(
                ArgumentEvidence(
                    argument=argument,
                    warrant=warrant,
                    results={},
                    error=str(result),
                )
            )
        else:
            results.append(
                ArgumentEvidence(
                    argument=argument,
                    warrant=warrant,
                    results=result,
                )
            )

    return results


# Thread-based version (legacy)
async def search_multiple_arguments_threaded(
    arguments: list[tuple[str, str, list[str]]],
    topic: str,
    side: str,
    model: str = "gemini-2.5-pro",
) -> list[ArgumentEvidence]:
    """Search for evidence for multiple arguments using thread-based async.

    This is a convenience wrapper that always uses the thread-based
    implementation.

    Args:
        arguments (list[tuple[str, str, list[str]]]): A list of tuples
            containing the argument, warrant, and evidence needed.
        topic (str): The debate topic.
        side (str): Which side of the debate.
        model (str, optional): The Gemini model to use.
            Defaults to "gemini-2.5-pro".

    Returns:
        list[ArgumentEvidence]: A list of ArgumentEvidence objects containing
            results for each argument.
    """
    return await search_multiple_arguments(
        arguments, topic, side, model, use_threaded=True
    )
