import asyncio
import logging
from google.genai import types
from typing import Any, List, Optional
from dataclasses import dataclass, field
from llm_debate_assistant.core.client import get_async_gemini_client

logger = logging.getLogger(__name__)

# ==============================================================
# Data Models
# ==============================================================


@dataclass
class SearchSource:
    """Represents a single source citation from Google Search."""

    uri: str
    title: str


@dataclass
class SearchResult:
    """Container for the result of a single search query."""

    query: str
    content: str  # The summarized answer/text from Gemini
    sources: List[SearchSource] = field(default_factory=list)
    raw_metadata: Optional[Any] = None
    error: Optional[str] = None


# ==============================================================
# Helper Functions (Extraction Logic)
# ==============================================================


def _extract_search_data(response: Any, query: str) -> SearchResult:
    """Parses the raw Gemini response to extract text content and grounding sources.

    Args:
        response (Any): Raw response from Gemini API
        query (str): The search query string

    Returns:
        SearchResult: Parsed search result containing content and sources
    """
    result = SearchResult(query=query, content="")

    if not (hasattr(response, "candidates") and response.candidates):
        return result

    candidate = response.candidates[0]

    # 1. Extract Text Content
    # Note: When API returns TOO_MANY_TOOL_CALLS or other edge cases,
    # parts might be None instead of a list, so we must check explicitly
    text_parts = []
    if (
        hasattr(candidate, "content")
        and hasattr(candidate.content, "parts")
        and candidate.content.parts is not None
    ):
        for part in candidate.content.parts:
            if hasattr(part, "text") and part.text:
                text_parts.append(part.text)
    result.content = "".join(text_parts)

    # 2. Extract Grounding Metadata (Sources)
    if hasattr(candidate, "grounding_metadata"):
        gm = candidate.grounding_metadata
        result.raw_metadata = gm

        # Must check if gm.grounding_chunks is not None before iterating
        if hasattr(gm, "grounding_chunks") and gm.grounding_chunks is not None:
            for chunk in gm.grounding_chunks:
                # Check for 'web' attribute which contains the source info
                if hasattr(chunk, "web"):
                    result.sources.append(
                        SearchSource(
                            uri=getattr(chunk.web, "uri", ""),
                            title=getattr(chunk.web, "title", "Unknown Source"),
                        )
                    )

    return result


# ==============================================================
# Core Search Functions
# ==============================================================


async def search_single_query(
    query: str,
    model: str = "gemini-2.5-flash",
) -> SearchResult:
    """Executes a single search query using Gemini's Google Search Grounding.

    Args:
        query (str): The search query string
        model (str, optional): The Gemini model to use. Defaults to "gemini-2.5-flash".

    Returns:
        SearchResult: The parsed search result containing content and sources
    """
    client = get_async_gemini_client()

    prompt_content = (
        f"Please search Google for the following query: '{query}'. "
        f"Summarize the key facts found. If specific data or cases are found, cite them."
    )

    google_search_tool = types.Tool(google_search=types.GoogleSearch())

    try:
        response = await client.aio.models.generate_content(
            model=model,
            contents=prompt_content,
            config=types.GenerateContentConfig(
                tools=[google_search_tool],
                response_modalities=["TEXT"],
                temperature=0.0,  # Fact-based, keep it strict
            ),
        )
        return _extract_search_data(response, query)

    except Exception as e:
        logger.error(f"Search failed for query '{query}': {str(e)}")
        return SearchResult(query=query, content="", error=str(e))


async def search_queries(queries: List[str], model: str = "gemini-2.5-flash") -> List[SearchResult]:
    """Executes a list of search queries concurrently

    Args:
        queries (List[str]): List of search query strings
        model (str, optional): The Gemini model to use. Defaults to "gemini-2.5-flash".

    Returns:
        List[SearchResult]: List of parsed search results containing content and sources
    """
    if not queries:
        return []

    logger.info(f"Starting batch search for {len(queries)} queries...")

    # Create tasks for all queries
    tasks = [search_single_query(q, model=model) for q in queries]

    # Run concurrently
    results = await asyncio.gather(*tasks)

    logger.info(f"Completed batch search. Success: {sum(1 for r in results if not r.error)}")
    return list(results)
