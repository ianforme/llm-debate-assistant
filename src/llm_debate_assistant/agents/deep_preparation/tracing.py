"""
Tracing and logging utilities for agent workflow monitoring.

This module provides:
- Shared Rich Console instance for consistent output
- Accurate token counting using tiktoken
- Extracting token usage from API responses (OpenAI/Gemini)
- Tracking cumulative token usage across a workflow
- Logging message statistics (counts, chars, tokens)
- Formatting verbose output for debugging

Traces workflow execution, resource consumption, and message flow.
"""

from typing import Optional

import tiktoken
from tiktoken import Encoding
from rich.console import Console

# ============================================================================
# Shared Console Instance
# ============================================================================

# Global console object for consistent output across all modules
console = Console()

# ============================================================================
# Token Tracking
# ============================================================================

# Initialize tiktoken encoder for accurate token counting
# cl100k_base works well for GPT-4, Claude, and similar models
_tiktoken_encoder: Optional[Encoding] = None
try:
    _tiktoken_encoder = tiktoken.get_encoding("cl100k_base")
except Exception:
    pass

# Cumulative token tracking for workflow
_token_usage = {
    "total_input_tokens": 0,
    "total_output_tokens": 0,
    "call_count": 0,
}


def count_tokens(text: str) -> int:
    """Count tokens in text using tiktoken.

    Args:
        text (str): Text to count tokens for

    Returns:
        int: Token count (uses tiktoken if available, falls back to char estimate)
    """
    if _tiktoken_encoder and text:
        return len(_tiktoken_encoder.encode(text))
    # Fallback: rough estimate (1 token ≈ 2 chars for Chinese content)
    return len(text) // 2 if text else 0


def extract_token_usage(response) -> dict:
    """Extract token usage from LLM response metadata.

    Supports both OpenAI and Google/Gemini response formats.

    Args:
        response (Any): LLM response object

    Returns:
        dict: Dict with input_tokens, output_tokens, total_tokens (0 if not available)
    """
    usage = {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
    }

    if not response:
        return usage

    # Try response_metadata (LangChain standard)
    if hasattr(response, "response_metadata"):
        metadata = response.response_metadata

        # OpenAI format
        if "token_usage" in metadata:
            token_usage = metadata["token_usage"]
            usage["input_tokens"] = token_usage.get("prompt_tokens", 0)
            usage["output_tokens"] = token_usage.get("completion_tokens", 0)
            usage["total_tokens"] = token_usage.get("total_tokens", 0)

        # Google/Gemini format
        elif "usage_metadata" in metadata:
            usage_metadata = metadata["usage_metadata"]
            usage["input_tokens"] = usage_metadata.get("prompt_token_count", 0)
            usage["output_tokens"] = usage_metadata.get("candidates_token_count", 0)
            usage["total_tokens"] = usage_metadata.get("total_token_count", 0)

    return usage


def reset_token_usage() -> None:
    """Reset cumulative token tracking to zero."""
    global _token_usage
    _token_usage = {
        "total_input_tokens": 0,
        "total_output_tokens": 0,
        "call_count": 0,
    }


def get_token_usage() -> dict:
    """Get current cumulative token usage statistics.

    Returns:
        dict: Dict with total_input_tokens, total_output_tokens, call_count
    """
    return _token_usage.copy()


def update_token_usage(input_tokens: int, output_tokens: int) -> None:
    """Update cumulative token usage with a new LLM call.

    Args:
        input_tokens (int): Number of input tokens for this call
        output_tokens (int): Number of output tokens for this call
    """
    global _token_usage
    _token_usage["total_input_tokens"] += input_tokens
    _token_usage["total_output_tokens"] += output_tokens
    _token_usage["call_count"] += 1


# ============================================================================
# Message Logging
# ============================================================================


def get_message_content_length(msg) -> int:
    """Get the character length of a message's content.

    Args:
        msg (Any): A LangChain message object

    Returns:
        int: Total character count of the message content
    """
    if hasattr(msg, "content"):
        content = msg.content
        if isinstance(content, str):
            return len(content)
        elif isinstance(content, list):
            # Handle list of content blocks
            total = 0
            for block in content:
                if isinstance(block, str):
                    total += len(block)
                elif isinstance(block, dict) and "text" in block:
                    total += len(block["text"])
            return total
    return 0


def log_message_stats(messages: list, label: str = "Message Stats", verbose: bool = True) -> int:
    """Log statistics about message types and lengths.

    Args:
        messages (list): List of messages to analyze
        label (str): Label for the log line
        verbose (bool): If False, skip logging

    Returns:
        int: Estimated token count for input
    """
    stats = {
        "SystemMessage": {"count": 0, "chars": 0, "tokens": 0},
        "HumanMessage": {"count": 0, "chars": 0, "tokens": 0},
        "AIMessage": {"count": 0, "chars": 0, "tokens": 0},
        "ToolMessage": {"count": 0, "chars": 0, "tokens": 0},
    }

    for msg in messages:
        msg_type = type(msg).__name__
        if msg_type in stats:
            stats[msg_type]["count"] += 1
            content = ""
            if hasattr(msg, "content"):
                if isinstance(msg.content, str):
                    content = msg.content
                elif isinstance(msg.content, list):
                    content = " ".join(
                        block.get("text", "") if isinstance(block, dict) else str(block)
                        for block in msg.content
                    )
            chars = len(content)
            tokens = count_tokens(content)
            stats[msg_type]["chars"] += chars
            stats[msg_type]["tokens"] += tokens

    total_msgs = sum(s["count"] for s in stats.values())
    total_chars = sum(s["chars"] for s in stats.values())
    total_tokens = sum(s["tokens"] for s in stats.values())

    if not verbose:
        return total_tokens

    # Build readable log line with clear labels
    parts = []
    for msg_type, data in stats.items():
        if data["count"] > 0:
            short_name = msg_type.replace("Message", "")
            parts.append(f"{short_name}: {data['count']} msg ({data['tokens']} tokens)")

    console.print(
        f"[dim][{label}] {total_msgs} messages, {total_chars} chars, "
        f"{total_tokens} tokens\n"
        f"    Breakdown: {' | '.join(parts)}[/dim]"
    )

    return total_tokens


def log_tool_output(tool_messages: list, verbose: bool = True) -> None:
    """Log statistics about tool message outputs.

    Args:
        tool_messages (list): List of tool messages to analyze
        verbose (bool): If False, skip logging
    """
    if not verbose or not tool_messages:
        return

    tool_chars = sum(get_message_content_length(tm) for tm in tool_messages)
    tool_names = [tm.name for tm in tool_messages]
    console.print(
        f"[dim][Tool Output] {len(tool_messages)} tool response(s), "
        f"{tool_chars} chars\n"
        f"    Tools executed: {', '.join(tool_names)}[/dim]"
    )


def log_token_usage(actual_usage: dict, cumulative_usage: dict, verbose: bool = True) -> None:
    """Log token usage from an LLM call.

    Args:
        actual_usage (dict): Token usage from this call (input_tokens,
            output_tokens, total_tokens)
        cumulative_usage (dict): Cumulative token usage (total_input_tokens,
            total_output_tokens, call_count)
        verbose (bool): If False, skip logging
    """
    if not verbose:
        return

    if actual_usage["total_tokens"] > 0:
        console.print(
            f"[dim][Agent Output] This call: {actual_usage['input_tokens']} "
            f"input tokens, {actual_usage['output_tokens']} output tokens, "
            f"{actual_usage['total_tokens']} total\n"
            f"    Cumulative: {cumulative_usage['total_input_tokens']} "
            f"input tokens, {cumulative_usage['total_output_tokens']} "
            f"output tokens ({cumulative_usage['call_count']} calls)[/dim]"
        )
    else:
        console.print(
            "[dim][Agent Output] Token usage not available from API "
            "(using tiktoken estimate instead)[/dim]"
        )


def log_compaction(
    original_length: int,
    summary_length: int,
    original_tokens: int,
    summary_tokens: int,
    verbose: bool = True,
) -> None:
    """Log compaction/summarization statistics.

    Args:
        original_length (int): Original content length in chars
        summary_length (int): Summarized content length in chars
        original_tokens (int): Original token count
        summary_tokens (int): Summarized token count
        verbose (bool): If False, skip logging
    """
    if not verbose:
        return

    saved_tokens = original_tokens - summary_tokens
    console.print(
        f"[dim][Compaction] Feedback summarized: {original_length} chars → "
        f"{summary_length} chars\n"
        f"    Tokens saved: {saved_tokens} tokens (from {original_tokens} "
        f"to {summary_tokens})[/dim]"
    )
