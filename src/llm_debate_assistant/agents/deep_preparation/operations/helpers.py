"""
Helper functions for loading data from filesystem or state.
"""

import json
from typing import Any

from ..schema import DeepPrepState


def load_outline_from_fs(state: DeepPrepState) -> dict[str, Any]:
    """Load outline from filesystem or state.

    Args:
        state (DeepPrepState): Deep preparation agent state

    Raises:
        ValueError: If no outline found in state or filesystem

    Returns:
        dict[str, Any]: Outline dictionary
    """
    # Try filesystem first
    filesystem = state.get("filesystem")
    if filesystem and filesystem.exists("/outline.json"):
        result = filesystem.read("/outline.json")
        if result["success"]:
            return json.loads(result["content"])

    # Fall back to state
    outline = state.get("outline")
    if outline:
        return outline

    raise ValueError("No outline found in state or filesystem")


def load_evidence_analysis(state: DeepPrepState, arg_index: int) -> str:
    """Load evidence from filesystem or state.

    Args:
        state (DeepPrepState): Deep preparation agent state
        arg_index (int): Argument index (1-based)

    Returns:
        str: Evidence analysis text
    """
    filesystem = state.get("filesystem")
    if filesystem:
        result = filesystem.read(f"/evidence/argument_{arg_index}.md")
        if result["success"]:
            content = result["content"]
            # Extract just the analysis section
            if "## Analysis" in content:
                analysis = content.split("## Analysis")[1].strip()
                return analysis

    # Fallback to state
    evidence_data = state.get("evidence_data", [])
    if evidence_data and arg_index <= len(evidence_data):
        return evidence_data[arg_index - 1].get("analysis", "")

    return ""


def load_current_draft(state: DeepPrepState) -> str:
    """Load current draft from filesystem or state.

    Args:
        state (DeepPrepState): Deep preparation agent state

    Raises:
        ValueError: If no draft found in state or filesystem

    Returns:
        str: Current draft of opening statement
    """
    # Try filesystem first
    filesystem = state.get("filesystem")
    if filesystem:
        result = filesystem.read("/current_draft.md")
        if result["success"]:
            return result["content"]

    # Fall back to state
    draft = state.get("draft")
    if draft:
        return draft

    raise ValueError("No draft found in state or filesystem")
