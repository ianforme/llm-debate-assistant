import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Literal, Optional, cast

from llm_debate_assistant.services.disk_filesystem import DiskFilesystem
from llm_debate_assistant.services.virtual_filesystem import VirtualFilesystem

from .schema import DeepPrepState, Filesystem


# ============================================================================
# Session Management
# ============================================================================


def save_session_metadata(
    filesystem: Filesystem,
    topic: str,
    side: Literal["正方", "反方"],
    segment: str = "topic_research",
) -> None:
    """Save session metadata for cache lookup.

    Args:
        filesystem: Filesystem instance to save metadata to
        topic: Debate topic
        side: Our side (正方 or 反方)
        segment: Which segment this session is for (e.g., "topic_research")
    """
    metadata = {
        "topic": topic,
        "side": side,
        "segment": segment,
        "created_at": datetime.now().isoformat(),
    }
    filesystem.write("/_metadata.json", json.dumps(metadata, ensure_ascii=False))


def find_session_by_topic(
    topic: str,
    side: Literal["正方", "反方"],
    segment: str = "topic_research",
    root_dir: Optional[Path] = None,
) -> Optional[str]:
    """Find existing session matching topic and side.

    Searches all sessions in the root directory for one matching the
    given topic, side, and segment. Returns the most recent match.

    Args:
        topic: Debate topic to search for
        side: Side to search for (正方 or 反方)
        segment: Segment type (default: "topic_research")
        root_dir: Root directory for sessions (default: .temp/debate_preparation_sessions)

    Returns:
        Optional[str]: Session ID of most recent match, or None if not found
    """
    if root_dir is None:
        root_dir = Path.cwd() / ".temp" / "debate_preparation_sessions"

    if not root_dir.exists():
        return None

    matches = []

    for session_dir in root_dir.iterdir():
        if not session_dir.is_dir():
            continue

        metadata_path = session_dir / "_metadata.json"
        if not metadata_path.exists():
            continue

        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

            # Check if this session matches our criteria
            if (
                metadata.get("topic") == topic
                and metadata.get("side") == side
                and metadata.get("segment") == segment
            ):
                created_at = metadata.get("created_at", "")
                matches.append((created_at, session_dir.name))
        except (json.JSONDecodeError, IOError):
            # Skip invalid metadata files
            continue

    if matches:
        # Return most recent session ID
        return max(matches)[1]

    return None


def create_filesystem(
    filesystem_type: Literal["virtual", "disk"] = "disk",
    session_id: Optional[str] = None,
    topic: Optional[str] = None,
    side: Optional[Literal["正方", "反方"]] = None,
    segment: str = "topic_research",
    use_cache: bool = True,
) -> tuple[Filesystem, Optional[str], bool]:
    """Factory function to create either virtual or disk filesystem.

    If topic and side are provided, will search for existing session
    matching those criteria and reuse it if use_cache=True.

    Args:
        filesystem_type: Type of filesystem to create ("virtual" or "disk")
        session_id: Optional session ID (only used for disk filesystem)
        topic: Optional debate topic (for cache lookup)
        side: Optional side (正方 or 反方, for cache lookup)
        segment: Segment type (default: "topic_research")
        use_cache: Whether to search for and reuse existing sessions

    Returns:
        tuple[Filesystem, Optional[str], bool]: Tuple of (filesystem instance,
            session path or None, cache_found)
    """
    if filesystem_type == "virtual":
        return VirtualFilesystem(), None, False
    elif filesystem_type == "disk":
        cache_found = False

        # Try to find existing session if topic/side provided and caching enabled
        if use_cache and topic and side and session_id is None:
            cached_session_id = find_session_by_topic(topic, side, segment)
            if cached_session_id:
                session_id = cached_session_id
                cache_found = True

        fs = DiskFilesystem(session_id=session_id)
        return fs, fs.get_session_path(), cache_found
    else:
        raise ValueError(f"Unknown filesystem_type: {filesystem_type}")


def initialize_filesystem_node(
    state: DeepPrepState,
    filesystem_type: Literal["virtual", "disk"] = "disk",
    session_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Initialize filesystem in state if not already present.

    This node should be called at the start of the graph to ensure
    the filesystem is available for all nodes.

    Args:
        state (DeepPrepState): Deep preparation agent state
        filesystem_type (Literal["virtual", "disk"]): Type of filesystem
            ("virtual" for in-memory, "disk" for persistent)
        session_id (Optional[str]): Optional session ID (only used for disk
            filesystem)

    Returns:
        Dict[str, Any]: Dict with filesystem and path information if
            initialized, else empty
    """
    if state.get("filesystem") is None:
        filesystem, session_path, _ = create_filesystem(filesystem_type, session_id)
        return {
            "filesystem": filesystem,
            "filesystem_path": session_path,
        }

    return {}


# Alias for backwards compatibility
initialize_filesystem_middleware = initialize_filesystem_node


# ======================================================================
# Context offloading functions for filesystem-based deep preparation
# ======================================================================


def save_outline_to_filesystem(state: DeepPrepState) -> Dict[str, Any]:
    """Save outline to filesystem after creation.

    Args:
        state (DeepPrepState): Deep preparation agent state

    Returns:
        Dict[str, Any]: Dictionary with updated filesystem
        information if saved, else empty
    """
    outline = state.get("outline")
    if outline is None:
        return {}

    filesystem = state.get("filesystem")
    if not filesystem:
        return {}

    import json

    # Save structured JSON (for loading in nodes)
    filesystem.write("/outline.json", json.dumps(outline, ensure_ascii=False))

    # Also save human-readable markdown
    content_parts = ["# Debate Outline\n"]
    content_parts.append(f"Topic: {state['topic']}")
    content_parts.append(f"Side: {state['side']}\n")

    content_parts.append("## Keyword Definitions")
    for kw in outline.get("keyword_definitions", []):
        content_parts.append(f"- **{kw['keyword']}**: {kw['definition']}")

    content_parts.append("\n## Comparison Standard")
    standard = outline.get("comparison_standard", {})
    content_parts.append(f"{standard.get('standard', '')}")
    content_parts.append(f"Justification: {standard.get('justification', '')}")

    content_parts.append("\n## Arguments")
    for i, arg in enumerate(outline.get("arguments", []), 1):
        content_parts.append(f"\n### Argument {i}")
        content_parts.append(f"**Claim**: {arg.get('claim', '')}")
        content_parts.append(f"**Warrant**: {arg.get('warrant', '')}")

    filesystem.write("/outline.md", "\n".join(content_parts))

    # Clear outline from state to save tokens - it's now on disk
    return {"outline": None}


def save_evidence_to_filesystem(state: DeepPrepState) -> Dict[str, Any]:
    """Save evidence to filesystem after search.

    Args:
        state (DeepPrepState): Deep preparation agent state

    Returns:
        Dict[str, Any]: Dictionary with updated filesystem
        information if saved, else empty
    """
    evidence_data = state.get("evidence_data")
    if evidence_data:
        for i, evidence in enumerate(evidence_data, 1):
            content_parts = [f"# Evidence for Argument {i}\n"]
            content_parts.append(f"**Argument**: {evidence.get('argument', '')}")
            content_parts.append(f"**Warrant**: {evidence.get('warrant', '')}\n")

            content_parts.append("## Search Results")
            for source in evidence.get("search_results", []):
                content_parts.append(
                    f"- [{source.get('title', 'Source')}]({source.get('uri', '')})"
                )

            content_parts.append("\n## Analysis")
            content_parts.append(evidence.get("analysis", ""))

            content = "\n".join(content_parts)

            filesystem = state.get("filesystem")
            if filesystem:
                filesystem.write(f"/evidence/argument_{i}.md", content)

        # Clear evidence_data from state to save tokens - it's now on disk
        if state.get("filesystem"):
            return {"evidence_data": None}

        return {}

    return {}


def save_draft_to_filesystem(state: DeepPrepState) -> Dict[str, Any]:
    """Save draft to filesystem after creation.

    Args:
        state (DeepPrepState): Deep preparation agent state with draft

    Returns:
        Dict[str, Any]: Dictionary with updated filesystem
        information if saved, else empty
    """
    draft = state.get("draft")
    if not draft:
        return {}

    iteration = state.get("iteration_count", 1)

    filesystem = state.get("filesystem")
    if not filesystem:
        return {}

    # Save as versioned draft
    filesystem.write(f"/drafts/draft_v{iteration}.md", draft)
    # Also save as current draft
    filesystem.write("/current_draft.md", draft)
    # Clear draft from state to save tokens - it's now on disk
    return {"draft": None}


def save_final_state_to_filesystem(state: DeepPrepState) -> Dict[str, Any]:
    """Save complete final state to filesystem for reference.

    Serializes the entire state (excluding non-serializable objects) to JSON
    for debugging and analysis after workflow completion.

    Args:
        state (DeepPrepState): Final deep preparation agent state

    Returns:
        Dict[str, Any]: Empty dict (doesn't modify state)
    """
    filesystem = state.get("filesystem")
    if not filesystem:
        return {}

    # Serialize state, handling non-serializable objects
    serializable_state: Dict[str, Any] = {}

    for key, value in state.items():
        # Skip non-serializable objects
        if key == "filesystem":
            serializable_state[key] = f"<{type(value).__name__}>"
            continue

        # Serialize messages to readable format (full content, no truncation)
        if key == "messages" and value is not None:
            serialized_messages: list[Dict[str, Any]] = []
            messages_list = value if isinstance(value, list) else []
            for msg in messages_list:
                msg_type = type(msg).__name__

                # Get content
                if hasattr(msg, "content"):
                    content = msg.content
                    if isinstance(content, str):
                        msg_content = content
                    else:
                        msg_content = str(content)
                else:
                    msg_content = ""

                # Get tool calls if present
                tool_calls = None
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    tool_calls = [
                        {"name": tc.get("name"), "args": tc.get("args", {})}
                        for tc in msg.tool_calls
                    ]

                serialized_msg: Dict[str, Any] = {
                    "type": msg_type,
                    "content": msg_content,  # Full content for reference
                }

                if tool_calls:
                    serialized_msg["tool_calls"] = tool_calls

                # Add tool name for ToolMessages
                if hasattr(msg, "name"):
                    serialized_msg["tool_name"] = msg.name

                serialized_messages.append(serialized_msg)

            serializable_state[key] = cast(Any, serialized_messages)
            continue

        # Regular serializable values
        try:
            json.dumps(value)  # Test if serializable
            serializable_state[key] = value
        except (TypeError, ValueError):
            serializable_state[key] = str(value)

    # Add metadata
    serializable_state["_metadata"] = {
        "saved_at": datetime.now().isoformat(),
        "message_count": len(state.get("messages", [])),
        "final_iteration": state.get("iteration_count", 1),
    }

    # Save as JSON (full state for reference)
    filesystem.write(
        "/final_state.json",
        json.dumps(serializable_state, ensure_ascii=False, indent=2),
    )

    return {}
