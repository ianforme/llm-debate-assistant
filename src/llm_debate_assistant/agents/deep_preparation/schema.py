from typing import Annotated, List, TypedDict, Dict, Any, Literal, Optional
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

from llm_debate_assistant.services.filesystem_protocol import FilesystemProtocol
from llm_debate_assistant.services.manage_todo_list import TodoList


# Type alias for filesystem (can be DiskFilesystem or VirtualFilesystem)
Filesystem = FilesystemProtocol

# Preparation mode: "lite" = topic_research only, "full" = all segments
PrepMode = Literal["lite", "full"]


class DeepPrepState(TypedDict, total=False):
    """
    Orchestrator State: Only holds Metadata, Pointers, and Chat History.
    NO HEAVY CONTENT.

    Note: total=False allows optional fields for incremental state updates.
    Required fields should be set during initialization.
    """

    # --- 1. Global Context ---
    topic: str
    side: Literal["正方", "反方"]
    session_id: str
    mode: PrepMode  # "lite" or "full"

    # --- 2. Memory (The "Brain") ---
    # Stores the interaction history between Agent and Tools.
    # Tools only return short messages like "Success: saved to /path/..." rather than full content.
    messages: Annotated[List[BaseMessage], add_messages]

    # --- 3. Status Board (The "Dashboard") ---
    # Used for UI progress display, or to help the Agent quickly decide the next step.
    # e.g., {"topic_research": "completed", "constructive_speech": "not_started"}
    # Note: This requires a Reducer or dedicated Node to update it.
    stage_status: Annotated[Dict[str, str], lambda x, y: {**x, **y}]

    # --- 4. Artifact Registry (The "File Explorer") ---
    # Only store paths! Only store paths! Only store paths!
    # e.g., {"research_json": "/research/research.json", "final_draft": "/speech/final.md"}
    artifact_paths: Annotated[Dict[str, str], lambda x, y: {**x, **y}]

    # --- 5. Task Management ---
    # TodoList instance for tracking preparation progress
    # Managed via the update_todo tool
    todo_list: Optional[TodoList]
