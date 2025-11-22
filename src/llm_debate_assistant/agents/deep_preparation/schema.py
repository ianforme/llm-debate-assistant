from typing import Annotated, Any, Optional, Union

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from llm_debate_assistant.services.disk_filesystem import DiskFilesystem
from llm_debate_assistant.services.virtual_filesystem import VirtualFilesystem

# Type alias for either filesystem implementation
Filesystem = Union[VirtualFilesystem, DiskFilesystem]


class DeepPrepState(TypedDict):
    """State for deep preparation agent with filesystem support.

    This state extends the debate preparation workflow with filesystem
    for managing research notes, outlines, drafts, and other intermediate
    work products.

    Supports both:
    - VirtualFilesystem: In-memory, transient storage
    - DiskFilesystem: Persistent storage in
        .temp/debate_preparation_sessions/<session_id>/
    """

    # Input parameters
    topic: str
    side: str

    # Filesystem (VirtualFilesystem or DiskFilesystem)
    filesystem: Optional[Filesystem]
    filesystem_path: Optional[str]  # Path to session directory (DiskFilesystem only)

    # Workflow outputs (structured)
    # Note: These are typically offloaded to filesystem and cleared from state
    outline: Optional[dict[str, Any]]
    evidence_data: Optional[list[dict[str, Any]]]
    draft: Optional[str]
    evaluation: Optional[dict[str, Any]]

    # Task planning and tracking (for agent-driven workflow)
    todos: Optional[list[dict[str, Any]]]

    # Control flow
    iteration_count: int
    max_iterations: int
    next_action: Optional[str]
    # "auto" | "redo_outline" | "redo_evidence" | "redo_draft"

    # Debug/logging options
    verbose: Optional[bool]  # Enable verbose logging (message stats, etc.)

    # Message history for LangGraph
    messages: Annotated[list[BaseMessage], add_messages]
