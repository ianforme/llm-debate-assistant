from typing import Annotated, Any, Optional

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class DebateState(TypedDict):
    """State for debate preparation reflection pattern."""

    # Input parameters (辩题，立场)
    topic: str
    side: str

    # Workflow outputs (structured)

    # 大纲
    outline: Optional[dict[str, Any]]
    # 论据
    evidence_data: Optional[list[dict[str, Any]]]
    # 初稿
    draft: Optional[str]
    # 评估结果
    evaluation: Optional[dict[str, Any]]

    # Control flow
    iteration_count: int
    max_iterations: int

    # Redo control (optional - for manual intervention)
    next_action: Optional[str]  # "auto" | "redo_outline" | "redo_evidence" | "redo_draft"

    # Message history for LangGraph
    messages: Annotated[list[BaseMessage], add_messages]
