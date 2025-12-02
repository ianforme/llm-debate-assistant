from langchain_core.tools import tool
from pydantic import BaseModel, Field
from typing import Dict, List, Optional


class TodoItem(BaseModel):
    """A single todo item."""

    content: str = Field(description="What needs to be done (imperative form)")
    status: str = Field(description="Status: 'pending', 'in_progress', or 'completed'")


class WriteTodosInput(BaseModel):
    """Input schema for write_todos tool."""

    todos: List[TodoItem] = Field(description="List of todo items with content and status")


class UpdateTodoStatusInput(BaseModel):
    """Input schema for update_todo_status tool (partial update)."""

    task_index: int = Field(description="Index of the task to update (0-based)")
    status: str = Field(description="New status: 'pending', 'in_progress', or 'completed'")


class MarkTodoCompleteInput(BaseModel):
    """Input schema for mark_todo_complete tool (partial update)."""

    task_index: int = Field(description="Index of the task to mark as completed (0-based)")


class WriteFileInput(BaseModel):
    """Input schema for write_file tool."""

    path: str = Field(description="File path to write to (e.g., '/notes.md')")
    content: str = Field(description="Content to write to the file")


class ReadFileInput(BaseModel):
    """Input schema for read_file tool."""

    path: str = Field(description="File path to read from (e.g., '/outline.json')")


class SearchEvidenceInput(BaseModel):
    """Input schema for search_evidence tool."""

    argument_claims: List[str] = Field(description="List of argument claims to search evidence for")
    search_queries: Optional[List[str]] = Field(
        default=None,
        description="Optional custom search queries (one per claim)",
    )


class CreateOutlineInput(BaseModel):
    """Input schema for create_outline tool."""

    topic: str = Field(description="Debate topic")
    side: str = Field(description="Debate side (正方 or 反方)")


class DraftStatementInput(BaseModel):
    """Input schema for draft_statement tool."""

    use_feedback: bool = Field(
        default=False,
        description="Whether to incorporate evaluation feedback",
    )


class EvaluateStatementInput(BaseModel):
    """Input schema for evaluate_statement tool."""

    # No parameters needed - evaluation uses state data


class ImproveStatementInput(BaseModel):
    """Input schema for improve_statement tool."""

    # No parameters needed - improvement uses evaluation feedback from state


# ============================================================================
# Core Workflow Tools
# ============================================================================


@tool(args_schema=CreateOutlineInput)
def create_outline_tool(topic: str, side: str) -> str:
    """Create a debate outline with keyword definitions, comparison standard,
    and arguments.

    This should be the first step in debate preparation. The outline will be
    saved to filesystem automatically.

    Args:
        topic: The debate topic
        side: The side you're arguing (正方 or 反方)

    Returns:
        Success message with outline summary
    """
    # This tool's implementation will be in the tool execution node
    # where it has access to state
    return "create_outline_requested"


@tool(args_schema=SearchEvidenceInput)
def search_evidence_tool(
    argument_claims: List[str],
    search_queries: Optional[List[str]] = None,
) -> str:
    """Search for evidence supporting debate arguments.

    Performs web search to find supporting evidence for each argument claim.
    Make sure you've created an outline first.

    Args:
        argument_claims: List of argument claims to find evidence for
        search_queries: Optional custom search queries (defaults to claims)

    Returns:
        Success message with evidence count
    """
    return "search_evidence_requested"


@tool(args_schema=DraftStatementInput)
def draft_statement_tool(use_feedback: bool = False) -> str:
    """Draft an opening statement based on outline and evidence.

    Combines the outline structure with gathered evidence to create a
    complete opening statement. Make sure you've created outline and
    gathered evidence first.

    Args:
        use_feedback: Whether to incorporate evaluation feedback

    Returns:
        Success message with draft summary
    """
    return "draft_statement_requested"


@tool(args_schema=EvaluateStatementInput)
def evaluate_statement_tool() -> str:
    """Evaluate the current draft opening statement.

    Analyzes the draft for strengths, weaknesses, and provides improvement
    suggestions. Make sure you've drafted a statement first.

    Returns:
        Evaluation results with score and feedback
    """
    return "evaluate_statement_requested"


@tool(args_schema=ImproveStatementInput)
def improve_statement_tool() -> str:
    """Improve the opening statement based on evaluation feedback.

    Uses the evaluation feedback to revise and enhance the draft.
    Make sure you've evaluated the statement first.

    Returns:
        Success message with improved draft summary
    """
    return "improve_statement_requested"


# ============================================================================
# Support Tools (Filesystem & Todos)
# ============================================================================


@tool(args_schema=WriteFileInput)
def write_file_tool(path: str, content: str) -> str:
    """Write content to a file in the persistent filesystem.

    Use this to save notes, intermediate results, or any data you want to
    persist. Files are saved to disk and can be read back later.

    Args:
        path: File path starting with / (e.g., '/notes/research.md')
        content: Content to write to the file

    Returns:
        Confirmation message
    """
    return "write_file_requested"


@tool(args_schema=ReadFileInput)
def read_file_tool(path: str) -> str:
    """Read content from a file in the persistent filesystem.

    Use this to retrieve previously saved notes, outlines, or other data.

    Args:
        path: File path starting with / (e.g., '/outline.json')

    Returns:
        File contents
    """
    return "read_file_requested"


@tool(args_schema=WriteTodosInput)
def write_todos_tool(todos: List[Dict[str, str]]) -> str:
    """Update the task planning and tracking list (full replacement).

    Use this to plan complex multi-step work, track progress, and organize
    your workflow. Each todo should have:
    - content: What needs to be done (imperative form)
    - status: 'pending', 'in_progress', or 'completed'

    Best practices:
    - Create todos at the start for planning
    - For status updates, prefer update_todo_status_tool or mark_todo_complete_tool
    - Be specific and actionable

    Args:
        todos: List of todo items with content and status

    Returns:
        Confirmation message with progress summary
    """
    return "write_todos_requested"


@tool(args_schema=UpdateTodoStatusInput)
def update_todo_status_tool(task_index: int, status: str) -> str:
    """Update a single task's status without rewriting the entire list.

    This is more token-efficient than write_todos_tool when you only need
    to update one task. The existing list is preserved.

    Args:
        task_index: Index of the task to update (0-based)
        status: New status: 'pending', 'in_progress', or 'completed'

    Returns:
        Confirmation message
    """
    return "update_todo_status_requested"


@tool(args_schema=MarkTodoCompleteInput)
def mark_todo_complete_tool(task_index: int) -> str:
    """Mark a task as completed without rewriting the entire list.

    This is the most token-efficient way to mark a task as done.
    Use this instead of write_todos_tool when you've finished a step.

    Args:
        task_index: Index of the task to mark as completed (0-based)

    Returns:
        Confirmation message
    """
    return "mark_todo_complete_requested"


# ============================================================================
# Tool Registry
# ============================================================================

# All available tools for the agent
ALL_TOOLS = [
    # Core workflow tools
    create_outline_tool,
    search_evidence_tool,
    draft_statement_tool,
    evaluate_statement_tool,
    improve_statement_tool,
    # Support tools
    write_file_tool,
    read_file_tool,
    write_todos_tool,
    # Partial update tools (token-efficient)
    update_todo_status_tool,
    mark_todo_complete_tool,
]


def get_tool_by_name(name: str):
    """Get a tool by its name.

    Args:
        name: Tool name

    Returns:
        Tool function

    Raises:
        ValueError: If tool not found
    """
    tool_map = {tool.name: tool for tool in ALL_TOOLS}
    if name not in tool_map:
        raise ValueError(f"Unknown tool: {name}")
    return tool_map[name]
