"""
Simple todo list management for LangGraph-based research agents.
"""

from enum import Enum
from typing import Any, Dict, List, Literal, Optional


class TaskStatus(str, Enum):
    """Status of a task."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class TodoItem:
    """A single todo item."""

    def __init__(
        self, id: str, description: str, status: TaskStatus = TaskStatus.PENDING
    ):
        self.id = id
        self.description = description
        self.status = status

    def __repr__(self):
        emoji = {
            "pending": "⏸️",
            "in_progress": "🔄",
            "completed": "✅",
            "failed": "❌",
        }
        status_val = self.status.value
        return f"{emoji[status_val]} [{self.id}] {self.description} ({status_val})"


class TodoList:
    """Simple todo list manager."""

    def __init__(self, goal: str = ""):
        self.goal = goal
        self.tasks: Dict[str, TodoItem] = {}

    def add_task(self, task_id: str, description: str) -> TodoItem:
        """Add a new task."""
        task = TodoItem(task_id, description)
        self.tasks[task_id] = task
        return task

    def update_status(self, task_id: str, status: TaskStatus) -> TodoItem:
        """Update task status."""
        if task_id not in self.tasks:
            raise ValueError(f"Task '{task_id}' not found")
        self.tasks[task_id].status = status
        return self.tasks[task_id]

    def get_pending(self) -> List[TodoItem]:
        """Get all pending tasks."""
        return [t for t in self.tasks.values() if t.status == TaskStatus.PENDING]

    def get_in_progress(self) -> List[TodoItem]:
        """Get all in-progress tasks."""
        return [t for t in self.tasks.values() if t.status == TaskStatus.IN_PROGRESS]

    def get_current_task(self) -> Optional[TodoItem]:
        """Get the first in-progress task (current task being worked on)."""
        in_progress = self.get_in_progress()
        return in_progress[0] if in_progress else None

    def mark_complete(self, task_id: str) -> TodoItem:
        """Mark a task as completed (shortcut for update_status)."""
        return self.update_status(task_id, TaskStatus.COMPLETED)

    def mark_in_progress(self, task_id: str) -> TodoItem:
        """Mark a task as in progress (shortcut for update_status)."""
        return self.update_status(task_id, TaskStatus.IN_PROGRESS)

    def get_task_by_index(self, index: int) -> Optional[TodoItem]:
        """Get a task by its index (0-based)."""
        task_list = list(self.tasks.values())
        if 0 <= index < len(task_list):
            return task_list[index]
        return None

    def update_by_index(self, index: int, status: TaskStatus) -> TodoItem:
        """Update task status by index (0-based)."""
        task = self.get_task_by_index(index)
        if not task:
            raise ValueError(f"Task at index {index} not found")
        return self.update_status(task.id, status)

    def get_progress(self) -> Dict[str, int]:
        """Get task counts by status."""
        counts = {"pending": 0, "in_progress": 0, "completed": 0, "failed": 0}
        for task in self.tasks.values():
            counts[task.status.value] += 1
        counts["total"] = len(self.tasks)
        return counts

    def format_summary(self) -> str:
        """Get a formatted summary."""
        progress = self.get_progress()
        lines = [
            f"📋 Todo List: {self.goal}",
            f"Progress: {progress['completed']}/{progress['total']} completed, "
            f"{progress['in_progress']} in progress, {progress['failed']} failed\n",
        ]

        for status in TaskStatus:
            tasks = [t for t in self.tasks.values() if t.status == status]
            if tasks:
                lines.append(f"{status.value.upper()}:")
                for task in tasks:
                    lines.append(f"  {task}")
                lines.append("")

        return "\n".join(lines)


def manage_todo_list(
    action: Literal[
        "create",
        "add",
        "update",
        "get_pending",
        "get_in_progress",
        "get_current",
        "progress",
        "summary",
        # Partial update actions (token-efficient)
        "mark_complete",
        "mark_in_progress",
        "update_by_index",
    ],
    state: Dict[str, Any],
    task_id: Optional[str] = None,
    task_index: Optional[int] = None,
    description: Optional[str] = None,
    status: Optional[str] = None,
    goal: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Manage a todo list in LangGraph state.

    Args:
        action: Action to perform
            - "create": Initialize new todo list
            - "add": Add a new task
            - "update": Update task status by ID
            - "get_pending": Get pending tasks
            - "get_in_progress": Get in-progress tasks
            - "get_current": Get current task being worked on
            - "progress": Get progress stats
            - "summary": Get formatted summary

            Partial update actions (token-efficient):
            - "mark_complete": Mark task as completed (by ID or index)
            - "mark_in_progress": Mark task as in progress (by ID or index)
            - "update_by_index": Update task status by index (0-based)

        state: Agent state dict (contains 'todo_list' key)
        task_id: Task ID (for add/update/mark_complete/mark_in_progress)
        task_index: Task index, 0-based (for update_by_index,
            mark_complete, mark_in_progress)
        description: Task description (for add)
        status: New status - "pending", "in_progress", "completed", or "failed"
        goal: Research goal (for create)

    Returns:
        Dict with 'success', 'message', 'todo_list', and 'data' keys

    Example:
        >>> # Create list
        >>> result = manage_todo_list("create", state, goal="Research AI")
        >>> state["todo_list"] = result["todo_list"]
        >>>
        >>> # Add tasks
        >>> manage_todo_list("add", state, task_id="1", description="Review papers")
        >>> manage_todo_list("add", state, task_id="2", description="Collect data")
        >>>
        >>> # Update status (full update)
        >>> manage_todo_list("update", state, task_id="1", status="completed")
        >>>
        >>> # Partial updates (token-efficient)
        >>> manage_todo_list("mark_complete", state, task_index=0)  # By index
        >>> manage_todo_list("mark_in_progress", state, task_id="2")  # By ID
        >>> manage_todo_list("update_by_index", state, task_index=1, status="completed")
        >>>
        >>> # Check progress
        >>> result = manage_todo_list("progress", state)
        >>> print(result["data"])
        # {'pending': 0, 'in_progress': 0, 'completed': 2, ...}
    """
    todo_list = state.get("todo_list")

    if action == "create":
        todo_list = TodoList(goal=goal or "")
        return {
            "success": True,
            "message": "Todo list created",
            "todo_list": todo_list,
            "data": None,
        }

    if not todo_list:
        raise ValueError("No todo list in state. Use action='create' first.")

    if action == "add":
        if not task_id or not description:
            raise ValueError("task_id and description required")
        task = todo_list.add_task(task_id, description)
        return {
            "success": True,
            "message": f"Added task '{task_id}'",
            "todo_list": todo_list,
            "data": task,
        }

    elif action == "update":
        if not task_id or not status:
            raise ValueError("task_id and status required")
        task_status = TaskStatus(status)
        task = todo_list.update_status(task_id, task_status)
        return {
            "success": True,
            "message": f"Task '{task_id}' -> {status}",
            "todo_list": todo_list,
            "data": task,
        }

    elif action == "get_pending":
        pending = todo_list.get_pending()
        return {
            "success": True,
            "message": f"{len(pending)} pending tasks",
            "todo_list": todo_list,
            "data": pending,
        }

    elif action == "get_in_progress":
        in_progress = todo_list.get_in_progress()
        return {
            "success": True,
            "message": f"{len(in_progress)} in-progress tasks",
            "todo_list": todo_list,
            "data": in_progress,
        }

    elif action == "get_current":
        current = todo_list.get_current_task()
        if current:
            return {
                "success": True,
                "message": f"Current task: {current.description}",
                "todo_list": todo_list,
                "data": current,
            }
        else:
            return {
                "success": True,
                "message": "No task currently in progress",
                "todo_list": todo_list,
                "data": None,
            }

    elif action == "mark_complete":
        # Support both task_id and task_index
        if task_index is not None:
            task = todo_list.update_by_index(task_index, TaskStatus.COMPLETED)
        elif task_id:
            task = todo_list.mark_complete(task_id)
        else:
            raise ValueError("task_id or task_index required for mark_complete")
        return {
            "success": True,
            "message": f"Task '{task.id}' marked as completed",
            "todo_list": todo_list,
            "data": task,
        }

    elif action == "mark_in_progress":
        # Support both task_id and task_index
        if task_index is not None:
            task = todo_list.update_by_index(task_index, TaskStatus.IN_PROGRESS)
        elif task_id:
            task = todo_list.mark_in_progress(task_id)
        else:
            raise ValueError("task_id or task_index required for mark_in_progress")
        return {
            "success": True,
            "message": f"Task '{task.id}' marked as in progress",
            "todo_list": todo_list,
            "data": task,
        }

    elif action == "update_by_index":
        if task_index is None:
            raise ValueError("task_index required for update_by_index")
        if not status:
            raise ValueError("status required for update_by_index")
        task_status = TaskStatus(status)
        task = todo_list.update_by_index(task_index, task_status)
        return {
            "success": True,
            "message": f"Task at index {task_index} -> {status}",
            "todo_list": todo_list,
            "data": task,
        }

    elif action == "progress":
        progress = todo_list.get_progress()
        return {
            "success": True,
            "message": "Progress retrieved",
            "todo_list": todo_list,
            "data": progress,
        }

    elif action == "summary":
        summary = todo_list.format_summary()
        return {
            "success": True,
            "message": "Summary generated",
            "todo_list": todo_list,
            "data": summary,
        }

    else:
        raise ValueError(f"Unknown action: {action}")
