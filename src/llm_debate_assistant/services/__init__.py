"""Services for LLM debate assistant agents."""

from .disk_filesystem import DiskFilesystem
from .filesystem_protocol import (
    FilesystemProtocol,
    delete_file,
    edit_file,
    file_exists,
    list_directory,
    manage_filesystem,
    read_file,
    write_file,
)
from .llm import DEFAULT_MODEL, get_llm
from .manage_todo_list import TaskStatus, TodoItem, TodoList, manage_todo_list
from .virtual_filesystem import VirtualFilesystem
from .web_search import (
    ArgumentEvidence,
    async_search_for_evidence,
    async_search_for_evidence_threaded,
    async_search_web,
    search_for_evidence,
    search_multiple_arguments,
    search_multiple_arguments_threaded,
    search_web,
)

__all__ = [
    # Filesystem implementations
    "VirtualFilesystem",
    "DiskFilesystem",
    "FilesystemProtocol",
    # Filesystem state management (works with any filesystem type)
    "manage_filesystem",
    "write_file",
    "read_file",
    "list_directory",
    "edit_file",
    "delete_file",
    "file_exists",
    # LLM configuration
    "get_llm",
    "DEFAULT_MODEL",
    # Todo list tools
    "TodoList",
    "TodoItem",
    "TaskStatus",
    "manage_todo_list",
    # Web search tools
    "search_web",
    "async_search_web",
    "search_for_evidence",
    "async_search_for_evidence",
    "async_search_for_evidence_threaded",
    "search_multiple_arguments",
    "search_multiple_arguments_threaded",
    "ArgumentEvidence",
]
