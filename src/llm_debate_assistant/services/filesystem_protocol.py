"""Common protocol for both virtual and disk-based filesystems.

This module provides:
- FilesystemProtocol: Interface for filesystem implementations
- Unified state management functions that work with any filesystem type
- Can be extended for future filesystem types (like cloud storage)
"""

from typing import Any, Dict, Literal, Optional, Protocol, runtime_checkable


@runtime_checkable
class FilesystemProtocol(Protocol):
    """Protocol defining the interface for both filesystem implementations.

    This allows nodes to work with either VirtualFilesystem (in-memory)
    or DiskFilesystem (persistent) interchangeably. Protocols also support
    forward compatibility if new methods are added later, e.g., managed
    services or cloud storage.
    """

    def write(self, path: str, content: str) -> Dict[str, Any]:
        """Write content to a file path

        Args:
            path (str): file path
            content (str): file content

        Returns:
            Dict[str, Any]: result dictionary with
            'success' and 'message'

        """
        ...

    def read(self, path: str, lines: Optional[int] = None) -> Dict[str, Any]:
        """Read content from a file.

        Args:
            path (str): file path to read
            lines (Optional[int]): optional line limit

        Returns:
            Dict[str, Any]: result dictionary with
            'success', 'message', and 'content'
        """
        ...

    def exists(self, path: str) -> bool:
        """Check if a file path exists

        Args:
            path (str): file path to check

        Returns:
            bool: True if file exists,
            False otherwise
        """
        ...

    def ls(self, directory: str = "/") -> Dict[str, Any]:
        """List files in a directory.

        Args:
            directory: Directory path to list

        Returns:
            Dict with 'success', 'message', and 'items'
        """
        ...

    def edit(self, path: str, old_text: str, new_text: str) -> Dict[str, Any]:
        """Edit a file by replacing text.

        Args:
            path (str): file path to edit
            old_text (str): text to find and replace
            new_text (str): replacement text

        Returns:
            Dict[str, Any]: result dictionary with
            'success', 'message', and 'replacements'
        """
        ...


# ==============================================================
# Unified State Management Functions
# ==============================================================


def manage_filesystem(
    action: Literal["write", "read", "ls", "edit", "delete", "exists"],
    state: Dict[str, Any],
    path: Optional[str] = None,
    content: Optional[str] = None,
    old_text: Optional[str] = None,
    new_text: Optional[str] = None,
    lines: Optional[int] = None,
    directory: Optional[str] = "/",
) -> Dict[str, Any]:
    """Manage filesystem operations in LangGraph state.

    This function works with ANY filesystem type that implements FilesystemProtocol
    (VirtualFilesystem, DiskFilesystem, or future implementations).

    Args:
        action: Action to perform
            - "write": Write file
            - "read": Read file
            - "ls": List directory
            - "edit": Edit file by replacing text
            - "delete": Delete file
            - "exists": Check if file exists
        state: Agent state dict (contains 'filesystem' key)
        path: File path (for write/read/edit/delete/exists)
        content: Content to write (for write)
        old_text: Text to replace (for edit)
        new_text: Replacement text (for edit)
        lines: Number of lines to read (for read)
        directory: Directory to list (for ls)

    Returns:
        Dict with 'success', 'message', 'filesystem', and 'data' keys

    Raises:
        ValueError: If filesystem not in state or required params missing

    Example:
        >>> # Assuming filesystem already initialized in state
        >>> result = manage_filesystem(
        ...     "write", state, path="/notes.txt", content="Hello"
        ... )
        >>> result = manage_filesystem("read", state, path="/notes.txt")
        >>> print(result["data"]["content"])  # "Hello"
    """
    filesystem = state.get("filesystem")

    if not filesystem:
        raise ValueError(
            "No filesystem in state. Initialize filesystem first using " "create_filesystem()"
        )

    if action == "write":
        if not path or content is None:
            raise ValueError("path and content required for write")
        result = filesystem.write(path, content)
        return {
            "success": result["success"],
            "message": result["message"],
            "filesystem": filesystem,
            "data": result,
        }

    elif action == "read":
        if not path:
            raise ValueError("path required for read")
        result = filesystem.read(path, lines=lines)
        return {
            "success": result["success"],
            "message": result["message"],
            "filesystem": filesystem,
            "data": result,
        }

    elif action == "ls":
        result = filesystem.ls(directory)
        return {
            "success": result["success"],
            "message": result["message"],
            "filesystem": filesystem,
            "data": result,
        }

    elif action == "edit":
        if not path or old_text is None or new_text is None:
            raise ValueError("path, old_text, and new_text required for edit")
        result = filesystem.edit(path, old_text, new_text)
        return {
            "success": result["success"],
            "message": result["message"],
            "filesystem": filesystem,
            "data": result,
        }

    elif action == "delete":
        if not path:
            raise ValueError("path required for delete")

        # Check if filesystem has delete method
        if not hasattr(filesystem, "delete"):
            return {
                "success": False,
                "message": "Filesystem does not support delete operation",
                "filesystem": filesystem,
                "data": None,
            }

        result = filesystem.delete(path)
        return {
            "success": result["success"],
            "message": result["message"],
            "filesystem": filesystem,
            "data": result,
        }

    elif action == "exists":
        if not path:
            raise ValueError("path required for exists")
        exists = filesystem.exists(path)
        return {
            "success": True,
            "message": f"File {'exists' if exists else 'does not exist'}: {path}",
            "filesystem": filesystem,
            "data": {"path": path, "exists": exists},
        }

    else:
        raise ValueError(f"Unknown action: {action}")


# ==============================================================
# Convenience Functions (Optional)
# ==============================================================


def write_file(state: Dict[str, Any], path: str, content: str) -> Dict[str, Any]:
    """Convenience function to write a file.

    Args:
        state: Agent state
        path: File path
        content: File content

    Returns:
        Result dict from manage_filesystem
    """
    return manage_filesystem("write", state, path=path, content=content)


def read_file(state: Dict[str, Any], path: str, lines: Optional[int] = None) -> Dict[str, Any]:
    """Convenience function to read a file.

    Args:
        state: Agent state
        path: File path
        lines: Optional line limit

    Returns:
        Result dict from manage_filesystem
    """
    return manage_filesystem("read", state, path=path, lines=lines)


def list_directory(state: Dict[str, Any], directory: str = "/") -> Dict[str, Any]:
    """Convenience function to list directory.

    Args:
        state: Agent state
        directory: Directory path

    Returns:
        Result dict from manage_filesystem
    """
    return manage_filesystem("ls", state, directory=directory)


def edit_file(state: Dict[str, Any], path: str, old_text: str, new_text: str) -> Dict[str, Any]:
    """Convenience function to edit a file.

    Args:
        state: Agent state
        path: File path
        old_text: Text to replace
        new_text: Replacement text

    Returns:
        Result dict from manage_filesystem
    """
    return manage_filesystem("edit", state, path=path, old_text=old_text, new_text=new_text)


def delete_file(state: Dict[str, Any], path: str) -> Dict[str, Any]:
    """Convenience function to delete a file.

    Args:
        state: Agent state
        path: File path

    Returns:
        Result dict from manage_filesystem
    """
    return manage_filesystem("delete", state, path=path)


def file_exists(state: Dict[str, Any], path: str) -> bool:
    """Convenience function to check if file exists.

    Args:
        state: Agent state
        path: File path

    Returns:
        True if file exists, False otherwise
    """
    result = manage_filesystem("exists", state, path=path)
    return result["data"]["exists"]
