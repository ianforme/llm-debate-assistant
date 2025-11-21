"""Disk-based filesystem for LangGraph agents.

Stores files in actual temp directory instead of memory,
avoiding state bloat and enabling inspection with normal tools.
"""

import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional
from uuid import uuid4

if TYPE_CHECKING:
    pass


class DiskFilesystem:
    """Filesystem that writes to actual disk in temp directory.

    Implements FilesystemProtocol for persistent disk storage.

    Files are written to .temp/debate_preparation_sessions/<session_id>/ and
    persist after the process ends.
    """

    def __init__(
        self, root_dir: Optional[Path] = None, session_id: Optional[str] = None
    ):
        """Initialize disk filesystem.

        Args:
            root_dir (Optional[Path]): Root directory for all sessions
                (default: .temp/debate_preparation_sessions)
            session_id (Optional[str]): Unique session ID
                (default: auto-generated UUID)
        """
        if root_dir is None:
            # Use .temp/debate_preparation_sessions in project directory
            root_dir = Path.cwd() / ".temp" / "debate_preparation_sessions"

        if session_id is None:
            session_id = str(uuid4())  # use fully random UUID

        self.session_id = session_id
        self.root_dir = Path(root_dir) / session_id
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def _resolve_path(self, path: str) -> Path:
        """Convert virtual path to actual disk path.

        Args:
            path (str): Virtual path like "/outline.md"

        Returns:
            Path: Actual path like ".temp/debate_preparation_sessions/abc123/outline.md"
        """
        # Remove leading slash and resolve relative to root
        relative_path = path.lstrip("/")
        return self.root_dir / relative_path

    def write(self, path: str, content: str) -> Dict[str, Any]:
        """Write content to a file on disk.

        Args:
            path (str): File path (e.g., "/research/notes.txt")
            content (str): File content to write

        Returns:
            Dict with success status and message
        """
        if not path.startswith("/"):
            return {
                "success": False,
                "message": f"Path must start with /: {path}",
                "path": path,
            }

        try:
            disk_path = self._resolve_path(path)
            disk_path.parent.mkdir(parents=True, exist_ok=True)
            disk_path.write_text(content, encoding="utf-8")

            return {
                "success": True,
                "message": f"Wrote {len(content)} chars to {path}",
                "path": path,
                "disk_path": str(disk_path),
                "size": len(content),
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"Failed to write {path}: {e}",
                "path": path,
            }

    def read(self, path: str, lines: Optional[int] = None) -> Dict[str, Any]:
        """Read content from a file on disk.

        Args:
            path (str): File path to read
            lines (Optional[int]): Optional number of lines to read (None = read all)

        Returns:
            Dict with success status, message, and content
        """
        disk_path = self._resolve_path(path)

        if not disk_path.exists():
            return {
                "success": False,
                "message": f"File not found: {path}",
                "path": path,
                "content": None,
            }

        try:
            content = disk_path.read_text(encoding="utf-8")

            # Optionally limit to N lines
            if lines is not None:
                content_lines = content.split("\n")
                content = "\n".join(content_lines[:lines])

            return {
                "success": True,
                "message": f"Read {path} ({len(content)} chars)",
                "path": path,
                "disk_path": str(disk_path),
                "content": content,
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"Failed to read {path}: {e}",
                "path": path,
                "content": None,
            }

    def ls(self, directory: str = "/") -> Dict[str, Any]:
        """List files in a directory.

        Args:
            directory (str): Directory path to list (default: "/")

        Returns:
            Dict[str, Any]: Dict with success status, message, and list of files
        """
        disk_path = self._resolve_path(directory)

        if not disk_path.exists():
            return {
                "success": True,
                "message": f"Directory does not exist: {directory}",
                "directory": directory,
                "items": [],
            }

        try:
            items = []
            for item in sorted(disk_path.iterdir()):
                if item.is_dir():
                    items.append(item.name + "/")
                else:
                    items.append(item.name)

            return {
                "success": True,
                "message": f"Listed {len(items)} items in {directory}",
                "directory": directory,
                "disk_path": str(disk_path),
                "items": items,
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"Failed to list {directory}: {e}",
                "directory": directory,
                "items": [],
            }

    def edit(self, path: str, old_text: str, new_text: str) -> Dict[str, Any]:
        """Edit a file by replacing text.

        Args:
            path (str): File path to edit
            old_text (str): Text to find and replace
            new_text (str): Replacement text

        Returns:
            Dict[str, Any]: Dict with success status and message
        """
        disk_path = self._resolve_path(path)

        if not disk_path.exists():
            return {
                "success": False,
                "message": f"File not found: {path}",
                "path": path,
            }

        try:
            content = disk_path.read_text(encoding="utf-8")

            if old_text not in content:
                return {
                    "success": False,
                    "message": f"Text not found in {path}: {old_text[:50]}...",
                    "path": path,
                }

            # Replace all occurrences
            new_content = content.replace(old_text, new_text)
            disk_path.write_text(new_content, encoding="utf-8")

            num_replacements = content.count(old_text)
            return {
                "success": True,
                "message": f"Edited {path} (replaced {num_replacements} occurrence(s))",
                "path": path,
                "disk_path": str(disk_path),
                "replacements": num_replacements,
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"Failed to edit {path}: {e}",
                "path": path,
            }

    def delete(self, path: str) -> Dict[str, Any]:
        """Delete a file at the given path.

        Args:
            path (str): File path to delete

        Returns:
            Dict[str, Any]: Dict with success status and message
        """
        disk_path = self._resolve_path(path)

        if not disk_path.exists():
            return {
                "success": False,
                "message": f"File not found: {path}",
                "path": path,
            }

        try:
            disk_path.unlink()
            return {
                "success": True,
                "message": f"Deleted {path}",
                "path": path,
                "disk_path": str(disk_path),
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"Failed to delete {path}: {e}",
                "path": path,
            }

    def exists(self, path: str) -> bool:
        """Check if a file exists at the given path.

        Args:
            path (str): File path to check

        Returns:
            bool: True if file exists, False otherwise
        """
        disk_path = self._resolve_path(path)
        return disk_path.exists()

    def get_all_files(self) -> List[str]:
        """Get list of all file paths (recursively).

        Returns:
            List[str]: List of all file paths relative to root
        """
        files = []
        for path in self.root_dir.rglob("*"):
            if path.is_file():
                # Convert to virtual path
                relative = path.relative_to(self.root_dir)
                virtual_path = "/" + str(relative)
                files.append(virtual_path)
        return sorted(files)

    def cleanup(self) -> None:
        """Delete the entire session directory."""
        if self.root_dir.exists():
            shutil.rmtree(self.root_dir)

    def get_session_path(self) -> str:
        """Get the absolute path to the session directory.

        Returns:
            str: Absolute path as string
        """
        return str(self.root_dir.absolute())
