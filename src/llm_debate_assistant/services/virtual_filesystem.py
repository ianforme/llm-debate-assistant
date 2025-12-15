"""Virtual/in-memory filesystem implementation.

This filesystem stores files in a Python dictionary (memory only).
Data is transient and lost when the process ends.

"""

from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    pass


class VirtualFilesystem:
    """Virtual/Transient filesystem stored in memory/state.

    Implements FilesystemProtocol for in-memory storage.

    All files are stored in a Python dictionary and lost when the
    process ends. No disk I/O is performed.
    """

    def __init__(self) -> None:
        self.files: Dict[str, str] = {}

    def write(self, path: str, content: str) -> Dict[str, Any]:
        """Write content to a virtual file.

        Args:
            path (str): File path (e.g., "/folder/file.txt")
            content (str): File content to write

        Returns:
            Dict[str, Any]: Result dictionary with success status and message

        Example:
            >>> vfs = VirtualFilesystem()
            >>> result = vfs.write("/example.txt", "Hello, World!")
            >>> print(result)
            {'success': True, 'message': 'Wrote 13 chars to /example.txt',
            'path': '/example.txt', 'size': 13}
        """

        if not path.startswith("/"):
            return {
                "success": False,
                "message": f"Path must start with /: {path}",
                "path": path,
            }

        self.files[path] = content
        return {
            "success": True,
            "message": f"Wrote {len(content)} chars to {path}",
            "path": path,
            "size": len(content),
        }

    def read(self, path: str, lines: Optional[int] = None) -> Dict[str, Any]:
        """Read content from a virtual file.

        Args:
            path (str): File path to read
            lines (Optional[int], optional): Number of lines to read
                (None = read all). Defaults to None.

        Returns:
            Dict[str, Any]: Result dictionary with success status, message, and content

        Example:
            >>> vfs = VirtualFilesystem()
            >>> vfs.write("/example.txt", "Line1\\nLine2\\nLine3")
            >>> result = vfs.read("/example.txt", lines=2)
            >>> print(result)
            {'success': True, 'message': 'Read /example.txt (12 chars)',
            'path': '/example.txt', 'content': 'Line1\\nLine2'}
        """
        if path not in self.files:
            return {
                "success": False,
                "message": f"File not found: {path}",
                "path": path,
                "content": None,
            }

        content = self.files[path]

        # Optionally limit to N lines
        if lines is not None:
            content_lines = content.split("\n")
            content = "\n".join(content_lines[:lines])

        return {
            "success": True,
            "message": f"Read {path} ({len(content)} chars)",
            "path": path,
            "content": content,
        }

    def ls(self, directory: str = "/") -> Dict[str, Any]:
        """List files in a directory.

        Args:
            directory (str, optional): Directory path to list. Defaults to "/".

        Returns:
            Dict[str, Any]: Result dictionary with success status, message,
                and list of items

        Example:
            >>> vfs = VirtualFilesystem()
            >>> vfs.write("/folder1/file1.txt", "Content1")
            >>> vfs.write("/folder1/file2.txt", "Content2")
            >>> vfs.write("/folder2/file3.txt", "Content3")
            >>> result = vfs.ls("/folder1")
            >>> print(result)
            {'success': True, 'message': 'Listed 2 items in /folder1/',
            'directory': '/folder1/', 'items': ['file1.txt', 'file2.txt']}
        """
        if not directory.startswith("/"):
            directory = "/" + directory

        if not directory.endswith("/"):
            directory = directory + "/"

        # Find all files that start with the directory path
        matching_files = []
        subdirs = set()

        for path in self.files.keys():
            if path.startswith(directory):
                # Get the relative path after the directory
                relative = path[len(directory) :]

                # If it contains /, it's in a subdirectory
                if "/" in relative:
                    subdir = relative.split("/")[0]
                    subdirs.add(subdir + "/")
                else:
                    # It's a file in this directory
                    matching_files.append(relative)

        # Combine subdirs and files
        items = sorted(subdirs) + sorted(matching_files)

        return {
            "success": True,
            "message": f"Listed {len(items)} items in {directory}",
            "directory": directory,
            "items": items,
        }

    def edit(self, path: str, old_text: str, new_text: str) -> Dict[str, Any]:
        """Edit an existing file by replacing text.

        Args:
            path (str): File path to edit
            old_text (str): Text to find and replace
            new_text (str): Replacement text

        Returns:
            Dict[str, Any]: Result dictionary with success status,
            message, and number of replacements

        Example:
            >>> vfs = VirtualFilesystem()
            >>> vfs.write("/example.txt", "Hello, World! Hello!")
            >>> result = vfs.edit("/example.txt", "Hello", "Hi")
            >>> print(result)
            {'success': True,
             'message': 'Edited /example.txt (replaced 2 occurrence(s))',
             'path': '/example.txt', 'replacements': 2}
        """
        if path not in self.files:
            return {
                "success": False,
                "message": f"File not found: {path}",
                "path": path,
            }

        content = self.files[path]

        if old_text not in content:
            return {
                "success": False,
                "message": f"Text not found in {path}: {old_text[:50]}...",
                "path": path,
            }

        # Replace all occurrences
        num_replacements = content.count(old_text)
        new_content = content.replace(old_text, new_text)
        self.files[path] = new_content

        return {
            "success": True,
            "message": f"Edited {path} (replaced {num_replacements} occurrence(s))",
            "path": path,
            "replacements": num_replacements,
        }

    def delete(self, path: str) -> Dict[str, Any]:
        """Delete a file.

        Args:
            path (str): File path to delete

        Returns:
            Dict[str, Any]: Result dictionary with success status and message

        Example:
            >>> vfs = VirtualFilesystem()
            >>> vfs.write("/example.txt", "Some content")
            >>> result = vfs.delete("/example.txt")
            >>> print(result)
            {'success': True, 'message': 'Deleted /example.txt',
            'path': '/example.txt'}
        """
        if path not in self.files:
            return {
                "success": False,
                "message": f"File not found: {path}",
                "path": path,
            }

        del self.files[path]
        return {
            "success": True,
            "message": f"Deleted {path}",
            "path": path,
        }

    def exists(self, path: str) -> bool:
        """Check if a file exists.

        Args:
            path (str): File path to check

        Returns:
            bool: True if file exists, False otherwise

        Example:
            >>> vfs = VirtualFilesystem()
            >>> vfs.write("/example.txt", "Content")
            >>> print(vfs.exists("/example.txt"))
            True
            >>> print(vfs.exists("/nonexistent.txt"))
            False
        """
        return path in self.files

    def get_all_files(self) -> List[str]:
        """Get list of all file paths.

        Returns:
            List[str]: List of all file paths

        Example:
            >>> vfs = VirtualFilesystem()
            >>> vfs.write("/file1.txt", "Content1")
            >>> vfs.write("/file2.txt", "Content2")
            >>> print(vfs.get_all_files())
            ['/file1.txt', '/file2.txt']
        """
        return sorted(self.files.keys())
