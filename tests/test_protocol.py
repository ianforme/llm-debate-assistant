import pytest

from llm_debate_assistant.services.disk_filesystem import DiskFilesystem
from llm_debate_assistant.services.filesystem_protocol import FilesystemProtocol
from llm_debate_assistant.services.virtual_filesystem import VirtualFilesystem


REQUIRED_METHODS = {"write", "read", "exists", "ls", "edit", "delete"}


@pytest.fixture
def virtual_filesystem():
    """Create a VirtualFilesystem instance."""
    return VirtualFilesystem()


@pytest.fixture
def disk_filesystem():
    """Create a DiskFilesystem instance with cleanup."""
    fs = DiskFilesystem(session_id="test-protocol")
    yield fs  # use yield instead of return to allow cleanup
    fs.cleanup()  # teardown


class TestProtocolCompliance:
    """Tests for protocol compliance verification."""

    @pytest.mark.parametrize("cls", [VirtualFilesystem, DiskFilesystem])
    def test_has_required_methods(self, cls):
        """Test that filesystem classes have all required methods."""
        for method_name in REQUIRED_METHODS:
            assert hasattr(cls, method_name), f"{cls.__name__} missing method: {method_name}"
            assert callable(
                getattr(cls, method_name)
            ), f"{cls.__name__}.{method_name} is not callable"

    def test_virtual_filesystem_isinstance(self, virtual_filesystem):
        """Test that VirtualFilesystem is instance of FilesystemProtocol."""
        assert isinstance(virtual_filesystem, FilesystemProtocol)

    def test_disk_filesystem_isinstance(self, disk_filesystem):
        """Test that DiskFilesystem is instance of FilesystemProtocol."""
        assert isinstance(disk_filesystem, FilesystemProtocol)


class TestFilesystemOperations:
    """Tests for filesystem operations."""

    @pytest.fixture(params=["virtual", "disk"])
    def filesystem(self, request, virtual_filesystem, disk_filesystem):
        """Parametrized fixture for both filesystem types."""
        if request.param == "virtual":
            return virtual_filesystem
        return disk_filesystem

    def test_write(self, filesystem):
        """Test write operation."""
        result = filesystem.write("/test.txt", "Hello, World!")
        assert result["success"] is True

    def test_exists_after_write(self, filesystem):
        """Test exists returns True after writing a file."""
        filesystem.write("/test.txt", "Hello, World!")
        assert filesystem.exists("/test.txt") is True

    def test_exists_nonexistent(self, filesystem):
        """Test exists returns False for nonexistent file."""
        assert filesystem.exists("/nonexistent.txt") is False

    def test_read(self, filesystem):
        """Test read operation."""
        filesystem.write("/test.txt", "Hello, World!")
        result = filesystem.read("/test.txt")
        assert result["success"] is True
        assert result["content"] == "Hello, World!"

    def test_ls(self, filesystem):
        """Test ls operation."""
        filesystem.write("/test.txt", "Hello, World!")
        result = filesystem.ls("/")
        assert result["success"] is True
        assert "test.txt" in result["items"]

    def test_edit(self, filesystem):
        """Test edit operation."""
        filesystem.write("/test.txt", "Hello, World!")
        result = filesystem.edit("/test.txt", "Hello", "Hi")
        assert result["success"] is True
        assert result["replacements"] == 1

        # Verify the edit was applied
        read_result = filesystem.read("/test.txt")
        assert read_result["content"] == "Hi, World!"

    def test_delete(self, filesystem):
        """Test delete operation."""
        filesystem.write("/test.txt", "Hello, World!")
        result = filesystem.delete("/test.txt")
        assert result["success"] is True
        assert filesystem.exists("/test.txt") is False
