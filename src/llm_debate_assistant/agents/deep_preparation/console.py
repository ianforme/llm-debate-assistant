"""
Console singleton for deep preparation agent.

This module provides a shared Rich Console instance to avoid circular imports.
"""

from rich.console import Console

# Global console object for consistent output
console = Console()
