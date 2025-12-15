# -*- coding: utf-8 -*-
"""
Utility functions for constructive speech operations.
"""


def count_visible_chars(text: str) -> int:
    """Count visible characters (excluding spaces, newlines, tabs).

    This is the standard for debate character limits - only counts
    Chinese characters, punctuation, English letters, and numbers.

    Args:
        text (str): The text to count.

    Returns:
        int: The count of visible characters.
    """
    return len("".join(c for c in text if not c.isspace()))
