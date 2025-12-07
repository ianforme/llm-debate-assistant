"""
Operations package for deep preparation workflow.

This package contains the core LLM operations:
- outline: Create debate outline
- evidence: Search for evidence
- draft: Draft opening statement
- evaluation: Evaluate and improve statement
"""

from .outline import create_outline_node_fs
from .evidence import search_evidence_node_fs
from .draft import draft_statement_node_fs
from .evaluation import evaluate_statement_node_fs, improve_statement_node_fs

__all__ = [
    "create_outline_node_fs",
    "search_evidence_node_fs",
    "draft_statement_node_fs",
    "evaluate_statement_node_fs",
    "improve_statement_node_fs",
]
