"""
Knowledge Base utilities - retrieval and indexing
"""

from .kb import (
    KnowledgeBaseIndex,
    get_kb,
    retrieve,
    retrieve_random_by_role,
    KB_COLUMNS,
    ROLE_TO_CSV,
)
from .kb_oqa import (
    preload_oqa_index,
    is_oqa_index_loaded,
)

__all__ = [
    "KnowledgeBaseIndex",
    "get_kb",
    "retrieve",
    "retrieve_random_by_role",
    "KB_COLUMNS",
    "ROLE_TO_CSV",
    "preload_oqa_index",
    "is_oqa_index_loaded",
]
