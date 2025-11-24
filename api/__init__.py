"""
API package - Modular API route definitions
"""

from .auth import router as auth_router
from .users import router as users_router
from .health import router as health_router
from .chat import router as chat_router
from .threads import router as threads_router
from .embeddings import router as embeddings_router
from .retrieval import router as retrieval_router

__all__ = [
    "auth_router",
    "users_router",
    "health_router",
    "chat_router",
    "threads_router",
    "embeddings_router",
    "retrieval_router",
]
