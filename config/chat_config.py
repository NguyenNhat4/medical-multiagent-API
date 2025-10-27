"""
Chat-related configuration settings
"""

import os
from typing import Optional


class ChatConfig:
    """Configuration for chat-related settings"""

    # Pagination settings
    DEFAULT_PAGE_SIZE: int = 50
    MAX_PAGE_SIZE: int = 200
    MIN_PAGE_SIZE: int = 1

    # Message settings
    MAX_MESSAGE_LENGTH: int = 10000
    MAX_THREAD_NAME_LENGTH: int = 255

    # Default messages
    DEFAULT_WELCOME_MESSAGE: str = "Xin chào 😊! Tôi là trợ lý AI của bạn. Rất vui được hỗ trợ bạn - Bạn cần tôi giúp gì hôm nay?"

    @classmethod
    def get_welcome_message(cls) -> str:
        """Get welcome message from environment or use default"""
        return os.getenv("WELCOME_MESSAGE", cls.DEFAULT_WELCOME_MESSAGE)


# Global config instance
chat_config = ChatConfig()
