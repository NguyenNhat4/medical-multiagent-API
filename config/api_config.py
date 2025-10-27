"""
API configuration settings
"""

import os


class APIConfig:
    """Configuration for API settings"""

    # API versioning
    API_V1_PREFIX: str = "/api"

    # Rate limiting (if implemented)
    RATE_LIMIT_PER_MINUTE: int = int(os.getenv("RATE_LIMIT_PER_MINUTE", "60"))

    # CORS settings
    ALLOWED_ORIGINS: list = os.getenv("ALLOWED_ORIGINS", "*").split(",")

    # Security
    ACCESS_TOKEN_EXPIRE_MINUTES: int = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))


# Global config instance
api_config = APIConfig()
