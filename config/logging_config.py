"""
Logging configuration settings
"""

import os
from typing import Optional


class LoggingConfig:
    """Configuration for logging"""

    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FORMAT: str = "%(asctime)s [VN] - %(name)s - %(levelname)s - %(message)s"

    # Log file settings
    LOG_FILE: Optional[str] = os.getenv("LOG_FILE")
    MAX_LOG_SIZE: int = int(os.getenv("MAX_LOG_SIZE", "10485760"))  # 10MB
    LOG_BACKUP_COUNT: int = int(os.getenv("LOG_BACKUP_COUNT", "5"))

    # Timezone settings
    USE_VIETNAM_TIMEZONE: bool = os.getenv("USE_VIETNAM_TIMEZONE", "true").lower() == "true"


# Global config instance
logging_config = LoggingConfig()
