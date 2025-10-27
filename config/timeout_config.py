"""
Timeout configuration settings
"""

import os


class TimeoutConfig:
    """Timeout configuration to prevent gateway timeouts"""

    # Cloudflare's timeout is ~100 seconds, we need to respond before that
    CLOUDFLARE_TIMEOUT_SECONDS: int = 100

    # Flow execution timeout - set to 85% of Cloudflare timeout for safety margin
    FLOW_EXECUTION_TIMEOUT: int = int(os.getenv("FLOW_EXECUTION_TIMEOUT", "85"))

    # LLM retry timeout - max time for a single LLM call with all retries
    LLM_RETRY_TIMEOUT: int = int(os.getenv("LLM_RETRY_TIMEOUT", "30"))

    # Jitter range for retry cooldown to prevent thundering herd
    RETRY_JITTER_MIN_SECONDS: float = 0.0
    RETRY_JITTER_MAX_SECONDS: float = 0.5

    # Minimum cooldown time when all API keys are exhausted
    MIN_COOLDOWN_SECONDS: int = 1

    @classmethod
    def get_timeout_message(cls) -> str:
        """Get user-friendly timeout message"""
        return (
            "Xin lỗi, hệ thống đang quá tải và không thể xử lý yêu cầu của bạn "
            "trong thời gian cho phép. Vui lòng thử lại sau ít phút hoặc rút ngắn câu hỏi của bạn."
        )


# Global config instance
timeout_config = TimeoutConfig()
