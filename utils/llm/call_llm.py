import os
import logging
import re
import random
import time
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()
logger = logging.getLogger(__name__)

from config.timeout_config import timeout_config


class APIOverloadException(Exception):
    """Exception raised when all API keys are overloaded or unavailable"""
    pass


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    vn_chars = len(re.findall(r'[àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđĐ]', text))
    total = len(text)
    return max(1, int(total / (3.2 if vn_chars > total * 0.1 else 3.8)))


def _has_exceeded_timeout(start_time: float, max_retry_time: int) -> bool:
    """Check if current execution has exceeded the maximum retry time"""
    elapsed = time.time() - start_time
    return elapsed > max_retry_time


def _would_exceed_timeout_after_sleep(start_time: float, sleep_duration: float, max_retry_time: int) -> bool:
    """Check if sleeping would cause us to exceed the timeout"""
    elapsed = time.time() - start_time
    return (elapsed + sleep_duration) > max_retry_time


def _calculate_jittered_sleep_time(base_sleep_seconds: float) -> float:
    """Calculate sleep time with jitter to prevent thundering herd"""
    jitter = random.uniform(
        timeout_config.RETRY_JITTER_MIN_SECONDS,
        timeout_config.RETRY_JITTER_MAX_SECONDS
    )
    return max(timeout_config.MIN_COOLDOWN_SECONDS, base_sleep_seconds) + jitter


def call_llm(prompt: str, fast_mode: bool = False, max_retry_time: int = None) -> str:
    """Call LLM with timeout protection and automatic retry logic
    
    Args:
        prompt: The prompt to send to the LLM
        fast_mode: Whether to use fast mode (disables thinking for supported models)
        max_retry_time: Maximum time (seconds) to spend on retries before giving up.
                       Defaults to LLM_RETRY_TIMEOUT from config.
    
    Returns:
        str: The LLM's response text
        
    Raises:
        APIOverloadException: When timeout is exceeded
    """
    if max_retry_time is None:
        max_retry_time = timeout_config.LLM_RETRY_TIMEOUT
    
    model_id = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    logger.info(f"🎯 model: {model_id}")

    # Use single key from environment
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        logger.error("❌ Missing GEMINI_API_KEY")
        return "Xin lỗi, hệ thống chưa cấu hình API key."

    max_attempts = 3  # Simple retry count since we don't have multiple keys
    last_err = None
    start_time = time.time()

    for attempt in range(max_attempts):
        try:
            client = genai.Client(api_key=api_key)

            # Nếu bạn dùng model thinking, mới set thinking_config; còn không thì để None
            cfg = None
            if "thinking" in model_id and not fast_mode:
                cfg = types.GenerateContentConfig(
                    thinking_config=types.ThinkingConfig(thinking_budget=0)
                )

            response = client.models.generate_content(
                model=model_id,
                contents=prompt,
                config=cfg
            )

            text = getattr(response, "text", None)
            if not text:
                # fallback nhẹ nếu SDK không fill .text
                cands = getattr(response, "candidates", None)
                if cands and getattr(cands[0], "content", None) and cands[0].content.parts:
                    text = getattr(cands[0].content.parts[0], "text", None)

            if not text:
                logger.error("❌ Không lấy được text trong response")
                return "Xin lỗi, không thể tạo response."

            logger.info(f"✅ OK, out len={len(text)}, est tokens={estimate_tokens(text)}")
            logger.info(f"📤 preview: {text[:200]}…")
            return text

        except Exception as e:
            es = str(e)
            last_err = es
            logger.error(f"❌ attempt {attempt+1}/{max_attempts} error: {es}")

            es_low = es.lower()

            # Check if we should retry
            should_retry = False
            if "resource_exhausted" in es_low or "429" in es or "quota" in es_low:
                should_retry = True
            elif any(s in es_low for s in ["temporarily unavailable", "overload", "503", "500"]):
                should_retry = True

            # Permanent errors: 401, 403, not found
            if any(code in es for code in ["401", "403"]) or "not_found" in es_low or "model not found" in es_low:
                logger.error("💥 Permanent error encountered")
                raise APIOverloadException(f"Permanent error: {es}")

            if not should_retry and attempt < max_attempts - 1:
                # If not explicitly a transient error but not permanent, we might still retry
                # or we can decide to fail fast. Let's retry for generic errors unless it's the last attempt.
                should_retry = True

            if should_retry:
                 # Check if we've exceeded max retry time
                if _has_exceeded_timeout(start_time, max_retry_time):
                    elapsed = time.time() - start_time
                    logger.error(
                        f"⏱️ Max retry time exceeded: {elapsed:.1f}s > {max_retry_time}s"
                    )
                    raise APIOverloadException(f"Max retry time exceeded: {elapsed:.1f}s")

                # Sleep before retry
                sleep_duration = _calculate_jittered_sleep_time(5.0) # Default 5s base for single key retry

                if _would_exceed_timeout_after_sleep(start_time, sleep_duration, max_retry_time):
                     elapsed = time.time() - start_time
                     logger.error(f"⏱️ Cannot wait {sleep_duration:.1f}s (timeout)")
                     raise APIOverloadException("Timeout reached during retry wait")

                logger.warning(f"⏳ Retrying in {sleep_duration:.1f}s...")
                time.sleep(sleep_duration)
            else:
                break
    
    logger.error(f"💥 Failed after {max_attempts} attempts. Last error: {last_err}")
    return "Xin lỗi, hiện chưa xử lý được yêu cầu."

if __name__ == "__main__":
    print(call_llm("Hello, how are you?", fast_mode=True))
