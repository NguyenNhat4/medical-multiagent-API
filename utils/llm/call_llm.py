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

from utils.auth.APIKeyManager import api_manager, APIOverloadException
from config.timeout_config import timeout_config


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


def _all_keys_cooling_down(status: dict, total_keys: int) -> bool:
    """Check if all usable API keys are currently in cooldown"""
    failed_count = len(status.get("failed", []))
    cooldown_count = len(status.get("cooldowns", {}))
    usable_keys = total_keys - failed_count
    return cooldown_count == usable_keys and usable_keys > 0


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
        APIOverloadException: When all API keys are exhausted or timeout is exceeded
    """
    if max_retry_time is None:
        max_retry_time = timeout_config.LLM_RETRY_TIMEOUT
    
    model_id = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    logger.info(f"🎯 model: {model_id}")

    max_attempts = max(1, len(api_manager.api_keys))
    last_err = None
    start_time = time.time()

    for attempt in range(max_attempts):
        # 1) Chọn 1 key khả dụng cho lần thử này
        key, idx = api_manager.pick_key()

        try:
            client = genai.Client(api_key=key)

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

            logger.info(f"✅ key {idx} OK, out len={len(text)}, est tokens={estimate_tokens(text)}")
            logger.info(f"📤 preview: {text[:200]}…")
            return text

        except Exception as e:
            es = str(e)
            last_err = es
            logger.error(f"❌ key {idx} error: {es}")

            es_low = es.lower()

            # 2) Phân loại lỗi và đánh dấu trạng thái key
            if "resource_exhausted" in es_low or "429" in es or "quota" in es_low:
                # Quota hết → cooldown theo RetryInfo/Retry-After
                api_manager.mark_quota_exhausted(idx, err_msg=es)

            elif any(s in es_low for s in ["temporarily unavailable", "overload", "503", "500"]):
                # Lỗi tạm thời → cooldown ngắn
                api_manager.mark_transient_error(idx)

            elif any(code in es for code in ["401", "403"]) or "not_found" in es_low or "model not found" in es_low:
                # Key hỏng / model sai → loại khỏi pool
                api_manager.mark_permanent_fail(idx)

            else:
                # Lỗi khác → cooldown ngắn để tránh spam
                api_manager.mark_transient_error(idx)

            # 3) Handle cooldown when all keys are temporarily unavailable
            st = api_manager.status()
            if _all_keys_cooling_down(st, len(api_manager.api_keys)):
                # Calculate how long to wait with jitter
                min_cooldown = min(st["cooldowns"].values()) if st["cooldowns"] else timeout_config.MIN_COOLDOWN_SECONDS
                sleep_duration = _calculate_jittered_sleep_time(min_cooldown)
                
                # Check if sleeping would exceed our timeout budget
                if _would_exceed_timeout_after_sleep(start_time, sleep_duration, max_retry_time):
                    elapsed = time.time() - start_time
                    logger.error(
                        f"⏱️ Cannot wait {sleep_duration:.1f}s (would exceed max retry time of {max_retry_time}s). "
                        f"Elapsed: {elapsed:.1f}s"
                    )
                    raise APIOverloadException("All API keys cooling down and max retry time reached")
                
                elapsed = time.time() - start_time
                logger.warning(
                    f"⏳ All keys cooling down. Sleeping {sleep_duration:.1f}s... "
                    f"(elapsed: {elapsed:.1f}s / {max_retry_time}s)"
                )
                time.sleep(sleep_duration)

            # Check if we've exceeded max retry time after attempting retry
            if _has_exceeded_timeout(start_time, max_retry_time):
                elapsed = time.time() - start_time
                logger.error(
                    f"⏱️ Max retry time exceeded: {elapsed:.1f}s > {max_retry_time}s"
                )
                raise APIOverloadException(f"Max retry time exceeded: {elapsed:.1f}s")
            
            # Continue to next attempt with a different key or after cooldown

    # Hết attempts - kiểm tra xem có phải do tất cả keys đều overload không
    st = api_manager.status()
    available_keys = len(api_manager.api_keys) - len(st.get("failed", []))
    if available_keys == 0:
        logger.error("💥 All API keys are permanently failed")
        raise APIOverloadException("All API keys are permanently failed")
    elif len(st.get("cooldowns", {})) == available_keys:
        logger.error("💥 All API keys are in cooldown")
        raise APIOverloadException("All API keys are in cooldown")
    
    logger.error(f"💥 Failed after {max_attempts} attempts. Last error: {last_err}")
    return "Xin lỗi, hiện chưa xử lý được yêu cầu."

if __name__ == "__main__":
    print(call_llm("Hello, how are you?", fast_mode=True))
