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
from .APIKeyManager import api_manager, APIOverloadException


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    vn_chars = len(re.findall(r'[àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđĐ]', text))
    total = len(text)
    return max(1, int(total / (3.2 if vn_chars > total * 0.1 else 3.8)))




def call_llm(prompt: str, fast_mode: bool = False) -> str:
    model_id = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    logger.info(f"🎯 model: {model_id}")

    max_attempts = max(1, len(api_manager.api_keys))  # thử tối đa = số key
    last_err = None

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

            # 3) Nếu tất cả key đang cooldown: ngủ đến khi key gần nhất hết cooldown (thêm jitter)
            st = api_manager.status()
            if len(st.get("cooldowns", {})) == len(api_manager.api_keys) - len(st.get("failed", [])):
                # tất cả usable key đều đang cooldown
                wait_secs = min(st["cooldowns"].values()) if st["cooldowns"] else 1
                sleep_for = max(1, wait_secs) + random.uniform(0, 0.5)
                logger.warning(f"⏳ All keys cooling down. Sleeping {sleep_for:.1f}s…")
                time.sleep(sleep_for)

            # Tiếp tục vòng for: sẽ pick key khác (hoặc key vừa hết cooldown)

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
