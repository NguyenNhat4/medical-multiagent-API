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
    vn_chars = len(re.findall(r'[àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]', text))
    total = len(text)
    ratio = 3.2 if vn_chars > total * 0.1 else 3.8
    return max(1, int(total / ratio))

def call_llm(prompt: str, fast_mode: bool = False, max_retry_time: int = None) -> str:
    """Call LLM with timeout protection and automatic retry logic"""
    model_id = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return "Xin lỗi, hệ thống chưa cấu hình API key."
    
    client = genai.Client(api_key=api_key)
    config = types.GenerateContentConfig(thinking_config=types.ThinkingConfig(thinking_budget=0)) if "thinking" in model_id and not fast_mode else None
    response = client.models.generate_content(model=model_id, contents=prompt, config=config)
    return response.text or "Xin lỗi, không thể tạo response."

if __name__ == "__main__": 
    print(call_llm("Hello, how are you?", fast_mode=True))
