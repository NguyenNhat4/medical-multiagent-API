import os
import logging
from google import genai
from dotenv import load_dotenv
load_dotenv()
# Configure logging for this module
logger = logging.getLogger(__name__)
from google.genai import types

# Initialize client at module level to avoid overhead on each call
api_key = os.getenv("GEMINI_API_KEY", "")
if not api_key:
    logger.error(" GEMINI_API_KEY is not set")
    raise RuntimeError("GEMINI_API_KEY is not set")

client = genai.Client(api_key=api_key)

# Learn more about calling the LLM: https://the-pocket.github.io/PocketFlow/utility_function/llm.html
def call_llm(prompt: str, fast_mode: bool = True) -> str:
    logger.info("🤖 Bắt đầu gọi LLM...")
    
    # Dynamic model selection based on fast_mode
    if fast_mode:
        model = os.getenv("GEMINI_MODEL", "gemini-1.5-flash-8b")  # Super fast
    else:
        model = os.getenv("GEMINI_MODEL_QUALITY", "gemini-2.5-flash")  # High quality
    
    
    try:
        logger.info("⏳ Đang gửi request đến Gemini...")
        response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.5,
            top_p=0.9
        ),
    )
            
        response_text = response.text
        
        # Handle case where response.text is None
        if response_text is None:
            logger.warning("⚠️ Response text is None, checking response object...")
            logger.info(f"Response object: {response}")
            if hasattr(response, 'candidates') and response.candidates:
                logger.info(f"Candidates: {response.candidates}")
                if response.candidates[0].content and response.candidates[0].content.parts:
                    response_text = response.candidates[0].content.parts[0].text
                    logger.info(f"✅ Extracted text from candidates: {len(response_text)} characters")
                else:
                    logger.error("❌ No text found in candidates")
                    response_text = "Xin lỗi, không thể tạo response."
            else:
                logger.error("❌ No candidates in response")
                response_text = "Xin lỗi, không thể tạo response."
        
        if response_text:
            logger.info(f"✅ Nhận được response từ LLM: {len(response_text)} characters")
            logger.info(f"📤 Response preview: {response_text}")

        return response_text or "Xin lỗi, không thể tạo response."
        
    except Exception as e:
        logger.error(f"❌ Lỗi khi gọi LLM: {str(e)}")
        raise

if __name__ == "__main__":
    test_prompt = "Hello, how are you?"
    print(call_llm(test_prompt))
