import os
import logging
import re
import random
from google import genai
from dotenv import load_dotenv
load_dotenv()
# Configure logging for this module
logger = logging.getLogger(__name__)
from google.genai import types

class APIOverloadException(Exception):
    """Exception raised when API is overloaded or quota exceeded"""
    pass

class APIKeyManager:
    """Manages multiple API keys and handles rotation when overload occurs"""
    
    def __init__(self):
        self.api_keys = []
        self.current_key_index = 0
        self.failed_keys = set()  # Track temporarily failed keys
        
        # Load API keys from environment
        self._load_api_keys()
        
        # Initialize clients for all keys
        self.clients = {}
        self._initialize_clients()
    
    def _load_api_keys(self):
        """Load API keys from environment variables"""
        # Try new multi-key format first
        multi_keys = os.getenv("GEMINI_API_KEYS", "")
        if multi_keys:
            self.api_keys = [key.strip() for key in multi_keys.split(",") if key.strip()]
            logger.info(f"🔑 Loaded {len(self.api_keys)} API keys from GEMINI_API_KEYS")
        else:
            # Fallback to single key format
            single_key = os.getenv("GEMINI_API_KEY", "")
            if single_key:
                self.api_keys = [single_key]
                logger.info("🔑 Loaded 1 API key from GEMINI_API_KEY")
            else:
                logger.error("❌ No API keys found. Set GEMINI_API_KEYS or GEMINI_API_KEY")
                raise RuntimeError("No Gemini API keys configured")
        
        # Shuffle keys for load balancing
        random.shuffle(self.api_keys)
        logger.info(f"🔀 API keys shuffled for load balancing")
    
    def _initialize_clients(self):
        """Initialize Gemini clients for all API keys"""
        for i, api_key in enumerate(self.api_keys):
            try:
                self.clients[i] = genai.Client(api_key=api_key)
                logger.info(f"✅ Client {i} initialized successfully")
            except Exception as e:
                logger.error(f"❌ Failed to initialize client {i}: {e}")
    
    def get_current_client(self):
        """Get the current active client"""
        available_indices = [i for i in range(len(self.api_keys)) if i not in self.failed_keys]
        
        if not available_indices:
            # Reset failed keys if all are failed (maybe they recovered)
            logger.warning("⚠️ All keys failed, resetting failed keys set")
            self.failed_keys.clear()
            available_indices = list(range(len(self.api_keys)))
        
        # Use current index if available, otherwise pick first available
        if self.current_key_index in available_indices:
            return self.clients[self.current_key_index], self.current_key_index
        else:
            self.current_key_index = available_indices[0]
            return self.clients[self.current_key_index], self.current_key_index
    
    def mark_key_failed(self, key_index):
        """Mark a key as temporarily failed"""
        self.failed_keys.add(key_index)
        logger.warning(f"🚫 Marked API key {key_index} as failed")
    
    def switch_to_next_key(self):
        """Switch to the next available API key"""
        available_indices = [i for i in range(len(self.api_keys)) if i not in self.failed_keys]
        
        if len(available_indices) <= 1:
            logger.warning("⚠️ No more API keys available for switching")
            return False
        
        # Remove current key from available and pick next
        if self.current_key_index in available_indices:
            available_indices.remove(self.current_key_index)
        
        old_index = self.current_key_index
        self.current_key_index = available_indices[0]
        logger.info(f"🔄 Switched from API key {old_index} to {self.current_key_index}")
        return True

# Initialize the API key manager
api_manager = APIKeyManager()

def reset_failed_keys():
    """Reset all failed keys - useful for periodic recovery attempts"""
    api_manager.failed_keys.clear()
    logger.info("🔄 Reset all failed API keys - attempting recovery")

def get_api_key_status():
    """Get current status of all API keys"""
    total_keys = len(api_manager.api_keys)
    failed_keys = len(api_manager.failed_keys)
    active_keys = total_keys - failed_keys
    current_key = api_manager.current_key_index
    
    return {
        "total_keys": total_keys,
        "active_keys": active_keys,
        "failed_keys": failed_keys,
        "current_key_index": current_key,
        "failed_key_indices": list(api_manager.failed_keys)
    }


def estimate_tokens(text: str) -> int:
    """
    Estimate token count for Vietnamese and English text.
    Uses a simple heuristic: ~4 characters per token for Vietnamese, ~3.5 for English.
    This is a rough approximation for logging purposes.
    """
    if not text:
        return 0
    
    # Count Vietnamese characters (with diacritics)
    vietnamese_chars = len(re.findall(r'[àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđĐ]', text))
    # Count total characters
    total_chars = len(text)
    
    # Rough estimation: Vietnamese text tends to have more tokens per character
    if vietnamese_chars > total_chars * 0.1:  # If >10% Vietnamese chars
        estimated_tokens = int(total_chars / 3.2)  # Vietnamese: ~3.2 chars per token
    else:
        estimated_tokens = int(total_chars / 3.8)  # English: ~3.8 chars per token
    
    return max(1, estimated_tokens)


# Learn more about calling the LLM: https://the-pocket.github.io/PocketFlow/utility_function/llm.html
def call_llm(prompt: str, fast_mode: bool = True) -> str:
    logger.info("🤖 Bắt đầu gọi LLM...")
    
    # Log token count before API call
    estimated_tokens = estimate_tokens(prompt)
    logger.info(f"📊 Estimated input tokens: {estimated_tokens} (prompt length: {len(prompt)} chars)")
    
    # Dynamic model selection based on fast_mode
    if fast_mode:
        model = os.getenv("GEMINI_MODEL", "gemini-1.5-flash-8b")  # Super fast
    else:
        model = os.getenv("GEMINI_MODEL_QUALITY", "gemini-2.5-flash")  # High quality
    
    logger.info(f"🎯 Using model: {model}")
    
    # Try with current API key, with automatic switching on overload
    max_retries = len(api_manager.api_keys)  # Try all available keys
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            # Get current client and key index
            current_client, key_index = api_manager.get_current_client()
            logger.info(f"⏳ Đang gửi request đến Gemini với API key {key_index}... (attempt {retry_count + 1}/{max_retries})")
            
            response = current_client.models.generate_content(
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
                output_tokens = estimate_tokens(response_text)
                logger.info(f"✅ Nhận được response từ LLM với API key {key_index}: {len(response_text)} characters")
                logger.info(f"📊 Estimated output tokens: {output_tokens}")
                logger.info(f"📤 Response preview: {response_text[:200]}...")
                logger.info(f"🔍 Full response for debugging: {response_text}")

            return response_text or "Xin lỗi, không thể tạo response."
            
        except Exception as e:
            error_str = str(e)
            logger.error(f"❌ Lỗi khi gọi LLM với API key {key_index}: {error_str}")
            
            # Handle quota exceeded and API overload specifically
            if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str or "quota" in error_str.lower():
                logger.warning(f"⚠️ API key {key_index} quota exceeded or overloaded")
                api_manager.mark_key_failed(key_index)
                
                # Try to switch to next key
                if api_manager.switch_to_next_key():
                    retry_count += 1
                    logger.info(f"🔄 Switching to next API key, retry {retry_count}/{max_retries}")
                    continue
                else:
                    logger.error("❌ No more API keys available, all quota exceeded")
                    raise APIOverloadException("All API keys quota exceeded")
            
            # Handle other API errors
            elif "500" in error_str or "503" in error_str or "overload" in error_str.lower():
                logger.warning(f"⚠️ API server overloaded for key {key_index}")
                api_manager.mark_key_failed(key_index)
                
                # Try to switch to next key
                if api_manager.switch_to_next_key():
                    retry_count += 1
                    logger.info(f"🔄 Switching to next API key, retry {retry_count}/{max_retries}")
                    continue
                else:
                    logger.error("❌ No more API keys available, all servers overloaded")
                    raise APIOverloadException("All API servers overloaded")
            else:
                # For other errors, don't mark key as failed, just return fallback
                logger.warning("⚠️ Unexpected API error. Using fallback response.")
                return "Xin lỗi hiện tại mình đang bị quá tải bạn chờ một chút nhé."
    
    # If we've exhausted all retries
    logger.error("❌ Exhausted all API key retries")
    return "Xin lỗi hiện tại mình đang bị quá tải bạn chờ một chút nhé."

if __name__ == "__main__":
    test_prompt = "Hello, how are you?"
    print(call_llm(test_prompt))
