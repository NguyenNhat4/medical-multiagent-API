# Core framework import
from pocketflow import Node

# Standard library imports
import logging

# Configure logging for this module with Vietnam timezone
from utils.timezone_utils import setup_vietnam_logging
from config.logging_config import logging_config

if logging_config.USE_VIETNAM_TIMEZONE:
    logger = setup_vietnam_logging(__name__, 
                                 level=getattr(logging, logging_config.LOG_LEVEL.upper()),
                                 format_str=logging_config.LOG_FORMAT)
else:
    logger = logging.getLogger(__name__)
    logger.setLevel(getattr(logging, logging_config.LOG_LEVEL.upper()))


class QueryCreatingForRetrievalAgent(Node):
    """ Dựa vào  hội thoại đã được tóm tắt (context_summary) , role của người dùng ,
    
    và input hiện toại của họ (query), trả về 1 retrieval_query dùng để retrieev thông tin chính xác để trả lời người dùng.)
    """

    def prep(self, shared):
        logger.info("🔍 [QueryCreatingForRetrievalAgent] PREP - Đọc query và context")
        query = shared.get("retrieval_query") or shared.get("query")

        role = shared.get("role", "")
        demuc = shared.get("demuc", "")
        chu_de_con = shared.get("chu_de_con", "")
        context_summary = shared.get("context_summary", "")
        reason = shared.get('create_retrieval_query_reason' , "")
        logger.info(f"🔍 [QueryCreatingForRetrievalAgent] PREP - Query: {query[:50]}..., Role: {role}, DEMUC: {demuc}, CHU_DE_CON: {chu_de_con}")
        return {
            "query": query,
            "role": role,
            "demuc": demuc,
            "chu_de_con": chu_de_con,
            "context_summary": context_summary,
            "reason": reason
        }

    def exec(self, inputs):
        # Import dependencies only when needed
        from utils.llm import call_llm
        from utils.parsing import parse_yaml_with_schema
        from utils.auth import APIOverloadException
        from config.timeout_config import timeout_config
        from utils.role_enum import RoleEnum, ROLE_DISPLAY_NAME
        
        current_user_input = inputs["query"]
        role = inputs["role"]
        demuc = inputs["demuc"]
        chu_de_con = inputs["chu_de_con"]
        context_summary = inputs["context_summary"]
        reason = inputs["reason"]
        vietnameseRole = ROLE_DISPLAY_NAME.get(RoleEnum(role), "Người dùng") # VD role = 'patient_dental' -> vietnameseRole='Bệnh nhân nha khoa'
        
        
        # Build topic context if available
        topic_context = f"\nChủ đề đã xác định: DEMUC='{demuc}'" if demuc else ""
            
        reason_final = f"- Lý do cần tạo là: {reason}" if reason else ""
    
        
        prompt = f"""
BỐI CẢNH:
-Tóm tắt hội thoại trước đó: {context_summary}
- Câu hỏi hiện tại của người dùng: "{current_user_input}"
{reason_final}
- Người dùng là {vietnameseRole} 
        {topic_context}

NHIỆM VỤ:
- Viết lại câu hỏi người dùng cho rõ ràng hơn để truy vấn thông tin từ vector database.

Trả về CHỈ một code block YAML hợp lệ:

```yaml
retrieval_query: "Câu hỏi đã được viết lại cho rõ ràng hơn"
reason: "Lý do ngắn gọn về cách tạo query"
confidence: "high"  # hoặc medium, low
```"""

        try:
            logger.info(f"🔍 [QueryCreatingForRetrievalAgent] EXEC - prompts: '{prompt}")
            
            resp = call_llm(prompt, fast_mode=True, max_retry_time=timeout_config.LLM_RETRY_TIMEOUT)
            logger.info(f"🔍 [QueryCreatingForRetrievalAgent] EXEC - LLM response: {resp[:200]}...")

            result = parse_yaml_with_schema(
                resp,
                required_fields=["retrieval_query", "reason"],
                optional_fields=["confidence"],
                field_types={"retrieval_query": str, "reason": str, "confidence": str}
            )

            if result:
                logger.info(f"🔍 [QueryCreatingForRetrievalAgent] EXEC - Created retrieval query: '{result}'")
                return result
        except APIOverloadException as e:
            logger.warning(f"🔍 [QueryCreatingForRetrievalAgent] EXEC - API overloaded: {e}")
            return {"retrieval_query": current_user_input, "confidence": "low", "reason": "API overloaded, using original query", "api_overload": True}
        except Exception as e:
            logger.warning(f"🔍 [QueryCreatingForRetrievalAgent] EXEC - Query creation failed: {e}")

        # Fallback: return original query
        logger.info(f"🔍 [QueryCreatingForRetrievalAgent] EXEC - Fallback: using original query")
        return {"retrieval_query": current_user_input, "confidence": "low", "reason": "Failed to create optimized query"}

    def post(self, shared, prep_res, exec_res):
        logger.info(f"🔍 [QueryCreatingForRetrievalAgent] POST - Storing retrieval query")
        
        # Extract results
        retrieval_query = exec_res.get("retrieval_query", "")
        confidence = exec_res.get("confidence", "low")
        reason = exec_res.get("reason", "")
        
        # Store original query if not already stored
        if "original_query" not in shared:
            shared["original_query"] = shared.get("query", "")
        
        # Store retrieval query in shared state
        shared["retrieval_query"] = retrieval_query
        shared["retrieval_query_confidence"] = confidence
        shared["retrieval_query_reason"] = reason
        
        logger.info(f"🔍 [QueryCreatingForRetrievalAgent] POST - Original: '{shared.get('original_query', '')[:50]}...'")
        logger.info(f"🔍 [QueryCreatingForRetrievalAgent] POST - Retrieval query: '{retrieval_query[:50]}...' (confidence: {confidence})")
        
        # Check for API overload
        if exec_res.get("api_overload", False):
            logger.warning("🔍 [QueryCreatingForRetrievalAgent] POST - API overload detected, routing to fallback")
            return "fallback"
        
        return "default"


