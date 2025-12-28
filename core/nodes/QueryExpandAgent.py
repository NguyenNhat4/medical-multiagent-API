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


class QueryExpandAgent(Node):
    """Agent mở rộng câu hỏi mơ hồ thành câu hỏi cụ thể hơn"""

    def prep(self, shared):
        query = shared.get("retrieval_query") or shared.get("query")

        role = shared.get("role", "")
        formatted_history = shared.get("formatted_conversation_history", "")
        demuc = shared.get("demuc", "")
        chu_de_con = shared.get("chu_de_con", "")
        return {
            "query": query,
            "role": role,
            "demuc": demuc,
            "chu_de_con": chu_de_con,
            "formatted_history": formatted_history
        }

    def exec(self, inputs):
        # Import dependencies only when needed
        from utils.llm import call_llm
        from utils.parsing import parse_yaml_with_schema
        from utils.llm.call_llm import APIOverloadException
        from config.timeout_config import timeout_config

        query = inputs["query"]
        role = inputs["role"]
        demuc = inputs["demuc"]
        chu_de_con = inputs["chu_de_con"]
        formatted_history = inputs["formatted_history"]

        # Build context about the topic classification
        topic_context = ""
        if demuc and chu_de_con:
            topic_context = f"\nĐã xác định được chủ đề: DEMUC='{demuc}', CHU_DE_CON='{chu_de_con}'"

        prompt = f"""
Bạn là trợ lý y khoa chuyên mở rộng và làm rõ câu hỏi của người dùng.


Câu hỏi hiện tại của người dùng: "{query}"
Role của người dùng: {role}
{topic_context}

NHIỆM VỤ:
Mở rộng câu hỏi thành một câu hỏi CỤ THỂ HƠN, RÕ RÀNG HƠN, CHI TIẾT HƠN.
- Nếu câu hỏi đã đủ cụ thể, có thể giữ nguyên hoặc bổ sung chi tiết nhỏ.
- Nếu câu hỏi mơ hồ, hãy làm rõ dựa trên ngữ cảnh hội thoại và chủ đề đã xác định.

YÊU CẦU:
- expanded_query: câu hỏi đã được mở rộng/cụ thể hóa
- confidence: high/medium/low - mức độ tự tin về việc mở rộng đúng ý người dùng
- reason: lý do ngắn gọn về cách mở rộng

VÍ DỤ:
Input: "Tôi muốn hỏi về bệnh"
Context: DEMUC="BỆNH LÝ ĐTĐ", CHU_DE_CON="Định nghĩa và phân loại"
Output:
```yaml
expanded_query: "Định nghĩa và phân loại bệnh đái tháo đường là gì?"
confidence: "high"
reason: "Mở rộng dựa trên chủ đề đã xác định"
```

Trả về CHỈ một code block YAML hợp lệ:

```yaml
expanded_query: "Câu hỏi đã mở rộng"
confidence: "high"
reason: "Lý do ngắn gọn"
```
"""

        try:
            resp = call_llm(prompt, fast_mode=True, max_retry_time=timeout_config.LLM_RETRY_TIMEOUT)
            result = parse_yaml_with_schema(
                resp,
                required_fields=["expanded_query"],
                optional_fields=["confidence", "reason"],
                field_types={"expanded_query": str, "confidence": str, "reason": str}
            )

            if result:
                return result
        except APIOverloadException as e:
            logger.warning(f"🔍 [QueryExpandAgent] EXEC - API overloaded: {e}")
            return {"expanded_query": query, "confidence": "low", "api_overload": True}
        except Exception as e:
            logger.warning(f"🔍 [QueryExpandAgent] EXEC - Expansion failed: {e}")

        # Fallback: return original query
        return {"expanded_query": query, "confidence": "low"}

    def post(self, shared, prep_res, exec_res):
        logger.info(f"🔍 [QueryExpandAgent] POST - Expansion result: {exec_res}")

        # Update query with expanded version
        original_query = shared.get("query", "")
        expanded_query = exec_res.get("expanded_query", original_query)

        shared["original_query"] = original_query
        shared["query"] = expanded_query  # Replace with expanded query
        shared["expansion_confidence"] = exec_res.get("confidence", "low")

        logger.info(f"🔍 [QueryExpandAgent] POST - Query expanded from '{original_query}...' to '{expanded_query}...'")

        # Check for API overload
        if exec_res.get("api_overload", False):
            return "fallback"

        # Update RAG state and route back to RagAgent
        shared["rag_state"] = "expanded"
        logger.info("🔍 [QueryExpandAgent] POST - Routing back to RagAgent")
        return "default"



