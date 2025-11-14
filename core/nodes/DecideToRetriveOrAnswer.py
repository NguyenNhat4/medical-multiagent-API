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


class DecideToRetriveOrAnswer(Node):
    """Main decision agent - ONLY decides between RAG agent or chitchat agent"""

    def prep(self, shared):
        logger.info("[DecideToRetriveOrAnswer] PREP - Đọc query và formatted history để phân loại RAG vs chitchat")
        query = shared.get("query", "").strip()
        role = shared.get("role", "")
        formatted_history = shared.get("formatted_conversation_history", "")
        logger.info(f"[DecideToRetriveOrAnswer] PREP - Query: {query[:50]}..., Has history: {bool(formatted_history)}")
        return query, role, formatted_history

    def exec(self, inputs):
        # Import dependencies only when needed
        from utils.llm import call_llm
        from utils.parsing import parse_yaml_with_schema
        from utils.auth import APIOverloadException
        from config.timeout_config import timeout_config

        query, role, formatted_history = inputs
        logger.info("[DecideToRetriveOrAnswer] EXEC - Deciding and responding")

        # Build conversation history context if available
        history_context = ""
        if formatted_history:
            history_context = f"""
Lịch sử hội thoại gần đây:
{formatted_history}

"""

        # Prompt: decide type AND generate response/new_query
        prompt = f"""Bạn là bot trợ lý y tế, chỉ trao đổi quanh chủ đề y tế.

{history_context}
user input: "{query}"

Chọn 1 trong 2 Hành động:
- direct_response: trao đổi xuồng sả, hỏi để biết người dùng cần hỗ trợ gì về y tế
- retrieve_kb: tra cơ sở tri thức y tế bác sĩ chuẩn bị trước khi trả lời câu hỏi y tế
Lưu ý:
- Hãy dựa vào ngữ cảnh hội thoại để hiểu câu hỏi và quyết định phù hợp
- trong cơ sở tri thức chỉ dùng một term chung là "Đái Tháo Đường". 
Trả về YAML:
Nếu chọn direct_response:
```yaml
type: direct_response
explanation: "Câu trả lời của bạn ở đây"
new_query: "<phải để trống>"
```

Nếu chọn retrieve_kb (thực hiện hybrid search trên user query và compose agent sẽ trả lời dựa trên user input và thông tin retrieve,
agent này sẽ không thấy được lịch sử chat nên có thể update bằng viết lại new_query cho rõ ràng.):
```yaml
type: retrieve_kb
explanation: "<phải để trống>"
new_query: "< Viết lại user input cho rõ ràng, chỉ khi nó mơ hồ.>"
```
"""
        logger.info(f"[DecideToRetriveOrAnswer] prompt: {prompt}")

        try:
            resp = call_llm(prompt, fast_mode=True, max_retry_time=timeout_config.LLM_RETRY_TIMEOUT)

            result = parse_yaml_with_schema(
                resp,
                required_fields=["type"],
                optional_fields=["explanation", "new_query"],
                field_types={"type": str, "explanation": str, "new_query": str}
            )

            decision_type = result.get("type", "")
            explanation = result.get("explanation", "")
            new_query = result.get("new_query", "")

            logger.info(f"[DecideToRetriveOrAnswer] EXEC - Type: {decision_type}, Explanation length: {len(explanation)}, New query: '{new_query if new_query else 'N/A'}...'")

            return {"type": decision_type, "explanation": explanation, "new_query": new_query}

        except APIOverloadException as e:
            logger.warning(f"[DecideToRetriveOrAnswer] EXEC - API overloaded, triggering fallback: {e}")
            return {"type": "api_overload", "explanation": "", "new_query": ""}
        except Exception as e:
            logger.warning(f"[DecideToRetriveOrAnswer] EXEC - LLM classification failed: {e}")
            return {"type": "default", "explanation": "", "new_query": ""}

    def post(self, shared, prep_res, exec_res):
        logger.info(f"[DecideToRetriveOrAnswer] POST - Classification result: {exec_res}")
        input_type = exec_res.get("type", "")
        explanation = exec_res.get("explanation", "")
        new_query = exec_res.get("new_query", "")

        # Save explanation to shared if direct_response
        if input_type == "direct_response" and explanation:
            shared["answer_obj"] = {
                "explain": explanation,
                "preformatted": True,
                "suggestion_questions": []
            }
            shared["explain"] = explanation
            shared["suggestion_questions"] = []
            logger.info(f"[DecideToRetriveOrAnswer] POST - Direct response saved to 'explain': {explanation[:80]}...")
            return "direct_response"
        elif input_type == "retrieve_kb":
            # Update query if new_query is provided (context-aware query enhancement)
            original_query = shared.get("query", "")
            if new_query and new_query.strip():
                shared["original_query"] = original_query
                shared["query"] = new_query.strip()
                logger.info(f"[DecideToRetriveOrAnswer] POST - Query updated from '{original_query[:50]}...' to '{new_query[:50]}...'")
            else:
                logger.info(f"[DecideToRetriveOrAnswer] POST - No query update, keeping original: '{original_query[:50]}...'")

            # Initialize retrieve attempts counter for RAG pipeline
            shared["retrieve_attempts"] = 0
            logger.info("[DecideToRetriveOrAnswer] POST - Complex question, routing to retrieve_kb (attempts=0)")
            return "retrieve_kb"
        elif input_type == "api_overload" or input_type == "default":
            logger.warning("[DecideToRetriveOrAnswer] POST - API issue, routing to fallback")
            return "fallback"
        else:
            # Fallback: if unknown type or no explanation, route to fallback
            logger.warning(f"[DecideToRetriveOrAnswer] POST - Unknown type '{input_type}', routing to fallback")
            return "fallback"
