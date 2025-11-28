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


class DecideSummarizeConversationToRetriveOrDirectlyAnswer(Node):

    def prep(self, shared):
        query = shared.get("query")

        role = shared.get("role", "")
        formatted_history = shared.get("formatted_conversation_history", "")
        return {
            "query": query,
            "role": role,
            "formatted_history": formatted_history
        }

    def exec(self, inputs):
        # Import dependencies only when needed
        from utils.llm import call_llm
        from utils.parsing import parse_yaml_with_schema
        from utils.auth import APIOverloadException
        from config.timeout_config import timeout_config
        from utils.role_enum import RoleEnum, ROLE_DISPLAY_NAME
        query = inputs["query"]
        role = inputs["role"]
        formatted_history = inputs["formatted_history"]
        user_role_name =  ROLE_DISPLAY_NAME.get(RoleEnum(role))
        # Build conversation history context if available
        history_context = ""
        if formatted_history:
            history_context = f"""
Lịch sử hội thoại gần đây:
{formatted_history}

"""
        prompt = f"""Bạn là bot trợ lý y tế, chỉ trao đổi quanh chủ đề y tế.
{history_context}
current user input: "{query}"
user role: {user_role_name}
Chọn 1 trong 2 Hành động:
- direct_response: chào, hỏi người dùng  để hiểu họ cần hỗ trợ gì về y tế, trả lời trực tiếp hội thoại đủ thông tin, hoặc bị lặp lại.
- retrieve_kb: chuyển tiếp cho rag agent tra kiến thức y tế chuẩn để trả lời.
Lưu ý:
- Hãy dựa vào ngữ cảnh hội thoại để hiểu câu hỏi và quyết định phù hợp

Nếu chọn direct_response:
```yaml
type: direct_response
explanation: <Câu trả lời của bạn gửi tới user ở đây>
```

Nếu chọn retrieve_kb:
```yaml
type: retrieve_kb
context_summary: |
    <sẽ mô tả ngắn gọn lại ngữ cảnh hội thoại để agent khác hiểu>
```
Trả về YAML như mẫu :
"""
        logger.info(f"[DecideSummarizeConversationToRetriveOrDirectlyAnswer] prompt: {prompt}")

        try:
            resp = call_llm(prompt, fast_mode=True, max_retry_time=timeout_config.LLM_RETRY_TIMEOUT)

            result = parse_yaml_with_schema(
                resp,
                required_fields=["type"],
                optional_fields=["explanation", "context_summary"],
                field_types={"type": str, "explanation": str, "context_summary": str}
            )

            decision_type = result.get("type", "")
            explanation = result.get("explanation", "")
            context_summary = result.get("context_summary", "")
            if decision_type == "direct_response":
                assert explanation != "" , "Câu trả lời không được rỗng"
                

            return {"type": decision_type, "explanation": explanation, "context_summary": context_summary}

        except APIOverloadException as e:
            logger.warning(f"[DecideSummarizeConversationToRetriveOrDirectlyAnswer] EXEC - API overloaded, triggering fallback: {e}")
            return {"type": "api_overload", "explanation": "", "context_summary": ""}
        except Exception as e:
            logger.warning(f"[DecideSummarizeConversationToRetriveOrDirectlyAnswer] EXEC - LLM classification failed: {e}")

    def post(self, shared, prep_res, exec_res):
        input_type = exec_res.get("type", "")
        explanation = exec_res.get("explanation", "")
        context_summary = exec_res.get("context_summary", "")

        # Save explanation to shared if direct_response
        if input_type == "direct_response" and explanation:
            shared["answer_obj"] = {
                "explain": explanation,
                "preformatted": True,
                "suggestion_questions": []
            }
            shared["explain"] = explanation
            shared["suggestion_questions"] = []
            return "direct_response"
        elif input_type == "retrieve_kb":
            # Save context summary if provided
            if context_summary and context_summary.strip():
                shared["context_summary"] = context_summary.strip()

            # Initialize retrieve attempts counter for RAG pipeline
            return "retrieve_kb"
        elif input_type == "api_overload" or input_type == "default":
            return "fallback"
        else:
            # Fallback: if unknown type or no explanation, route to fallback
            return "fallback"
