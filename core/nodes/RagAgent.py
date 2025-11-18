

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


class RagAgent(Node):
    """
    RAG Agent - intelligent decision maker that orchestrates the retrieval pipeline:
    1. Decide if we need to classify topic (get DEMUC, CHU_DE_CON)
    2. Decide if we need to expand query
    3. Trigger retrieval when ready
    4. Route to compose answer after retrieval
    
    State machine:
    - init -> classify (if no metadata) -> expand (if needed) -> retrieve -> compose_answer
    """

    def prep(self, shared):
        logger.info("  [RagAgent] PREP - Analyzing current state and making decision")
        query = shared.get("retrieval_query") or shared.get("query")
        rag_state = shared.get("rag_state", "init") 
        retrieve_attempts = shared.get("retrieve_attempts", 1)
        selected_questions = shared.get("selected_questions", "Chưa có  câu hỏi  nào được retrieve")
        context_summary  = shared.get("context_summary",  "")
        return query, rag_state, retrieve_attempts,selected_questions,context_summary

    def exec(self, inputs):
        from utils.llm import call_llm
        from utils.parsing import parse_yaml_with_schema
        from utils.auth import APIOverloadException
        from config.timeout_config import timeout_config
        
        query, rag_state, retrieve_attempts,selected_questions,context_summary= inputs
        conversation_context = f"Ngữ cảnh hội thoại:{context_summary}" if  context_summary else ""
        
        
        prompt = f"""Bạn là Orchestrator RAG Agent đưa ra quyết định dựa vào thông tin sau.
User query: "{query}"
Retrieve attempts: {retrieve_attempts}/{self.max_retries}
Trạng thái tại hiện tại:{rag_state}
Danh sách câu hỏi đã retrieve: 
{selected_questions}
{conversation_context}
Chọn một trong các actions sau:
- create_retrieval_query: Update lại user query nếu nó không đủ thông tin để retrieve.
- retrieve_kb: truy xuất thông tin QA dùng user query ,nếu không có câu hỏi đã retrieve nào liên quan tới user query.
- compose_answer: Chuyển tiếp cho agent khác để soạn trả lời nếu các câu hỏi được truy xuất có liên quan cao. 


```yaml
next_action: <chọn 1 trong Actions>
reason: <>
```

Trả về chính xác câu trúc yml trên:
"""

        try:
            logger.info(f"  [RagAgent] EXEC - prompts :{prompt}")
            
            resp = call_llm(prompt, fast_mode=True, max_retry_time=timeout_config.LLM_RETRY_TIMEOUT)

            result = parse_yaml_with_schema(
                resp,
                required_fields=["next_action", "reason"],
                field_types={"next_action": str, "reason": str}
            )

            # Validate action
            valid_actions = ["retrieve_kb", "compose_answer","create_retrieval_query"]
            if result["next_action"] not in valid_actions:
                raise ValueError(f"Invalid action: {result['next_action']}")
            logger.info(f"  [RagAgent] Decision: {result['next_action']} - {result['reason']}")
            return result

        except APIOverloadException:
            logger.error("  [RagAgent] API overloaded")
            raise
        except Exception as e:
            logger.error(f"  [RagAgent] Error: {e}")
            raise

    def post(self, shared, prep_res, exec_res):
        next_action = exec_res["next_action"]
        reason = exec_res.get("reason", "")
        current_attempts = shared.get("retrieve_attempts", 0)
        shared['create_retrieval_query_reason'] = ""
        logger.info(f"  [RagAgent] POST - Next action: '{next_action}' | Reason: {reason} | Current attempts: {current_attempts}")

        # Update state based on next action
        if next_action == "retrieve_kb":
            # Increment retrieve attempts counter
            shared["retrieve_attempts"] = current_attempts + 1
            shared["rag_state"] = "init"  # Reset to init for retrieve_flow to start fresh
            logger.info(f"  [RagAgent] POST - Retrying retrieval pipeline (attempt {current_attempts + 1}/2)")
            return "retrieve_kb"
        elif next_action == "compose_answer":
            shared["rag_state"] = "composing"
            logger.info("  [RagAgent] POST - Proceeding to compose answer")
            return "compose_answer"
        elif next_action == "create_retrieval_query":
            shared['create_retrieval_query_reason'] = reason
            shared["rag_state"] = "just created new query"
            return "create_retrieval_query"
        else:
            logger.warning(f"  [RagAgent] POST - Unknown action '{next_action}', defaulting to compose_answer")
            return "compose_answer"
