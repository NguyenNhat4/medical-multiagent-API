

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

# Constants
MAX_RETRIEVAL_LOOPS = 2  # Maximum number of retrieval attempts before forcing compose_answer


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
        query = shared.get("retrieval_query") or shared.get("query")
        rag_state = shared.get("rag_state", "init") 
        attempts = shared.get("attempts", 1)
        selected_questions = shared.get("selected_questions", "Chưa có câu hỏi nào được retrieve")
        context_summary = shared.get("context_summary", "")
        action_history = shared.get("action_history", [])

        # Hard check: Force compose_answer if max attempts reached to prevent infinite loops
        if attempts > MAX_RETRIEVAL_LOOPS:
            return None  # Signal to exec to skip LLM call and return compose_answer
        
        return query, rag_state, attempts, selected_questions, context_summary,action_history

    def exec(self, inputs):
        from utils.llm import call_llm
        from utils.parsing import parse_yaml_with_schema
        from utils.auth import APIOverloadException
        from config.timeout_config import timeout_config
        
        # Handle hard check fallback from prep()
        if inputs is None:
            return {"next_action": "compose_answer", "reason": "Max retrieval attempts reached"}
        
        query, rag_state, attempts, selected_questions, context_summary,action_history = inputs
        conversation_context = f"Hội thoại tóm tắt (Context): {context_summary}" if context_summary else "Hội thoại vừa bắt đầu."
        current_knowledge = selected_questions if selected_questions else "Chưa có thông tin (Empty)"
        prompt = f"""Bạn là Orchestrator RAG Agent đưa ra quyết định dựa vào thông tin sau.
User query: "{query}"
Attempts: {attempts}/{MAX_RETRIEVAL_LOOPS}

Trạng thái trước đó: {rag_state}
Thông tin đã tìm được với query: 
{current_knowledge}
{conversation_context}
Tiêu chí đánh giá:
Chọn một trong các actions sau:
- create_retrieval_query: Update lại retrieval query nếu  Query bị thiếu ngữ cảnh,.
- retrieve_kb: Truy xuất thông tin QA dùng user query, nếu không có câu hỏi đã retrieve nào liên quan tới user query.
- compose_answer: Chuyển tiếp cho agent khác để soạn trả lời nếu các câu hỏi được truy xuất có liên quan cao, Và bắt buộc  nếu  Retrieve attempts lớn hơn {MAX_RETRIEVAL_LOOPS}.

```yaml
reason: <nếu chọn create_retrieval_query cân giải thích để agent khác hiểu tại sao và cần update lại như thế nào>
next_action: <create_retrieval_query | retrieve_kb | compose_answer>
```

Trả về chính xác cấu trúc yml trên:
"""

        try:
            logger.info(f"  [RagAgent] EXEC - prompt :{prompt}")
            
            resp = call_llm(prompt, fast_mode=True, max_retry_time=timeout_config.LLM_RETRY_TIMEOUT)
            logger.info(f"  [RagAgent] EXEC - resp :{resp}")

            result = parse_yaml_with_schema(
                resp,
                required_fields=["next_action", "reason"],
                field_types={"next_action": str, "reason": str}
            )
            action_history.append(result["next_action"])
            if action_history[-1] == "create_retrieval_query":
                action_history.append("retrieve_kb")
                
                
            valid_actions = ["retrieve_kb", "compose_answer", "create_retrieval_query"]
            if result["next_action"] not in valid_actions:
                return {"next_action": "compose_answer", "reason": "Invalid LLM action","attempts": attempts}
            assert result["next_action"] in valid_actions , f"Next action must be in valid actions: {valid_actions}"
            result["attempts"] = attempts
            result["action_history"] = action_history
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
        action_history = exec_res.get("action_history", [])
        current_attempts = exec_res.get("attempts", 0)
        shared['create_retrieval_query_reason'] = ""
        
        # Update state based on next action
        if next_action == "retrieve_kb":
            # Safety check: prevent infinite loops even if LLM decides to retrieve again
            if current_attempts >= MAX_RETRIEVAL_LOOPS:
                shared["rag_state"] = "composing"
                return "compose_answer"
            
            return "retrieve_kb"
        elif next_action == "compose_answer":
            shared["rag_state"] = "composing"
            return "compose_answer"
        elif next_action == "create_retrieval_query":
            shared['create_retrieval_query_reason'] = reason
            shared["rag_state"] = "create_retrieval_query_reason"
            return "create_retrieval_query"
        else:
            return "compose_answer"
