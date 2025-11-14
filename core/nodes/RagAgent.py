

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
        query = shared.get("query", "")
        user_role = shared.get("role", "")
        demuc = shared.get("demuc", "")
        chu_de_con = shared.get("chu_de_con", "")
        rag_state = shared.get("rag_state", "init")
        retrieved_candidates = shared.get("retrieved_candidates", [])
        selected_ids = shared.get("selected_ids", [])
        expansion_tried = shared.get("expansion_tried", False)
        retrieve_attempts = shared.get("retrieve_attempts", 0)

        # Load filtered questions (selected by FilterAgent)
        filtered_questions = []
        if selected_ids and retrieved_candidates:
            # Map selected IDs to actual questions
            candidate_map = {c["id"]: c["CAUHOI"] for c in retrieved_candidates}
            filtered_questions = [
                {"id": qid, "question": candidate_map.get(qid, "")}
                for qid in selected_ids
                if qid in candidate_map
            ]

        logger.info(f"  [RagAgent] PREP - state='{rag_state}', query='{query[:50]}...', {len(filtered_questions)} filtered questions, attempts={retrieve_attempts}")
        return query, user_role, demuc, chu_de_con, rag_state, filtered_questions, expansion_tried, retrieve_attempts

    def exec(self, inputs):
        from utils.llm import call_llm
        from utils.parsing import parse_yaml_with_schema
        from utils.auth import APIOverloadException
        from config.timeout_config import timeout_config

        query, user_role, demuc, chu_de_con, rag_state, filtered_questions, expansion_tried, retrieve_attempts = inputs
        logger.info(f"  [RagAgent] EXEC - Current state: {rag_state}, {len(filtered_questions)} questions, attempts: {retrieve_attempts}")

        # Format filtered questions for LLM
        questions_str = ""
        if filtered_questions:
            questions_str = "\n".join([
                f"{i}. {q['question'][:80]}..." if len(q['question']) > 80 else f"{i}. {q['question']}"
                for i, q in enumerate(filtered_questions, 1)
            ])

        # Build context
        context = f"""Query: "{query}"
Metadata: DEMUC="{demuc}", CHU_DE_CON="{chu_de_con}"
State: {rag_state}
Retrieve attempts: {retrieve_attempts}/2

Filtered questions ({len(filtered_questions)}):
{questions_str if questions_str else "(none)"}"""


        prompt = f"""RAG Agent quyết định bước tiếp.

{context}

Actions:
- retry_retrieve: Thử lại retrieval
- compose_answer: Soạn trả lời

Rules:
1. Nếu attempts >= 2 → BẮT BUỘC compose_answer (đã hết lượt retry)
2. Nếu có đủ câu hỏi (≥ 2) → compose_answer
3. Nếu không có câu hỏi + attempts < 2 → retry_retrieve

YAML:
```yaml
next_action: "..."
reason: "..."
```"""

        try:
            resp = call_llm(prompt, fast_mode=True, max_retry_time=timeout_config.LLM_RETRY_TIMEOUT)

            result = parse_yaml_with_schema(
                resp,
                required_fields=["next_action", "reason"],
                field_types={"next_action": str, "reason": str}
            )

            # Validate action
            valid_actions = ["retry_retrieve", "compose_answer"]
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

        logger.info(f"  [RagAgent] POST - Next action: '{next_action}' | Reason: {reason} | Current attempts: {current_attempts}")

        # Update state based on next action
        if next_action == "retry_retrieve":
            # Increment retrieve attempts counter
            shared["retrieve_attempts"] = current_attempts + 1
            shared["rag_state"] = "init"  # Reset to init for retrieve_flow to start fresh
            logger.info(f"  [RagAgent] POST - Retrying retrieval pipeline (attempt {current_attempts + 1}/2)")
            return "retry_retrieve"
        elif next_action == "compose_answer":
            shared["rag_state"] = "composing"
            logger.info("  [RagAgent] POST - Proceeding to compose answer")
            return "compose_answer"
        else:
            logger.warning(f"  [RagAgent] POST - Unknown action '{next_action}', defaulting to compose_answer")
            return "compose_answer"
