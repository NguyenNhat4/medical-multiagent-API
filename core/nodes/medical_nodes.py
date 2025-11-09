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


# ========== Medical Agent Nodes ==========

class IngestQuery(Node):
    def prep(self, shared):
        logger.info("🔍 [IngestQuery] PREP - Đọc role và input từ shared")
        role = shared.get("role", "")
        user_input = shared.get("input", "")
        logger.info(f"🔍 [IngestQuery] PREP - Role: {role}, Users Input : {user_input}")
        return role, user_input

    def exec(self, inputs):
        logger.info("🔍 [IngestQuery] EXEC - Xử lý role và query")
        role, user_input = inputs
        result = {"role": role, "query": user_input.strip()}
        logger.info(f"🔍 [IngestQuery] EXEC - Processed: {result}")
        return result

    def post(self, shared, prep_res, exec_res):
        logger.info("🔍 [IngestQuery] POST - Lưu role và query vào shared")
        shared["role"] = exec_res["role"]
        shared["query"] = exec_res["query"]
        logger.info(f"🔍 [IngestQuery] POST - Saved role: {exec_res['role']}, query: {exec_res['query'][:50]}...")
        return "default"

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
        logger.info("🤖 [RagAgent] PREP - Analyzing current state and making decision")
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

        logger.info(f"🤖 [RagAgent] PREP - state='{rag_state}', query='{query[:50]}...', {len(filtered_questions)} filtered questions, attempts={retrieve_attempts}")
        return query, user_role, demuc, chu_de_con, rag_state, filtered_questions, expansion_tried, retrieve_attempts

    def exec(self, inputs):
        from utils.llm import call_llm
        from utils.parsing import parse_yaml_with_schema
        from utils.auth import APIOverloadException
        from config.timeout_config import timeout_config

        query, user_role, demuc, chu_de_con, rag_state, filtered_questions, expansion_tried, retrieve_attempts = inputs
        logger.info(f"🤖 [RagAgent] EXEC - Current state: {rag_state}, {len(filtered_questions)} questions, attempts: {retrieve_attempts}")

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

            logger.info(f"🤖 [RagAgent] Decision: {result['next_action']} - {result['reason']}")
            return result

        except APIOverloadException:
            logger.error("🤖 [RagAgent] API overloaded")
            raise
        except Exception as e:
            logger.error(f"🤖 [RagAgent] Error: {e}")
            raise

    def post(self, shared, prep_res, exec_res):
        next_action = exec_res["next_action"]
        reason = exec_res.get("reason", "")
        current_attempts = shared.get("retrieve_attempts", 0)

        logger.info(f"🤖 [RagAgent] POST - Next action: '{next_action}' | Reason: {reason} | Current attempts: {current_attempts}")

        # Update state based on next action
        if next_action == "retry_retrieve":
            # Increment retrieve attempts counter
            shared["retrieve_attempts"] = current_attempts + 1
            shared["rag_state"] = "init"  # Reset to init for retrieve_flow to start fresh
            logger.info(f"🤖 [RagAgent] POST - Retrying retrieval pipeline (attempt {current_attempts + 1}/2)")
            return "retry_retrieve"
        elif next_action == "compose_answer":
            shared["rag_state"] = "composing"
            logger.info("🤖 [RagAgent] POST - Proceeding to compose answer")
            return "compose_answer"
        else:
            logger.warning(f"🤖 [RagAgent] POST - Unknown action '{next_action}', defaulting to compose_answer")
            return "compose_answer"


class FilterAgent(Node):
    """
    Filter candidates using LLM semantic understanding.

    Selects most relevant questions from candidates.
    Output: selected_ids (list of IDs)
    """

    def prep(self, shared):
        logger.info("🔍 [FilterAgent] PREP - Reading query and candidates")
        query = shared.get("query", "")
        candidates = shared.get("retrieved_candidates", [])

        logger.info(f"🔍 [FilterAgent] PREP - Query: '{query[:50]}...', Candidates: {len(candidates)}")
        return query, candidates

    def exec(self, inputs):
        from utils.llm import call_llm
        from utils.parsing import parse_yaml_with_schema
        from utils.auth import APIOverloadException
        from config.timeout_config import timeout_config

        query, candidates = inputs
        logger.info(f"🔍 [FilterAgent] EXEC - Filtering {len(candidates)} candidates")

        # Handle empty candidates
        if not candidates:
            logger.warning("🔍 [FilterAgent] EXEC - No candidates to filter")
            return []

        # Handle very few candidates (≤ 3) - return all
        if len(candidates) <= 3:
            logger.info(f"🔍 [FilterAgent] EXEC - Only {len(candidates)} candidates, returning all")
            return [c["id"] for c in candidates]

        # Format candidates for LLM
        candidate_list_str = self._format_candidates(candidates)

        prompt = f"""Chọn tối đa 6 câu hỏi liên quan nhất để trả lời user.

User: "{query}"

Candidates:
{candidate_list_str}

YAML:
```yaml
selected_ids: [...]
```"""

        try:
            resp = call_llm(prompt, fast_mode=True, max_retry_time=timeout_config.LLM_RETRY_TIMEOUT)

            result = parse_yaml_with_schema(
                resp,
                required_fields=["selected_ids"],
                field_types={"selected_ids": list}
            )

            if result and result["selected_ids"]:
                # Cap at 6
                selected_ids = result["selected_ids"][:6]
                logger.info(f"🔍 [FilterAgent] EXEC - Selected {len(selected_ids)} IDs")
                return selected_ids
            else:
                # Fallback: top 6
                logger.warning("🔍 [FilterAgent] EXEC - LLM parsing failed, using top 6")
                return [c["id"] for c in candidates[:6]]

        except (APIOverloadException, Exception) as e:
            logger.warning(f"🔍 [FilterAgent] EXEC - Error: {e}, using top 6")
            return [c["id"] for c in candidates[:6]]

    def _format_candidates(self, candidates: list) -> str:
        """Format candidates compactly for LLM prompt"""
        lines = []
        for i, c in enumerate(candidates, 1):
            question = c["CAUHOI"][:100] + "..." if len(c["CAUHOI"]) > 100 else c["CAUHOI"]
            lines.append(f"{i}. ID={c['id']}: \"{question}\"")
        return "\n".join(lines)

    def post(self, shared, prep_res, exec_res):
        # exec_res is just a list of IDs
        selected_ids = exec_res if isinstance(exec_res, list) else []

        # Save to shared store
        shared["selected_ids"] = selected_ids
        shared["rag_state"] = "filtered"

        logger.info(f"🔍 [FilterAgent] POST - Saved {len(selected_ids)} IDs")

        return "default"


class RetrieveFromKB(Node):
    """
    Retrieve relevant QA pairs from Qdrant vector database using hybrid search.

    ID-based architecture - no scoring needed (FilterAgent handles semantic filtering):
    - prep(): Read query and metadata from shared
    - exec(): Call Qdrant retrieval utility
    - post(): Write lightweight {id, CAUHOI} to shared

    Output: shared["retrieved_candidates"] - list of lightweight candidates
    """

    def prep(self, shared):
        logger.info("📚 [RetrieveFromKB] PREP - Đọc query và metadata từ shared")

        # Read from shared store ONLY
        query = shared.get("query", "")
        demuc = shared.get("demuc", "")
        chu_de_con = shared.get("chu_de_con", "")

        logger.info(f"📚 [RetrieveFromKB] PREP - query='{str(query)[:80]}...', demuc='{demuc}', chu_de_con='{chu_de_con}'")
        return query, demuc, chu_de_con

    def exec(self, inputs):
        query, demuc, chu_de_con = inputs
        logger.info("📚 [RetrieveFromKB] EXEC - Bắt đầu retrieve từ Qdrant")

        # Call Qdrant retrieval utility function
        from utils.knowledge_base.qdrant_retrieval import retrieve_from_qdrant

        # Retrieve with filters if available
        retrieved_results = retrieve_from_qdrant(
            query=query,
            demuc=demuc if demuc else None,
            chu_de_con=chu_de_con if chu_de_con else None,
            top_k=20
        )

        # Extract lightweight candidates: {id, CAUHOI}
        candidates = [
            {
                "id": result["id"],
                "CAUHOI": result["CAUHOI"]
            }
            for result in retrieved_results
        ]

        # Log top results
        if candidates:
            lines = ["\n📚 [RetrieveFromKB] TOP CANDIDATES:"]
            for i, candidate in enumerate(candidates[:5], 1):
                lines.append(
                    f"  {i}. id={candidate['id']} | Q: {candidate['CAUHOI'][:80]}..."
                )
            logger.info("\n".join(lines))

        logger.info(f"📚 [RetrieveFromKB] EXEC - Retrieved {len(candidates)} candidates")
        return candidates

    def post(self, shared, prep_res, exec_res):
        logger.info("📚 [RetrieveFromKB] POST - Lưu kết quả retrieve")

        candidates = exec_res

        # Save lightweight candidates to shared store
        shared["retrieved_candidates"] = candidates

        # Update RAG state
        shared["rag_state"] = "retrieved"

        logger.info(f"📚 [RetrieveFromKB] POST - Saved {len(candidates)} candidates to 'retrieved_candidates'")

        return "default" 

class GreetingResponse(Node):
    """Deprecated: Chào hỏi được gom vào ChitChatRespond."""
    def post(self, shared, prep_res, exec_res):
        return "default"

class ChitChatRespond(Node):
    """Node xử lý tất cả trường hợp không cần RAG (bao gồm chào hỏi)."""

    def prep(self, shared):
        role = shared.get("role", "")
        query = shared.get("query", "")
        conversation_history = shared.get("conversation_history", [])
        return role, query, conversation_history

    def exec(self, inputs):
        # Import dependencies only when needed
        from utils.role_enum import PERSONA_BY_ROLE, ROLE_DESCRIPTION_BY_VALUE
        from utils.llm import call_llm, PROMPT_CHITCHAT_RESPONSE
        from utils.auth import APIOverloadException
        from config.timeout_config import timeout_config

        role, query, conversation_history = inputs
        # Lấy 3 cặp gần nhất (6 tin)
        history_lines = []
        for msg in conversation_history[-6:]:
            try:
                who = msg.get("role")
                content = msg.get("content", "")
                history_lines.append(f"- {who}: {content}")
            except Exception:
                continue
        formatted_history = "\n".join(history_lines)

        # Lấy persona theo role (fallback an toàn)
        if role in PERSONA_BY_ROLE:
            persona = PERSONA_BY_ROLE[role]
            audience = persona.get('audience', 'người dùng phổ thông')
            tone = persona.get('tone', 'thân thiện, rõ ràng')
        else:
             audience, tone =  'người dùng phổ thông', 'thân thiện, rõ ràng'

        # Lấy description từ role value
        role_purpose_description = ROLE_DESCRIPTION_BY_VALUE.get(role, "Người dùng")

        prompt = PROMPT_CHITCHAT_RESPONSE.format(
            conversation_history=formatted_history,
            query=query,
            role=role,
            description=role_purpose_description,
            audience=audience,
            tone=tone
        )

        try:
            resp = call_llm(prompt, max_retry_time=timeout_config.LLM_RETRY_TIMEOUT)
        except APIOverloadException:
            # Đánh dấu API overload để route sang fallback
            resp = "Cảm ơn bạn đã chia sẻ. Mình luôn sẵn sàng hỗ trợ về thông tin y khoa nếu bạn cần nhé!"
            return {"reply": resp, "api_overload": True}

        return {"reply": resp, "api_overload": False}

    def post(self, shared, prep_res, exec_res):
        shared["answer_obj"] = {"explain": exec_res.get("reply", ""), "preformatted": True}
        shared["explain"] = exec_res.get("reply", "")
        if exec_res.get("api_overload", False):
            return "fallback"
        return "default"



class ComposeAnswer(Node):
    def prep(self, shared):
        # Import dependencies
        from utils.knowledge_base.qdrant_retrieval import get_full_qa_by_ids

        role = shared.get("role", "")
        query = shared.get("query", "")
        selected_ids = shared.get("selected_ids", [])
        score = shared.get("retrieval_score", 0.0)
        conversation_history = shared.get("conversation_history", [])

        logger.info(f"✍️ [ComposeAnswer] PREP - Role: '{role}', Query: '{query[:50]}...', Selected IDs: {selected_ids}")

        # Fetch full QA data from Qdrant using IDs
        if selected_ids:
            retrieved_qa = get_full_qa_by_ids(selected_ids)
            logger.info(f"✍️ [ComposeAnswer] PREP - Retrieved {len(retrieved_qa)} full QA pairs from Qdrant")
        else:
            logger.warning("✍️ [ComposeAnswer] PREP - No selected IDs, using empty list")
            retrieved_qa = []

        return (role, query, retrieved_qa, score, conversation_history)

    def exec(self, inputs):
        # Import dependencies only when needed
        import time
        from utils.role_enum import PERSONA_BY_ROLE
        from utils.helpers import format_kb_qa_list, format_conversation_history
        from utils.llm import call_llm, PROMPT_COMPOSE_ANSWER
        from utils.parsing import parse_yaml_with_schema
        from utils.auth import APIOverloadException
        from config.timeout_config import timeout_config

        role, query, retrieved,  score, conversation_history = inputs

        # Handle missing or invalid role with fallback
        if role not in PERSONA_BY_ROLE:
            logger.warning(f"✍️ [ComposeAnswer] EXEC - Invalid role '{role}', using default patient_diabetes role")
            role = "patient_diabetes"  # Default fallback role

        persona = PERSONA_BY_ROLE[role]
        # Compact KB context
        relevant_info_from_kb = format_kb_qa_list(retrieved, max_items=6)

        # Format conversation history
        formatted_history = format_conversation_history(conversation_history)

        prompt = PROMPT_COMPOSE_ANSWER.format(
            ai_role=persona['persona'],
            audience=persona['audience'],
            tone=persona['tone'],
            query=query,
            relevant_info_from_kb=relevant_info_from_kb if relevant_info_from_kb else "Không có thông tin từ cơ sở tri thức",
            conversation_history = formatted_history
        )
        logger.info(f"✍️ [ComposeAnswer] EXEC - prompt: {prompt}")

        try:
            start_time = time.time()
            result = call_llm(prompt, max_retry_time=timeout_config.LLM_RETRY_TIMEOUT)
            end_time = time.time()

            # Log LLM timing

            logger.info(f"✍️ [ComposeAnswer] EXEC - LLM response received")
            result = parse_yaml_with_schema(result, required_fields=["explanation", "suggestion_questions"], field_types={"explanation": str, "suggestion_questions": list})
            logger.info(f"✍️ [ComposeAnswer] EXEC - result: {result}")

            if not result or  isinstance(result, str):
                logger.warning("[ComposeAnswer] EXEC - Invalid LLM response, using fallback")
                resp = "Xin lỗi, tôi không thể tạo câu trả lời phù hợp lúc này. Bạn đặt câu hỏi khác được không? "
                return {"explain": resp, "suggestion_questions": [], "preformatted": True}

            return {"explain": result.get("explanation", ""), "suggestion_questions": result.get("suggestion_questions", []), "preformatted": True}

        except APIOverloadException as e:
            logger.warning(f"✍️ [ComposeAnswer] EXEC - API overloaded, triggering fallback mode: {e}")
            # Return flag to indicate API overload - will be handled in post method
            resp = "API hiện đang quá tải, đang chuyển sang chế độ fallback..."
            return {"explain": resp, "suggestion_questions": [], "preformatted": True, "api_overload": True}


    def post(self, shared, prep_res, exec_res):
        logger.info("✍️ [ComposeAnswer] POST - Lưu answer object")
        shared["answer_obj"] = exec_res
        shared["explain"] = exec_res.get("explain", "")
        shared["suggestion_questions"] = exec_res.get("suggestion_questions", [])
        logger.info(f"✍️ [ComposeAnswer] POST - Answer keys: {list(exec_res.keys())}")
        logger.info(f"✍️ [ComposeAnswer] POST - Answer preview: {exec_res.get('explain')}")
        
        # Check if API overload occurred and route to fallback
        if exec_res.get("api_overload", False):
            logger.info("✍️ [ComposeAnswer] POST - API overloaded, routing to fallback")
            return "fallback"
        
        return "default"


class TopicClassifyAgent(Node):
    """
    Agent phân loại chủ đề chính (DEMUC only).

    Refactored to follow PocketFlow best practices:
    - prep(): Read from shared store ONLY (no DB/API calls)
    - exec(): Call utility functions to classify DEMUC based on role's CSV file
    - post(): Write to shared store ONLY

    Classification:
    - Classify DEMUC from query based on role
    - CHU_DE_CON is always left empty (not classified)
    """

    def prep(self, shared):
        logger.info("🏷️ [TopicClassifyAgent] PREP - Đọc query và metadata từ shared")

        # Read ALL data from shared store - no external calls
        query = shared.get("query", "").strip()
        role = shared.get("role", "")
        current_demuc = shared.get("demuc", "")
        current_chu_de_con = shared.get("chu_de_con", "")

        logger.info(f"🏷️ [TopicClassifyAgent] PREP - Role: '{role}', Query: '{query[:50]}...', DEMUC: '{current_demuc}'")

        return query, role, current_demuc, current_chu_de_con

    def exec(self, inputs):
        query, role, current_demuc, current_chu_de_con = inputs

        from utils.knowledge_base.metadata_utils import (
            get_demuc_list_for_role,
            format_demuc_list_for_prompt
        )
        from utils.llm.classify_topic import classify_demuc_with_llm

        # Only classify DEMUC (no CHU_DE_CON classification)
        logger.info(f"🏷️ [TopicClassifyAgent] EXEC - Classifying DEMUC for query: '{query[:50]}...'")

        # Get DEMUC list for role
        demuc_list = get_demuc_list_for_role(role)
        if not demuc_list:
            logger.warning(f"🏷️ [TopicClassifyAgent] EXEC - No DEMUC list found for role '{role}'")
            return {"demuc": "", "chu_de_con": "", "confidence": "low"}

        demuc_list_str = format_demuc_list_for_prompt(demuc_list)
        logger.info(f"🏷️ [TopicClassifyAgent] EXEC - Available DEMUCs: {demuc_list}")

        # Classify DEMUC
        demuc_result = classify_demuc_with_llm(
            query=query,
            role=role,
            demuc_list_str=demuc_list_str
        )

        if demuc_result.get("api_overload"):
            return {"demuc": "", "chu_de_con": "", "confidence": "low", "api_overload": True}

        classified_demuc = demuc_result.get("demuc", "")
        logger.info(f"🏷️ [TopicClassifyAgent] EXEC - Classification result: DEMUC='{classified_demuc}'")

        # Return with DEMUC only (no CHU_DE_CON)
        return {
            "demuc": classified_demuc,
            "chu_de_con": "",  # Always empty - we don't classify CHU_DE_CON
            "confidence": demuc_result.get("confidence", "low"),
            "reason": demuc_result.get("reason", "")
        }

    def post(self, shared, prep_res, exec_res):
        logger.info(f"🏷️ [TopicClassifyAgent] POST - Classification result: {exec_res}")

        # Update shared with classification results - WRITE ONLY
        shared["demuc"] = exec_res.get("demuc", "")
        shared["chu_de_con"] = exec_res.get("chu_de_con", "")  # Always empty now
        shared["classification_confidence"] = exec_res.get("confidence", "low")

        logger.info(f"🏷️ [TopicClassifyAgent] POST - Updated: DEMUC='{shared['demuc']}'")

        # Check for API overload
        if exec_res.get("api_overload", False):
            return "fallback"

        # Classification complete - proceed to retrieval
        logger.info("🏷️ [TopicClassifyAgent] POST - Classification complete, routing to retrieval")
        return "default"  # Go to next node (RetrieveFromKB)


class QueryExpandAgent(Node):
    """Agent mở rộng câu hỏi mơ hồ thành câu hỏi cụ thể hơn"""

    def prep(self, shared):
        logger.info("🔍 [QueryExpandAgent] PREP - Đọc query và context")
        query = shared.get("query", "").strip()
        role = shared.get("role", "")
        conversation_history = shared.get("conversation_history", [])
        demuc = shared.get("demuc", "")
        chu_de_con = shared.get("chu_de_con", "")

        # Format conversation history
        history_lines = []
        for msg in conversation_history[-6:]:
            try:
                who = msg.get("role")
                content = msg.get("content", "")
                history_lines.append(f"- {who}: {content}")
            except Exception:
                continue
        formatted_history = "\n".join(history_lines)

        return query, role, demuc, chu_de_con, formatted_history

    def exec(self, inputs):
        # Import dependencies only when needed
        from utils.llm import call_llm
        from utils.parsing import parse_yaml_with_schema
        from utils.auth import APIOverloadException
        from config.timeout_config import timeout_config

        query, role, demuc, chu_de_con, formatted_history = inputs
        logger.info(f"🔍 [QueryExpandAgent] EXEC - Query: '{query[:50]}...', DEMUC: '{demuc}', CHU_DE_CON: '{chu_de_con}'")

        # Build context about the topic classification
        topic_context = ""
        if demuc and chu_de_con:
            topic_context = f"\nĐã xác định được chủ đề: DEMUC='{demuc}', CHU_DE_CON='{chu_de_con}'"

        prompt = f"""
Bạn là trợ lý y khoa chuyên mở rộng và làm rõ câu hỏi của người dùng.

Ngữ cảnh hội thoại gần đây:
{formatted_history}

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
            logger.info(f"🔍 [QueryExpandAgent] EXEC - LLM response: {resp}")

            result = parse_yaml_with_schema(
                resp,
                required_fields=["expanded_query"],
                optional_fields=["confidence", "reason"],
                field_types={"expanded_query": str, "confidence": str, "reason": str}
            )

            if result:
                logger.info(f"🔍 [QueryExpandAgent] EXEC - Expanded result: {result}")
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

        logger.info(f"🔍 [QueryExpandAgent] POST - Query expanded from '{original_query[:50]}...' to '{expanded_query[:50]}...'")

        # Check for API overload
        if exec_res.get("api_overload", False):
            return "fallback"

        # Update RAG state and route back to RagAgent
        shared["rag_state"] = "expanded"
        logger.info("🔍 [QueryExpandAgent] POST - Routing back to RagAgent")
        return "default"


class MainDecisionAgent(Node):
    """Main decision agent - ONLY decides between RAG agent or chitchat agent"""

    def prep(self, shared):
        logger.info("[MainDecision] PREP - Đọc query để phân loại RAG vs chitchat")
        query = shared.get("query", "").strip()
        role = shared.get("role", "")
        conversation_history = shared.get("conversation_history", [])
        # Lấy 3 cặp gần nhất (6 tin)
        history_lines = []
        for msg in conversation_history[-6:]:
            try:
                who = msg.get("role")
                content = msg.get("content", "")
                history_lines.append(f"- {who}: {content}")
            except Exception:
                continue
        formatted_history = "\n".join(history_lines)
        return query, role, formatted_history

    def exec(self, inputs):
        # Import dependencies only when needed
        from utils.llm import call_llm
        from utils.parsing import parse_yaml_with_schema
        from utils.auth import APIOverloadException
        from config.timeout_config import timeout_config

        query, role, formatted_history = inputs
        logger.info("[MainDecision] EXEC - Deciding and responding")

        # Prompt: decide type AND generate response if direct_response
        prompt = f"""Bạn là trợ lý y tế nha khoa và nội tiết. Phân tích câu hỏi và quyết định.

Câu hỏi: "{query}"

Hành động:
- direct_response: trao đổi xuồng sả.  
- retrieve_kb: câu hỏi về y tế cần tra kiến thức y tế. 

Trả về YAML:
```yaml
type: direct_response
explanation: "Câu trả lời của bạn ở đây"
```

HOẶC nếu cần tra KB:
```yaml
type: retrieve_kb
explanation: ""
```"""

        try:
            resp = call_llm(prompt, fast_mode=True, max_retry_time=timeout_config.LLM_RETRY_TIMEOUT)

            result = parse_yaml_with_schema(
                resp,
                required_fields=["type"],
                optional_fields=["explanation"],
                field_types={"type": str, "explanation": str}
            )

            decision_type = result.get("type", "")
            explanation = result.get("explanation", "")

            logger.info(f"[MainDecision] EXEC - Type: {decision_type}, Explanation length: {len(explanation)}")

            return {"type": decision_type, "explanation": explanation}

        except APIOverloadException as e:
            logger.warning(f"[MainDecision] EXEC - API overloaded, triggering fallback: {e}")
            return {"type": "api_overload", "explanation": ""}
        except Exception as e:
            logger.warning(f"[MainDecision] EXEC - LLM classification failed: {e}")
            return {"type": "default", "explanation": ""}

    def post(self, shared, prep_res, exec_res):
        logger.info(f"[MainDecision] POST - Classification result: {exec_res}")
        input_type = exec_res.get("type", "")
        explanation = exec_res.get("explanation", "")

        # Save explanation to shared if direct_response
        if input_type == "direct_response" and explanation:
            shared["answer_obj"] = {
                "explain": explanation,
                "preformatted": True,
                "suggestion_questions": []
            }
            shared["explain"] = explanation
            shared["suggestion_questions"] = []
            logger.info(f"[MainDecision] POST - Direct response saved to 'explain': {explanation[:80]}...")
            return "direct_response"
        elif input_type == "retrieve_kb":
            # Initialize retrieve attempts counter for RAG pipeline
            shared["retrieve_attempts"] = 0
            logger.info("[MainDecision] POST - Complex question, routing to retrieve_kb (attempts=0)")
            return "retrieve_kb"
        elif input_type == "api_overload" or input_type == "default":
            logger.warning("[MainDecision] POST - API issue, routing to fallback")
            return "fallback"
        else:
            # Fallback: if unknown type or no explanation, route to fallback
            logger.warning(f"[MainDecision] POST - Unknown type '{input_type}', routing to fallback")
            return "fallback"

class FallbackNode(Node):
    """Node fallback khi API quá tải - retrieve query và trả kết quả dựa trên score"""
    
    def prep(self, shared):
        logger.info("🔄 [FallbackNode] PREP - Xử lý fallback khi API quá tải")
        query = shared.get("query", "")
        role = shared.get("role", "")
        return query, role

    def exec(self, inputs):
        # Import dependencies only when needed
        from unidecode import unidecode
        from utils.knowledge_base import get_kb, ROLE_TO_CSV, retrieve_random_by_role
        from utils.helpers import aggregate_retrievals, format_kb_qa_list

        query, role = inputs
        logger.info(f"🔄 [FallbackNode] EXEC - Fallback search cho role: {role} với query: '{query[:50]}...'")

        try:
            # 1) Tìm tuần tự trong CSV theo role, so khớp HOÀN TOÀN với cột CAUHOI
            kb = get_kb()
            role_lower = (role or "").lower()
            role_csv = ROLE_TO_CSV.get(role_lower)

            def _norm_text(s: str) -> str:
                s = unidecode((s or "").lower())
                return " ".join(s.split())

            q_norm = _norm_text(query)
            exact_matches = []

            if role_csv and role_csv in kb.role_dataframes:
                df = kb.role_dataframes[role_csv]
                for _, row in df.iterrows():
                    q_text = str(row.get("CAUHOI", ""))
                    a_text = str(row.get("CAUTRALOI", ""))
                    qn = _norm_text(q_text)
                    if qn and q_norm and qn == q_norm:
                        exact_matches.append({
                            "cau_hoi": q_text,
                            "cau_tra_loi": a_text,
                            "de_muc": row.get("DEMUC", ""),
                            "chu_de_con": row.get("CHUDECON", ""),
                            "ma_so": row.get("MASO", ""),
                            "keywords": row.get("keywords", ""),
                            "giai_thich": row.get("GIAITHICH", ""),
                        })

            # Build retrieval queries: use only the main query
            retrieval_queries = []
            if query:
                retrieval_queries.append(query)

            # Use aggregate_retrievals helper function
            retrieved_results, _ = aggregate_retrievals(retrieval_queries, role=role, top_k=15)

            try:
                formatted = format_kb_qa_list(retrieved_results, max_items=15)
                if formatted:
                    logger.info("\n📚 [FallbackNode] RETRIEVE - Aggregated Results:\n" + formatted)
            except Exception:
                pass

            # Log thêm bảng điểm cho retrieved_results
            if retrieved_results:
                lines = ["\n🏷️ [FallbackNode] TOP SCORES (desc):"]
                for i, it in enumerate(retrieved_results, 1):
                    q = str(it.get('cau_hoi', ''))
                    sc = float(it.get('score', 0.0))
                    lines.append(f"  {i}. score={sc:.4f} | Q: {q[:140]}")
                logger.info("\n".join(lines))

            if exact_matches:
                best = exact_matches[0]
                explain = best.get("cau_tra_loi", "")
                # Suggestions: top4 từ retrieve (khác câu exact match)
                suggestion_questions = []
                exact_q_norm = _norm_text(best.get("cau_hoi", ""))
                for it in retrieved_results:
                    q = it.get('cau_hoi', '')
                    if q and _norm_text(q) != exact_q_norm:
                        suggestion_questions.append(q)
                        if len(suggestion_questions) >= 4:
                            break
                score = 1.0
                # Log lựa chọn cuối
                logger.info("\n✅ [FallbackNode] EXPLAIN (exact match): score=1.0000 | Q (exact): " + str(best.get("cau_hoi", ""))[:140])
                if suggestion_questions:
                    # map score theo câu hỏi để log
                    score_map = {str(it.get('cau_hoi', '')): float(it.get('score', 0.0)) for it in retrieved_results}
                    sug_lines = ["📌 [FallbackNode] SUGGESTIONS (top4):"]
                    for idx, sq in enumerate(suggestion_questions, 1):
                        sug_lines.append(f"  {idx}. score={score_map.get(sq, 0.0):.4f} | Q: {sq[:140]}")
                    logger.info("\n".join(sug_lines))
            else:
                # Không có exact match: nếu có retrieved_results, dùng top1 làm explain và còn lại làm suggestion
                if retrieved_results:
                    best_answer = retrieved_results[0]
                    explain = best_answer.get("cau_tra_loi", "")
                    suggestion_questions = [it.get('cau_hoi', '') for it in retrieved_results[1:5] if it.get('cau_hoi')]
                    score = float(best_answer.get('score', 0.0))
                    # Log lựa chọn cuối
                    logger.info(f"\n✅ [FallbackNode] EXPLAIN (retrieve top1): score={score:.4f} | Q: {str(best_answer.get('cau_hoi',''))[:140]}")
                    if suggestion_questions:
                        sug_lines = ["📌 [FallbackNode] SUGGESTIONS (next4):"]
                        for idx, it in enumerate(retrieved_results[1:5], 1):
                            if not it.get('cau_hoi'):
                                continue
                            sug_lines.append(f"  {idx}. score={float(it.get('score', 0.0)):.4f} | Q: {str(it.get('cau_hoi'))[:140]}")
                        logger.info("\n".join(sug_lines))
                else:
                    explain = "Hiện tại tôi chưa có đủ thông tin liên quan để trả lời câu hỏi này của bạn, Bạn có thể đặt lại câu hỏi khác hoặc diễn đạt lại câu hỏi của bạn! Hoặc bạn có thể chọn các câu hỏi gợi ý dưới đây!"
                    random_questions = retrieve_random_by_role(role, amount=5)
                    suggestion_questions = [q['cau_hoi'] for q in random_questions]
                    score = 0.0

            result = {
                "explain": explain,
                "suggestion_questions": suggestion_questions,
                "retrieval_score": score,
                "preformatted": True
            }

            logger.info(f"🔄 [FallbackNode] EXEC - Generated response with {len(suggestion_questions)} suggestions")
            return result

        except Exception as e:
            logger.error(f"🔄 [FallbackNode] EXEC - Error during fallback: {e}")
            # Fallback tối thiểu
            return {
                "explain": "Xin lỗi, hệ thống đang gặp sự cố. Vui lòng thử lại sau.",
                "suggestion_questions": [],
                "retrieval_score": 0.0,
                "preformatted": True
            }
    
    def post(self, shared, prep_res, exec_res):
        logger.info("🔄 [FallbackNode] POST - Lưu fallback response")
        shared["answer_obj"] = exec_res
        shared["explain"] = exec_res.get("explain", "")
        shared["suggestion_questions"] = exec_res.get("suggestion_questions", [])
        shared["retrieval_score"] = exec_res.get("retrieval_score", 0.0)
        return "default"

