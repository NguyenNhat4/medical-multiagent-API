from math import log
from pocketflow import Node
from utils.llm import call_llm, PROMPT_CLASSIFY_INPUT, PROMPT_COMPOSE_ANSWER, PROMPT_CHITCHAT_RESPONSE
from utils.auth import APIOverloadException
from config.timeout_config import timeout_config
from utils.knowledge_base import retrieve_random_by_role, get_kb, ROLE_TO_CSV
from utils.parsing import parse_yaml_with_schema
from utils.helpers import (
    format_kb_qa_list,
    get_score_threshold,
    format_conversation_history,
    aggregate_retrievals
)
from utils.role_enum import (
    PERSONA_BY_ROLE,
    ROLE_DESCRIPTION_BY_VALUE
)
import logging
from unidecode import unidecode
import time

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
        retrieved = shared.get("retrieved", [])
        
        logger.info(f"🤖 [RagAgent] PREP - state='{rag_state}', query='{query[:50]}...', demuc='{demuc}', chu_de_con='{chu_de_con}'")
        return query, user_role, demuc, chu_de_con, rag_state, retrieved

    def exec(self, inputs):
        query, user_role, demuc, chu_de_con, rag_state, retrieved = inputs
        logger.info(f"🤖 [RagAgent] EXEC - Current state: {rag_state}")

        # State machine logic
        if rag_state == "init":
            # First time - decide if we need topic classification
            if not demuc and not chu_de_con:
                logger.info("🤖 [RagAgent] No metadata - need topic classification")
                return {"next_action": "classify", "reason": "No DEMUC/CHU_DE_CON available"}
            else:
                # Has some metadata - check if need expansion
                logger.info("🤖 [RagAgent] Has metadata - checking if need expansion")
                return self._check_expansion_need(query, demuc, chu_de_con)
        
        elif rag_state == "classified":
            # After classification - check if need query expansion
            logger.info("🤖 [RagAgent] After classification - checking if need expansion")
            return self._check_expansion_need(query, demuc, chu_de_con)
        
        elif rag_state == "expanded":
            # After expansion - proceed to retrieve
            logger.info("🤖 [RagAgent] After expansion - ready to retrieve")
            return {"next_action": "retrieve", "reason": "Query expanded, ready to retrieve"}
        
        elif rag_state == "retrieved":
            # After retrieval - compose answer
            logger.info("🤖 [RagAgent] After retrieval - ready to compose answer")
            return {"next_action": "compose_answer", "reason": "Retrieved data available"}
        
        else:
            # Unknown state - default to retrieve
            logger.warning(f"🤖 [RagAgent] Unknown state '{rag_state}' - defaulting to retrieve")
            return {"next_action": "retrieve", "reason": "Unknown state - fallback"}

    def _check_expansion_need(self, query: str, demuc: str, chu_de_con: str) -> dict:
        """Check if query needs expansion based on query length and metadata availability"""
        
        # If we have both DEMUC and CHU_DE_CON and query is reasonably specific
        if demuc and chu_de_con and len(query.split()) >= 4:
            logger.info("🤖 [RagAgent] Have full metadata and specific query - no expansion needed")
            return {"next_action": "retrieve", "reason": "Full metadata + specific query"}
        
        # If query is very short or vague
        if len(query.split()) < 4:
            logger.info("🤖 [RagAgent] Query too short/vague - need expansion")
            return {"next_action": "expand", "reason": "Query too short or vague"}
        
        # Default: proceed to retrieve
        logger.info("🤖 [RagAgent] Query acceptable - proceeding to retrieve")
        return {"next_action": "retrieve", "reason": "Query acceptable as-is"}

    def post(self, shared, prep_res, exec_res):
        next_action = exec_res["next_action"]
        reason = exec_res.get("reason", "")
        
        logger.info(f"🤖 [RagAgent] POST - Next action: '{next_action}' | Reason: {reason}")
        
        # Update state based on next action
        if next_action == "classify":
            shared["rag_state"] = "init"  # Will be updated to "classified" by TopicClassifyAgent
            return "classify"
        elif next_action == "expand":
            shared["rag_state"] = "classified"  # Will be updated to "expanded" by QueryExpandAgent
            return "expand"
        elif next_action == "retrieve":
            shared["rag_state"] = "expanded"  # Will be updated to "retrieved" by RetrieveFromKB
            return "retrieve"
        elif next_action == "compose_answer":
            shared["rag_state"] = "retrieved"
            return "compose_answer"
        else:
            logger.warning(f"🤖 [RagAgent] POST - Unknown action '{next_action}', defaulting to retrieve")
            return "retrieve"


class RetrieveFromKB(Node):
    def prep(self, shared):
        logger.info("📚 [RetrieveFromKB] PREP - Đọc query để retrieve")
        query = shared.get("query", "")
        user_role = shared.get("role", "")
        demuc = shared.get("demuc", "")
        chu_de_con = shared.get("chu_de_con", "")
        logger.info(f"📚 [RetrieveFromKB] PREP - query='{str(query)[:80]}...', demuc='{demuc}', chu_de_con='{chu_de_con}'")
        return query, user_role, demuc, chu_de_con

    def exec(self, inputs):
        query, user_role, demuc, chu_de_con = inputs
        logger.info("📚 [RetrieveFromKB] EXEC - Bắt đầu retrieve với expanded query")

        # Use only the main query (which may have been expanded)
        retrieval_queries = []
        if query:
            retrieval_queries.append(query)

        # Use aggregate_retrievals helper function
        retrieved_results, top_score = aggregate_retrievals(retrieval_queries, role=user_role, top_k=15)

        # Log formatted QA list and score table
        try:
            formatted = format_kb_qa_list(retrieved_results, max_items=15)
            if formatted:
                logger.info("\n📚 [RetrieveFromKB] FORMATTED Top Results:\n" + formatted)
        except Exception:
            pass

        if retrieved_results:
            lines = ["\n🏷️ [RetrieveFromKB] TOP SCORES (desc):"]
            for i, it in enumerate(retrieved_results, 1):
                lines.append(f"  {i}. score={float(it.get('score',0.0)):.4f} | Q: {str(it.get('cau_hoi',''))[:140]}")
            logger.info("\n".join(lines))

        logger.info(f"📚 [RetrieveFromKB] EXEC - Aggregated results={len(retrieved_results)}, top_score={top_score:.4f}")
        return retrieved_results, top_score

    def post(self, shared, prep_res, exec_res):
        logger.info("📚 [RetrieveFromKB] POST - Lưu kết quả retrieve")
        results, score = exec_res
        shared["retrieved"] = results
        shared["retrieval_score"] = score
        shared["need_clarify"] = score < get_score_threshold()
        
        # Update RAG state and route back to RagAgent
        shared["rag_state"] = "retrieved"
        logger.info(
            f"📚 [RetrieveFromKB] POST - Saved {len(results)} results, score: {score:.4f}, "
            f"routing back to RagAgent"
        )
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
        role = shared.get("role", "")
        query = shared.get("query", "")
        retrieved = shared.get("retrieved", [])
        score = shared.get("retrieval_score", 0.0)
        conversation_history = shared.get("conversation_history", [])
        logger.info(f"✍️ [ComposeAnswer] PREP - Role: '{role}', Query: '{query[:50]}...', Retrieved: {len(retrieved)} items")
        return (role, query, retrieved, score, conversation_history)

    def exec(self, inputs):
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
    Agent phân loại chủ đề 2 bước: DEMUC -> CHU_DE_CON

    Refactored to follow PocketFlow best practices:
    - prep(): Read from shared store ONLY (no DB/API calls)
    - exec(): Call utility functions for 2-step classification
    - post(): Write to shared store ONLY

    Two-step classification:
    1. If no DEMUC: Classify DEMUC from query
    2. If have DEMUC: Classify CHU_DE_CON within that DEMUC
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
            get_chu_de_con_for_demuc,
            format_demuc_list_for_prompt,
            format_chu_de_con_list_for_prompt
        )
        from utils.llm.classify_topic import (
            classify_demuc_with_llm,
            classify_chu_de_con_with_llm
        )

        # Step 1: Classify DEMUC if not present
        if not current_demuc:
            logger.info(f"🏷️ [TopicClassifyAgent] EXEC - STEP 1: Classifying DEMUC for query: '{query[:50]}...'")

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
            logger.info(f"🏷️ [TopicClassifyAgent] EXEC - STEP 1 result: DEMUC='{classified_demuc}'")

            # Return with only DEMUC for now - CHU_DE_CON will be classified in next run
            return {
                "demuc": classified_demuc,
                "chu_de_con": "",  # Will be classified in next step
                "confidence": demuc_result.get("confidence", "low"),
                "reason": demuc_result.get("reason", "")
            }

        # Step 2: Classify CHU_DE_CON within current DEMUC
        else:
            logger.info(f"🏷️ [TopicClassifyAgent] EXEC - STEP 2: Classifying CHU_DE_CON within DEMUC='{current_demuc}'")

            # Get CHU_DE_CON list for this DEMUC
            chu_de_con_list = get_chu_de_con_for_demuc(role, current_demuc)
            if not chu_de_con_list:
                logger.warning(f"🏷️ [TopicClassifyAgent] EXEC - No CHU_DE_CON found for DEMUC '{current_demuc}' in role '{role}'")
                return {"demuc": current_demuc, "chu_de_con": "", "confidence": "low"}

            chu_de_con_list_str = format_chu_de_con_list_for_prompt(chu_de_con_list)
            logger.info(f"🏷️ [TopicClassifyAgent] EXEC - Available CHU_DE_CONs: {chu_de_con_list}")

            # Classify CHU_DE_CON
            chu_de_con_result = classify_chu_de_con_with_llm(
                query=query,
                demuc=current_demuc,
                chu_de_con_list_str=chu_de_con_list_str
            )

            if chu_de_con_result.get("api_overload"):
                return {"demuc": current_demuc, "chu_de_con": "", "confidence": "low", "api_overload": True}

            classified_chu_de_con = chu_de_con_result.get("chu_de_con", "")
            logger.info(f"🏷️ [TopicClassifyAgent] EXEC - STEP 2 result: CHU_DE_CON='{classified_chu_de_con}'")

            return {
                "demuc": current_demuc,  # Keep existing DEMUC
                "chu_de_con": classified_chu_de_con,
                "confidence": chu_de_con_result.get("confidence", "low"),
                "reason": chu_de_con_result.get("reason", "")
            }

    def post(self, shared, prep_res, exec_res):
        logger.info(f"🏷️ [TopicClassifyAgent] POST - Classification result: {exec_res}")

        # Update shared with classification results - WRITE ONLY
        shared["demuc"] = exec_res.get("demuc", "")
        shared["chu_de_con"] = exec_res.get("chu_de_con", "")
        shared["classification_confidence"] = exec_res.get("confidence", "low")

        logger.info(f"🏷️ [TopicClassifyAgent] POST - Updated: DEMUC='{shared['demuc']}', CHU_DE_CON='{shared['chu_de_con']}'")

        # Check for API overload
        if exec_res.get("api_overload", False):
            return "fallback"

        # Check if we need another round
        # If we have DEMUC but not CHU_DE_CON, route back to classify again
        if shared["demuc"] and not shared["chu_de_con"]:
            logger.info("🏷️ [TopicClassifyAgent] POST - Have DEMUC but no CHU_DE_CON, route back to classify again")
            return "classify_again"  # This action should route back to TopicClassifyAgent

        # Both DEMUC and CHU_DE_CON are classified
        shared["rag_state"] = "classified"
        logger.info("🏷️ [TopicClassifyAgent] POST - Classification complete, routing back to RagAgent")
        return "default"


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
        query, role, formatted_history = inputs
        logger.info("[MainDecision] EXEC - Classifying: RAG or chitchat")

        # Simplified prompt: only decide between RAG and chitchat
        prompt = f"""
Phân loại DUY NHẤT input thành một trong: medical_question | chitchat.

Định nghĩa:
- medical_question: BẤT KỲ câu hỏi nào liên quan đến kiến thức y khoa (dù rõ ràng hay mơ hồ) - cần tra cứu cơ sở tri thức (RAG).
- chitchat: chỉ chào hỏi/trò chuyện xã giao KHÔNG LIÊN QUAN đến y khoa.

Ví dụ medical_question (bao gồm cả câu hỏi mơ hồ):
- "Triệu chứng của bệnh đái tháo đường type 2 là gì?" (cụ thể)
- "Tôi muốn hỏi về bệnh" (mơ hồ nhưng vẫn là medical_question)
- "Cho tôi biết về điều trị" (mơ hồ nhưng vẫn là medical_question)
- "Làm sao để chăm sóc răng miệng?" (medical_question)
- "Có thông tin gì về đái tháo đường?" (medical_question)

Ví dụ chitchat:
- "Xin chào"
- "Cảm ơn bạn"
- "Tạm biệt"
- "Bạn khỏe không?"

QUAN TRỌNG: Nếu câu hỏi có BẤT KỲ yếu tố y khoa nào (dù mơ hồ), phân loại là medical_question.

Ngữ cảnh hội thoại gần đây:
{formatted_history}

Input của user: "{query}"
Role của user: {role}

Trả về CHỈ một code block YAML hợp lệ:

```yaml
type: medical_question  # hoặc chitchat
confidence: high  # hoặc medium, low
reason: "Lý do ngắn gọn"
```
"""

        try:
            resp = call_llm(prompt, fast_mode=True, max_retry_time=timeout_config.LLM_RETRY_TIMEOUT)

            logger.info(f"[MainDecision] EXEC - resp: {resp}")
            result = parse_yaml_with_schema(
                resp,
                required_fields=["type"],
                optional_fields=["confidence", "reason"],
                field_types={"type": str, "confidence": str, "reason": str}
            )
            logger.info(f"[MainDecision] EXEC - result after parse: {result}")

            if result:
                logger.info(f"[MainDecision] EXEC - LLM classification: {result}")
                return result
        except APIOverloadException as e:
            logger.warning(f"[MainDecision] EXEC - API overloaded, triggering fallback: {e}")
            return {"type": "api_overload", "confidence": "high"}
        except Exception as e:
            logger.warning(f"[MainDecision] EXEC - LLM classification failed: {e}")

        return {"type": "default", "confidence": "high"}

    def post(self, shared, prep_res, exec_res):
        logger.info(f"[MainDecision] POST - Classification result: {exec_res}")
        shared["input_type"] = exec_res["type"]
        shared["classification_confidence"] = exec_res.get("confidence", "low")
        shared["classification_reason"] = exec_res.get("reason", "")

        # Route based on classification - ONLY two options
        input_type = exec_res["type"]

        if input_type == "medical_question":
            logger.info("[MainDecision] POST - Medical question detected, routing to RAG")
            return "retrieve_kb"
        elif input_type == "chitchat":
            logger.info("[MainDecision] POST - Chitchat detected, routing to chitchat handler")
            return "chitchat"
        elif input_type == "api_overload" or input_type == "default":
            return "fallback"
        else:
            # Fallback mặc định
            return "chitchat"

class FallbackNode(Node):
    """Node fallback khi API quá tải - retrieve query và trả kết quả dựa trên score"""
    
    def prep(self, shared):
        logger.info("🔄 [FallbackNode] PREP - Xử lý fallback khi API quá tải")
        query = shared.get("query", "")
        role = shared.get("role", "")
        return query, role

    def exec(self, inputs):
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

