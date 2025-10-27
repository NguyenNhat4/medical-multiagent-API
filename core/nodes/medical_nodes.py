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

class RetrieveFromKB(Node):
    def prep(self, shared):
        logger.info("📚 [RetrieveFromKB] PREP - Đọc query và danh sách rag_questions để retrieve tuần tự")
        query = shared.get("query", "")
        rag_questions = shared.get("rag_questions", [])
        user_role =  shared.get("role", "")
        logger.info(f"📚 [RetrieveFromKB] PREP - query='{str(query)[:80]}...', rag_questions={len(rag_questions) if rag_questions else 0}")
        return query, rag_questions, user_role

    def exec(self, inputs):
        query, rag_questions, user_role = inputs
        logger.info("📚 [RetrieveFromKB] EXEC - Bắt đầu retrieve tuần tự: user input trước, sau đó RAG")

        # Xây danh sách truy vấn: ưu tiên user input trước, rồi đến rag_questions
        retrieval_queries = []
        if query:
            retrieval_queries.append(query)
        if rag_questions:
            retrieval_queries.extend([q for q in rag_questions if q])

        # Use aggregate_retrievals helper function
        top5, top_score = aggregate_retrievals(retrieval_queries, role=user_role, top_k=5)

        # Log formatted QA list and score table
        try:
            formatted = format_kb_qa_list(top5, max_items=5)
            if formatted:
                logger.info("\n📚 [RetrieveFromKB] FORMATTED Top-5:\n" + formatted)
        except Exception:
            pass

        if top5:
            lines = ["\n🏷️ [RetrieveFromKB] TOP-5 SCORES (desc):"]
            for i, it in enumerate(top5, 1):
                lines.append(f"  {i}. score={float(it.get('score',0.0)):.4f} | Q: {str(it.get('cau_hoi',''))[:140]}")
            logger.info("\n".join(lines))

        logger.info(f"📚 [RetrieveFromKB] EXEC - Aggregated top5={len(top5)}, top_score={top_score:.4f}")
        return top5, top_score

    def post(self, shared, prep_res, exec_res):
        logger.info("📚 [RetrieveFromKB] POST - Lưu kết quả retrieve")
        results, score = exec_res
        shared["retrieved"] = results
        shared["retrieval_score"] = score
        shared["need_clarify"] = score < get_score_threshold()
        
        # Always continue to next node via default edge (ScoreDecisionNode)
        input_type = shared.get("input_type", "medical_question")
        logger.info(
            f"📚 [RetrieveFromKB] POST - Saved {len(results)} results, score: {score:.4f}, "
            f"input_type={input_type} -> routing via 'default' to ScoreDecision"
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

class ClarifyQuestionNode(Node):
    """Node xử lý clarification cho medical questions có score thấp"""
    
    def prep(self, shared):
        role = shared.get("role", "")
        query = shared.get("query", "")
        retrieved = shared.get("retrieved", [])
        rag_questions = shared.get("rag_questions", [])
        logger.info(f"[ClarifyQuestion] PREP - Role: {role}, Query: '{query[:50]}...', RAG Questions: {len(rag_questions)}")
        return role, query, retrieved, rag_questions
    
    def exec(self, inputs):
        role, query, retrieved, rag_questions = inputs
        logger.info(f"[ClarifyQuestion] EXEC - Generating clarification for low-score medical query")
        
        # Lấy danh sách câu hỏi từ retrieved hoặc random nếu retrieved trống
        if not retrieved:
            suggestion_questions = [q['cau_hoi'] for q in retrieve_random_by_role(role, amount=4)]
        else:
            # Lấy câu hỏi từ retrieved data
            suggestion_questions = [item.get('cau_hoi', '') for item in retrieved if item.get('cau_hoi')][:5]
        
        
        result = {
            "explain": "Hiện tại tôi chưa có đủ thông tin liên quan để trả lời câu hỏi này của bạn, Bạn có thể đặt lại câu hỏi khác hoặc diễn đạt lại câu hỏi của bạn! Hoặc bạn có thể chọn các câu hỏi gợi ý dưới đây!",
            "suggestion_questions": suggestion_questions,
            "preformatted": True,
        }
        
        logger.info(f"[ClarifyQuestion] EXEC - Generated {len(suggestion_questions)} clarification questions")
        return result
    
    def post(self, shared, prep_res, exec_res):
        logger.info("[ClarifyQuestion] POST - Lưu clarification response")
        shared["answer_obj"] = exec_res
        shared["explain"] = exec_res.get("explain", "")
        shared["suggestion_questions"] = exec_res.get("suggestion_questions", [])
        return "default"

class MainDecisionAgent(Node):
    """Main decision agent - chỉ phân loại input và routing"""
    
    def prep(self, shared):
        logger.info("[MainDecision] PREP - Đọc query để phân loại")
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
        logger.info("[MainDecision] EXEC - Using LLM for classification")
        prompt = PROMPT_CLASSIFY_INPUT.format(query=query, role=role, conversation_history=formatted_history)
        
        try:
            resp = call_llm(prompt, fast_mode=True, max_retry_time=timeout_config.LLM_RETRY_TIMEOUT)
            
            
            logger.info(f"[MainDecision] EXEC - resp: {resp}")
            result = parse_yaml_with_schema(
                resp,
                required_fields=["type"],
                optional_fields=["confidence", "reason", "rag_questions"],
                field_types={"type": str, "confidence": str, "reason": str, "rag_questions": list}
            )
            logger.info(f"[MainDecision] EXEC - result after parse: {result}")
            
            if result:
                logger.info(f"[MainDecision] EXEC - LLM classification: {result}")
                return result       
        except APIOverloadException as e:
            logger.warning(f"[MainDecision] EXEC - API overloaded, triggering fallback: {e}")
            return {"type": "api_overload", "confidence": "high", "rag_questions": []}
        except Exception as e:
            logger.warning(f"[MainDecision] EXEC - LLM classification failed: {e}")
        
        return {"type": "default", "confidence": "high", "rag_questions": []}
    
    def post(self, shared, prep_res, exec_res):
        logger.info(f"[MainDecision] POST - Classification result: {exec_res}")
        shared["input_type"] = exec_res["type"]
        shared["classification_confidence"] = exec_res.get("confidence", "low")
        shared["classification_reason"] = exec_res.get("reason", "")
        shared["rag_questions"] = exec_res.get("rag_questions", [])
        
        # Route based on classification
        input_type = exec_res["type"]
        
        if input_type == "medical_question":
            return "retrieve_kb"
        elif input_type == "chitchat":
            return "chitchat"
        elif input_type == "api_overload" or input_type == "default":
            return "fallback"
        else:
            # Mặc định không đẩy sang topic_suggest nữa
            return "chitchat"

class FallbackNode(Node):
    """Node fallback khi API quá tải - retrieve query và trả kết quả dựa trên score"""
    
    def prep(self, shared):
        logger.info("🔄 [FallbackNode] PREP - Xử lý fallback khi API quá tải")
        query = shared.get("query", "")
        role = shared.get("role", "")
        rag_questions = shared.get("rag_questions", [])
        return query, role, rag_questions
    
    def exec(self, inputs):
        query, role, rag_questions = inputs
        logger.info(f"🔄 [FallbackNode] EXEC - Fallback search cho role: {role} với query: '{query[:50]}...', rag_questions: {len(rag_questions) if rag_questions else 0}")
        
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

            # Build retrieval queries: user input first, then rag_questions (if any)
            retrieval_queries = []
            if query:
                retrieval_queries.append(query)
            if rag_questions:
                retrieval_queries.extend([q for q in rag_questions if q])

            # Use aggregate_retrievals helper function
            top5, _ = aggregate_retrievals(retrieval_queries, role=role, top_k=5)

            try:
                formatted = format_kb_qa_list(top5, max_items=5)
                if formatted:
                    logger.info("\n📚 [FallbackNode] RETRIEVE - Aggregated Results (Top 5):\n" + formatted)
            except Exception:
                pass

            # Log thêm bảng điểm cho top5
            if top5:
                lines = ["\n🏷️ [FallbackNode] TOP-5 SCORES (desc):"]
                for i, it in enumerate(top5, 1):
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
                for it in top5:
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
                    score_map = {str(it.get('cau_hoi', '')): float(it.get('score', 0.0)) for it in top5}
                    sug_lines = ["📌 [FallbackNode] SUGGESTIONS (top4):"]
                    for idx, sq in enumerate(suggestion_questions, 1):
                        sug_lines.append(f"  {idx}. score={score_map.get(sq, 0.0):.4f} | Q: {sq[:140]}")
                    logger.info("\n".join(sug_lines))
            else:
                # Không có exact match: nếu có top5, dùng top1 làm explain và còn lại làm suggestion
                if top5:
                    best_answer = top5[0]
                    explain = best_answer.get("cau_tra_loi", "")
                    suggestion_questions = [it.get('cau_hoi', '') for it in top5[1:5] if it.get('cau_hoi')]
                    score = float(best_answer.get('score', 0.0))
                    # Log lựa chọn cuối
                    logger.info(f"\n✅ [FallbackNode] EXPLAIN (retrieve top1): score={score:.4f} | Q: {str(best_answer.get('cau_hoi',''))[:140]}")
                    if suggestion_questions:
                        sug_lines = ["📌 [FallbackNode] SUGGESTIONS (next4):"]
                        for idx, it in enumerate(top5[1:5], 1):
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

class ScoreDecisionNode(Node):
    """Node quyết định dựa trên retrieval score"""
    
    def prep(self, shared):
        logger.info("[ScoreDecision] PREP - Kiểm tra retrieval score")
        input_type = shared.get("input_type", "")
        retrieval_score = shared.get("retrieval_score", 0.0)
        return input_type, retrieval_score
    
    
    def exec(self, inputs):
        input_type, retrieval_score = inputs
        score_threshold = get_score_threshold()
        
        logger.info(f"[ScoreDecision] EXEC - Input: '{input_type}', Score: {retrieval_score:.4f}, Threshold: {score_threshold}")
        
        if input_type == "medical_question":

            if retrieval_score >= score_threshold:
                return {"action": "compose_answer", "context": "medical_high_score"}
          
            
        return {"action": "clarify", "context": "default_fallback"}
   
    def post(self, shared, prep_res, exec_res):
        shared["response_context"] = exec_res["context"]
        logger.info(f"[ScoreDecision] POST - Decision: {exec_res['action']}, Context: {exec_res['context']}")
        return exec_res["action"]
