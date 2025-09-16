from math import log
from unittest import result
from pocketflow import Node
from utils.call_llm import call_llm, APIOverloadException
from utils.kb import retrieve, retrieve_random_by_role

from utils.response_parser import parse_yaml_response, validate_yaml_structure, parse_yaml_with_schema
from utils.prompts import (
    PROMPT_CLASSIFY_INPUT, 
    PROMPT_COMPOSE_ANSWER
)
from utils.helpers import (
    format_kb_qa_list,
    get_score_threshold,
    format_conversation_history,
    log_llm_timing
)
from utils.role_ENUM import (
    PERSONA_BY_ROLE
)
from typing import Any, Dict, List, Tuple
import textwrap
import yaml
import logging
import re
import time

# Configure logging for this module
logger = logging.getLogger(__name__)

class AnswerNode(Node):
    def prep(self, shared):
        return shared["question"]
    
    def exec(self, question):
        start_time = time.time()
        result = call_llm(question)
        end_time = time.time()
        
        # Log LLM timing
        log_llm_timing("AnswerNode", start_time, end_time, len(question))
        
        return result
    
    def post(self, shared, prep_res, exec_res):
        shared["answer"] = exec_res




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
        logger.info("📚 [RetrieveFromKB] PREP - Đọc query và rag_questions để retrieve")
        query = shared.get("query", "")
        rag_questions = shared.get("rag_questions", [])
        # Kết hợp query gốc với các câu hỏi RAG để tìm kiếm toàn diện hơn
        all_queries = [query] + rag_questions
        search_term = " ".join(all_queries)   
        user_role =  shared.get("role", "")
        logger.info(f"📚 [RetrieveFromKB] PREP - Search Term: '{search_term[:100]}...'")
        return search_term, user_role

    def exec(self, inputs):
        search_term, user_role = inputs
        logger.info("📚 [RetrieveFromKB] EXEC - Bắt đầu retrieve từ knowledge base")
        logger.info(f"📚 [RetrieveFromKB] EXEC - Query: {search_term}")
        import time

        start_time = time.time()
        # Reduce retrieval breadth
        results, score = retrieve(search_term, user_role, top_k=5)
        elapsed_time = time.time() - start_time

        # Log elapsed time to a file
        with open("retrieve_timing.log", "a", encoding="utf-8") as f:
            f.write(f" Time: {elapsed_time:.4f} seconds\n")
        logger.info(f"📚 [RetrieveFromKB] EXEC - Retrieved results: {results} , best score: {score:.4f}")
        return results, score

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
    """Node xử lý chào hỏi - set context và route đến topic suggestion"""
    def prep(self, shared):
        return shared.get("role", ""), shared.get("query", "")
    
    def exec(self, inputs):
        role, query = inputs
        return {"context_set": True, "role": role, "query": query}
    
    def post(self, shared, prep_res, exec_res):
        shared["explain"] = "Xin chào 😊! Tôi là trợ lý AI của bạn. Rất vui được hỗ trợ bạn - Bạn cần tôi giúp gì hôm nay? "
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
            result = call_llm(prompt)
            end_time = time.time()
            
            # Log LLM timing
            log_llm_timing("ComposeAnswer", start_time, end_time, len(prompt))
            
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
            "explain": "Hiện tại mình chưa có đủ thông tin liên quan để trả lời câu hỏi này của bạn, Bạn có thể đặt lại câu hỏi khác hoặc diễn đạt lại câu hỏi của bạn! Hoặc bạn có thể chọn các câu hỏi gợi ý dưới đây!",
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


class TopicSuggestResponse(Node):
    """Node xử lý gợi ý topic khi user yêu cầu gợi ý chủ đề"""
    def prep(self, shared):
        role = shared.get("role", "")
        query = shared.get("query", "")
        logger.info(f"[TopicSuggestResponse] PREP - Role: {role}, Query: '{query[:50]}...'")
        return role, query
    
    def exec(self, inputs):
        role, query = inputs
        logger.info(f"[TopicSuggestResponse] EXEC - Generating topic suggestions for role: {role}")
        
        # Get fewer topic suggestions to reduce tokens
        suggestion_questions = [q['cau_hoi'] for q in retrieve_random_by_role(role, amount=5)]
        
        result = {
            "explain": "Mình gợi ý bạn các chủ đề sau nhé! Bạn có thể chọn bất kỳ chủ đề nào mà bạn quan tâm 😊",
            "suggestion_questions": suggestion_questions,
            "preformatted": True,
        }
        
        logger.info(f"[TopicSuggestResponse] EXEC - Generated {len(suggestion_questions)} topic suggestions")
        return result
    
    def post(self, shared, prep_res, exec_res):
        logger.info("[TopicSuggestResponse] POST - Lưu topic suggestion response")
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
        return query, role
    
    def exec(self, inputs):
        query, role = inputs
        logger.info("[MainDecision] EXEC - Using LLM for classification")
        prompt = PROMPT_CLASSIFY_INPUT.format(query=query, role=role)
        
        try:
            start_time = time.time()
            resp = call_llm(prompt)
            end_time = time.time()
            
            # Log LLM timing
            log_llm_timing("MainDecisionAgent", start_time, end_time, len(prompt))
            
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
        elif input_type == "greeting":
            return "greeting"
        elif input_type == "api_overload" or input_type == "default":
            return "fallback"
        else:
            return "topic_suggest"


class FallbackNode(Node):
    """Node fallback khi API quá tải - retrieve query và trả kết quả dựa trên score"""
    
    def prep(self, shared):
        logger.info("🔄 [FallbackNode] PREP - Xử lý fallback khi API quá tải")
        query = shared.get("query", "")
        role = shared.get("role", "")
        return query, role
    
    def exec(self, inputs):
        query, role = inputs
        logger.info(f"🔄 [FallbackNode] EXEC - Retrieve từ query: '{query[:50]}...' cho role: {role}")
        
        try:
            # Retrieve từ knowledge base
            results, score = retrieve(query, role, top_k=5)
            logger.info(f"🔄 [FallbackNode] EXEC - Retrieved {len(results)} results, best score: {score:.4f}")
            logger.info(f"🔄 [FallbackNode] EXEC - Results: {results}")
            # Kiểm tra score threshold
            if score > 0.35:
                # Có kết quả tốt - lấy câu trả lời có score cao nhất
                best_answer = results[0] if results else None
                if best_answer:
                    explain = best_answer.get("cau_tra_loi", "")
                    # Lấy thêm câu hỏi gợi ý từ kết quả retrieve
                    suggestion_questions = [item.get('cau_hoi', '') for item in results[1:4] if item.get('cau_hoi')]
                else:
                    explain = "Xin lỗi, không thể lấy được thông tin phù hợp lúc này."
                    suggestion_questions = []
            else:
                # Score thấp - trả về thông báo mặc định + câu hỏi gợi ý từ retrieve
                explain = "Hiện tại mình chưa có đủ thông tin liên quan để trả lời câu hỏi này của bạn, Bạn có thể đặt lại câu hỏi khác hoặc diễn đạt lại câu hỏi của bạn! Hoặc bạn có thể chọn các câu hỏi gợi ý dưới đây!"
                # Lấy câu hỏi gợi ý từ kết quả retrieve (nếu có), fallback sang random nếu không có
                if results and len(results) > 0:
                    suggestion_questions = [item.get('cau_hoi', '') for item in results if item.get('cau_hoi')][:5]
                    # Nếu không đủ câu hỏi từ retrieve, bổ sung thêm từ random
                    if len(suggestion_questions) < 3:
                        random_questions = retrieve_random_by_role(role, amount=5-len(suggestion_questions))
                        suggestion_questions.extend([q['cau_hoi'] for q in random_questions])
                else:
                    # Không có kết quả retrieve, dùng random
                    random_questions = retrieve_random_by_role(role, amount=5)
                    suggestion_questions = [q['cau_hoi'] for q in random_questions]
            
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
            else:
                return {"action": "clarify", "context": "medical_low_score"}
            
        return {"action": "clarify", "context": "topic_suggestion"}
   
    def post(self, shared, prep_res, exec_res):
        shared["response_context"] = exec_res["context"]
        logger.info(f"[ScoreDecision] POST - Decision: {exec_res['action']}, Context: {exec_res['context']}")
        return exec_res["action"]
