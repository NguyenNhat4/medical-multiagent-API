from math import log
from unittest import result
from pocketflow import Node
from utils.call_llm import call_llm
from utils.kb import retrieve, retrieve_random_by_role
from utils.conversation_logger import log_user_message, log_bot_response, log_conversation_exchange
from utils.response_parser import parse_yaml_response, validate_yaml_structure, parse_yaml_with_schema
from utils.prompts import (
    PROMPT_CLASSIFY_INPUT, 
    PROMPT_CLARIFYING_QUESTIONS_GENERIC,
    PROMPT_COMPOSE_ANSWER,
    PROMPT_SUGGEST_FOLLOWUPS
)
from utils.helpers import (
    get_persona_for,
    get_topics_by_role,
    get_fallback_topics_by_role,
    get_most_relevant_QA,
    get_context_for_input_type,
    get_context_for_knowledge_case,
    get_score_threshold,
    
)
from typing import Any, Dict, List, Tuple
import textwrap
import yaml
import logging
import re

# Configure logging for this module
logger = logging.getLogger(__name__)


class AnswerNode(Node):
    def prep(self, shared):
        # Read question from shared
        return shared["question"]
    
    def exec(self, question):
        # Call LLM to get the answer
        return call_llm(question)
    
    def post(self, shared, prep_res, exec_res):
        # Store the answer in shared
        shared["answer"] = exec_res



# Removed _persona_for function - now imported from utils.helpers

# ========== Medical Agent Nodes ==========

class ClassifyInput(Node):
    """Node để phân loại input của user thành các loại khác nhau"""
    
    def prep(self, shared):
        logger.info("[ClassifyInput] PREP - Đọc query để phân loại")
        query = shared.get("query", "").strip()
        role = shared.get("role", "")
        logger.info(f"[ClassifyInput] PREP - Query: '{query}', Role: {role}")
        return query, role
    
    def exec(self, inputs):
        query, role = inputs

        pattern_result = classify_input_pattern(query)
        if pattern_result["confidence"] in ["high", "medium"]:
            logger.info(f"[ClassifyInput] EXEC - Pattern classification: {pattern_result}")
            return pattern_result
        
        # For ambiguous cases, use LLM
        if len(query) > 3:
            logger.info("[ClassifyInput] EXEC - Using LLM for classification")
            prompt = PROMPT_CLASSIFY_INPUT.format(query=query, role=role)
            
            try:
                resp = call_llm(prompt)
                result = parse_yaml_with_schema(
                    resp,
                    required_fields=["type"],
                    optional_fields=["confidence", "reason"],
                    field_types={"type": str, "confidence": str, "reason": str}
                )
                logger.info(f"[ClassifyInput] EXEC -resp {resp}")
                logger.info(f"[ClassifyInput] EXEC - result {result}")

                if result:
                    logger.info(f"[ClassifyInput] EXEC - LLM classification: {result}")
                    return result
            except Exception as e:
                logger.warning(f"[ClassifyInput] EXEC - LLM classification failed: {e}")
        
        return {"type": "topic_suggestion", "confidence": "high"}
    
    def post(self, shared, prep_res, exec_res):
        logger.info(f"[ClassifyInput] POST - Classification result: {exec_res}")
        shared["input_type"] = exec_res["type"]
        shared["classification_confidence"] = exec_res.get("confidence", "low")
        shared["classification_reason"] = exec_res.get("reason", "")
        
        # Return action based on classification
        return exec_res["type"]


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
        logger.info("📚 [RetrieveFromKB] PREP - Đọc query và keywords để retrieve")
        query = shared.get("query", "")
        keywords = shared.get("keywords", [])
        
        # Ưu tiên dùng keywords nếu có, nếu không thì dùng query gốc
        search_term = " ".join(keywords) if keywords else query
        logger.info(f"📚 [RetrieveFromKB] PREP - Search Term: '{search_term[:100]}...'")
        return search_term

    def exec(self, search_term: str):
        logger.info("📚 [RetrieveFromKB] EXEC - Bắt đầu retrieve từ knowledge base")
        logger.info(f"📚 [RetrieveFromKB] EXEC - Query: {search_term}")
        results, score = retrieve(search_term, top_k=4)
        logger.info(f"📚 [RetrieveFromKB] EXEC - Retrieved results: {results} , best score: {score:.4f}")
        return results, score

    def post(self, shared, prep_res, exec_res):
        logger.info("📚 [RetrieveFromKB] POST - Lưu kết quả retrieve")
        results, score = exec_res
        shared["retrieved"] = results
        shared["retrieval_score"] = score
        shared["need_clarify"] = score < 0.15
        
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
        shared["explain"] = "Chào bạn! Tôi là AI nha khoa. Nếu bạn có câu hỏi hoặc cần gợi ý các câu hỏi liên quan đến nha khoa và sức khỏe răng miệng hoặc đái tháo đường, thì cứ nói tôi nhé 😊"
        return "default"


class ComposeAnswer(Node):
    def prep(self, shared):
        role = shared.get("role", "")
        query = shared.get("query", "")
        retrieved = shared.get("retrieved", [])
        score = shared.get("retrieval_score", 0.0)
        conversation_history = shared.get("conversation_history", [])
        logger.info(f"✍️ [ComposeAnswer] retrieved: {retrieved}")
        return (role, query, retrieved, score, conversation_history)

    def exec(self, inputs):
        role, query, retrieved,  score, conversation_history = inputs
        persona = get_persona_for(role)

        relevant_info_from_kb = get_most_relevant_QA(retrieved)
        prompt = PROMPT_COMPOSE_ANSWER.format(
            ai_role=persona['persona'],
            audience=persona['audience'],
            tone=persona['tone'],
            query=query,
            relevant_info_from_kb=relevant_info_from_kb,
            conversation_history = conversation_history
        )
        logger.info(f"✍️ [ComposeAnswer] EXEC - prompt: {prompt}")
        result = call_llm(prompt)
        logger.info(f"✍️ [ComposeAnswer] EXEC - LLM response received")
        result = parse_yaml_with_schema(result, required_fields=["explanation", "suggestion_questions"], field_types={"explanation": str, "suggestion_questions": list})
        logger.info(f"✍️ [ComposeAnswer] EXEC - result: {result}")

        if not result or  isinstance(result, str):
            logger.warning("[ComposeAnswer] EXEC - Invalid LLM response, using fallback")
            resp = "Xin lỗi, tôi không thể tạo câu trả lời phù hợp lúc này. Bạn đặt câu hỏi khác được không? "
            return {"explain": resp, "suggestion_questions": [], "preformatted": True}
        
        return {"explain": result.get("explanation", ""), "suggestion_questions": result.get("suggestion_questions", []), "preformatted": True}


    def post(self, shared, prep_res, exec_res):
        logger.info("✍️ [ComposeAnswer] POST - Lưu answer object")
        shared["answer_obj"] = exec_res
        shared["explain"] = exec_res.get("explain", "")
        shared["suggestion_questions"] = exec_res.get("suggestion_questions", [])
        logger.info(f"✍️ [ComposeAnswer] POST - Answer keys: {list(exec_res.keys())}")
        logger.info(f"✍️ [ComposeAnswer] POST - Answer preview: {exec_res.get('explain')}")
        return "default"



class TopicSuggestResponse(Node):
    """Node xử lý gợi ý topic với template khác nhau cho từng context"""
    
    def prep(self, shared):
        role = shared.get("role", "")
        query = shared.get("query", "")
        retrieved = shared.get("retrieved", [])
        logger.info(f"[TopicSuggestResponse] content retrieve: {retrieved}")
        context = shared.get("response_context", "default") 
        retrieval_score = shared.get("retrieval_score", 0.0)
        return role, query, retrieved, context, retrieval_score
    
    
    def exec(self, inputs):
        role, query, retrieved, context, retrieval_score = inputs
        result =    {
            "explain": "",
            "suggestion_questions" : [],
            "preformatted": True,
        }
        suggestion_questions = [q['cau_hoi'] for q in retrieve_random_by_role(role, amount=10)]
        # Handle low-score medical questions specifically
        if context == "medical_low_score":
            logger.info(f"[TopicSuggestResponse] EXEC - Handling low-score medical query: '{query}'")
            result["explain"] = "Hiện mình chưa tìm được câu trả lời trong dữ sẵn có. Bạn thông cảm nhé!. Mình có các hướng sau bạn có thể quan tâm nè."
        
        if context == "topic_suggestion":
            result["explain"] = "Mình gợi ý bạn các chủ đề sau nhé"
        
        result["suggestion_questions"] = suggestion_questions

        return result

    
    def post(self, shared, prep_res, exec_res):
        logger.info("[TopicSuggestResponse] POST - Lưu topic suggestion response")
        shared["answer_obj"] = exec_res
        shared["explain"] = exec_res.get("explain", "")
        shared["suggestion_questions"] = exec_res.get("suggestion_questions", [])  # Lấy 3 đầu làm suggestions
        return "default"



class LogConversationNode(Node):
    """Node để log cuộc trò chuyện user-bot vào file"""
    
    def prep(self, shared):
        logger.info("[LogConversation] PREP - Chuẩn bị log conversation")
        user_query = shared.get("query", "")
        bot_answer = shared.get("answer", "")
        return user_query, bot_answer
    
    def exec(self, inputs):
        logger.info("[LogConversation] EXEC - Logging conversation to file")
        user_query, bot_answer = inputs
        
        # Log the complete exchange
        log_conversation_exchange(user_query, bot_answer)
        
        return {"logged": True, "user_query": user_query, "bot_answer": bot_answer}
    
    def post(self, shared, prep_res, exec_res):
        logger.info(f"[LogConversation] POST - Logged conversation: user='{exec_res['user_query'][:50]}...' bot='{exec_res['bot_answer'][:50]}...'")
        shared["conversation_logged"] = exec_res["logged"]
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
            resp = call_llm(prompt  )
            logger.info(f"[MainDecision] EXEC - resp: {resp}")
            result = parse_yaml_with_schema(
                resp,
                required_fields=["type"],
                optional_fields=["confidence", "reason", "keywords"],
                field_types={"type": str, "confidence": str, "reason": str, "keywords": list}
            )
            logger.info(f"[MainDecision] EXEC - result after parse: {result}")
            
            if result:
                logger.info(f"[MainDecision] EXEC - LLM classification: {result}")
                return result       
        except Exception as e:
            logger.warning(f"[MainDecision] EXEC - LLM classification failed: {e}")
        
        # Default fallback
        return {"type": "topic_suggestion", "confidence": "high", "keywords": []}
    
    def post(self, shared, prep_res, exec_res):
        logger.info(f"[MainDecision] POST - Classification result: {exec_res}")
        shared["input_type"] = exec_res["type"]
        shared["classification_confidence"] = exec_res.get("confidence", "low")
        shared["classification_reason"] = exec_res.get("reason", "")
        shared["keywords"] = exec_res.get("keywords", [])
        
        # Route based on classification
        input_type = exec_res["type"]
        
        if input_type == "medical_question":
            return "retrieve_kb"
        elif input_type == "topic_suggestion":
            return "retrieve_kb"
        elif input_type == "greeting":
            return "greeting"
        else:
            return "topic_suggestion"


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
                return {"action": "topic_suggest", "context": "medical_low_score"}
        
        return {"action": "topic_suggest", "context": "topic_suggestion"}
   
    def post(self, shared, prep_res, exec_res):
        shared["response_context"] = exec_res["context"]
        logger.info(f"[ScoreDecision] POST - Decision: {exec_res['action']}, Context: {exec_res['context']}")
        return exec_res["action"]
