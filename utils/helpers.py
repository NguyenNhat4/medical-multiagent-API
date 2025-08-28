"""
Helper functions for medical agent nodes
"""

from typing import Dict, List, Tuple, Any
import logging
import re
import yaml
from .call_llm import call_llm
from .kb import retrieve, retrieve_random_by_role
from .response_parser import parse_yaml_response, validate_yaml_structure

logger = logging.getLogger(__name__)


def get_persona_for(role: str) -> Dict[str, str]:
    """Get persona configuration based on user role"""
    rl = role.lower()
    if "bác sĩ nha khoa" in rl:
        return {
            "persona": "Bác sĩ nội tiết (chuyên ĐTĐ)",
            "audience": "bác sĩ nha khoa",
            "tone": (
                "Thái độ thân thiện, giải thích ngắn gọn, dùng từ phù hợp để bác sĩ nha khoa hiểu, mục tiêu giúp bác sĩ nha khoa hiểu hơn về nội tiết "
            ),
        }
    if "bác sĩ nội tiết" in rl:
        return {
            "persona": "Bác sĩ nha khoa",
            "audience": "bác sĩ nội tiết",
            "tone": (
                "Thái độ thân thiện, giải thích ngắn gọn, dùng từ phù hợp để bác sĩ nội tiết hiểu, mục tiêu giúp bác sĩ nội tiết hiểu hơn về nha khoa phục vụ cho nghành nghề của mình"
            ),
        }
    if "bệnh nhân đái tháo đường" in rl or "đái tháo đường" in rl:
        return {
            "persona": "Bác sĩ nội tiết",
            "audience": "bệnh nhân",
            "tone": (
                "Thái độ thân thiện, Ngôn ngữ giản dị, không nói dài dòng,không dùng từ chuyên môn. "
            ),
        }
    # Bệnh nhân nha khoa (mặc định)
    return {
        "persona": "Bác sĩ nha khoa",
        "audience": "bệnh nhân",
        "tone": (
            "Thái độ thân thiện, Ngôn ngữ thân thiện, không  dùng từ chuyên môn"
        ),
    }


def get_topics_by_role(role: str) -> Tuple[List[str], str]:
    """Retrieve topics from KB based on user role - returns topics and intro"""
    try:
        # Retrieve ngẫu nhiên từ CSV file tương ứng với role
        results = retrieve_random_by_role(role, count=7)
        
        if results:
            topics = []
            for item in results:
                # Sử dụng câu hỏi làm topic
                topic = item.get('cau_hoi', '').strip()
                if topic and len(topic) > 10:  # Lọc câu hỏi có ý nghĩa
                    topics.append(topic)
            
            # Tạo intro phù hợp với role
            persona = get_persona_for(role)
            intro = f"Dưới đây là một số chủ đề phù hợp với {persona['audience']} từ cơ sở tri thức:"
            
            return topics[:7]
        else:
            return [], ""
            
    except Exception as e:
        logger.warning(f"Failed to retrieve random topics for role '{role}': {e}")
        return [], ""


def get_fallback_topics_by_role(role: str) -> List[str]:
    """Fallback topics based on role when cannot retrieve from KB"""
    role_lower = role.lower()
    
    if "bác sĩ nha khoa" in role_lower:
        return [
            "Quản lý bệnh nhân đái tháo đường trong nha khoa",
            "Điều trị viêm nha chu ở bệnh nhân ĐTĐ",
            "Phối hợp với bác sĩ nội tiết trong điều trị",
            "Biến chứng nha khoa do đái tháo đường",
            "Kháng sinh trong điều trị nha khoa bệnh nhân ĐTĐ"
        ]
    elif "bác sĩ nội tiết" in role_lower:
        return [
            "Mối liên hệ giữa kiểm soát đường huyết và sức khỏe nha chu",
            "Khi nào giới thiệu bệnh nhân đến nha khoa",
            "Thuốc đái tháo đường ảnh hưởng đến răng miệng",
            "Biến chứng răng miệng ở bệnh nhân ĐTĐ type 1 và type 2",
            "Tư vấn chăm sóc răng miệng cho bệnh nhân ĐTĐ"
        ]
    elif "đái tháo đường" in role_lower:
        return [
            "Cách chăm sóc răng miệng khi bị đái tháo đường",
            "Triệu chứng cảnh báo ở răng miệng cần chú ý",
            "Khi nào cần đi khám nha khoa",
            "Chế độ ăn tốt cho răng miệng và đường huyết",
            "Cách đánh răng đúng cách cho người ĐTĐ"
        ]
    else:
        # Default cho bệnh nhân nha khoa hoặc không xác định
        return [
            "Cách chăm sóc răng miệng hàng ngày",
            "Dấu hiệu cần khám nha khoa ngay",
            "Phòng ngừa sâu răng và viêm nướu",
            "Chế độ ăn uống tốt cho răng miệng",
            "Tần suất khám nha khoa định kỳ"
        ]

def get_most_relevant_QA(hits: List[Dict[str, Any]]) -> str:
    """Return the first Q&A from KB hits that has a non-empty answer"""
    if not hits:
        return ""
    # Find the first hit with a non-empty answer
    for item in hits:
        answer = str(item.get('cau_tra_loi', '')).strip()
        question = str(item.get('cau_hoi', '')).strip()
        if answer:
            q_part = f"Q: {question}\n" if question else ""
            return f"{q_part}A: {answer}".strip()
    # No answers found
    return ""


def build_history_text(history: List[Dict[str, str]]) -> str:
    """Format conversation history for prompts"""
    return "\n".join([f"{m.get('role')}: {m.get('content')}" for m in history[-3:]])


def generate_clarifying_questions(query: str, history: List[Dict[str, str]], hits: List[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Generate clarifying questions for generic or low-score queries"""
    from .prompts import PROMPT_CLARIFYING_QUESTIONS_GENERIC
    
    history_text = build_history_text(history)
    
    # Unified handling: always use the generic prompt,
    # build KB context from hits if available, otherwise provide a placeholder
    ctx_lines = []
    if hits:
        for h in hits[:5]:
            ctx_lines.append(f"- {h['de_muc']} > {h['chu_de_con']} | Q: {h['cau_hoi']}")
    kb_ctx = "\n".join(ctx_lines) if ctx_lines else "(Không có ngữ cảnh KB phù hợp)"

    prompt = PROMPT_CLARIFYING_QUESTIONS_GENERIC.format(
        query=query, history_text=history_text, kb_ctx=kb_ctx
    )
    
    try:
        resp = call_llm(prompt)
        data = parse_yaml_response(resp)
        
        # Handle the new simple response format for low-score queries
        if "response" in data:
            return {
                "final": data["response"],
                "preformatted": True,
                "need_clarify": False,
                "suggestions": [],
                "context": "low_score_no_info"
            }

        if data and validate_yaml_structure(data, required_fields=["questions"]):
            qs = [q for q in data.get("questions", []) if isinstance(q, str) and q.strip()]
            if len(qs) < 2:
                raise ValueError("Not enough questions")
            lead = data.get("lead", "Bạn quan tâm đến nội dung nào?")
            return {
                "explain": "",
                "summary": "",
                "final": lead,
                "clarify_questions": qs[:5],
                "need_clarify": True,
                "ask_first_title": lead,
                "preformatted": False,
            }
    except Exception as e:
        logger.warning(f"Failed to parse clarifying questions: {e}")
    
    # Fallback
    return {
        "final": "Xin lỗi, tôi không thể xử lý yêu cầu của bạn lúc này. Vui lòng thử lại sau.",
        "preformatted": True,
        "need_clarify": False,
        "suggestions": [],
        "context": "error"
    }


def get_context_for_input_type(input_type: str) -> str:
    """Get appropriate context for input type"""
    context_map = {
        "greeting": "greeting",
        "oos": "oos"
    }
    return context_map.get(input_type, "oos")


def get_context_for_knowledge_case(input_type: str) -> str:
    """Get context for knowledge-based cases"""
    if input_type == "medical_question":
        return "medical_low_score"
    else:
        return "topic_suggestion"


def get_score_threshold() -> float:
    """Get retrieval score threshold for decision making"""
    return 0.15


def generate_clarifying_questions_for_topic(topic: str, role: str) -> Dict[str, Any]:
    """
    When a user mentions a topic but doesn't ask a specific question,
    this function generates a response to clarify their intent.
    """
    # Retrieve related questions from the knowledge base
    related_questions, _ = retrieve_random_by_role(topic, top_k=3)
    
    intro = f'Mình thấy bạn đang quan tâm đến “{topic}”. Bạn muốn hỏi điều nào?'
    
    
    if related_questions:
        # Use retrieved questions as suggestions
        question_list = [q['cau_hoi'] for q in related_questions]
        formatted_questions = "\n".join([f"{i+1}) “{q}”" for i, q in enumerate(question_list)])
        final_answer = f"{intro}\n{formatted_questions}"
        suggestion_questions = question_list
    else:
        # Fallback if no related questions are found
        final_answer = f"{intro}\n(Không tìm thấy câu hỏi gợi ý cụ thể)"
        suggestion_questions = []

    return {
        "final": final_answer,
        "preformatted": True,
        "need_clarify": True,
        "suggestion_questions": suggestion_questions,
        "context": "clarification"
    }
