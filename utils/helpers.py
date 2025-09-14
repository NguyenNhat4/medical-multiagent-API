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
from .role_ENUM import RoleEnum

logger = logging.getLogger(__name__)



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
    """Fallback topics based on role when cannot retrieve from KB.

    Supports RoleEnum values; falls back to Vietnamese phrase detection.
    """
    topics_by_role: Dict[RoleEnum, List[str]] = {
        RoleEnum.DOCTOR_DENTAL: [
            "Quản lý bệnh nhân đái tháo đường trong nha khoa",
            "Điều trị viêm nha chu ở bệnh nhân ĐTĐ",
            "Phối hợp với bác sĩ nội tiết trong điều trị",
            "Biến chứng nha khoa do đái tháo đường",
            "Kháng sinh trong điều trị nha khoa bệnh nhân ĐTĐ",
        ],
        RoleEnum.DOCTOR_ENDOCRINE: [
            "Mối liên hệ giữa kiểm soát đường huyết và sức khỏe nha chu",
            "Khi nào giới thiệu bệnh nhân đến nha khoa",
            "Thuốc đái tháo đường ảnh hưởng đến răng miệng",
            "Biến chứng răng miệng ở bệnh nhân ĐTĐ type 1 và type 2",
            "Tư vấn chăm sóc răng miệng cho bệnh nhân ĐTĐ",
        ],
        RoleEnum.PATIENT_DIABETES: [
            "Cách chăm sóc răng miệng khi bị đái tháo đường",
            "Triệu chứng cảnh báo ở răng miệng cần chú ý",
            "Khi nào cần đi khám nha khoa",
            "Chế độ ăn tốt cho răng miệng và đường huyết",
            "Cách đánh răng đúng cách cho người ĐTĐ",
        ],
        RoleEnum.PATIENT_DENTAL: [
            "Cách chăm sóc răng miệng hàng ngày",
            "Dấu hiệu cần khám nha khoa ngay",
            "Phòng ngừa sâu răng và viêm nướu",
            "Chế độ ăn uống tốt cho răng miệng",
            "Tần suất khám nha khoa định kỳ",
        ],
    }

    role_lower = (role or "").lower()
    # Try RoleEnum value match
    for enum_role in RoleEnum:
        if role_lower == enum_role.value:
            return topics_by_role[enum_role]

    # Vietnamese phrase detection for legacy strings
    if "bác sĩ nha khoa" in role_lower:
        return topics_by_role[RoleEnum.DOCTOR_DENTAL]
    if "bác sĩ nội tiết" in role_lower:
        return topics_by_role[RoleEnum.DOCTOR_ENDOCRINE]
    if "đái tháo đường" in role_lower:
        return topics_by_role[RoleEnum.PATIENT_DIABETES]

    # Default
    return topics_by_role[RoleEnum.PATIENT_DENTAL]

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


def format_kb_qa_list(hits: List[Dict[str, Any]], max_items: int = 5) -> str:
    """Format multiple KB hits as a readable Q&A list for prompting.

    Each entry is rendered as:
    Q: <question>
    A: <answer>
    (score: <0..1>)  # optional if score exists

    Entries are separated by a blank line. Only items with non-empty answers are included.
    """
    if not hits:
        return ""

    lines: List[str] = []
    added = 0
    for item in hits:
        answer = str(item.get("cau_tra_loi", "")).strip()
        question = str(item.get("cau_hoi", "")).strip()
        if not answer:
            continue
        if question:
            lines.append(f"Q: {question}")
        else:
            lines.append("Q: (không có tiêu đề)")
        lines.append(f"A: {answer}")
        score = item.get("score")
        if isinstance(score, (int, float)):
            try:
                lines.append(f"(score: {float(score):.3f})")
            except Exception:
                pass
        lines.append("")  # separator
        added += 1
        if added >= max_items:
            break

    return "\n".join(lines).strip()

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
    return 0.1


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

def serialize_conversation_history(messages):
    """
    Serialize SQLAlchemy message objects to plain Python dicts
    
    Args:
        messages: SQLAlchemy relationship collection of ChatMessage objects
        
    Returns:
        list: List of serialized message dictionaries
    """
    conversation_history = []
    for msg in messages:
        conversation_history.append({
            "role": msg.role,
            "content": msg.content,
            "api_role": msg.api_role,
            "input_type": msg.input_type
        })
    return conversation_history