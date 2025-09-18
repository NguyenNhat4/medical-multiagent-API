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



def format_kb_qa_list(hits: List[Dict[str, Any]], max_items: int = 10) -> str:
    """Format multiple KB hits as a readable Q&A list for prompting.

    Each entry is rendered as:
    Q: <question>
    A: <answer>

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
        lines.append("")  # separator
        added += 1
        if added >= max_items:
            break

    return "\n".join(lines).strip()



def get_score_threshold() -> float:
    """Get retrieval score threshold for decision making"""
    return 0.1



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

def format_conversation_history(conversation_history):
    """Format conversation history from list of dicts to readable text"""
    if not conversation_history:
        return "Không có cuộc hội thoại trước đó"
    
    formatted_messages = []
    for msg in conversation_history:
        role = msg.get('role', '')
        content = msg.get('content', '')
        
        if role == 'user':
            formatted_messages.append(f"Người dùng: {content}")
        elif role == 'bot':
            formatted_messages.append(f"Bot: {content}")
        else:
            formatted_messages.append(f"{role}: {content}")
    
    return "\n".join(formatted_messages)
