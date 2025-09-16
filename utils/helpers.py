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


def log_llm_timing(node_name: str, start_time: float, end_time: float, prompt_length: int = 0):
    """Log LLM call timing to logs/llm.log file"""
    import os
    
    elapsed_time = end_time - start_time
    
    # Ensure logs directory exists
    os.makedirs('logs', exist_ok=True)
    
    # Create a separate logger for LLM timing
    llm_logger = logging.getLogger('llm_timing')
    llm_logger.setLevel(logging.INFO)
    
    # Create file handler if it doesn't exist
    if not llm_logger.handlers:
        file_handler = logging.FileHandler('logs/llm.log', encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        file_handler.setFormatter(formatter)
        
        llm_logger.addHandler(file_handler)
        llm_logger.propagate = True  # Don't propagate to root logger
    
    llm_logger.info(f"{node_name} - Duration: {elapsed_time:.4f}s, Prompt: {prompt_length} chars")


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

def _format_conversation_history(conversation_history):
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
