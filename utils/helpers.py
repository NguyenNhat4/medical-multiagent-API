"""
Helper functions for medical agent nodes
"""

from typing import Dict, List, Tuple, Any, Optional
import logging
import re
import yaml
from unidecode import unidecode
from utils.llm import call_llm
from utils.knowledge_base import retrieve, retrieve_random_by_role
from utils.parsing.response_parser import parse_yaml_response, validate_yaml_structure
from utils.role_enum import RoleEnum

# Configure logging with Vietnam timezone
from utils.timezone_utils import setup_vietnam_logging
from config.logging_config import logging_config

if logging_config.USE_VIETNAM_TIMEZONE:
    logger = setup_vietnam_logging(__name__, 
                                 level=getattr(logging, logging_config.LOG_LEVEL.upper()),
                                 format_str=logging_config.LOG_FORMAT)
else:
    logger = logging.getLogger(__name__)
    logger.setLevel(getattr(logging, logging_config.LOG_LEVEL.upper()))



def format_kb_qa_list(hits: List[Dict[str, Any]], max_items: int = 10, include_explanation: bool = True) -> str:
    """Format multiple KB hits as a readable Q&A list for prompting.

    Each entry is rendered as:
    Q: <question>
    A: <answer>
    [Optional] Giải thích: <explanation>

    Entries are separated by a blank line. Only items with non-empty answers are included.

    Args:
        hits: List of KB retrieval results
        max_items: Maximum number of items to format
        include_explanation: Whether to include GIẢI THÍCH field if available
    """
    if not hits:
        return ""

    lines: List[str] = []
    added = 0
    for item in hits:
        answer = str(item.get("cau_tra_loi", "")).strip()
        question = str(item.get("cau_hoi", "")).strip()
        explanation = str(item.get("giai_thich", "")).strip()

        if not answer:
            continue
        if question:
            lines.append(f"Q: {question}")
        else:
            lines.append("Q: (không có tiêu đề)")
        lines.append(f"A: {answer}")

        # Add explanation if available and requested
        if include_explanation and explanation:
            lines.append(f"Giải thích: {explanation}")

        lines.append("")  # separator
        added += 1
        if added >= max_items:
            break

    return "\n".join(lines).strip()



def aggregate_retrievals(
    queries: List[str],
    role: Optional[str] = None,
    top_k: int = 5
) -> Tuple[List[Dict[str, Any]], float]:
    """
    Aggregate retrieval results from multiple queries with deduplication.

    This function:
    1. Retrieves results for each query
    2. Aggregates all results
    3. Deduplicates by ma_so or normalized question (keeps highest score)
    4. Sorts by score descending
    5. Returns top_k results and best score

    Args:
        queries: List of query strings to retrieve for
        role: User role for role-specific search (optional)
        top_k: Number of top results to return

    Returns:
        Tuple of (deduplicated_top_k_results, best_score)
    """
    if not queries:
        return [], 0.0

    # Helper function to normalize text for deduplication
    def _norm_text(s: str) -> str:
        return " ".join(unidecode((s or "").lower()).split())

    # Helper function to generate deduplication key
    def _key(item: Dict[str, Any]) -> str:
        return item.get('ma_so') or _norm_text(item.get('cau_hoi', ''))

    # Aggregate results from all queries
    aggregated: List[Dict[str, Any]] = []
    best_seen_score = 0.0

    for query in queries:
        if not query or not query.strip():
            continue
        try:
            results, score = retrieve(query, role, top_k=5)
            logger.info(
                f"📚 [aggregate_retrievals] Retrieved for '{query[:60]}...': "
                f"{len(results) if results else 0} results, best score: {score:.4f}"
            )
            if results:
                aggregated.extend(results)
                if score > best_seen_score:
                    best_seen_score = score
        except Exception as e:
            logger.warning(f"📚 [aggregate_retrievals] Retrieval failed for query '{query[:40]}...': {e}")
            continue

    # Deduplicate: keep highest score per unique key
    seen_max: Dict[str, Dict[str, Any]] = {}
    for item in aggregated:
        k = _key(item)
        if not k:
            continue
        cur = seen_max.get(k)
        if cur is None or float(item.get('score', 0.0)) > float(cur.get('score', 0.0)):
            seen_max[k] = item

    # Sort by score descending and take top_k
    unique_results = list(seen_max.values())
    unique_results.sort(key=lambda x: x.get('score', 0.0), reverse=True)
    top_results = unique_results[:top_k]

    # Calculate top score
    top_score = float(top_results[0].get('score', 0.0)) if top_results else 0.0

    logger.info(
        f"📚 [aggregate_retrievals] Aggregated {len(aggregated)} total, "
        f"{len(unique_results)} unique, returning top {len(top_results)}, "
        f"best score: {top_score:.4f}"
    )

    return top_results, top_score


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
