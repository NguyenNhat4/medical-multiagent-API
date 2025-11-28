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



class IngestQuery(Node):
    def prep(self, shared):
        logger.info("🔍 [IngestQuery] PREP - Đọc role, input và conversation history từ shared")
        role = shared.get("role", "")
        user_input = shared.get("input", "")
        conversation_history = shared.get("conversation_history", [])
        logger.info(f"🔍 [IngestQuery] PREP - Role: {role}, Users Input: {user_input}")
        return {
            "role": role,
            "user_input": user_input,
            "conversation_history": conversation_history
        }

    def exec(self, inputs):
        logger.info("🔍 [IngestQuery] EXEC - Xử lý role, query và format conversation history")
        role = inputs["role"]
        user_input = inputs["user_input"]
        conversation_history = inputs["conversation_history"]

        # Format conversation history (lấy 6 tin nhắn gần nhất - 3 cặp, bot responses truncated)
        formatted_history = self._format_conversation_history(conversation_history)
        
        # Format full conversation history (toàn bộ, không truncate)
        full_history = self._format_full_conversation_history(conversation_history)

        result = {
            "role": role,
            "query": user_input.strip(),
            "formatted_conversation_history": formatted_history,
            "full_conversation_history": full_history
        }
        logger.info(f"🔍 [IngestQuery] EXEC - Processed: role={role}, query length={len(user_input)}, history items={len(conversation_history)}")
        return result

    def post(self, shared, prep_res, exec_res):
        logger.info("🔍 [IngestQuery] POST - Lưu role, query và conversation history vào shared")
        shared["role"] = exec_res["role"]
        shared["query"] = exec_res["query"]
        shared["formatted_conversation_history"] = exec_res["formatted_conversation_history"]
        shared["full_conversation_history"] = exec_res["full_conversation_history"]
        logger.info(f"🔍 [IngestQuery] POST - Saved role: {exec_res['role']}, query: {exec_res['query'][:50]}...")
        return "default"

    def _format_conversation_history(self, conversation_history):
        """Format conversation history từ list of dicts thành readable text.
        Lấy 6 tin nhắn gần nhất và truncate bot responses thành 20 ký tự đầu + "...".

        Args:
            conversation_history: List of message dicts with 'role' and 'content' keys

        Returns:
            str: Formatted conversation history string với bot responses truncated
        """
        if not conversation_history:
            return ""

        # Lấy 6 tin nhắn gần nhất (3 cặp)
        recent_messages = conversation_history[-6:]
        history_lines = []

        for msg in recent_messages:
            try:
                role = msg.get("role", "")
                content = msg.get("content", "")

                # Format theo role, truncate bot responses
                if role == "user":
                    history_lines.append(f"- Người dùng: {content}")
                elif role == "bot":
                    # Truncate bot response: 20 ký tự đầu + "..."
                    truncated_content = content[:20] + "..." if len(content) > 20 else content
                    history_lines.append(f"- Bot: {truncated_content}")
                else:
                    history_lines.append(f"- {role}: {content}")
            except Exception as e:
                logger.warning(f"🔍 [IngestQuery] Error formatting message: {e}")
                continue

        return "\n".join(history_lines)

    def _format_full_conversation_history(self, conversation_history):
        """Format toàn bộ conversation history từ list of dicts thành readable text.
        Không truncate, giữ nguyên toàn bộ nội dung.

        Args:
            conversation_history: List of message dicts with 'role' and 'content' keys

        Returns:
            str: Formatted full conversation history string
        """
        if not conversation_history:
            return ""

        history_lines = []

        for msg in conversation_history:
            try:
                role = msg.get("role", "")
                content = msg.get("content", "")

                # Format theo role, không truncate
                if role == "user":
                    history_lines.append(f"- Người dùng: {content}")
                elif role == "bot":
                    history_lines.append(f"- Bot: {content}")
                else:
                    history_lines.append(f"- {role}: {content}")
            except Exception as e:
                logger.warning(f"🔍 [IngestQuery] Error formatting message in full history: {e}")
                continue

        return "\n".join(history_lines)

