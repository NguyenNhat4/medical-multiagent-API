# Core framework import
from pocketflow import Node

# Standard library imports
import logging

# Local imports
from utils.knowledge_base.memory_retrieval import retrieve_user_memory

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


class RetrieveFromMemory(Node):
    def prep(self, shared):
        user_id = shared.get("user_id")
        # Use 'input' as the query to find relevant past memories
        query = shared.get("original_query", "") or shared.get("query", "") or shared.get("input", "")
        context_summary = shared.get("context_summary", "")
        role = shared.get("role", "")

        logger.info(f"🧠 [RetrieveFromMemory] PREP - User ID: {user_id}, Query: {query[:50] if query else 'None'}...")

        return {
            "user_id": user_id,
            "query": query,
            "context_summary": context_summary,
            "role": role
        }

    def exec(self, inputs):
        from utils.llm import call_llm
        from utils.parsing import parse_yaml_with_schema
        from utils.llm.call_llm import APIOverloadException
        from config.timeout_config import timeout_config
        from utils.role_enum import RoleEnum, ROLE_DISPLAY_NAME

        user_id = inputs["user_id"]
        query = inputs["query"]
        context_summary = inputs["context_summary"]
        role = inputs["role"]

        if not user_id:
            logger.warning("🧠 [RetrieveFromMemory] EXEC - Missing user_id, cannot retrieve memories")
            return []

        if not query:
            logger.warning("🧠 [RetrieveFromMemory] EXEC - Missing query, cannot retrieve memories")
            return []

        # Generate optimized memory retrieval query using LLM
        vietnameseRole = ROLE_DISPLAY_NAME.get(RoleEnum(role), "Người dùng") if role else "Người dùng"

        prompt = f"""
BỐI CẢNH:
- Tóm tắt hội thoại trước: {context_summary}
- Câu hỏi hiện tại của người dùng ({vietnameseRole}): "{query}"

NHIỆM VỤ:
Tạo một câu truy vấn tối ưu để tìm kiếm các thông tin cá nhân/lịch sử tương tác đã lưu trữ của người dùng này.
Câu truy vấn cần tập trung vào:
- Thông tin cá nhân liên quan (tên, tuổi, tình trạng sức khỏe, tiền sử bệnh...)
- Các cuộc hội thoại trước đây có liên quan
- Ngữ cảnh và mối quan tâm của người dùng

vui lòng trả lời dưới định dạng YAML như sau:
```yaml
memory_query: |
    Câu truy vấn tối ưu để tìm kiếm memory
reason: |
    Lý do ngắn gọn về cách tạo query
```"""

        logger.info(f"🧠 [RetrieveFromMemory] EXEC - Generating memory query with LLM")

        resp = call_llm(prompt, fast_mode=True, max_retry_time=timeout_config.LLM_RETRY_TIMEOUT)

        result = parse_yaml_with_schema(
            resp,
            required_fields=["memory_query"],
            optional_fields=["reason"],
            field_types={"memory_query": str, "reason": str}
        )

        assert isinstance(result, dict), f"Failed to parse LLM response, got: {resp}"

        memory_query = result.get("memory_query", query)
        logger.info(f"🧠 [RetrieveFromMemory] EXEC - Generated memory query: '{memory_query[:100]}...'")

        # Retrieve memories using the optimized query
        memories = retrieve_user_memory(user_id=user_id, current_query=memory_query, top_k=10)

        logger.info(f"🧠 [RetrieveFromMemory] EXEC - Retrieved {len(memories)} memories")
        return memories

    def exec_fallback(self, inputs, exc):
        """Fallback when memory query generation fails - use original query"""
        user_id = inputs.get("user_id")
        query = inputs.get("query", "")
        logger.error(f"🧠 [RetrieveFromMemory] FALLBACK - Failed after max retries: {exc}, using original query")

        if not user_id or not query:
            return []

        # Retrieve memories using original query as fallback
        memories = retrieve_user_memory(user_id=user_id, current_query=query, top_k=10)
        logger.info(f"🧠 [RetrieveFromMemory] FALLBACK - Retrieved {len(memories)} memories with original query")
        return memories

    def post(self, shared, prep_res, exec_res):
        shared["relevant_memories"] = exec_res

        # Log a snippet of the first memory if available
        if exec_res:
            first_memory = exec_res[0].get('query', '')
            logger.info(f"🧠 [RetrieveFromMemory] POST - Top memory: {first_memory[:50]}...")
        else:
            logger.info("🧠 [RetrieveFromMemory] POST - No memories found")

        return "default"
