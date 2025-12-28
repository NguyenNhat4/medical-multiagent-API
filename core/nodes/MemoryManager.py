# Core framework import
from core.pocketflow import AsyncNode

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


class MemoryManager(AsyncNode):
    """
    Memory Manager - Orchestrator node that analyzes conversation and existing memories
    to decide which operations (INSERT/UPDATE/DELETE) should be performed.

    This node uses LLM to intelligently determine operations that will be executed
    by specialized worker nodes (AddMemory, UpdateMemory, DeleteMemory).
    """

    async def prep_async(self, shared):
        user_id = shared.get("user_id")
        query = shared.get("original_query") or shared.get("input", "")
        context_summary = shared.get("context_summary", "")
        role = shared.get("role", "")
        # Try multiple fields for AI response
        ai_response = (shared.get("final_answer", "") or
                      shared.get("response", "") or
                      shared.get("explain", "") or
                      (shared.get("answer_obj", {}).get("explain", "") if isinstance(shared.get("answer_obj"), dict) else ""))
        relevant_memories = shared.get("relevant_memories", [])

        logger.info(f"🎯 [MemoryManager] PREP - User ID: {user_id}, Query: {query[:50] if query else 'None'}...")
        logger.info(f"🎯 [MemoryManager] PREP - Analyzing {len(relevant_memories)} existing memories")

        return {
            "user_id": user_id,
            "query": query,
            "context_summary": context_summary,
            "role": role,
            "ai_response": ai_response,
            "relevant_memories": relevant_memories
        }

    async def exec_async(self, inputs):
        from utils.llm import call_llm
        from utils.parsing import parse_yaml_with_schema
        from utils.llm.call_llm import APIOverloadException
        from config.timeout_config import timeout_config
        from utils.role_enum import RoleEnum, ROLE_DISPLAY_NAME

        user_id = inputs["user_id"]
        query = inputs["query"]
        context_summary = inputs["context_summary"]
        role = inputs["role"]
        ai_response = inputs["ai_response"]
        relevant_memories = inputs["relevant_memories"]

        if not user_id:
            logger.warning("🎯 [MemoryManager] EXEC - Missing user_id, cannot manage memories")
            return {
                "success": False,
                "operations": {"insert": [], "update": [], "delete": []},
                "reason": "Missing user_id"
            }

        if not query:
            logger.warning("🎯 [MemoryManager] EXEC - Missing query, cannot manage memories")
            return {
                "success": False,
                "operations": {"insert": [], "update": [], "delete": []},
                "reason": "Missing query"
            }

        # Format existing memories for the prompt
        vietnameseRole = ROLE_DISPLAY_NAME.get(RoleEnum(role), "Người dùng") if role else "Người dùng"

        memories_context = ""
        if relevant_memories:
            memories_list = []
            for i, mem in enumerate(relevant_memories[:10], 1):  # Show top 10
                memories_list.append(
                    f"  - ID: {mem.get('id')}\n"
                    f"    Nội dung: {mem.get('query', '')}\n"
                    f"    Score: {mem.get('score', 0):.3f}"
                )
            memories_context = f"\n# CÁC MEMORY ĐÃ TỒN TẠI (Top 10):\n" + "\n".join(memories_list)
        else:
            memories_context = "\n# CÁC MEMORY ĐÃ TỒN TẠI: Không có memory nào."

        prompt = f"""
# NHIỆM VỤ:
Bạn là Memory Manager - hệ thống quản lý bộ nhớ thông minh. Phân tích hội thoại và quyết định các thao tác cần thực hiện.

# BỐI CẢNH HỘI THOẠI:
- Tóm tắt hội thoại trước: {context_summary}
- Người dùng ({vietnameseRole}): "{query}"
- AI trả lời: "{ai_response[:300]}..."
{memories_context}

# CÁC THAO TÁC:
1. **INSERT**: Thêm memory mới - thông tin hoàn toàn mới và quan trọng
2. **UPDATE**: Cập nhật memory cũ - thông tin đã thay đổi/bổ sung
3. **DELETE**: Xóa memory cũ - thông tin sai/lỗi thời/không còn liên quan

# QUY TẮC:
- INSERT: Thông tin cá nhân mới (tên, tuổi, nghề), sức khỏe, sở thích, gia đình chưa có
- UPDATE: Thông tin cũ cần cập nhật (tuổi mới, công việc mới, tình trạng sức khỏe thay đổi)
- DELETE: Thông tin trong memory hoàn toàn sai hoặc người dùng đã sửa/phủ nhận
- SKIP ALL: Chào hỏi xã giao, thông tin tổng quát, hoặc đã đầy đủ trong memory

# YÊU CẦU ĐỊNH DẠNG (QUAN TRỌNG):
- Sử dụng Block Scalar (|) cho văn bản
- Tổ chức operations theo loại: insert_operations, update_operations, delete_operations
- Mỗi operation có: memory_id (nếu UPDATE/DELETE), content (nếu INSERT/UPDATE)

# VÍ DỤ:
```yaml
insert_operations:
  - content: |
      Người dùng có sở thích đọc sách triết học
update_operations:
  - memory_id: "abc-123"
    content: |
      Người dùng An, 30 tuổi (cập nhật từ 29), nghề giáo viên
delete_operations:
  - memory_id: "xyz-456"
reason: |
  Cập nhật tuổi, thêm sở thích mới, xóa thông tin sai
importance: "high"
```

Trả về duy nhất một block code YAML:
"""

        logger.info(f"🎯 [MemoryManager] EXEC - Analyzing operations with LLM")

        resp = call_llm(prompt, fast_mode=True, max_retry_time=timeout_config.LLM_RETRY_TIMEOUT)

        result = parse_yaml_with_schema(
            resp,
            required_fields=["reason"],
            optional_fields=["insert_operations", "update_operations", "delete_operations", "importance"],
            field_types={
                "insert_operations": list,
                "update_operations": list,
                "delete_operations": list,
                "reason": str,
                "importance": str
            }
        )

        assert isinstance(result, dict), f"Failed to parse LLM response, got: {resp}"

        insert_ops = result.get("insert_operations", [])
        update_ops = result.get("update_operations", [])
        delete_ops = result.get("delete_operations", [])
        reason = result.get("reason", "")
        importance = result.get("importance", "medium")

        total_ops = len(insert_ops) + len(update_ops) + len(delete_ops)
        logger.info(f"🎯 [MemoryManager] EXEC - Decided {total_ops} operations: "
                  f"INSERT={len(insert_ops)}, UPDATE={len(update_ops)}, DELETE={len(delete_ops)}")
        logger.info(f"🎯 [MemoryManager] EXEC - Reason: {reason}")

        return {
            "success": True,
            "operations": {
                "insert": insert_ops,
                "update": update_ops,
                "delete": delete_ops
            },
            "reason": reason,
            "importance": importance,
            "user_id": user_id
        }

    async def exec_fallback_async(self, inputs, exc):
        logger.error(f"🎯 [MemoryManager] FALLBACK - Failed after {self.max_retries} retries: {exc}")
        user_id = inputs.get("user_id")
        return {
            "success": True,
            "operations": {"insert": [], "update": [], "delete": []},
            "reason": "Failed to analyze operations, skipping",
            "importance": "low",
            "user_id": user_id
        }

    async def post_async(self, shared, prep_res, exec_res):
        # Store operation decisions in shared state for worker nodes
        shared["memory_operations"] = exec_res.get("operations", {})
        shared["memory_manager_reason"] = exec_res.get("reason", "")
        shared["memory_importance"] = exec_res.get("importance", "medium")

        operations = exec_res.get("operations", {})
        insert_count = len(operations.get("insert", []))
        update_count = len(operations.get("update", []))
        delete_count = len(operations.get("delete", []))

        total = insert_count + update_count + delete_count

        if total == 0:
            logger.info(f"🎯 [MemoryManager] POST - No operations needed, returning 'skip'")
            return "skip"  # No operations, skip to end

        # Return specific routes based on what operations exist
        # This allows conditional routing in the flow
        routes = []
        if insert_count > 0:
            routes.append("insert")
        if update_count > 0:
            routes.append("update")
        if delete_count > 0:
            routes.append("delete")

        logger.info(f"🎯 [MemoryManager] POST - Operations planned: "
                   f"INSERT={insert_count}, UPDATE={update_count}, DELETE={delete_count}")
        logger.info(f"🎯 [MemoryManager] POST - Operations details: {operations}")
        logger.info(f"🎯 [MemoryManager] POST - Returning routes: {routes}")

        # Return first route (or "default" if routing doesn't support multiple returns)
        # For now, return "default" and let worker nodes check if they have work
        return "default"  # Proceed to worker nodes
