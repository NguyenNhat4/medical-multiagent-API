# Core framework import
from core.pocketflow import AsyncParallelBatchNode

# Standard library imports
import logging

# Local imports
from utils.knowledge_base.memory_retrieval import save_user_memory

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


class AddMemory(AsyncParallelBatchNode):
    """
    AddMemory - Worker node that executes INSERT operations.
    Creates new memory entries based on decisions from MemoryManager.
    """

    async def prep_async(self, shared):
        user_id = shared.get("user_id")
        memory_operations = shared.get("memory_operations", {})
        insert_operations = memory_operations.get("insert", [])

        logger.info(f"➕ [AddMemory] PREP - User ID: {user_id}, {len(insert_operations)} insert operation(s)")

        # If no insert operations, return empty list to skip execution
        if not insert_operations:
            logger.info(f"➕ [AddMemory] PREP - No insert operations, skipping")
            return []

        logger.info(f"➕ [AddMemory] PREP - Memory operations from shared: {memory_operations}")
        logger.info(f"➕ [AddMemory] PREP - Insert operations: {insert_operations}")

        # Return list of items for batch processing (each item is a dict with operation data)
        batch_items = [{"user_id": user_id, "content": op.get("content"), "index": i}
                       for i, op in enumerate(insert_operations, 1)]

        logger.info(f"➕ [AddMemory] PREP - Returning {len(batch_items)} batch items")
        return batch_items

    async def exec_async(self, item):
        """Execute a single INSERT operation (called in parallel for each item)"""
        user_id = item["user_id"]
        content = item["content"]
        index = item["index"]

        if not user_id:
            logger.warning(f"➕ [AddMemory] EXEC [{index}] - Missing user_id")
            return {"index": index, "success": False, "reason": "Missing user_id"}

        if not content or not content.strip():
            logger.warning(f"➕ [AddMemory] EXEC [{index}] - Empty content, skipping")
            return {"index": index, "success": False, "reason": "Empty content"}

        # Run synchronous save_user_memory in executor to avoid blocking
        import asyncio
        loop = asyncio.get_event_loop()
        success = await loop.run_in_executor(None, save_user_memory, user_id, content)

        if success:
            logger.info(f"➕ [AddMemory] EXEC [{index}] - INSERT successful - '{content[:50]}...'")
            return {"index": index, "success": True, "content": content[:100]}
        else:
            logger.error(f"➕ [AddMemory] EXEC [{index}] - INSERT failed - '{content[:50]}...'")
            return {"index": index, "success": False, "reason": "Save operation failed"}

    async def exec_fallback_async(self, item, exc):
        """Fallback when INSERT operation fails after max retries"""
        index = item.get("index", 0)
        content = item.get("content", "")
        logger.error(f"➕ [AddMemory] FALLBACK [{index}] - Failed after {self.max_retries} retries: {exc}")
        return {"index": index, "success": False, "reason": f"Failed after {self.max_retries} retries", "content": content[:50]}

    async def post_async(self, shared, prep_res, exec_res):
        # exec_res is a list of results from all parallel executions
        if not exec_res:
            logger.info(f"➕ [AddMemory] POST - No operations executed")
            shared["add_memory_result"] = {"success": True, "inserted": 0, "total": 0, "results": []}
            return "default"

        success_count = sum(1 for r in exec_res if r.get("success"))
        total = len(exec_res)

        result = {
            "success": success_count > 0,
            "inserted": success_count,
            "total": total,
            "results": exec_res
        }

        # Store results in shared state
        shared["add_memory_result"] = result

        logger.info(f"➕ [AddMemory] POST - Completed: {success_count}/{total} successful (parallel execution)")

        return "default"
