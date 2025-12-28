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


class UpdateMemory(AsyncParallelBatchNode):
    """
    UpdateMemory - Worker node that executes UPDATE operations in parallel.
    Updates existing memory entries based on decisions from MemoryManager.
    Uses AsyncParallelBatchNode for concurrent processing.
    """

    async def prep_async(self, shared):
        user_id = shared.get("user_id")
        memory_operations = shared.get("memory_operations", {})
        update_operations = memory_operations.get("update", [])

        logger.info(f"🔄 [UpdateMemory] PREP - User ID: {user_id}, {len(update_operations)} update operation(s)")

        # If no update operations, return empty list to skip execution
        if not update_operations:
            logger.info(f"🔄 [UpdateMemory] PREP - No update operations, skipping")
            return []

        # Return list of items for batch processing
        batch_items = [
            {
                "user_id": user_id,
                "index": i,
                "memory_id": op.get("memory_id"),
                "content": op.get("content")
            }
            for i, op in enumerate(update_operations, 1)
        ]

        logger.info(f"🔄 [UpdateMemory] PREP - Returning {len(batch_items)} batch items")
        return batch_items

    async def exec_async(self, item):
        """Execute a single UPDATE operation"""
        user_id = item["user_id"]
        index = item["index"]
        memory_id = item["memory_id"]
        content = item["content"]

        if not user_id:
            logger.warning(f"🔄 [UpdateMemory] EXEC [{index}] - Missing user_id")
            return {
                "index": index,
                "memory_id": memory_id,
                "success": False,
                "reason": "Missing user_id"
            }

        if not memory_id:
            logger.warning(f"🔄 [UpdateMemory] EXEC [{index}] - Missing memory_id, skipping")
            return {
                "index": index,
                "success": False,
                "reason": "Missing memory_id"
            }

        if not content or not content.strip():
            logger.warning(f"🔄 [UpdateMemory] EXEC [{index}] - Empty content, skipping")
            return {
                "index": index,
                "memory_id": memory_id,
                "success": False,
                "reason": "Empty content"
            }

        # Run synchronous save_user_memory in executor to avoid blocking
        import asyncio
        loop = asyncio.get_event_loop()
        success = await loop.run_in_executor(
            None,
            lambda: save_user_memory(user_id=user_id, query=content, point_id=memory_id)
        )

        if success:
            logger.info(f"🔄 [UpdateMemory] EXEC [{index}] - UPDATE [{memory_id}] successful - '{content[:50]}...'")
            return {
                "index": index,
                "memory_id": memory_id,
                "success": True,
                "content": content[:100]
            }
        else:
            logger.error(f"🔄 [UpdateMemory] EXEC [{index}] - UPDATE [{memory_id}] failed - '{content[:50]}...'")
            return {
                "index": index,
                "memory_id": memory_id,
                "success": False,
                "reason": "Update operation failed"
            }

    async def exec_fallback_async(self, item, exc):
        """Fallback when UPDATE operation fails after max retries"""
        index = item.get("index", 0)
        memory_id = item.get("memory_id", "")
        content = item.get("content", "")
        logger.error(f"🔄 [UpdateMemory] FALLBACK [{index}] - Failed after max retries: {exc}")
        return {
            "index": index,
            "memory_id": memory_id,
            "success": False,
            "reason": f"Failed after max retries",
            "content": content[:50]
        }

    async def post_async(self, shared, prep_res, exec_res):
        """Aggregate results from parallel execution"""
        if not exec_res:
            logger.info(f"🔄 [UpdateMemory] POST - No operations executed")
            shared["update_memory_result"] = {"success": True, "updated": 0, "total": 0, "results": []}
            return "default"

        results = exec_res or []
        success_count = sum(1 for r in results if r.get("success", False))
        total = len(results)

        logger.info(f"🔄 [UpdateMemory] POST - Completed: {success_count}/{total} successful (parallel)")

        # Store results in shared state
        result_data = {
            "success": success_count > 0,
            "updated": success_count,
            "total": total,
            "results": results
        }
        shared["update_memory_result"] = result_data

        if success_count > 0:
            logger.info(f"🔄 [UpdateMemory] POST - Successfully updated {success_count}/{total} memories")
        else:
            logger.info(f"🔄 [UpdateMemory] POST - No memories updated")

        return "default"
