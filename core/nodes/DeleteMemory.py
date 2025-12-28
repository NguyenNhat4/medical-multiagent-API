# Core framework import
from core.pocketflow import AsyncNode

# Standard library imports
import logging

# Local imports
from utils.knowledge_base.memory_retrieval import delete_user_memory

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


class DeleteMemory(AsyncNode):
    """
    DeleteMemory - Worker node that executes DELETE operations.
    Removes memory entries based on decisions from MemoryManager.
    Uses AsyncNode with batch delete for efficiency.
    """

    async def prep_async(self, shared):
        user_id = shared.get("user_id")
        memory_operations = shared.get("memory_operations", {})
        delete_operations = memory_operations.get("delete", [])

        logger.info(f"🗑️ [DeleteMemory] PREP - User ID: {user_id}, {len(delete_operations)} delete operation(s)")

        # If no delete operations, return None to skip execution
        if not delete_operations:
            logger.info(f"🗑️ [DeleteMemory] PREP - No delete operations, skipping")
            return None

        return {
            "user_id": user_id,
            "delete_operations": delete_operations
        }

    async def exec_async(self, inputs):
        """Execute DELETE operations in a single batch call"""
        # Handle case when prep_async returns None (no operations)
        if inputs is None:
            logger.info("🗑️ [DeleteMemory] EXEC - No inputs (prep returned None), skipping")
            return {"success": True, "deleted": 0, "results": []}

        user_id = inputs["user_id"]
        delete_operations = inputs["delete_operations"]

        if not user_id:
            logger.warning("🗑️ [DeleteMemory] EXEC - Missing user_id, cannot delete memories")
            return {"success": False, "deleted": 0, "results": []}

        if not delete_operations:
            logger.info("🗑️ [DeleteMemory] EXEC - No delete operations to perform")
            return {"success": True, "deleted": 0, "results": []}

        # Collect all memory IDs to delete
        memory_ids_to_delete = []
        results = []

        for i, op in enumerate(delete_operations, 1):
            memory_id = op.get("memory_id")

            if not memory_id:
                logger.warning(f"🗑️ [DeleteMemory] EXEC - Operation {i}: Missing memory_id, skipping")
                results.append({
                    "index": i,
                    "success": False,
                    "reason": "Missing memory_id"
                })
                continue

            memory_ids_to_delete.append(memory_id)

        # Execute batch delete if we have IDs
        if memory_ids_to_delete:
            # Run synchronous delete_user_memory in executor to avoid blocking
            import asyncio
            loop = asyncio.get_event_loop()
            success = await loop.run_in_executor(
                None,
                lambda: delete_user_memory(point_ids=memory_ids_to_delete)
            )

            if success:
                logger.info(f"🗑️ [DeleteMemory] EXEC - Successfully deleted {len(memory_ids_to_delete)} memories (batch)")
                # Mark all as successful
                for i, memory_id in enumerate(memory_ids_to_delete, 1):
                    results.append({
                        "index": i,
                        "memory_id": memory_id,
                        "success": True
                    })

                return {
                    "success": True,
                    "deleted": len(memory_ids_to_delete),
                    "total": len(delete_operations),
                    "results": results
                }
            else:
                logger.error(f"🗑️ [DeleteMemory] EXEC - Failed to delete {len(memory_ids_to_delete)} memories")
                # Mark all as failed
                for i, memory_id in enumerate(memory_ids_to_delete, 1):
                    results.append({
                        "index": i,
                        "memory_id": memory_id,
                        "success": False,
                        "reason": "Delete operation failed"
                    })

                return {
                    "success": False,
                    "deleted": 0,
                    "total": len(delete_operations),
                    "results": results
                }

        # No valid IDs to delete
        logger.info("🗑️ [DeleteMemory] EXEC - No valid memory IDs to delete")
        return {
            "success": True,
            "deleted": 0,
            "total": len(delete_operations),
            "results": results
        }

    async def exec_fallback_async(self, inputs, exc):
        """Fallback when DELETE operation fails after max retries"""
        if inputs is None:
            return {"success": True, "deleted": 0, "results": []}

        delete_operations = inputs.get("delete_operations", [])
        logger.error(f"🗑️ [DeleteMemory] FALLBACK - Failed after max retries: {exc}")
        return {
            "success": False,
            "deleted": 0,
            "total": len(delete_operations),
            "results": [],
            "reason": f"Failed after max retries"
        }

    async def post_async(self, shared, prep_res, exec_res):
        """Store results in shared state"""
        deleted = exec_res.get("deleted", 0)
        total = exec_res.get("total", 0)

        # Store results in shared state
        shared["delete_memory_result"] = exec_res

        if deleted > 0:
            logger.info(f"🗑️ [DeleteMemory] POST - Successfully deleted {deleted}/{total} memories")
        else:
            logger.info(f"🗑️ [DeleteMemory] POST - No memories deleted")

        return "default"
