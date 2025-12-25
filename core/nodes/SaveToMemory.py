# Core framework import
from pocketflow import Node

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


class SaveToMemory(Node):
    def prep(self, shared):
        user_id = shared.get("user_id")
        # Use 'retrieval_query' if available (often better formatted), else 'query', else 'input'
        query = shared.get("retrieval_query") or shared.get("query") or shared.get("input", "")

        logger.info(f"💾 [SaveToMemory] PREP - User ID: {user_id}, Query: {query[:50] if query else 'None'}...")

        return {
            "user_id": user_id,
            "query": query
        }

    def exec(self, inputs):
        user_id = inputs["user_id"]
        query = inputs["query"]

        if not user_id:
            logger.warning("💾 [SaveToMemory] EXEC - Missing user_id, cannot save memory")
            return {"success": False}

        if not query:
            logger.warning("💾 [SaveToMemory] EXEC - Missing query, cannot save memory")
            return {"success": False}

        success = save_user_memory(user_id=user_id, query=query)

        if success:
            logger.info(f"💾 [SaveToMemory] EXEC - Successfully saved memory for user {user_id}")
        else:
            logger.error(f"💾 [SaveToMemory] EXEC - Failed to save memory for user {user_id}")

        return {"success": success}

    def post(self, shared, prep_res, exec_res):
        # Nothing specific to update in shared state for now, just logging
        logger.info(f"💾 [SaveToMemory] POST - Memory save status: {exec_res['success']}")
        return "default"
