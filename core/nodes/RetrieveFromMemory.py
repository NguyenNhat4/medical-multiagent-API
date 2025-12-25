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
        query = shared.get("input", "") or shared.get("query", "")

        logger.info(f"🧠 [RetrieveFromMemory] PREP - User ID: {user_id}, Query: {query[:50] if query else 'None'}...")

        return {
            "user_id": user_id,
            "query": query
        }

    def exec(self, inputs):
        user_id = inputs["user_id"]
        query = inputs["query"]

        if not user_id:
            logger.warning("🧠 [RetrieveFromMemory] EXEC - Missing user_id, cannot retrieve memories")
            return []

        if not query:
            logger.warning("🧠 [RetrieveFromMemory] EXEC - Missing query, cannot retrieve memories")
            return []

        memories = retrieve_user_memory(user_id=user_id, current_query=query, top_k=10)

        logger.info(f"🧠 [RetrieveFromMemory] EXEC - Retrieved {len(memories)} memories")
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
