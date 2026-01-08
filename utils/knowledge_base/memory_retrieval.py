"""
Utility function for User Memory retrieval and storage in Qdrant.
"""

import logging
import uuid
import time
from typing import List, Dict, Any, Optional
from qdrant_client import models

from utils.qdrant.operations import QdrantOperations
from utils.qdrant.config import COLLECTION_USER_MEMORY, VECTOR_DENSE, VECTOR_SPARSE, VECTOR_LATE
from utils.qdrant.embeddings import EmbeddingService
from utils.qdrant.filters import FilterBuilder

logger = logging.getLogger(__name__)


def ensure_memory_collection_exists(
    qdrant_url: str = None, # Deprecated
    collection_name: str = COLLECTION_USER_MEMORY
) -> bool:
    """
    Ensure the user memory collection exists with the correct configuration.
    """
    return QdrantOperations.ensure_collection(collection_name)


def save_user_memory(
    user_id: str,
    query: str,
    qdrant_url: str = None,
    collection_name: str = COLLECTION_USER_MEMORY,
    point_id: Optional[str] = None
) -> bool:
    """
    Save a user query to the memory collection.
    """
    if not query or not query.strip():
        logger.warning("[Memory] Empty query, skipping save")
        return False

    try:
        # Ensure collection exists
        ensure_memory_collection_exists(collection_name=collection_name)

        # Embed query
        vectors = EmbeddingService.embed_query(query)

        # Create or update Point
        if point_id is None:
            point_id = str(uuid.uuid4())
            timestamp = time.time()
            action = "Saved new"
        else:
            timestamp = time.time()  # Update timestamp on modification
            action = "Updated"

        point = models.PointStruct(
            id=point_id,
            vector={
                VECTOR_DENSE: vectors['dense'],
                VECTOR_SPARSE: vectors['sparse'].as_object(),
                VECTOR_LATE: vectors['late'],
            },
            payload={
                "user_id": user_id,
                "query": query,
                "timestamp": timestamp
            }
        )

        # Upsert
        success = QdrantOperations.upsert(collection_name, [point])
        if success:
            logger.info(f"[Memory] {action} memory for user {user_id}: '{query[:50]}...'")
        return success

    except Exception as e:
        logger.error(f"[Memory] Error saving memory: {e}")
        return False


def delete_user_memory(
    point_ids: List[str],
    qdrant_url: str = None,
    collection_name: str = COLLECTION_USER_MEMORY
) -> bool:
    """
    Delete user memories by point IDs.
    """
    if not point_ids:
        logger.warning("[Memory] No point IDs provided for deletion")
        return False

    return QdrantOperations.delete(collection_name, point_ids)


def retrieve_user_memory(
    user_id: str,
    current_query: str,
    top_k: int = 10,
    qdrant_url: str = None,
    collection_name: str = COLLECTION_USER_MEMORY
) -> List[Dict[str, Any]]:
    """
    Retrieve relevant past queries for a user.
    """
    try:
        # Ensure collection exists
        ensure_memory_collection_exists(collection_name=collection_name)

        filter_builder = FilterBuilder()
        filter_builder.add_user_id(user_id)

        results, _ = QdrantOperations.search(
            collection_name=collection_name,
            query_text=current_query,
            query_filter=filter_builder.build(),
            top_k=top_k,
            use_late_interaction=True
        )

        memories = []
        for point in results:
            memories.append({
                "id": point["id"],
                "query": point.get("query", ""),
                "timestamp": point.get("timestamp", 0),
                "score": point["score"]
            })

        logger.info(f"[Memory] Retrieved {len(memories)} memories for user {user_id}")
        return memories

    except Exception as e:
        logger.error(f"[Memory] Error retrieving memories: {e}")
        return []
