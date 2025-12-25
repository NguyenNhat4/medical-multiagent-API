"""
Utility function for User Memory retrieval and storage in Qdrant.
"""

import logging
import uuid
import time
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient, models
import os
from dotenv import load_dotenv

# Import the existing embedding model loader to reuse models
from utils.knowledge_base.qdrant_retrieval import _get_embedding_models

logger = logging.getLogger(__name__)
load_dotenv(override=False)

MEMORY_COLLECTION_NAME = "user_memory"
QDRANT_URL = os.getenv("QDRANT_URL")

# Configuration for the memory collection (same as knowledge base for consistency)
DENSE_VECTOR_SIZE = 384  # all-MiniLM-L6-v2
LATE_INTERACTION_VECTOR_SIZE = 128  # colbertv2.0


def ensure_memory_collection_exists(
    qdrant_url: str = QDRANT_URL,
    collection_name: str = MEMORY_COLLECTION_NAME
) -> bool:
    """
    Ensure the user memory collection exists with the correct configuration.

    Args:
        qdrant_url: Qdrant server URL
        collection_name: Name of the collection

    Returns:
        True if collection exists or was created, False on error
    """
    try:
        client = QdrantClient(url=qdrant_url)

        # Check if collection exists
        collections = client.get_collections().collections
        exists = any(col.name == collection_name for col in collections)

        if exists:
            # logger.info(f"[Memory] Collection '{collection_name}' already exists")
            return True

        logger.info(f"[Memory] Creating collection '{collection_name}' with hybrid search config")

        client.create_collection(
            collection_name=collection_name,
            vectors_config={
                "all-MiniLM-L6-v2": models.VectorParams(
                    size=DENSE_VECTOR_SIZE,
                    distance=models.Distance.COSINE,
                ),
                "colbertv2.0": models.VectorParams(
                    size=LATE_INTERACTION_VECTOR_SIZE,
                    distance=models.Distance.COSINE,
                    multivector_config=models.MultiVectorConfig(
                        comparator=models.MultiVectorComparator.MAX_SIM,
                    ),
                    hnsw_config=models.HnswConfigDiff(m=0)  # Disable HNSW for reranking
                ),
            },
            sparse_vectors_config={
                "bm25": models.SparseVectorParams(
                    modifier=models.Modifier.IDF
                )
            }
        )

        logger.info(f"[Memory] Collection '{collection_name}' created successfully")
        return True

    except Exception as e:
        logger.error(f"[Memory] Error creating collection '{collection_name}': {e}")
        return False


def save_user_memory(
    user_id: str,
    query: str,
    qdrant_url: str = QDRANT_URL,
    collection_name: str = MEMORY_COLLECTION_NAME
) -> bool:
    """
    Save a user query to the memory collection.

    Args:
        user_id: The user's ID
        query: The user's query text
        qdrant_url: Qdrant server URL
        collection_name: Memory collection name

    Returns:
        True if saved successfully, False otherwise
    """
    if not query or not query.strip():
        logger.warning("[Memory] Empty query, skipping save")
        return False

    try:
        # Ensure collection exists
        ensure_memory_collection_exists(qdrant_url, collection_name)

        # Get embedding models
        dense_model, sparse_model, late_interaction_model = _get_embedding_models()

        # Embed query
        dense_vectors = next(dense_model.embed([query]))
        sparse_vectors = next(sparse_model.embed([query]))
        late_vectors = next(late_interaction_model.embed([query]))

        # Create Point
        point_id = str(uuid.uuid4())
        timestamp = time.time()

        point = models.PointStruct(
            id=point_id,
            vector={
                "all-MiniLM-L6-v2": dense_vectors,
                "bm25": sparse_vectors.as_object(),
                "colbertv2.0": late_vectors,
            },
            payload={
                "user_id": user_id,
                "query": query,
                "timestamp": timestamp
            }
        )

        # Upsert
        client = QdrantClient(url=qdrant_url)
        client.upsert(
            collection_name=collection_name,
            points=[point]
        )

        logger.info(f"[Memory] Saved memory for user {user_id}: '{query[:50]}...'")
        return True

    except Exception as e:
        logger.error(f"[Memory] Error saving memory: {e}")
        return False


def retrieve_user_memory(
    user_id: str,
    current_query: str,
    top_k: int = 10,
    qdrant_url: str = QDRANT_URL,
    collection_name: str = MEMORY_COLLECTION_NAME
) -> List[Dict[str, Any]]:
    """
    Retrieve relevant past queries for a user.

    Args:
        user_id: The user's ID
        current_query: The current query to find similar memories for
        top_k: Number of memories to retrieve
        qdrant_url: Qdrant server URL
        collection_name: Memory collection name

    Returns:
        List of memory dictionaries containing {id, query, timestamp, score}
    """
    try:
        # Ensure collection exists (just in case it's the first time)
        ensure_memory_collection_exists(qdrant_url, collection_name)

        # Get embedding models
        dense_model, sparse_model, late_interaction_model = _get_embedding_models()

        # Embed current query
        dense_vectors = next(dense_model.query_embed(current_query))
        sparse_vectors = next(sparse_model.query_embed(current_query))
        late_vectors = next(late_interaction_model.query_embed(current_query))

        client = QdrantClient(url=qdrant_url)

        # Build prefetch for hybrid search
        prefetch = [
            models.Prefetch(
                query=dense_vectors,
                using="all-MiniLM-L6-v2",
                limit=top_k + 20, # Fetch a bit more for reranking
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="user_id",
                            match=models.MatchValue(value=user_id)
                        )
                    ]
                )
            ),
            models.Prefetch(
                query=models.SparseVector(**sparse_vectors.as_object()),
                using="bm25",
                limit=top_k + 20,
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="user_id",
                            match=models.MatchValue(value=user_id)
                        )
                    ]
                )
            ),
        ]

        # Execute hybrid search with Late Interaction (ColBERT) reranking
        results = client.query_points(
            collection_name=collection_name,
            prefetch=prefetch,
            query=late_vectors,
            using="colbertv2.0",
            with_payload=True,
            limit=top_k,
            # Also apply filter at the root level just to be safe
            query_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="user_id",
                        match=models.MatchValue(value=user_id)
                    )
                ]
            )
        )

        memories = []
        for point in results.points:
            memories.append({
                "id": point.id,
                "query": point.payload.get("query", ""),
                "timestamp": point.payload.get("timestamp", 0),
                "score": point.score
            })

        logger.info(f"[Memory] Retrieved {len(memories)} memories for user {user_id}")
        return memories

    except Exception as e:
        logger.error(f"[Memory] Error retrieving memories: {e}")
        return []
