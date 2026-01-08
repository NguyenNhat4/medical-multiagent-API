"""
Utility function for Qdrant hybrid search retrieval.

This is an external utility function for vector database operations.
According to PocketFlow best practices, this should be independent and easily testable.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple

from utils.qdrant.operations import QdrantOperations
from utils.qdrant.filters import FilterBuilder
from utils.qdrant.config import ALL_COLLECTIONS
from utils.qdrant.embeddings import EmbeddingService

logger = logging.getLogger(__name__)

# Keep for backward compatibility if imported elsewhere
def _get_embedding_models():
    """
    Deprecated: Use EmbeddingService.get_models() instead.
    """
    return EmbeddingService.get_models()


def retrieve_from_qdrant(
    query: str,
    demuc: Optional[str] = None,
    chu_de_con: Optional[str] = None,
    top_k: int = 20,
    collection_name: str = "bnrhm",
    qdrant_url: str = None, # Deprecated but kept for signature
    use_late_interaction: bool = True
) -> List[Dict[str, Any]]:
    """
    Retrieve documents from Qdrant using hybrid search.
    Legacy wrapper around QdrantOperations.
    """
    try:
        logger.info(f"[retrieve_from_qdrant] Query: '{query[:50]}...', Filters: demuc={demuc}, sub={chu_de_con}")

        # Build filter
        filter_builder = FilterBuilder()
        filter_builder.add_demuc(demuc)
        filter_builder.add_subtopic(chu_de_con)
        
        # Execute search
        results, _ = QdrantOperations.search(
            collection_name=collection_name,
            query_text=query,
            query_filter=filter_builder.build(),
            top_k=top_k,
            use_late_interaction=use_late_interaction,
            return_embeddings=False
        )

        # Format results (QdrantOperations already returns a good format, but let's ensure keys match legacy)
        formatted_results = []
        for r in results:
            formatted_results.append({
                "id": r["id"],
                "score": r["score"],
                "collection": r["collection"],
                "DEMUC": r.get("DEMUC", ""),
                "CHUDECON": r.get("CHUDECON", ""),
                "CAUHOI": r.get("CAUHOI", ""),
                "CAUTRALOI": r.get("CAUTRALOI", ""),
                "GIAITHICH": r.get("GIAITHICH", "")
            })

        return formatted_results

    except Exception as e:
        logger.error(f"[retrieve_from_qdrant] Error during retrieval: {e}")
        return []


def get_full_qa_by_ids(
    ids: List[int],
    collection_name: str = "bnrhm",
    qdrant_url: str = None
) -> List[Dict[str, Any]]:
    """
    Get full QA pairs by document IDs from Qdrant.
    """
    try:
        results = QdrantOperations.retrieve_by_ids(collection_name, ids)

        # Ensure keys exist
        formatted = []
        for r in results:
            formatted.append({
                "id": r["id"],
                "DEMUC": r.get("DEMUC", ""),
                "CHUDECON": r.get("CHUDECON", ""),
                "CAUHOI": r.get("CAUHOI", ""),
                "CAUTRALOI": r.get("CAUTRALOI", ""),
                "GIAITHICH": r.get("GIAITHICH", "")
            })
        return formatted

    except Exception as e:
        logger.error(f"[get_full_qa_by_ids] Error retrieving by IDs: {e}")
        return []

def retrieve_from_qdrant_with_cached_embeddings(
    query: str,
    demuc: Optional[str] = None,
    chu_de_con: Optional[str] = None,
    top_k: int = 20,
    collection_name: str = "bnrhm",
    qdrant_url: str = None,
    use_late_interaction: bool = True,
    embeddings: Optional[Dict] = None,
    return_embeddings: bool = False
) -> Tuple[List[Dict[str, Any]], Optional[Dict]]:
    """
    Retrieve documents from Qdrant with support for embedding reuse.
    """
    try:
        filter_builder = FilterBuilder()
        filter_builder.add_demuc(demuc)
        filter_builder.add_subtopic(chu_de_con)

        results, out_embeddings = QdrantOperations.search(
            collection_name=collection_name,
            query_text=query,
            query_filter=filter_builder.build(),
            top_k=top_k,
            use_late_interaction=use_late_interaction,
            embeddings=embeddings,
            return_embeddings=return_embeddings
        )
        
        return results, out_embeddings

    except Exception as e:
        logger.error(f"[retrieve_cached] ❌ Error: {e}", exc_info=True)
        return [], None
