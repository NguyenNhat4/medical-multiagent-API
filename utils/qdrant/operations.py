import logging
from typing import List, Dict, Any, Optional
from qdrant_client import models

from utils.qdrant.client import get_client
from utils.qdrant.config import (
    VECTOR_DENSE,
    VECTOR_SPARSE,
    VECTOR_LATE,
    get_vector_params,
    get_sparse_vector_params
)
from utils.qdrant.embeddings import EmbeddingService

logger = logging.getLogger(__name__)

class QdrantOperations:
    @staticmethod
    def ensure_collection(collection_name: str, recreate: bool = False) -> bool:
        """
        Ensures a collection exists. Creates it with standard config if not.
        """
        client = get_client()
        try:
            exists = client.collection_exists(collection_name)

            if exists and recreate:
                client.delete_collection(collection_name)
                exists = False

            if not exists:
                logger.info(f"[QdrantOps] Creating collection '{collection_name}'")

                # We need to pass qdrant_client.models to config helper
                client.create_collection(
                    collection_name=collection_name,
                    vectors_config=get_vector_params(models),
                    sparse_vectors_config=get_sparse_vector_params(models)
                )
                return True
            return True
        except Exception as e:
            logger.error(f"[QdrantOps] Error ensuring collection {collection_name}: {e}")
            return False

    @staticmethod
    def search(
        collection_name: str,
        query_text: str = None,
        query_filter: Optional[models.Filter] = None,
        top_k: int = 20,
        use_late_interaction: bool = True,
        embeddings: Optional[Dict] = None,
        return_embeddings: bool = False,
        score_threshold: float = None
    ) -> tuple[List[Dict[str, Any]], Optional[Dict]]:
        """
        Performs a hybrid search (Dense + Sparse + Optional Late Interaction).
        """
        try:
            # 1. Get Embeddings
            if embeddings:
                vectors = embeddings
            elif query_text:
                vectors = EmbeddingService.embed_query(query_text, use_late_interaction)
            else:
                raise ValueError("Either query_text or embeddings must be provided")

            dense_vec = vectors['dense']
            sparse_vec = vectors['sparse']
            late_vec = vectors.get('late')

            client = get_client()

            # 2. Build Prefetch
            prefetch = [
                models.Prefetch(
                    query=dense_vec,
                    using=VECTOR_DENSE,
                    limit=top_k + 50, # Fetch more for reranking
                    filter=query_filter
                ),
                models.Prefetch(
                    query=models.SparseVector(**sparse_vec.as_object()),
                    using=VECTOR_SPARSE,
                    limit=top_k + 50,
                    filter=query_filter
                ),
            ]

            # 3. Execute Query
            if use_late_interaction and late_vec is not None:
                search_result = client.query_points(
                    collection_name=collection_name,
                    prefetch=prefetch,
                    query=late_vec,
                    using=VECTOR_LATE,
                    limit=top_k,
                    filter=query_filter,
                    score_threshold=score_threshold
                )
            else:
                # RRF Fusion if no Late Interaction
                search_result = client.query_points(
                    collection_name=collection_name,
                    prefetch=prefetch,
                    query=models.FusionQuery(fusion=models.Fusion.RRF),
                    limit=top_k,
                    filter=query_filter,
                    score_threshold=score_threshold
                )

            # 4. Format Results
            results = []
            for point in search_result.points:
                payload = point.payload or {}
                results.append({
                    "id": point.id,
                    "score": point.score,
                    "collection": collection_name,
                    **payload
                })

            returned_embeddings = vectors if return_embeddings else None
            return results, returned_embeddings

        except Exception as e:
            logger.error(f"[QdrantOps] Search error in {collection_name}: {e}", exc_info=True)
            return [], None

    @staticmethod
    def retrieve_by_ids(
        collection_name: str,
        ids: List[Any]
    ) -> List[Dict[str, Any]]:
        """Retrieves full points by IDs."""
        try:
            client = get_client()
            records = client.retrieve(
                collection_name=collection_name,
                ids=ids,
                with_payload=True,
                with_vectors=False
            )
            return [
                {"id": r.id, **(r.payload or {})}
                for r in records
            ]
        except Exception as e:
            logger.error(f"[QdrantOps] Error retrieving IDs in {collection_name}: {e}")
            return []

    @staticmethod
    def upsert(
        collection_name: str,
        points: List[models.PointStruct]
    ) -> bool:
        """Upserts points into a collection."""
        try:
            client = get_client()
            client.upsert(
                collection_name=collection_name,
                points=points
            )
            return True
        except Exception as e:
            logger.error(f"[QdrantOps] Upsert error in {collection_name}: {e}")
            return False

    @staticmethod
    def delete(
        collection_name: str,
        point_ids: List[Any]
    ) -> bool:
        """Deletes points from a collection."""
        try:
            client = get_client()
            client.delete(
                collection_name=collection_name,
                points_selector=models.PointIdsList(points=point_ids)
            )
            return True
        except Exception as e:
            logger.error(f"[QdrantOps] Delete error in {collection_name}: {e}")
            return False
