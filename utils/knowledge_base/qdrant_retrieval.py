"""
Utility function for Qdrant hybrid search retrieval.

This is an external utility function for vector database operations.
According to PocketFlow best practices, this should be independent and easily testable.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from qdrant_client import QdrantClient, models
from fastembed import TextEmbedding, LateInteractionTextEmbedding, SparseTextEmbedding
import os 
from dotenv import load_dotenv
logger = logging.getLogger(__name__)
import shutil

load_dotenv(override=False)

# Cache directory for embedding models
FASTEMBED_CACHE = os.getenv("FASTEMBED_CACHE_PATH", "./models")

# Global embedding models (lazy loaded)
_dense_model = None
_sparse_model = None
_late_interaction_model = None


def _get_embedding_models():
    """
    Lazy load embedding models (singleton pattern) with auto-recovery from corruption.
    """
    global _dense_model, _sparse_model, _late_interaction_model

    if _dense_model is None:
        logger.info(f"[Qdrant] Loading embedding models from cache: {FASTEMBED_CACHE}")

        # Helper function to load models, helps reuse retry logic
        def load_models():
            return (
                TextEmbedding("sentence-transformers/all-MiniLM-L6-v2", cache_dir=FASTEMBED_CACHE, providers=['CPUExecutionProvider']),
                SparseTextEmbedding("Qdrant/bm25", cache_dir=FASTEMBED_CACHE),
                LateInteractionTextEmbedding("colbert-ir/colbertv2.0", cache_dir=FASTEMBED_CACHE)
            )

        def clear_all_model_caches():
            """Clear all model caches in case of corruption."""
            model_patterns = [
                "models--sentence-transformers--all-MiniLM-L6-v2",
                "models--qdrant--all-MiniLM-L6-v2-onnx",
                "models--Qdrant--bm25",
                "models--colbert-ir--colbertv2.0",
            ]

            for pattern in model_patterns:
                model_path = os.path.join(FASTEMBED_CACHE, pattern)
                if os.path.exists(model_path):
                    logger.warning(f"[Qdrant] Deleting potentially corrupted cache: {model_path}")
                    try:
                        shutil.rmtree(model_path)
                    except Exception as rm_error:
                        logger.error(f"[Qdrant] Could not delete {model_path}: {rm_error}")

        max_retries = 3
        for attempt in range(1, max_retries + 1):
            try:
                logger.info(f"[Qdrant] Loading models (attempt {attempt}/{max_retries})...")
                _dense_model, _sparse_model, _late_interaction_model = load_models()
                logger.info("[Qdrant] ✅ Embedding models loaded successfully")
                break

            except Exception as e:
                error_msg = str(e)

                # Check if it's a corruption or download error
                is_corruption = any(keyword in error_msg.lower() for keyword in [
                    'modelproto does not have a graph',
                    'onnxruntimeerror',
                    'corrupted',
                    'download',
                    'incomplete',
                    'could not download'
                ])

                if is_corruption:
                    logger.warning(f"[Qdrant] ⚠️ Model corruption/download error detected (attempt {attempt}/{max_retries}): {e}")

                    if attempt < max_retries:
                        logger.info(f"[Qdrant] 🧹 Clearing model caches and retrying...")
                        clear_all_model_caches()

                        # Wait before retry
                        import time
                        wait_time = 2 ** attempt  # Exponential backoff: 2, 4, 8 seconds
                        logger.info(f"[Qdrant] ⏳ Waiting {wait_time} seconds before retry...")
                        time.sleep(wait_time)
                    else:
                        logger.error(f"[Qdrant] ❌ Failed to load models after {max_retries} attempts")
                        logger.error(f"[Qdrant] Please check your internet connection and Docker build logs")
                        raise RuntimeError(
                            f"Failed to load embedding models after {max_retries} attempts. "
                            "The models may be corrupted. Please rebuild the Docker image or clear the model cache."
                        ) from e
                else:
                    # Non-corruption error, raise immediately
                    logger.error(f"[Qdrant] ❌ Failed to load embedding models: {e}")
                    raise

    return _dense_model, _sparse_model, _late_interaction_model

def retrieve_from_qdrant(
    query: str,
    demuc: Optional[str] = None,
    chu_de_con: Optional[str] = None,
    top_k: int = 20,
    collection_name: str = "bnrhm",
    qdrant_url: str = os.getenv("QDRANT_URL"),
    use_late_interaction: bool = True
) -> List[Dict[str, Any]]:
    """
    Retrieve documents from Qdrant using hybrid search (dense + sparse + [optional] late interaction).

    Input:
        - query (str): User's question
        - demuc (str, optional): Filter by DEMUC if provided
        - chu_de_con (str, optional): Filter by CHU_DE_CON if provided
        - top_k (int): Number of results to return (default: 20)
        - collection_name (str): Qdrant collection name (role-specific: bndtd, bsnt, bnrhm, bsrhm)
        - qdrant_url (str): Qdrant server URL
        - use_late_interaction (bool): Whether to use ColBERT late interaction (default: True)

    Output:
        List of dicts with keys: id, score, DEMUC, CHUDECON, CAUHOI, CAUTRALOI, GIAITHICH
    """
    try:
        logger.info(f"[retrieve_from_qdrant] Query: '{query}...', Filters: demuc={demuc}, sub={chu_de_con}, LateInteraction={use_late_interaction}")

        # Get embedding models
        dense_model, sparse_model, late_interaction_model = _get_embedding_models()

        # Embed query
        dense_vectors = next(dense_model.query_embed(query))
        sparse_vectors = next(sparse_model.query_embed(query))
        
        # Only compute late interaction vectors if needed
        late_vectors = None
        if use_late_interaction:
            late_vectors = next(late_interaction_model.query_embed(query))

        logger.info(f"[retrieve_from_qdrant] Query embeddings generated (LI={use_late_interaction})")

        # Create Qdrant client
        client = QdrantClient(url=qdrant_url)

        # Build prefetch for hybrid search
        prefetch = [
            models.Prefetch(
                query=dense_vectors,
                using="all-MiniLM-L6-v2",
                limit=top_k + 100,
            ),
            models.Prefetch(
                query=models.SparseVector(**sparse_vectors.as_object()),
                using="bm25",
                limit=top_k + 100,
            ),
        ]

        # Build filter based on DEMUC and CHU_DE_CON
        query_filter = None
        if demuc or chu_de_con:
            conditions = []
            if demuc:
                conditions.append(
                    models.FieldCondition(
                        key="DEMUC",
                        match=models.MatchValue(value=demuc)
                    )
                )
            if chu_de_con:
                conditions.append(
                    models.FieldCondition(
                        key="CHUDECON",
                        match=models.MatchValue(value=chu_de_con)
                    )
                )
            query_filter = models.Filter(must=conditions)
            logger.info(f"[retrieve_from_qdrant] Applying filters: DEMUC={demuc}, CHU_DE_CON={chu_de_con}")

        # Execute hybrid search
        if use_late_interaction and late_vectors is not None:
            # Case 1: Late Interaction (ColBERT) as root query
            results = client.query_points(
                collection_name,
                prefetch=prefetch,
                query=late_vectors,
                using="colbertv2.0",
                with_payload=True,
                limit=top_k,
                query_filter=query_filter
            )
        else:
            # Case 2: No Late Interaction -> Use Fusion (RRF) of prefetch results
            results = client.query_points(
                collection_name,
                prefetch=prefetch,
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                with_payload=True,
                limit=top_k,
                query_filter=query_filter
            )

        logger.info(f"[retrieve_from_qdrant] Retrieved {len(results.points)} results")

        # Format results
        formatted_results = []
        for point in results.points:
            formatted_results.append({
                "id": point.id,
                "score": point.score,
                "collection": collection_name,  # Add collection name for multi-collection search
                "DEMUC": point.payload.get("DEMUC", ""),
                "CHUDECON": point.payload.get("CHUDECON", ""),
                "CAUHOI": point.payload.get("CAUHOI", ""),
                "CAUTRALOI": point.payload.get("CAUTRALOI", ""),
                "GIAITHICH": point.payload.get("GIAITHICH", "")
            })

        # Log top results
        if formatted_results:
            logger.info(f"[retrieve_from_qdrant] Top 3 results:")
            for i, result in enumerate(formatted_results[:3], 1):
                logger.info(f"  {i}. score={result['score']:.4f} | Q: {result['CAUHOI'][:80]}...")

        return formatted_results

    except Exception as e:
        logger.error(f"[retrieve_from_qdrant] Error during retrieval: {e}")
        return []


def get_full_qa_by_ids(
    ids: List[int],
    collection_name: str = "bnrhm",
    qdrant_url: str = os.getenv("QDRANT_URL")
) -> List[Dict[str, Any]]:
    """
    Get full QA pairs by document IDs from Qdrant.

    Input:
        - ids (List[int]): List of document IDs
        - collection_name (str): Qdrant collection name (default: bnrhm for patient_dental)
        - qdrant_url (str): Qdrant server URL

    Output:
        List of dicts with full QA information

    Necessity: Used to get complete QA pairs after retrieval
    """
    try:
        logger.info(f"[get_full_qa_by_ids] Retrieving {len(ids)} documents by IDs")

        client = QdrantClient(url=qdrant_url)

        records = client.retrieve(
            collection_name=collection_name,
            ids=ids,
            with_payload=True,
            with_vectors=False
        )

        results = []
        for record in records:
            results.append({
                "id": record.id,
                "DEMUC": record.payload.get("DEMUC", ""),
                "CHUDECON": record.payload.get("CHUDECON", ""),
                "CAUHOI": record.payload.get("CAUHOI", ""),
                "CAUTRALOI": record.payload.get("CAUTRALOI", ""),
                "GIAITHICH": record.payload.get("GIAITHICH", "")
            })

        logger.info(f"[get_full_qa_by_ids] Retrieved {len(results)} full QA pairs")
        return results

    except Exception as e:
        logger.error(f"[get_full_qa_by_ids] Error retrieving by IDs: {e}")
        return []


if __name__ == "__main__":
    # Test the utility function
    print("=" * 80)
    print("Testing Qdrant retrieval utility")
    print("=" * 80)

    # Test 1: Basic retrieval
    print("\nTest 1: Basic retrieval without filters")
    query = "tại sao tiểu đường nguy hiểm"
    results = retrieve_from_qdrant(query=query, top_k=5)

    print(f"\nQuery: '{query}'")
    print(f"Results: {len(results)}")
    if results:
        print("\nTop 3 results:")
        for i, r in enumerate(results[:3], 1):
            print(f"{i}. Score: {r['score']:.4f}")
            print(f"   Q: {r['CAUHOI'][:100]}...")
            print(f"   DEMUC: {r['DEMUC']}")

    # Test 2: Retrieval with DEMUC filter
    print("\n" + "=" * 80)
    print("Test 2: Retrieval with DEMUC filter")
    results_filtered = retrieve_from_qdrant(
        query=query,
        demuc="BỆNH ĐÁI THÁO ĐƯỜNG",
        top_k=5
    )

    print(f"\nQuery: '{query}'")
    print(f"Filter: DEMUC='BỆNH ĐÁI THÁO ĐƯỜNG'")
    print(f"Results: {len(results_filtered)}")

    # Test 3: Get full QA by IDs
    if results:
        print("\n" + "=" * 80)
        print("Test 3: Get full QA by IDs")
        ids = [r["id"] for r in results[:3]]
        full_qa = get_full_qa_by_ids(ids)

        print(f"\nIDs: {ids}")
        print(f"Full QA pairs retrieved: {len(full_qa)}")
        if full_qa:
            print("\nFirst QA pair:")
            qa = full_qa[0]
            print(f"Q: {qa.get('CAUHOI', 'N/A')}")
            print(f"A: {qa.get('CAUTRALOI', 'N/A')[:150]}...")


def retrieve_from_qdrant_with_cached_embeddings(
    query: str,
    demuc: Optional[str] = None,
    chu_de_con: Optional[str] = None,
    top_k: int = 20,
    collection_name: str = "bnrhm",
    qdrant_url: str = os.getenv("QDRANT_URL"),
    use_late_interaction: bool = True,
    embeddings: Optional[Dict] = None,
    return_embeddings: bool = False
) -> Tuple[List[Dict[str, Any]], Optional[Dict]]:
    """
    Retrieve documents from Qdrant with support for embedding reuse.
    
    This optimized version allows reusing embeddings across multiple searches,
    significantly improving performance when searching multiple collections.
    
    Args:
        embeddings: Optional dict with keys 'dense', 'sparse', 'late' to reuse
        return_embeddings: If True, return embeddings for reuse in next calls
        ... (other args same as retrieve_from_qdrant)
    
    Returns:
        Tuple of (results, embeddings) if return_embeddings=True, else (results, None)
    """
    try:
        logger.info(f"[retrieve_cached] Query: '{query[:50]}...', Collection: {collection_name}")

        # Get embedding models (singleton, loaded once)
        dense_model, sparse_model, late_interaction_model = _get_embedding_models()

        # Reuse embeddings if provided, otherwise compute new ones
        if embeddings:
            logger.info(f"[retrieve_cached] ✨ Reusing cached embeddings")
            dense_vectors = embeddings['dense']
            sparse_vectors = embeddings['sparse']
            late_vectors = embeddings.get('late')
        else:
            logger.info(f"[retrieve_cached] 🔄 Computing new embeddings")
            dense_vectors = next(dense_model.query_embed(query))
            sparse_vectors = next(sparse_model.query_embed(query))
            late_vectors = None
            if use_late_interaction:
                late_vectors = next(late_interaction_model.query_embed(query))

        # Create Qdrant client
        client = QdrantClient(url=qdrant_url)

        # Build prefetch for hybrid search
        prefetch = [
            models.Prefetch(
                query=dense_vectors,
                using="all-MiniLM-L6-v2",
                limit=top_k + 100,
            ),
            models.Prefetch(
                query=models.SparseVector(**sparse_vectors.as_object()),
                using="bm25",
                limit=top_k + 100,
            ),
        ]

        # Build filter based on DEMUC and CHU_DE_CON
        query_filter = None
        if demuc or chu_de_con:
            conditions = []
            if demuc:
                conditions.append(
                    models.FieldCondition(
                        key="DEMUC",
                        match=models.MatchValue(value=demuc)
                    )
                )
            if chu_de_con:
                conditions.append(
                    models.FieldCondition(
                        key="CHUDECON",
                        match=models.MatchValue(value=chu_de_con)
                    )
                )
            query_filter = models.Filter(must=conditions)

        # Execute hybrid search with optional late interaction reranking
        if use_late_interaction and late_vectors is not None:
            search_result = client.query_points(
                collection_name=collection_name,
                prefetch=prefetch,
                query=late_vectors,
                using="colbertv2.0",
                limit=top_k,
                query_filter=query_filter,
            )
        else:
            # Fallback: search without late interaction
            search_result = client.query_points(
                collection_name=collection_name,
                prefetch=prefetch,
                query=dense_vectors,
                using="all-MiniLM-L6-v2",
                limit=top_k,
                query_filter=query_filter,
            )

        # Format results
        results = []
        for point in search_result.points:
            results.append({
                "id": point.id,
                "score": point.score,
                "collection": collection_name,
                **point.payload
            })

        logger.info(f"[retrieve_cached] ✅ Retrieved {len(results)} results")

        # Return embeddings for reuse if requested
        if return_embeddings:
            emb_cache = {
                'dense': dense_vectors,
                'sparse': sparse_vectors,
                'late': late_vectors
            }
            return results, emb_cache
        
        return results, None

    except Exception as e:
        logger.error(f"[retrieve_cached] ❌ Error: {e}", exc_info=True)
        return [], None
