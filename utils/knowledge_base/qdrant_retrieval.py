"""
Utility function for Qdrant hybrid search retrieval.

This is an external utility function for vector database operations.
According to PocketFlow best practices, this should be independent and easily testable.
"""

import logging
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient, models
from fastembed import TextEmbedding, LateInteractionTextEmbedding, SparseTextEmbedding
import os 
from dotenv import load_dotenv
logger = logging.getLogger(__name__)

load_dotenv(override=False)
# Global embedding models (lazy loaded)
_dense_model = None
_sparse_model = None
_late_interaction_model = None


def _get_embedding_models():
    """
    Lazy load embedding models (singleton pattern).

    Returns: (dense_model, sparse_model, late_interaction_model)
    """
    global _dense_model, _sparse_model, _late_interaction_model

    if _dense_model is None:
        logger.info("[Qdrant] Loading embedding models...")
        _dense_model = TextEmbedding("sentence-transformers/all-MiniLM-L6-v2")
        _sparse_model = SparseTextEmbedding("Qdrant/bm25")
        _late_interaction_model = LateInteractionTextEmbedding("colbert-ir/colbertv2.0")
        logger.info("[Qdrant] Embedding models loaded successfully")

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
