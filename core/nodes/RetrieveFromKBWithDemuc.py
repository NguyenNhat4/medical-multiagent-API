# Core framework import
from pocketflow import Node

# Standard library imports
import logging

# Configure logging for this module with Vietnam timezone
from utils.timezone_utils import setup_vietnam_logging
from config.logging_config import logging_config
from utils.role_enum import RoleEnum

if logging_config.USE_VIETNAM_TIMEZONE:
    logger = setup_vietnam_logging(__name__,
                                 level=getattr(logging, logging_config.LOG_LEVEL.upper()),
                                 format_str=logging_config.LOG_FORMAT)
else:
    logger = logging.getLogger(__name__)
    logger.setLevel(getattr(logging, logging_config.LOG_LEVEL.upper()))


# Role to collection name mapping
ROLE_TO_COLLECTION = {
    RoleEnum.PATIENT_DIABETES.value: "bndtd",
    RoleEnum.DOCTOR_ENDOCRINE.value: "bsnt",
    RoleEnum.PATIENT_DENTAL.value: "bnrhm",
    RoleEnum.DOCTOR_DENTAL.value: "bsrhm",
}


class RetrieveFromKBWithDemuc(Node):
    """
    Retrieve relevant QA pairs from Qdrant WITH demuc filter + global search.
    
    Hybrid retrieval strategy:
    - prep(): Read query, metadata (demuc), role, and top_k from shared
    - exec(): 
        1. Search WITH demuc filter (narrow context)
        2. Search WITHOUT filters (global context)
        3. Combine, deduplicate, and sort by score
    - post(): Write lightweight {id, CAUHOI, score} candidates to shared
    
    Output: shared["retrieved_candidates"] - list of lightweight candidates from hybrid search
    """

    def prep(self, shared):
        # Read from shared store ONLY
        query = shared.get("retrieval_query") or shared.get("query")
        demuc = shared.get("demuc", "")
        top_k = shared.get("top_k", 20)
        role = shared.get("role", RoleEnum.PATIENT_DENTAL.value)
        return {
            "query": query,
            "demuc": demuc,
            "role": role,
            "top_k": top_k
        }

    def exec(self, inputs):
        retrieve_query = inputs["query"]
        demuc = inputs["demuc"]
        role = inputs["role"]
        top_k = inputs["top_k"]
        
        # Call Qdrant retrieval utility function
        from utils.knowledge_base.qdrant_retrieval import retrieve_from_qdrant

        # Map role to collection name
        collection_name = ROLE_TO_COLLECTION.get(role, "bnrhm")

        # Strategy:
        # 1. Search WITH demuc filter (narrow context)
        # 2. Search WITHOUT filters (global context)
        # 3. Combine and deduplicate

        # 1. Filtered search (by demuc only)
        retrieved_results_filtered = retrieve_from_qdrant(
            query=retrieve_query,
            demuc=demuc,
            chu_de_con=None,  # Ignore sub-topic
            top_k=top_k,
            collection_name=collection_name
        )
        
        # 2. Global search (no filters)
        retrieved_results_global = retrieve_from_qdrant(
            query=retrieve_query,
            demuc=None,
            chu_de_con=None,
            top_k=top_k,
            collection_name=collection_name
        )
        
        # 3. Combine results: Filtered first (more relevant), then Global
        retrieved_results = retrieved_results_filtered + retrieved_results_global
        
        # Deduplicate by ID
        seen_ids = set()
        unique_results = []
        for entry in retrieved_results:
            if entry["id"] not in seen_ids:
                seen_ids.add(entry["id"])
                unique_results.append(entry)

        # Sort by score descending
        unique_results.sort(key=lambda x: x.get("score", 0), reverse=True)
        
        # Take top k results
        top_results = unique_results[:top_k]

        # Extract lightweight candidates: {id, CAUHOI, score}
        candidates = [
            {
                "id": result["id"],
                "CAUHOI": result["CAUHOI"],
                "score": result.get("score", 0)
            }
            for result in top_results
        ]
        
        logger.info(f"📚 [RetrieveFromKBWithDemuc] Query: {retrieve_query}")
        logger.info(f"📚 [RetrieveFromKBWithDemuc] Demuc filter: {demuc}")
        logger.info(f"📚 [RetrieveFromKBWithDemuc] Retrieved {len(candidates)} candidates (hybrid search)")
        logger.info(f"📚 [RetrieveFromKBWithDemuc] Filtered results: {len(retrieved_results_filtered)}, Global results: {len(retrieved_results_global)}")
        
        return candidates

    def post(self, shared, prep_res, exec_res):
        candidates = exec_res

        # Save lightweight candidates to shared store
        shared["retrieved_candidates"] = candidates
        
        # Direct pass-through: Save selected IDs and questions
        shared["selected_ids"] = [c["id"] for c in candidates]
        shared["selected_questions"] = [c["CAUHOI"] for c in candidates]
        
        # Update RAG state
        shared["rag_state"] = "retrieved"
        return "default"

