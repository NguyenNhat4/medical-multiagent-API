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





class RetrieveFromKB(Node):
    """
    Retrieve relevant QA pairs from Qdrant vector database using hybrid search.

    ID-based architecture - no scoring needed (FilterAgent handles semantic filtering):
    - prep(): Read query, metadata, and role from shared
    - exec(): Call Qdrant retrieval utility with role-specific collection
    - post(): Write lightweight {id, CAUHOI} to shared

    Output: shared["retrieved_candidates"] - list of lightweight candidates
    """

    def prep(self, shared):
        # Read from shared store ONLY
        query = shared.get("retrieval_query") or shared.get("query")
        demuc = shared.get("demuc", "")
        # chu_de_con = shared.get("chu_de_con", "") # Not used anymore
        top_k = shared.get("top_k", 20)
        role = shared.get("role", RoleEnum.PATIENT_DENTAL.value)
        return query, demuc, role, top_k

    def exec(self, inputs):
        retrieve_query, demuc, role, top_k = inputs
        # Call Qdrant retrieval utility function
        from utils.knowledge_base.qdrant_retrieval import retrieve_from_qdrant

        # Map role to collection name
        collection_name = ROLE_TO_COLLECTION.get(role, "bnrhm")

        # Strategy:
        # 1. Search WITHOUT filters (global context)
        # 2. Search WITH demuc filter (narrow context) - if demuc exists
        # 3. Combine and deduplicate

        # 1. Global search (no filters)
        retrieved_results_global = retrieve_from_qdrant(
            query=retrieve_query,
            demuc=None,
            chu_de_con=None,
            top_k=top_k,
            collection_name=collection_name
        )
        
        retrieved_results_filtered = []
        if demuc:
            # 2. Filtered search (only by demuc, ignore chu_de_con)
            retrieved_results_filtered = retrieve_from_qdrant(
                query=retrieve_query,
                demuc=demuc,
                chu_de_con=None, # Ignore sub-topic
                top_k=top_k, # Slightly less than global
                collection_name=collection_name
            )
          
        # Combine results: Filtered first (more relevant), then Global
        retrieved_results = retrieved_results_filtered + retrieved_results_global
        
        # Filter out duplicate
        seen_ids = set()
        unique_results = []
        for e in retrieved_results:
            if e["id"] not in seen_ids:
                seen_ids.add(e["id"])
                unique_results.append(e)

        # Sort by score descending
        unique_results.sort(key=lambda x: x.get("score", 0), reverse=True)
        
        # Take top k highest score
        top_results = unique_results

        # Extract lightweight candidates: {id, CAUHOI}
        candidates = [
            {
                "id": result["id"],
                "CAUHOI": result["CAUHOI"],
                "score": result.get("score", 0)
            }
            for result in top_results
        ]
        
        questions = [q["CAUHOI"] for q in candidates]
        logger.info(f"📚 [RetrieveFromKB] Query used to retrieve: {retrieve_query}")
        logger.info(f"📚 [RetrieveFromKB] Retrieved {len(candidates)} top candidates (sorted by score)")
        
        return candidates

    def post(self, shared, prep_res, exec_res):

        candidates = exec_res

        # Save lightweight candidates to shared store
        shared["retrieved_candidates"] = candidates
        
        # Direct pass-through (replacing FilterAgent logic):
        # Save selected_ids and selected_questions directly from top candidates
        shared["selected_ids"] = [c["id"] for c in candidates]
        shared["selected_questions"] = [c["CAUHOI"] for c in candidates]
        
        # Update RAG state
        shared["rag_state"] = "retrieved"
        return "default" 
