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


class RetrieveFromKBWithoutDemuc(Node):
    """
    Retrieve relevant QA pairs from Qdrant WITHOUT demuc filter (global search only).
    
    Simple retrieval strategy:
    - prep(): Read query, role, and top_k from shared
    - exec(): Call Qdrant retrieval utility without any filters
    - post(): Write lightweight {id, CAUHOI, score} candidates to shared
    
    Output: shared["retrieved_candidates"] - list of lightweight candidates from global search
    """

    def prep(self, shared):
        # Read from shared store ONLY
        query = shared.get("retrieval_query") or shared.get("query")
        top_k = shared.get("top_k", 20)
        role = shared.get("role", RoleEnum.PATIENT_DENTAL.value)
        return {
            "query": query,
            "role": role,
            "top_k": top_k
        }

    def exec(self, inputs):
        retrieve_query = inputs["query"]
        role = inputs["role"]
        top_k = inputs["top_k"]
        
        # Call Qdrant retrieval utility function
        from utils.knowledge_base.qdrant_retrieval import retrieve_from_qdrant

        # Map role to collection name
        collection_name = ROLE_TO_COLLECTION.get(role, "bnrhm")

        # Global search WITHOUT any filters
        retrieved_results = retrieve_from_qdrant(
            query=retrieve_query,
            demuc=None,
            chu_de_con=None,
            top_k=top_k,
            collection_name=collection_name
        )

        # Extract lightweight candidates: {id, collection, CAUHOI, score}
        candidates = [
            {
                "id": result["id"],
                "collection": result.get("collection", collection_name),  # Preserve collection info
                "CAUHOI": result["CAUHOI"],
                "score": result.get("score", 0)
            }
            for result in retrieved_results
        ]
        
        logger.info(f"📚 [RetrieveFromKBWithoutDemuc] Query: {retrieve_query}")
        logger.info(f"📚 [RetrieveFromKBWithoutDemuc] Retrieved {len(candidates)} candidates (global search)")
        
        return candidates

    def post(self, shared, prep_res, exec_res):
        candidates = exec_res

        # Save lightweight candidates to shared store
        shared["retrieved_candidates"] = candidates
        
        # Group IDs by collection for efficient multi-collection retrieval
        ids_by_collection = {}
        for c in candidates:
            collection = c.get("collection", "bnrhm")
            if collection not in ids_by_collection:
                ids_by_collection[collection] = []
            ids_by_collection[collection].append(c["id"])
        
        # Save both formats for compatibility
        shared["selected_ids"] = [c["id"] for c in candidates]  # Legacy format
        shared["selected_ids_by_collection"] = ids_by_collection  # New format for multi-collection
        shared["selected_questions"] = [c["CAUHOI"] for c in candidates]
        
        logger.info(f"📚 [RetrieveFromKBWithoutDemuc] POST - IDs grouped by collection: {[(k, len(v)) for k, v in ids_by_collection.items()]}")
        
        # Update RAG state
        shared["rag_state"] = "retrieved"
        return "default"

