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
        query = shared.get("query", "")
        demuc = shared.get("demuc", "")
        chu_de_con = shared.get("chu_de_con", "")
        role = shared.get("role", RoleEnum.PATIENT_DENTAL.value)
        return query, demuc, chu_de_con, role

    def exec(self, inputs):
        query, demuc, chu_de_con, role = inputs
        # Call Qdrant retrieval utility function
        from utils.knowledge_base.qdrant_retrieval import retrieve_from_qdrant

        # Map role to collection name
        collection_name = ROLE_TO_COLLECTION.get(role, "bnrhm")

        logger.info(f"📚 [RetrieveFromKB] Role: {role} -> Collection: {collection_name}")

        # Retrieve with filters if available
        retrieved_results = retrieve_from_qdrant(
            query=query,
            demuc=demuc if demuc else None,
            chu_de_con=chu_de_con if chu_de_con else None,
            top_k=20,
            collection_name=collection_name
        )
        logger.info(f"📚 [RetrieveFromKB] retrieve results: {retrieved_results}")

        # Extract lightweight candidates: {id, CAUHOI}
        candidates = [
            {
                "id": result["id"],
                "CAUHOI": result["CAUHOI"]
            }
            for result in retrieved_results
        ]

        # Log top results
        if candidates:
            lines = ["\n📚 [RetrieveFromKB] TOP CANDIDATES:"]
            for i, candidate in enumerate(candidates[:5], 1):
                lines.append(
                    f"  {i}. id={candidate['id']} | Q: {candidate['CAUHOI']}..."
                )
            logger.info("\n".join(lines))

        return candidates

    def post(self, shared, prep_res, exec_res):

        candidates = exec_res

        # Save lightweight candidates to shared store
        shared["retrieved_candidates"] = candidates
        # Update RAG state
        shared["rag_state"] = "retrieved"
        return "default" 


