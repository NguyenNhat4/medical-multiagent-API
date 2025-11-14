# Core framework import
from pocketflow import Node

# Standard library imports
import logging

# Configure logging for this module with Vietnam timezone
from utils.timezone_utils import setup_vietnam_logging
from config.logging_config import logging_config

if logging_config.USE_VIETNAM_TIMEZONE:
    logger = setup_vietnam_logging(__name__, 
                                 level=getattr(logging, logging_config.LOG_LEVEL.upper()),
                                 format_str=logging_config.LOG_FORMAT)
else:
    logger = logging.getLogger(__name__)
    logger.setLevel(getattr(logging, logging_config.LOG_LEVEL.upper()))



class TopicClassifyAgent(Node):
    """
    Agent phân loại chủ đề chính (DEMUC only).

    Refactored to follow PocketFlow best practices:
    - prep(): Read from shared store ONLY (no DB/API calls)
    - exec(): Call utility functions to classify DEMUC based on role's CSV file
    - post(): Write to shared store ONLY

    Classification:
    - Classify DEMUC from query based on role
    - CHU_DE_CON is always left empty (not classified)
    """

    def prep(self, shared):

        # Read ALL data from shared store - no external calls
        query = shared.get("query", "").strip()
        role = shared.get("role", "")
        current_demuc = shared.get("demuc", "")
        current_chu_de_con = shared.get("chu_de_con", "")

        return query, role, current_demuc, current_chu_de_con

    def exec(self, inputs):
        query, role, current_demuc, current_chu_de_con = inputs

        from utils.knowledge_base.metadata_utils import (
            get_demuc_list_for_role,
            format_demuc_list_for_prompt
        )
        from utils.llm.classify_topic import classify_demuc_with_llm

        # Only classify DEMUC (no CHU_DE_CON classification)

        # Get DEMUC list for role
        demuc_list = get_demuc_list_for_role(role)
        if not demuc_list:
            logger.warning(f"🏷️ [TopicClassifyAgent] EXEC - No DEMUC list found for role '{role}'")
            return {"demuc": "", "chu_de_con": "", "confidence": "low"}

        demuc_list_str = format_demuc_list_for_prompt(demuc_list)
        logger.info(f"🏷️ [TopicClassifyAgent] EXEC - Available DEMUCs: {demuc_list}")

        # Classify DEMUC
        demuc_result = classify_demuc_with_llm(
            query=query,
            role=role,
            demuc_list_str=demuc_list_str
        )

        if demuc_result.get("api_overload"):
            return {"demuc": "", "chu_de_con": "", "confidence": "low", "api_overload": True}

        classified_demuc = demuc_result.get("demuc", "")
        logger.info(f'🏷️ [TopicClassifyAgent] EXEC - Classification result: DEMUC="{classified_demuc}", confidence="{demuc_result.get("confidence", "low")}", reason="{demuc_result.get("reason", "")}" ')

        # Return with DEMUC only (no CHU_DE_CON)
        return {
            "demuc": classified_demuc,
            "chu_de_con": "",  # Always empty - we don't classify CHU_DE_CON
            "confidence": demuc_result.get("confidence", "low"),
            "reason": demuc_result.get("reason", "")
        }

    def post(self, shared, prep_res, exec_res):

        # Update shared with classification results - WRITE ONLY
        shared["demuc"] = exec_res.get("demuc", "")
        shared["chu_de_con"] = exec_res.get("chu_de_con", "")  # Always empty now
        shared["classification_confidence"] = exec_res.get("confidence", "low")


        # Check for API overload
        if exec_res.get("api_overload", False):
            return "fallback"
        # Classification complete - proceed to retrieval
        return "default"  # Go to next node (RetrieveFromKB)

