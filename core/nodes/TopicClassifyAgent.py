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
    Agent phân loại chủ đề:
      - DEMUC (bắt buộc, nếu chưa có)
      - CHU_DE_CON (ĐÃ BỎ - KHÔNG CẦN PHÂN LOẠI NỮA)

    Quy ước PocketFlow:
      - prep(): chỉ đọc từ shared
      - exec(): gọi util/LLM để phân loại
      - post(): chỉ ghi vào shared

    Luồng:
      A) Nếu chưa có demuc:
         - Lấy danh sách DEMUC theo role
         - Gọi classify_demuc_with_llm
         - Trả về demuc, chu_de_con=""
    """

    def prep(self, shared):

        # Read ALL data from shared store - no external calls
        query = shared.get("retrieval_query") or shared.get("query")

        role = shared.get("role", "")
        current_demuc = shared.get("demuc", "")
        # current_chu_de_con = shared.get("chu_de_con", "") # Not used anymore
        current_chu_de_con = "" 
        rag_state = shared.get("rag_state", "")
        
        if rag_state == "create_retrieval_query_reason":
            current_demuc = ""
            
        return query, role, current_demuc, current_chu_de_con, rag_state

    def exec(self, inputs):
        query, role, current_demuc, current_chu_de_con, rag_state = inputs
        demuc_result = {"confidence": "", "reason": ""}
        from utils.knowledge_base.metadata_utils import (
            get_demuc_list_for_role,
            format_demuc_list_for_prompt
        )
        
        if not current_demuc:
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
            
            current_demuc = demuc_result.get("demuc", "")
            # assert current_demuc != "", "DEMUC is not classified" 
            
            logger.info(f'🏷️ [TopicClassifyAgent] EXEC - Classification result: DEMUC="{demuc_result.get("demuc", "")}", confidence="{demuc_result.get("confidence", "low")}", reason="{demuc_result.get("reason", "")}" ')

        # NO LONGER CLASSIFY SUB-TOPIC (CHU_DE_CON)
        # if not current_chu_de_con and rag_state != "create_retrieval_query_reason":          
        #     chu_de_con_list_str = get_chu_de_con_for_demuc(role=role,demuc=current_demuc)
        #     logger.info(f"🏷️ [TopicClassifyAgent] EXEC - Available chu_de_cons: {chu_de_con_list_str}")
            
        #     chu_de_con_result = classify_chu_de_con_with_llm(query = query, demuc =current_demuc,chu_de_con_list_str=chu_de_con_list_str)
        #     current_chu_de_con = chu_de_con_result.get("chu_de_con","")
        #     logger.info(f"🏷️ [TopicClassifyAgent] EXEC - Classification chu_de_cons: {current_chu_de_con}")
     
        return {
            "demuc": current_demuc,
            "chu_de_con": "", # Always empty
            "confidence": demuc_result.get("confidence", "low"),
            "reason": demuc_result.get("reason", "")
        }     
     
    def post(self, shared, prep_res, exec_res):

        # Update shared with classification results - WRITE ONLY
        shared["demuc"] = exec_res.get("demuc", "")
        shared["chu_de_con"] = "" # exec_res.get("chu_de_con", "")  # Always empty now
        shared["classification_confidence"] = exec_res.get("confidence", "low")


        # Check for API overload
        if exec_res.get("api_overload", False):
            return "fallback"
        # Classification complete - proceed to retrieval
        return "default"  # Go to next node (RetrieveFromKB)
