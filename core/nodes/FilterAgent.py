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



class FilterAgent(Node):
    """
    Filter candidates using LLM semantic understanding.

    Selects most relevant questions from candidates.
    Output: selected_ids (list of IDs)
    """

    def prep(self, shared):
        logger.info("🔍 [FilterAgent] PREP - Reading query and candidates")
        query = shared.get("retrieval_query") or shared.get("query")

        role =  shared.get("role","")  # patient_dental, patient_diabetes,vv
        display_user_role_name =  ""
        
        candidates = shared.get("retrieved_candidates", [])
        
        logger.info(f"🔍 [FilterAgent] PREP - Query: '{query}...', Candidates: {len(candidates)}")
        return query, candidates,role 

    def exec(self, inputs):
        from utils.llm import call_llm
        from utils.parsing import parse_yaml_with_schema
        from utils.auth import APIOverloadException
        from config.timeout_config import timeout_config
        from  utils.role_enum import RoleEnum, ROLE_DISPLAY_NAME
        query, candidates,role = inputs
        logger.info(f"🔍 [FilterAgent] EXEC - Filtering {len(candidates)} candidates")
        
        vietnamese_user_role  = ROLE_DISPLAY_NAME[RoleEnum(role)]   # Vd: vietnamese_user_role = Bệnh Nhân Nha Khoa , .. 
        # Handle empty candidates
        if not candidates:
            logger.warning("🔍 [FilterAgent] EXEC - No candidates to filter")
            return []

        # Handle very few candidates (≤ 3) - return all
        if len(candidates) <= 3:
            logger.info(f"🔍 [FilterAgent] EXEC - Only {len(candidates)} candidates, returning all")
            return [c["id"] for c in candidates]

        # Format candidates for LLM
        candidate_list_str = self._format_candidates(candidates)
        logger.info(f"🔍 [FilterAgent] EXEC danh sách câu hỏi : {candidate_list_str}")
        prompt = f"""
        Bạn là trợ lý y khoa 
Query hiện tại của user: {query}
user role: {vietnamese_user_role}

danh sách câu hỏi được retrieve:
{candidate_list_str}
Chọn tối đa 10 câu hỏi liên quan nhất tới câu hỏi của user. ( câu trả lời sẽ được retrieve sau đó)

YAML:
```yaml
selected_ids: [...]
```"""

        try:
            resp = call_llm(prompt, fast_mode=True, max_retry_time=timeout_config.LLM_RETRY_TIMEOUT)

            result = parse_yaml_with_schema(
                resp,
                required_fields=["selected_ids"],
                field_types={"selected_ids": list}
            )

            if result and result["selected_ids"]:
                # Cap at 6
                selected_ids = result["selected_ids"]
                selected_questions = [e.get('CAUHOI') for e in candidates if e.get('id') in selected_ids]
                logger.info(f"🔍 [FilterAgent] EXEC - Selected {selected_questions} ")
                return selected_ids,selected_questions
            else:
                # Fallback: top 6
                logger.warning("🔍 [FilterAgent] EXEC - LLM parsing failed, using top 6")
                return [c["id"] for c in candidates[:6]], [e.get('CAUHOI') for e in candidates[:6]]

        except (APIOverloadException, Exception) as e:
            logger.warning(f"🔍 [FilterAgent] EXEC - Error: {e}, using top 6")
            return [c["id"] for c in candidates[:6]]

    def _format_candidates(self, candidates: list) -> str:
        """Format candidates compactly for LLM prompt"""
        lines = []
        for i, c in enumerate(candidates, 1):
            question = c["CAUHOI"][:100] + "..." if len(c["CAUHOI"]) > 100 else c["CAUHOI"]
            lines.append(f"{i}. ID={c['id']}: \"{question}\"")
        return "\n".join(lines)

    def post(self, shared, prep_res, exec_res):
        selected_ids,selected_questions= exec_res 
        
        shared["selected_ids"] = selected_ids
        shared["selected_questions"] = selected_questions
        shared["rag_state"] = "filtered"

        logger.info(f"🔍 [FilterAgent] POST - Saved {len(selected_ids)} IDs")
        if  not selected_ids:
            return "fallback"
        return "default"

