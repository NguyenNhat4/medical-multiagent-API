# Core framework import
from pocketflow import Node

# Standard library imports
import logging

# Third-party imports
from utils.knowledge_base.qdrant_retrieval import get_full_qa_by_ids
from utils.role_enum import RoleEnum, PERSONA_BY_ROLE
from utils.helpers import format_kb_qa_list
from utils.llm import call_llm
from utils.parsing import parse_yaml_with_schema
from utils.llm.call_llm import APIOverloadException
from config.timeout_config import timeout_config
from config.chat_config import chat_config

# Configure logging for this module with Vietnam timezone
from utils.timezone_utils import setup_vietnam_logging
from config.logging_config import logging_config
from typing import List
if logging_config.USE_VIETNAM_TIMEZONE:
    logger = setup_vietnam_logging(__name__, 
                                 level=getattr(logging, logging_config.LOG_LEVEL.upper()),
                                 format_str=logging_config.LOG_FORMAT)
else:
    logger = logging.getLogger(__name__)
    logger.setLevel(getattr(logging, logging_config.LOG_LEVEL.upper()))


class ComposeAnswer(Node):
    def prep(self, shared):
        # Role to collection mapping
        ROLE_TO_COLLECTION = {
            RoleEnum.PATIENT_DIABETES.value: "bndtd",
            RoleEnum.DOCTOR_ENDOCRINE.value: "bsnt",
            RoleEnum.PATIENT_DENTAL.value: "bnrhm",
            RoleEnum.DOCTOR_DENTAL.value: "bsrhm",
        }
        context_summary = shared.get("context_summary", "")
        role = shared.get("role", "")
        
        # Prioritize retrieval_query over query for KB retrieval
        query = shared.get("retrieval_query") or shared.get("query")
        query_source = "retrieval_query" if shared.get("retrieval_query") else "query"
        
        # Check if we have multi-collection IDs (new format)
        selected_ids_by_collection = shared.get("selected_ids_by_collection", {})
        selected_ids = shared.get("selected_ids", [])  # Legacy fallback
        
        # Get relevant memories
        relevant_memories = shared.get("relevant_memories", [])

        # Map role to collection name (for legacy fallback)
        collection_name = ROLE_TO_COLLECTION.get(role, "bnrhm")

        logger.info(f"✍️ [ComposeAnswer] PREP - Role: '{role}' -> Collection: '{collection_name}', Query source: '{query_source}', Query: '{query[:50] if query else 'None'}...'")

        # Fetch full QA data from Qdrant using IDs
        retrieved_qa = []
        
        if selected_ids_by_collection:
            # New format: fetch from multiple collections
            logger.info(f"✍️ [ComposeAnswer] PREP - Fetching from multiple collections: {list(selected_ids_by_collection.keys())}")
            for coll_name, ids in selected_ids_by_collection.items():
                if ids:
                    qa_batch = get_full_qa_by_ids(ids, collection_name=coll_name)
                    retrieved_qa.extend(qa_batch)
                    logger.info(f"✍️ [ComposeAnswer] PREP - Retrieved {len(qa_batch)} QA pairs from '{coll_name}'")
            logger.info(f"✍️ [ComposeAnswer] PREP - Total retrieved: {len(retrieved_qa)} full QA pairs from all collections")
        elif selected_ids:
            # Legacy format: fetch from single collection
            logger.info(f"✍️ [ComposeAnswer] PREP - Using legacy format, fetching from single collection: '{collection_name}'")
            retrieved_qa = get_full_qa_by_ids(selected_ids, collection_name=collection_name)
            logger.info(f"✍️ [ComposeAnswer] PREP - Retrieved {len(retrieved_qa)} full QA pairs from Qdrant")
        else:
            logger.warning("✍️ [ComposeAnswer] PREP - No selected IDs, using empty list")
            retrieved_qa = []

        return {
            "role": role,
            "query": query,
            "retrieved_qa": retrieved_qa,
            "context_summary": context_summary,
            "relevant_memories": relevant_memories
        }

    def exec(self, inputs):
        role = inputs["role"]
        query = inputs["query"]
        retrieved = inputs["retrieved_qa"]
        context_summary = inputs["context_summary"]
        relevant_memories = inputs.get("relevant_memories", [])

        # Handle missing or invalid role with fallback
        if role not in PERSONA_BY_ROLE:
            logger.warning(f"✍️ [ComposeAnswer] EXEC - Invalid role '{role}', using default patient_diabetes role")
            role = "patient_diabetes"  # Default fallback role

        persona = PERSONA_BY_ROLE[role]
        # Compact KB context
        relevant_info_from_kb = format_kb_qa_list(retrieved)

        # Build memory context
        memory_context = ""
        if relevant_memories:
            memory_list = "\n".join([f"- {m.get('query', '')}" for m in relevant_memories[:3]])
            memory_context = f"\nThông tin từ các câu hỏi trước đây của người dùng (tham khảo thêm):\n{memory_list}\n"

        prompt = f"""
Hay cung cấp tri thức y khoa dựa trên cơ sở tri thức do bác sĩ biên soạn.
User là :{ persona["audience"] }
Câu hỏi cần trả lời: {query}

Danh sách Q&A đã retrieve:
{relevant_info_from_kb}

{memory_context}

Lưu ý quan trọng:
1) Phong cách: { persona["tone"]}.
2) Kết thúc bằng một dòng tóm lược bắt đầu bằng "👉 Tóm lại,".

```yaml
explanation: |
  <viết câu trả lời trực tiếp vào vấn đề dựa vào thông tin từ danh sách Q&A đã retrieve; KHÔNG bắt đầu bằng Chào bạn; dùng **nhấn mạnh** cho các từ khoá quan trọng>
  👉 Tóm lại, <tóm lược ngắn gọn>
suggestion_questions:
  - "Câu hỏi gợi ý 1"
  - "Câu hỏi gợi ý 2"
  - "Câu hỏi gợi ý 3"
```

Trả về chính xác cấu trúc yaml như ở trên (chú ý suggestion_questions là list, KHÔNG có dấu |):
"""
        # Log prompt with truncation to avoid flooding logs
        logger.info(f"✍️ [ComposeAnswer] EXEC - Full prompt: {prompt}")

        # Use proper timeout from config instead of hardcoded 1 second
        result = call_llm(prompt, max_retry_time=timeout_config.LLM_RETRY_TIMEOUT)
        logger.info(f"✍️ [ComposeAnswer] EXEC - LLM response received: {result}")
        # Parse and validate response structure
        parsed_result = parse_yaml_with_schema(
            result, 
            required_fields=["explanation", "suggestion_questions"], 
            field_types={"explanation": str, "suggestion_questions": list}
        )
        # import yaml
        # clean_result_dictionary = yaml.safe_load(result.split("```yaml")[1].split("```")[0].strip())
        # assert isinstance(clean_result_dictionary , dict) , "parse fail,"
        return parsed_result
        


    def post(self, shared, prep_res, exec_res):
   
        logger.info("✍️ [ComposeAnswer] POST - Lưu answer object")
        shared["answer_obj"] = exec_res
        shared["explain"] = exec_res.get("explanation", "")
        shared["suggestion_questions"] = exec_res.get("suggestion_questions", [])
        logger.info(f"✍️ [ComposeAnswer] POST - Answer keys: {list(exec_res.keys())}")
        
        # Log answer preview with safe truncation
        answer_preview = exec_res.get('explain', '')
        preview_text = answer_preview[:100] if answer_preview else 'None'
        logger.info(f"✍️ [ComposeAnswer] POST - Answer preview: {preview_text}...")
        
        # Check if API overload occurred and route to fallback
        if exec_res.get("api_overload", False):
            logger.info("✍️ [ComposeAnswer] POST - API overloaded, routing to fallback")
            return "fallback"
        
        return "default"
