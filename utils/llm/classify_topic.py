"""
Utility functions for topic classification using LLM.

These are external utility functions that handle LLM calls for 2-step topic classification:
1. classify_demuc_with_llm: Find DEMUC from query
2. classify_chu_de_con_with_llm: Find CHU_DE_CON within a DEMUC

According to PocketFlow best practices, these should be independent and easily testable.
"""

import logging
from typing import Dict, Any
from utils.llm import call_llm
from utils.parsing import parse_yaml_with_schema
from utils.auth import APIOverloadException
from config.timeout_config import timeout_config

logger = logging.getLogger(__name__)


def classify_demuc_with_llm(
    query: str,
    role: str,
    demuc_list_str: str
) -> Dict[str, Any]:
    """
    Step 1: Call LLM to classify DEMUC (main topic) from query.

    Input:
        - query (str): User's question
        - role (str): User's role (e.g., "patient_diabetes")
        - demuc_list_str (str): Formatted DEMUC list as string

    Output:
        Dict with keys: demuc, confidence, reason, api_overload (optional)

    Necessity: Used by TopicClassifyAgent for STEP 1 - finding DEMUC
    """
    try:
        prompt = f"""
Bạn là trợ lý y khoa chuyên phân loại chủ đề câu hỏi.

Câu hỏi của người dùng: "{query}"
Role: {role}

Danh sách DEMUC (đề mục) có sẵn:
{demuc_list_str}

NHIỆM VỤ: Chọn DEMUC phù hợp nhất từ danh sách trên.

YÊU CẦU:
- demuc: chọn CHÍNH XÁC một DEMUC từ danh sách (viết đúng y hệt)
- confidence: high/medium/low
- reason: lý do ngắn gọn

VÍ DỤ:
Input: "Tôi muốn hỏi về bệnh đái tháo đường"
Danh sách: ["BỆNH LÝ ĐTĐ", "DINH DƯỠNG", "ĐIỀU TRỊ"]
Output:
```yaml
demuc: "BỆNH LÝ ĐTĐ"
confidence: "high"
reason: "Câu hỏi về bệnh đái tháo đường"
```

Trả về CHỈ một code block YAML hợp lệ:

```yaml
demuc: "TÊN ĐỀ MỤC"
confidence: "high"
reason: "Lý do"
```
"""

        logger.info(f"[classify_demuc_with_llm] Calling LLM to classify DEMUC")

        resp = call_llm(prompt, fast_mode=True, max_retry_time=timeout_config.LLM_RETRY_TIMEOUT)
        logger.info(f"[classify_demuc_with_llm] LLM response received")

        result = parse_yaml_with_schema(
            resp,
            required_fields=["demuc"],
            optional_fields=["confidence", "reason"],
            field_types={"demuc": str, "confidence": str, "reason": str}
        )

        if result:
            logger.info(f"[classify_demuc_with_llm] DEMUC classification successful: {result}")
            return result
        else:
            logger.warning("[classify_demuc_with_llm] Failed to parse LLM response")
            return {"demuc": "", "confidence": "low"}

    except APIOverloadException as e:
        logger.warning(f"[classify_demuc_with_llm] API overloaded: {e}")
        return {"demuc": "", "confidence": "low", "api_overload": True}
    except Exception as e:
        logger.warning(f"[classify_demuc_with_llm] Classification failed: {e}")
        return {"demuc": "", "confidence": "low"}


def classify_chu_de_con_with_llm(
    query: str,
    demuc: str,
    chu_de_con_list_str: str
) -> Dict[str, Any]:
    """
    Step 2: Call LLM to classify CHU_DE_CON (subtopic) within a DEMUC.

    Input:
        - query (str): User's question
        - demuc (str): Already classified DEMUC (e.g., "BỆNH LÝ ĐTĐ")
        - chu_de_con_list_str (str): Formatted CHU_DE_CON list for this DEMUC

    Output:
        Dict with keys: chu_de_con, confidence, reason, api_overload (optional)

    Necessity: Used by TopicClassifyAgent for STEP 2 - finding CHU_DE_CON within DEMUC
    """
    try:
        prompt = f"""
Bạn là trợ lý y khoa. Đã xác định được DEMUC="{demuc}".

Câu hỏi của người dùng: "{query}"
DEMUC hiện tại: "{demuc}"

Danh sách CHU_DE_CON (chủ đề con) có sẵn trong DEMUC "{demuc}":
{chu_de_con_list_str}

NHIỆM VỤ: Chọn CHU_DE_CON phù hợp nhất từ danh sách trên.

YÊU CẦU:
- chu_de_con: chọn CHÍNH XÁC một CHU_DE_CON từ danh sách (viết đúng y hệt)
- confidence: high/medium/low
- reason: lý do ngắn gọn

VÍ DỤ:
Input: "Triệu chứng của đái tháo đường là gì?"
DEMUC: "BỆNH LÝ ĐTĐ"
Danh sách: ["Định nghĩa và phân loại", "Triệu chứng", "Chẩn đoán"]
Output:
```yaml
chu_de_con: "Triệu chứng"
confidence: "high"
reason: "Câu hỏi rõ ràng về triệu chứng"
```

Trả về CHỈ một code block YAML hợp lệ:

```yaml
chu_de_con: "TÊN CHỦ ĐỀ CON"
confidence: "high"
reason: "Lý do"
```
"""

        logger.info(f"[classify_chu_de_con_with_llm] Calling LLM to classify CHU_DE_CON for DEMUC='{demuc}'")

        resp = call_llm(prompt, fast_mode=True, max_retry_time=timeout_config.LLM_RETRY_TIMEOUT)
        logger.info(f"[classify_chu_de_con_with_llm] LLM response received")

        result = parse_yaml_with_schema(
            resp,
            required_fields=["chu_de_con"],
            optional_fields=["confidence", "reason"],
            field_types={"chu_de_con": str, "confidence": str, "reason": str}
        )

        if result:
            logger.info(f"[classify_chu_de_con_with_llm] CHU_DE_CON classification successful: {result}")
            return result
        else:
            logger.warning("[classify_chu_de_con_with_llm] Failed to parse LLM response")
            return {"chu_de_con": "", "confidence": "low"}

    except APIOverloadException as e:
        logger.warning(f"[classify_chu_de_con_with_llm] API overloaded: {e}")
        return {"chu_de_con": "", "confidence": "low", "api_overload": True}
    except Exception as e:
        logger.warning(f"[classify_chu_de_con_with_llm] Classification failed: {e}")
        return {"chu_de_con": "", "confidence": "low"}


if __name__ == "__main__":
    # Test the utility functions
    print("=" * 80)
    print("Testing topic classification utility functions")
    print("=" * 80)

    # Mock data
    mock_demuc_list = ["BỆNH LÝ ĐTĐ", "DINH DƯỠNG", "ĐIỀU TRỊ"]
    mock_chu_de_con_list = ["Định nghĩa và phân loại", "Triệu chứng", "Chẩn đoán"]

    from utils.knowledge_base.metadata_utils import (
        format_demuc_list_for_prompt,
        format_chu_de_con_list_for_prompt
    )

    demuc_list_str = format_demuc_list_for_prompt(mock_demuc_list)
    chu_de_con_list_str = format_chu_de_con_list_for_prompt(mock_chu_de_con_list)

    # Test 1: Classify DEMUC
    print("\nTest 1: Classify DEMUC")
    print(f"Query: 'Tôi muốn hỏi về bệnh đái tháo đường type 2'")
    result = classify_demuc_with_llm(
        query="Tôi muốn hỏi về bệnh đái tháo đường type 2",
        role="patient_diabetes",
        demuc_list_str=demuc_list_str
    )
    print(f"Result: {result}")

    # Test 2: Classify CHU_DE_CON
    print("\nTest 2: Classify CHU_DE_CON")
    print(f"Query: 'Triệu chứng là gì?'")
    print(f"DEMUC: 'BỆNH LÝ ĐTĐ'")
    result = classify_chu_de_con_with_llm(
        query="Triệu chứng là gì?",
        demuc="BỆNH LÝ ĐTĐ",
        chu_de_con_list_str=chu_de_con_list_str
    )
    print(f"Result: {result}")
