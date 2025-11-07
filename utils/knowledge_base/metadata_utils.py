"""
Utility functions for knowledge base metadata operations.

These are external utility functions that can be easily tested and mocked.
According to PocketFlow best practices, these should be independent and
have clear input/output contracts.
"""

import logging
from typing import Dict, Any, List
import pandas as pd

logger = logging.getLogger(__name__)


def get_demuc_list_for_role(role: str) -> List[str]:
    """
    Get list of DEMUC (topics) available for a role.

    Input: role (str) - e.g., "patient_diabetes"
    Output: List of DEMUC names

    Example output:
    ["BỆNH LÝ ĐTĐ", "DINH DƯỠNG", "ĐIỀU TRỊ"]

    Necessity: Used by TopicClassifyAgent for FULL classification
               to show LLM all available DEMUC options
    """
    try:
        from utils.knowledge_base.kb import get_df_metadata_for_role

        df_metadata = get_df_metadata_for_role(role)

        # Get unique DEMUC list
        demuc_list = df_metadata["DEMUC"].unique().tolist()

        logger.info(f"Loaded {len(demuc_list)} DEMUCs for role '{role}': {demuc_list}")
        return demuc_list

    except Exception as e:
        logger.warning(f"Could not load DEMUC list for role '{role}': {e}")
        return []


def get_chu_de_con_for_demuc(role: str, demuc: str) -> List[str]:
    """
    Get list of CHU_DE_CON (subtopics) for a specific DEMUC within a role.

    Input:
        - role (str): e.g., "patient_diabetes"
        - demuc (str): e.g., "BỆNH LÝ ĐTĐ"

    Output: List of CHU_DE_CON names

    Example output:
    ["Định nghĩa và phân loại", "Triệu chứng", "Chẩn đoán"]

    Necessity: Used by TopicClassifyAgent for DEMUC-ONLY classification
               when DEMUC is already known, only need to choose CHU_DE_CON
    """
    try:
        from utils.knowledge_base.kb import get_df_metadata_for_role

        df_metadata = get_df_metadata_for_role(role)

        # Filter for specific DEMUC and get CHU_DE_CON list
        filtered_df = df_metadata[df_metadata["DEMUC"] == demuc]
        chu_de_con_list = filtered_df["CHUDECON"].tolist()

        logger.info(f"Loaded {len(chu_de_con_list)} CHU_DE_CON for DEMUC '{demuc}' in role '{role}'")
        return chu_de_con_list

    except Exception as e:
        logger.warning(f"Could not load CHU_DE_CON for DEMUC '{demuc}' in role '{role}': {e}")
        return []


def format_demuc_list_for_prompt(demuc_list: List[str]) -> str:
    """
    Format DEMUC list as simple string for LLM prompt.

    Input: List of DEMUC names
    Output: formatted string

    Example:
    Input: ["BỆNH LÝ ĐTĐ", "DINH DƯỠNG"]
    Output: "- BỆNH LÝ ĐTĐ\n- DINH DƯỠNG"

    Necessity: Used to prepare DEMUC list for LLM classification prompt
    """
    return "\n".join([f"- {demuc}" for demuc in demuc_list])


def format_chu_de_con_list_for_prompt(chu_de_con_list: List[str]) -> str:
    """
    Format CHU_DE_CON list as simple string for LLM prompt.

    Input: List of CHU_DE_CON names
    Output: formatted string

    Example:
    Input: ["Định nghĩa và phân loại", "Triệu chứng"]
    Output: "- Định nghĩa và phân loại\n- Triệu chứng"

    Necessity: Used to prepare CHU_DE_CON list for LLM classification prompt
    """
    return "\n".join([f"- {chu_de_con}" for chu_de_con in chu_de_con_list])


if __name__ == "__main__":
    # Test the utility functions
    print("=" * 80)
    print("Testing metadata utility functions")
    print("=" * 80)

    test_role = "patient_diabetes"

    # Test 1: Get DEMUC list
    print(f"\nTest 1: Get DEMUC list for role '{test_role}'")
    demuc_list = get_demuc_list_for_role(test_role)
    print(f"Number of DEMUCs: {len(demuc_list)}")
    print(f"DEMUC list: {demuc_list}")

    if demuc_list:
        # Test formatting
        print("\nFormatted DEMUC list:")
        formatted = format_demuc_list_for_prompt(demuc_list)
        print(formatted)

        # Test 2: Get CHU_DE_CON for first DEMUC
        first_demuc = demuc_list[0]
        print(f"\nTest 2: Get CHU_DE_CON for DEMUC '{first_demuc}'")
        chu_de_con_list = get_chu_de_con_for_demuc(test_role, first_demuc)
        print(f"Number of CHU_DE_CON: {len(chu_de_con_list)}")
        print(f"CHU_DE_CON list: {chu_de_con_list}")

        # Test formatting
        print("\nFormatted CHU_DE_CON list:")
        formatted = format_chu_de_con_list_for_prompt(chu_de_con_list)
        print(formatted)
    else:
        print("No metadata found!")
