"""
Utility functions for knowledge base metadata operations.

These are external utility functions that can be easily tested and mocked.
According to PocketFlow best practices, these should be independent and
have clear input/output contracts.
"""

import logging
from typing import Dict, Any, List
from pathlib import Path
import pandas as pd

logger = logging.getLogger(__name__)


def get_demuc_list_for_role(role: str) -> List[str]:
    """
    Get list of DEMUC (topics) available for a role.

    READS DIRECTLY FROM CSV based on role mapping (same source as Qdrant data).

    Input: role (str) - e.g., "patient_diabetes", "patient_dental"
    Output: List of DEMUC names for that role's CSV file

    Example output:
    ["BỆNH ĐÁI THÁO ĐƯỜNG", "BỆNH LÝ RĂNG MIỆNG", ...]

    Necessity: Used by TopicClassifyAgent for FULL classification
               to show LLM all available DEMUC options
    """
    try:
        from utils.role_enum import ROLE_TO_CSV

        # Get CSV file for this role
        csv_file = ROLE_TO_CSV.get(role,"")

        if not csv_file:
            logger.warning(f"No CSV file mapping found for role '{role}'")
            return []

        # Read directly from role-specific CSV (same source as Qdrant)
        # Build a robust absolute path so it works regardless of CWD (tests, docker, IDE)
        project_root = Path(__file__).resolve().parents[2]
        csv_path = project_root / "medical_knowledge_base" / csv_file
        df = pd.read_csv(str(csv_path), encoding="utf-8-sig")
        logger.info(csv_path)
        # Get unique DEMUC list
        demuc_list =  df["DEMUC"].dropna().unique().tolist() 
        logger.info("test:",(df["DEMUC"].unique().tolist()))
        logger.info(f"Loaded {len(demuc_list)} DEMUCs from {csv_path} for role '{role}': {demuc_list}")
        return demuc_list

    except Exception as e:
        logger.warning(f"Could not load DEMUC list for role '{role}': {e}")
        return []


def get_chu_de_con_for_demuc(role: str, demuc: str) -> List[str]:
    """
    Get list of CHU_DE_CON (subtopics) for a specific DEMUC within a role.

    READS DIRECTLY FROM CSV based on role mapping (same source as Qdrant data).

    Input:
        - role (str): e.g., "patient_diabetes", "patient_dental"
        - demuc (str): e.g., "BỆNH ĐÁI THÁO ĐƯỜNG"

    Output: List of CHU_DE_CON names

    Example output:
    ["Định nghĩa", "Biến chứng", "ĐTĐ type 1", ...]

    Necessity: Used by TopicClassifyAgent for DEMUC-ONLY classification
               when DEMUC is already known, only need to choose CHU_DE_CON
    """
    try:
        from utils.role_enum import ROLE_TO_CSV

        # Get CSV file for this role
        csv_file = ROLE_TO_CSV.get(role)

        if not csv_file:
            logger.warning(f"No CSV file mapping found for role '{role}'")
            return []

        # Read directly from role-specific CSV (same source as Qdrant)
        # Build a robust absolute path so it works regardless of CWD (tests, docker, IDE)
        project_root = Path(__file__).resolve().parents[2]
        csv_path = project_root / "medical_knowledge_base" / csv_file
        df = pd.read_csv(str(csv_path), encoding="utf-8-sig")

        # Filter for specific DEMUC and get unique CHU_DE_CON list
        filtered_df = df[df["DEMUC"] == demuc]
        chu_de_con_list = sorted(filtered_df["CHUDECON"].unique().tolist())

        logger.info(f"Loaded {len(chu_de_con_list)} CHU_DE_CON for DEMUC '{demuc}' from {csv_file}")
        return chu_de_con_list

    except Exception as e:
        logger.warning(f"Could not load CHU_DE_CON for DEMUC '{demuc}' from role '{role}': {e}")
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
