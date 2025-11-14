"""
Test TopicClassifyAgent với 2-step classification (DEMUC -> CHU_DE_CON)
Sử dụng mock data - KHÔNG CẦN DB hay LLM API
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

# Mock the database dependency before any imports
sys.modules['database'] = type(sys)('database')
sys.modules['database.db'] = type(sys)('database.db')

# Mock get_db function
import types
mock_db = types.ModuleType('mock_db')
mock_db.get_db = lambda: None
sys.modules['database.db'] = mock_db

from unittest.mock import patch, MagicMock
mock_database = types.ModuleType('database')
mock_models = types.ModuleType('models')

mock_models.Users = type('Users', (), {})  # Mock Users class
mock_database.models = mock_models

mock_db = types.ModuleType('db')
mock_db.get_db = lambda: None
mock_database.db = mock_db


sys.modules['database'] = mock_database
sys.modules['database.db'] = mock_db
sys.modules['database.models'] = mock_models

# ===== NOW SAFE TO IMPORT =====
from unittest.mock import patch, MagicMock


def test_step1_classify_demuc():
    """
    Test BƯỚC 1: Phân loại DEMUC khi chưa có DEMUC
    """
    from core.nodes.medical_nodes import TopicClassifyAgent

    print("\n" + "=" * 80)
    print("TEST BƯỚC 1: Phân loại DEMUC (chưa có DEMUC)")
    print("=" * 80)

    # Shared state - CHƯA CÓ DEMUC
    shared = {
        "query": "Tạo sao tiểu đường nghiêm trọng ",
        "role": "patient_diabetes",
        "demuc": "",  # Chưa có
        "chu_de_con": ""
    }

    # Mock data
    mock_demuc_list = ["BỆNH LÝ ĐTĐ", "DINH DƯỠNG", "ĐIỀU TRỊ", "BIẾN CHỨNG"]
    mock_llm_result = {
        "demuc": "BỆNH LÝ ĐTĐ",
        "confidence": "high",
        "reason": "Câu hỏi về bệnh đái tháo đường"
    }

    print(f"\n📝 Input:")
    print(f"  Query: {shared['query']}")
    print(f"  Role: {shared['role']}")
    print(f"  DEMUC hiện tại: '{shared['demuc']}' (trống)")
    print(f"\n🔧 Mock DEMUC list: {mock_demuc_list}")

    # Mock utility functions
    with patch('utils.knowledge_base.metadata_utils.get_demuc_list_for_role') as mock_get_demuc, \
         patch('utils.llm.classify_topic.classify_demuc_with_llm') as mock_classify:

        mock_get_demuc.return_value = mock_demuc_list
        mock_classify.return_value = mock_llm_result

        # Run node
        node = TopicClassifyAgent()
        action = node.run(shared)

        print(f"\n✅ Output:")
        print(f"  DEMUC: '{shared['demuc']}'")
        print(f"  CHU_DE_CON: '{shared['chu_de_con']}' (chưa có)")
        print(f"  Confidence: {shared['classification_confidence']}")
        print(f"  Action: {action}")

        # Verify
        assert shared["demuc"] == "BỆNH LÝ ĐTĐ", "DEMUC should be classified"
        assert shared["chu_de_con"] == "", "CHU_DE_CON should still be empty"
        assert action == "classify_again", "Should route back to classify CHU_DE_CON"

        print("\n✅ Test PASSED - DEMUC classified successfully!")

    return shared


def test_step2_classify_chu_de_con():
    """
    Test BƯỚC 2: Phân loại CHU_DE_CON khi đã có DEMUC
    """
    from core.nodes.medical_nodes import TopicClassifyAgent

    print("\n" + "=" * 80)
    print("TEST BƯỚC 2: Phân loại CHU_DE_CON (đã có DEMUC)")
    print("=" * 80)

    # Shared state - ĐÃ CÓ DEMUC
    shared = {
        "query": "Triệu chứng của bệnh đái tháo đường là gì?",
        "role": "patient_diabetes",
        "demuc": "BỆNH LÝ ĐTĐ",  # Đã có từ bước 1
        "chu_de_con": ""  # Chưa có
    }

    # Mock data
    mock_chu_de_con_list = [
        "Định nghĩa và phân loại",
        "Triệu chứng",
        "Chẩn đoán",
        "Nguyên nhân"
    ]
    mock_llm_result = {
        "chu_de_con": "Triệu chứng",
        "confidence": "high",
        "reason": "Câu hỏi rõ ràng về triệu chứng"
    }

    print(f"\n📝 Input:")
    print(f"  Query: {shared['query']}")
    print(f"  Role: {shared['role']}")
    print(f"  DEMUC hiện tại: '{shared['demuc']}'")
    print(f"  CHU_DE_CON hiện tại: '{shared['chu_de_con']}' (trống)")
    print(f"\n🔧 Mock CHU_DE_CON list cho DEMUC '{shared['demuc']}': {mock_chu_de_con_list}")

    # Mock utility functions
    with patch('utils.knowledge_base.metadata_utils.get_chu_de_con_for_demuc') as mock_get_chudecon, \
         patch('utils.llm.classify_topic.classify_chu_de_con_with_llm') as mock_classify:

        mock_get_chudecon.return_value = mock_chu_de_con_list
        mock_classify.return_value = mock_llm_result

        # Run node
        node = TopicClassifyAgent()
        action = node.run(shared)

        print(f"\n✅ Output:")
        print(f"  DEMUC: '{shared['demuc']}' (giữ nguyên)")
        print(f"  CHU_DE_CON: '{shared['chu_de_con']}'")
        print(f"  Confidence: {shared['classification_confidence']}")
        print(f"  Action: {action}")

        # Verify
        assert shared["demuc"] == "BỆNH LÝ ĐTĐ", "DEMUC should remain same"
        assert shared["chu_de_con"] == "Triệu chứng", "CHU_DE_CON should be classified"
        assert action == "default", "Should route to next node (classification complete)"

        print("\n✅ Test PASSED - CHU_DE_CON classified successfully!")

    return shared


def test_full_2step_flow():
    """
    Test FULL FLOW: 2 bước liên tiếp
    Bước 1: Classify DEMUC
    Bước 2: Classify CHU_DE_CON
    """
    from core.nodes.medical_nodes import TopicClassifyAgent

    print("\n" + "=" * 80)
    print("TEST FULL FLOW: 2 bước classification liên tiếp")
    print("=" * 80)

    # Initial state - chưa có gì
    shared = {
        "query": "Bệnh đái tháo đường là gì?",
        "role": "patient_diabetes",
        "demuc": "",
        "chu_de_con": ""
    }

    # Mock data cho cả 2 bước
    mock_demuc_list = ["BỆNH LÝ ĐTĐ", "DINH DƯỠNG", "ĐIỀU TRỊ"]
    mock_chu_de_con_list = ["Định nghĩa và phân loại", "Triệu chứng", "Chẩn đoán"]

    mock_demuc_result = {
        "demuc": "BỆNH LÝ ĐTĐ",
        "confidence": "high",
        "reason": "Về bệnh đái tháo đường"
    }

    mock_chu_de_con_result = {
        "chu_de_con": "Định nghĩa và phân loại",
        "confidence": "high",
        "reason": "Hỏi về định nghĩa"
    }

    print(f"\n📝 Initial State:")
    print(f"  Query: {shared['query']}")
    print(f"  DEMUC: '{shared['demuc']}' (trống)")
    print(f"  CHU_DE_CON: '{shared['chu_de_con']}' (trống)")

    # Mock all utility functions
    with patch('utils.knowledge_base.metadata_utils.get_demuc_list_for_role') as mock_get_demuc, \
         patch('utils.knowledge_base.metadata_utils.get_chu_de_con_for_demuc') as mock_get_chudecon, \
         patch('utils.llm.classify_topic.classify_demuc_with_llm') as mock_classify_demuc, \
         patch('utils.llm.classify_topic.classify_chu_de_con_with_llm') as mock_classify_chudecon:

        mock_get_demuc.return_value = mock_demuc_list
        mock_get_chudecon.return_value = mock_chu_de_con_list
        mock_classify_demuc.return_value = mock_demuc_result
        mock_classify_chudecon.return_value = mock_chu_de_con_result

        node = TopicClassifyAgent()

        # BƯỚC 1: Classify DEMUC
        print(f"\n" + "-" * 80)
        print("BƯỚC 1: Classify DEMUC")
        print("-" * 80)
        action1 = node.run(shared)

        print(f"  Sau bước 1:")
        print(f"    DEMUC: '{shared['demuc']}'")
        print(f"    CHU_DE_CON: '{shared['chu_de_con']}'")
        print(f"    Action: {action1}")

        assert shared["demuc"] == "BỆNH LÝ ĐTĐ"
        assert shared["chu_de_con"] == ""
        assert action1 == "classify_again"

        # BƯỚC 2: Classify CHU_DE_CON
        print(f"\n" + "-" * 80)
        print("BƯỚC 2: Classify CHU_DE_CON")
        print("-" * 80)
        action2 = node.run(shared)

        print(f"  Sau bước 2:")
        print(f"    DEMUC: '{shared['demuc']}'")
        print(f"    CHU_DE_CON: '{shared['chu_de_con']}'")
        print(f"    Action: {action2}")

        assert shared["demuc"] == "BỆNH LÝ ĐTĐ"
        assert shared["chu_de_con"] == "Định nghĩa và phân loại"
        assert action2 == "default"

        print("\n✅ Test PASSED - Full 2-step flow works correctly!")

    return shared


def test_api_overload_handling():
    """
    Test xử lý khi API overload
    """
    from core.nodes.medical_nodes import TopicClassifyAgent

    print("\n" + "=" * 80)
    print("TEST: Xử lý API Overload")
    print("=" * 80)

    shared = {
        "query": "Test query",
        "role": "patient_diabetes",
        "demuc": "",
        "chu_de_con": ""
    }

    # Mock API overload
    mock_demuc_list = ["BỆNH LÝ ĐTĐ", "DINH DƯỠNG"]
    mock_overload_result = {
        "demuc": "",
        "confidence": "low",
        "api_overload": True
    }

    print(f"\n📝 Simulating API overload...")

    with patch('utils.knowledge_base.metadata_utils.get_demuc_list_for_role') as mock_get_demuc, \
         patch('utils.llm.classify_topic.classify_demuc_with_llm') as mock_classify:

        mock_get_demuc.return_value = mock_demuc_list
        mock_classify.return_value = mock_overload_result

        node = TopicClassifyAgent()
        action = node.run(shared)

        print(f"\n✅ Output:")
        print(f"  DEMUC: '{shared['demuc']}' (trống do overload)")
        print(f"  Action: {action}")

        assert action == "fallback", "Should route to fallback on API overload"
        print("\n✅ Test PASSED - API overload handled correctly!")

    return shared


if __name__ == "__main__":

    try:
        # Run all tests
        # test_step1_classify_demuc()
        test_step2_classify_chu_de_con()
        # test_full_2step_flow()
        # test_api_overload_handling()


    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()