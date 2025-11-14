"""
Test RetrieveFromKB node với Qdrant hybrid search.
Sử dụng mock data - KHÔNG CẦN Qdrant server để test node logic.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Mock database before imports
import types
mock_database = types.ModuleType('database')
mock_models = types.ModuleType('models')
mock_models.Users = type('Users', (), {})
mock_database.models = mock_models
mock_db = types.ModuleType('db')
mock_db.get_db = lambda: None
mock_database.db = mock_db

sys.modules['database'] = mock_database
sys.modules['database.db'] = mock_db
sys.modules['database.models'] = mock_models

from unittest.mock import patch


def test_retrieve_without_filters():
    """
    Test retrieval KHÔNG có filter (DEMUC, CHU_DE_CON)
    """
    from core.nodes.medical_nodes import RetrieveFromKB

    print("\n" + "=" * 80)
    print("TEST 1: Retrieval KHÔNG CÓ filters")
    print("=" * 80)

    # Shared state - không có DEMUC/CHU_DE_CON
    shared = {
        "query": "Tại sao tiểu đường nguy hiểm?",
        "demuc": "",
        "chu_de_con": "",
        "rag_state": "expanded"
    }

    # Mock Qdrant results
    mock_results = [
        {
            "id": 1,
            "score": 22.376976,
            "DEMUC": "BỆNH ĐÁI THÁO ĐƯỜNG",
            "CHUDECON": "Định nghĩa",
            "CAUHOI": "Vì sao bệnh đái tháo đường lại nguy hiểm?",
            "CAUTRALOI": "Nguy hiểm vì đường huyết cao không gây triệu chứng rõ ràng...",
            "GIAITHICH": "Đái tháo đường nguy hiểm vì..."
        },
        {
            "id": 2,
            "score": 21.5,
            "DEMUC": "BỆNH ĐÁI THÁO ĐƯỜNG",
            "CHUDECON": "Biến chứng",
            "CAUHOI": "Biến chứng của tiểu đường là gì?",
            "CAUTRALOI": "Biến chứng bao gồm...",
            "GIAITHICH": ""
        }
    ]

    print(f"\n📝 Input:")
    print(f"  Query: {shared['query']}")
    print(f"  DEMUC: '{shared['demuc']}' (trống)")
    print(f"  CHU_DE_CON: '{shared['chu_de_con']}' (trống)")

    # Mock Qdrant utility
    with patch('utils.knowledge_base.qdrant_retrieval.retrieve_from_qdrant') as mock_retrieve:
        mock_retrieve.return_value = mock_results

        # Run node
        node = RetrieveFromKB()
        action = node.run(shared)

        print(f"\n✅ Output:")
        print(f"  Retrieved: {len(shared['question_retrieved_list'])} results")
        print(f"  Top score: {shared['retrieval_score']:.4f}")
        print(f"  Action: {action}")

        # Show first result
        if shared["question_retrieved_list"]:
            first = shared["question_retrieved_list"][0]
            print(f"\n  First result:")
            print(f"    Q: {first['CAUHOI']}")
            print(f"    A: {first['CAUTRALOI'][:80]}...")
            print(f"    DEMUC: {first['DEMUC']}")
            print(f"    Score: {first['score']:.4f}")

        # Verify
        assert len(shared["question_retrieved_list"]) == 2
        assert shared["retrieval_score"] == 22.376976
        assert shared["question_retrieved_list"][0]["CAUHOI"] == "Vì sao bệnh đái tháo đường lại nguy hiểm?"

        # Verify utility was called correctly
        mock_retrieve.assert_called_once_with(
            query="Tại sao tiểu đường nguy hiểm?",
            demuc=None,
            chu_de_con=None,
            top_k=20
        )

        print("\n✅ Test PASSED - Retrieval without filters works!")

    return shared


def test_retrieve_with_demuc_filter():
    """
    Test retrieval CÓ filter DEMUC
    """
    from core.nodes.medical_nodes import RetrieveFromKB

    print("\n" + "=" * 80)
    print("TEST 2: Retrieval VỚI DEMUC filter")
    print("=" * 80)

    # Shared state - có DEMUC
    shared = {
        "query": "Triệu chứng là gì?",
        "demuc": "BỆNH ĐÁI THÁO ĐƯỜNG",
        "chu_de_con": "",
        "rag_state": "classified"
    }

    # Mock Qdrant results (filtered by DEMUC)
    mock_results = [
        {
            "id": 10,
            "score": 23.5,
            "DEMUC": "BỆNH ĐÁI THÁO ĐƯỜNG",
            "CHUDECON": "Triệu chứng",
            "CAUHOI": "Triệu chứng của đái tháo đường là gì?",
            "CAUTRALOI": "Triệu chứng bao gồm khát nước nhiều, tiểu nhiều...",
            "GIAITHICH": ""
        }
    ]

    print(f"\n📝 Input:")
    print(f"  Query: {shared['query']}")
    print(f"  DEMUC: '{shared['demuc']}'")
    print(f"  CHU_DE_CON: '{shared['chu_de_con']}' (trống)")

    with patch('utils.knowledge_base.qdrant_retrieval.retrieve_from_qdrant') as mock_retrieve:
        mock_retrieve.return_value = mock_results

        node = RetrieveFromKB()
        action = node.run(shared)

        print(f"\n✅ Output:")
        print(f"  Retrieved: {len(shared['question_retrieved_list'])} results")
        print(f"  All results from DEMUC: {all(r['DEMUC'] == 'BỆNH ĐÁI THÁO ĐƯỜNG' for r in shared['question_retrieved_list'])}")

        # Verify utility was called with DEMUC filter
        mock_retrieve.assert_called_once_with(
            query="Triệu chứng là gì?",
            demuc="BỆNH ĐÁI THÁO ĐƯỜNG",
            chu_de_con=None,
            top_k=20
        )

        print("\n✅ Test PASSED - DEMUC filter works!")

    return shared


def test_retrieve_with_both_filters():
    """
    Test retrieval CÓ cả DEMUC và CHU_DE_CON filters
    """
    from core.nodes.medical_nodes import RetrieveFromKB

    print("\n" + "=" * 80)
    print("TEST 3: Retrieval VỚI CẢ DEMUC và CHU_DE_CON filters")
    print("=" * 80)

    # Shared state - có cả DEMUC và CHU_DE_CON
    shared = {
        "query": "Làm sao phát hiện sớm?",
        "demuc": "BỆNH ĐÁI THÁO ĐƯỜNG",
        "chu_de_con": "Chẩn đoán",
        "rag_state": "classified"
    }

    # Mock Qdrant results (filtered by both)
    mock_results = [
        {
            "id": 20,
            "score": 24.0,
            "DEMUC": "BỆNH ĐÁI THÁO ĐƯỜNG",
            "CHUDECON": "Chẩn đoán",
            "CAUHOI": "Làm sao phát hiện sớm đái tháo đường?",
            "CAUTRALOI": "Phát hiện sớm qua xét nghiệm đường huyết định kỳ...",
            "GIAITHICH": ""
        }
    ]

    print(f"\n📝 Input:")
    print(f"  Query: {shared['query']}")
    print(f"  DEMUC: '{shared['demuc']}'")
    print(f"  CHU_DE_CON: '{shared['chu_de_con']}'")

    with patch('utils.knowledge_base.qdrant_retrieval.retrieve_from_qdrant') as mock_retrieve:
        mock_retrieve.return_value = mock_results

        node = RetrieveFromKB()
        action = node.run(shared)

        print(f"\n✅ Output:")
        print(f"  Retrieved: {len(shared['question_retrieved_list'])} results")
        print(f"  Matching filters: DEMUC='{mock_results[0]['DEMUC']}', CHU_DE_CON='{mock_results[0]['CHUDECON']}'")

        # Verify utility was called with both filters
        mock_retrieve.assert_called_once_with(
            query="Làm sao phát hiện sớm?",
            demuc="BỆNH ĐÁI THÁO ĐƯỜNG",
            chu_de_con="Chẩn đoán",
            top_k=20
        )

        print("\n✅ Test PASSED - Both filters work!")

    return shared


def test_empty_results():
    """
    Test xử lý khi không có kết quả
    """
    from core.nodes.medical_nodes import RetrieveFromKB

    print("\n" + "=" * 80)
    print("TEST 4: Xử lý khi KHÔNG CÓ kết quả")
    print("=" * 80)

    shared = {
        "query": "Some unrelated query",
        "demuc": "",
        "chu_de_con": "",
        "rag_state": "expanded"
    }

    # Mock empty results
    mock_results = []

    print(f"\n📝 Input:")
    print(f"  Query: {shared['query']}")

    with patch('utils.knowledge_base.qdrant_retrieval.retrieve_from_qdrant') as mock_retrieve:
        mock_retrieve.return_value = mock_results

        node = RetrieveFromKB()
        action = node.run(shared)

        print(f"\n✅ Output:")
        print(f"  Retrieved: {len(shared['question_retrieved_list'])} results")
        print(f"  Top score: {shared['retrieval_score']}")

        # Verify
        assert len(shared["question_retrieved_list"]) == 0
        assert shared["retrieval_score"] == 0.0

        print("\n✅ Test PASSED - Empty results handled correctly!")

    return shared


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("TESTING RetrieveFromKB with Qdrant")
    print("=" * 80)

    try:
        # Run all tests
        test_retrieve_without_filters()
        test_retrieve_with_demuc_filter()
        test_retrieve_with_both_filters()
        test_empty_results()

        print("\n" + "=" * 80)
        print("ALL TESTS PASSED!")
        print("=" * 80)
        print("\nKết luận:")
        print("- RetrieveFromKB hoạt động đúng với Qdrant")
        print("- KHÔNG CẦN Qdrant server để test node logic")
        print("- Filter theo DEMUC/CHU_DE_CON hoạt động")
        print("- Output đúng format: shared['question_retrieved_list']")

    except AssertionError as e:
        print(f"\nTEST FAILED: {e}")
        import traceback
        traceback.print_exc()
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
