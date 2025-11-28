"""
Test to verify that all nodes return dictionaries from prep() and exec()
This helps Langfuse display structured data better than tuples.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.nodes import (
    IngestQuery,
    DecideSummarizeConversationToRetriveOrDirectlyAnswer,
    RagAgent,
    ComposeAnswer,
    QueryCreatingForRetrievalAgent,
    QueryExpandAgent,
    TopicClassifyAgent,
    RetrieveFromKBWithDemuc,
    RetrieveFromKBWithoutDemuc,
    FallbackNode
)


def test_ingest_query_prep_returns_dict():
    """Test IngestQuery.prep() returns dictionary"""
    node = IngestQuery()
    shared = {
        "role": "patient_dental",
        "input": "Test query",
        "conversation_history": []
    }
    result = node.prep(shared)
    
    assert isinstance(result, dict), "prep() should return dict"
    assert "role" in result, "Missing 'role' key"
    assert "user_input" in result, "Missing 'user_input' key"
    assert "conversation_history" in result, "Missing 'conversation_history' key"
    print("✅ IngestQuery.prep() returns dict with correct keys")


def test_compose_answer_prep_returns_dict():
    """Test ComposeAnswer.prep() returns dictionary"""
    node = ComposeAnswer()
    shared = {
        "role": "patient_dental",
        "query": "Test query",
        "retrieval_query": "Test retrieval query",
        "context_summary": "Test context",
        "selected_ids": []
    }
    result = node.prep(shared)
    
    assert isinstance(result, dict), "prep() should return dict"
    assert "role" in result, "Missing 'role' key"
    assert "query" in result, "Missing 'query' key"
    assert "retrieved_qa" in result, "Missing 'retrieved_qa' key"
    assert "context_summary" in result, "Missing 'context_summary' key"
    print("✅ ComposeAnswer.prep() returns dict with correct keys")


def test_rag_agent_prep_returns_dict():
    """Test RagAgent.prep() returns dictionary"""
    node = RagAgent()
    shared = {
        "query": "Test query",
        "rag_state": "init",
        "attempts": 1,
        "selected_questions": "Test questions",
        "context_summary": "Test context",
        "action_history": []
    }
    result = node.prep(shared)
    
    assert isinstance(result, dict), "prep() should return dict"
    assert "query" in result, "Missing 'query' key"
    assert "rag_state" in result, "Missing 'rag_state' key"
    assert "attempts" in result, "Missing 'attempts' key"
    assert "selected_questions" in result, "Missing 'selected_questions' key"
    assert "context_summary" in result, "Missing 'context_summary' key"
    assert "action_history" in result, "Missing 'action_history' key"
    print("✅ RagAgent.prep() returns dict with correct keys")


def test_retrieve_with_demuc_prep_returns_dict():
    """Test RetrieveFromKBWithDemuc.prep() returns dictionary"""
    node = RetrieveFromKBWithDemuc()
    shared = {
        "query": "Test query",
        "demuc": "Test demuc",
        "role": "patient_dental",
        "top_k": 20
    }
    result = node.prep(shared)
    
    assert isinstance(result, dict), "prep() should return dict"
    assert "query" in result, "Missing 'query' key"
    assert "demuc" in result, "Missing 'demuc' key"
    assert "role" in result, "Missing 'role' key"
    assert "top_k" in result, "Missing 'top_k' key"
    print("✅ RetrieveFromKBWithDemuc.prep() returns dict with correct keys")


def test_retrieve_without_demuc_prep_returns_dict():
    """Test RetrieveFromKBWithoutDemuc.prep() returns dictionary"""
    node = RetrieveFromKBWithoutDemuc()
    shared = {
        "query": "Test query",
        "role": "patient_dental",
        "top_k": 20
    }
    result = node.prep(shared)
    
    assert isinstance(result, dict), "prep() should return dict"
    assert "query" in result, "Missing 'query' key"
    assert "role" in result, "Missing 'role' key"
    assert "top_k" in result, "Missing 'top_k' key"
    print("✅ RetrieveFromKBWithoutDemuc.prep() returns dict with correct keys")


def test_fallback_node_returns_dict():
    """Test FallbackNode.prep() and exec() return dictionaries"""
    node = FallbackNode()
    shared = {}
    
    prep_result = node.prep(shared)
    assert isinstance(prep_result, dict), "prep() should return dict"
    print("✅ FallbackNode.prep() returns dict")
    
    exec_result = node.exec(prep_result)
    assert isinstance(exec_result, dict), "exec() should return dict"
    print("✅ FallbackNode.exec() returns dict")


if __name__ == "__main__":
    print("\n🧪 Testing dictionary returns from nodes...\n")
    
    test_ingest_query_prep_returns_dict()
    test_compose_answer_prep_returns_dict()
    test_rag_agent_prep_returns_dict()
    test_retrieve_with_demuc_prep_returns_dict()
    test_retrieve_without_demuc_prep_returns_dict()
    test_fallback_node_returns_dict()
    
    print("\n✅ All tests passed! Nodes now return dictionaries for better Langfuse visualization.\n")

