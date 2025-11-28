"""
Test script for RetrieveFromKBWithDemuc and RetrieveFromKBWithoutDemuc nodes
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.nodes import (
    RetrieveFromKBWithDemuc,
    RetrieveFromKBWithoutDemuc
)
from utils.role_enum import RoleEnum


def test_retrieve_without_demuc():
    """Test RetrieveFromKBWithoutDemuc - global search only"""
    print("\n" + "="*60)
    print("TEST 1: RetrieveFromKBWithoutDemuc (Global Search)")
    print("="*60)
    
    # Create node
    node = RetrieveFromKBWithoutDemuc()
    
    # Prepare shared store
    shared = {
        "query": "Tôi bị đau răng phải làm sao?",
        "retrieval_query": "Tôi bị đau răng phải làm sao?",
        "role": RoleEnum.PATIENT_DENTAL.value,
        "top_k": 5
    }
    
    # Run node
    print(f"Query: {shared['query']}")
    print(f"Role: {shared['role']}")
    print(f"Top K: {shared['top_k']}")
    
    action = node.run(shared)
    
    print(f"\nAction returned: {action}")
    print(f"Retrieved {len(shared.get('retrieved_candidates', []))} candidates")
    print(f"RAG State: {shared.get('rag_state')}")
    
    # Show top 3 candidates
    if shared.get("retrieved_candidates"):
        print("\nTop 3 candidates:")
        for i, candidate in enumerate(shared["retrieved_candidates"][:3], 1):
            print(f"\n{i}. [ID: {candidate['id']}] Score: {candidate.get('score', 0):.4f}")
            print(f"   Question: {candidate['CAUHOI'][:100]}...")
    
    return shared


def test_retrieve_with_demuc():
    """Test RetrieveFromKBWithDemuc - hybrid search (filtered + global)"""
    print("\n" + "="*60)
    print("TEST 2: RetrieveFromKBWithDemuc (Hybrid Search)")
    print("="*60)
    
    # Create node
    node = RetrieveFromKBWithDemuc()
    
    # Prepare shared store with demuc
    shared = {
        "query": "Tôi bị đau răng phải làm sao?",
        "retrieval_query": "Tôi bị đau răng phải làm sao?",
        "demuc": "Hỏi đáp chung",  # Add demuc filter
        "role": RoleEnum.PATIENT_DENTAL.value,
        "top_k": 5
    }
    
    # Run node
    print(f"Query: {shared['query']}")
    print(f"Demuc: {shared['demuc']}")
    print(f"Role: {shared['role']}")
    print(f"Top K: {shared['top_k']}")
    
    action = node.run(shared)
    
    print(f"\nAction returned: {action}")
    print(f"Retrieved {len(shared.get('retrieved_candidates', []))} candidates")
    print(f"RAG State: {shared.get('rag_state')}")
    
    # Show top 3 candidates
    if shared.get("retrieved_candidates"):
        print("\nTop 3 candidates:")
        for i, candidate in enumerate(shared["retrieved_candidates"][:3], 1):
            print(f"\n{i}. [ID: {candidate['id']}] Score: {candidate.get('score', 0):.4f}")
            print(f"   Question: {candidate['CAUHOI'][:100]}...")
    
    return shared


def compare_results(shared_without, shared_with):
    """Compare results from two retrieval strategies"""
    print("\n" + "="*60)
    print("COMPARISON")
    print("="*60)
    
    candidates_without = shared_without.get("retrieved_candidates", [])
    candidates_with = shared_with.get("retrieved_candidates", [])
    
    print(f"\nWithout Demuc: {len(candidates_without)} candidates")
    print(f"With Demuc: {len(candidates_with)} candidates")
    
    # Check overlap
    ids_without = {c["id"] for c in candidates_without}
    ids_with = {c["id"] for c in candidates_with}
    
    overlap = ids_without & ids_with
    print(f"Overlap: {len(overlap)} candidates")
    print(f"Unique to Without Demuc: {len(ids_without - ids_with)} candidates")
    print(f"Unique to With Demuc: {len(ids_with - ids_without)} candidates")


if __name__ == "__main__":
    try:
        # Test both nodes
        shared_without = test_retrieve_without_demuc()
        shared_with = test_retrieve_with_demuc()
        
        # Compare results
        compare_results(shared_without, shared_with)
        
        print("\n" + "="*60)
        print("✅ ALL TESTS COMPLETED SUCCESSFULLY")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()

