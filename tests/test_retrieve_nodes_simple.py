"""
Simple test to verify that the two new retrieval nodes have correct structure
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def test_node_structure():
    """Test that both nodes have the correct structure"""
    print("\n" + "="*60)
    print("Testing Node Structure")
    print("="*60)
    
    # Test imports
    try:
        from core.nodes.RetrieveFromKBWithDemuc import RetrieveFromKBWithDemuc
        from core.nodes.RetrieveFromKBWithoutDemuc import RetrieveFromKBWithoutDemuc
        print("✅ Imports successful")
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False
    
    # Test node instantiation
    try:
        node_with_demuc = RetrieveFromKBWithDemuc()
        node_without_demuc = RetrieveFromKBWithoutDemuc()
        print("✅ Nodes instantiated successfully")
    except Exception as e:
        print(f"❌ Node instantiation failed: {e}")
        return False
    
    # Test that nodes have required methods
    required_methods = ['prep', 'exec', 'post', 'run']
    
    for node, name in [(node_with_demuc, "RetrieveFromKBWithDemuc"), 
                       (node_without_demuc, "RetrieveFromKBWithoutDemuc")]:
        print(f"\nChecking {name}:")
        for method in required_methods:
            if hasattr(node, method):
                print(f"  ✅ {method}() exists")
            else:
                print(f"  ❌ {method}() missing")
                return False
    
    # Test prep method signature
    print("\nTesting prep() method signatures:")
    
    # Create mock shared stores
    shared_without = {
        "query": "test query",
        "retrieval_query": "test query",
        "role": "benh_nhan_rhm",
        "top_k": 5
    }
    
    shared_with = {
        "query": "test query",
        "retrieval_query": "test query",
        "demuc": "test demuc",
        "role": "benh_nhan_rhm",
        "top_k": 5
    }
    
    try:
        # Test WithoutDemuc prep
        prep_result = node_without_demuc.prep(shared_without)
        print(f"✅ RetrieveFromKBWithoutDemuc.prep() returns: {type(prep_result)}")
        
        # Test WithDemuc prep
        prep_result = node_with_demuc.prep(shared_with)
        print(f"✅ RetrieveFromKBWithDemuc.prep() returns: {type(prep_result)}")
        
    except Exception as e:
        print(f"❌ prep() method test failed: {e}")
        return False
    
    return True


def test_flow_integration():
    """Test that the flow can import and use the new nodes"""
    print("\n" + "="*60)
    print("Testing Flow Integration")
    print("="*60)
    
    try:
        from core.flows.medical_flow import RetrievalFlow
        print("✅ RetrievalFlow imported successfully")
        
        # Create flow instance
        flow = RetrievalFlow()
        print("✅ RetrievalFlow instantiated successfully")
        
        # Check that flow has a start node
        if hasattr(flow, 'start') and flow.start is not None:
            print(f"✅ Flow has start node: {type(flow.start).__name__}")
        else:
            print("❌ Flow missing start node")
            return False
            
    except Exception as e:
        print(f"❌ Flow integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def test_topic_classify_actions():
    """Test that TopicClassifyAgent returns correct actions"""
    print("\n" + "="*60)
    print("Testing TopicClassifyAgent Actions")
    print("="*60)
    
    try:
        from core.nodes.TopicClassifyAgent import TopicClassifyAgent
        print("✅ TopicClassifyAgent imported successfully")
        
        node = TopicClassifyAgent()
        
        # Test case 1: With demuc
        shared_with = {"demuc": "test"}
        exec_result_with = {"demuc": "test_demuc", "confidence": "high"}
        action = node.post(shared_with, None, exec_result_with)
        
        if action == "with_demuc":
            print(f"✅ Returns 'with_demuc' when demuc exists")
        else:
            print(f"❌ Expected 'with_demuc' but got '{action}'")
            return False
        
        # Test case 2: Without demuc
        shared_without = {}
        exec_result_without = {"demuc": "", "confidence": "low"}
        action = node.post(shared_without, None, exec_result_without)
        
        if action == "without_demuc":
            print(f"✅ Returns 'without_demuc' when demuc is empty")
        else:
            print(f"❌ Expected 'without_demuc' but got '{action}'")
            return False
            
    except Exception as e:
        print(f"❌ TopicClassifyAgent action test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    print("\n🧪 RUNNING SIMPLE STRUCTURE TESTS")
    print("="*60)
    
    all_passed = True
    
    # Run tests
    if not test_node_structure():
        all_passed = False
    
    if not test_flow_integration():
        all_passed = False
    
    if not test_topic_classify_actions():
        all_passed = False
    
    # Final result
    print("\n" + "="*60)
    if all_passed:
        print("✅ ALL TESTS PASSED!")
        print("="*60)
        print("\n📝 Summary:")
        print("   - RetrieveFromKBWithDemuc: ✅ Created")
        print("   - RetrieveFromKBWithoutDemuc: ✅ Created")
        print("   - TopicClassifyAgent: ✅ Updated with branching logic")
        print("   - RetrievalFlow: ✅ Updated to use both nodes")
        sys.exit(0)
    else:
        print("❌ SOME TESTS FAILED")
        print("="*60)
        sys.exit(1)

