#!/usr/bin/env python3
"""
Test script to reproduce the ORTHODONTIST import error
"""

print("Testing imports...")

try:
    print("1. Testing utils.role_enum...")
    from utils.role_enum import RoleEnum, ROLE_DISPLAY_NAME, ROLE_DESCRIPTION, PERSONA_BY_ROLE
    print(f"   ✅ RoleEnum imported successfully")
    print(f"   ✅ All roles: {[r.value for r in RoleEnum]}")
    print(f"   ✅ ORTHODONTIST value: {RoleEnum.ORTHODONTIST.value}")
    
except Exception as e:
    print(f"   ❌ Error importing role_enum: {e}")
    import traceback
    traceback.print_exc()

try:
    print("2. Testing core.flows...")
    from core.flows import create_med_agent_flow, create_oqa_orthodontist_flow
    print(f"   ✅ Flow functions imported successfully")
    
except Exception as e:
    print(f"   ❌ Error importing flows: {e}")
    import traceback
    traceback.print_exc()

try:
    print("3. Testing flow creation...")
    med_flow = create_med_agent_flow()
    print(f"   ✅ Medical flow created successfully")
    
except Exception as e:
    print(f"   ❌ Error creating medical flow: {e}")
    import traceback
    traceback.print_exc()

try:
    print("4. Testing OQA flow creation...")
    oqa_flow = create_oqa_orthodontist_flow()
    print(f"   ✅ OQA flow created successfully")
    
except Exception as e:
    print(f"   ❌ Error creating OQA flow: {e}")
    import traceback
    traceback.print_exc()

try:
    print("5. Testing app import...")
    from app import app
    print(f"   ✅ App imported successfully")
    
except Exception as e:
    print(f"   ❌ Error importing app: {e}")
    import traceback
    traceback.print_exc()

print("Test completed!")
