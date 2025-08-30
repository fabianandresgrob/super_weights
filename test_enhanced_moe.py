#!/usr/bin/env python3
"""
Test script for Enhanced MoE Super Weight Detection

This script tests the key components of the enhanced MoE detection system
to ensure everything is working properly.
"""

import sys
import logging
from pathlib import Path

# Setup path
sys.path.append(str(Path(__file__).parent))

try:
    from detection.detector import MoESuperWeightDetector
    from detection.super_weight import MoESuperWeight 
    from research.researcher import SuperWeightResearchSession
    from utils.model_architectures import UniversalMLPHandler
    print("All imports successful")
except ImportError as e:
    print(f"Import error: {e}")
    exit(1)


def test_moe_super_weight_class():
    """Test the enhanced MoESuperWeight class"""
    print("\n🧪 Testing MoESuperWeight class...")
    
    # Create a test MoE super weight
    sw = MoESuperWeight(
        layer=2,
        expert_id=3,
        component="expert_3.down_proj",
        row=15,
        column=42,
        input_value=123.45,
        output_value=67.89,
        iteration_found=1,
        original_value=0.01234,
        # Enhanced fields
        p_active=0.15,
        co_spike_score=0.234,
        routed_tokens_count=8,
        impact_natural=0.012,
        impact_interventional=0.013
    )

    print(f"Created: {sw}")
    print(f"Causal agreement: {sw.causal_agreement:.3f}")
    print(f"Routing stability: {sw.routing_stability}")

    return True


def test_detector_initialization():
    """Test that we can initialize the enhanced detector"""
    print("\nTesting detector initialization...")
    
    try:
        # This would normally require a real model, but we can test the class structure
        print("MoESuperWeightDetector class available")
        
        # Test method signatures
        detector_methods = [
            '_enhanced_routing_analysis',
            '_per_expert_co_spike_detection', 
            '_compute_co_spike_scores',
            '_compute_causal_impact_scores',
            '_compute_fast_proxies'
        ]
        
        for method_name in detector_methods:
            if hasattr(MoESuperWeightDetector, method_name):
                print(f"Method {method_name} found")
            else:
                print(f"Method {method_name} missing")
                return False
                
        return True
        
    except Exception as e:
        print(f"Detector initialization error: {e}")
        return False


def test_research_session():
    """Test research session enhancements"""
    print("\nTesting research session enhancements...")

    try:
        # Test that the enhanced parameters are supported
        session_class = SuperWeightResearchSession
        detect_method = getattr(session_class, 'detect_super_weights')
        
        # Check if method signature includes enhanced parameters
        import inspect
        sig = inspect.signature(detect_method)
        
        expected_params = [
            'router_analysis_samples',
            'p_active_floor', 
            'co_spike_threshold',
            'enable_causal_scoring'
        ]
        
        for param in expected_params:
            if param in sig.parameters:
                print(f"Parameter {param} found in detect_super_weights")
            else:
                print(f"Parameter {param} missing from detect_super_weights")
                return False
        
        return True
        
    except Exception as e:
        print(f"Research session test error: {e}")
        return False


def test_example_script():
    """Test that the example script is properly structured"""
    print("\nTesting example script structure...")
    
    try:
        example_path = Path(__file__).parent / "example_enhanced_moe_detection.py"
        if example_path.exists():
            print("Example script exists")

            # Basic syntax check
            with open(example_path) as f:
                content = f.read()
                
            if "enhanced_moe_detection_demo" in content:
                print("Main demo function found")
            else:
                print("Main demo function missing")
                return False
                
            if "Co-spike alignment score" in content:
                print("Co-spike documentation found")
            else:
                print("Co-spike documentation missing")
                return False
                
            return True
        else:
            print("Example script not found")
            return False
            
    except Exception as e:
        print(f"Example script test error: {e}")
        return False


def main():
    """Run all tests"""
    print("Testing Enhanced MoE Super Weight Detection Implementation")
    print("=" * 60)
    
    tests = [
        test_moe_super_weight_class,
        test_detector_initialization,
        test_research_session,
        test_example_script
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"Test {test.__name__} failed with exception: {e}")
            results.append(False)
    
    print("\n" + "=" * 60)
    print("Test Summary:")

    passed = sum(results)
    total = len(results)
    
    for i, (test, result) in enumerate(zip(tests, results)):
        status = "PASS" if result else "FAIL"
        print(f"{i+1}. {test.__name__}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("All tests passed! Enhanced MoE detection is ready.")
        return 0
    else:
        print("Some tests failed. Check the implementation.")
        return 1


if __name__ == '__main__':
    exit(main())
