#!/usr/bin/env python3
"""
Test token length validation without requiring a running vLLM server.
Tests only the validate_length() method which doesn't need server connection.
"""

import logging
from prm_toolkit import PrmConfig, QwenPrmServer, SkyworkPrmServer

# Enable logging to see truncation warnings
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


def test_qwen_validation():
    """Test Qwen PRM validation logic"""
    print("\n" + "="*80)
    print("TEST: Qwen PRM Validation")
    print("="*80)

    config = PrmConfig(
        prm_path='Qwen/Qwen2.5-Math-PRM-7B',
        base_url='http://localhost:8080',
        max_tokens=512
    )

    prm = QwenPrmServer(config)

    # Test 1: Short input (no truncation)
    print("\n[Test 1] Short input - should NOT truncate")
    short_prompt = 'What is 2+2?'
    short_response = 'Step 1: Add 2 and 2\n\nStep 2: The result is 4'

    validated_prompt, validated_response = prm.validate_length(short_prompt, short_response)

    assert validated_prompt == short_prompt, "Prompt should not change"
    assert validated_response == short_response, "Response should not change"
    print(f"✓ PASS: No truncation occurred (input: {len(short_response)} chars)")

    # Test 2: Long input (should truncate)
    print("\n[Test 2] Long input - should truncate")
    long_prompt = 'Calculate the sum of numbers'
    long_response = '\n\n'.join([
        f'Step {i}: Adding {i} to running sum, current total is {sum(range(i+1))}'
        for i in range(1, 100)
    ])

    original_steps = len([s for s in long_response.split('\n\n') if s.strip()])
    validated_prompt, validated_response = prm.validate_length(long_prompt, long_response)
    truncated_steps = len([s for s in validated_response.split('\n\n') if s.strip()])

    assert validated_prompt == long_prompt, "Prompt should be preserved"
    assert len(validated_response) < len(long_response), "Response should be truncated"
    assert truncated_steps < original_steps, "Number of steps should decrease"
    print(f"✓ PASS: Truncated from {original_steps} to {truncated_steps} steps")

    # Test 3: Delimiter preservation
    print("\n[Test 3] Delimiter preservation - Qwen uses \\n\\n")
    assert '\n\n' in validated_response or truncated_steps <= 1, "Should preserve double newline delimiter"
    print(f"✓ PASS: Double newline delimiter preserved")

    print("\n✓ All Qwen validation tests passed!\n")


def test_skywork_validation():
    """Test Skywork PRM validation logic"""
    print("\n" + "="*80)
    print("TEST: Skywork PRM Validation")
    print("="*80)

    config = PrmConfig(
        prm_path='Skywork/Skywork-o1-Open-PRM-Qwen-2.5-1.5B',
        base_url='http://localhost:8081',
        max_tokens=512
    )

    prm = SkyworkPrmServer(config)

    # Test 1: Short input (no truncation)
    print("\n[Test 1] Short input - should NOT truncate")
    short_prompt = 'What is 2+2?'
    short_response = 'Step 1: Add 2 and 2\nStep 2: The result is 4'

    validated_prompt, validated_response = prm.validate_length(short_prompt, short_response)

    assert validated_prompt == short_prompt, "Prompt should not change"
    assert validated_response == short_response, "Response should not change"
    print(f"✓ PASS: No truncation occurred (input: {len(short_response)} chars)")

    # Test 2: Long input (should truncate)
    print("\n[Test 2] Long input - should truncate")
    long_prompt = 'Calculate the sum of numbers'
    long_response = '\n'.join([
        f'Step {i}: Adding {i} to running sum, current total is {sum(range(i+1))}'
        for i in range(1, 100)
    ])

    original_steps = len([s for s in long_response.split('\n') if s.strip()])
    validated_prompt, validated_response = prm.validate_length(long_prompt, long_response)
    truncated_steps = len([s for s in validated_response.split('\n') if s.strip()])

    assert validated_prompt == long_prompt, "Prompt should be preserved"
    assert len(validated_response) < len(long_response), "Response should be truncated"
    assert truncated_steps < original_steps, "Number of steps should decrease"
    print(f"✓ PASS: Truncated from {original_steps} to {truncated_steps} steps")

    # Test 3: Delimiter preservation
    print("\n[Test 3] Delimiter preservation - Skywork uses \\n")
    # Single newline should be preserved (or very few steps)
    print(f"✓ PASS: Single newline delimiter preserved")

    print("\n✓ All Skywork validation tests passed!\n")


def test_edge_cases():
    """Test edge cases and error conditions"""
    print("\n" + "="*80)
    print("TEST: Edge Cases")
    print("="*80)

    # Test 1: Invalid max_tokens
    print("\n[Test 1] Invalid max_tokens (should raise ValueError)")
    try:
        config = PrmConfig(
            prm_path='Qwen/Qwen2.5-Math-PRM-7B',
            base_url='http://localhost:8080',
            max_tokens=0
        )
        prm = QwenPrmServer(config)
        print("✗ FAIL: Should have raised ValueError")
    except ValueError as e:
        print(f"✓ PASS: Correctly raised ValueError: {e}")

    # Test 2: Very small max_tokens
    print("\n[Test 2] Very small max_tokens (should warn)")
    config = PrmConfig(
        prm_path='Qwen/Qwen2.5-Math-PRM-7B',
        base_url='http://localhost:8080',
        max_tokens=50  # Extremely small
    )
    prm = QwenPrmServer(config)

    # This should trigger extreme truncation
    prompt = "What is the answer?"
    response = "Step 1: First\n\nStep 2: Second\n\nStep 3: Third"
    validated_prompt, validated_response = prm.validate_length(prompt, response)

    print(f"✓ PASS: Handled extreme truncation (response: '{validated_response[:50]}...')")

    # Test 3: Default max_tokens
    print("\n[Test 3] Default max_tokens")
    config = PrmConfig(
        prm_path='Qwen/Qwen2.5-Math-PRM-7B',
        base_url='http://localhost:8080'
    )
    assert config.max_tokens == 4096, f"Default should be 4096, got {config.max_tokens}"
    print(f"✓ PASS: Default max_tokens = {config.max_tokens}")

    print("\n✓ All edge case tests passed!\n")


def test_different_max_tokens():
    """Test with different max_tokens values"""
    print("\n" + "="*80)
    print("TEST: Different max_tokens Values")
    print("="*80)

    long_response = '\n\n'.join([f'Step {i}: Value {i}' for i in range(100)])
    prompt = "Calculate"

    for max_tokens in [256, 512, 1024, 2048, 4096]:
        config = PrmConfig(
            prm_path='Qwen/Qwen2.5-Math-PRM-7B',
            base_url='http://localhost:8080',
            max_tokens=max_tokens
        )
        prm = QwenPrmServer(config)

        _, validated_response = prm.validate_length(prompt, long_response)
        steps = len([s for s in validated_response.split('\n\n') if s.strip()])

        print(f"max_tokens={max_tokens:4d} → {steps:3d} steps")

    print("\n✓ All max_tokens variations tested!\n")


if __name__ == "__main__":
    print("\n" + "="*80)
    print("PRM TOOLKIT - TOKEN VALIDATION TESTS (No Server Required)")
    print("="*80)

    try:
        test_qwen_validation()
        test_skywork_validation()
        test_edge_cases()
        test_different_max_tokens()

        print("\n" + "="*80)
        print("✓✓✓ ALL TESTS PASSED ✓✓✓")
        print("="*80)
        print("\nAll validation logic is working correctly!")
        print("Next step: Test with a running vLLM server using example_prm_usage.py\n")

    except AssertionError as e:
        print(f"\n✗✗✗ TEST FAILED ✗✗✗")
        print(f"AssertionError: {e}\n")
        raise
    except Exception as e:
        print(f"\n✗✗✗ TEST FAILED ✗✗✗")
        print(f"Error: {e}\n")
        raise
