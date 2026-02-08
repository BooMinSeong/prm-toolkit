#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Validation script for local mode functionality"""

from prm_toolkit import PrmConfig, load_prm_server
import sys


def test_qwen_local():
    """Test Qwen PRM in local mode"""
    print("=" * 80)
    print("Testing Qwen PRM (local mode)")
    print("=" * 80)

    config = PrmConfig(
        prm_path="Qwen/Qwen2.5-Math-PRM-7B",
        use_local_mode=True,
        max_tokens=1024,
        gpu_memory_utilization=0.7,
    )

    print("Loading model into GPU...")
    prm = load_prm_server(config)

    prompt = "What is 2+2?"
    response = "Step 1: Add 2 and 2\n\nStep 2: The result is 4"

    print(f"Prompt: {prompt}")
    print(f"Response: {response}")
    print("\nScoring...")

    rewards = prm.score(prompt, response)

    print("\n✓ Success!")
    print(f"  Steps: {len(rewards)}")
    print(f"  Rewards: {[f'{r:.4f}' for r in rewards]}")
    print(f"  Average: {sum(rewards)/len(rewards):.4f}")

    prm.cleanup()
    print("\n✓ Cleanup complete")


def test_skywork_local():
    """Test Skywork PRM in local mode"""
    print("=" * 80)
    print("Testing Skywork PRM (local mode)")
    print("=" * 80)

    config = PrmConfig(
        prm_path="Skywork/Skywork-o1-Open-PRM-Qwen-2.5-1.5B",
        use_local_mode=True,
        max_tokens=1024,
        gpu_memory_utilization=0.7,
    )

    print("Loading model into GPU...")
    prm = load_prm_server(config)

    prompt = "What is 2+2?"
    response = "Step 1: Add 2 and 2\nStep 2: The result is 4"

    print(f"Prompt: {prompt}")
    print(f"Response: {response}")
    print("\nScoring...")

    rewards = prm.score(prompt, response)

    print("\n✓ Success!")
    print(f"  Steps: {len(rewards)}")
    print(f"  Rewards: {[f'{r:.4f}' for r in rewards]}")
    print(f"  Average: {sum(rewards)/len(rewards):.4f}")

    prm.cleanup()
    print("\n✓ Cleanup complete")


if __name__ == "__main__":
    model = sys.argv[1] if len(sys.argv) > 1 else "qwen"

    if model == "qwen":
        test_qwen_local()
    elif model == "skywork":
        test_skywork_local()
    else:
        print(f"Unknown model: {model}")
        print("Usage: python test_local_mode.py [qwen|skywork]")
        sys.exit(1)
