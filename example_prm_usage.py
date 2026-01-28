#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Example usage of the unified PRM server architecture.

This script demonstrates how to use the PrmServer classes with both
Qwen and Skywork PRM models.

Usage:
    # For Qwen PRM (start server first):
    # Terminal 1: vllm serve Qwen/Qwen2.5-Math-PRM-7B --port 8080 --trust-remote-code
    # Terminal 2: python example_prm_usage.py --model qwen

    # For Skywork PRM (start server first):
    # Terminal 1: python start_reward_server.py
    # Terminal 2: python example_prm_usage.py --model skywork
"""

import argparse
from prm_server import PrmConfig, create_prm_server


def get_qwen_example():
    """Example problem for Qwen PRM (uses double newline as step delimiter)"""
    prompt = (
        "Sue lives in a fun neighborhood. One weekend, the neighbors decided to "
        "play a prank on Sue. On Friday morning, the neighbors placed 18 pink "
        "plastic flamingos out on Sue's front yard. On Saturday morning, the "
        "neighbors took back one third of the flamingos, painted them white, and "
        "put these newly painted white flamingos back out on Sue's front yard. "
        "Then, on Sunday morning, they added another 18 pink plastic flamingos to "
        "the collection. At noon on Sunday, how many more pink plastic flamingos "
        "were out than white plastic flamingos?"
    )

    # Qwen expects steps separated by double newline
    response = """To find out how many more pink plastic flamingos were out than white plastic flamingos at noon on Sunday, we can break down the problem into steps. First, on Friday, the neighbors start with 18 pink plastic flamingos.

On Saturday, they take back one third of the flamingos. Since there were 18 flamingos, (1/3 \\times 18 = 6) flamingos are taken back. So, they have (18 - 6 = 12) flamingos left in their possession. Then, they paint these 6 flamingos white and put them back out on Sue's front yard. Now, Sue has the original 12 pink flamingos plus the 6 new white ones. Thus, by the end of Saturday, Sue has (12 + 6 = 18) pink flamingos and 6 white flamingos.

On Sunday, the neighbors add another 18 pink plastic flamingos to Sue's front yard. By the end of Sunday morning, Sue has (18 + 18 = 36) pink flamingos and still 6 white flamingos.

To find the difference, subtract the number of white flamingos from the number of pink flamingos: (36 - 6 = 30). Therefore, at noon on Sunday, there were 30 more pink plastic flamingos out than white plastic flamingos. The answer is (\\boxed{30})."""

    return prompt, response


def get_skywork_example():
    """Example problem for Skywork PRM (uses single newline as step delimiter)"""
    prompt = (
        "Janet's ducks lay 16 eggs per day. She eats three for breakfast every "
        "morning and bakes muffins for her friends every day with four. She sells "
        "the remainder at the farmers' market daily for $2 per fresh duck egg. How "
        "much in dollars does she make every day at the farmers' market?"
    )

    # Skywork expects steps separated by single newline
    response = """Step 1: Calculate total eggs laid per day: 16 eggs
Step 2: Calculate eggs used for breakfast: 3 eggs
Step 3: Calculate eggs used for muffins: 4 eggs
Step 4: Calculate total eggs used: 3 + 4 = 7 eggs
Step 5: Calculate remaining eggs to sell: 16 - 7 = 9 eggs
Step 6: Calculate daily earnings: 9 eggs × $2 = $18"""

    return prompt, response


def main():
    parser = argparse.ArgumentParser(description="Example PRM usage")
    parser.add_argument(
        "--model",
        type=str,
        choices=["qwen", "skywork"],
        required=True,
        help="Which PRM model to use"
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="Server base URL (default: http://localhost:8080 for Qwen, http://localhost:8081 for Skywork)"
    )
    args = parser.parse_args()

    # Configure based on model selection
    if args.model == "qwen":
        model_name = "Qwen/Qwen2.5-Math-PRM-7B"
        base_url = args.base_url or "http://localhost:8082"
        prompt, response = get_qwen_example()
        print("=" * 80)
        print("QWEN PRM EXAMPLE")
        print("=" * 80)
    else:  # skywork
        model_name = "Skywork/Skywork-o1-Open-PRM-Qwen-2.5-1.5B"
        base_url = args.base_url or "http://localhost:8081"
        prompt, response = get_skywork_example()
        print("=" * 80)
        print("SKYWORK PRM EXAMPLE")
        print("=" * 80)

    # Create configuration
    config = PrmConfig(
        model=model_name,
        base_url=base_url,
        timeout=300,
        trust_remote_code=True
    )

    print(f"\nConfiguration:")
    print(f"  Model: {config.model}")
    print(f"  Base URL: {config.base_url}")
    print(f"  Timeout: {config.timeout}s")

    # Create PRM server instance
    print(f"\nInitializing PRM server...")
    prm = create_prm_server(config)
    print(f"  Model type: {prm.model_type}")

    # Display problem
    print(f"\n{'=' * 80}")
    print("PROBLEM")
    print("=" * 80)
    print(prompt)

    print(f"\n{'=' * 80}")
    print("RESPONSE")
    print("=" * 80)
    print(response)

    # Score the response
    print(f"\n{'=' * 80}")
    print("SCORING")
    print("=" * 80)
    print("Sending request to PRM server...")

    try:
        rewards = prm.score(prompt=prompt, response=response)

        print(f"\n{'=' * 80}")
        print("RESULTS")
        print("=" * 80)
        print(f"Number of steps: {len(rewards)}")
        print(f"\nStep-wise rewards:")

        for i, reward in enumerate(rewards, 1):
            print(f"  Step {i}: {reward:.6f}")

        avg_reward = sum(rewards) / len(rewards) if rewards else 0
        print(f"\nAverage reward: {avg_reward:.6f}")
        print(f"Min reward: {min(rewards):.6f}")
        print(f"Max reward: {max(rewards):.6f}")

        print(f"\n{'=' * 80}")
        print("SUCCESS")
        print("=" * 80)

    except Exception as e:
        print(f"\n{'=' * 80}")
        print("ERROR")
        print("=" * 80)
        print(f"Failed to score response: {e}")
        print("\nTroubleshooting:")
        print(f"  1. Ensure the vLLM server is running at {base_url}")
        print(f"  2. Check that the model is correctly loaded")
        print(f"  3. Verify network connectivity")


if __name__ == "__main__":
    main()
