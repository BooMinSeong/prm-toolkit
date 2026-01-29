#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Example usage of the unified PRM server architecture.

This script demonstrates how to use the PrmServer classes with both
Qwen and Skywork PRM models, supporting both single and batch scoring.

Usage:
    # For Qwen PRM (start server first):
    # Terminal 1: vllm serve Qwen/Qwen2.5-Math-PRM-7B --port 8080 --trust-remote-code
    # Terminal 2 (single): python example_prm_usage.py --model qwen
    # Terminal 2 (batch):  python example_prm_usage.py --model qwen --batch

    # For Skywork PRM (start server first):
    # Terminal 1: python start_reward_server.py
    # Terminal 2 (single): python example_prm_usage.py --model skywork
    # Terminal 2 (batch):  python example_prm_usage.py --model skywork --batch
"""

import argparse
from prm_server import PrmConfig, load_prm_server


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
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Run batch scoring example instead of single scoring"
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
        prm_path=model_name,
        base_url=base_url,
        timeout=300,
        trust_remote_code=True
    )

    print(f"\nConfiguration:")
    print(f"  Model: {config.prm_path}")
    print(f"  Base URL: {config.base_url}")
    print(f"  Timeout: {config.timeout}s")

    # Create PRM server instance
    print(f"\nInitializing PRM server...")
    prm = load_prm_server(config)
    print(f"  Model type: {prm.model_type}")

    # Run batch or single scoring based on argument
    if args.batch:
        run_batch_example(prm, prompt, response, args.model, base_url)
    else:
        run_single_example(prm, prompt, response, base_url)


def run_single_example(prm, prompt, response, base_url):
    """Run single scoring example"""
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


def run_batch_example(prm, example_prompt, example_response, model_type, base_url):
    """Run batch scoring example with multiple prompt-response pairs"""
    print(f"\n{'=' * 80}")
    print("BATCH SCORING EXAMPLE")
    print("=" * 80)

    # Create multiple test cases
    if model_type == "qwen":
        # Qwen examples use double newline as delimiter
        prompts = [
            example_prompt,  # Original flamingo problem
            "What is 15 + 27?",
            "If a book costs $12 and you buy 3 books, how much do you spend?",
        ]*10
        responses = [
            example_response,  # Original flamingo response
            "First, I need to add 15 and 27.\n\n15 + 27 = 42\n\nTherefore, the answer is \\boxed{42}.\n\n"*30,
            "Step 1: Identify the cost per book: $12\n\nStep 2: Multiply by the number of books: $12 × 3 = $36\n\nStep 3: The total cost is \\boxed{36} dollars.\n\n"*30,
        ]*10
    else:  # skywork
        # Skywork examples use single newline as delimiter
        prompts = [
            example_prompt,  # Original Janet's ducks problem
            "What is 15 + 27?",
            "If a book costs $12 and you buy 3 books, how much do you spend?",
        ]
        responses = [
            example_response,  # Original Janet's ducks response
            "Step 1: Add the two numbers: 15 + 27\nStep 2: Calculate: 15 + 27 = 42\nStep 3: The answer is 42",
            "Step 1: Cost per book is $12\nStep 2: Number of books is 3\nStep 3: Total cost: $12 × 3 = $36\nStep 4: Answer is $36",
        ]

    print(f"Processing {len(prompts)} prompt-response pairs...")
    print(f"Using single batch API call for efficiency\n")

    # Display all problems briefly
    for i, (prompt, response) in enumerate(zip(prompts, responses), 1):
        print(f"Problem {i}:")
        print(f"  Prompt: {prompt[:60]}{'...' if len(prompt) > 60 else ''}")
        print(f"  Response: {response[:60]}{'...' if len(response) > 60 else ''}")

    # Score batch
    print(f"\n{'=' * 80}")
    print("BATCH SCORING")
    print("=" * 80)
    print("Sending batch request to PRM server...")

    try:
        batch_rewards = prm.score_batch(prompts=prompts, responses=responses)

        print(f"\n{'=' * 80}")
        print("BATCH RESULTS")
        print("=" * 80)
        print(f"Successfully scored {len(batch_rewards)} responses\n")

        # Display results for each item
        for i, (prompt, response, rewards) in enumerate(zip(prompts, responses, batch_rewards), 1):
            print(f"{'─' * 80}")
            print(f"Result {i}/{len(batch_rewards)}")
            print(f"{'─' * 80}")
            print(f"Prompt: {prompt[:80]}{'...' if len(prompt) > 80 else ''}")
            print(f"\nNumber of steps: {len(rewards)}")
            print("Step-wise rewards:")

            for j, reward in enumerate(rewards, 1):
                print(f"  Step {j}: {reward:.6f}")

            avg_reward = sum(rewards) / len(rewards) if rewards else 0
            print(f"\nAverage reward: {avg_reward:.6f}")
            print(f"Min reward: {min(rewards):.6f}")
            print(f"Max reward: {max(rewards):.6f}")
            print()

        # Summary statistics
        print(f"{'=' * 80}")
        print("BATCH SUMMARY")
        print("=" * 80)
        all_avg_rewards = [sum(r)/len(r) if r else 0 for r in batch_rewards]
        print(f"Overall average reward: {sum(all_avg_rewards)/len(all_avg_rewards):.6f}")
        print(f"Best performing response: Problem {all_avg_rewards.index(max(all_avg_rewards)) + 1} ({max(all_avg_rewards):.6f})")
        print(f"Worst performing response: Problem {all_avg_rewards.index(min(all_avg_rewards)) + 1} ({min(all_avg_rewards):.6f})")

        print(f"\n{'=' * 80}")
        print("SUCCESS")
        print("=" * 80)
        print("Batch processing completed successfully!")
        print(f"Network efficiency: 1 API call for {len(prompts)} inputs")

    except Exception as e:
        print(f"\n{'=' * 80}")
        print("ERROR")
        print("=" * 80)
        print(f"Failed to score batch: {e}")
        print("\nTroubleshooting:")
        print(f"  1. Ensure the vLLM server is running at {base_url}")
        print(f"  2. Check that the model is correctly loaded")
        print(f"  3. Verify network connectivity")
        print(f"  4. Check server logs for batch processing errors")


if __name__ == "__main__":
    main()
