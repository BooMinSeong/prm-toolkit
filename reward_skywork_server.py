#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Skywork-o1-Open-PRM using vLLM server + /pooling endpoint (FIXED)
#
# Usage:
#   1. Start vLLM server in separate terminal:
#      python start_reward_server.py
#
#   2. Run this client script:
#      python reward_skywork_server_fixed.py

import argparse
import requests
import time
import json
from typing import List, Dict, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from transformers import AutoTokenizer
from skywork_utils import prepare_input, sigmoid


def derive_step_rewards_from_pooling(pooling_response, reward_flags_list):
    """
    Extract step-wise rewards from vLLM /pooling endpoint response.

    Args:
        pooling_response: JSON response from /pooling endpoint
        reward_flags_list: List of reward flag arrays, where 1 indicates step end position

    Returns:
        List of step-wise rewards for each input
    """
    batch_step_rewards = []

    for item, reward_flags in zip(pooling_response["data"], reward_flags_list):
        # item["data"] is a list of rewards (one per token)
        # Each reward is a list with a single value: [[reward1], [reward2], ...]
        rewards_data = item["data"]

        # Extract rewards only at step positions (where reward_flags == 1)
        step_rewards = []
        for reward_list, flag in zip(rewards_data, reward_flags):
            if flag == 1:
                # Extract the reward value (first element)
                reward_value = reward_list[0]
                # Apply sigmoid to normalize to [0, 1]
                step_rewards.append(sigmoid(reward_value))

        batch_step_rewards.append(step_rewards)

    return batch_step_rewards


def parse_args():
    parser = argparse.ArgumentParser(description="Skywork-o1-Open-PRM client with vLLM server (stress test mode)")
    parser.add_argument(
        "--model",
        type=str,
        default="Skywork/Skywork-o1-Open-PRM-Qwen-2.5-1.5B",
        help="Model name or path"
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default="http://localhost:8081",
        help="vLLM server base URL"
    )
    parser.add_argument(
        "--problems-file",
        type=str,
        default="math_problems_stress_test.json",
        help="JSON file containing math problems"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Number of problems to process in each batch (default: all problems)"
    )
    parser.add_argument(
        "--num-iterations",
        type=int,
        default=1,
        help="Number of times to repeat the test for stress testing"
    )
    parser.add_argument(
        "--concurrent-requests",
        type=int,
        default=1,
        help="Number of concurrent requests to send simultaneously (default: 1 for sequential)"
    )
    parser.add_argument(
        "--show-detailed-results",
        action="store_true",
        help="Show detailed results for each problem"
    )
    return parser.parse_args()


def load_math_problems(file_path: str) -> List[Dict[str, str]]:
    """Load math problems from a JSON file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            problems = json.load(f)
        return problems
    except FileNotFoundError:
        print(f"Error: Problems file '{file_path}' not found.")
        print("Please ensure the file exists or specify a different file with --problems-file")
        exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Failed to parse JSON from '{file_path}': {e}")
        exit(1)


def send_single_request(
    base_url: str,
    input_ids_list: List,
    reward_flags_list: List,
    request_id: int,
    timeout: int = 300
) -> Tuple[bool, float, int, str]:
    """
    Send a single request to the vLLM server.

    Returns:
        Tuple of (success, request_time, request_id, error_message)
    """
    request_start = time.time()
    try:
        response = requests.post(
            f"{base_url}/pooling",
            json={"input": list(input_ids_list)},
            headers={"Content-Type": "application/json"},
            timeout=timeout
        )
        request_time = time.time() - request_start

        if response.status_code != 200:
            return False, request_time, request_id, f"Status {response.status_code}: {response.text[:100]}"

        # Validate response
        pooling_response = response.json()
        step_rewards = derive_step_rewards_from_pooling(pooling_response, reward_flags_list)

        return True, request_time, request_id, ""

    except requests.exceptions.Timeout:
        request_time = time.time() - request_start
        return False, request_time, request_id, f"Timeout after {timeout}s"
    except requests.exceptions.RequestException as e:
        request_time = time.time() - request_start
        return False, request_time, request_id, f"Request error: {str(e)[:100]}"
    except Exception as e:
        request_time = time.time() - request_start
        return False, request_time, request_id, f"Unexpected error: {str(e)[:100]}"


def main(args):
    print("=" * 80)
    print("STRESS TEST MODE - Skywork-o1-Open-PRM Client")
    print("=" * 80)
    print(f"Server URL: {args.base_url}")
    print(f"Problems file: {args.problems_file}")
    print(f"Batch size: {args.batch_size if args.batch_size else 'ALL'}")
    print(f"Iterations: {args.num_iterations}")
    print(f"Concurrent requests: {args.concurrent_requests}")
    print(f"Total requests: {args.num_iterations * args.concurrent_requests}")
    print("=" * 80)

    # Load tokenizer
    print(f"\nLoading tokenizer from {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=True
    )

    # Load math problems from external file
    print(f"Loading problems from {args.problems_file}...")
    all_problems = load_math_problems(args.problems_file)
    print(f"Loaded {len(all_problems)} problems from file")

    # Select batch if specified
    if args.batch_size:
        datas = all_problems[:args.batch_size]
        print(f"Using first {len(datas)} problems for testing")
    else:
        datas = all_problems
        print(f"Using all {len(datas)} problems for testing")

    # Prepare inputs using Skywork's official format
    print(f"\nPreparing {len(datas)} problems...")
    processed_data = [
        prepare_input(
            data["problem"],
            data["response"],
            tokenizer=tokenizer,
            step_token="\n"  # Skywork-o1-Open-PRM uses newlines as step delimiters
        )
        for data in datas
    ]

    # Unpack processed data
    input_ids_list, steps_list, reward_flags_list = zip(*processed_data)

    # Calculate statistics
    total_tokens = sum(len(ids) for ids in input_ids_list)
    total_steps = sum(len(steps) for steps in steps_list)
    print(f"Total tokens: {total_tokens:,}")
    print(f"Total steps: {total_steps}")
    print(f"Average tokens per problem: {total_tokens // len(datas):,}")
    print(f"Average steps per problem: {total_steps // len(datas):.1f}")

    # Performance tracking
    all_request_times = []
    successful_requests = 0
    failed_requests = 0
    error_messages = []

    # Run stress test
    print(f"\n{'=' * 80}")
    print(f"STARTING STRESS TEST")
    print(f"Mode: {'CONCURRENT' if args.concurrent_requests > 1 else 'SEQUENTIAL'}")
    print("=" * 80)

    total_start = time.time()

    # Prepare all requests
    total_requests = args.num_iterations * args.concurrent_requests

    if args.concurrent_requests > 1:
        # Concurrent mode: use ThreadPoolExecutor
        print(f"\nSending {total_requests} requests with {args.concurrent_requests} concurrent workers...")

        with ThreadPoolExecutor(max_workers=args.concurrent_requests) as executor:
            # Submit all requests
            futures = []
            for i in range(total_requests):
                future = executor.submit(
                    send_single_request,
                    args.base_url,
                    input_ids_list,
                    reward_flags_list,
                    i + 1,
                    300  # timeout
                )
                futures.append(future)

            # Collect results as they complete
            completed = 0
            for future in as_completed(futures):
                success, request_time, request_id, error_msg = future.result()
                completed += 1

                if success:
                    successful_requests += 1
                    all_request_times.append(request_time)
                    tokens_per_sec = total_tokens / request_time
                    print(f"  [{completed}/{total_requests}] ✓ Request #{request_id} completed in {request_time:.2f}s ({tokens_per_sec:.1f} tokens/s)")
                else:
                    failed_requests += 1
                    error_messages.append((request_id, error_msg))
                    print(f"  [{completed}/{total_requests}] ✗ Request #{request_id} failed: {error_msg}")

    else:
        # Sequential mode: send requests one by one
        print(f"\nSending {total_requests} requests sequentially...")

        for i in range(total_requests):
            success, request_time, request_id, error_msg = send_single_request(
                args.base_url,
                input_ids_list,
                reward_flags_list,
                i + 1,
                300  # timeout
            )

            if success:
                successful_requests += 1
                all_request_times.append(request_time)
                tokens_per_sec = total_tokens / request_time
                print(f"  [{i + 1}/{total_requests}] ✓ Request #{request_id} completed in {request_time:.2f}s ({tokens_per_sec:.1f} tokens/s)")
            else:
                failed_requests += 1
                error_messages.append((request_id, error_msg))
                print(f"  [{i + 1}/{total_requests}] ✗ Request #{request_id} failed: {error_msg}")

    total_time = time.time() - total_start

    # Print final statistics
    print(f"\n{'=' * 80}")
    print("STRESS TEST RESULTS")
    print("=" * 80)
    print(f"Total requests sent: {total_requests}")
    print(f"Successful requests: {successful_requests}")
    print(f"Failed requests: {failed_requests}")
    print(f"Success rate: {successful_requests / total_requests * 100:.1f}%")
    print(f"\nTotal test duration: {total_time:.2f}s")

    if all_request_times:
        avg_request_time = sum(all_request_times) / len(all_request_times)
        min_request_time = min(all_request_times)
        max_request_time = max(all_request_times)
        avg_throughput = total_tokens / avg_request_time

        print(f"\nRequest Time Statistics:")
        print(f"  Average: {avg_request_time:.2f}s")
        print(f"  Min: {min_request_time:.2f}s")
        print(f"  Max: {max_request_time:.2f}s")
        print(f"  Std dev: {(sum((t - avg_request_time) ** 2 for t in all_request_times) / len(all_request_times)) ** 0.5:.2f}s")

        print(f"\nThroughput Statistics (per request):")
        print(f"  Average: {avg_throughput:.1f} tokens/s")
        print(f"  Average: {len(datas) / avg_request_time:.2f} problems/s")
        print(f"  Peak: {total_tokens / min_request_time:.1f} tokens/s")

        if args.concurrent_requests > 1:
            # Calculate aggregate throughput for concurrent mode
            aggregate_tokens = successful_requests * total_tokens
            aggregate_throughput = aggregate_tokens / total_time
            print(f"\nAggregate Throughput (all concurrent requests):")
            print(f"  Total tokens processed: {aggregate_tokens:,}")
            print(f"  Aggregate throughput: {aggregate_throughput:.1f} tokens/s")
            print(f"  Effective speedup: {aggregate_throughput / avg_throughput:.2f}x")

    if error_messages:
        print(f"\nErrors encountered ({len(error_messages)}):")
        for req_id, error_msg in error_messages[:10]:  # Show first 10 errors
            print(f"  Request #{req_id}: {error_msg}")
        if len(error_messages) > 10:
            print(f"  ... and {len(error_messages) - 10} more errors")

    print("=" * 80)


if __name__ == "__main__":
    args = parse_args()
    main(args)
