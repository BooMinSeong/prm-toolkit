#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Qwen2.5-Math-PRM-7B using vLLM server + /pooling endpoint (STRESS TEST)
#
# Usage:
#   1. Start vLLM server in separate terminal:
#      python start_qwen_prm_server.py
#
#   2. Run this stress test client:
#      python reward_qwen_prm_server_stress.py --problems-file math_problems.json --num-iterations 10
#
# Features:
#   - Load problems from external JSON file
#   - Batch size control
#   - Multiple iterations
#   - Concurrent request support
#   - Detailed performance statistics

import argparse
import requests
import time
import json
from typing import List, Dict, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from transformers import AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Qwen2.5-Math-PRM-7B client with vLLM server (stress test mode)")
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-Math-PRM-7B",
        help="Model name or path"
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default="http://localhost:8082",
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


def prepare_qwen_prm_input(problem: str, response: List[str], system: str, tokenizer) -> Tuple[str, List[int], List[str], int]:
    """
    Prepare input for Qwen PRM with <extra_0> step delimiters.

    Args:
        problem: Question text
        response: List of reasoning steps
        system: System prompt
        tokenizer: Tokenizer instance

    Returns:
        Tuple of (prompt_text, input_ids, steps, num_extra_tokens)
    """
    # Format according to Qwen PRM specification
    answer = "<extra_0>".join(response) + "<extra_0>"
    prompt = f"<im_start>system\n{system}<im_end>\n<im_start>user\n{problem}<im_end>\n<im_start>assistant\n{answer}<im_end><|endoftext|>"

    # Tokenize for statistics (but we'll send text to server)
    input_ids = tokenizer.encode(prompt, add_special_tokens=False)

    # Count <extra_0> tokens for validation
    extra_0_token_id = tokenizer.encode("<extra_0>", add_special_tokens=False)[0]
    num_extra_tokens = sum(1 for token_id in input_ids if token_id == extra_0_token_id)

    return prompt, input_ids, response, num_extra_tokens


def send_single_request(
    base_url: str,
    prompts_list: List[str],
    expected_rewards_count: int,
    request_id: int,
    timeout: int = 300
) -> Tuple[bool, float, int, int, str]:
    """
    Send a single request to the vLLM server.

    Returns:
        Tuple of (success, request_time, request_id, num_rewards, error_message)
    """
    request_start = time.time()
    try:
        response = requests.post(
            f"{base_url}/pooling",
            json={"input": prompts_list},
            headers={"Content-Type": "application/json"},
            timeout=timeout
        )
        request_time = time.time() - request_start

        if response.status_code != 200:
            return False, request_time, request_id, 0, f"Status {response.status_code}: {response.text[:100]}"

        # Validate response
        pooling_response = response.json()

        # Count rewards
        total_rewards = 0
        for item in pooling_response.get("data", []):
            rewards_data = item.get("data", [])
            total_rewards += len(rewards_data)

        # Check if we got the expected number of rewards
        if total_rewards == 0:
            return False, request_time, request_id, 0, "Empty rewards (missing <extra_0> tokens?)"

        return True, request_time, request_id, total_rewards, ""

    except requests.exceptions.Timeout:
        request_time = time.time() - request_start
        return False, request_time, request_id, 0, f"Timeout after {timeout}s"
    except requests.exceptions.RequestException as e:
        request_time = time.time() - request_start
        return False, request_time, request_id, 0, f"Request error: {str(e)[:100]}"
    except Exception as e:
        request_time = time.time() - request_start
        return False, request_time, request_id, 0, f"Unexpected error: {str(e)[:100]}"


def main(args):
    print("=" * 80)
    print("STRESS TEST MODE - Qwen2.5-Math-PRM-7B Client")
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

    # Get default system prompt
    default_system = "Please reason step by step, and put your final answer within \\boxed{}."

    # Prepare inputs using Qwen PRM format
    print(f"\nPreparing {len(datas)} problems...")
    processed_data = []
    for data in datas:
        system = data.get("system", default_system)
        problem = data["problem"]
        response = data["response"] if isinstance(data["response"], list) else [data["response"]]

        prompt_text, input_ids, steps, num_extra_tokens = prepare_qwen_prm_input(
            problem, response, system, tokenizer
        )
        processed_data.append((prompt_text, input_ids, steps, num_extra_tokens))

    # Unpack processed data
    prompts_list, input_ids_list, steps_list, extra_tokens_list = zip(*processed_data)

    # Calculate statistics
    total_tokens = sum(len(ids) for ids in input_ids_list)
    total_steps = sum(len(steps) for steps in steps_list)
    total_extra_tokens = sum(extra_tokens_list)

    print(f"Total tokens: {total_tokens:,}")
    print(f"Total steps: {total_steps}")
    print(f"Total <extra_0> tokens: {total_extra_tokens}")
    print(f"Average tokens per problem: {total_tokens // len(datas):,}")
    print(f"Average steps per problem: {total_steps / len(datas):.1f}")
    print(f"Average <extra_0> per problem: {total_extra_tokens / len(datas):.1f}")

    # Performance tracking
    all_request_times = []
    successful_requests = 0
    failed_requests = 0
    error_messages = []
    total_rewards_received = 0

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
                    list(prompts_list),
                    total_extra_tokens,
                    i + 1,
                    300  # timeout
                )
                futures.append(future)

            # Collect results as they complete
            completed = 0
            for future in as_completed(futures):
                success, request_time, request_id, num_rewards, error_msg = future.result()
                completed += 1

                if success:
                    successful_requests += 1
                    all_request_times.append(request_time)
                    total_rewards_received += num_rewards
                    tokens_per_sec = total_tokens / request_time
                    print(f"  [{completed}/{total_requests}] ✓ Request #{request_id} completed in {request_time:.2f}s ({tokens_per_sec:.1f} tokens/s, {num_rewards} rewards)")
                else:
                    failed_requests += 1
                    error_messages.append((request_id, error_msg))
                    print(f"  [{completed}/{total_requests}] ✗ Request #{request_id} failed: {error_msg}")

    else:
        # Sequential mode: send requests one by one
        print(f"\nSending {total_requests} requests sequentially...")

        for i in range(total_requests):
            success, request_time, request_id, num_rewards, error_msg = send_single_request(
                args.base_url,
                list(prompts_list),
                total_extra_tokens,
                i + 1,
                300  # timeout
            )

            if success:
                successful_requests += 1
                all_request_times.append(request_time)
                total_rewards_received += num_rewards
                tokens_per_sec = total_tokens / request_time
                print(f"  [{i + 1}/{total_requests}] ✓ Request #{request_id} completed in {request_time:.2f}s ({tokens_per_sec:.1f} tokens/s, {num_rewards} rewards)")
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
        avg_rewards_per_request = total_rewards_received / successful_requests if successful_requests > 0 else 0

        print(f"\nRequest Time Statistics:")
        print(f"  Average: {avg_request_time:.2f}s")
        print(f"  Min: {min_request_time:.2f}s")
        print(f"  Max: {max_request_time:.2f}s")
        print(f"  Std dev: {(sum((t - avg_request_time) ** 2 for t in all_request_times) / len(all_request_times)) ** 0.5:.2f}s")

        print(f"\nThroughput Statistics (per request):")
        print(f"  Average: {avg_throughput:.1f} tokens/s")
        print(f"  Average: {len(datas) / avg_request_time:.2f} problems/s")
        print(f"  Peak: {total_tokens / min_request_time:.1f} tokens/s")

        print(f"\nReward Statistics:")
        print(f"  Total rewards received: {total_rewards_received}")
        print(f"  Average rewards per request: {avg_rewards_per_request:.1f}")
        print(f"  Expected <extra_0> tokens: {total_extra_tokens}")

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
