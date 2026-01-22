#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Qwen2.5-Math-PRM-7B using vLLM server + OpenAI client
#
# Usage:
#   1. Start vLLM server in separate terminal:
#      python start_qwen_prm_server.py
#
#      Or manually:
#      vllm serve Qwen/Qwen2.5-Math-PRM-7B \
#          --host 0.0.0.0 \
#          --port 8082 \
#          --tensor-parallel-size 1 \
#          --gpu-memory-utilization 0.9 \
#          --enable-prefix-caching \
#          --dtype auto \
#          --trust-remote-code
#
#   2. Run this client script:
#      python reward_qwen_prm_server.py
#
# Note: Qwen2.5-Math-PRM-7B uses STEP pooling with <extra_0> tokens
#       as step delimiters. Rewards are extracted at <extra_0> positions.

import argparse
import requests


def parse_args():
    parser = argparse.ArgumentParser(description="Qwen2.5-Math-PRM-7B client with vLLM server")
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
        "--api-key",
        type=str,
        default="EMPTY",
        help="API key (use 'EMPTY' for local vLLM server)"
    )
    return parser.parse_args()


def math_step_prompts():
    """Returns math problems with step-by-step solutions in Qwen PRM format"""
    # ruff: noqa: E501
    data = {
        "system": "Please reason step by step, and put your final answer within \\boxed{}. ",
        "query": "Sue lives in a fun neighborhood.  One weekend, the neighbors decided to play a prank on Sue.  On Friday morning, the neighbors placed 18 pink plastic flamingos out on Sue's front yard.  On Saturday morning, the neighbors took back one third of the flamingos, painted them white, and put these newly painted white flamingos back out on Sue's front yard.  Then, on Sunday morning, they added another 18 pink plastic flamingos to the collection. At noon on Sunday, how many more pink plastic flamingos were out than white plastic flamingos?",
        "response": [
            "To find out how many more pink plastic flamingos were out than white plastic flamingos at noon on Sunday, we can break down the problem into steps. First, on Friday, the neighbors start with 18 pink plastic flamingos.",
            "On Saturday, they take back one third of the flamingos. Since there were 18 flamingos, (1/3 \\times 18 = 6) flamingos are taken back. So, they have (18 - 6 = 12) flamingos left in their possession. Then, they paint these 6 flamingos white and put them back out on Sue's front yard. Now, Sue has the original 12 pink flamingos plus the 6 new white ones. Thus, by the end of Saturday, Sue has (12 + 6 = 18) pink flamingos and 6 white flamingos.",
            "On Sunday, the neighbors add another 18 pink plastic flamingos to Sue's front yard. By the end of Sunday morning, Sue has (18 + 18 = 36) pink flamingos and still 6 white flamingos.",
            "To find the difference, subtract the number of white flamingos from the number of pink flamingos: (36 - 6 = 30). Therefore, at noon on Sunday, there were 30 more pink plastic flamingos out than white plastic flamingos. The answer is (\\boxed{30}).",
        ],
    }

    # Format according to Qwen PRM specification:
    # Steps are joined with <extra_0> tokens
    answer = "<extra_0>".join(data["response"]) + "<extra_0>"
    prompt = f"<im_start>system\n{data['system']}<im_end>\n<im_start>user\n{data['query']}<im_end>\n<im_start>assistant\n{answer}<im_end><|endoftext|>"

    return [prompt], [data["response"]]


def main(args):
    print(f"Connecting to vLLM server at {args.base_url}...")

    # Get example math problems
    prompts, steps_list = math_step_prompts()

    print(f"\nPrepared {len(prompts)} problem(s) with <extra_0> step delimiters")
    print(f"Preview: {prompts[0][:200]}...")

    # Get model name from server
    try:
        models_response = requests.get(f"{args.base_url}/v1/models")
        models_response.raise_for_status()
        model_name = models_response.json()["data"][0]["id"]
        print(f"\nUsing model: {model_name}")
    except Exception as e:
        print(f"Warning: Could not get model name from server: {e}")
        model_name = args.model

    # Get embeddings (rewards) from vLLM server using /pooling endpoint
    # Note: For Qwen PRM, the input must already contain <extra_0> tokens
    print("\nRequesting embeddings from vLLM /pooling endpoint...")
    pooling_response = requests.post(
        f"{args.base_url}/pooling",
        json={"input": prompts},
        headers={"Content-Type": "application/json"},
        timeout=300
    )
    pooling_response.raise_for_status()
    embeddings_response = pooling_response.json()

    # Debug: Print raw response
    print("\n" + "=" * 80)
    print("DEBUG: Raw pooling response")
    print("=" * 80)
    print(f"Response type: {type(embeddings_response)}")
    print(f"Response keys: {embeddings_response.keys() if isinstance(embeddings_response, dict) else 'N/A'}")
    if "data" in embeddings_response:
        print(f"Number of data items: {len(embeddings_response['data'])}")
        for idx, item in enumerate(embeddings_response["data"]):
            print(f"\nItem {idx}:")
            print(f"  Type: {type(item)}")
            print(f"  Keys: {item.keys() if isinstance(item, dict) else 'N/A'}")
            if "data" in item:
                rewards = item["data"]
                print(f"  Rewards type: {type(rewards)}")
                print(f"  Rewards length: {len(rewards) if hasattr(rewards, '__len__') else 'N/A'}")
                if hasattr(rewards, '__len__') and len(rewards) > 0:
                    print(f"  Sample values: {rewards[:min(3, len(rewards))]}")
    print("=" * 80)

    # Process results
    # For Qwen PRM with STEP pooling, embeddings correspond to <extra_0> token positions
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)

    for idx, (prompt, steps, embedding_data) in enumerate(zip(prompts, steps_list, embeddings_response["data"])):
        # Extract rewards from /pooling endpoint response
        # Response format: {"data": [{"data": [[reward1, reward2], [reward3, reward4], ...]}]}
        rewards_raw = embedding_data["data"]
        num_steps = len(steps)

        print(f"\n{'=' * 80}")
        print(f"Problem {idx + 1}:")
        print(f"{'=' * 80}")
        print(f"Number of steps: {num_steps}")
        print(f"Number of rewards: {len(rewards_raw)}")

        if len(rewards_raw) == 0:
            print("\n⚠️  WARNING: Empty rewards array!")
            print("    Make sure the input contains <extra_0> tokens.")
            print("    See: https://github.com/vllm-project/vllm/issues/27343")
        elif len(rewards_raw) != num_steps:
            print(f"\n⚠️  WARNING: Reward count ({len(rewards_raw)}) != step count ({num_steps})")
            print("    This may be expected depending on the tokenization.")

        print(f"\nStep-wise rewards:")

        # Match rewards to steps (assuming 1:1 correspondence)
        # Each reward may be a list (e.g., [positive_logit, negative_logit])
        for step_idx in range(min(len(steps), len(rewards_raw))):
            step = steps[step_idx]
            reward_data = rewards_raw[step_idx]
            # If reward is a list, show all values
            if isinstance(reward_data, list):
                reward_str = f"[{', '.join(f'{r:.6f}' for r in reward_data)}]"
            else:
                reward_str = f"{reward_data:.6f}"
            step_preview = step[:80].replace("\n", " ") if len(step) > 80 else step.replace("\n", " ")
            print(f"  Step {step_idx + 1}: {reward_str} | {step_preview}...")

        # Show any extra rewards if present
        if len(rewards_raw) > len(steps):
            print(f"\n  Extra rewards (beyond step count):")
            for extra_idx in range(len(steps), len(rewards_raw)):
                reward_data = rewards_raw[extra_idx]
                if isinstance(reward_data, list):
                    reward_str = f"[{', '.join(f'{r:.6f}' for r in reward_data)}]"
                else:
                    reward_str = f"{reward_data:.6f}"
                print(f"  Reward {extra_idx + 1}: {reward_str}")

        if len(rewards_raw) > 0:
            # Calculate average from first value if rewards are lists
            if isinstance(rewards_raw[0], list):
                avg_reward = sum(r[0] for r in rewards_raw) / len(rewards_raw)
            else:
                avg_reward = sum(rewards_raw) / len(rewards_raw)
            print(f"\nAverage reward: {avg_reward:.6f}")

    print("\n" + "=" * 80)
    print("Inference completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    args = parse_args()
    main(args)
