# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import numpy as np
from argparse import Namespace
from transformers import AutoTokenizer
from vllm import LLM, EngineArgs
from vllm.utils.argparse_utils import FlexibleArgumentParser
from skywork_utils import prepare_input, sigmoid


def derive_step_rewards_from_vllm_outputs(vllm_outputs, batch_reward_flags):
    """
    Extract step-wise rewards from vLLM LLM.reward() outputs.

    Args:
        vllm_outputs: List of PoolingRequestOutput from LLM.reward()
        batch_reward_flags: List of reward flag arrays, where 1 indicates step end position

    Returns:
        List of step-wise rewards for each input
    """
    batch_step_rewards = []

    for output, reward_flags in zip(vllm_outputs, batch_reward_flags):
        # Extract rewards from vLLM output
        # output.outputs.data is a list of tensors with shape (num_tokens, num_labels)
        rewards_data = output.outputs.data

        # Extract rewards only at step positions (where reward_flags == 1)
        step_rewards = []
        for reward_tensor, flag in zip(rewards_data, reward_flags):
            if flag == 1:
                # Extract the reward value (first element of the tensor)
                reward_value = reward_tensor[0].item()
                # Apply sigmoid to normalize to [0, 1]
                step_rewards.append(sigmoid(reward_value))

        batch_step_rewards.append(step_rewards)

    return batch_step_rewards


def parse_args():
    parser = FlexibleArgumentParser()
    parser = EngineArgs.add_cli_args(parser)
    # Set example specific arguments
    parser.set_defaults(
        model="Skywork/Skywork-o1-Open-PRM-Qwen-2.5-1.5B",
        runner="pooling",
        max_model_len=1024,
        trust_remote_code=True,
    )
    return parser.parse_args()

def math_step_prompts(tokenizer):
    # ruff: noqa: E501
    # Skywork-o1-Open-PRM uses newline-separated steps
    data = {
        "problem": "Please reason step by step, and put your final answer within \\boxed{}. Sue lives in a fun neighborhood.  One weekend, the neighbors decided to play a prank on Sue.  On Friday morning, the neighbors placed 18 pink plastic flamingos out on Sue's front yard.  On Saturday morning, the neighbors took back one third of the flamingos, painted them white, and put these newly painted white flamingos back out on Sue's front yard.  Then, on Sunday morning, they added another 18 pink plastic flamingos to the collection. At noon on Sunday, how many more pink plastic flamingos were out than white plastic flamingos?",
        "response": "\n".join([
            "To find out how many more pink plastic flamingos were out than white plastic flamingos at noon on Sunday, we can break down the problem into steps. First, on Friday, the neighbors start with 18 pink plastic flamingos.",
            "On Saturday, they take back one third of the flamingos. Since there were 18 flamingos, (1/3 \\times 18 = 6) flamingos are taken back. So, they have (18 - 6 = 12) flamingos left in their possession. Then, they paint these 6 flamingos white and put them back out on Sue's front yard. Now, Sue has the original 12 pink flamingos plus the 6 new white ones. Thus, by the end of Saturday, Sue has (12 + 6 = 18) pink flamingos and 6 white flamingos.",
            "On Sunday, the neighbors add another 18 pink plastic flamingos to Sue's front yard. By the end of Sunday morning, Sue has (18 + 18 = 36) pink flamingos and still 6 white flamingos.",
            "To find the difference, subtract the number of white flamingos from the number of pink flamingos: (36 - 6 = 30). Therefore, at noon on Sunday, there were 30 more pink plastic flamingos out than white plastic flamingos. The answer is (\\boxed{30}).",
        ]),
    }

    # Use Skywork's official prepare_input function
    input_ids, steps, reward_flags = prepare_input(
        data["problem"],
        data["response"],
        tokenizer=tokenizer,
        step_token="\n"  # Skywork-o1-Open-PRM uses newlines as step delimiters
    )

    # Return prompts, steps, and reward_flags for later processing
    return [{"prompt_token_ids": input_ids, "multi_modal_data": None}], [steps], [reward_flags]

def main(args: Namespace):
    # Load tokenizer with trust_remote_code (required for Skywork model)
    # Note: The SkyworkQwen2ForPrmModel is automatically registered via vLLM plugin system
    # Make sure to run: pip install -e .
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=True
    )

    # Create an LLM.
    # You should pass runner="pooling" for reward models
    llm = LLM(**vars(args))

    # Get prompts, steps, and reward_flags from the function
    prompts, steps_list, reward_flags_list = math_step_prompts(tokenizer)

    # Generate rewards. The output is a list of PoolingRequestOutput.
    # This returns rewards for ALL tokens (not just step positions)
    outputs = llm.reward(prompts)

    # Extract step-wise rewards using reward_flags
    # This filters the rewards to get only the step-end positions
    step_rewards = derive_step_rewards_from_vllm_outputs(outputs, reward_flags_list)

    # Print the outputs.
    print("\n" + "=" * 80)
    print("SKYWORK-O1-OPEN-PRM STEP-WISE REWARDS")
    print("=" * 80)

    for idx, (steps, rewards) in enumerate(zip(steps_list, step_rewards)):
        print(f"\nProblem {idx + 1}:")
        print("-" * 80)
        print(f"Number of steps: {len(steps)}")
        print(f"\nStep-wise rewards:")
        for step_idx, (step, reward) in enumerate(zip(steps, rewards)):
            step_preview = step[:80].replace("\n", " ") if len(step) > 80 else step.replace("\n", " ")
            print(f"  Step {step_idx + 1}: {reward:.4f} | {step_preview}...")

        avg_reward = sum(rewards) / len(rewards) if rewards else 0
        print(f"\nAverage reward (for Best-of-N ranking): {avg_reward:.4f}")
        print("-" * 80)


if __name__ == "__main__":
    args = parse_args()
    main(args)
