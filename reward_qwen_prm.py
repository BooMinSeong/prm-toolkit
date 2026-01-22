# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from argparse import Namespace

from vllm import LLM, EngineArgs
from vllm.utils.argparse_utils import FlexibleArgumentParser


def parse_args():
    parser = FlexibleArgumentParser()
    parser = EngineArgs.add_cli_args(parser)
    # Set example specific arguments
    parser.set_defaults(
        model="Qwen/Qwen2.5-Math-PRM-7B",
        runner="pooling",
        max_model_len=1024,
        trust_remote_code=True,
    )
    return parser.parse_args()

def math_step_prompts():
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
    answer = "<extra_0>".join(data["response"]) + "<extra_0>"
    prompt = f"<im_start>system\n{data['system']}<im_end>\n<im_start>user\n{data['query']}<im_end>\n<im_start>assistant\n{answer}<im_end><|endoftext|>"
    return [prompt]

def main(args: Namespace):

    # Create an LLM.
    # You should pass runner="pooling" for reward models
    llm = LLM(**vars(args))

    # Get prompts from the function
    prompts = math_step_prompts()

    # Generate rewards. The output is a list of PoolingRequestOutput.
    outputs = llm.reward(prompts)

    print(outputs)
    # Print the outputs.
    print("\nGenerated Outputs:\n" + "-" * 60)
    for prompt, output in zip(prompts, outputs):
        rewards = output.outputs.data
        rewards_trimmed = (
            (str(rewards[:16])[:-1] + ", ...]") if len(rewards) > 16 else rewards
        )
        print(f"Prompt: {prompt!r} \nReward: {rewards_trimmed} (size={len(rewards)})")
        print("-" * 60)


if __name__ == "__main__":
    args = parse_args()
    main(args)
