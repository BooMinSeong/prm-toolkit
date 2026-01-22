#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Skywork-o1-Open-PRM utility functions.
Extracted from skywork-o1-prm-inference for standalone use.
"""

import numpy as np


def prepare_input(problem, response, tokenizer, step_token):
    """
    Prepare input for Skywork-o1-Open-PRM inference.

    Args:
        problem: Problem statement text
        response: Step-by-step response text (steps separated by step_token)
        tokenizer: HuggingFace tokenizer
        step_token: Token used to separate steps (e.g., "\\n")

    Returns:
        tuple: (input_ids, steps, reward_flags)
            - input_ids: List of token IDs for the full input
            - steps: List of step texts
            - reward_flags: List where 1 marks step end positions
    """
    # Encode problem with BOS token
    prompt_ids = tokenizer.encode(tokenizer.bos_token + problem + "\n")

    response_ids = []
    steps = []
    reward_flags = [0] * len(prompt_ids)

    # Get step token ID
    step_token_id = tokenizer.encode(step_token)[-1]

    # Process each step
    for idx, step in enumerate(response.split(step_token)):
        if step != "":
            step_ids = tokenizer.encode(step)
        else:
            step_ids = []

        # Add step token at the end
        step_ids += [step_token_id]
        step = step + step_token

        # Create flags: 1 only at step end position
        flag = [0] * len(step_ids)
        flag[-1] = 1

        response_ids.extend(step_ids)
        reward_flags.extend(flag)
        steps.append(step)

    input_ids = prompt_ids + response_ids
    return input_ids, steps, reward_flags


def sigmoid(x):
    """Apply sigmoid function for reward normalization to [0, 1] range."""
    return 1 / (np.exp(-x) + 1)
