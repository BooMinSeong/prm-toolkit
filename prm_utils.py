#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Unified utility functions for Process Reward Models (PRM).
Supports both Qwen2.5-Math-PRM-7B and Skywork-o1-Open-PRM-Qwen-2.5-1.5B.
"""

from typing import Union, List, Dict, Tuple, Any, Optional
import numpy as np


def detect_model_type(model_name: str) -> str:
    """
    Detect model type from model name.

    Args:
        model_name: Model name or path (e.g., "Qwen/Qwen2.5-Math-PRM-7B")

    Returns:
        "qwen" or "skywork"

    Raises:
        ValueError: If model type cannot be determined
    """
    model_name_lower = model_name.lower()

    if "qwen2.5-math-prm" in model_name_lower or "qwen/qwen2.5-math-prm" in model_name_lower:
        return "qwen"
    elif "skywork" in model_name_lower:
        return "skywork"
    else:
        raise ValueError(
            f"Cannot determine model type from '{model_name}'. "
            "Please specify --model-type explicitly (qwen or skywork)"
        )


def normalize_input(data: dict) -> dict:
    """
    Normalize input data to support flexible input formats.

    Args:
        data: Input dictionary with 'problem' and 'response' keys

    Returns:
        Normalized data dictionary with response as list
    """
    normalized = data.copy()

    # Convert response to list if it's a string
    if isinstance(normalized["response"], str):
        # Split by newline if present, otherwise treat as single step
        if "\n" in normalized["response"]:
            normalized["response"] = [
                step for step in normalized["response"].split("\n")
                if step.strip()  # Filter out empty lines
            ]
        else:
            normalized["response"] = [normalized["response"]]

    # Ensure response is a list
    if not isinstance(normalized["response"], list):
        normalized["response"] = [str(normalized["response"])]

    return normalized


def prepare_prm_input(
    model_type: str,
    problem: str,
    response: Union[str, List[str]],
    system: Optional[str] = None,
    tokenizer = None
) -> Tuple[Union[str, List[int]], List[str], Any]:
    """
    Prepare input for PRM inference in model-specific format.

    Args:
        model_type: "qwen" or "skywork"
        problem: Problem statement text
        response: Step-by-step response (string with newlines or list of steps)
        system: System prompt (optional, defaults differ by model)
        tokenizer: HuggingFace tokenizer (required for Skywork)

    Returns:
        Tuple of (input_data, steps, metadata):
            - For Qwen: (prompt_text, steps_list, None)
            - For Skywork: (input_ids_list, steps_list, reward_flags)
    """
    # Normalize response to list
    if isinstance(response, str):
        if "\n" in response:
            steps = [step for step in response.split("\n") if step.strip()]
        else:
            steps = [response]
    else:
        steps = response

    if model_type == "qwen":
        # Qwen PRM format
        # Default system prompt for math reasoning
        if system is None:
            system = "Please reason step by step, and put your final answer within \\boxed{}."

        # Format with <extra_0> step delimiters
        answer = "<extra_0>".join(steps) + "<extra_0>"
        prompt = (
            f"<im_start>system\n{system}<im_end>\n"
            f"<im_start>user\n{problem}<im_end>\n"
            f"<im_start>assistant\n{answer}<im_end><|endoftext|>"
        )

        return prompt, steps, None

    elif model_type == "skywork":
        # Skywork PRM format
        if tokenizer is None:
            raise ValueError("Tokenizer is required for Skywork model")

        # Import Skywork utilities
        from skywork_utils import prepare_input

        # Join steps with newline
        response_text = "\n".join(steps)

        # Use Skywork's prepare_input function
        input_ids, processed_steps, reward_flags = prepare_input(
            problem=problem,
            response=response_text,
            tokenizer=tokenizer,
            step_token="\n"
        )

        return input_ids, processed_steps, reward_flags

    else:
        raise ValueError(f"Unsupported model type: {model_type}")


def parse_prm_response(
    model_type: str,
    response: dict,
    steps: List[str],
    reward_flags: Optional[List[int]] = None
) -> Dict[str, Any]:
    """
    Parse PRM response into unified output format.

    Args:
        model_type: "qwen" or "skywork"
        response: JSON response from /pooling endpoint
        steps: List of step texts
        reward_flags: Reward flags (required for Skywork)

    Returns:
        Dictionary with structure:
        {
            "steps": ["step1", "step2", ...],
            "rewards": [0.95, 0.87, ...],
            "average_reward": 0.91,
            "num_steps": 2,
            "metadata": {...}
        }
    """
    if model_type == "qwen":
        return _parse_qwen_response(response, steps)
    elif model_type == "skywork":
        if reward_flags is None:
            raise ValueError("reward_flags is required for Skywork model")
        return _parse_skywork_response(response, steps, reward_flags)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


def _parse_qwen_response(response: dict, steps: List[str]) -> Dict[str, Any]:
    """
    Parse Qwen PRM response.

    Qwen2.5-Math-PRM outputs 2 logits per token (negative/positive class).
    We apply softmax normalization and extract the positive class probability.

    Reference: Qwen/Qwen2.5-Math-PRM-7B official example
    https://huggingface.co/Qwen/Qwen2.5-Math-PRM-7B
    """
    # Extract rewards from response
    # Format: {"data": [{"data": [[logit0, logit1], [logit0, logit1], ...]}]}
    rewards_raw = response["data"][0]["data"]

    # Convert to numpy array for efficient computation
    logits_array = np.array(rewards_raw, dtype=np.float32)

    # Check shape
    if logits_array.ndim == 1:
        # Single value per step - already processed or old format
        rewards = logits_array.tolist()
    elif logits_array.ndim == 2 and logits_array.shape[1] == 2:
        # Standard Qwen PRM format: [num_steps, 2]
        # Apply softmax to convert logits to probabilities
        # softmax(x) = exp(x) / sum(exp(x))
        exp_logits = np.exp(logits_array - np.max(logits_array, axis=1, keepdims=True))
        probabilities = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        # Extract positive class (index 1)
        rewards = probabilities[:, 1].tolist()
    else:
        # Unexpected format - fallback to first value
        if logits_array.ndim == 2:
            rewards = logits_array[:, 0].tolist()
        else:
            rewards = [float(logits_array.flatten()[0])]

    # Calculate average
    avg_reward = sum(rewards) / len(rewards) if rewards else 0.0

    return {
        "steps": steps[:len(rewards)],  # Match steps to rewards count
        "rewards": rewards,
        "average_reward": avg_reward,
        "num_steps": len(rewards),
        "metadata": {
            "expected_steps": len(steps),
            "actual_rewards": len(rewards),
            "format": "qwen_prm",
            "logits_shape": logits_array.shape
        }
    }


def _parse_skywork_response(
    response: dict,
    steps: List[str],
    reward_flags: List[int]
) -> Dict[str, Any]:
    """Parse Skywork PRM response."""
    from skywork_utils import sigmoid

    # Extract rewards from response
    # Format: {"data": [{"data": [[reward1], [reward2], ...]}]}
    rewards_data = response["data"][0]["data"]

    # Extract rewards only at step positions (where reward_flags == 1)
    step_rewards = []
    for reward_list, flag in zip(rewards_data, reward_flags):
        if flag == 1:
            # Extract the reward value (first element)
            reward_value = reward_list[0]
            # Apply sigmoid to normalize to [0, 1]
            step_rewards.append(float(sigmoid(reward_value)))

    # Calculate average
    avg_reward = sum(step_rewards) / len(step_rewards) if step_rewards else 0.0

    return {
        "steps": steps,
        "rewards": step_rewards,
        "average_reward": avg_reward,
        "num_steps": len(step_rewards),
        "metadata": {
            "total_tokens": len(reward_flags),
            "step_positions": sum(reward_flags),
            "format": "skywork_prm"
        }
    }


def format_results_summary(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Create a summary of results in unified format.

    Args:
        results: List of parsed results from parse_prm_response

    Returns:
        Summary dictionary with statistics
    """
    if not results:
        return {
            "problems": [],
            "summary": {
                "total_problems": 0,
                "average_reward": 0.0,
                "total_steps": 0
            }
        }

    # Calculate statistics
    total_problems = len(results)
    total_reward = sum(r["average_reward"] for r in results)
    average_reward = total_reward / total_problems
    total_steps = sum(r["num_steps"] for r in results)

    # Format problems with IDs
    problems = []
    for idx, result in enumerate(results):
        problem_data = {
            "problem_id": idx,
            "steps": result["steps"],
            "rewards": result["rewards"],
            "average_reward": result["average_reward"],
            "metadata": result.get("metadata", {})
        }
        problems.append(problem_data)

    return {
        "problems": problems,
        "summary": {
            "total_problems": total_problems,
            "average_reward": average_reward,
            "total_steps": total_steps,
            "min_reward": min(r["average_reward"] for r in results),
            "max_reward": max(r["average_reward"] for r in results)
        }
    }
