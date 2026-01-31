#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Unified PRM (Process Reward Model) Server Architecture

Provides a server-based interface for evaluating step-by-step reasoning
with different PRM models (Qwen, Skywork, etc.).

Usage:
    from prm_toolkit import PrmConfig, load_prm_server

    # Create configuration
    config = PrmConfig(
        prm_path="Qwen/Qwen2.5-Math-PRM-7B",
        base_url="http://localhost:8080"
    )

    # Create PRM server instance
    prm = load_prm_server(config)

    # Score a response
    rewards = prm.score(
        prompt="What is 2+2?",
        response="Step 1: Add 2 and 2\n\nStep 2: The result is 4"
    )
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import requests
import numpy as np
import logging

logger = logging.getLogger(__name__)


@dataclass
class PrmConfig:
    """Configuration for PRM server"""
    prm_path: str                    # Model name/path
    base_url: str                 # vLLM server URL (e.g., "http://localhost:8081")
    timeout: int = 300            # Request timeout in seconds
    trust_remote_code: bool = True
    max_tokens: int = 4096        # Maximum token length for validation


class PrmServer(ABC):
    """Base class for Process Reward Model servers"""

    def __init__(self, config: PrmConfig):
        self.config = config
        self.base_url = config.base_url

        # Validate max_tokens
        if self.config.max_tokens <= 0:
            raise ValueError(f"max_tokens must be positive, got {self.config.max_tokens}")
        if self.config.max_tokens < 128:
            logger.warning(f"max_tokens={self.config.max_tokens} is very small and may truncate the prompt")

        self.model_type = self.model_check()
        self._init_tokenizer()

    @abstractmethod
    def model_check(self) -> str:
        """Detect and return model type from config"""
        pass

    @abstractmethod
    def _init_tokenizer(self):
        """Initialize model-specific tokenizer"""
        pass

    @abstractmethod
    def validate_length(self, prompt: str, response: str) -> tuple[str, str]:
        """
        Validate and truncate input if it exceeds max_tokens.

        Uses tail truncation: truncates from the END to preserve the prompt.

        Args:
            prompt: Problem statement
            response: Step-by-step solution

        Returns:
            (validated_prompt, validated_response) tuple
            If no truncation needed, returns originals unchanged
            If truncated, returns modified versions that fit within max_tokens
        """
        pass

    @abstractmethod
    def preprocess_input(self, prompt: str, response: str) -> Dict[str, Any]:
        """
        Transform prompt and response into model-specific format.

        Args:
            prompt: Problem statement or question
            response: Step-by-step response/solution

        Returns:
            Dictionary containing processed data ready for server request
        """
        pass

    def send_request(self, processed_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Send request to vLLM server /pooling endpoint.

        Args:
            processed_data: Output from preprocess_input()

        Returns:
            Raw server response (JSON)
        """
        try:
            response = requests.post(
                f"{self.base_url}/pooling",
                json={"input": processed_data["input"]},
                headers={"Content-Type": "application/json"},
                timeout=self.config.timeout
            )
            response.raise_for_status()
            return {
                "response": response.json(),
                "metadata": processed_data.get("metadata", {})
            }
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"PRM server request failed: {e}")

    @abstractmethod
    def post_process_output(self, raw_results: Dict[str, Any]) -> List[float]:
        """
        Extract and normalize step-wise rewards from raw server response.

        Args:
            raw_results: Output from send_request()

        Returns:
            List of normalized step-wise rewards [0, 1]
        """
        pass


    def score(self, prompt: str, response: str) -> List[float]:
        """
        Main entry point: compute step-wise rewards for a response.

        Args:
            prompt: Problem statement
            response: Step-by-step solution

        Returns:
            List of normalized step-wise rewards
        """
        # Validate and potentially truncate input
        validated_prompt, validated_response = self.validate_length(prompt, response)

        processed_data = self.preprocess_input(validated_prompt, validated_response)
        raw_results = self.send_request(processed_data)
        rewards = self.post_process_output(raw_results)
        return rewards

    def score_batch(self, prompts: List[str], responses: List[str]) -> List[List[float]]:
        """
        Batch processing: compute step-wise rewards for multiple (prompt, response) pairs.

        Uses vLLM's batch inference to process all inputs in a single API call,
        significantly reducing network overhead and leveraging GPU batch processing.

        Args:
            prompts: List of problem statements
            responses: List of step-by-step solutions

        Returns:
            List of step-wise reward lists, one per input pair

        Example:
            >>> prompts = ["What is 2+2?", "What is 3+3?"]
            >>> responses = ["Step 1: Add\\n\\nStep 2: Result is 4",
            ...              "Step 1: Add\\n\\nStep 2: Result is 6"]
            >>> batch_rewards = prm.score_batch(prompts, responses)
            >>> # batch_rewards[0] = rewards for first input
            >>> # batch_rewards[1] = rewards for second input
        """
        if len(prompts) != len(responses):
            raise ValueError(f"Length mismatch: {len(prompts)} prompts vs {len(responses)} responses")

        if len(prompts) == 0:
            return []

        # Preprocess all inputs with validation
        all_inputs = []
        all_metadata = []
        for prompt, response in zip(prompts, responses):
            # Validate length first
            validated_prompt, validated_response = self.validate_length(prompt, response)
            processed = self.preprocess_input(validated_prompt, validated_response)
            all_inputs.extend(processed["input"])
            all_metadata.append(processed["metadata"])

        # Single batch API call
        try:
            batch_response = requests.post(
                f"{self.base_url}/pooling",
                json={"input": all_inputs},
                headers={"Content-Type": "application/json"},
                timeout=self.config.timeout
            )
            batch_response.raise_for_status()
            pooling_response = batch_response.json()
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"PRM server batch request failed: {e}")

        # Post-process each result separately
        batch_rewards = []
        for i, metadata in enumerate(all_metadata):
            # Reconstruct raw_results format for post_process_output
            item_result = {
                "response": {
                    "data": [pooling_response["data"][i]]
                },
                "metadata": metadata
            }
            rewards = self.post_process_output(item_result)
            batch_rewards.append(rewards)

        return batch_rewards


class QwenPrmServer(PrmServer):
    """Qwen2.5-Math-PRM implementation"""

    def model_check(self) -> str:
        if "Qwen" in self.config.prm_path and "PRM" in self.config.prm_path:
            return "qwen-prm"
        raise ValueError(f"Model {self.config.prm_path} not compatible with QwenPrmServer")

    def _init_tokenizer(self):
        # Initialize tokenizer for validation (Qwen server handles actual tokenization)
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.prm_path,
            trust_remote_code=self.config.trust_remote_code
        )

    def validate_length(self, prompt: str, response: str) -> tuple[str, str]:
        """
        Validate and truncate input if it exceeds max_tokens.

        Uses Qwen-specific formatting with <extra_0> delimiters and chat template.
        """
        # Format using Qwen template with <extra_0> delimiters
        steps = [s.strip() for s in response.split('\n\n') if s.strip()]
        formatted_response = "<extra_0>".join(steps) + "<extra_0>"

        system_msg = "Please reason step by step, and put your final answer within \\boxed{}."
        formatted_text = (
            f"<im_start>system\n{system_msg}<im_end>\n"
            f"<im_start>user\n{prompt}<im_end>\n"
            f"<im_start>assistant\n{formatted_response}<im_end><|endoftext|>"
        )

        # Tokenize and check length
        tokens = self.tokenizer.encode(formatted_text)

        if len(tokens) <= self.config.max_tokens:
            return prompt, response  # No truncation

        # Tail truncation
        truncated_tokens = tokens[:self.config.max_tokens]
        truncated_text = self.tokenizer.decode(truncated_tokens, skip_special_tokens=False)

        # Parse back: extract response section
        if "<im_start>assistant\n" in truncated_text:
            start = truncated_text.find("<im_start>assistant\n") + len("<im_start>assistant\n")
            response_section = truncated_text[start:].replace("<im_end>", "").replace("<|endoftext|>", "")

            # Convert <extra_0> back to double newlines
            truncated_steps = [s.strip() for s in response_section.split("<extra_0>") if s.strip()]
            truncated_response = "\n\n".join(truncated_steps)

            logger.warning(f"Qwen: Truncated {len(tokens)} → {self.config.max_tokens} tokens ({len(steps)} → {len(truncated_steps)} steps)")
            return prompt, truncated_response
        else:
            # Extreme case: prompt itself was truncated
            logger.error(f"Qwen: Extreme truncation - prompt exceeded {self.config.max_tokens} tokens")
            return prompt, ""

    def preprocess_input(self, prompt: str, response: str) -> Dict[str, Any]:
        """
        Format input using Qwen's chat template with <extra_0> step delimiters.

        IMPORTANT: Does NOT tokenize - passes string directly to server.
        Server handles tokenization internally.

        Format:
        <im_start>system\n{system_prompt}<im_end>\n
        <im_start>user\n{prompt}<im_end>\n
        <im_start>assistant\n{response_with_<extra_0>}<im_end><|endoftext|>
        """
        # Split response into steps and format
        steps = [s.strip() for s in response.split('\n\n') if s.strip()]
        formatted_response = "<extra_0>".join(steps) + "<extra_0>"

        system_msg = "Please reason step by step, and put your final answer within \\boxed{}."
        formatted_prompt = (
            f"<im_start>system\n{system_msg}<im_end>\n"
            f"<im_start>user\n{prompt}<im_end>\n"
            f"<im_start>assistant\n{formatted_response}<im_end><|endoftext|>"
        )

        # Return STRING prompt (NOT tokenized)
        return {
            "input": [formatted_prompt],
            "metadata": {
                "num_steps": len(steps),
                "steps": steps
            }
        }

    def post_process_output(self, raw_results: Dict[str, Any]) -> List[float]:
        """
        Extract step-wise rewards from Qwen PRM response.

        Qwen returns [negative_prob, positive_prob] pairs.
        We extract positive probability (index 1) as the reward.
        """
        pooling_response = raw_results["response"]

        # Extract rewards from first item (single input)
        rewards_data = pooling_response["data"][0]["data"]

        # Qwen returns [[neg_prob, pos_prob], [neg_prob, pos_prob], ...]
        # Use positive probability (index 1) as reward
        step_rewards = [r[1] for r in rewards_data]

        return step_rewards


class SkyworkPrmServer(PrmServer):
    """Skywork-o1-Open-PRM implementation"""

    def model_check(self) -> str:
        if "Skywork" in self.config.prm_path and "PRM" in self.config.prm_path:
            return "skywork-prm"
        raise ValueError(f"Model {self.config.prm_path} not compatible with SkyworkPrmServer")

    def _init_tokenizer(self):
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.prm_path,
            trust_remote_code=self.config.trust_remote_code
        )

    def validate_length(self, prompt: str, response: str) -> tuple[str, str]:
        """
        Validate and truncate input if it exceeds max_tokens.

        Uses SAME tokenization logic as preprocess_input() to ensure accurate validation.
        """
        # Tokenize prompt (same as preprocess_input)
        prompt_ids = self.tokenizer.encode(self.tokenizer.bos_token + prompt + "\n")

        # Tokenize response step-by-step (same as preprocess_input)
        step_token = "\n"
        step_token_id = self.tokenizer.encode(step_token)[-1]

        response_ids = []
        steps = []

        for step in response.split(step_token):
            if step != "":
                step_ids = self.tokenizer.encode(step)
            else:
                step_ids = []

            step_ids += [step_token_id]
            response_ids.extend(step_ids)
            steps.append(step)

        # Total token count (must match preprocess_input exactly)
        total_tokens = len(prompt_ids) + len(response_ids)

        if total_tokens <= self.config.max_tokens:
            return prompt, response  # No truncation

        # Tail truncation: remove steps from the end
        tokens_available = self.config.max_tokens - len(prompt_ids)

        if tokens_available <= 0:
            logger.error(f"Skywork: Prompt alone ({len(prompt_ids)} tokens) exceeds max_tokens={self.config.max_tokens}")
            return prompt, ""

        # Rebuild response with as many steps as fit
        truncated_response_ids = []
        truncated_steps = []

        for step in steps:
            if step != "":
                step_ids = self.tokenizer.encode(step)
            else:
                step_ids = []
            step_ids += [step_token_id]

            if len(truncated_response_ids) + len(step_ids) <= tokens_available:
                truncated_response_ids.extend(step_ids)
                truncated_steps.append(step)
            else:
                break  # Can't fit more steps

        truncated_response = step_token.join(truncated_steps)
        if truncated_steps:  # Add trailing newline if we have steps
            truncated_response += step_token

        logger.warning(
            f"Skywork: Truncated {total_tokens} → {len(prompt_ids) + len(truncated_response_ids)} tokens "
            f"({len(steps)} → {len(truncated_steps)} steps)"
        )

        return prompt, truncated_response

    def preprocess_input(self, prompt: str, response: str) -> Dict[str, Any]:
        """
        Format input using Skywork's tokenization with reward_flags.

        INTEGRATED from skywork_utils.prepare_input() - no external calls.

        Skywork uses newline as step delimiter.
        Creates reward_flags array to mark step-end positions.
        """
        prompt_ids = self.tokenizer.encode(self.tokenizer.bos_token + prompt + "\n")

        response_ids = []
        steps = []
        reward_flags = [0] * len(prompt_ids)

        step_token = "\n"
        step_token_id = self.tokenizer.encode(step_token)[-1]

        for step in response.split(step_token):
            if step != "":
                step_ids = self.tokenizer.encode(step)
            else:
                step_ids = []

            step_ids += [step_token_id]
            step_text = step + step_token

            flag = [0] * len(step_ids)
            flag[-1] = 1

            response_ids.extend(step_ids)
            reward_flags.extend(flag)
            steps.append(step_text)

        input_ids = prompt_ids + response_ids

        return {
            "input": [input_ids],
            "metadata": {
                "steps": steps,
                "reward_flags": reward_flags,
                "num_steps": len(steps)
            }
        }

    def post_process_output(self, raw_results: Dict[str, Any]) -> List[float]:
        """
        Extract step-wise rewards using reward_flags and apply sigmoid.

        INTEGRATED sigmoid from skywork_utils - no external calls.
        """
        pooling_response = raw_results["response"]
        reward_flags = raw_results["metadata"]["reward_flags"]

        # Extract rewards from server response
        rewards_data = pooling_response["data"][0]["data"]

        # Filter rewards at step-end positions (where reward_flags == 1)
        step_rewards = []
        for reward_list, flag in zip(rewards_data, reward_flags):
            if flag == 1:
                reward_value = reward_list[0]
                # Apply sigmoid normalization to [0, 1]
                normalized_reward = 1 / (np.exp(-reward_value) + 1)
                step_rewards.append(normalized_reward)

        return step_rewards


def load_prm_server(config: PrmConfig) -> PrmServer:
    """Factory to instantiate appropriate PRM server based on model"""
    model_lower = config.prm_path.lower()

    # Check Skywork first (more specific) since it contains "qwen" in the name
    if "skywork" in model_lower and "prm" in model_lower:
        return SkyworkPrmServer(config)
    elif "qwen" in model_lower and "prm" in model_lower:
        return QwenPrmServer(config)
    else:
        raise ValueError(f"Unknown PRM model: {config.prm_path}")
