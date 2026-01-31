#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Custom Qwen2ForPrmModel implementation for Skywork-o1-Open-PRM
Compatible with vLLM 0.14.1

This module provides a custom implementation that bridges Skywork's model architecture
with vLLM 0.14.1's API, specifically handling the v_head parameter structure.
"""

from collections.abc import Iterable
from typing import Optional

import torch
from torch import nn

from vllm.config import VllmConfig
from vllm.model_executor.layers.pooler import Pooler
from vllm.model_executor.layers.pooler.tokwise import pooler_for_token_classify
from vllm.sequence import IntermediateTensors

from vllm.model_executor.models.interfaces import SupportsLoRA, SupportsPP
from vllm.model_executor.models.interfaces_base import default_pooling_type
from vllm.model_executor.models.qwen2 import Qwen2Model
from vllm.model_executor.models.utils import AutoWeightsLoader, maybe_prefix


class ValueHead(nn.Module):
    """
    Value head for process reward models.
    Returns a scalar for each output token.
    """

    def __init__(self, config, **kwargs):
        super().__init__()
        if not hasattr(config, "summary_dropout_prob"):
            summary_dropout_prob = kwargs.pop("summary_dropout_prob", 0.1)
        else:
            summary_dropout_prob = config.summary_dropout_prob

        self.dropout = (
            nn.Dropout(summary_dropout_prob) if summary_dropout_prob else nn.Identity()
        )

        if hasattr(config, "hidden_size"):
            hidden_size = config.hidden_size
        else:
            raise ValueError("Config must have hidden_size attribute")

        self.summary = nn.Linear(hidden_size, 1)
        self.flatten = nn.Flatten()

    def forward(self, hidden_states):
        output = self.dropout(hidden_states)

        # Force upcast to fp32 if needed for numerical stability
        if output.dtype != self.summary.weight.dtype:
            output = output.to(self.summary.weight.dtype)

        output = self.summary(output)
        return output


@default_pooling_type(tok_pooling_type="STEP")
class SkyworkQwen2ForPrmModel(nn.Module, SupportsLoRA, SupportsPP):
    """
    Skywork-specific Process Reward Model implementation.

    This model implements Skywork-o1-Open-PRM's architecture which differs from
    vLLM's standard Qwen2ForProcessRewardModel in its value head structure:
    - Uses v_head (ValueHead) instead of score
    - Qwen2Model as the base transformer
    - ValueHead for reward prediction with dropout
    - STEP pooling for process-level rewards
    """

    is_pooling_model = True

    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    # LoRA specific attributes
    supported_lora_modules = [
        "qkv_proj",
        "o_proj",
        "gate_up_proj",
        "down_proj",
    ]
    embedding_modules = {}
    embedding_padding_modules = []

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        pooler_config = vllm_config.model_config.pooler_config

        self.config = config
        self.quant_config = quant_config

        # Initialize Qwen2 base model
        self.model = Qwen2Model(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )

        # Initialize value head (specific to Skywork's architecture)
        self.v_head = ValueHead(config)

        # Set up pooler for STEP pooling (same as Qwen2ForProcessRewardModel)
        assert pooler_config is not None, "PoolerConfig is required for PRM models"

        self.pooler = pooler_for_token_classify(pooler_config)

        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor | IntermediateTensors:
        """
        Forward pass through the model.

        Args:
            input_ids: Input token IDs
            positions: Position indices
            intermediate_tensors: Optional intermediate tensors for pipeline parallelism
            inputs_embeds: Optional pre-computed input embeddings

        Returns:
            Logits from the value head
        """
        # Get hidden states from base model
        hidden_states = self.model(
            input_ids, positions, intermediate_tensors, inputs_embeds
        )

        # Apply value head to get rewards
        logits = self.v_head(hidden_states)

        return logits

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """
        Load model weights.

        Args:
            weights: Iterable of (parameter_name, tensor) tuples

        Returns:
            Set of loaded weight names
        """
        loader = AutoWeightsLoader(
            self,
            ignore_unexpected_prefixes=["lm_head."]
        )
        return loader.load_weights(weights)


def register_skywork_prm_model():
    """
    Register SkyworkQwen2ForPrmModel with vLLM's ModelRegistry.

    Maps the "Qwen2ForPrmModel" architecture name (from Skywork's config.json)
    to our custom SkyworkQwen2ForPrmModel implementation, which handles
    Skywork's specific v_head parameter structure.
    """
    from vllm import ModelRegistry

    if "Qwen2ForPrmModel" not in ModelRegistry.get_supported_archs():
        ModelRegistry.register_model(
            "Qwen2ForPrmModel",  # Name from model config.json
            "prm_toolkit.skywork_prm_model:SkyworkQwen2ForPrmModel"  # Our implementation
        )
        print("✓ Registered SkyworkQwen2ForPrmModel (Qwen2ForPrmModel) for Skywork-o1-Open-PRM")
    else:
        print("✓ SkyworkQwen2ForPrmModel (Qwen2ForPrmModel) already registered")
