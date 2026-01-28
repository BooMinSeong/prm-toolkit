# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a test project for vLLM reward model inference, specifically exploring process reward models (PRM) for evaluating mathematical reasoning steps:
- **Qwen2.5-Math-PRM-7B**: Process reward model for math reasoning
- **Skywork-o1-Open-PRM-Qwen-2.5-1.5B**: Skywork's open-source PRM (with vLLM 0.14.1 compatibility fix)

## Setup

**IMPORTANT: Install the vLLM plugin first (required for Skywork-o1-Open-PRM):**

```bash
source .venv/bin/activate
pip install -e .  # Registers SkyworkQwen2ForPrmModel with vLLM
```

## Running the Scripts

### Unified PRM Server (Recommended)

```bash
source .venv/bin/activate

# Qwen PRM example
# Terminal 1: Start Qwen PRM server
vllm serve Qwen/Qwen2.5-Math-PRM-7B --port 8080 --trust-remote-code

# Terminal 2: Run example
python example_prm_usage.py --model qwen

# Skywork PRM example
# Terminal 1: Start Skywork PRM server
python start_reward_server.py

# Terminal 2: Run example
python example_prm_usage.py --model skywork
```

### Legacy Scripts

```bash
source .venv/bin/activate

# Basic reward model tests
python reward.py           # Basic reward model test with simple prompts
python reward_qwen_prm.py  # Math step evaluation with Qwen PRM format

# Skywork-o1-Open-PRM (two options)
# Option A: Direct execution
python reward_skywork_o1_prm.py

# Option B: Server/client mode
# Terminal 1: Start server
python start_reward_server.py
# Terminal 2: Run client
python reward_skywork_server.py
```

All scripts accept vLLM engine arguments via CLI (e.g., `--model`, `--max-model-len`, `--tensor-parallel-size`).

## Architecture

### Unified PRM Server (Recommended)

**NEW**: Unified server-based architecture for all PRM models:

- **prm_server.py**: Main module with unified PRM interface
  - `PrmConfig`: Dataclass for type-safe configuration
  - `PrmServer`: Abstract base class with `score(prompt, response)` API
  - `QwenPrmServer`: Qwen2.5-Math-PRM implementation
  - `SkyworkPrmServer`: Skywork-o1-Open-PRM implementation
  - `create_prm_server()`: Factory function for model instantiation

- **example_prm_usage.py**: Demonstration script showing usage with both models

**Usage pattern:**
```python
from prm_server import PrmConfig, create_prm_server

# Create configuration
config = PrmConfig(
    model="Qwen/Qwen2.5-Math-PRM-7B",
    base_url="http://localhost:8080"
)

# Create PRM server and score
prm = create_prm_server(config)
rewards = prm.score(prompt="...", response="...")
```

**Key features:**
- Server-only access (no local LLM.reward() calls)
- Model-specific preprocessing and postprocessing
- Unified interface across different PRM models
- Type-safe configuration with dataclass

**Model-specific details:**
- **Qwen**: Uses `\n\n` (double newline) as step delimiter, `<extra_0>` tokens, returns `[neg_prob, pos_prob]` pairs
- **Skywork**: Uses `\n` (single newline) as step delimiter, reward_flags for step positions, sigmoid normalization

### Legacy Scripts (Reference)

- **reward.py**: Basic vLLM pooling/reward model example using simple text prompts
- **reward_qwen_prm.py**: Specialized script for Qwen2.5-Math-PRM format, demonstrating step-by-step math reasoning evaluation with `<extra_0>` step delimiters and `<im_start>`/`<im_end>` chat template markers
- **reward_qwen_prm_server.py**: Qwen PRM server mode (reference for QwenPrmServer implementation)
- **reward_skywork_o1_prm.py**: Skywork-o1-Open-PRM direct execution script
- **start_reward_server.py**: Helper script to start vLLM server with Skywork-o1-Open-PRM
- **reward_skywork_server.py**: Client script for server/client mode using OpenAI-compatible API
- **skywork_prm_model.py**: Custom `SkyworkQwen2ForPrmModel` implementation (vLLM plugin)
- **pyproject.toml**: Package configuration with vLLM plugin entry point

Legacy scripts use vLLM's `LLM.reward()` API or direct server calls. **Use the unified PRM server for new code.**

## vLLM 0.14.1 Compatibility

Skywork-o1-Open-PRM uses `Qwen2ForPrmModel` architecture with a `v_head` parameter structure that differs from vLLM's standard `Qwen2ForProcessRewardModel`. This repository provides a custom model implementation (`skywork_prm_model.py`) that:

- Implements Skywork's exact architecture (ValueHead with `v_head` parameters)
- Uses vLLM 0.14.1's native APIs (Pooler, DispatchPooler)
- Supports STEP pooling for process-level rewards
- Registers `SkyworkQwen2ForPrmModel` with vLLM's ModelRegistry

### vLLM 0.14+ Notes

- **Async Scheduling**: Enabled by default in v0.14.0. If issues occur, disable with `--disable-async-output-proc`
- **PyTorch Requirement**: v2.5.0+ required
- **Plugin System**: No changes to entry point registration

### vLLM Plugin System

The custom model is registered using vLLM's official plugin system via `pyproject.toml`:

```toml
[project.entry-points."vllm.general_plugins"]
register_skywork_prm = "skywork_prm_model:register_skywork_prm_model"
```

**Installation:**
```bash
pip install -e .
```

This automatically registers the model when vLLM starts. You should see:
```
✓ Registered SkyworkQwen2ForPrmModel (Qwen2ForPrmModel) for Skywork-o1-Open-PRM
```

See inline documentation in `prm_server.py` and `skywork_prm_model.py` for technical details.

## Migration Guide

### From Legacy Scripts to Unified PRM Server

**Old approach (reward_qwen_prm_server.py):**
```python
# Manual request handling
prompts, steps_list = math_step_prompts()
pooling_response = requests.post(
    f"{base_url}/pooling",
    json={"input": prompts},
    ...
)
rewards_raw = pooling_response.json()["data"][0]["data"]
rewards = [r[1] for r in rewards_raw]  # Extract positive probability
```

**New approach (unified prm_server.py):**
```python
from prm_server import PrmConfig, create_prm_server

config = PrmConfig(
    model="Qwen/Qwen2.5-Math-PRM-7B",
    base_url="http://localhost:8080"
)
prm = create_prm_server(config)
rewards = prm.score(prompt="...", response="...")
```

**Benefits:**
- Single line to score responses
- Automatic preprocessing (step splitting, formatting)
- Automatic postprocessing (reward extraction, normalization)
- Type-safe configuration
- Model-agnostic interface

### Step Delimiter Differences

**Qwen PRM:**
- Input: Steps separated by `\n\n` (double newline)
- Internal: Converted to `<extra_0>` tokens
- Output: Positive probability from `[neg_prob, pos_prob]` pairs

**Skywork PRM:**
- Input: Steps separated by `\n` (single newline)
- Internal: Tokenized with reward_flags marking step positions
- Output: Sigmoid-normalized rewards [0, 1]

### Starting Servers

**Qwen PRM Server:**
```bash
vllm serve Qwen/Qwen2.5-Math-PRM-7B \
    --port 8080 \
    --trust-remote-code \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.9
```

**Skywork PRM Server:**
```bash
# Requires vLLM plugin installation first: pip install -e .
python start_reward_server.py

# Or manually:
vllm serve Skywork/Skywork-o1-Open-PRM-Qwen-2.5-1.5B \
    --port 8081 \
    --trust-remote-code \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.9
```
