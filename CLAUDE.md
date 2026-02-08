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

**Server Mode (requires vLLM server running):**

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

**Local Mode (NEW - no server needed, loads model directly into GPU):**

```bash
source .venv/bin/activate

# Qwen PRM - local mode
python example_prm_usage.py --model qwen --local

# Skywork PRM - local mode
python example_prm_usage.py --model skywork --local

# Batch processing in local mode
python example_prm_usage.py --model qwen --local --batch

# Custom GPU settings
python example_prm_usage.py --model qwen --local --gpu-memory-utilization 0.7

# Validation tests
python test_local_mode.py qwen
python test_local_mode.py skywork
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
  - `load_prm_server()`: Factory function for model instantiation

- **example_prm_usage.py**: Demonstration script showing usage with both models

**Usage pattern:**

Server Mode:
```python
from prm_toolkit import PrmConfig, load_prm_server

# Create configuration for server mode
config = PrmConfig(
    prm_path="Qwen/Qwen2.5-Math-PRM-7B",
    base_url="http://localhost:8080"
)

# Create PRM server and score
prm = load_prm_server(config)
rewards = prm.score(prompt="...", response="...")
```

Local Mode (NEW):
```python
from prm_toolkit import PrmConfig, load_prm_server

# Create configuration for local mode
config = PrmConfig(
    prm_path="Qwen/Qwen2.5-Math-PRM-7B",
    use_local_mode=True,
    gpu_memory_utilization=0.7,  # Use 70% of GPU memory
)

# Create PRM server and score
prm = load_prm_server(config)
rewards = prm.score(prompt="...", response="...")

# Cleanup GPU resources when done
prm.cleanup()
```

**Key features:**
- Two modes: Server (HTTP) or Local (direct GPU)
- Model-specific preprocessing and postprocessing
- Unified interface across different PRM models
- Type-safe configuration with dataclass
- Automatic GPU memory management in local mode

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

**New approach (unified prm_toolkit):**
```python
from prm_toolkit import PrmConfig, load_prm_server

config = PrmConfig(
    prm_path="Qwen/Qwen2.5-Math-PRM-7B",
    base_url="http://localhost:8080"
)
prm = load_prm_server(config)
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

### Mode Selection Guide

**Server Mode - Use when:**
- Running multiple experiments/clients against same model
- Sharing model across processes/machines
- Need long-running service

**Local Mode - Use when:**
- Single script/notebook using the PRM
- Want simpler setup (no server management)
- Prototyping or one-off analysis
- Have dedicated GPU for the task

### Starting Servers (Server Mode Only)

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

### Local Mode Details

**Configuration Options:**
- `use_local_mode=True`: Enable local mode
- `gpu_memory_utilization`: GPU memory fraction (default 0.9)
- `tensor_parallel_size`: Number of GPUs for tensor parallelism (default 1)
- `max_model_len`: Maximum sequence length (defaults to max_tokens)

**Memory Management:**
- Model loads immediately in `__init__` (not lazy)
- Call `prm.cleanup()` to free GPU memory when done
- Destructor (`__del__`) provides safety net but manual cleanup is recommended

**Example with Cleanup:**
```python
prm = load_prm_server(config)
try:
    rewards = prm.score(prompt, response)
    # Process rewards...
finally:
    prm.cleanup()
```
