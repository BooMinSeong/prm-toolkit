# PRM Server Unified Architecture

A unified, server-based interface for Process Reward Model (PRM) inference with vLLM.

## Overview

This architecture provides a clean, consistent API for evaluating step-by-step reasoning across different PRM models (Qwen, Skywork). All models are accessed via vLLM server HTTP endpoints, ensuring scalability and consistency.

## Quick Start

### 1. Install Dependencies

```bash
source .venv/bin/activate
pip install -e .  # Registers Skywork PRM plugin with vLLM
```

### 2. Start a PRM Server

**For Qwen PRM:**
```bash
vllm serve Qwen/Qwen2.5-Math-PRM-7B --port 8080 --trust-remote-code
```

**For Skywork PRM:**
```bash
python start_reward_server.py
# Or: vllm serve Skywork/Skywork-o1-Open-PRM-Qwen-2.5-1.5B --port 8081 --trust-remote-code
```

### 3. Use the Unified API

```python
from prm_server import PrmConfig, create_prm_server

# Configure
config = PrmConfig(
    model="Qwen/Qwen2.5-Math-PRM-7B",
    base_url="http://localhost:8080"
)

# Create PRM instance
prm = create_prm_server(config)

# Score a response
prompt = "What is 15 + 27?"
response = "Step 1: Add 15 and 27\n\nStep 2: The result is 42"
rewards = prm.score(prompt, response)

# Results: [0.95, 0.98] (example rewards for each step)
```

## Architecture

### Core Components

```
prm_server.py
├── PrmConfig              # Type-safe configuration dataclass
├── PrmServer              # Abstract base class
│   ├── preprocess_input() # Model-specific formatting
│   ├── send_request()     # HTTP communication
│   ├── post_process_output() # Reward extraction
│   └── score()            # Main entry point
├── QwenPrmServer          # Qwen2.5-Math-PRM implementation
├── SkyworkPrmServer       # Skywork-o1-Open-PRM implementation
└── create_prm_server()    # Factory function
```

### Configuration

```python
@dataclass
class PrmConfig:
    model: str                    # Model name (e.g., "Qwen/Qwen2.5-Math-PRM-7B")
    base_url: str                 # Server URL (e.g., "http://localhost:8080")
    timeout: int = 300            # Request timeout in seconds
    trust_remote_code: bool = True
```

### Interface

All PRM servers implement the same interface:

```python
class PrmServer(ABC):
    def score(self, prompt: str, response: str) -> List[float]:
        """
        Score a step-by-step response.

        Args:
            prompt: Problem statement or question
            response: Step-by-step solution (steps separated by model-specific delimiter)

        Returns:
            List of normalized rewards (one per step)
        """
```

## Model-Specific Details

### Qwen2.5-Math-PRM-7B

**Step Delimiter:** `\n\n` (double newline)

**Input Format:**
```python
prompt = "Solve: 2 + 2 = ?"
response = "Step 1: Add 2 and 2\n\nStep 2: The result is 4"
```

**Internal Processing:**
- Wraps in chat template: `<im_start>system\n...<im_end>\n<im_start>user\n...<im_end>\n<im_start>assistant\n...`
- Converts steps to `<extra_0>` tokens: `Step 1<extra_0>Step 2<extra_0>`
- Server returns `[negative_prob, positive_prob]` pairs
- Extracts positive probability (index 1) as reward

**Output:** Probabilities in [0, 1] range (no additional normalization)

**Example:**
```python
config = PrmConfig(
    model="Qwen/Qwen2.5-Math-PRM-7B",
    base_url="http://localhost:8080"
)
prm = create_prm_server(config)

prompt = "Calculate 15 + 27"
response = "First, add the ones place: 5 + 7 = 12\n\nThen add the tens place: 10 + 20 = 30\n\nFinally: 30 + 12 = 42"
rewards = prm.score(prompt, response)
# Returns: [0.92, 0.95, 0.98] (3 steps)
```

### Skywork-o1-Open-PRM-Qwen-2.5-1.5B

**Step Delimiter:** `\n` (single newline)

**Input Format:**
```python
prompt = "Solve: 2 + 2 = ?"
response = "Step 1: Add 2 and 2\nStep 2: The result is 4"
```

**Internal Processing:**
- Tokenizes with BOS token + problem + response
- Creates reward_flags array (1 at step-end positions)
- Server returns raw logits
- Filters by reward_flags and applies sigmoid: `1 / (1 + exp(-x))`

**Output:** Sigmoid-normalized rewards in [0, 1] range

**Example:**
```python
config = PrmConfig(
    model="Skywork/Skywork-o1-Open-PRM-Qwen-2.5-1.5B",
    base_url="http://localhost:8081"
)
prm = create_prm_server(config)

prompt = "Janet's ducks lay 16 eggs per day. She eats 3 for breakfast and uses 4 for muffins. How many can she sell?"
response = "Step 1: Total eggs: 16\nStep 2: Used: 3 + 4 = 7\nStep 3: Remaining: 16 - 7 = 9"
rewards = prm.score(prompt, response)
# Returns: [0.88, 0.91, 0.96] (3 steps)
```

## Usage Patterns

### Basic Usage

```python
from prm_server import PrmConfig, create_prm_server

config = PrmConfig(model="Qwen/Qwen2.5-Math-PRM-7B", base_url="http://localhost:8080")
prm = create_prm_server(config)
rewards = prm.score(prompt="...", response="...")
```

### With LLM Generation

```python
from prm_server import PrmConfig, create_prm_server
from vllm import LLM, SamplingParams

# Generate response
llm = LLM(model="meta-llama/Llama-3.1-8B-Instruct")
outputs = llm.generate(
    prompts=["Solve: 15 + 27 = ?"],
    sampling_params=SamplingParams(temperature=0.7, max_tokens=200)
)
response = outputs[0].outputs[0].text

# Score with PRM
prm_config = PrmConfig(model="Qwen/Qwen2.5-Math-PRM-7B", base_url="http://localhost:8080")
prm = create_prm_server(prm_config)
rewards = prm.score(prompt="Solve: 15 + 27 = ?", response=response)
```

### Batch Processing

```python
problems = [
    ("Problem 1", "Solution 1 with\n\nsteps"),
    ("Problem 2", "Solution 2 with\n\nsteps"),
]

results = []
for prompt, response in problems:
    rewards = prm.score(prompt, response)
    avg_reward = sum(rewards) / len(rewards)
    results.append((prompt, avg_reward))

# Find best solution
best_prompt, best_score = max(results, key=lambda x: x[1])
```

### Error Handling

```python
from prm_server import PrmConfig, create_prm_server

config = PrmConfig(model="Qwen/Qwen2.5-Math-PRM-7B", base_url="http://localhost:8080")

try:
    prm = create_prm_server(config)
    rewards = prm.score(prompt="...", response="...")
except ValueError as e:
    print(f"Configuration error: {e}")
except RuntimeError as e:
    print(f"Server communication error: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

## Advanced Features

### Custom Timeout

```python
config = PrmConfig(
    model="Qwen/Qwen2.5-Math-PRM-7B",
    base_url="http://localhost:8080",
    timeout=600  # 10 minutes for large inputs
)
```

### Server Health Check

```python
import requests

def check_server(base_url: str) -> bool:
    try:
        response = requests.get(f"{base_url}/v1/models", timeout=5)
        return response.status_code == 200
    except:
        return False

if check_server("http://localhost:8080"):
    prm = create_prm_server(config)
else:
    print("Server not available")
```

## Comparison: Legacy vs Unified

### Legacy Approach (reward_qwen_prm_server.py)

```python
# Manual formatting
steps = response.split("\n\n")
formatted = "<extra_0>".join(steps) + "<extra_0>"
prompt = f"<im_start>system\n{system}<im_end>\n<im_start>user\n{query}<im_end>\n<im_start>assistant\n{formatted}<im_end><|endoftext|>"

# Manual request
response = requests.post(
    f"{base_url}/pooling",
    json={"input": [prompt]},
    timeout=300
)

# Manual extraction
rewards_raw = response.json()["data"][0]["data"]
rewards = [r[1] for r in rewards_raw]  # Extract positive probability
```

### Unified Approach

```python
from prm_server import PrmConfig, create_prm_server

config = PrmConfig(model="Qwen/Qwen2.5-Math-PRM-7B", base_url="http://localhost:8080")
prm = create_prm_server(config)
rewards = prm.score(prompt="...", response="...")
```

**Benefits:**
- 3 lines instead of ~15 lines
- No manual formatting required
- Type-safe configuration
- Consistent error handling
- Works with any supported PRM model

## Troubleshooting

### Server Not Running

**Error:** `RuntimeError: PRM server request failed: Connection refused`

**Solution:**
1. Start the appropriate server:
   ```bash
   # Qwen
   vllm serve Qwen/Qwen2.5-Math-PRM-7B --port 8080 --trust-remote-code

   # Skywork
   python start_reward_server.py
   ```
2. Verify server is running: `curl http://localhost:8080/health`

### Wrong Port

**Error:** `RuntimeError: PRM server request failed: Connection refused`

**Solution:** Check `base_url` matches server port:
- Qwen default: `http://localhost:8080`
- Skywork default: `http://localhost:8081`

### Empty Rewards

**Error:** Empty rewards list `[]`

**Possible Causes:**
1. **Qwen:** No `\n\n` (double newline) delimiters in response
2. **Skywork:** No `\n` (single newline) delimiters in response

**Solution:** Use correct step delimiter for your model:
```python
# Qwen: double newline
response = "Step 1: ...\n\nStep 2: ..."

# Skywork: single newline
response = "Step 1: ...\nStep 2: ..."
```

### Model Not Supported

**Error:** `ValueError: Unknown PRM model: custom/my-model`

**Solution:** Currently supports:
- Models with "Qwen" and "PRM" in name → `QwenPrmServer`
- Models with "Skywork" and "PRM" in name → `SkyworkPrmServer`

To add custom model: Subclass `PrmServer` and implement required methods.

### Request Timeout

**Error:** `RuntimeError: PRM server request failed: Request timeout`

**Solution:** Increase timeout or reduce input size:
```python
config = PrmConfig(
    model="...",
    base_url="...",
    timeout=600  # Increase from default 300s
)
```

## Performance Tips

1. **Reuse PRM instances:** Create once, call `score()` multiple times
   ```python
   prm = create_prm_server(config)
   for prompt, response in data:
       rewards = prm.score(prompt, response)
   ```

2. **Batch at application level:** Process multiple (prompt, response) pairs sequentially
   ```python
   results = [prm.score(p, r) for p, r in zip(prompts, responses)]
   ```

3. **Server scaling:** Run multiple vLLM servers on different ports for parallel processing

4. **Tokenizer caching (Skywork only):** Tokenizer loaded once during initialization

## API Reference

### PrmConfig

```python
@dataclass
class PrmConfig:
    model: str                    # Model identifier
    base_url: str                 # Server HTTP URL
    timeout: int = 300            # Request timeout (seconds)
    trust_remote_code: bool = True
```

### PrmServer

```python
class PrmServer(ABC):
    def score(self, prompt: str, response: str) -> List[float]:
        """Score step-by-step response"""

    def preprocess_input(self, prompt: str, response: str) -> Dict[str, Any]:
        """Model-specific input formatting"""

    def send_request(self, processed_data: Dict[str, Any]) -> Dict[str, Any]:
        """HTTP request to /pooling endpoint"""

    def post_process_output(self, raw_results: Dict[str, Any]) -> List[float]:
        """Extract and normalize rewards"""
```

### Factory Function

```python
def create_prm_server(config: PrmConfig) -> PrmServer:
    """
    Create appropriate PRM server instance based on model name.

    Args:
        config: PrmConfig with model identifier and server URL

    Returns:
        QwenPrmServer or SkyworkPrmServer instance

    Raises:
        ValueError: If model type is not recognized
    """
```

## Examples

See `example_prm_usage.py` for complete working examples with both Qwen and Skywork models.

Run examples:
```bash
# Qwen (start server first on port 8080)
python example_prm_usage.py --model qwen

# Skywork (start server first on port 8081)
python example_prm_usage.py --model skywork
```

## Contributing

To add support for a new PRM model:

1. Create a new class inheriting from `PrmServer`
2. Implement required abstract methods:
   - `model_check()`: Validate model compatibility
   - `_init_tokenizer()`: Initialize tokenizer if needed
   - `preprocess_input()`: Format (prompt, response) for server
   - `post_process_output()`: Extract rewards from server response
3. Update `create_prm_server()` factory with new model detection
4. Add example to `example_prm_usage.py`
5. Update documentation

## License

Apache-2.0
