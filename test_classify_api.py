#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Test vLLM classify/pooling API directly

import requests
import json
from transformers import AutoTokenizer
from skywork_utils import prepare_input

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    "Skywork/Skywork-o1-Open-PRM-Qwen-2.5-1.5B",
    trust_remote_code=True
)

# Prepare a simple test case
data = {
    "problem": "What is 2+2?",
    "response": "Let's think step by step.\nFirst, 2+2 equals 4.\nTherefore, the answer is 4."
}

input_ids, steps, reward_flags = prepare_input(
    data["problem"],
    data["response"],
    tokenizer=tokenizer,
    step_token="\n"
)

print(f"Input IDs length: {len(input_ids)}")
print(f"Number of steps: {len(steps)}")
print(f"Reward flags: {reward_flags}")

# Test different API endpoints
base_url = "http://localhost:8082"

# Test 1: /pooling endpoint
print("\n" + "=" * 80)
print("TEST 1: /pooling endpoint")
print("=" * 80)
try:
    response = requests.post(
        f"{base_url}/pooling",
        json={"input": input_ids},
        headers={"Content-Type": "application/json"}
    )
    print(f"Status: {response.status_code}")
    print(f"Response: {response.text[:500]}")
except Exception as e:
    print(f"Error: {e}")

# Test 2: /classify endpoint
print("\n" + "=" * 80)
print("TEST 2: /classify endpoint")
print("=" * 80)
try:
    response = requests.post(
        f"{base_url}/classify",
        json={"input": input_ids},
        headers={"Content-Type": "application/json"}
    )
    print(f"Status: {response.status_code}")
    print(f"Response: {response.text[:500]}")
except Exception as e:
    print(f"Error: {e}")

# Test 3: /v1/embeddings endpoint with token IDs
print("\n" + "=" * 80)
print("TEST 3: /v1/embeddings endpoint (token IDs)")
print("=" * 80)
try:
    response = requests.post(
        f"{base_url}/v1/embeddings",
        json={
            "input": [input_ids],
            "model": "Skywork/Skywork-o1-Open-PRM-Qwen-2.5-1.5B"
        },
        headers={"Content-Type": "application/json"}
    )
    print(f"Status: {response.status_code}")
    print(f"Response: {response.text[:500]}")
    if response.status_code == 200:
        data = response.json()
        print(f"\nFull response structure:")
        print(json.dumps(data, indent=2)[:1000])
except Exception as e:
    print(f"Error: {e}")

# Test 4: /v1/embeddings endpoint with text
print("\n" + "=" * 80)
print("TEST 4: /v1/embeddings endpoint (text)")
print("=" * 80)
try:
    text = tokenizer.decode(input_ids)
    response = requests.post(
        f"{base_url}/v1/embeddings",
        json={
            "input": [text],
            "model": "Skywork/Skywork-o1-Open-PRM-Qwen-2.5-1.5B"
        },
        headers={"Content-Type": "application/json"}
    )
    print(f"Status: {response.status_code}")
    print(f"Response: {response.text[:500]}")
except Exception as e:
    print(f"Error: {e}")
