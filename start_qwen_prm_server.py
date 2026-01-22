#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Server startup script for Qwen2.5-Math-PRM-7B with vLLM
#
# Usage:
#   python start_qwen_prm_server.py [vllm serve arguments]
#
# Example:
#   python start_qwen_prm_server.py \
#       --model Qwen/Qwen2.5-Math-PRM-7B \
#       --host 0.0.0.0 \
#       --port 8082 \
#       --tensor-parallel-size 1 \
#       --gpu-memory-utilization 0.9 \
#       --trust-remote-code

import sys
import subprocess


def start_vllm_server(args):
    """Start vLLM server with the provided arguments"""
    # Default arguments if none provided
    default_args = [
        "Qwen/Qwen2.5-Math-PRM-7B",
        "--runner", "pooling",
        "--host", "0.0.0.0",
        "--port", "8082",
        "--tensor-parallel-size", "1",
        "--gpu-memory-utilization", "0.9",
        "--enable-prefix-caching",
        "--dtype", "auto",
        "--trust-remote-code"
    ]

    # Use provided arguments if any, otherwise use defaults
    server_args = args if args else default_args

    print(f"\nStarting vLLM server with arguments: {' '.join(server_args)}\n")
    print("=" * 80)
    print("IMPORTANT: Qwen2.5-Math-PRM-7B uses STEP pooling with <extra_0> tokens")
    print("Make sure to include <extra_0> tokens in your input strings!")
    print("=" * 80)

    # Start vLLM server using subprocess
    cmd = ["vllm", "serve"] + server_args

    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\n\nServer stopped by user (Ctrl+C)")
    except subprocess.CalledProcessError as e:
        print(f"\nError: vLLM server failed with exit code {e.returncode}")
        sys.exit(e.returncode)


def main():
    """Main entry point"""
    print("=" * 80)
    print("Qwen2.5-Math-PRM-7B vLLM Server Startup")
    print("=" * 80)
    print()

    # Get command-line arguments (skip script name)
    args = sys.argv[1:]

    # Start the vLLM server
    start_vllm_server(args)


if __name__ == "__main__":
    main()
