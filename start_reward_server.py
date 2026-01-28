#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Server startup script for Skywork-o1-Open-PRM with vLLM 0.14.1
#
# Prerequisites:
#   pip install -e .  # Install the vLLM plugin for Skywork PRM
#
# Usage:
#   python start_reward_server.py [vllm serve arguments]
#
# Example:
#   python start_reward_server.py \
#       --model Skywork/Skywork-o1-Open-PRM-Qwen-2.5-1.5B \
#       --host 0.0.0.0 \
#       --port 8081 \
#       --tensor-parallel-size 1 \
#       --gpu-memory-utilization 0.9 \
#       --trust-remote-code

import sys
import subprocess


def start_vllm_server(args):
    """Start vLLM server with the provided arguments"""
    # Default arguments if none provided
    default_args = [
        "Skywork/Skywork-o1-Open-PRM-Qwen-2.5-1.5B",
        "--host", "0.0.0.0",
        "--port", "8081",
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

    # Start vLLM server using subprocess
    # The Skywork PRM model will be automatically registered via the vLLM plugin system
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
    print("Skywork-o1-Open-PRM vLLM Server Startup")
    print("=" * 80)
    print()

    # Get command-line arguments (skip script name)
    args = sys.argv[1:]

    # Start the vLLM server
    start_vllm_server(args)


if __name__ == "__main__":
    main()
