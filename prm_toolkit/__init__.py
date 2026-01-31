"""
PRM Toolkit: Unified Process Reward Model inference toolkit for vLLM

Supports:
- Qwen2.5-Math-PRM-7B
- Skywork-o1-Open-PRM-Qwen-2.5-1.5B
"""

from prm_toolkit.prm_server import (
    PrmConfig,
    PrmServer,
    QwenPrmServer,
    SkyworkPrmServer,
    load_prm_server,
)

__all__ = [
    "PrmConfig",
    "PrmServer",
    "QwenPrmServer",
    "SkyworkPrmServer",
    "load_prm_server",
]

__version__ = "0.1.0"
