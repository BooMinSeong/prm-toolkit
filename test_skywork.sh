#!/bin/bash
# Test script for Skywork-o1-Open-PRM with vLLM 0.14.1
# Updated to use custom model implementation

echo "=========================================="
echo "Testing Skywork-o1-Open-PRM Implementation"
echo "=========================================="
echo ""

echo "Step 1: Checking Python environment..."
which python
python --version
echo ""

echo "Step 2: Verifying custom model file..."
if [ -f "skywork_prm_model.py" ]; then
    echo "✓ skywork_prm_model.py found"
    python -m py_compile skywork_prm_model.py && echo "✓ Syntax check passed"
else
    echo "✗ skywork_prm_model.py not found!"
    exit 1
fi
echo ""

echo "Step 3: Testing custom model registration..."
python -c "
from skywork_prm_model import register_skywork_prm_model
from vllm import ModelRegistry

print('Available architectures before:', len(ModelRegistry.get_supported_archs()))
register_skywork_prm_model()
print('Available architectures after:', len(ModelRegistry.get_supported_archs()))
print('Qwen2ForPrmModel registered:', 'Qwen2ForPrmModel' in ModelRegistry.get_supported_archs())
"
echo ""

echo "Step 4: Running reward_skywork_o1_prm.py..."
echo "Command: CUDA_VISIBLE_DEVICES=1 python reward_skywork_o1_prm.py"
echo ""
CUDA_VISIBLE_DEVICES=1 python reward_skywork_o1_prm.py

echo ""
echo "=========================================="
echo "Test completed!"
echo "=========================================="
