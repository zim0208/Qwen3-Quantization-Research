# Qwen3-Quantization-Research
A comprehensive evaluation of quantization methods for Qwen3 models on NVIDIA's Blackwell architecture, comparing BF16, FP8, and FP4 precision formats.

# Environment Setup - Complete Guide

This section documents the **correct and reliable method** for setting up the Qwen3 quantization research environment on RTX 5090, based on extensive testing and community validation.

## ⚡ Quick Setup (15 Minutes)

**Use the pre-built container approach for immediate success:**

```bash
# 1. Start the working container (RTX 5090 optimized)
docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
    -v $(pwd):/workspace/project \
    -it nvcr.io/nvidia/tritonserver:25.05-vllm-python-py3 /bin/bash

# 2. Verify environment (should work immediately)
python3 -c "import vllm; print(f'vLLM: {vllm.__version__}')"
python3 -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')"

# 3. Test quantization support
python3 -c "import torch; print(f'BF16: {torch.cuda.is_bf16_supported()}, CUDA: {torch.cuda.get_device_capability(0)}')"
```

**Expected Output:**
```
vLLM: 0.8.4+dc1a3e10.nv25.05
GPU: NVIDIA GeForce RTX 5090 Laptop GPU
BF16: True, CUDA: (12, 0)
```

## 📋 Prerequisites

### Hardware Requirements
- **GPU:** NVIDIA RTX 5090 (Desktop or Laptop)
- **VRAM:** 24GB (Desktop) or 23.9GB (Laptop) 
- **System RAM:** 16GB+ recommended for Docker
- **Storage:** 20GB+ free space for models and datasets

**Tested Configuration:**
- ✅ RTX 5090 Laptop (23.9GB VRAM) - Fully tested and verified
- ✅ RTX 5090 Desktop - Expected to work (same architecture)

### Software Requirements
- **OS:** Windows 10/11 with WSL2, or Linux
- **Docker:** Docker Desktop 4.0+ or Docker Engine 20.10+
- **NVIDIA Driver:** 570.86+ (required for RTX 5090)
- **NVIDIA Container Toolkit:** Latest version

### Docker Setup (If Not Installed)

**Windows with WSL2:**
```bash
# 1. Install Docker Desktop from https://docker.com/products/docker-desktop
# 2. Enable WSL2 integration in Docker Desktop settings
# 3. Install NVIDIA Container Toolkit:
wsl --install Ubuntu-24.04
# Follow NVIDIA Container Toolkit installation guide
```

**Linux:**
```bash
# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Install NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

## ✅ Environment Setup Method

### Step-by-Step Setup

**1. Start Container:**
```bash
docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
    -v $(pwd):/workspace/project \
    -it nvcr.io/nvidia/tritonserver:25.05-vllm-python-py3 /bin/bash
```

**Container Parameters Explained:**
- `--gpus all` - Access all GPUs (RTX 5090)
- `--ipc=host` - Shared memory for multi-processing
- `--ulimit memlock=-1` - Unlimited locked memory
- `--ulimit stack=67108864` - Large stack size for CUDA
- `-v $(pwd):/workspace/project` - Mount current directory

**2. Environment Verification:**
```bash
# Check vLLM installation
python3 -c "import vllm; print(f'vLLM version: {vllm.__version__}')"

# Verify RTX 5090 detection
python3 -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.version.cuda}')
print(f'GPU: {torch.cuda.get_device_name(0)}')
print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
print(f'BF16 Support: {torch.cuda.is_bf16_supported()}')
print(f'CUDA Capability: {torch.cuda.get_device_capability(0)}')
print(f'Blackwell Architecture: {\"Yes\" if torch.cuda.get_device_capability(0)[0] == 12 else \"No\"}')
"
```

**3. Test Model Loading:**
```bash
# Quick inference test
python3 -c "
from vllm import LLM
print('Testing vLLM with RTX 5090...')
llm = LLM('microsoft/DialoGPT-medium', dtype='bfloat16')
outputs = llm.generate(['Hello, how are you?'], max_tokens=10)
print('✅ Environment setup successful!')
print(f'Response: {outputs[0].outputs[0].text}')
"
```

**Expected Success Indicators:**
- vLLM version: 0.8.4+
- PyTorch version: 2.8.0+
- CUDA version: 12.9
- RTX 5090 detected with ~24GB VRAM (23.9GB on laptop)
- BF16 support: True
- CUDA capability: (12, 0) - Blackwell architecture
- Blackwell Architecture: Yes
- Model loads and generates text successfully

## 🧪 Environment Testing

### Automated Test Suite
```bash
# Run comprehensive environment tests
python3 tests/test_environment.py
```

### Verification Checklist
- [ ] Docker container starts successfully
- [ ] RTX 5090 detected correctly
- [ ] vLLM imports without errors
- [ ] BF16 support enabled
- [ ] CUDA capability shows (12, 0)
- [ ] Model loading works (tested with small model)
- [ ] Inference produces text output
- [ ] Memory usage reasonable (<80% VRAM)

### Performance Baseline Test
```bash
# Establish your system's baseline performance
python3 experiments/baseline/qwen3_bf16_baseline.py

# Run Math-500 evaluation
python3 experiments/evaluation/math500_eval.py --model Qwen/Qwen3-4B --precision bf16
```

## 🐛 Troubleshooting

### Common Issues and Solutions

**Issue: "docker: command not found"**
```bash
# Solution: Install Docker
# Windows: Download Docker Desktop
# Linux: curl -fsSL https://get.docker.com -o get-docker.sh && sudo sh get-docker.sh
```

**Issue: "RuntimeError: No CUDA GPUs are available"**
```bash
# Check NVIDIA driver
nvidia-smi

# Install NVIDIA Container Toolkit if needed
# Follow: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html
```

**Issue: "python: command not found" in container**
```bash
# Use python3 instead of python
python3 -c "import vllm"
```

**Issue: Container exits immediately**
```bash
# Check if GPU is accessible
docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi

# If this fails, check NVIDIA Docker setup
```

**Issue: Out of memory during inference**
```bash
# Use smaller model or reduce memory utilization
llm = LLM('model_name', dtype='bfloat16', gpu_memory_utilization=0.7)
```

**Issue: Model download fails**
```bash
# Check internet connection
# Try smaller model first: 'microsoft/DialoGPT-medium'
# Ensure sufficient disk space (20GB+)
```
# Benchmark Evaluation

## Math-500
* **Prompt Template for Qwen3**
   * Math Question + "Please reason step by step, and put your final answer within \boxed{}."
   * Refer to https://huggingface.co/Qwen/Qwen3-4B for more details
* **Hyper-parameters**
   * https://github.com/QwenLM/Qwen3/issues/1483
   * Thinking mode, use **temperature=0.6, top_p=0.95, and top_k=20**
   * Non-Thinking mode, use **temperature = 0.7, top_p = 0.8, top_k = 20, and presence_penalty = 1.5**
   * Max Sequence Length: **32768**
* **Evaluation Method**
   * **Math-verify https://github.com/huggingface/Math-Verify**

## Performance Results

Based on the comparison shown in Table 17, our quantization research targets the following performance benchmarks:

| Model | Math-500 Score | Architecture | Parameters |
|-------|---------------|--------------|------------|
| Qwen3-8B | 87.5 | Dense | 8B |
| Qwen3-4B | 83.7 | Dense | 4B |
| DeepSeek-R1-Distill-Qwen-14B | 93.9 | Dense | 14B |
| DeepSeek-R1-Distill-Qwen-32B | 94.3 | Dense | 32B |

**Quantization Performance Targets:**
- **NVFP8:** Maintain >95% of baseline Math-500 performance
- **NVFP4:** Maintain >90% of baseline Math-500 performance
- **Accuracy Threshold:** <5% degradation from BF16 baseline

### Math-500 Benchmark Targets
- **Qwen3-4B BF16 Baseline:** 83.7% (target to maintain)
- **Qwen3-8B BF16 Baseline:** 87.5% (aspirational target)
- **NVFP8 Quantized:** >79.5% (>95% of baseline)
- **NVFP4 Quantized:** >75.3% (>90% of baseline)

