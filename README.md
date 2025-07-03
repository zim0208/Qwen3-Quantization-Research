# Qwen3-Quantization-Research
A comprehensive evaluation of quantization methods for Qwen3 models on NVIDIA's Blackwell architecture, comparing BF16, FP8, and FP4 precision formats.

# Environment Setup - Complete Guide

This section documents the **correct and reliable method** for setting up the Qwen3 quantization research environment on RTX 5090, based on extensive testing and community validation.

## ⚡ Quick Setup (Recommended - 15 Minutes)

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
GPU: NVIDIA GeForce RTX 5090 [Desktop/Laptop]
BF16: True, CUDA: (12, 0)
```

## 📋 Prerequisites

### Hardware Requirements
- **GPU:** RTX 5090 (Desktop or Laptop)
- **VRAM:** 24GB (automatic detection)
- **System RAM:** 16GB+ recommended for Docker
- **Storage:** 20GB+ free space for models and datasets

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

## ✅ Method 1: Pre-built Container (Recommended)

### Why This Method Works
- ✅ **No compilation required** - saves 4-6 hours
- ✅ **Professional NVIDIA optimization** for RTX 5090
- ✅ **Pre-solved Flash Attention 3 issues**
- ✅ **Community tested and verified**
- ✅ **All quantization kernels included**
- ✅ **Perfect for research work**

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
- RTX 5090 detected with ~24GB VRAM
- BF16 support: True
- CUDA capability: (12, 0) - Blackwell architecture
- Model loads and generates text successfully

## ❌ Method 2: Compilation from Source (Not Recommended)

### Why This Method Often Fails
- ❌ **6+ hour compilation time**
- ❌ **High memory requirements** (64GB+ RAM needed)
- ❌ **Flash Attention 3 memory crashes**
- ❌ **Complex troubleshooting required**
- ❌ **Hardware limitations on typical systems**

### The Problems We Encountered

**Memory Issues:**
```bash
# Typical compilation failure
nvcc error : '"$CICC_PATH/cicc"' died due to signal 9 (Kill signal)

# Why it happens:
Flash Attention 3 files: 12-16GB RAM each during compilation
RTX 5090 laptop systems: Limited to 32GB container memory
Even MAX_JOBS=1: Single files exceed memory limits
```

**Build Complexity:**
- 345 files to compile (FA2 + FA3)
- Multiple restart cycles required
- Environment variable confusion
- CMake cache issues

### If You Must Compile from Source

**⚠️ Warning:** Only attempt if you have 64GB+ system RAM and specific customization needs.

```bash
# Use basic PyTorch container
docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
    --memory=64g -it nvcr.io/nvidia/pytorch:25.02-py3 /bin/bash

# Clone and setup
git clone https://github.com/vllm-project/vllm.git && cd vllm
python use_existing_torch.py
pip install -r requirements/build.txt
pip install setuptools_scm
apt-get update && apt-get install ccache -y
mkdir /tmp/ccache

# Force Flash Attention 2 only (avoid FA3 memory issues)
export VLLM_FLASH_ATTN_VERSION=2
export VLLM_BUILD_FA3=0

# Conservative build (single-threaded)
MAX_JOBS=1 CCACHE_DIR=/tmp/ccache python setup.py develop
```

**Expected Compilation Time:** 2-4 hours minimum

## 🧪 Environment Testing

### Automated Test Suite
```bash
# Run comprehensive environment tests
python3 tests/test_environment.py
```

### Manual Verification Checklist
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

## 🔧 Advanced Configuration

### For RTX 5090 Desktop
```bash
# Maximum performance configuration
llm = LLM(
    model='Qwen/Qwen2.5-1.5B-Instruct',
    dtype='bfloat16',
    gpu_memory_utilization=0.95,  # Use most VRAM
    max_model_len=32768,
    tensor_parallel_size=1
)
```

### For RTX 5090 Laptop
```bash
# Conservative configuration (thermal management)
llm = LLM(
    model='Qwen/Qwen2.5-1.5B-Instruct',
    dtype='bfloat16',
    gpu_memory_utilization=0.8,   # Thermal headroom
    max_model_len=16384,          # Reduced context
    tensor_parallel_size=1
)
```

### Custom Container Persistence
```bash
# Save container state for reuse
docker commit <container_id> my-qwen3-research:latest

# Run saved container
docker run --gpus all -it my-qwen3-research:latest /bin/bash
```

## 📊 Performance Expectations

### Typical Performance on RTX 5090
- **Model Loading:** 3-8 seconds for 1.5B parameter model
- **Inference Speed:** 20-50 tokens/second (depends on model size)
- **Memory Usage:** 3-8GB VRAM for 1.5B model in BF16
- **Context Length:** Up to 32K tokens supported

### Quantization Performance Targets
- **NVFP8:** 1.5-2x speed improvement, 50% memory reduction
- **NVFP4:** 2-3x speed improvement, 75% memory reduction
- **Accuracy:** <5% degradation on Math-500 benchmark

## 🎯 Success Criteria

**Your environment is ready when:**
1. ✅ All verification tests pass
2. ✅ BF16 baseline model loads and runs
3. ✅ Math-500 evaluation completes successfully
4. ✅ Memory usage is stable and reasonable
5. ✅ Performance meets expected benchmarks

## 📚 Additional Resources

- **Official vLLM Docs:** [https://docs.vllm.ai/](https://docs.vllm.ai/)
- **RTX 5090 Setup Issue:** [vLLM GitHub #14452](https://github.com/vllm-project/vllm/issues/14452)
- **NVIDIA Container Toolkit:** [Installation Guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- **Docker Installation:** [https://docs.docker.com/get-docker/](https://docs.docker.com/get-docker/)

---

**Environment Setup Summary:**
- ✅ **Recommended:** Pre-built container (15 minutes)
- ❌ **Not Recommended:** Compilation from source (6+ hours, high failure rate)
- 🎯 **Goal:** Working vLLM + RTX 5090 + quantization support
- 📊 **Success Rate:** 95%+ with pre-built container method
