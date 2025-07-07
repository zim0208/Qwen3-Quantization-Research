# Qwen3-Quantization-Research

A comprehensive evaluation of quantization methods for Qwen3 models, comparing BF16, FP8, and FP4 precision formats.

## Quick Setup

Start the pre-built container:

```bash
docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
    -v $(pwd):/workspace/project \
    -it nvcr.io/nvidia/tritonserver:25.05-vllm-python-py3 /bin/bash
```

Verify environment:

```bash
python3 -c "import vllm; print(f'vLLM: {vllm.__version__}')"
python3 -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')"
```

## Prerequisites

**Hardware Requirements:**
- Modern NVIDIA GPU with 16GB+ VRAM
- 16GB+ system RAM
- 20GB+ storage space

*Note: NVFP4 quantization requires Blackwell architecture (RTX 50-series) for optimal performance*

**Software Requirements:**
- Docker with GPU support
- NVIDIA drivers 570.86+
- NVIDIA Container Toolkit

## Setup

**1. Start Container:**
```bash
docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
    -v $(pwd):/workspace/project \
    -it nvcr.io/nvidia/tritonserver:25.05-vllm-python-py3 /bin/bash
```

**2. Test Installation:**
```bash
python3 -c "
from vllm import LLM
llm = LLM('microsoft/DialoGPT-medium', dtype='bfloat16')
outputs = llm.generate(['Hello'], max_tokens=5)
print('✅ Setup successful!')
"
```

## Docker Installation

**Linux:**
```bash
curl -fsSL https://get.docker.com -o get-docker.sh && sudo sh get-docker.sh
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

**Windows:**
Install Docker Desktop and enable WSL2 integration.

## Benchmark Evaluation

### Math-500
- **Prompt:** Math Question + "Please reason step by step, and put your final answer within \boxed{}."
- **Parameters:** 
  - Thinking mode: temperature=0.6, top_p=0.95, top_k=20
  - Non-thinking mode: temperature=0.7, top_p=0.8, top_k=20, presence_penalty=1.5
  - Max sequence length: 32768
- **Evaluation:** Math-verify framework

### Performance Targets

| Model | Baseline Score | NVFP8 Target | NVFP4 Target |
|-------|---------------|--------------|--------------|
| Qwen3-4B | 83.7% | >79.5% | >75.3% |
| Qwen3-8B | 87.5% | >83.1% | >78.8% |

**Quantization Goals:**
- **NVFP8:** Maintain >95% of baseline performance
- **NVFP4:** Maintain >90% of baseline performance  
- **Accuracy threshold:** <5% degradation from BF16

## Troubleshooting

**GPU not detected:**
```bash
nvidia-smi  # Check driver
docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi  # Test Docker GPU access
```

**Out of memory:**
```bash
# Reduce memory usage
llm = LLM('model_name', dtype='bfloat16', gpu_memory_utilization=0.7)
```

**Container exits:**
```bash
# Check NVIDIA Container Toolkit installation
docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi
```
