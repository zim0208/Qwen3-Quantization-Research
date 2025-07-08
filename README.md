# Qwen3 Quantization Research

This project evaluates quantization methods for Qwen3 models. It compares BF16 and standard FP8 baselines with two quantization techniques: NVIDIA (NV) and Microscaling (MX), both supporting FP8 and FP4 formats. The goal is to measure accuracy, performance, and memory efficiency on GPU systems.

## About

This study benchmarks Qwen3 models using BF16 and FP8 as baselines. It compares them with NV and MX quantization, both offering FP8 and FP4 formats. The results help identify trade-offs in deploying quantized models on GPUs such as Ampere, Hopper, and Blackwell.

## Quick Start

Clone the repository:

git clone https://github.com/zim0208/Qwen3-Quantization-Research.git
cd Qwen3-Quantization-Research

Start all inference services:

docker-compose -f docker-compose.inference.yml up -d

Start only FP8:

docker-compose --profile fp8 -f docker-compose.inference.yml up -d

If using VS Code:

1. Open the repo in VS Code
2. Open Command Palette → “Dev Containers: Attach to Running Container”
3. Select the container to start developing

## Requirements

Hardware:

- NVIDIA GPU with 16 GB+ VRAM
- 16 GB+ RAM
- 20 GB+ disk space
- Blackwell GPU for FP4

Software:

- Docker with GPU support
- NVIDIA Container Toolkit
- NVIDIA driver 570.86+
- (Optional) VS Code with Dev Containers extension

## Benchmark Evaluation

Dataset: Math-500  
Prompt: "Question. Please reason step by step, and put your final answer within \boxed{}."

Generation settings:

Thinking mode:
- temperature: 0.6
- top_p: 0.95
- top_k: 20

Non-thinking mode:
- temperature: 0.7
- top_p: 0.8
- top_k: 20
- presence_penalty: 1.5

Max sequence length: 32768  
Evaluation tool: math-verify

| Model     | BF16 | FP8 | NV FP8 | MX FP8 | NV FP4 | MX FP4 |
|-----------|------|-----|--------|--------|--------|--------|
| Qwen3-4B  | 83.7 | 82.1| >79.5  | >78.6  | >75.3  | >74.1  |
| Qwen3-8B  | 87.5 | 86.0| >83.1  | >82.5  | >78.8  | >77.0  |

Targets:

- NV/MX FP8: ≥95% of BF16
- NV/MX FP4: ≥90% of BF16
- Max drop: <5% from BF16

## Troubleshooting

Check GPU access:

nvidia-smi
docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi

Out of memory:

LLM(model_name, dtype="bfloat16", gpu_memory_utilization=0.7)

Container exits:

docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi

## Structure

- docker-compose.inference.yml: Inference services
- logs/: Inference logs
- notebooks/: Evaluation scripts
- README.md: Documentation
