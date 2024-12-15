# AI Benchmarking Script

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Generic badge](https://img.shields.io/badge/Python-3.9,_3.11-Green.svg)](https://shields.io/)
[![Generic badge](https://img.shields.io/badge/Win_OS-Win_10_(22H2),_Win_11_(22H2)-Green.svg)](https://shields.io/)
[![Generic badge](https://img.shields.io/badge/Mac_OS-Sonoma_14.2_(M1),_Sequoia_15.2_(M2_Pro,_M3_Pro)-Green.svg)](https://shields.io/)

This Python script is designed for benchmarking a VGG16 neural network model using PyTorch on the CIFAR-10 dataset. The script includes functionality for monitoring training time, system resource utilization, and supports multiple computing devices (CPU, CUDA, Apple Silicon).

## Current Leaderboard

| Timestamp   | OS                | CPU Model      | CPU Cores       | RAM      | GPU Model | CUDA Version | CUDA Cores | VRAM | PyTorch Version | Total Time | F1 Score |
|-------------|-------------------|----------------|------------------|----------|-----------|--------------|------------|------|------------------|------------|----------|
| 2024-12-05  | Windows 10       | AMD Ryzen 7 5800H | 8 (16 threads) | 31.86 GB | RTX 3070 Laptop | 12.4 | 8 | 8.0 GB | 2.5.1+cu124 | 198.04 | 0.7895 |
| 2024-12-05  | Darwin 24.2.0    | Apple M2 Pro   | 10 (10 threads) | 16.0 GB  | M2 Pro    | N.A.         | N.A.       | N.A. | 2.5.1 | 399.32 | 0.7896 |
| 2024-12-05  | Darwin 24.2.0    | Apple M3 Pro   | 11 (11 threads) | 36.0 GB  | M3 Pro    | N.A.         | N.A.       | N.A. | 2.5.1 | 408.91 | 0.7961 |

### Results Analysis

The benchmark results reveal interesting performance characteristics across different hardware configurations. The NVIDIA RTX 3070 Laptop GPU significantly outperforms both Apple Silicon variants, completing the training in 198.04 seconds - less than half the time of either M2 Pro (399.32s) or M3 Pro (408.91s).

Between the Apple Silicon chips, the M2 Pro slightly edges out the M3 Pro in terms of training time (399.32s vs 408.91s), despite the M3 Pro having more cores and RAM. However, the M3 Pro achieves a marginally higher F1 score (0.7961 vs 0.7896). This small variation in F1 scores across all devices (ranging from 0.7895 to 0.7961) suggests that the model's performance is relatively consistent across different hardware configurations, with the main differentiator being training speed.

The RTX 3070 setup's superior performance can be attributed to its dedicated CUDA cores and 8GB VRAM, which are specifically optimized for deep learning workloads. This demonstrates the continued advantage of dedicated GPUs for neural network training tasks, even when compared to Apple's latest integrated solutions.

## Features

- VGG16 architecture implementation from scratch
- Multi-device support (CPU, CUDA GPU, Apple Silicon)
- System information detection and reporting
- Training progress visualization with tqdm
- Benchmark results tracking and leaderboard
- Reproducible results with seed setting
- F1 score evaluation

## Prerequisites

```bash
conda create --prefix=venv python=3.11 -y
conda activate ./venv
python -m pip cache purge # if there are issues with installation or upgrading your conda version
python -m pip install --force-reinstall -r requirements.txt
```

### Additional Requirements
For Windows:
```bash
python -m pip install --default-timeout=1000 --force-reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

## Usage

### Clone the repository:

```bash
git clone https://github.com/AmirYunus/ai-benchmark.git
cd ai-benchmark
```

### Run the script:
```bash
python main.py
```

The script will:
1. Detect available computing devices
2. Display system information
3. Load the CIFAR-10 dataset
4. Train a VGG16 model
5. Evaluate model performance
6. Save results to a CSV file
7. Display a leaderboard of benchmark results

## Benchmark Results

The benchmark results are saved to `benchmark_results.csv` and include:
- Timestamp
- Operating System details
- CPU model and core count
- RAM capacity
- GPU model (if available)
- CUDA information (if available)
- PyTorch version
- Training time and F1 score

The results are automatically sorted to create a leaderboard based on F1 score and total training time.

## Important Notes

- The script will automatically detect and use the best available device (CUDA GPU > Apple Silicon > CPU)
- Training parameters can be adjusted by modifying the `PCBenchmark` class attributes
- The benchmark uses a fixed random seed for reproducibility
- The script may utilize significant system resources during training

## License

This project is licensed under the MIT License - see the LICENSE file for details.