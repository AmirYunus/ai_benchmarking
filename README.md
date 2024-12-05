# AI Benchmarking Script

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Generic badge](https://img.shields.io/badge/Python-3.9,_3.11-Green.svg)](https://shields.io/)
[![Generic badge](https://img.shields.io/badge/Win_OS-Win_10_(22H2),_Win_11_(22H2)-Green.svg)](https://shields.io/)
[![Generic badge](https://img.shields.io/badge/Mac_OS-Sonoma_14.2_(M1),_Sequoia_15.2_(M2_Pro,_M3_Pro)-Green.svg)](https://shields.io/)

This Python script is designed for benchmarking a VGG16 neural network model using PyTorch on the CIFAR-10 dataset. The script includes functionality for monitoring training time, system resource utilization, and supports multiple computing devices (CPU, CUDA, Apple Silicon).

## Current Leaderboard

| Timestamp   | OS                | CPU Model      | CPU Cores       | RAM      | GPU Model | CUDA Version | CUDA Cores | VRAM | PyTorch Version | Total Time | F1 Score |
|-------------|-------------------|----------------|------------------|----------|-----------|--------------|------------|------|------------------|------------|----------|
| 2024-12-05  | Darwin 24.2.0    | Apple M3 Pro   | 11 (11 threads)  | 36.0 GB  | M3 Pro    | N.A.         | N.A.       | N.A. | 2.5.1           | 705.11     | 0.8000   |
| 2024-12-05  | Darwin 24.2.0    | Apple M2 Pro   | 10 (10 threads)  | 16.0 GB  | M2 Pro    | N.A.         | N.A.       | N.A. | 2.5.1           | 467.46     | 0.7937   |

### Results Analysis

The M3 Pro device, despite having superior specifications such as 36 GB of RAM, exhibited slower training times compared to the M2 Pro. This could be attributed to several factors, including differences in how the two architectures handle workloads, potential thermal throttling under heavy load, or variations in resource management during training. 

Interestingly, the M3 Pro achieved a higher F1 score, which is unexpected given that the experiments are deterministic with a fixed random seed. This means that the results should be consistent across runs, and better hardware alone should not lead to an increased F1 score. This discrepancy suggests that the M3 Pro may have some advantages in model performance that are not solely related to its hardware specifications.

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
```

Ensure you have the required dependencies by running:

```bash
python -m pip install -r requirements.txt
```

For Windows with CUDA support:
```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
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