# AI Benchmarking Script

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Generic badge](https://img.shields.io/badge/Python-3.9,_3.11-Green.svg)](https://shields.io/)
[![Generic badge](https://img.shields.io/badge/Win_OS-Win_10_(22H2),_Win_11_(22H2)-Green.svg)](https://shields.io/)
[![Generic badge](https://img.shields.io/badge/Mac_OS-Sonoma_14.2_(M1),_Sequoia_15.2_(M2_Pro,_M3_Pro)-Green.svg)](https://shields.io/)

This Python script is designed for benchmarking a VGG16 neural network model using PyTorch on the CIFAR-10 dataset. The script includes functionality for monitoring training time, system resource utilization, and supports multiple computing devices (CPU, CUDA, Apple Silicon).

## Current Leaderboard

| OS                | CPU Model      | CPU Cores       | RAM      | GPU Model | CUDA Version | CUDA Cores | VRAM | PyTorch Version | Total Time | F1 Score | Benchmark Score |
|-------------------|----------------|------------------|----------|-----------|--------------|------------|------|------------------|------------|----------|------------------|
| Linux 6.8.0-49-generic | AMD Ryzen 7 7700X 8-Core Processor | 8 (16 threads) | 30.56 GB | NVIDIA GeForce RTX 4070 Ti SUPER | 12.4 | 8 | 15.7 | 2.5.1+cu124 | 54.64 | 0.7906 | 0.7906 |
| Windows 10       | Intel(R) Core(TM) i7-14700KF | 20 (28 threads) | 31.84 GB | NVIDIA GeForce RTX 4090 | 12.4 | 8 | 23.99 | 2.5.1+cu124 | 78.75 | 0.7897 | 0.5479 |
| Windows 10       | AMD Ryzen 7 7700X 8-Core Processor | 8 (16 threads) | 31.21 GB | NVIDIA GeForce RTX 4070 Ti SUPER | 12.4 | 8 | 15.99 | 2.5.1+cu124 | 96.57 | 0.7906 | 0.4473 |
| Windows 10       | AMD Ryzen 7 5800H with Radeon Graphics | 8 (16 threads) | 31.86 GB | NVIDIA GeForce RTX 3070 Laptop GPU | 12.4 | 8 | 8.0 | 2.5.1+cu124 | 198.04 | 0.7895 | 0.2178 |
| Darwin 24.2.0    | Apple M2 Pro   | 10 (10 threads) | 16.0 GB  | M2 Pro    | N.A.         | N.A.       | N.A. | 2.5.1 | 397.97 | 0.7896 | 0.1084 |
| Darwin 24.2.0    | Apple M3 Pro   | 11 (11 threads) | 36.0 GB  | M3 Pro    | N.A.         | N.A.       | N.A. | 2.5.1 | 411.66 | 0.7961 | 0.1057 |
| Darwin 24.2.0    | Intel(R) Core(TM) i7-8569U CPU @ 2.80GHz | 4 (8 threads) | 16.0 GB | None | N.A. | N.A. | N.A. | 2.2.2 | 5137.11 | 0.7918 | 0.0084 |

### Results Analysis

The benchmark results reveal significant performance variations across different hardware configurations. The NVIDIA RTX 4070 Ti SUPER on Linux emerges as the leader in benchmark score, while the RTX 4090 on Windows demonstrates impressive raw performance despite system overhead. Training times range from 54.64 seconds for the fastest setup to over 5,000 seconds for CPU-only processing.

#### Key Observations:

1. **High-End NVIDIA GPUs Performance**:
   - The NVIDIA RTX 4070 Ti SUPER on Linux achieves the highest benchmark score, completing training in just 54.64 seconds with an F1 score of 0.7906.
   - The RTX 4090, despite having superior specifications (23.99GB VRAM vs 15.7GB), completed training in 78.75 seconds. This slightly longer time might be attributed to running on Windows rather than Linux, as we see similar OS-based performance differences with other hardware.
   - Both cards demonstrate exceptional performance for deep learning tasks, with the RTX 4090's higher VRAM capacity potentially offering advantages for larger models or batch sizes not tested in this benchmark.

2. **Operating System Impact**:
   - The RTX 4070 Ti SUPER shows a stark performance difference between operating systems: 54.64 seconds on Linux versus 96.57 seconds on Windows, despite identical hardware.
   - This nearly 2x performance gap suggests that Linux environments may offer significant advantages for deep learning workloads, possibly due to better driver optimization and lower system overhead.
   - The pattern indicates that users might achieve substantial performance gains by switching to Linux, particularly for production environments.

3. **Mobile vs Desktop GPU Performance**:
   - The RTX 3070 Laptop GPU, while still powerful, shows the performance gap between mobile and desktop solutions, requiring 198.04 seconds for training.
   - Despite being a mobile variant, it still outperforms both Apple Silicon chips, highlighting the advantages of dedicated GPU architecture for deep learning tasks.

4. **Apple Silicon Performance**:
   - The M2 Pro slightly outperforms the M3 Pro in training time (397.97 vs 411.66 seconds), despite the M3 Pro having more cores and RAM.
   - Interestingly, the M3 Pro achieves the highest F1 score (0.7961) among all tested configurations, suggesting potential benefits in model accuracy despite longer training times.
   - Both chips demonstrate competitive performance for their integrated architecture, though they lag behind dedicated GPUs in raw training speed.

5. **F1 Score Consistency**:
   - F1 scores remain remarkably consistent across all configurations (0.7895-0.7961), indicating that hardware choices primarily affect training speed rather than model quality.
   - This consistency validates the robustness of the VGG16 architecture and training process across different hardware platforms.

6. **Performance Scaling**:
   - The benchmark scores show clear tiers of performance:
     * High-end NVIDIA GPUs on Linux (0.7906)
     * High-end NVIDIA GPUs on Windows (0.5479-0.4473)
     * Mobile NVIDIA GPUs (0.2178)
     * Apple Silicon (0.1084-0.1057)
     * CPU-only (0.0084)
   - This scaling demonstrates the critical importance of hardware selection for deep learning workloads.

7. **Implications for Users**:
   - For maximum performance, a combination of high-end NVIDIA GPU and Linux OS appears optimal.
   - The choice between RTX 4090 and 4070 Ti SUPER might depend more on other workload requirements (like VRAM needs) than raw training speed for this specific benchmark.
   - Apple Silicon devices offer a balanced option for users prioritizing portability and integration over raw training speed.
   - The significant performance gap between GPU and CPU-only training (54.64s vs 5137.11s) emphasizes the necessity of GPU acceleration for practical deep learning work.

Overall, these results provide valuable insights for hardware selection in deep learning applications, highlighting the importance of both hardware choice and operating system in achieving optimal performance.

## Features

- VGG16 architecture implementation from scratch
- Multi-device support (CPU, CUDA GPU, Apple Silicon)
- System information detection and reporting
- Training progress visualization with tqdm
- Benchmark results tracking and leaderboard
- Reproducible results with seed setting
- F1 score evaluation

## Prerequisites

1. Make sure you have conda installed and at least version 24.0.0.
```bash
conda -V
```
> **Note:**
> 
> If you have conda version lower than 24.0.0, you can upgrade it by running:
> ```bash
> conda activate base
> conda update --all
> conda update conda
> ```

2. Create a new conda environment with Python 3.11 and install the required packages.

```bash
conda create --prefix=venv python=3.11 -y
conda activate ./venv
python -m pip cache purge # if there are issues with installation or if you upgraded your conda version to 24.0.0 or higher
python -m pip install --default-timeout=1000 --force-reinstall -r requirements.txt
```

> **Note:**
> 
> There were issues with installing PyTorch and running the script on the Intel(R) Core(TM) i7-8569U on MacOS. If you faced something similar, you may try to use the following requirements file instead:
> 
> ```bash
> python -m pip install --force-reinstall -r requirements_bk.txt
> ```

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

## Disclaimer

The AI benchmarking results presented in this document are based solely on tests conducted for Computer Vision use cases, specifically using the VGG16 neural network model on the CIFAR-10 dataset. While these benchmarks provide valuable insights into the performance of various hardware configurations for this specific application, users should be aware that results may vary significantly for other types of tasks.

Future benchmarks and leaderboard updates may consider additional use cases, including Natural Language Processing (NLP) and other Machine Learning (ML) applications. As the benchmarking framework evolves, we aim to provide a more comprehensive overview of hardware performance across a wider range of AI tasks.

## Citations

If you use this AI benchmarking framework or the results presented in this document in your research or projects, please consider citing it as follows:

```
@misc{ai_benchmark,
  author = {Amir Yunus},
  title = {AI Benchmarking Script for VGG16 on CIFAR-10},
  year = {2024},
  url = {https://github.com/AmirYunus/ai-benchmark},
  note = {Accessed: YYYY-MM-DD}
}
```

Replace "YYYY-MM-DD" with the date you accessed the repository. This helps us track the usage of the benchmark and contributes to the academic community.