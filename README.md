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
| Windows 10       | AMD Ryzen 7 7700X 8-Core Processor | 8 (16 threads) | 31.21 GB | NVIDIA GeForce RTX 4070 Ti SUPER | 12.4 | 8 | 15.99 | 2.5.1+cu124 | 96.57 | 0.7906 | 0.4473 |
| Windows 10       | AMD Ryzen 7 5800H with Radeon Graphics | 8 (16 threads) | 31.86 GB | NVIDIA GeForce RTX 3070 Laptop GPU | 12.4 | 8 | 8.0 | 2.5.1+cu124 | 198.04 | 0.7895 | 0.2178 |
| Darwin 24.2.0    | Apple M2 Pro   | 10 (10 threads) | 16.0 GB  | M2 Pro    | N.A.         | N.A.       | N.A. | 2.5.1 | 397.97 | 0.7896 | 0.1084 |
| Darwin 24.2.0    | Apple M3 Pro   | 11 (11 threads) | 36.0 GB  | M3 Pro    | N.A.         | N.A.       | N.A. | 2.5.1 | 411.66 | 0.7961 | 0.1057 |
| Darwin 24.2.0    | Intel(R) Core(TM) i7-8569U CPU @ 2.80GHz | 4 (8 threads) | 16.0 GB | None | N.A. | N.A. | N.A. | 2.2.2 | 5137.11 | 0.7918 | 0.0084 |

### Results Analysis

The benchmark results reveal significant performance variations across different hardware configurations. The NVIDIA RTX 4070 Ti SUPER emerges as the clear leader, completing the training in just 54.64 seconds on Linux 6.8.0-49-generic. This performance is approximately half the time of the RTX 3070 Laptop GPU, which took 198.04 seconds, and about a quarter of the time taken by the Apple Silicon variants, which ranged from 397.97 seconds for the M2 Pro to 411.66 seconds for the M3 Pro.

#### Key Observations:

1. **NVIDIA GPUs vs. Apple Silicon**:
   - The NVIDIA RTX 4070 Ti SUPER demonstrates a substantial advantage in training speed, attributed to its dedicated CUDA cores and VRAM, which are optimized for deep learning tasks. This highlights the continued superiority of dedicated GPUs over integrated solutions, particularly for computationally intensive workloads like neural network training.
   - The RTX 3070 Laptop GPU, while slower than the 4070 Ti, still outperforms both Apple Silicon variants, indicating that NVIDIA's architecture is more efficient for this type of task.

2. **Comparison of RTX 4070 Ti SUPER on Different OS**:
   - The performance of the RTX 4070 Ti SUPER varies slightly between operating systems. On **Linux 6.8.0-49-generic**, it completed the training in **54.64 seconds**, while on **Windows 10**, the same GPU took **96.57 seconds**. This indicates that the Linux environment may provide better optimization for deep learning tasks, likely due to more efficient resource management and lower overhead compared to Windows.
   - The difference in training times suggests that users running deep learning workloads on NVIDIA GPUs may benefit from using Linux, especially for large-scale projects where training time is critical.

3. **Apple Silicon Performance**:
   - Among the Apple Silicon chips, the M2 Pro slightly edges out the M3 Pro in terms of training time (397.97 seconds vs. 411.66 seconds). This is intriguing given that the M3 Pro has more cores and RAM, suggesting that architectural differences or optimizations in the M2 Pro may play a role in its performance.
   - Despite the M3 Pro's longer training time, it achieves a marginally higher F1 score (0.7961 vs. 0.7896), indicating that while it may take longer to train, it could potentially yield better model performance in terms of classification accuracy.

4. **Consistency in F1 Scores**:
   - The small variation in F1 scores across all devices (ranging from 0.7895 to 0.7961) suggests that the model's performance is relatively consistent across different hardware configurations. This indicates that while training speed varies significantly, the underlying model architecture and training process are robust enough to deliver similar performance metrics regardless of the hardware used.
   - The F1 score is a critical metric for evaluating model performance, especially in classification tasks, as it considers both precision and recall. The consistency in these scores across diverse hardware suggests that users can expect reliable performance from the VGG16 model, regardless of whether they are using high-end NVIDIA GPUs or Apple Silicon.

5. **Implications for Users**:
   - For users prioritizing training speed, investing in high-performance NVIDIA GPUs is advisable, especially for large-scale deep learning tasks. The results clearly demonstrate the advantages of dedicated hardware in reducing training times.
   - Conversely, users with Apple Silicon may still achieve satisfactory results, particularly for smaller datasets or less time-sensitive applications. The trade-off between training time and model performance should be considered based on the specific use case.

Overall, these results underscore the importance of hardware selection in deep learning tasks and provide valuable insights for users looking to optimize their benchmarking and training processes.

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
python -m pip install --force-reinstall -r requirements.txt
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