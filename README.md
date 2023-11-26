# AI Benchmarking Script

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Generic badge](https://img.shields.io/badge/Python-3.9.18-Green.svg)](https://shields.io/)
[![Generic badge](https://img.shields.io/badge/Mac_OS-Sonoma_14.2_(M1)-Green.svg)](https://shields.io/)

This Python script is designed for benchmarking a convolutional neural network (CNN) model using TensorFlow/Keras on the CIFAR-10 dataset. The script includes functionality for monitoring training time, early stopping, and system resource utilization.

## Prerequisites

Ensure you have the required dependencies by running the following command:

```bash
pip install -r requirements.txt
```

The requirements.txt file includes the necessary dependencies:

```plaintext
tensorflow
scikit-learn
numpy
psutil
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

The script will load the CIFAR-10 dataset, preprocess the data, build and train a CNN model, and then evaluate the model's performance. The training process is monitored by a custom callback that stops training if a specified time limit is reached.

## Custom Callbacks

### `TimerCallback`
A custom callback (`TimerCallback`) is implemented to monitor training time and stop training if a specified time limit is exceeded. The maximum training time can be set by modifying the `max_minutes` parameter in the `TimerCallback` instantiation.

```python
# Example: Set maximum training time to 10 minutes
timer_callback = TimerCallback(max_minutes=10)
```

### `EarlyStopping`
The script utilises the `EarlyStopping` callback from `TensorFlow` / `Keras` to halt training when the validation accuracy plateaus. Parameters such as `monitor`, `patience`, and `restore_best_weights` can be adjusted as needed.

```python
# Example: Adjust EarlyStopping parameters
callbacks.EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True)
```

## Benchmark Results

After running the script, the benchmark results will be saved to a text file (`benchmark_report.txt`) and printed to the console. The report includes information about the CPU, memory, and disk usage, as well as the F1 score, training time, and a benchmark score.

## Important Note

The script may consume a significant amount of CPU resources, and it's advised to run it on a system where you can afford to utilise maximum CPU capacity temporarily.

Ensure that you have the necessary permissions to terminate processes on your system.

## License

This project is licensed under the ANU Affero General Public License - see the LICENSE file for details.