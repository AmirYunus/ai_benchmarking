import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
import numpy as np
import time
import psutil
import os
import urllib
import tarfile
import pickle
import platform

# Check if CUDA is available and set the device
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

# Define the version of the benchmarking script
version = '2407301136'
# Define the maximum training time in minutes
minutes = 1

# Define a custom callback to monitor training time and stop training when a specified time limit is reached
class TimerCallback:
    """
    A custom callback to monitor training time and stop training when a specified time limit is reached.

    Attributes:
    - max_minutes (int): The maximum allowed training time in minutes.
    - start_time (float): The time when training begins.
    """
    def __init__(self, max_minutes: int):
        self.max_minutes = max_minutes
        self.start_time = 0

    def on_train_begin(self, logs=None):
        """
        Initializes the start time when training begins.
        """
        self.start_time = time.time()

    # def on_epoch_end(self, epoch: int, logs=None):
    #     """
    #     Checks if the elapsed time exceeds the maximum allowed time after each epoch.
    #     If so, stops the training.

    #     Parameters:
    #     - epoch (int): The current epoch.
    #     - logs (dict): The training logs.
    #     """
    #     current_time = time.time()
    #     elapsed_time = current_time - self.start_time

    #     if elapsed_time > self.max_minutes * 60:
    #         self.model.stop_training = True

# Define the local folder to store CIFAR-10 data
local_folder = './data/'

# Function to download and extract the CIFAR-10 dataset
def download_and_extract_cifar10():
    """
    Downloads and extracts the CIFAR-10 dataset to the local folder.
    """
    os.makedirs(local_folder, exist_ok=True)
    url = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
    file_path = os.path.join(local_folder, 'cifar-10-python.tar.gz')
    
    # Download the CIFAR-10 dataset
    urllib.request.urlretrieve(url, file_path)
    
    # Extract the contents of the tar file
    with tarfile.open(file_path, 'r:gz') as tar:
        tar.extractall(local_folder)
    
    # Remove the downloaded tar file
    os.remove(file_path)

# Function to load and preprocess the CIFAR-10 dataset
def load_and_preprocess_data():
    """
    Loads the CIFAR-10 dataset from the local folder, normalizes pixel values,
    and splits the data into training and validation sets.

    Returns:
    - Tuple of numpy arrays: (train_images, train_labels, test_images, test_labels, val_images, val_labels)
    """
    # Check if CIFAR-10 data is already downloaded
    if not os.path.exists(os.path.join(local_folder, 'cifar-10-batches-py')):
        download_and_extract_cifar10()
    
    train_images, train_labels = [], []
    for i in range(1, 6):
        file_path = os.path.join(local_folder, f'cifar-10-batches-py/data_batch_{i}')
        with open(file_path, 'rb') as file:
            batch_data = pickle.load(file, encoding='bytes')
            train_images.append(batch_data[b'data'])
            train_labels.extend(batch_data[b'labels'])
    
    train_images = np.concatenate(train_images, axis=0).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    train_labels = np.array(train_labels)
    
    # Load test data
    file_path = os.path.join(local_folder, 'cifar-10-batches-py/test_batch')
    with open(file_path, 'rb') as file:
        test_data = pickle.load(file, encoding='bytes')
        test_images = test_data[b'data'].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
        test_labels = np.array(test_data[b'labels'])
    
    # Normalize pixel values
    train_images, test_images = train_images / 255.0, test_images / 255.0
    
    # Split into training and validation sets
    train_images, val_images, train_labels, val_labels = train_test_split(
        train_images, train_labels, test_size=0.2, random_state=42
    )
    
    return train_images, train_labels, test_images, test_labels, val_images, val_labels

# Function to build and compile a convolutional neural network (CNN) model using PyTorch
def build_and_compile_model():
    """
    Builds and compiles a convolutional neural network (CNN) model using PyTorch.

    Returns:
    - A PyTorch model object
    """
    model = nn.Sequential(
        nn.Conv2d(3, 32, 3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),
        nn.Conv2d(32, 64, 3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),
        nn.Conv2d(64, 64, 3, padding=1),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(64 * 8 * 8, 1028),
        nn.ReLU(),
        nn.Linear(1028, 10)
    )
    
    # Move the model to the device
    model = model.to(device)
    
    return model

# Function to train the model with EarlyStopping and TimerCallback
def train_model(model, train_images, train_labels, val_images, val_labels, timer_callback):
    """
    Trains the provided model with EarlyStopping and TimerCallback.

    Parameters:
    - model: A PyTorch model object.
    - train_images, train_labels: Numpy arrays containing training data.
    - val_images, val_labels: Numpy arrays containing validation data.
    - timer_callback: An instance of TimerCallback for monitoring training time.

    Returns:
    - The training history object.
    """
    train_dataset = TensorDataset(torch.from_numpy(train_images).float(), torch.from_numpy(train_labels))
    val_dataset = TensorDataset(torch.from_numpy(val_images).float(), torch.from_numpy(val_labels))

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    history = []
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    for epoch in range(minutes * 10):
        for batch in train_loader:
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs.permute(0, 3, 1, 2))
            loss = nn.CrossEntropyLoss()(outputs, labels.long())
            loss.backward()
            optimizer.step()
            history.append({'loss': loss.item()})
        if val_loader:
            model.eval()
            total_correct = 0
            with torch.no_grad():
                for batch in val_loader:
                    inputs, labels = batch
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs.permute(0, 3, 1, 2))
                    _, predicted = torch.max(outputs, 1)
                    total_correct += (predicted == labels).sum().item()
            accuracy = total_correct / len(val_loader.dataset)
            history[-1]['val_accuracy'] = accuracy
            # if timer_callback.on_epoch_end(epoch, history):
            #     break
    return history

# Function to evaluate the model on the test set and calculate F1 score
def evaluate_model(model, test_images, test_labels, minutes, training_time):
    """
    Evaluates the model on the test set and calculates F1 score.

    Parameters:
    - model: A trained PyTorch model.
    - test_images, test_labels: Numpy arrays containing test data.
    - minutes: Minutes registered for TimerCallback
    - history: Training history

    Returns:
    - Tuple: (F1 score, benchmark score)
    """
    test_predictions = np.argmax(model(torch.from_numpy(test_images).permute(0, 3, 1, 2).float().to(device)).cpu().detach().numpy(), axis=1)
    f1 = f1_score(test_labels, test_predictions, average='weighted')
    # benchmark = round(((minutes * 60) / training_time) * ((minutes * 10) - len(history.history['accuracy'])) * 100)
    benchmark = round((((minutes * 60) / training_time) * 1_000) / 3)
    return f1, benchmark

# Function to retrieve information about the system's CPU, memory, and disk
def get_system_information():
    """
    Retrieves information about the system's CPU, memory, and disk.

    Returns:
    - Tuple of dictionaries: (cpu_info, memory_info, disk_info)
    """
    cpu_info = {
        'Platform': platform.platform(),
        'CPU Usage (%)': round(min(max([x / psutil.cpu_count() * 100 for x in psutil.getloadavg()]), 100), 1),
        'CPU Cores': psutil.cpu_count(logical=False),
    }

    memory_info = {
        'Total Memory (GB)': round(psutil.virtual_memory().total / (1024**3), 2),
        'Memory Usage (%)': psutil.virtual_memory().percent
    }

    disk_info = {
        'Total Disk Space (GB)': round(psutil.disk_usage('/').total / (1024**3), 2),
        'Disk Usage (%)': psutil.disk_usage('/').percent
    }

    return cpu_info, memory_info, disk_info

# Function to save benchmark information to a text file
def save_benchmark_report(filename, cpu_info, memory_info, disk_info, f1, training_time, benchmark):
    """
    Saves benchmark information to a text file.

    Parameters:
    - filename (str): The name of the file to save the benchmark report.
    - cpu_info, memory_info, disk_info: Dictionaries containing system information.
    - f1 (float): F1 score.
    - training_time (float): Time taken for training.
    - benchmark (int): Benchmark score.
    """
    with open(filename, 'w') as file:
        file.write(f"===== AI Benchmarking | Version: {version} =====\n")
        file.write("===== CPU Information =====\n")
        for key, value in cpu_info.items():
            file.write(f"{key}: {value}\n")

        file.write("\n===== Memory Information =====\n")
        for key, value in memory_info.items():
            file.write(f"{key}: {value}\n")

        file.write("\n===== Disk Information =====\n")
        for key, value in disk_info.items():
            file.write(f"{key}: {value}\n")

        file.write("\n===== Benchmark Results =====\n")
        file.write(f'F1 Score (%): {round(f1 * 100, 2)}\n')
        file.write(f'Training Time (s): {int(round(training_time, 0))}\n')
        file.write(f'Benchmark Score: {int(benchmark)}')

# Function to print system information and benchmark results to the console
def print_system_information(cpu_info, memory_info, disk_info, f1, training_time, benchmark):
    """
    Prints system information and benchmark results to the console.

    Parameters:
    - cpu_info, memory_info, disk_info: Dictionaries containing system information.
    - f1 (float): F1 score.
    - training_time (float): Time taken for training.
    - benchmark (int): Benchmark score.
    """
    print("===== CPU Information =====")
    for key, value in cpu_info.items():
        print(f"{key}: {value}")

    print("\n===== Memory Information =====")
    for key, value in memory_info.items():
        print(f"{key}: {value}")

    print("\n===== Disk Information =====")
    for key, value in disk_info.items():
        print(f"{key}: {value}")

    print("\n===== Benchmark Results =====")
    print(f'F1 Score (%): {round(f1 * 100, 2)}')
    print(f'Training Time (s): {int(round(training_time, 0))}')
    print(f'Benchmark Score: {int(benchmark)}\n')

# Main function that orchestrates the entire benchmarking process
def main():
    """
    The main function that orchestrates the entire benchmarking process.
    """
    # Load and preprocess data
    train_images, train_labels, test_images, test_labels, val_images, val_labels = load_and_preprocess_data()

    # Build and compile the model
    model = build_and_compile_model()

    # Define TimerCallback with a maximum time of 5 minutes
    timer_callback = TimerCallback(max_minutes=minutes)

    # Print version information
    print(f'\nAI Benchmarking | Version: {version}')
    if device.type == 'cuda' or device.type == 'mps':
        print(f'This test will take up to {minutes + 1} minutes on {device.type.upper()}\n')
    else:
        print(f'This test will take up to {minutes * 2} minutes on {device.type.upper()}\n')

    start_time = time.time()

    # Train the model with EarlyStopping and TimerCallback
    train_model(model, train_images, train_labels, val_images, val_labels, timer_callback)

    end_time = time.time()

    # Calculate training time
    training_time = end_time - start_time

    # Evaluate the model on the test set
    f1, benchmark = evaluate_model(model, test_images, test_labels, minutes, training_time)

    # Get system information
    cpu_info, memory_info, disk_info = get_system_information()

    # Define the filename for the benchmark report
    filename = './benchmark_report.txt'

    # Save benchmark information to the file
    save_benchmark_report(filename, cpu_info, memory_info, disk_info, f1, training_time, benchmark)

    # Print benchmark information to the console
    print_system_information(cpu_info, memory_info, disk_info, f1, training_time, benchmark)

    # Print the location where the benchmark report is saved
    print(f"Benchmark report saved to {filename}")

if __name__ == "__main__":
    torch.manual_seed(786)
    main()