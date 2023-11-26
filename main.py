import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
import numpy as np
import time
import psutil

class TimerCallback(callbacks.Callback):
    """
    A custom callback to monitor training time and stop training when a specified time limit is reached.

    Attributes:
    - max_minutes (int): The maximum allowed training time in minutes.
    - start_time (float): The time when training begins.
    """
    def __init__(self, max_minutes):
        super().__init__()
        self.max_minutes = max_minutes
        self.start_time = 0

    def on_train_begin(self, logs=None):
        """
        Initializes the start time when training begins.
        """
        self.start_time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        """
        Checks if the elapsed time exceeds the maximum allowed time after each epoch.
        If so, stops the training.

        Parameters:
        - epoch (int): The current epoch.
        - logs (dict): The training logs.
        """
        current_time = time.time()
        elapsed_time = current_time - self.start_time

        if elapsed_time > self.max_minutes * 60:
            self.model.stop_training = True

def load_and_preprocess_data():
    """
    Loads the CIFAR-10 dataset, normalizes pixel values, and splits the data into training and validation sets.

    Returns:
    - Tuple of numpy arrays: (train_images, train_labels, test_images, test_labels, val_images, val_labels)
    """
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()
    train_images, test_images = train_images / 255.0, test_images / 255.0
    train_images, val_images, train_labels, val_labels = train_test_split(
        train_images, train_labels, test_size=0.2, random_state=42
    )
    return train_images, train_labels, test_images, test_labels, val_images, val_labels

def build_and_compile_model():
    """
    Builds and compiles a convolutional neural network (CNN) model using TensorFlow/Keras.

    Returns:
    - A Keras model object
    """
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10))

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    return model

def train_model(model, train_images, train_labels, val_images, val_labels, timer_callback):
    """
    Trains the provided model with EarlyStopping and TimerCallback.

    Parameters:
    - model: A Keras model object.
    - train_images, train_labels: Numpy arrays containing training data.
    - val_images, val_labels: Numpy arrays containing validation data.
    - timer_callback: An instance of TimerCallback for monitoring training time.

    Returns:
    - The training history object.
    """
    history = model.fit(
        train_images,
        train_labels,
        epochs=1000,
        validation_data=(val_images, val_labels),
        callbacks=[callbacks.EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True), timer_callback],
        verbose=0
    )
    return history

def evaluate_model(model, test_images, test_labels, training_time, history):
    """
    Evaluates the model on the test set and calculates F1 score.

    Parameters:
    - model: A trained Keras model.
    - test_images, test_labels: Numpy arrays containing test data.
    - training_time: Time spent on training
    - history: Training history

    Returns:
    - Tuple: (F1 score, benchmark score)
    """
    test_predictions = np.argmax(model.predict(test_images, verbose=0), axis=1)
    f1 = f1_score(test_labels, test_predictions, average='weighted')
    benchmark = round((1 / training_time) * f1 * 10000 * len(history.history['accuracy']), 0)
    return f1, benchmark

def get_system_information():
    """
    Retrieves information about the system's CPU, memory, and disk.

    Returns:
    - Tuple of dictionaries: (cpu_info, memory_info, disk_info)
    """
    cpu_info = {
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

def main():
    """
    The main function that orchestrates the entire benchmarking process.
    """
    # Load and preprocess data
    train_images, train_labels, test_images, test_labels, val_images, val_labels = load_and_preprocess_data()

    # Build and compile the model
    model = build_and_compile_model()

    # Define TimerCallback with a maximum time of 5 minutes
    minutes = 5
    timer_callback = TimerCallback(max_minutes=minutes)

    # Print version information
    print(f'\nAI Benchmarking | Version: 2311261655')
    print(f'This test will take up to {minutes + 5} minutes\n')

    start_time = time.time()

    # Train the model with EarlyStopping and TimerCallback
    history = train_model(model, train_images, train_labels, val_images, val_labels, timer_callback)

    end_time = time.time()

    # Calculate training time
    training_time = end_time - start_time

    # Evaluate the model on the test set
    f1, benchmark = evaluate_model(model, test_images, test_labels, training_time, history)

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
    main()