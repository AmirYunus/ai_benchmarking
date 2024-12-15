import psutil
import platform
import numpy as np
import pandas as pd
import time
from datetime import datetime
import cpuinfo
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from tqdm import tqdm
import warnings
import random
import hashlib
import subprocess
import re

warnings.filterwarnings('ignore')

def set_seed(seed=42):
    """Set seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def _create_vgg16():
    """Create VGG16 architecture from scratch"""
    cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
    
    layers = []
    in_channels = 3
    for x in cfg:
        if x == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            layers += [
                nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                nn.BatchNorm2d(x),
                nn.ReLU(inplace=True)
            ]
            in_channels = x
            
    return nn.Sequential(
        *layers,
        nn.AdaptiveAvgPool2d((7, 7)),
        nn.Flatten(),
        nn.Linear(512 * 7 * 7, 4096),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(4096, 4096),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(4096, 10)
    )

class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        self.features = _create_vgg16()
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.features(x)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

class PCBenchmark:
    def __init__(self):
        self.results_file = "benchmark_results.csv"
        self.system_info = self._get_system_info()
        self.available_devices = self._detect_devices()
        self.num_epochs = 5
        self.batch_size = 128
        set_seed()
        
    def _detect_devices(self):
        """Detect available computing devices"""
        devices = {'cpu': True}
        
        # Check for CUDA
        devices['cuda'] = torch.cuda.is_available()
        if devices['cuda']:
            devices['cuda_device'] = torch.cuda.get_device_name(0)
            devices['cuda_count'] = torch.cuda.device_count()
        
        # Check for MPS (Apple Silicon)
        devices['mps'] = torch.backends.mps.is_available()
        
        return devices
        
    def _get_apple_silicon_info(self):
        """Get detailed information about Apple Silicon chip"""
        try:
            import subprocess
            
            # Run system_profiler command to get detailed hardware info
            cmd = ['system_profiler', 'SPHardwareDataType']
            result = subprocess.run(cmd, capture_output=True, text=True)
            output = result.stdout
            
            # Look for the chip information
            for line in output.split('\n'):
                if 'Chip' in line:
                    # Extract chip info (e.g., "Apple M1 Pro" or "Apple M2 Max")
                    chip_info = line.split(':')[1].strip()
                    # Remove "Apple " prefix if present
                    chip_info = chip_info.replace('Apple ', '')
                    return chip_info
            
            return "Apple Silicon (Unknown Model)"
        except Exception as e:
            return "Apple Silicon (Detection Failed)"
        
    def _get_system_info(self):
        """Gather system specifications"""
        cpu_info = cpuinfo.get_cpu_info()
        system_info = {
            'cpu_model': cpu_info['brand_raw'],
            'cpu_cores': psutil.cpu_count(logical=False),
            'cpu_threads': psutil.cpu_count(logical=True),
            'ram_gb': round(psutil.virtual_memory().total / (1024**3), 2),
            'os': f"{platform.system()} {platform.release()}",
            'python_version': platform.python_version()
        }
        
        # Add GPU information if available
        if torch.cuda.is_available():
            system_info['gpu_model'] = torch.cuda.get_device_name(0)
        elif torch.backends.mps.is_available():
            system_info['gpu_model'] = self._get_apple_silicon_info()
        else:
            system_info['gpu_model'] = "None"
            
        return system_info
    
    def _prepare_data(self):
        """Prepare CIFAR-10 dataset"""
        g = torch.Generator()
        g.manual_seed(42)
        
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        # Download and load training data
        trainset = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True, transform=transform
        )
        trainloader = DataLoader(
            trainset, batch_size=self.batch_size, shuffle=True, 
            num_workers=2, worker_init_fn=seed_worker, generator=g
        )
        
        # Download and load test data
        testset = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True, transform=transform
        )
        testloader = DataLoader(
            testset, batch_size=self.batch_size, shuffle=False, 
            num_workers=2, worker_init_fn=seed_worker, generator=g
        )
        
        return trainloader, testloader

    def run_benchmark(self, device):
        """Run the complete benchmark including training and testing"""
        total_start_time = time.time()  # Start timing total process
        
        print("\nPreparing dataset...")
        trainloader, testloader = self._prepare_data()
        
        print("Initialising benchmark model...")
        model = VGG16().to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.num_epochs)

        print(f"\nStarting benchmark training...")
        epoch_start_time = time.time()
        
        for epoch in range(self.num_epochs):
            model.train()
            running_loss = 0.0
            
            # Create progress bar for training
            pbar = tqdm(trainloader, desc=f'Epoch {epoch+1}/{self.num_epochs}',
                       unit='batch', leave=True)
            
            for inputs, labels in pbar:
                inputs, labels = inputs.to(device), labels.to(device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()

                # Update progress bar description with current loss
                pbar.set_postfix({
                    'loss': f'{running_loss/len(pbar):.4f}',
                    'epoch_time': f'{time.time() - epoch_start_time:.1f}s'
                })
            
            scheduler.step()
            pbar.close()
            epoch_start_time = time.time()  # Reset for next epoch

        # Perform validation within each epoch after training
        model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in testloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        f1 = f1_score(all_labels, all_preds, average='weighted')
        
        training_time = time.time() - total_start_time
        
        # Calculate F1 score on test set
        print("\nEvaluating benchmark training...")
        model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in testloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        f1 = f1_score(all_labels, all_preds, average='weighted')
        
        total_time = time.time() - total_start_time
        
        return {
            'training_time': round(training_time, 2),
            'total_time': round(total_time, 2),
            'f1_score': round(f1, 4),
        }
    
    def run_full_benchmark(self):
        """Run complete benchmark and return results"""
        # Select best available device
        device = (torch.device('cuda') if torch.cuda.is_available() else 
                 torch.device('mps') if torch.backends.mps.is_available() else 
                 torch.device('cpu'))
        
        print(f"Running benchmark on {device}...")
        benchmark_results = self.run_benchmark(device)
        
        # Get the minimum total time from all records (including current benchmark)
        if os.path.exists(self.results_file):
            existing_df = pd.read_csv(self.results_file)
            all_times = existing_df['total_time'].tolist() + [benchmark_results['total_time']]
            alpha = min(all_times)
        else:
            alpha = benchmark_results['total_time']
            
        # Calculate benchmark score for current result
        benchmark_score = round((alpha / benchmark_results['total_time']) * benchmark_results['f1_score'], 4)
        
        # Remove the premature update and save of existing records
        # The benchmark scores will be updated in save_results instead
        
        results = {
            'timestamp': datetime.now().strftime('%Y-%m-%d'),
            'os': f"{platform.system()} {platform.release()}",
            'cpu_model': self.system_info['cpu_model'],
            'cpu_cores': f"{self.system_info['cpu_cores']} ({self.system_info['cpu_threads']} threads)",
            'ram': self._get_memory_info(),
            'gpu_model': self.system_info['gpu_model'],
            'pytorch_version': torch.__version__,
            'training_time': benchmark_results['training_time'],
            'total_time': benchmark_results['total_time'],
            'f1_score': benchmark_results['f1_score'],
            'benchmark_score': benchmark_score,
        }
        
        # Add CUDA information if available
        cuda_info = self._get_cuda_info()
        results.update({
            'cuda_version': cuda_info['cuda_version'],
            'cuda_cores': cuda_info['cuda_cores'],
            'vram': (cuda_info['vram_gb'] if cuda_info['vram_gb'] != 'N.A.' 
                    else self._get_memory_info() if 'Apple' in str(results['gpu_model'])
                    else 'N.A.')
        })
        
        return results
    
    def _generate_unique_id(self, results):
        """Generate a unique identifier based on system specifications"""
        unique_string = f"{results['os']}_{results['cpu_model']}_{results['cpu_cores']}_{results['ram']}_{results['gpu_model']}_{results['cuda_version']}_{results['cuda_cores']}_{results['vram']}_{results['pytorch_version']}"
        return hashlib.sha256(unique_string.encode()).hexdigest()  # Generate a SHA-256 hash

    def save_results(self, results):
        """Save benchmark results to CSV file"""
        results['unique_id'] = self._generate_unique_id(results)
        
        df = pd.DataFrame([results])
        
        if os.path.exists(self.results_file):
            existing_df = pd.read_csv(self.results_file)
            if results['unique_id'] in existing_df['unique_id'].values:
                existing_df = existing_df[existing_df['unique_id'] != results['unique_id']]
            df = pd.concat([existing_df, df], ignore_index=True)
        
        # Calculate alpha (minimum total time) from the combined dataset
        alpha = df['total_time'].min()
        
        # Update all benchmark scores using the final alpha value
        df['benchmark_score'] = round((alpha / df['total_time']) * df['f1_score'], 4)
        
        # Convert benchmark_score to numeric and sort
        df['benchmark_score'] = df['benchmark_score'].astype(float)
        df = df.sort_values(by='benchmark_score', ascending=False, ignore_index=True)
        
        # Reorder columns
        columns = [
            'unique_id', 'os', 'cpu_model', 'cpu_cores', 'ram',
            'gpu_model', 'cuda_version', 'cuda_cores', 'vram',
            'pytorch_version', 'total_time', 'f1_score', 'benchmark_score'
        ]
        df = df[columns]
        
        df.to_csv(self.results_file, index=False)
        return df
    
    def _get_memory_info(self):
        """Get total RAM information"""
        return f"{round(psutil.virtual_memory().total / (1024**3), 2)} GB"
    
    def _get_cuda_info(self):
        """Get CUDA information if available"""
        if not torch.cuda.is_available():
            return {
                'cuda_version': 'N.A.',
                'cuda_cores': 'N.A.',
                'vram_gb': 'N.A.'
            }
        
        # Assuming you have a way to get CUDA version and cores
        cuda_version = torch.version.cuda
        cuda_cores = torch.cuda.get_device_capability(0)[0]  # Example for getting CUDA cores
        vram_gb = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)
        
        return {
            'cuda_version': cuda_version,
            'cuda_cores': cuda_cores,
            'vram_gb': vram_gb
        }

def main():
    benchmark = PCBenchmark()
    print("Starting PC Benchmark...")
    
    print("\nAvailable Devices:")
    for device, available in benchmark.available_devices.items():
        if isinstance(available, bool):
            print(f"{device}: {'Available' if available else 'Not Available'}")
        else:
            print(f"{device}: {available}")
    
    print("\nSystem Information:")
    for key, value in benchmark.system_info.items():
        print(f"{key}: {value}")
    
    results = benchmark.run_full_benchmark()
    
    print("\nBenchmark Results:")
    print(f"Total Time: {results['total_time']} seconds")
    print(f"F1 Score: {results['f1_score']}")
    
    print("\nSaving results and updating leaderboard...")
    leaderboard = benchmark.save_results(results)
    
    print("\nLeaderboard (Top 5):")
    print(leaderboard.drop('unique_id', axis=1).head())

if __name__ == "__main__":
    main()
