import psutil
import torch
import time
import argparse
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import json
from datetime import datetime

class MemoryMonitor:
    def __init__(self, interval=1.0, window_size=60):
        self.interval = interval
        self.window_size = window_size
        self.system_memory = []
        self.gpu_memory = []
        self.timestamps = []
        self.running = True
        self.log_file = f"memory_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # Initialize torch memory tracking
        self.has_cuda = torch.cuda.is_available()
        if self.has_cuda:
            self.initial_gpu_memory = torch.cuda.memory_allocated()
    
    def get_memory_usage(self):
        # System memory
        system_mem = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        # GPU memory if available
        gpu_mem = 0
        if self.has_cuda:
            gpu_mem = torch.cuda.memory_allocated() / 1024 / 1024  # MB
        
        return system_mem, gpu_mem
    
    def monitor_memory(self):
        while self.running:
            system_mem, gpu_mem = self.get_memory_usage()
            current_time = time.time()
            
            self.system_memory.append(system_mem)
            self.gpu_memory.append(gpu_mem)
            self.timestamps.append(current_time)
            
            # Keep only window_size samples
            if len(self.timestamps) > self.window_size:
                self.system_memory.pop(0)
                self.gpu_memory.pop(0)
                self.timestamps.pop(0)
            
            # Log to file
            self.log_memory(system_mem, gpu_mem, current_time)
            
            # Print current usage
            print(f"System Memory: {system_mem:.1f}MB | GPU Memory: {gpu_mem:.1f}MB")
            
            time.sleep(self.interval)
    
    def log_memory(self, system_mem, gpu_mem, timestamp):
        log_entry = {
            "timestamp": timestamp,
            "system_memory_mb": system_mem,
            "gpu_memory_mb": gpu_mem
        }
        with open(self.log_file, 'a') as f:
            json.dump(log_entry, f)
            f.write('\n')
    
    def plot_memory(self, fig):
        if not self.timestamps:
            return
        
        relative_times = [(t - self.timestamps[0]) for t in self.timestamps]
        
        plt.clf()
        
        # Plot system memory
        plt.subplot(2, 1, 1)
        plt.title("System Memory Usage")
        plt.plot(relative_times, self.system_memory, 'b-', label='System')
        plt.ylabel("MB")
        plt.legend()
        plt.grid(True)
        
        # Plot GPU memory
        plt.subplot(2, 1, 2)
        plt.title("GPU Memory Usage")
        plt.plot(relative_times, self.gpu_memory, 'r-', label='GPU')
        plt.xlabel("Time (seconds)")
        plt.ylabel("MB")
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
    
    def start_monitoring(self, plot=False):
        from threading import Thread
        monitor_thread = Thread(target=self.monitor_memory, daemon=True)
        monitor_thread.start()
        
        if plot:
            fig = plt.figure(figsize=(10, 8))
            ani = FuncAnimation(fig, lambda i: self.plot_memory(fig), interval=self.interval * 1000)
            plt.show()
        
        try:
            while True:
                time.sleep(0.1)
        except KeyboardInterrupt:
            self.running = False
            print(f"\nMonitoring stopped. Log saved to {self.log_file}")

def main():
    parser = argparse.ArgumentParser(description="Memory Usage Monitor")
    parser.add_argument("--interval", type=float, default=1.0, help="Update interval in seconds")
    parser.add_argument("--window", type=int, default=60, help="Window size for plotting")
    parser.add_argument("--plot", action="store_true", help="Show real-time plot")
    args = parser.parse_args()
    
    monitor = MemoryMonitor(interval=args.interval, window_size=args.window)
    monitor.start_monitoring(plot=args.plot)

if __name__ == "__main__":
    main()
