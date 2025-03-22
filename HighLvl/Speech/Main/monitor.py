import psutil
import time
import argparse
import os
import sys
import threading
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

class ResourceMonitor:
    def __init__(self, interval=1.0, window_size=60):
        self.interval = interval
        self.window_size = window_size
        self.cpu_history = []
        self.ram_history = []
        self.timestamps = []
        self.running = True
        
        # Initialize GPU monitoring if available
        self.has_gpu = False
        self.gpu_history = []
        try:
            import torch
            if torch.cuda.is_available():
                self.has_gpu = True
        except ImportError:
            pass
            
    def monitor_resources(self):
        while self.running:
            # Collect CPU and RAM data
            cpu_percent = psutil.cpu_percent(interval=None)
            ram_percent = psutil.virtual_memory().percent
            
            self.cpu_history.append(cpu_percent)
            self.ram_history.append(ram_percent)
            self.timestamps.append(time.time())
            
            # Keep window_size items
            if len(self.cpu_history) > self.window_size:
                self.cpu_history.pop(0)
                self.ram_history.pop(0)
                self.timestamps.pop(0)
            
            # Collect GPU data if available
            if self.has_gpu:
                try:
                    import torch
                    gpu_percent = torch.cuda.utilization()
                    self.gpu_history.append(gpu_percent)
                    if len(self.gpu_history) > self.window_size:
                        self.gpu_history.pop(0)
                except:
                    self.gpu_history.append(0)
                    if len(self.gpu_history) > self.window_size:
                        self.gpu_history.pop(0)
            
            # Print current stats
            current_time = time.strftime("%H:%M:%S", time.localtime())
            gpu_str = f", GPU: {self.gpu_history[-1]}%" if self.has_gpu else ""
            print(f"{current_time} - CPU: {cpu_percent:.1f}%, RAM: {ram_percent:.1f}%{gpu_str}")
            
            # Sleep until next interval
            time.sleep(self.interval)
    
    def plot_resources(self, fig, ax_cpu, ax_ram, ax_gpu=None):
        # Plot data
        if not self.timestamps:
            return
            
        relative_times = [(t - self.timestamps[0]) for t in self.timestamps]
        
        ax_cpu.clear()
        ax_cpu.set_title("CPU Usage")
        ax_cpu.set_ylabel("Percent")
        ax_cpu.set_ylim(0, 100)
        ax_cpu.plot(relative_times, self.cpu_history, 'b-')
        ax_cpu.set_xticklabels([])
        
        ax_ram.clear()
        ax_ram.set_title("RAM Usage")
        ax_ram.set_ylabel("Percent")
        ax_ram.set_ylim(0, 100)
        ax_ram.plot(relative_times, self.ram_history, 'g-')
        
        if self.has_gpu and ax_gpu:
            ax_gpu.clear()
            ax_gpu.set_title("GPU Usage")
            ax_gpu.set_ylabel("Percent")
            ax_gpu.set_ylim(0, 100)
            ax_gpu.plot(relative_times, self.gpu_history, 'r-')
            ax_gpu.set_xlabel("Time (seconds)")
        else:
            ax_ram.set_xlabel("Time (seconds)")
            
    def start_monitoring(self, plot=False):
        # Start monitoring thread
        monitor_thread = threading.Thread(target=self.monitor_resources)
        monitor_thread.daemon = True
        monitor_thread.start()
        
        if plot:
            plt.style.use('ggplot')
            if self.has_gpu:
                fig, (ax_cpu, ax_ram, ax_gpu) = plt.subplots(3, 1, figsize=(10, 8))
            else:
                fig, (ax_cpu, ax_ram) = plt.subplots(2, 1, figsize=(10, 6))
                ax_gpu = None
                
            ani = FuncAnimation(
                fig, 
                lambda i: self.plot_resources(fig, ax_cpu, ax_ram, ax_gpu),
                interval=self.interval * 1000
            )
            
            plt.tight_layout()
            plt.show()
            
        try:
            # Keep running until keyboard interrupt
            while True:
                time.sleep(0.1)
        except KeyboardInterrupt:
            self.running = False
            print("Monitoring stopped.")

def main():
    parser = argparse.ArgumentParser(description="System Resource Monitor")
    parser.add_argument("--interval", type=float, default=1.0, help="Update interval in seconds")
    parser.add_argument("--window", type=int, default=60, help="History window size")
    parser.add_argument("--plot", action="store_true", help="Show graphical plot")
    args = parser.parse_args()
    
    print("System Resource Monitor")
    print("Press Ctrl+C to stop monitoring")
    
    monitor = ResourceMonitor(interval=args.interval, window_size=args.window)
    monitor.start_monitoring(plot=args.plot)

if __name__ == "__main__":
    main()
