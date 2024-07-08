import torch
import gc
import time
import psutil
from functools import wraps

def get_gpu_memory():
    return torch.cuda.memory_allocated() / 1024**2  # Convert to MB

def get_gpu_max_memory():
    return torch.cuda.max_memory_allocated() / 1024**2  # Convert to MB

def get_system_memory_usage():
    memory = psutil.virtual_memory()
    return memory.used / (1024 * 1024)  # Convert to MB

def get_process_memory_usage():
    process = psutil.Process()
    return process.memory_info().rss / (1024 * 1024)  # Convert to MB

def get_cpu_usage():
    return psutil.cpu_percent(interval=0.1)

print("Initial System Memory:", get_system_memory_usage(), "MB")
print("Initial Process Memory:", get_process_memory_usage(), "MB")
initial_system_memory = get_system_memory_usage()
initial_process_memory = get_process_memory_usage()

def track_resources(*metrics_to_track):
    all_metrics = {'time', 'sys_mem', 'proc_mem', 'gpu_mem', 'cpu'}
    metrics_set = set(metrics_to_track) if metrics_to_track else all_metrics

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.perf_counter() if 'time' in metrics_set else None
            start_sys_mem = get_system_memory_usage() if 'sys_mem' in metrics_set else None
            start_proc_mem = get_process_memory_usage() if 'proc_mem' in metrics_set else None
            start_gpu_mem = get_gpu_memory() if 'gpu_mem' in metrics_set else None
            start_cpu = get_cpu_usage() if 'cpu' in metrics_set else None

            if 'gpu_mem' in metrics_set:
                torch.cuda.reset_peak_memory_stats()

            result = func(*args, **kwargs)

            print(f"{func.__name__}:")

            if 'time' in metrics_set:
                elapsed_time = time.perf_counter() - start_time
                print(f"  Time: {elapsed_time:.4f}s")

            if 'sys_mem' in metrics_set:
                end_sys_mem = get_system_memory_usage()
                sys_mem_increase = end_sys_mem - start_sys_mem
                total_sys_mem_increase = end_sys_mem - initial_system_memory
                print(f"  System RAM: {start_sys_mem:.2f} MB -> {end_sys_mem:.2f} MB (Difference: {sys_mem_increase:.2f} MB, Total: {total_sys_mem_increase:.2f} MB)")

            if 'proc_mem' in metrics_set:
                end_proc_mem = get_process_memory_usage()
                proc_mem_increase = end_proc_mem - start_proc_mem
                total_proc_mem_increase = end_proc_mem - initial_process_memory
                print(f"  Process RAM: {start_proc_mem:.2f} MB -> {end_proc_mem:.2f} MB (Difference: {proc_mem_increase:.2f} MB, Total: {total_proc_mem_increase:.2f} MB)")

            if 'gpu_mem' in metrics_set:
                end_gpu_mem = get_gpu_memory()
                max_gpu_mem = get_gpu_max_memory()
                gpu_mem_increase = end_gpu_mem - start_gpu_mem
                print(f"  GPU Memory: {start_gpu_mem:.2f} MB -> {end_gpu_mem:.2f} MB (Difference: {gpu_mem_increase:.2f} MB, Peak: {max_gpu_mem:.2f} MB)")

            if 'cpu' in metrics_set:
                end_cpu = get_cpu_usage()
                cpu_usage = end_cpu - start_cpu
                print(f"  CPU Usage: {cpu_usage:.2f}%")

            return result
        return wrapper
    return decorator