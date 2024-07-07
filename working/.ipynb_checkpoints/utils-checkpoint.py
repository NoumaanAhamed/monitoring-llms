import torch
import gc
import time
import psutil
from functools import wraps

# def get_gpu_memory():
#     return torch.cuda.memory_allocated() / 1024**2  # Convert to MB

# def get_gpu_max_memory():
#     return torch.cuda.max_memory_allocated() / 1024**2  # Convert to MB

# def get_system_memory_usage():
#     memory = psutil.virtual_memory()
#     return memory.used / (1024 * 1024)  # Convert to MB

# def format_memory(memory_mb):
#     if memory_mb >= 1024:
#         return f"{memory_mb/1024:.2f} GB"
#     else:
#         return f"{memory_mb:.2f} MB"

# print("SYS:",get_system_memory_usage())

# def get_process_memory_usage():
#     process = psutil.Process()
#     return process.memory_info().rss / (1024 * 1024)  # Convert to MB

# print("PROC:",get_process_memory_usage())

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

print("Initial System Memory:", get_system_memory_usage(), "MB")
print("Initial Process Memory:", get_process_memory_usage(), "MB")

initial_system_memory = get_system_memory_usage()
initial_process_memory = get_process_memory_usage()

def track_resources(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        start_sys_mem = get_system_memory_usage()
        start_proc_mem = get_process_memory_usage()
        start_gpu_mem = get_gpu_memory()
        
        torch.cuda.reset_peak_memory_stats()
        result = func(*args, **kwargs)
        
        end_time = time.perf_counter()
        end_sys_mem = get_system_memory_usage()
        end_proc_mem = get_process_memory_usage()
        end_gpu_mem = get_gpu_memory()
        max_gpu_mem = get_gpu_max_memory()
        
        elapsed_time = end_time - start_time
        sys_mem_increase = end_sys_mem - start_sys_mem
        proc_mem_increase = end_proc_mem - start_proc_mem
        total_sys_mem_increase = end_sys_mem - initial_system_memory
        total_proc_mem_increase = end_proc_mem - initial_process_memory
        gpu_mem_increase = end_gpu_mem - start_gpu_mem
        
        print(f"{func.__name__}:")
        print(f"  Time: {elapsed_time:.4f}s")
        print(f"  System RAM: {start_sys_mem:.2f} MB -> {end_sys_mem:.2f} MB (Difference: {sys_mem_increase:.2f} MB, Total: {total_sys_mem_increase:.2f} MB)")
        print(f"  Process RAM: {start_proc_mem:.2f} MB -> {end_proc_mem:.2f} MB (Difference: {proc_mem_increase:.2f} MB, Total: {total_proc_mem_increase:.2f} MB)")
        print(f"  GPU Memory: {start_gpu_mem:.2f} MB -> {end_gpu_mem:.2f} MB (Difference: {gpu_mem_increase:.2f} MB, Peak: {max_gpu_mem:.2f} MB)")
        return result
    return wrapper
