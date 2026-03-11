#!/usr/bin/env python3
"""
GPU Memory Cleanup Script for FSDP Training

This script helps clear GPU memory that might be stuck after FSDP operations.
Run this when your training script is not running to clean up any lingering GPU memory.
"""

import torch
import gc
import subprocess
import os
import signal
import psutil


def kill_gpu_processes():
    """Kill any running Python processes that might be holding GPU memory"""
    print("Checking for Python processes using GPU...")
    
    try:
        # Get GPU processes
        result = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,process_name', '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True)
        
        if result.returncode == 0 and result.stdout.strip():
            lines = result.stdout.strip().split('\n')
            for line in lines:
                if line.strip():
                    parts = line.split(', ')
                    if len(parts) >= 2:
                        pid = int(parts[0])
                        process_name = parts[1]
                        print(f"Found GPU process: PID {pid}, Name: {process_name}")
                        
                        # Only kill Python processes (be careful here)
                        if 'python' in process_name.lower():
                            try:
                                process = psutil.Process(pid)
                                print(f"Killing Python process: {pid}")
                                process.terminate()
                                process.wait(timeout=5)
                            except (psutil.NoSuchProcess, psutil.TimeoutExpired):
                                try:
                                    os.kill(pid, signal.SIGKILL)
                                except ProcessLookupError:
                                    pass
                            except Exception as e:
                                print(f"Could not kill process {pid}: {e}")
    except Exception as e:
        print(f"Error checking GPU processes: {e}")


def clear_gpu_memory():
    """Clear GPU memory using PyTorch operations"""
    print("Clearing GPU memory with PyTorch...")
    
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        print(f"Found {device_count} GPU(s)")
        
        for i in range(device_count):
            print(f"Clearing GPU {i}...")
            with torch.cuda.device(i):
                # Clear cache
                torch.cuda.empty_cache()
                
                # Force garbage collection
                gc.collect()
                
                # Clear IPC memory if available
                if hasattr(torch.cuda, 'ipc_collect'):
                    torch.cuda.ipc_collect()
                
                # Reset peak memory stats
                torch.cuda.reset_peak_memory_stats(i)
                torch.cuda.reset_accumulated_memory_stats(i)
                
                # Check memory after cleanup
                allocated = torch.cuda.memory_allocated(i) / 1024**3
                cached = torch.cuda.memory_reserved(i) / 1024**3
                print(f"GPU {i} - Allocated: {allocated:.2f} GB, Cached: {cached:.2f} GB")
    else:
        print("CUDA not available")


def reset_cuda_context():
    """Reset CUDA context (nuclear option)"""
    print("Resetting CUDA context...")
    try:
        # This will reset the CUDA context for the current process
        torch.cuda.empty_cache()
        if hasattr(torch.cuda, 'reset_accumulated_memory_stats'):
            for i in range(torch.cuda.device_count()):
                torch.cuda.reset_accumulated_memory_stats(i)
                torch.cuda.reset_peak_memory_stats(i)
        print("CUDA context reset completed")
    except Exception as e:
        print(f"Error resetting CUDA context: {e}")


def show_gpu_status():
    """Show current GPU memory status"""
    print("\n" + "="*50)
    print("GPU Memory Status")
    print("="*50)
    
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        print(result.stdout)
    except Exception as e:
        print(f"Could not run nvidia-smi: {e}")
    
    if torch.cuda.is_available():
        print("\nPyTorch CUDA Memory Info:")
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            cached = torch.cuda.memory_reserved(i) / 1024**3
            print(f"GPU {i}: Allocated {allocated:.2f} GB, Cached {cached:.2f} GB")


def main():
    print("GPU Memory Cleanup Script")
    print("="*30)
    
    # Show initial status
    show_gpu_status()
    
    # Option 1: Gentle cleanup
    print("\n1. Performing gentle cleanup...")
    clear_gpu_memory()
    show_gpu_status()
    
    # Option 2: Reset CUDA context
    print("\n2. Resetting CUDA context...")
    reset_cuda_context()
    show_gpu_status()
    
    # Option 3: Kill GPU processes (if needed)
    response = input("\nDo you want to kill Python processes using GPU? (y/N): ").lower()
    if response == 'y':
        kill_gpu_processes()
        show_gpu_status()
    
    print("\nCleanup completed!")


if __name__ == "__main__":
    main()
