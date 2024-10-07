import torch
import subprocess
import re

def get_gpu_memory_map():
    """Get the current GPU usage using nvidia-smi.

    Returns
    -------
    memory_map: dict
        Keys are device ids as integers.
        Values are memory free on that device in MB as integers.
    """
    # Run nvidia-smi command to get memory information
    result = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=memory.free,memory.total', '--format=csv,nounits,noheader'], encoding='utf-8'
    )

    # Extract the memory information
    gpu_memory = [tuple(map(int, x.split(','))) for x in result.strip().split('\n')]

    # Create a memory map as a dictionary
    memory_map = {i: free for i, (free, total) in enumerate(gpu_memory)}
    return memory_map

def select_gpu_with_most_free_memory():
    """Select the GPU with the most available memory."""
    if torch.cuda.is_available():
        memory_map = get_gpu_memory_map()
        best_gpu = max(memory_map, key=memory_map.get)
        torch.cuda.set_device(best_gpu)
        print(f"Selected GPU {best_gpu} with {memory_map[best_gpu]} MB free memory.")
        return best_gpu
    else:
        print("No GPU available, using CPU.")
        return None