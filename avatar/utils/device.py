import torch
import subprocess
import numpy as np
import logging
import os
from typing import List, Union


def get_gpu_memory_map() -> np.ndarray:
    """
    Get the current GPU memory usage.

    Returns:
        np.ndarray: Array of memory usage for each GPU.
    """
    try:
        result = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,nounits,noheader'],
            encoding='utf-8'
        )
        gpu_memory = np.array([int(x) for x in result.strip().split('\n')])
        return gpu_memory
    except subprocess.CalledProcessError as e:
        logging.error(f"Error while fetching GPU memory usage: {e}")
        return np.zeros(torch.cuda.device_count())  # Return zeroes if there's an error


def auto_select_device(
    cuda_visible: Union[List[int], str] = [],
    memory_max: int = 20000,
    memory_bias: int = 200,
    strategy: str = 'random'
) -> str:
    """
    Auto select a GPU device based on memory usage.

    Args:
        cuda_visible (Union[List[int], str], optional): List of visible CUDA devices or a string representing the list. Defaults to [].
        memory_max (int, optional): Maximum allowed memory usage to consider a GPU. Defaults to 20000.
        memory_bias (int, optional): Bias to add to the memory usage for random selection. Defaults to 200.
        strategy (str, optional): Strategy to select the GPU, either 'greedy' or 'random'. Defaults to 'random'.

    Returns:
        str: Selected device ('cuda:<id>' or 'cpu').
    """
    if not torch.cuda.is_available():
        return 'cpu'

    try:
        memory_raw = get_gpu_memory_map()
    except subprocess.CalledProcessError:
        memory_raw = np.ones(torch.cuda.device_count()) * 1e6  # Set high value if fetching fails

    if isinstance(cuda_visible, str):
        cuda_visible = eval(cuda_visible)

    if not cuda_visible:
        cuda_visible = list(range(len(memory_raw)))

    invisible_device = np.ones(len(memory_raw), dtype=bool)
    invisible_device[cuda_visible] = False
    memory_raw[invisible_device] = 1e6  # Set high memory to exclude invisible devices

    if strategy == 'greedy' or np.all(memory_raw > memory_max):
        cuda = np.argmin(memory_raw)
        logging.info(f'Greedy select GPU, select GPU {cuda} with mem: {memory_raw[cuda]}')
    elif strategy == 'random':
        memory = 1 / (memory_raw + memory_bias)
        memory[memory_raw > memory_max] = 0
        memory[invisible_device] = 0
        gpu_prob = memory / memory.sum()
        np.random.seed()
        cuda = np.random.choice(len(gpu_prob), p=gpu_prob)
        logging.info(f'Random select GPU, select GPU {cuda} with mem: {memory_raw[cuda]}')

    logging.info(f'GPU Mem: {memory_raw}')
    return f'cuda:{cuda}' if 'cuda' in locals() else 'cpu'
