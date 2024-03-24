from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from Utils.print_color_V2_NEW import print_Color_V2
from Utils.print_color_V1_OLD import print_Color
from tabulate import tabulate
from numba import cuda
import numpy as np
import pickle
import gzip


def GPU_memUsage(Print=True):
    """Prints GPU memory usage for each GPU.

    Args:
    Print (bool): Whether to print the memory usage.
        If True, prints the memory usage.
        If False, returns the free and total memory as a tuple.

    Returns:
    If Print is False, returns a tuple (free, total) with the free
    and total memory in bytes for the GPU.
    """
    gpus = cuda.gpus.lst
    for gpu in gpus:
        with gpu:
            meminfo = cuda.current_context().get_memory_info()
            if Print:
                print_Color(
                    f"~*(GPU-MEM)~*--{gpu}--[free: {meminfo.free / (1024** 3):.2f}GB, used: {meminfo.total / (1024** 3) - meminfo.free / (1024** 3):.2f}GB, total, {meminfo.total / (1024** 3):.2f}GB]",
                    ["green", "cyan"],
                    advanced_mode=True,
                )
            else:
                return meminfo.free, meminfo.total


def save_list(history, filename, compress=True):
    """Saves a list to a file.

    Args:
        history: The list to save.
        filename: The file to save the list to.
        compress: Whether to gzip compress the file. Default is True.

    """
    if compress:
        with gzip.open(filename, "wb") as f:
            pickle.dump(history, f)
    else:
        with open(filename, "wb") as f:
            pickle.dump(history, f)


def load_list(filename, compressed=True):
    """Loads a list from a file.

    Args:
        filename: The file to load from.
        compressed: Whether the file is gzip compressed. Default is True.

    Returns:
        The loaded list from the file.
    """
    if compressed:
        with gzip.open(filename, "rb") as f:
            return pickle.load(f)
    else:
        with open(filename, "rb") as f:
            return pickle.load(f)


def P_warning(msg):
    """Prints a warning message to the console.

    Args:
        msg (str): The warning message to print.
    """
    print_Color_V2(f"<light_red>Warning: <yellow>{msg} (⚠️)")


def P_verbose(msg):
    """Prints a verbose message to the console.

    Args:
        msg (str): The verbose message to print.
    """
    print_Color_V2(f"<light_cyan>Verbose: <normal>{msg}")
