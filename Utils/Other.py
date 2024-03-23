from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from Utils.print_color_V2_NEW import print_Color_V2
from Utils.print_color_V1_OLD import print_Color
import keras.backend as K
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


def evaluate_model_full(y_test, model_pred, model=None, x_test=None):
    """Evaluates a machine learning model on a test set.

    Args:
        x_test: Test set features.
        y_test: Test set labels.
        model_pred: Model predictions.
        model: The model object.

    Returns:
        None. Prints a table with accuracy, precision, recall and
        F1 score.
    """
    # Get the model predictions
    if model_pred is None:
        y_pred = model.predict(x_test)
    else:
        y_pred = model_pred

    # Convert one-hot encoded predictions and labels to label encoded form
    y_pred_bin = np.argmax(y_pred, axis=1)
    y_test_bin = np.argmax(y_test, axis=1)

    # Calculate normal metrics
    accuracy = accuracy_score(y_test_bin, y_pred_bin)

    # Calculate weighted metrics
    weighted_precision = precision_score(y_test_bin, y_pred_bin, average="macro")
    weighted_f1 = f1_score(y_test_bin, y_pred_bin, average="macro")
    weighted_recall = recall_score(y_test_bin, y_pred_bin, average="macro")

    # Prepare data for the table
    metrics = [
        ["Accuracy", round(accuracy * 100, 6)],
        ["Precision", round(weighted_precision * 100, 6)],
        ["F1 Score", round(weighted_f1 * 100, 6)],
        ["Recall", round(weighted_recall * 100, 6)],
    ]

    # Print the table
    print(tabulate(metrics, headers=["Metric", "Value"], tablefmt="pretty"))


def set_optimizer_attribute(optimizer, attribute, value, verbose: bool = False):
    """Sets an attribute on the given optimizer to the specified value.

    Args:
        optimizer: The optimizer instance to modify.
        attribute: The attribute name to set.
        value: The value to set the attribute to.
        verbose: Whether to print a message if the attribute does not exist.
    """
    if hasattr(optimizer, attribute):
        K.set_value(getattr(optimizer, attribute), value)
    else:
        print(f"The optimizer does not have an attribute named '{attribute}'") if verbose else None


def print_optimizer_info(model):
    """Prints information about the optimizer used by a Keras model.

    Prints the optimizer class name and its parameter values. Useful
    for inspecting optimizer configuration.

    Args:
        model: The Keras model whose optimizer to print info for.

    """
    if model.optimizer:
        print_Color(f"~*Optimizer: ~*{model.optimizer.__class__.__name__}", ["cyan", "green"], advanced_mode=True)
        print_Color(" <Opt> Parameters:", ["cyan"])
        for param, value in model.optimizer.get_config().items():
            print_Color(f"~* <Opt> -- ~*{param}: ~*{value}", ["cyan", "light_cyan", "green"], advanced_mode=True)
    else:
        print_Color("No optimizer found in the model.", ["red"])