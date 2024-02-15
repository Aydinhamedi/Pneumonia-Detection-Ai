from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from Utils.print_color_V2_NEW import print_Color_V2
from Utils.print_color_V1_OLD import print_Color
from tabulate import tabulate
import numpy as np
import pickle
import gzip

def save_list(history, filename, compress=True):
    """Saves a list to a file.

    Args:
        history: The list to save.
        filename: The file to save the list to.
        compress: Whether to gzip compress the file. Default is True.

    """
    if compress:
        with gzip.open(filename, 'wb') as f:
            pickle.dump(history, f)
    else:
        with open(filename, 'wb') as f:
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
        with gzip.open(filename, 'rb') as f:
            return pickle.load(f)
    else:
        with open(filename, 'rb') as f:
            return pickle.load(f)


def P_warning(msg):
    """Prints a warning message to the console.

    Args:
        msg (str): The warning message to print.
    """
    print_Color_V2(f'<light_red>Warning: <yellow>{msg}')

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
    weighted_precision = precision_score(
        y_test_bin, y_pred_bin, average='macro')
    weighted_f1 = f1_score(y_test_bin, y_pred_bin, average='macro')
    weighted_recall = recall_score(y_test_bin, y_pred_bin, average='macro')

    # Prepare data for the table
    metrics = [["Accuracy", round(accuracy * 100, 6)],
               ["Precision", round(weighted_precision * 100, 6)],
               ["F1 Score", round(weighted_f1 * 100, 6)],
               ["Recall", round(weighted_recall * 100, 6)]]

    # Print the table
    print(tabulate(metrics, headers=["Metric", "Value"], tablefmt="pretty"))

