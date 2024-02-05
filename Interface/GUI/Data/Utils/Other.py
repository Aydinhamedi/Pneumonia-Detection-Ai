from Utils.print_color_V2_NEW import print_Color_V2
from Utils.print_color_V1_OLD import print_Color
import pickle
import gzip

def save_list(history, filename, compress=True):
    # Saves the given history list to the specified filename.
    # If compress is True, the file will be gzip compressed.
    # Otherwise it will be saved as a normal pickle file.
    if compress:
        with gzip.open(filename, 'wb') as f:
            pickle.dump(history, f)
    else:
        with open(filename, 'wb') as f:
            pickle.dump(history, f)


def load_list(filename, compressed=True):
    # Loads a pickled object from a file.
    # If compressed=True, it will load from a gzip compressed file.
    # Otherwise loads from a regular file.
    if compressed:
        with gzip.open(filename, 'rb') as f:
            return pickle.load(f)
    else:
        with open(filename, 'rb') as f:
            return pickle.load(f)
def P_warning(msg):
    # Prints a warning message with color formatting.
    # msg: The message to print as a warning.
    print_Color_V2(f'<light_red>Warning: <yellow>{msg}')
        
