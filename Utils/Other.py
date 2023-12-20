import pickle
import gzip

def save_list(history, filename, compress=True):
    if compress:
        with gzip.open(filename, 'wb') as f:
            pickle.dump(history, f)
    else:
        with open(filename, 'wb') as f:
            pickle.dump(history, f)

def load_list(filename, compressed=True):
    if compressed:
        with gzip.open(filename, 'rb') as f:
            return pickle.load(f)
    else:
        with open(filename, 'rb') as f:
            return pickle.load(f)
        
