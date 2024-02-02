import pickle 
import gzip
import pandas as pd

def load_list(filename, compressed=True):
    if compressed:
        with gzip.open(filename, 'rb') as f:
            return pickle.load(f)
    else:
        with open(filename, 'rb') as f:
            return pickle.load(f)

Data = load_list('history\\model_history.pkl.gz', compressed=True)

df = pd.DataFrame(Data)
df.to_csv(r'history\\model_history_CSV.csv')