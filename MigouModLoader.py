'''
# MigouModLoader.py
import pickle

def load_migou(path):
    """
    Loads MigouMod dataset from pickle.
    Format: {(mod, snr): samples} or similar.
    """
    with open(path, "rb") as f:
        data = pickle.load(f, encoding="latin1")
    print(f"✅ Loaded Migou dataset from {path}, keys={len(data)}")
    return data
'''
# MigouModLoader.py
import pickle
import numpy as np
from tensorflow.keras.utils import to_categorical

def load_migou(path):
    with open(path, "rb") as f:
        return pickle.load(f, encoding="latin1")

def migou_generator(dataset, le, batch_size=1024):
    """
    Generator for MigouMod testing dataset.
    Returns X, y batches without shuffling (for evaluation)
    """
    keys = list(dataset.keys())
    for key in keys:
        samples = dataset[key]
        mod = str(key[0] if isinstance(key, tuple) else key)
        labels = np.full(samples.shape[0], mod, dtype=object)
        labels_enc = le.transform(labels)
        y_onehot = to_categorical(labels_enc, num_classes=len(le.classes_))
        for start in range(0, samples.shape[0], batch_size):
            end = min(start + batch_size, samples.shape[0])
            X_batch = np.expand_dims(samples[start:end].astype(np.float32), -1)
            y_batch = y_onehot[start:end]
            yield X_batch, y_batch
