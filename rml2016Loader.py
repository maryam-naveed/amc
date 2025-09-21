# rml2016Loader.py
'''
import pickle

def load_rml2016(path):
    """
    Loads RML2016.10a dataset (dict of {(mod, snr): samples}).
    Returns the raw dict.
    """
    with open(path, "rb") as f:
        data = pickle.load(f, encoding="latin1")
    print(f"✅ Loaded RML2016.10a from {path}, keys={len(data)}")
    return data
'''
# radioml2016Loader.py
import pickle
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

# --------------------------
# GPU / CPU check
# --------------------------
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    print(f"✅ GPU detected: {gpus[0].name}")
    try:
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
        print("✅ Mixed precision enabled.")
    except Exception as e:
        print("⚠️ Could not enable mixed precision:", e)
else:
    print("⚠️ No GPU detected, using CPU.")

# --------------------------
# Dataset loader
# --------------------------
def load_rml2016(path):
    with open(path, "rb") as f:
        return pickle.load(f, encoding="latin1")

# --------------------------
# Generator
# --------------------------
def rml2016_generator(dataset, le, batch_size=1024, mode="train", split=0.8):
    keys = list(dataset.keys())
    np.random.shuffle(keys)
    cutoff = int(len(keys) * split)
    if mode == "train":
        keys = keys[:cutoff]
    else:
        keys = keys[cutoff:]

    while True:
        np.random.shuffle(keys)
        for key in keys:
            samples = dataset[key]
            mod = str(key[0] if isinstance(key, tuple) else key)
            labels = np.full(samples.shape[0], mod, dtype=object)
            labels_enc = le.transform(labels)
            
            idx = np.arange(samples.shape[0])
            np.random.shuffle(idx)

            for start in range(0, samples.shape[0], batch_size):
                end = min(start + batch_size, samples.shape[0])
                batch_idx = idx[start:end]
                X_batch = samples[batch_idx].astype(np.float32)
                y_batch = to_categorical(labels_enc[batch_idx], num_classes=len(le.classes_))
                X_batch = np.expand_dims(X_batch, -1)  # (2,128,1)
                yield X_batch, y_batch
