# DataLoader.py
import numpy as np
import tensorflow as tf

# --------------------------
# Collect all labels from datasets
# --------------------------
def collect_labels(datasets):
    """
    Collect all unique labels from multiple datasets.
    datasets: list of dict-like {key: samples}
    Returns: sorted list of unique labels as strings
    """
    labels = set()
    for dataset in datasets:
        for key in dataset.keys():
            mod = str(key[0] if isinstance(key, tuple) else key)
            labels.add(mod)
    return sorted(labels)

# --------------------------
# Batch generator
# --------------------------
def batch_generator(datasets, le, batch_size=1024, shuffle=True):
    """
    Generator yielding (X, y) batches.
    datasets: list of dict-like datasets
    le: pre-fitted LabelEncoder
    batch_size: int
    shuffle: bool
    """
    # Flatten all data
    X_all, y_all = [], []
    for dataset in datasets:
        for key, samples in dataset.items():
            mod = str(key[0] if isinstance(key, tuple) else key)
            X_all.append(samples)
            y_all.append(np.full(samples.shape[0], mod, dtype=object))

    X_all = np.vstack(X_all).astype(np.float32)
    y_all = np.concatenate(y_all)

    # Encode labels using the **pre-fitted** LabelEncoder
    y_all_enc = le.transform(y_all)
    y_all_cat = tf.keras.utils.to_categorical(y_all_enc, num_classes=len(le.classes_))

    # Expand dims
    X_all = np.expand_dims(X_all, -1)

    # Shuffle indices
    indices = np.arange(X_all.shape[0])
    if shuffle:
        np.random.shuffle(indices)

    while True:
        for start in range(0, len(indices), batch_size):
            end = start + batch_size
            batch_idx = indices[start:end]
            yield X_all[batch_idx], y_all_cat[batch_idx]
