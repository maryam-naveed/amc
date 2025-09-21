# rml2018Loader.py
'''
import os
import h5py
import numpy as np

def load_rml2018(folder_path):
    """
    Loads the RML2018 dataset stored in multiple .h5 parts.
    Each file is expected to contain 'X' and 'Y' datasets.
    
    Args:
        folder_path (str): Path to the folder containing part0.h5 ... partN.h5
    
    Returns:
        X_all (np.ndarray): All samples concatenated
        Y_all (np.ndarray): All labels concatenated
    """
    X_parts, Y_parts = [], []

    files = sorted([f for f in os.listdir(folder_path) if f.endswith(".h5")])
    print(f"Found {len(files)} RML2018 parts in {folder_path}")

    for file in files:
        path = os.path.join(folder_path, file)
        print(f"Loading {path}...")

        with h5py.File(path, "r") as f:
            # Load signals
            X_parts.append(np.array(f["X"], dtype=np.float32))

            # Load labels (leave dtype as-is for flexibility)
            Y_parts.append(np.array(f["Y"]))

            # Debug info (only on first file)
            if file == files[0]:
                print("Keys in file:", list(f.keys()))
                for k in f.keys():
                    print(f"  {k}: shape={f[k].shape}, dtype={f[k].dtype}")

    X_all = np.concatenate(X_parts, axis=0)
    Y_all = np.concatenate(Y_parts, axis=0)

    print("RML2018 loaded:", X_all.shape, Y_all.shape)
    return X_all, Y_all


# Quick test
if __name__ == "__main__":
    # Change this path to where your part0.h5 ... part20.h5 are located
    folder = "C:/Users/HP/Downloads/AMC/datasets/ExtractDataset"
    X, Y = load_rml2018(folder)
    print("Sample batch:", X[:5].shape, Y[:5])

'''
'''
# RML2018Loader.py
import os, h5py, numpy as np

def load_rml2018(folder, parts=None):
    """
    Load RadioML2018 dataset split in part*.h5 files.
    Args:
        folder: folder containing part0.h5 ... partN.h5
        parts: list of part indices (e.g., [0,1,2]) or None=all
    Returns:
        X: signals, shape (N, 1024, 2)
        Y: labels (one-hot, shape (N, 24))
    """
    files = sorted([f for f in os.listdir(folder) if f.endswith(".h5")])
    if parts is not None:
        files = [f"part{p}.h5" for p in parts]

    X_list, Y_list = [], []
    for file in files:
        path = os.path.join(folder, file)
        with h5py.File(path, "r") as f:
            X_list.append(f["X"][:])
            Y_list.append(f["Y"][:])
    X = np.vstack(X_list)
    Y = np.vstack(Y_list)
    print(f"RML2018 loaded from {len(files)} files: {X.shape}, {Y.shape}")
    return X, Y
'''

# radioml2018Loader.py
import os
import h5py
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

def rml2018_generator(folder, le, batch_size=1024, mode="train", split=0.8, parts=None):
    """
    Generator for RML2018 dataset stored in multiple .h5 files.
    Args:
        folder: path to RML2018 folder
        le: LabelEncoder
        batch_size: batch size
        mode: 'train' or 'val'
        split: train/val split
        parts: list of part indices to use (None=all)
    """
    files = sorted([f for f in os.listdir(folder) if f.endswith(".h5")])
    if parts is not None:
        files = [f"part{p}.h5" for p in parts]

    # Load indices only (to avoid loading all in memory)
    file_indices = []
    for file in files:
        with h5py.File(os.path.join(folder, file), "r") as f:
            n_samples = f["X"].shape[0]
            file_indices.append((file, n_samples))

    # Split
    total_samples = sum([n for _, n in file_indices])
    cutoff = int(total_samples * split)

    while True:
        for file, n_samples in file_indices:
            filepath = os.path.join(folder, file)
            with h5py.File(filepath, "r") as f:
                X_data = f["X"][:].astype(np.float32)
                Y_data = f["Y"][:]
                X_data = np.transpose(X_data, (0, 2, 1))[:, :, :128]
                y_labels = np.argmax(Y_data, axis=1)
                y_labels_str = le.inverse_transform(y_labels)
                y_encoded = le.transform(y_labels_str)
                y_onehot = to_categorical(y_encoded, num_classes=len(le.classes_))
                
                idx = np.arange(X_data.shape[0])
                np.random.shuffle(idx)
                for start in range(0, X_data.shape[0], batch_size):
                    end = min(start + batch_size, X_data.shape[0])
                    batch_idx = idx[start:end]
                    X_batch = np.expand_dims(X_data[batch_idx], -1)
                    y_batch = y_onehot[batch_idx]
                    yield X_batch, y_batch
