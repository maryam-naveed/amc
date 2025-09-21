import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from MigouModLoader import load_migou, migou_generator
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# --------------------------
# Paths
# --------------------------
MODEL_PATH = "C:/Users/HP/Downloads/AMC/amc_model_finetuned.h5"
ENCODER_PATH = "label_encoder.pkl"
MIGOU_PATH = "C:/Users/HP/Downloads/AMC/datasets/migou_dataset_19.08_400000x128.pkl"

# --------------------------
# Load model + encoder
# --------------------------
print(f"Loading model from {MODEL_PATH} ...")
model = load_model(MODEL_PATH)
with open(ENCODER_PATH, "rb") as f:
    le = pickle.load(f)
print("✅ Model and encoder loaded.")

# --------------------------
# Load MigouMod dataset
# --------------------------
print(f"Loading MigouMod dataset from {MIGOU_PATH} ...")
migou = load_migou(MIGOU_PATH)
print(f"Total samples (keys): {len(migou)}")

# --------------------------
# Use a fraction of the dataset for testing
# --------------------------
migou_keys = list(migou.keys())
test_fraction = 0.1  # 10% of dataset for quick testing
num_test_samples = max(1, int(len(migou_keys) * test_fraction))
test_keys_subset = migou_keys[:num_test_samples]
test_data_subset = {k: migou[k] for k in test_keys_subset}
print(f"Testing on subset: {len(test_data_subset)} samples")

# --------------------------
# Generator wrapper for testing
# --------------------------
def test_generator(dataset, label_encoder, batch_size=1024):
    gen = migou_generator(dataset, label_encoder, batch_size=batch_size)
    for X, y in gen:
        # Fix shape if needed
        if X.ndim == 5:  # accidentally (batch, 2, 128, 1, 1)
            X = np.squeeze(X, axis=-1)
        if X.ndim == 3:  # missing channel dim
            X = np.expand_dims(X, -1)
        yield X.astype(np.float32), y.astype(np.float32)

# --------------------------
# Evaluate
# --------------------------
y_true_all = []
y_pred_all = []

test_gen = test_generator(test_data_subset, le, batch_size=1024)
for X_batch, y_batch in test_gen:
    y_true_all.extend(np.argmax(y_batch, axis=1))
    y_pred_all.extend(np.argmax(model.predict(X_batch, verbose=0), axis=1))

y_true_all = np.array(y_true_all)
y_pred_all = np.array(y_pred_all)

# --------------------------
# Metrics
# --------------------------
acc = accuracy_score(y_true_all, y_pred_all)
print(f"\n📊 Overall Accuracy on subset: {acc:.4f}")

cm = confusion_matrix(y_true_all, y_pred_all)
plt.figure(figsize=(12,10))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

print("\nClassification Report:")
print(classification_report(y_true_all, y_pred_all, target_names=le.classes_))

# Sample predictions
print("\n🔍 Sample Predictions:")
for i in range(min(10, len(y_true_all))):
    true_label = le.classes_[y_true_all[i]]
    pred_label = le.classes_[y_pred_all[i]]
    correct = "✔️" if y_true_all[i] == y_pred_all[i] else "❌"
    print(f"Sample {i}: True={true_label}, Pred={pred_label} {correct}")


'''
# --------------------------
# TRY THIS
# --------------------------
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from MigouModLoader import load_migou, migou_generator
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# --------------------------
# Paths
# --------------------------
MODEL_PATH = "C:/Users/HP/Downloads/AMC/amc_model_finetuned.h5"
ENCODER_PATH = "label_encoder.pkl"
MIGOU_PATH = "C:/Users/HP/Downloads/AMC/datasets/migou_dataset_19.08_400000x128.pkl"

# --------------------------
# Load model + encoder
# --------------------------
print(f"Loading model from {MODEL_PATH} ...")
model = load_model(MODEL_PATH)
with open(ENCODER_PATH, "rb") as f:
    le = pickle.load(f)
print("✅ Model and encoder loaded.")

# --------------------------
# Load MigouMod dataset
# --------------------------
print(f"Loading MigouMod dataset from {MIGOU_PATH} ...")
migou = load_migou(MIGOU_PATH)
print(f"Total samples (keys): {len(migou)}")

# --------------------------
# Random subset for faster testing
# --------------------------
test_fraction = 0.05  # 5% of dataset
migou_keys = list(migou.keys())
num_test_samples = max(1, int(test_fraction * len(migou_keys)))
test_keys = np.random.choice(migou_keys, size=num_test_samples, replace=False)
test_data = {k: migou[k] for k in test_keys}
print(f"Testing on subset: {len(test_data)} samples")

# --------------------------
# Generator wrapper for testing
# --------------------------
def test_generator(dataset, label_encoder, batch_size=1024):
    gen = migou_generator(dataset, label_encoder, batch_size=batch_size)
    for X, y in gen:
        # Fix shape issues
        if X.ndim == 5:  # accidentally (batch,2,128,1,1)
            X = np.squeeze(X, axis=-1)
        if X.ndim == 3:  # missing channel dim
            X = np.expand_dims(X, -1)
        yield X.astype(np.float32), y.astype(np.float32)

# --------------------------
# Evaluate
# --------------------------
y_true_all = []
y_pred_all = []

test_gen = test_generator(test_data, le, batch_size=1024)
for X_batch, y_batch in test_gen:
    y_true_all.extend(np.argmax(y_batch, axis=1))
    y_pred_all.extend(np.argmax(model.predict(X_batch, verbose=0), axis=1))

y_true_all = np.array(y_true_all)
y_pred_all = np.array(y_pred_all)

# --------------------------
# Metrics
# --------------------------
acc = accuracy_score(y_true_all, y_pred_all)
print(f"\n📊 Overall Accuracy on subset: {acc:.4f}")

# Confusion matrix
cm = confusion_matrix(y_true_all, y_pred_all)
plt.figure(figsize=(12,10))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

# Classification report
present_classes = np.unique(np.concatenate([y_true_all, y_pred_all]))
print("\nClassification Report:")
print(classification_report(
    y_true_all,
    y_pred_all,
    labels=present_classes,
    target_names=[le.classes_[i] for i in present_classes]
))

# Sample predictions
print("\n🔍 Sample Predictions:")
for i in range(min(10, len(y_true_all))):
    true_label = le.classes_[y_true_all[i]]
    pred_label = le.classes_[y_pred_all[i]]
    correct = "✔️" if y_true_all[i] == y_pred_all[i] else "❌"
    print(f"Sample {i}: True={true_label}, Pred={pred_label} {correct}")
'''