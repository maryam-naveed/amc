import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from MigouModLoader import load_migou, migou_generator

# --------------------------
# Paths
# --------------------------
MODEL_PATH = "C:/Users/HP/Downloads/AMC/amc_model.h5"
ENCODER_PATH = "label_encoder.pkl"
MIGOU_PATH = "C:/Users/HP/Downloads/AMC/datasets/migou_dataset_19.08_400000x128.pkl"
FT_MODEL_PATH = "C:/Users/HP/Downloads/AMC/amc_model_finetuned.h5"

# --------------------------
# Load pre-trained model + label encoder
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
migou_keys = list(migou.keys())
num_samples = len(migou_keys)
print(f"Total samples (keys): {num_samples}")

# --------------------------
# Split dataset for fine-tuning and testing
# --------------------------
# Use at least 1 key for fine-tuning, even for very small datasets
finetune_size = max(1, int(0.2 * num_samples))  # 20% or at least 1
np.random.shuffle(migou_keys)
finetune_keys = migou_keys[:finetune_size]
test_keys = migou_keys[finetune_size:]

finetune_data = {k: migou[k] for k in finetune_keys}
test_data = {k: migou[k] for k in test_keys}
print(f"Fine-tuning samples: {len(finetune_data)}, Test samples: {len(test_data)}")

# --------------------------
# Generator wrapper for fine-tuning
# --------------------------
def finetune_generator(dataset, label_encoder, batch_size=1024):
    gen = migou_generator(dataset, label_encoder, batch_size=batch_size)
    for X, y in gen:
        # Ensure correct shape: (batch_size, 2, 128, 1)
        if X.ndim == 5:  # accidentally (batch,2,128,1,1)
            X = np.squeeze(X, axis=-1)
        if X.ndim == 3:  # missing channel dim
            X = np.expand_dims(X, -1)
        yield X.astype(np.float32), y.astype(np.float32)

batch_size = 1024
epochs = 5
finetune_gen = finetune_generator(finetune_data, le, batch_size=batch_size)
steps_per_epoch = max(1, sum([v.shape[0] for v in finetune_data.values()]) // batch_size)

# --------------------------
# Compile model for fine-tuning
# --------------------------
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),  # smaller LR for fine-tuning
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# --------------------------
# Fine-tuning
# --------------------------
print("Starting fine-tuning on subset of MigouMod ...")
model.fit(
    finetune_gen,
    steps_per_epoch=steps_per_epoch,
    epochs=epochs
)

# --------------------------
# Save fine-tuned model
# --------------------------
model.save(FT_MODEL_PATH)
print(f"✅ Fine-tuning complete. Model saved to {FT_MODEL_PATH}.")