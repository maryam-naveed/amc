'''
# Training.py
import pickle
import numpy as np
from tensorflow.keras import models, layers, optimizers
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from DataLoader import collect_labels
from rml2016Loader import load_rml2016
from rml2018Loader import load_rml2018

# --------------------------
# CNN model
# --------------------------
def build_model(input_shape, num_classes):
    model = models.Sequential([
        layers.Conv2D(64, (1, 3), activation='relu', input_shape=input_shape),
        layers.Conv2D(64, (2, 3), activation='relu'),
        layers.MaxPooling2D((1, 2)),
        layers.Dropout(0.3),
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax', dtype='float32')
    ])
    model.compile(
        optimizer=optimizers.Adam(1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

# --------------------------
# Main
# --------------------------
if __name__ == "__main__":
    print("Loading datasets...")
    rml2016 = load_rml2016("C:/Users/HP/Downloads/AMC/datasets/RML2016.10a_dict.pkl")
    rml2018_X, rml2018_Y = load_rml2018("C:/Users/HP/Downloads/AMC/datasets/ExtractDataset")

    # Convert 2018 to dict-like format for compatibility
    rml2018 = {}
    for i in range(rml2018_X.shape[0]):
        mod = str(np.argmax(rml2018_Y[i]))  # one-hot → string index
        rml2018[(mod, 0)] = rml2018_X[i:i+1]

    # --------------------------
    # Combine datasets into X, y
    # --------------------------
    X_list, y_list = [], []
    for dataset in [rml2016, rml2018]:
        for key, samples in dataset.items():
            mod = str(key[0] if isinstance(key, tuple) else key)
            # Ensure samples have shape (num_samples, 2, 128)
            if samples.shape[1] != 2 or samples.shape[2] != 128:
                samples = samples[:, :, :128]  # trim or pad to 128
            X_list.append(samples)
            y_list.extend([mod] * samples.shape[0])

    X_all = np.vstack(X_list).astype(np.float32)
    y_all = np.array(y_list, dtype=object)

    print("Combined dataset shape:", X_all.shape)
    print("Unique classes:", np.unique(y_all))

    # --------------------------
    # Label encoding
    # --------------------------
    le = LabelEncoder()
    y_all_enc = le.fit_transform(y_all)
    y_all_cat = np.eye(len(le.classes_))[y_all_enc]

    # --------------------------
    # Train/Validation Split
    # --------------------------
    X_train, X_val, y_train, y_val = train_test_split(
        X_all, y_all_cat, test_size=0.2, random_state=42, stratify=y_all_enc
    )

    # --------------------------
    # Expand dims for CNN input
    # --------------------------
    X_train = np.expand_dims(X_train, -1)
    X_val = np.expand_dims(X_val, -1)

    input_shape = X_train.shape[1:]  # (2, 128, 1)
    num_classes = len(le.classes_)

    # --------------------------
    # Model
    # --------------------------
    model = build_model(input_shape, num_classes)
    model.summary()

    # --------------------------
    # Training
    # --------------------------
    model.fit(
        X_train, y_train,
        batch_size=1024,
        epochs=10,
        validation_data=(X_val, y_val),
        shuffle=True
    )

    # --------------------------
    # Save model + encoder
    # --------------------------
    model.save("C:/Users/HP/Downloads/AMC/amc_model.h5")
    with open("label_encoder.pkl", "wb") as f:
        pickle.dump(le, f)

    print("✅ Training complete. Model and encoder saved.")
'''
# Training.py
from rml2016Loader import load_rml2016, rml2016_generator
from rml2018Loader import rml2018_generator
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import models, layers, optimizers
import numpy as np

# --------------------------
# CNN Model
# --------------------------
def build_model(input_shape, num_classes):
    model = models.Sequential([
        layers.Conv2D(64, (1,3), activation='relu', input_shape=input_shape),
        layers.Conv2D(64, (2,3), activation='relu'),
        layers.MaxPooling2D((1,2)),
        layers.Dropout(0.3),
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax', dtype='float32')
    ])
    model.compile(optimizer=optimizers.Adam(1e-3),
                  loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# --------------------------
# Main
# --------------------------
if __name__ == "__main__":
    print("Loading datasets...")
    rml2016 = load_rml2016("C:/Users/HP/Downloads/AMC/datasets/RML2016.10a_dict.pkl")
    rml2018_folder = "C:/Users/HP/Downloads/AMC/datasets/ExtractDataset"

    # Collect all unique labels
    all_labels = list(np.unique(list(rml2016.keys())))  # RML2016 labels
    # Add RML2018 labels (24 classes)
    rml2018_labels = ["OOK","4ASK","8ASK","BPSK","QPSK","8PSK","16PSK","32PSK",
                      "16APSK","32APSK","64APSK","128APSK","16QAM","32QAM","64QAM",
                      "128QAM","256QAM","AM-SSB","AM-DSB","WBFM","GFSK","CPFSK",
                      "PAM4","B-FM"]
    all_labels = list(np.unique(all_labels + rml2018_labels))
    print("Classes:", all_labels)

    le = LabelEncoder()
    le.fit(all_labels)

    # Generators
    train_gen_rml16 = rml2016_generator(rml2016, le, batch_size=1024, mode="train")
    val_gen_rml16 = rml2016_generator(rml2016, le, batch_size=1024, mode="val")
    train_gen_rml18 = rml2018_generator(rml2018_folder, le, batch_size=1024, mode="train")
    val_gen_rml18 = rml2018_generator(rml2018_folder, le, batch_size=1024, mode="val")

    # Merge generators
    def combined_gen(train=True):
        while True:
            for X, y in train_gen_rml16 if train else val_gen_rml16:
                yield X, y
            for X, y in train_gen_rml18 if train else val_gen_rml18:
                yield X, y

    X_batch, y_batch = next(train_gen_rml16)
    input_shape = X_batch.shape[1:]

    model = build_model(input_shape, len(le.classes_))
    model.summary()

    model.fit(combined_gen(train=True), steps_per_epoch=200,
              validation_data=combined_gen(train=False), validation_steps=50,
              epochs=10)

    # Save model + encoder
    model.save("C:/Users/HP/Downloads/AMC/amc_model.h5")
    with open("label_encoder.pkl","wb") as f:
        import pickle
        pickle.dump(le, f)

    print("✅ Training complete.")
