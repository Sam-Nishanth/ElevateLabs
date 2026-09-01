# ============================================================
# RTSign Language Recognition
# BiLSTM Sign Language Model Training
# ============================================================

from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
import json

# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

FEATURE_DIR = BASE_DIR / "features"
MODEL_DIR = BASE_DIR / "models"

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

MODEL_PATH = MODEL_DIR / "sign_bilstm.keras"
CLASS_PATH = MODEL_DIR / "isl_classes_bilstm.npy"

# ============================================================
# SETTINGS
# ============================================================

RANDOM_STATE = 42

TEST_SIZE = 0.20

BATCH_SIZE = 32

EPOCHS = 30

LEARNING_RATE = 0.001

# ============================================================
# GPU CHECK
# ============================================================

gpus = tf.config.list_physical_devices("GPU")

print("=" * 70)
print("RTSign — BiLSTM SIGN LANGUAGE TRAINING")
print("=" * 70)

if gpus:
    print("GPU detected:")
    for gpu in gpus:
        print(" ", gpu)
else:
    print("GPU not detected — using CPU")

# ============================================================
# FIND FEATURE FILES
# ============================================================

print("\nFeature directory:")
print(FEATURE_DIR)

if not FEATURE_DIR.exists():
    raise FileNotFoundError(
        f"Feature directory not found:\n{FEATURE_DIR}"
    )

# Search for numpy feature files
feature_files = sorted(
    FEATURE_DIR.rglob("*.npy")
)

print(
    "\nNumpy files found:",
    len(feature_files)
)

if not feature_files:
    raise RuntimeError(
        "No .npy feature files found."
    )

# ============================================================
# LOAD FEATURES
# ============================================================

X_list = []
y_list = []
class_names = []

print("\nLoading feature files...")

for feature_file in feature_files:

    try:

        data = np.load(
            feature_file,
            allow_pickle=True
        )

        # ----------------------------------------------------
        # Determine class from parent directory
        # ----------------------------------------------------

        class_name = feature_file.parent.name

        # Ignore files that are not actual samples
        if class_name in [
            "features",
            "feature",
            "data"
        ]:

            continue

        # ----------------------------------------------------
        # Convert to float32
        # ----------------------------------------------------

        data = np.asarray(
            data,
            dtype=np.float32
        )

        # ----------------------------------------------------
        # Require sequential data
        # ----------------------------------------------------

        if data.ndim != 2:

            print(
                "Skipping:",
                feature_file.name,
                "shape:",
                data.shape
            )

            continue

        X_list.append(data)
        class_names.append(class_name)

    except Exception as e:

        print(
            "Failed:",
            feature_file,
            e
        )

# ============================================================
# CHECK DATA
# ============================================================

if not X_list:

    raise RuntimeError(
        "No valid feature sequences were loaded."
    )

# ============================================================
# CLASS MAPPING
# ============================================================

class_names = sorted(
    set(class_names)
)

class_to_id = {
    name: index
    for index, name in enumerate(class_names)
}

print(
    "\nClasses found:",
    len(class_names)
)

for index, name in enumerate(
    class_names
):

    print(
        f"{index:3d} : {name}"
    )

# ============================================================
# SECOND PASS
# CREATE X AND y
# ============================================================

X = []
y = []

print("\nBuilding dataset...")

for feature_file in feature_files:

    try:

        data = np.load(
            feature_file,
            allow_pickle=True
        )

        data = np.asarray(
            data,
            dtype=np.float32
        )

        if data.ndim != 2:
            continue

        class_name = feature_file.parent.name

        if class_name not in class_to_id:
            continue

        X.append(data)

        y.append(
            class_to_id[class_name]
        )

    except Exception:
        continue

# ============================================================
# CHECK SEQUENCE SHAPES
# ============================================================

sequence_lengths = [
    item.shape[0]
    for item in X
]

feature_sizes = [
    item.shape[1]
    for item in X
]

if len(set(sequence_lengths)) != 1:

    raise ValueError(
        "Feature sequences have different lengths."
    )

if len(set(feature_sizes)) != 1:

    raise ValueError(
        "Feature vectors have different sizes."
    )

SEQUENCE_LENGTH = sequence_lengths[0]
FEATURE_SIZE = feature_sizes[0]

print("\nDataset shape information:")
print(
    "Samples        :",
    len(X)
)

print(
    "Sequence length:",
    SEQUENCE_LENGTH
)

print(
    "Features/frame :",
    FEATURE_SIZE
)

# ============================================================
# CONVERT TO NUMPY
# ============================================================

X = np.asarray(
    X,
    dtype=np.float32
)

y = np.asarray(
    y,
    dtype=np.int32
)

print(
    "\nX shape:",
    X.shape
)

print(
    "y shape:",
    y.shape
)

# ============================================================
# NORMALIZE FEATURES
# ============================================================

mean = np.mean(
    X,
    axis=(0, 1),
    keepdims=True
)

std = np.std(
    X,
    axis=(0, 1),
    keepdims=True
)

std = np.maximum(
    std,
    1e-6
)

X = (
    X - mean
) / std

print(
    "✓ Feature normalization complete"
)

# ============================================================
# TRAIN / VALIDATION SPLIT
# ============================================================

X_train, X_val, y_train, y_val = (
    train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )
)

print("\nDataset split:")

print(
    "Training  :",
    X_train.shape,
    y_train.shape
)

print(
    "Validation:",
    X_val.shape,
    y_val.shape
)

# ============================================================
# BUILD BiLSTM MODEL
# ============================================================

print("\nBuilding BiLSTM model...")

inputs = keras.Input(
    shape=(
        SEQUENCE_LENGTH,
        FEATURE_SIZE
    )
)

x = layers.Bidirectional(
    layers.LSTM(
        128,
        return_sequences=True
    )
)(inputs)

x = layers.Dropout(
    0.3
)(x)

x = layers.Bidirectional(
    layers.LSTM(
        64
    )
)(x)

x = layers.Dropout(
    0.3
)(x)

x = layers.Dense(
    128,
    activation="relu"
)(x)

x = layers.Dropout(
    0.3
)(x)

outputs = layers.Dense(
    len(class_names),
    activation="softmax"
)(x)

model = keras.Model(
    inputs=inputs,
    outputs=outputs
)

# ============================================================
# COMPILE
# ============================================================

model.compile(

    optimizer=keras.optimizers.Adam(
        learning_rate=LEARNING_RATE
    ),

    loss="sparse_categorical_crossentropy",

    metrics=[
        "accuracy"
    ]
)

# ============================================================
# MODEL SUMMARY
# ============================================================

model.summary()

# ============================================================
# CALLBACKS
# ============================================================

callbacks = [

    keras.callbacks.ModelCheckpoint(

        filepath=str(
            MODEL_PATH
        ),

        monitor="val_accuracy",

        mode="max",

        save_best_only=True,

        verbose=1
    ),

    keras.callbacks.ReduceLROnPlateau(

        monitor="val_loss",

        factor=0.5,

        patience=3,

        min_lr=1e-6,

        verbose=1
    ),

    keras.callbacks.EarlyStopping(

        monitor="val_accuracy",

        mode="max",

        patience=7,

        restore_best_weights=True,

        verbose=1
    )
]

# ============================================================
# TRAIN
# ============================================================

print("\n" + "=" * 70)
print("STARTING TRAINING")
print("=" * 70)

history = model.fit(

    X_train,
    y_train,

    validation_data=(
        X_val,
        y_val
    ),

    epochs=EPOCHS,

    batch_size=BATCH_SIZE,

    callbacks=callbacks,

    shuffle=True,

    verbose=1
)

# ============================================================
# SAVE FINAL MODEL
# ============================================================

model.save(
    MODEL_PATH
)

# ============================================================
# SAVE CLASS NAMES
# ============================================================

np.save(
    CLASS_PATH,
    np.asarray(
        class_names,
        dtype=object
    )
)

# ============================================================
# SAVE NORMALIZATION PARAMETERS
# ============================================================

np.save(
    MODEL_DIR / "sign_bilstm_mean.npy",
    mean
)

np.save(
    MODEL_DIR / "sign_bilstm_std.npy",
    std
)

# ============================================================
# FINAL EVALUATION
# ============================================================

loss, accuracy = model.evaluate(
    X_val,
    y_val,
    verbose=0
)

# ============================================================
# RESULTS
# ============================================================

print("\n" + "=" * 70)
print("TRAINING COMPLETE")
print("=" * 70)

print(
    "Validation loss    :",
    loss
)

print(
    "Validation accuracy:",
    f"{accuracy * 100:.2f}%"
)

print(
    "\nModel saved:"
)

print(
    MODEL_PATH
)

print(
    "\nClasses saved:"
)

print(
    CLASS_PATH
)

print(
    "\nNormalization saved:"
)

print(
    MODEL_DIR / "sign_bilstm_mean.npy"
)

print(
    MODEL_DIR / "sign_bilstm_std.npy"
)

print("=" * 70)