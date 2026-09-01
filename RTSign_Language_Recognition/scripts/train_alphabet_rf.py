# ============================================================
# RTSign Language Recognition
# Alphabet Coordinate Random Forest Training
#
# Input:
#   data/alphabet_coordinates.csv
#
# Format:
#   label + 63 coordinate features
#   21 landmarks × (x, y, z)
#
# Output:
#   models/alphabet_coordinate_rf.pkl
#   models/alphabet_coordinate_rf_classes.npy
#
# ============================================================

from pathlib import Path
import time
import pickle

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)

DATA_FILE = (
    BASE_DIR
    / "data"
    / "alphabet_coordinates.csv"
)

MODEL_DIR = (
    BASE_DIR
    / "models"
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

MODEL_FILE = (
    MODEL_DIR
    / "alphabet_coordinate_rf.pkl"
)

CLASS_FILE = (
    MODEL_DIR
    / "alphabet_coordinate_rf_classes.npy"
)


# ============================================================
# SETTINGS
# ============================================================

TEST_SIZE = 0.20

RANDOM_STATE = 42

N_ESTIMATORS = 300

N_JOBS = -1


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("RTSign — ALPHABET COORDINATE RANDOM FOREST")
print("=" * 70)


# ============================================================
# CHECK DATASET
# ============================================================

if not DATA_FILE.exists():

    raise FileNotFoundError(
        f"\nDataset not found:\n{DATA_FILE}"
    )


# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading dataset...")

df = pd.read_csv(
    DATA_FILE
)

print(
    "Dataset shape:",
    df.shape
)


# ============================================================
# CHECK LABEL
# ============================================================

if "label" not in df.columns:

    raise ValueError(
        "Dataset must contain a 'label' column."
    )


# ============================================================
# SEPARATE FEATURES AND LABELS
# ============================================================

y = df["label"]

X = df.drop(
    columns=["label"]
)


# ============================================================
# VERIFY FEATURE COUNT
# ============================================================

print(
    "\nFeature count:",
    X.shape[1]
)

if X.shape[1] != 63:

    raise ValueError(
        f"Expected 63 coordinate features, "
        f"but found {X.shape[1]}."
    )


# ============================================================
# CONVERT FEATURES
# ============================================================

X = X.to_numpy(
    dtype=np.float32
)

y = y.to_numpy()


# ============================================================
# CLASS NAMES
# ============================================================

class_names = np.array(
    sorted(
        np.unique(y)
    )
)

NUM_CLASSES = len(
    class_names
)


print(
    "Number of classes:",
    NUM_CLASSES
)

print(
    "\nClasses:"
)

for class_id, class_name in enumerate(
    class_names
):

    count = np.sum(
        y == class_name
    )

    print(
        f"{class_id:3d} : "
        f"{class_name:10s} "
        f"{count}"
    )


# ============================================================
# DATA VALIDATION
# ============================================================

print("\nChecking dataset...")

if not np.isfinite(X).all():

    raise ValueError(
        "Dataset contains NaN or infinite values."
    )


if len(X) != len(y):

    raise ValueError(
        "X and y have different numbers of samples."
    )


print(
    "✓ No invalid feature values"
)

print(
    "✓ X/y sample counts match"
)


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

print("\n" + "=" * 70)
print("TRAIN / TEST SPLIT")
print("=" * 70)


X_train, X_test, y_train, y_test = (
    train_test_split(

        X,

        y,

        test_size=TEST_SIZE,

        random_state=RANDOM_STATE,

        stratify=y
    )
)


print(
    "Training samples:",
    len(X_train)
)

print(
    "Testing samples :",
    len(X_test)
)


# ============================================================
# BUILD RANDOM FOREST
# ============================================================

print("\n" + "=" * 70)
print("BUILDING RANDOM FOREST")
print("=" * 70)


model = RandomForestClassifier(

    n_estimators=N_ESTIMATORS,

    random_state=RANDOM_STATE,

    n_jobs=N_JOBS,

    criterion="gini",

    max_depth=None,

    min_samples_split=2,

    min_samples_leaf=1,

    class_weight=None,

    verbose=1
)


print(
    "Trees:",
    N_ESTIMATORS
)

print(
    "Features:",
    X.shape[1]
)

print(
    "Classes:",
    NUM_CLASSES
)

print(
    "CPU threads:",
    N_JOBS
)


# ============================================================
# TRAIN
# ============================================================

print("\n" + "=" * 70)
print("STARTING TRAINING")
print("=" * 70)


start_time = time.time()


model.fit(
    X_train,
    y_train
)


training_time = (
    time.time()
    -
    start_time
)


# ============================================================
# PREDICTION
# ============================================================

print("\n" + "=" * 70)
print("EVALUATING MODEL")
print("=" * 70)


y_pred = model.predict(
    X_test
)


# ============================================================
# ACCURACY
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)


print(
    "\nTest accuracy:",
    f"{accuracy * 100:.2f}%"
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print(
    "\nClassification report:"
)

print(

    classification_report(

        y_test,

        y_pred,

        labels=class_names,

        target_names=class_names,

        zero_division=0

    )

)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(

    y_test,

    y_pred,

    labels=class_names

)


print(
    "Confusion matrix shape:",
    cm.shape
)


# ============================================================
# SAVE MODEL
# ============================================================

print("\nSaving Random Forest model...")


with open(
    MODEL_FILE,
    "wb"
) as f:

    pickle.dump(
        model,
        f,
        protocol=pickle.HIGHEST_PROTOCOL
    )


# ============================================================
# SAVE CLASS NAMES
# ============================================================

np.save(
    CLASS_FILE,
    class_names
)


# ============================================================
# SAVE TRAINING INFORMATION
# ============================================================

INFO_FILE = (
    MODEL_DIR
    / "alphabet_coordinate_rf_info.json"
)


import json


info = {

    "dataset": str(
        DATA_FILE
    ),

    "samples": int(
        len(X)
    ),

    "features": int(
        X.shape[1]
    ),

    "classes": int(
        NUM_CLASSES
    ),

    "train_samples": int(
        len(X_train)
    ),

    "test_samples": int(
        len(X_test)
    ),

    "n_estimators": int(
        N_ESTIMATORS
    ),

    "random_state": int(
        RANDOM_STATE
    ),

    "test_accuracy": float(
        accuracy
    ),

    "training_time_seconds": float(
        training_time
    )

}


with open(
    INFO_FILE,
    "w"
) as f:

    json.dump(
        info,
        f,
        indent=2
    )


# ============================================================
# FINAL REPORT
# ============================================================

print("\n" + "=" * 70)
print("ALPHABET RANDOM FOREST TRAINING COMPLETE")
print("=" * 70)

print(
    "Total samples     :",
    len(X)
)

print(
    "Features/sample   :",
    X.shape[1]
)

print(
    "Classes           :",
    NUM_CLASSES
)

print(
    "Training samples  :",
    len(X_train)
)

print(
    "Testing samples   :",
    len(X_test)
)

print(
    "Trees             :",
    N_ESTIMATORS
)

print(
    "Test accuracy     :",
    f"{accuracy * 100:.2f}%"
)

print(
    "Training time     :",
    f"{training_time / 60:.2f} minutes"
)

print("\nSaved:")

print(
    MODEL_FILE
)

print(
    CLASS_FILE
)

print(
    INFO_FILE
)

print("=" * 70)