import pandas as pd
import joblib

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


CSV_PATH = "data/alphabet_coordinates.csv"
MODEL_PATH = "models/alphabet_coordinate_rf.pkl"


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(CSV_PATH)

print("=" * 60)
print("Alphabet Coordinate Random Forest Evaluation")
print("=" * 60)

print()
print("Dataset shape:", df.shape)


# ============================================================
# FEATURES / LABEL
# ============================================================

X = df.drop(columns=["label"])

y = df["label"]


print("Feature shape:", X.shape)
print("Samples:", len(X))


# ============================================================
# LOAD MODEL
# ============================================================

model_data = joblib.load(
    MODEL_PATH
)

if isinstance(model_data, dict):

    model = model_data["model"]

else:

    model = model_data


print()
print("Model:", type(model))


# ============================================================
# CLASS MAPPING
# ============================================================

class_names = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

print()
print("Class mapping:")

for i, name in enumerate(class_names):

    print(f"{i:2d} -> {name}")


# ============================================================
# PREDICTION
# ============================================================

print()
print("Running prediction...")


# Convert DataFrame to NumPy
# because the Random Forest was trained without feature names

X_numpy = X.to_numpy(
    dtype="float32"
)


predictions_numeric = model.predict(
    X_numpy
)


# ============================================================
# CONVERT NUMERIC PREDICTIONS → LETTERS
# ============================================================

predictions = [
    class_names[int(p)]
    for p in predictions_numeric
]


# ============================================================
# ACCURACY
# ============================================================

accuracy = accuracy_score(
    y,
    predictions
)


print()
print("=" * 60)

print(
    f"Accuracy: {accuracy * 100:.2f}%"
)

print("=" * 60)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print()
print("Classification Report:")

print(
    classification_report(
        y,
        predictions,
        labels=class_names,
        zero_division=0
    )
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

print()
print("Confusion Matrix:")

cm = confusion_matrix(
    y,
    predictions,
    labels=class_names
)

print(cm)