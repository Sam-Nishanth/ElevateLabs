# ============================================================
# RTSign Language Recognition
# Alphabet Feature Extraction
#
# Pipeline:
#
# Alphabet images
#       ↓
# MediaPipe Hand Landmarker
#       ↓
# 21 landmarks
#       ↓
# XYZ coordinates
#       ↓
# Wrist-relative coordinates
#       ↓
# Scale normalization
#       ↓
# 63 features
#       ↓
# alphabet_features.npz
#
# 21 landmarks × 3 coordinates = 63 features
# ============================================================

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import mediapipe as mp


# ============================================================
# PATHS
# ============================================================

BASE_DIR = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)

DATA_DIR = (
    BASE_DIR
    / "data"
    / "alphabet"
)

OUTPUT_DIR = (
    BASE_DIR
    / "features"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "alphabet_features.npz"
)


# ============================================================
# MEDIAPIPE MODEL
# ============================================================

HAND_MODEL_PATH = (
    BASE_DIR
    / "models"
    / "hand_landmarker.task"
)


# ============================================================
# SETTINGS
# ============================================================

NUM_HANDS = 1

FEATURES_PER_HAND = 63

NUM_LANDMARKS = 21


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("RTSign — ALPHABET FEATURE EXTRACTION")
print("=" * 70)

print(
    "Dataset:",
    DATA_DIR
)

print(
    "Output:",
    OUTPUT_FILE
)

print(
    "Hand model:",
    HAND_MODEL_PATH
)


# ============================================================
# CHECK MODEL
# ============================================================

if not HAND_MODEL_PATH.exists():

    raise FileNotFoundError(
        "\nMediaPipe model not found:\n"
        f"{HAND_MODEL_PATH}"
    )


# ============================================================
# CHECK DATASET
# ============================================================

if not DATA_DIR.exists():

    raise FileNotFoundError(
        "\nAlphabet dataset not found:\n"
        f"{DATA_DIR}"
    )


# ============================================================
# FIND IMAGE FILES
# ============================================================

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}


image_records = []


for class_dir in sorted(
    DATA_DIR.iterdir()
):

    if not class_dir.is_dir():
        continue

    class_name = class_dir.name

    for image_path in sorted(
        class_dir.rglob("*")
    ):

        if (
            image_path.is_file()
            and image_path.suffix.lower()
            in IMAGE_EXTENSIONS
        ):

            image_records.append(
                (
                    class_name,
                    image_path
                )
            )


print(
    "\nImages found:",
    len(image_records)
)


if not image_records:

    raise RuntimeError(
        "No alphabet images were found."
    )


# ============================================================
# CLASS MAPPING
# ============================================================

class_names = sorted(
    {
        record[0]
        for record in image_records
    }
)


class_to_id = {
    name: index
    for index, name in enumerate(
        class_names
    )
}


print(
    "Classes:",
    len(class_names)
)


print("\nClass mapping:")

for class_id, class_name in enumerate(
    class_names
):

    print(
        f"{class_id:3d} : {class_name}"
    )


# ============================================================
# MEDIAPIPE SETUP
# ============================================================

BaseOptions = (
    mp.tasks.BaseOptions
)

HandLandmarker = (
    mp.tasks.vision.HandLandmarker
)

HandLandmarkerOptions = (
    mp.tasks.vision.HandLandmarkerOptions
)

RunningMode = (
    mp.tasks.vision.RunningMode
)


options = HandLandmarkerOptions(

    base_options=BaseOptions(
        model_asset_path=str(
            HAND_MODEL_PATH
        )
    ),

    running_mode=RunningMode.IMAGE,

    num_hands=NUM_HANDS,

    min_hand_detection_confidence=0.5,

    min_hand_presence_confidence=0.5,

    min_tracking_confidence=0.5
)


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_features(
    result
):
    """
    Extract one hand.

    21 landmarks × XYZ
    = 63 features

    Normalization:
        1. Wrist-relative coordinates
        2. Scale normalization
    """

    if not result.hand_landmarks:

        return None


    # --------------------------------------------------------
    # Use first detected hand
    # --------------------------------------------------------

    hand = (
        result.hand_landmarks[0]
    )


    # --------------------------------------------------------
    # Convert landmarks to XYZ
    # --------------------------------------------------------

    points = np.array(

        [
            [
                landmark.x,
                landmark.y,
                landmark.z
            ]

            for landmark in hand
        ],

        dtype=np.float32
    )


    # --------------------------------------------------------
    # Verify landmark count
    # --------------------------------------------------------

    if points.shape != (
        NUM_LANDMARKS,
        3
    ):

        return None


    # --------------------------------------------------------
    # Wrist-relative coordinates
    # --------------------------------------------------------

    wrist = points[0].copy()

    points = (
        points - wrist
    )


    # --------------------------------------------------------
    # Scale normalization
    # --------------------------------------------------------

    distances = np.linalg.norm(
        points,
        axis=1
    )


    scale = np.max(
        distances
    )


    if scale > 1e-6:

        points = (
            points / scale
        )


    # --------------------------------------------------------
    # Flatten
    # --------------------------------------------------------

    features = (
        points.flatten()
    )


    # --------------------------------------------------------
    # Verify
    # --------------------------------------------------------

    if features.shape != (
        FEATURES_PER_HAND,
    ):

        return None


    return features.astype(
        np.float32
    )


# ============================================================
# EXTRACT DATASET
# ============================================================

X = []

y = []

successful = 0

failed = 0

no_hand = 0

failure_records = []


print("\n" + "=" * 70)
print("PROCESSING IMAGES")
print("=" * 70)


start_time = __import__(
    "time"
).time()


# ============================================================
# CREATE LANDMARKER
# ============================================================

with HandLandmarker.create_from_options(
    options
) as landmarker:


    for index, (
        class_name,
        image_path
    ) in enumerate(
        image_records,
        start=1
    ):


        # ----------------------------------------------------
        # Read image
        # ----------------------------------------------------

        image = cv2.imread(
            str(image_path)
        )


        if image is None:

            failed += 1

            failure_records.append({

                "class": class_name,

                "path": str(
                    image_path
                ),

                "reason":
                    "Could not read image"

            })

            continue


        # ----------------------------------------------------
        # BGR → RGB
        # ----------------------------------------------------

        rgb = cv2.cvtColor(

            image,

            cv2.COLOR_BGR2RGB

        )


        # ----------------------------------------------------
        # MediaPipe image
        # ----------------------------------------------------

        mp_image = mp.Image(

            image_format=(
                mp.ImageFormat.SRGB
            ),

            data=rgb

        )


        # ----------------------------------------------------
        # Detect hand
        # ----------------------------------------------------

        try:

            result = (
                landmarker.detect(
                    mp_image
                )
            )

        except Exception as e:

            failed += 1

            failure_records.append({

                "class": class_name,

                "path": str(
                    image_path
                ),

                "reason":
                    f"MediaPipe error: {e}"

            })

            continue


        # ----------------------------------------------------
        # Extract features
        # ----------------------------------------------------

        features = extract_features(
            result
        )


        if features is None:

            no_hand += 1

            failure_records.append({

                "class": class_name,

                "path": str(
                    image_path
                ),

                "reason":
                    "No valid hand detected"

            })

            continue


        # ----------------------------------------------------
        # Store
        # ----------------------------------------------------

        X.append(
            features
        )

        y.append(
            class_to_id[
                class_name
            ]
        )

        successful += 1


        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        if (
            index % 500 == 0
            or index == len(
                image_records
            )
        ):

            elapsed = (
                __import__(
                    "time"
                ).time()
                -
                start_time
            )

            rate = (
                index / elapsed
                if elapsed > 0
                else 0
            )

            remaining = (
                len(image_records)
                -
                index
            )

            eta = (
                remaining / rate
                if rate > 0
                else 0
            )

            print(

                f"{index}/"
                f"{len(image_records)} "
                f"| OK: {successful} "
                f"| No hand: {no_hand} "
                f"| Failed: {failed} "
                f"| ETA: "
                f"{eta / 60:.1f} min"

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
    dtype=np.int64
)


# ============================================================
# VERIFY
# ============================================================

print("\n" + "=" * 70)
print("VERIFYING DATASET")
print("=" * 70)


print(
    "X shape:",
    X.shape
)

print(
    "y shape:",
    y.shape
)

print(
    "Classes:",
    len(class_names)
)


if X.ndim != 2:

    raise ValueError(
        f"Unexpected X shape: {X.shape}"
    )


if X.shape[1] != FEATURES_PER_HAND:

    raise ValueError(
        f"Expected 63 features, "
        f"got {X.shape[1]}"
    )


if len(X) != len(y):

    raise ValueError(
        "X and y sample counts do not match."
    )


if not np.isfinite(X).all():

    raise ValueError(
        "Feature data contains NaN or Inf."
    )


print(
    "✓ Feature shape verified"
)

print(
    "✓ Labels verified"
)

print(
    "✓ No NaN/Inf values"
)


# ============================================================
# SAVE
# ============================================================

print("\nSaving feature dataset...")


np.savez_compressed(

    OUTPUT_FILE,

    X_train=X,

    y_train=y,

    y_test=np.array(
        [],
        dtype=np.int64
    ),

    class_names=np.asarray(
        class_names
    )

)


# ============================================================
# SAVE FAILURE REPORT
# ============================================================

if failure_records:

    failure_file = (
        OUTPUT_DIR
        / "alphabet_feature_failures.csv"
    )

    pd.DataFrame(
        failure_records
    ).to_csv(

        failure_file,

        index=False
    )

    print(
        "\nFailure report:",
        failure_file
    )


# ============================================================
# FINAL REPORT
# ============================================================

total_time = (
    __import__(
        "time"
    ).time()
    -
    start_time
)


print("\n" + "=" * 70)
print("ALPHABET FEATURE EXTRACTION COMPLETE")
print("=" * 70)

print(
    "Images found   :",
    len(image_records)
)

print(
    "Successful      :",
    successful
)

print(
    "No hand         :",
    no_hand
)

print(
    "Failed          :",
    failed
)

print(
    "Classes         :",
    len(class_names)
)

print(
    "Features/sample :",
    FEATURES_PER_HAND
)

print(
    "X shape         :",
    X.shape
)

print(
    "y shape         :",
    y.shape
)

print(
    "Time            :",
    f"{total_time / 60:.2f} minutes"
)

print(
    "\nSaved:",
    OUTPUT_FILE
)

print("=" * 70)