# ============================================================
# RTSign Language Recognition
# ISL Feature Extraction
#
# Output:
#   features/isl_features_normalized.npz
#
# Feature format:
#   32 frames × 126 features
#
# 126 = 2 hands × 21 landmarks × (x,y,z)
# ============================================================

import os
import time
import json
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
    / "isl_sign_project"
)

MODEL_FILE = (
    BASE_DIR
    / "models"
    / "hand_landmarker.task"
)

FEATURE_DIR = (
    BASE_DIR
    / "features"
)

FEATURE_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_FILE = (
    FEATURE_DIR
    / "isl_features_normalized.npz"
)

FAILURE_FILE = (
    FEATURE_DIR
    / "isl_feature_failures.csv"
)


# ============================================================
# SETTINGS
# ============================================================

SEQUENCE_LENGTH = 32

NUM_HANDS = 2

LANDMARKS_PER_HAND = 21

FEATURES_PER_LANDMARK = 3

FEATURES_PER_FRAME = (
    NUM_HANDS
    * LANDMARKS_PER_HAND
    * FEATURES_PER_LANDMARK
)

VIDEO_EXTENSIONS = {
    ".mp4",
    ".avi",
    ".mov",
    ".mkv",
    ".webm"
}


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("RTSign — ISL FEATURE EXTRACTION")
print("=" * 70)

print(
    "Dataset:",
    DATA_DIR
)

print(
    "MediaPipe model:",
    MODEL_FILE
)

print(
    "Output:",
    OUTPUT_FILE
)

print(
    "Sequence length:",
    SEQUENCE_LENGTH
)

print(
    "Features/frame:",
    FEATURES_PER_FRAME
)


# ============================================================
# CHECK PATHS
# ============================================================

if not DATA_DIR.exists():

    raise FileNotFoundError(
        f"\nISL dataset not found:\n{DATA_DIR}"
    )

if not MODEL_FILE.exists():

    raise FileNotFoundError(
        f"\nMediaPipe hand model not found:\n{MODEL_FILE}"
    )


# ============================================================
# FIND VIDEOS
# ============================================================

video_records = []

for class_dir in sorted(
    DATA_DIR.iterdir()
):

    if not class_dir.is_dir():
        continue

    class_name = class_dir.name

    for video_path in sorted(
        class_dir.rglob("*")
    ):

        if (
            video_path.is_file()
            and video_path.suffix.lower()
            in VIDEO_EXTENSIONS
        ):

            video_records.append(
                (
                    class_name,
                    video_path
                )
            )


print(
    "\nVideos found:",
    len(video_records)
)


if not video_records:

    raise RuntimeError(
        "No ISL videos were found."
    )


# ============================================================
# CLASS MAPPING
# ============================================================

class_names = sorted(
    {
        item[0]
        for item in video_records
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

BaseOptions = mp.tasks.BaseOptions

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
            MODEL_FILE
        )
    ),

    running_mode=RunningMode.VIDEO,

    num_hands=NUM_HANDS,

    min_hand_detection_confidence=0.5,

    min_hand_presence_confidence=0.5,

    min_tracking_confidence=0.5
)


# ============================================================
# LANDMARK CONVERSION
# ============================================================

def hand_to_array(hand):
    """
    Convert MediaPipe hand landmarks
    to a (21, 3) numpy array.
    """

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

    if points.shape != (
        LANDMARKS_PER_HAND,
        3
    ):

        return None

    return points


# ============================================================
# NORMALIZE ONE HAND
# ============================================================

def normalize_hand(points):
    """
    Wrist-relative + scale normalization.

    points:
        (21, 3)
    """

    points = points.copy()

    # --------------------------------------------------------
    # Wrist-relative coordinates
    # --------------------------------------------------------

    wrist = points[0].copy()

    points -= wrist


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

        points /= scale


    return points


# ============================================================
# EXTRACT ONE VIDEO
# ============================================================

def extract_video_features(
    video_path
):

    cap = cv2.VideoCapture(
        str(video_path)
    )

    if not cap.isOpened():

        raise RuntimeError(
            "Could not open video."
        )


    total_frames = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )


    if fps <= 0:

        fps = 30.0


    # --------------------------------------------------------
    # Choose 32 frame positions
    # --------------------------------------------------------

    if total_frames <= 0:

        cap.release()

        raise RuntimeError(
            "Video contains no frames."
        )


    if total_frames >= SEQUENCE_LENGTH:

        frame_indices = np.linspace(

            0,

            total_frames - 1,

            SEQUENCE_LENGTH

        ).astype(
            np.int64
        )

    else:

        # If video has fewer than 32 frames,
        # use all available frames and repeat
        # the last valid frame.

        frame_indices = np.linspace(

            0,

            total_frames - 1,

            total_frames

        ).astype(
            np.int64
        )


    frame_features = []


    # --------------------------------------------------------
    # Create a fresh VIDEO-mode landmarker
    # for this video.
    # --------------------------------------------------------

    with HandLandmarker.create_from_options(
        options
    ) as landmarker:


        last_feature = np.zeros(
            FEATURES_PER_FRAME,
            dtype=np.float32
        )


        for frame_index in frame_indices:


            cap.set(
                cv2.CAP_PROP_POS_FRAMES,
                int(frame_index)
            )


            ret, frame = cap.read()


            if not ret or frame is None:

                frame_features.append(
                    last_feature.copy()
                )

                continue


            rgb = cv2.cvtColor(

                frame,

                cv2.COLOR_BGR2RGB

            )


            mp_image = mp.Image(

                image_format=(
                    mp.ImageFormat.SRGB
                ),

                data=rgb

            )


            # ------------------------------------------------
            # IMPORTANT:
            # MediaPipe VIDEO mode requires
            # monotonically increasing timestamps.
            #
            # We process selected frames in increasing
            # order and assign timestamp from frame number.
            # ------------------------------------------------

            timestamp_ms = int(
                (
                    int(frame_index)
                    /
                    fps
                )
                * 1000
            )


            try:

                result = (
                    landmarker.detect_for_video(
                        mp_image,
                        timestamp_ms
                    )
                )

            except Exception as e:

                raise RuntimeError(
                    f"MediaPipe error: {e}"
                )


            # ------------------------------------------------
            # Create 2-hand feature vector
            # ------------------------------------------------

            feature_vector = np.zeros(
                FEATURES_PER_FRAME,
                dtype=np.float32
            )


            if result.hand_landmarks:

                # ------------------------------------------------
                # MediaPipe returns detected hands.
                #
                # We use at most two.
                # ------------------------------------------------

                for hand_index, hand in enumerate(
                    result.hand_landmarks[:NUM_HANDS]
                ):

                    points = hand_to_array(
                        hand
                    )


                    if points is None:
                        continue


                    points = normalize_hand(
                        points
                    )


                    start = (
                        hand_index
                        * 63
                    )

                    end = (
                        start
                        + 63
                    )


                    feature_vector[
                        start:end
                    ] = points.flatten()


            # ------------------------------------------------
            # Temporal fallback
            #
            # If no hand was detected, use the previous
            # valid feature vector.
            # ------------------------------------------------

            if not result.hand_landmarks:

                feature_vector = (
                    last_feature.copy()
                )


            else:

                last_feature = (
                    feature_vector.copy()
                )


            frame_features.append(
                feature_vector
            )


    cap.release()


    # ========================================================
    # Convert sequence
    # ========================================================

    sequence = np.asarray(
        frame_features,
        dtype=np.float32
    )


    # ========================================================
    # Pad to exactly 32 frames
    # ========================================================

    if len(sequence) < SEQUENCE_LENGTH:

        if len(sequence) == 0:

            sequence = np.zeros(

                (
                    SEQUENCE_LENGTH,
                    FEATURES_PER_FRAME
                ),

                dtype=np.float32

            )

        else:

            padding = np.repeat(

                sequence[-1:],

                SEQUENCE_LENGTH
                -
                len(sequence),

                axis=0

            )

            sequence = np.concatenate(

                [
                    sequence,
                    padding
                ],

                axis=0

            )


    # ========================================================
    # Ensure exactly 32
    # ========================================================

    sequence = sequence[
        :SEQUENCE_LENGTH
    ]


    if sequence.shape != (
        SEQUENCE_LENGTH,
        FEATURES_PER_FRAME
    ):

        raise RuntimeError(
            f"Unexpected sequence shape: "
            f"{sequence.shape}"
        )


    return sequence


# ============================================================
# PROCESS DATASET
# ============================================================

X = []

y = []

failures = []

successful = 0

failed = 0

start_time = time.time()


print("\n" + "=" * 70)
print("PROCESSING VIDEOS")
print("=" * 70)


for index, (
    class_name,
    video_path
) in enumerate(
    video_records,
    start=1
):

    try:

        sequence = extract_video_features(
            video_path
        )


        X.append(
            sequence
        )

        y.append(
            class_to_id[class_name]
        )

        successful += 1


    except Exception as e:

        failed += 1

        failures.append({

            "class": class_name,

            "video": video_path.name,

            "path": str(
                video_path
            ),

            "error": str(e)

        })


    # --------------------------------------------------------
    # Progress
    # --------------------------------------------------------

    if (
        index % 10 == 0
        or index == len(video_records)
    ):

        elapsed = (
            time.time()
            -
            start_time
        )

        rate = (
            index / elapsed
            if elapsed > 0
            else 0
        )

        remaining = (
            len(video_records)
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
            f"{len(video_records)} "
            f"| OK: {successful} "
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
print("VERIFYING FEATURES")
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


expected_shape = (
    SEQUENCE_LENGTH,
    FEATURES_PER_FRAME
)


if len(X) > 0:

    if X.shape[1:] != expected_shape:

        raise ValueError(
            f"Expected each sequence to be "
            f"{expected_shape}, "
            f"got {X.shape[1:]}"
        )


if len(X) != len(y):

    raise ValueError(
        "X and y sample counts do not match."
    )


if len(X) > 0:

    if not np.isfinite(X).all():

        raise ValueError(
            "X contains NaN or Inf."
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
# SAVE FEATURE DATASET
# ============================================================

print("\nSaving features...")


np.savez_compressed(

    OUTPUT_FILE,

    X=X,

    y=y,

    class_names=np.asarray(
        class_names
    )

)


# ============================================================
# SAVE FAILURE REPORT
# ============================================================

if failures:

    pd.DataFrame(
        failures
    ).to_csv(

        FAILURE_FILE,

        index=False

    )

    print(
        "\nFailure report:",
        FAILURE_FILE
    )


# ============================================================
# FINAL REPORT
# ============================================================

total_time = (
    time.time()
    -
    start_time
)


print("\n" + "=" * 70)
print("ISL FEATURE EXTRACTION COMPLETE")
print("=" * 70)

print(
    "Videos found:",
    len(video_records)
)

print(
    "Successful:",
    successful
)

print(
    "Failed:",
    failed
)

print(
    "Classes:",
    len(class_names)
)

print(
    "Sequence length:",
    SEQUENCE_LENGTH
)

print(
    "Features/frame:",
    FEATURES_PER_FRAME
)

print(
    "Final X shape:",
    X.shape
)

print(
    "Final y shape:",
    y.shape
)

print(
    "Total time:",
    f"{total_time / 60:.2f} minutes"
)

print(
    "\nSaved:"
)

print(
    OUTPUT_FILE
)

print("=" * 70)