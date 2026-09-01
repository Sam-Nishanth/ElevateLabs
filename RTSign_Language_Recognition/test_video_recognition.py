import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp

from collections import deque, Counter

# ============================================================
# CONFIGURATION
# ============================================================

VIDEO_PATH = r"D:\Elevate labs\ElevateLabs\RTSign_Language_Recognition\data\isl\sign_project\11.monsoon\Monsoon 3.mp4"

MODEL_PATH = "models/sign_bilstm.keras"
CLASS_PATH = "models/isl_classes_bilstm.npy"
HAND_MODEL = "models/hand_landmarker.task"

SEQUENCE_LENGTH = 32

NUM_HANDS = 2
FEATURES_PER_HAND = 63
FEATURE_SIZE = 126

SMOOTHING_SIZE = 7
CONFIDENCE_THRESHOLD = 0.60

# Number of consecutive stable predictions required
STABLE_COUNT = 8


# ============================================================
# LOAD MODEL
# ============================================================

print()
print("======================================")
print("Loading ISL BiLSTM model...")
print("======================================")

model = tf.keras.models.load_model(MODEL_PATH)

class_names = np.load(
    CLASS_PATH,
    allow_pickle=True
)

print("BiLSTM loaded.")
print("ISL classes:", len(class_names))


# ============================================================
# MEDIAPIPE
# ============================================================

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=HAND_MODEL
    ),
    running_mode=RunningMode.VIDEO,
    num_hands=NUM_HANDS,

    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_frame_features(result):

    features = np.zeros(
        FEATURE_SIZE,
        dtype=np.float32
    )

    if not result.hand_landmarks:
        return features

    for hand_index, hand in enumerate(
        result.hand_landmarks[:NUM_HANDS]
    ):

        points = np.array(
            [
                [lm.x, lm.y, lm.z]
                for lm in hand
            ],
            dtype=np.float32
        )

        # ------------------------------------------
        # Wrist-relative coordinates
        # ------------------------------------------

        wrist = points[0].copy()

        points = points - wrist

        # ------------------------------------------
        # Scale normalization
        # ------------------------------------------

        distances = np.linalg.norm(
            points,
            axis=1
        )

        scale = np.max(distances)

        if scale > 1e-6:
            points = points / scale

        # ------------------------------------------
        # Store 63 coordinates
        # ------------------------------------------

        offset = (
            hand_index *
            FEATURES_PER_HAND
        )

        features[
            offset:
            offset + FEATURES_PER_HAND
        ] = points.flatten()

    return features


# ============================================================
# OPEN VIDEO
# ============================================================

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():

    print()
    print("ERROR: Could not open video.")
    print(VIDEO_PATH)

    raise SystemExit


fps = cap.get(cv2.CAP_PROP_FPS)

if fps <= 0:
    fps = 30

total_frames = int(
    cap.get(cv2.CAP_PROP_FRAME_COUNT)
)

print()
print("Video opened successfully.")
print("FPS:", fps)
print("Total frames:", total_frames)


# ============================================================
# BUFFERS
# ============================================================

sequence = deque(
    maxlen=SEQUENCE_LENGTH
)

prediction_history = deque(
    maxlen=SMOOTHING_SIZE
)

# Final recognized words/signs
final_words = []

# Last stable prediction
last_stable_prediction = None

# Count how long the same prediction remains stable
stable_counter = 0

# Prevent adding the same prediction repeatedly
last_added_prediction = None


# ============================================================
# TIMESTAMP
# ============================================================

timestamp = 0


# ============================================================
# VIDEO PROCESSING
# ============================================================

print()
print("======================================")
print("Processing video...")
print("======================================")
print("Press Q to stop.")
print()


with HandLandmarker.create_from_options(
    options
) as landmarker:

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        # ------------------------------------------
        # Mirror frame
        # ------------------------------------------

        frame = cv2.flip(
            frame,
            1
        )

        # ------------------------------------------
        # RGB conversion
        # ------------------------------------------

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        # ------------------------------------------
        # Timestamp
        # ------------------------------------------

        timestamp += 1

        # ------------------------------------------
        # MediaPipe detection
        # ------------------------------------------

        result = landmarker.detect_for_video(
            mp_image,
            timestamp
        )

        # ------------------------------------------
        # Extract coordinates
        # ------------------------------------------

        features = extract_frame_features(
            result
        )

        sequence.append(
            features
        )

        prediction_text = "Collecting..."
        confidence = 0.0

        # ==================================================
        # PREDICTION
        # ==================================================

        if len(sequence) == SEQUENCE_LENGTH:

            X = np.array(
                sequence,
                dtype=np.float32
            )

            X = np.expand_dims(
                X,
                axis=0
            )

            # Shape:
            # (1, 32, 126)

            probabilities = model.predict(
                X,
                verbose=0
            )[0]

            predicted_index = int(
                np.argmax(probabilities)
            )

            confidence = float(
                probabilities[
                    predicted_index
                ]
            )

            predicted_class = str(
                class_names[
                    predicted_index
                ]
            )

            # ----------------------------------------------
            # Confidence filter
            # ----------------------------------------------

            if confidence >= CONFIDENCE_THRESHOLD:

                prediction_history.append(
                    predicted_class
                )

                # ------------------------------------------
                # Most common recent prediction
                # ------------------------------------------

                current_prediction = Counter(
                    prediction_history
                ).most_common(1)[0][0]

                prediction_text = (
                    current_prediction
                )

                # ------------------------------------------
                # Stability detection
                # ------------------------------------------

                if current_prediction == last_stable_prediction:

                    stable_counter += 1

                else:

                    last_stable_prediction = (
                        current_prediction
                    )

                    stable_counter = 1

                # ------------------------------------------
                # Add to final sentence
                # ------------------------------------------

                if stable_counter >= STABLE_COUNT:

                    if (
                        current_prediction
                        != last_added_prediction
                    ):

                        final_words.append(
                            current_prediction
                        )

                        last_added_prediction = (
                            current_prediction
                        )

                        print(
                            "Recognized:",
                            current_prediction
                        )

                        # Reset so another occurrence
                        # of the same sign can be detected
                        stable_counter = 0

            else:

                prediction_text = "Uncertain"

        # ==================================================
        # DISPLAY
        # ==================================================

        # Small information box
        cv2.rectangle(
            frame,
            (10, 10),
            (520, 105),
            (0, 0, 0),
            -1
        )

        # Current prediction
        cv2.putText(
            frame,
            f"Sign: {prediction_text}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        # Confidence
        cv2.putText(
            frame,
            f"Confidence: {confidence * 100:.1f}%",
            (20, 68),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1
        )

        # Buffer
        cv2.putText(
            frame,
            f"Frames: {len(sequence)}/{SEQUENCE_LENGTH}",
            (20, 93),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (255, 255, 255),
            1
        )

        # ==================================================
        # FINAL OUTPUT
        # ==================================================

        sentence = " ".join(
            final_words
        )

        # Bottom black box
        height, width = frame.shape[:2]

        cv2.rectangle(
            frame,
            (10, height - 65),
            (width - 10, height - 10),
            (0, 0, 0),
            -1
        )

        cv2.putText(
            frame,
            f"Output: {sentence}",
            (20, height - 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2
        )

        # ------------------------------------------
        # Show video
        # ------------------------------------------

        cv2.imshow(
            "RTSign - ISL Video Recognition",
            frame
        )

        # ------------------------------------------
        # Q = quit
        # ------------------------------------------

        key = cv2.waitKey(
            max(1, int(1000 / fps))
        ) & 0xFF

        if key == ord("q"):
            break


# ============================================================
# CLEANUP
# ============================================================

cap.release()
cv2.destroyAllWindows()


# ============================================================
# FINAL RESULT
# ============================================================

print()
print("======================================")
print("VIDEO PROCESSING COMPLETE")
print("======================================")

print()
print("FINAL OUTPUT:")

if final_words:

    print(
        " ".join(final_words)
    )

else:

    print(
        "No stable signs recognized."
    )

print()
print("======================================")