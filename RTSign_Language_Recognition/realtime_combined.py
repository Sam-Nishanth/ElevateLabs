import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import joblib

from collections import deque, Counter


# ============================================================
# CONFIGURATION
# ============================================================

ALPHABET_MODEL_PATH = "models/alphabet_coordinate_rf.pkl"

ISL_MODEL_PATH = "models/sign_bilstm.keras"
ISL_CLASS_PATH = "models/isl_classes_bilstm.npy"

HAND_MODEL_PATH = "models/hand_landmarker.task"


# ISL settings
SEQUENCE_LENGTH = 32
NUM_HANDS = 2
FEATURES_PER_HAND = 63
ISL_FEATURE_SIZE = 126

SMOOTHING_SIZE = 5
CONFIDENCE_THRESHOLD = 0.60


# ============================================================
# LOAD ALPHABET RANDOM FOREST
# ============================================================

print("======================================")
print("Loading Alphabet Random Forest...")
print("======================================")

alphabet_data = joblib.load(
    ALPHABET_MODEL_PATH
)

# Handle both possible formats:
# 1. Direct model
# 2. Dictionary containing model
if isinstance(alphabet_data, dict):

    if "model" in alphabet_data:
        alphabet_model = alphabet_data["model"]

    else:
        raise ValueError(
            "Alphabet model dictionary does not contain 'model'."
        )

else:

    alphabet_model = alphabet_data


# Alphabet classes
ALPHABET_CLASSES = np.array(
    list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
)

print(
    "Alphabet model loaded."
)

print(
    "Alphabet classes:",
    len(ALPHABET_CLASSES)
)


# ============================================================
# LOAD ISL BILSTM
# ============================================================

print()
print("======================================")
print("Loading ISL BiLSTM...")
print("======================================")

isl_model = tf.keras.models.load_model(
    ISL_MODEL_PATH
)

isl_classes = np.load(
    ISL_CLASS_PATH,
    allow_pickle=True
)

print(
    "BiLSTM loaded."
)

print(
    "ISL classes:",
    len(isl_classes)
)


# ============================================================
# MEDIAPIPE
# ============================================================

print()
print("======================================")
print("Loading MediaPipe...")
print("======================================")


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
        model_asset_path=HAND_MODEL_PATH
    ),

    running_mode=RunningMode.VIDEO,

    num_hands=NUM_HANDS,

    min_hand_detection_confidence=0.5,

    min_hand_presence_confidence=0.5,

    min_tracking_confidence=0.5
)


# ============================================================
# ALPHABET FEATURE EXTRACTION
# ============================================================

def extract_alphabet_features(result):

    """
    Extract 21 hand landmarks.

    Output:
        63 values
        21 landmarks × XYZ
    """

    if not result.hand_landmarks:

        return None


    # Use first detected hand
    hand = result.hand_landmarks[0]


    points = np.array(

        [
            [lm.x, lm.y, lm.z]
            for lm in hand
        ],

        dtype=np.float32
    )


    # --------------------------------------------------------
    # Wrist-relative coordinates
    # --------------------------------------------------------

    wrist = points[0].copy()

    points = points - wrist


    # --------------------------------------------------------
    # Scale normalization
    # --------------------------------------------------------

    distances = np.linalg.norm(
        points,
        axis=1
    )

    scale = np.max(distances)


    if scale > 1e-6:

        points = points / scale


    return points.flatten()


# ============================================================
# ISL FEATURE EXTRACTION
# ============================================================

def extract_isl_features(result):

    """
    Extract up to 2 hands.

    Each hand:
        21 landmarks × XYZ = 63

    Two hands:
        63 × 2 = 126
    """

    features = np.zeros(
        ISL_FEATURE_SIZE,
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


        # Wrist-relative coordinates
        wrist = points[0].copy()

        points = points - wrist


        # Scale normalization
        distances = np.linalg.norm(
            points,
            axis=1
        )

        scale = np.max(distances)


        if scale > 1e-6:

            points = points / scale


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
# WEBCAM
# ============================================================

cap = cv2.VideoCapture(0)


if not cap.isOpened():

    print(
        "ERROR: Could not open webcam."
    )

    raise SystemExit


# ============================================================
# INITIAL STATE
# ============================================================

mode = "NONE"

sequence = deque(
    maxlen=SEQUENCE_LENGTH
)

prediction_history = deque(
    maxlen=SMOOTHING_SIZE
)


timestamp = 0


print()
print("======================================")
print("RTSign Language Recognition")
print("======================================")
print()
print("Press A -> Alphabet Recognition")
print("Press I -> ISL Recognition")
print("Press Q -> Quit")
print()
print("Only one recognition model runs at a time.")
print()


# ============================================================
# MEDIAPIPE
# ============================================================

with HandLandmarker.create_from_options(
    options
) as landmarker:


    while True:


        # ====================================================
        # READ CAMERA
        # ====================================================

        success, frame = cap.read()


        if not success:

            print(
                "Failed to read webcam."
            )

            break


        # Mirror camera
        frame = cv2.flip(
            frame,
            1
        )


        # ====================================================
        # RGB CONVERSION
        # ====================================================

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )


        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )


        # ====================================================
        # TIMESTAMP
        # ====================================================

        timestamp += 1


        # ====================================================
        # MEDIAPIPE DETECTION
        # ====================================================

        result = landmarker.detect_for_video(
            mp_image,
            timestamp
        )


        # ====================================================
        # DEFAULT DISPLAY
        # ====================================================

        prediction_text = "Select A or I"

        confidence = 0.0


        # ====================================================
        # ALPHABET MODE
        # ====================================================

        if mode == "ALPHABET":


            features = extract_alphabet_features(
                result
            )


            if features is None:

                prediction_text = (
                    "No hand detected"
                )


            else:

                X = features.reshape(
                    1,
                    -1
                )


                # --------------------------------------------
                # Random Forest prediction
                # --------------------------------------------

                prediction = alphabet_model.predict(
                    X
                )[0]


                # --------------------------------------------
                # Convert numeric prediction to letter
                # --------------------------------------------

                if isinstance(
                    prediction,
                    (np.integer, int)
                ):

                    prediction_index = int(
                        prediction
                    )

                    if (
                        0 <=
                        prediction_index <
                        len(ALPHABET_CLASSES)
                    ):

                        prediction_text = (
                            ALPHABET_CLASSES[
                                prediction_index
                            ]
                        )

                    else:

                        prediction_text = str(
                            prediction
                        )

                else:

                    prediction_text = str(
                        prediction
                    )


                # --------------------------------------------
                # Confidence
                # --------------------------------------------

                if hasattr(
                    alphabet_model,
                    "predict_proba"
                ):

                    probabilities = (
                        alphabet_model.predict_proba(
                            X
                        )[0]
                    )

                    confidence = float(
                        np.max(
                            probabilities
                        )
                    )


        # ====================================================
        # ISL MODE
        # ====================================================

        elif mode == "ISL":


            features = extract_isl_features(
                result
            )


            # Add frame to sequence
            sequence.append(
                features
            )


            # --------------------------------------------
            # Wait until 32 frames
            # --------------------------------------------

            if len(sequence) < SEQUENCE_LENGTH:

                prediction_text = (
                    "Collecting..."
                )

            else:


                input_sequence = np.array(
                    sequence,
                    dtype=np.float32
                )


                input_sequence = np.expand_dims(
                    input_sequence,
                    axis=0
                )


                # Shape:
                # (1, 32, 126)

                probabilities = (
                    isl_model.predict(
                        input_sequence,
                        verbose=0
                    )[0]
                )


                predicted_index = int(
                    np.argmax(
                        probabilities
                    )
                )


                confidence = float(
                    probabilities[
                        predicted_index
                    ]
                )


                predicted_class = str(
                    isl_classes[
                        predicted_index
                    ]
                )


                # ----------------------------------------
                # Confidence filter
                # ----------------------------------------

                if (
                    confidence >=
                    CONFIDENCE_THRESHOLD
                ):


                    prediction_history.append(
                        predicted_class
                    )


                    if prediction_history:

                        most_common = Counter(
                            prediction_history
                        ).most_common(
                            1
                        )[0][0]


                        prediction_text = (
                            str(
                                most_common
                            )
                        )


                else:

                    prediction_text = (
                        "Uncertain"
                    )


        # ====================================================
        # DISPLAY
        # ====================================================

        # Smaller text and smaller information box

        cv2.rectangle(

            frame,

            (10, 10),

            (420, 95),

            (0, 0, 0),

            -1
        )


        # Mode

        cv2.putText(

            frame,

            f"Mode: {mode}",

            (20, 35),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.55,

            (255, 255, 255),

            1,

            cv2.LINE_AA
        )


        # Prediction

        cv2.putText(

            frame,

            f"Prediction: {prediction_text}",

            (20, 60),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.65,

            (0, 255, 0),

            2,

            cv2.LINE_AA
        )


        # Confidence

        if mode == "ALPHABET" or mode == "ISL":

            cv2.putText(

                frame,

                f"Confidence: {confidence * 100:.1f}%",

                (20, 84),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.45,

                (255, 255, 255),

                1,

                cv2.LINE_AA
            )


        # ====================================================
        # ISL BUFFER DISPLAY
        # ====================================================

        if mode == "ISL":

            cv2.putText(

                frame,

                f"Frames: {len(sequence)}/{SEQUENCE_LENGTH}",

                (10, 120),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.5,

                (255, 255, 255),

                1,

                cv2.LINE_AA
            )


        # ====================================================
        # CONTROLS
        # ====================================================

        cv2.putText(

            frame,

            "A: Alphabet   I: ISL   Q: Quit",

            (10, frame.shape[0] - 20),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.5,

            (255, 255, 255),

            1,

            cv2.LINE_AA
        )


        # ====================================================
        # SHOW CAMERA
        # ====================================================

        cv2.imshow(

            "RTSign Language Recognition",

            frame
        )


        # ====================================================
        # KEYBOARD
        # ====================================================

        key = cv2.waitKey(1) & 0xFF


        # --------------------------------------------
        # Alphabet
        # --------------------------------------------

        if key == ord("a"):

            mode = "ALPHABET"

            sequence.clear()

            prediction_history.clear()

            print(
                "Mode changed -> ALPHABET"
            )


        # --------------------------------------------
        # ISL
        # --------------------------------------------

        elif key == ord("i"):

            mode = "ISL"

            sequence.clear()

            prediction_history.clear()

            print(
                "Mode changed -> ISL"
            )


        # --------------------------------------------
        # Quit
        # --------------------------------------------

        elif key == ord("q"):

            break


# ============================================================
# CLEANUP
# ============================================================

cap.release()

cv2.destroyAllWindows()


print()
print("======================================")
print("Recognition stopped.")
print("======================================")