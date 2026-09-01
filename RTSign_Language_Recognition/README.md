# RTSign Language Recognition

Real-time Indian Sign Language and alphabet recognition using MediaPipe, Random Forest, and BiLSTM.

## Requirements

* Windows 10/11
* Python 3.10+
* Git
* Webcam
* 8 GB RAM or more recommended
* NVIDIA GPU recommended for training, but not required for inference

---

## 1. Clone the Repository

Open PowerShell:

```powershell
git clone https://github.com/YOUR_USERNAME/RTSign_Language_Recognition.git
cd RTSign_Language_Recognition
```

---

## 2. Create Python Environment

```powershell
python -m venv tf_env
```

Activate it:

```powershell
.\tf_env\Scripts\Activate.ps1
```

If PowerShell blocks activation:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\tf_env\Scripts\Activate.ps1
```

---

## 3. Install Dependencies

Upgrade pip:

```powershell
python -m pip install --upgrade pip
```

Install the project requirements:

```powershell
pip install -r requirements.txt
```

Check the installation:

```powershell
python -c "import cv2, numpy, pandas, sklearn, tensorflow, mediapipe; print('Installation successful')"
```

---

# 4. Download MediaPipe Hand Model

Download the official MediaPipe Hand Landmarker model:

[Download hand_landmarker.task](https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task?utm_source=chatgpt.com)

Official documentation:

[MediaPipe Hand Landmarker documentation](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker?utm_source=chatgpt.com)

Create the models directory:

```powershell
mkdir models
```

Place the downloaded file here:

```text
models/
└── hand_landmarker.task
```

---

# 5. Download the Dataset

The trained models in this project were created from a project-specific dataset.

The exact dataset used for the original model is **not stored in this GitHub repository**.

If you have access to the original project dataset, place it according to the structure below.

## Alphabet Dataset

```text
data/
└── alphabet/
    ├── A/
    ├── B/
    ├── C/
    └── ...
```

Each folder represents an alphabet class.

---

## ISL Video Dataset

```text
data/
└── isl_sign_project/
    ├── class_1/
    ├── class_2/
    ├── class_3/
    └── ...
```

Each folder represents an ISL sign class.

### Public ISL dataset alternatives

If you do not have the original project dataset, you can obtain publicly available ISL datasets for retraining.

### INCLUDE Dataset

[INCLUDE dataset – AI4Bharat / Zenodo information](https://data.niaid.nih.gov/resources?id=zenodo_4010759&utm_source=chatgpt.com)

The INCLUDE dataset contains Indian Sign Language videos and provides downloadable train/test data. It is a **different dataset** from the original dataset used by this project, so its classes and training configuration will not exactly match the existing model.

---

# 6. Prepare the Model Directory

After downloading the MediaPipe model:

```text
RTSign_Language_Recognition/
│
├── models/
│   └── hand_landmarker.task
│
├── data/
│
├── features/
│
└── scripts/
```

---

# 7. Run the Existing Trained Models

If the trained model files have been provided separately, place them inside `models/`.

Required files:

```text
models/
├── hand_landmarker.task
├── alphabet_coordinate_rf.pkl
├── sign_bilstm.keras
└── isl_classes_bilstm.npy
```

Then run:

```powershell
python realtime_combined.py
```

The webcam-based recognition application will start.

---

# 8. Train the Models From Dataset

If the trained model files are not provided, they must be generated from the datasets.

## Alphabet Features

Run:

```powershell
python scripts/extract_alphabet_features.py
```

Then train the alphabet model:

```powershell
python scripts/train_alphabet_rf.py
```

The generated model will be:

```text
models/alphabet_coordinate_rf.pkl
```

---

## ISL Features

Place the ISL videos in:

```text
data/isl_sign_project/
```

Then run:

```powershell
python scripts/extract_isl_features.py
```

Train the BiLSTM:

```powershell
python scripts/train_sign_bilstm.py
```

The generated model will be:

```text
models/sign_bilstm.keras
```

---

# 9. Run the Application

After the required models are available:

```powershell
python realtime_combined.py
```

Allow camera access when Windows requests permission.

---

# 10. Test With a Video

If the video-testing script is available locally:

```powershell
python test_video_recognition.py
```

Provide the path of the video when requested.

Example:

```text
D:\Videos\test.mp4
```

For meaningful evaluation, use videos that were not included in the training dataset.

---

# 11. Check GPU

To check whether TensorFlow detects an NVIDIA GPU:

```powershell
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

If a GPU is detected, it will be displayed in the output.

The application can still run without a GPU.

---

# 12. Complete Setup

For a system where the datasets are already available:

```powershell
git clone <repo URL>

cd RTSign_Language_Recognition

python -m venv tf_env

.\tf_env\Scripts\Activate.ps1

pip install -r requirements.txt
```

Download:

```text
models/hand_landmarker.task
```

Place the datasets:

```text
data/alphabet/
data/isl_sign_project/
```

Generate features:

```powershell
python scripts/extract_alphabet_features.py

python scripts/extract_isl_features.py
```

Train:

```powershell
python scripts/train_alphabet_rf.py

python scripts/train_sign_bilstm.py
```

Finally run:

```powershell
python realtime_combined.py
```

---

# 13. Quick Start With Existing Models

If you already have the trained models, you do **not** need to regenerate the feature datasets or retrain the models.

Just:

```powershell
git clone https://github.com/YOUR_USERNAME/RTSign_Language_Recognition.git

cd RTSign_Language_Recognition

python -m venv tf_env

.\tf_env\Scripts\Activate.ps1

pip install -r requirements.txt
```

Place:

```text
models/
├── hand_landmarker.task
├── alphabet_coordinate_rf.pkl
├── sign_bilstm.keras
└── isl_classes_bilstm.npy
```

Then:

```powershell
python realtime_combined.py
```

---

## Dataset Note

The exact original project dataset should be used if you want to reproduce the existing trained models. Public datasets such as INCLUDE are alternatives for retraining, not drop-in replacements.
