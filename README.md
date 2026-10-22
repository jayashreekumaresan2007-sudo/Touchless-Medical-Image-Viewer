# Touchless-Medical-Image-Viewer
A touchless medical image viewer using Python, OpenCV and MediaPipe hand gestures.
# 🩻 Touchless Medical Image Viewer

A touchless medical image viewer built using Python, OpenCV, and MediaPipe. The application allows users to interact with medical images using hand gestures, reducing the need for direct mouse interaction.

## ✨ Features

- 🖐️ **Hand Gesture Recognition** — Detects hand movements using a webcam.
- 🔍 **Zoom** — Use a pinch gesture with your thumb and index finger.
- ✋ **Pan** — Move the index finger to navigate the image.
- 🔄 **Rotate** — Use an open-palm gesture to rotate the image.
- ✊ **Pause** — Use a closed-fist gesture to pause gesture-based interaction.
- 🖼️ **Image Selection** — Load medical images from your computer.
- 🖥️ **Graphical User Interface** — Dark-themed interface for image viewing and camera controls.
- ♻️ **Reset View** — Restore the image view to its original state.

## 🛠️ Technologies Used

- Python
- OpenCV
- MediaPipe
- NumPy
- CustomTkinter
- Pillow

## 📂 Project Structure

```text
Touchless-Medical-Image-Viewer/
├── ui.py
├── main.py
├── hand_landmarker.task
├── sample_xray.png
└── README.md
```

## ⚙️ Installation

### 1. Install Python

Install Python 3.11 or another version compatible with the required MediaPipe and project dependencies.

### 2. Clone the repository

```bash
git clone https://github.com/jayashreekumaresan2007-sudo/Touchless-Medical-Image-Viewer.git
cd Touchless-Medical-Image-Viewer
```

### 3. Create and activate a virtual environment

**Windows:**

```bash
python -m venv .venv
.venv\Scripts\activate
```

### 4. Install dependencies

```bash
pip install opencv-python mediapipe numpy customtkinter pillow
```

### 5. Run the application

```bash
python ui.py
```

Allow webcam access when prompted, click **Start Camera**, and select a medical image to begin interacting with it using hand gestures.

## 🎯 Project Objective

The goal of this project is to explore touchless human-computer interaction for medical image viewing using computer vision and hand gesture recognition.

## 🔮 Future Enhancements

- Support for DICOM medical imaging files
- Additional hand gestures and controls
- Improved image navigation and accessibility
- User-friendly image management
- Enhanced gesture recognition and performance

## ⚠️ Disclaimer

This project is an educational prototype developed
