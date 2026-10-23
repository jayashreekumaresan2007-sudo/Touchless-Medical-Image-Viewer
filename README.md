
<div align="center">

# ✳ MEDI VISION

### Touchless Medical Image Viewer

**Imaging, without touching the screen.**

A computer vision project exploring hands-free medical image interaction using Python, OpenCV and MediaPipe.

[**🌐 Live Website**](https://jayashreekumaresan2007-sudo.github.io/Touchless-Medical-Image-Viewer/) · [**💻 Source Code**](https://github.com/jayashreekumaresan2007-sudo/Touchless-Medical-Image-Viewer)

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![MediaPipe](https://img.shields.io/badge/MediaPipe-00A98F?style=for-the-badge&logo=google&logoColor=white)

</div>

---

## ◈ Project Overview

Medi Vision is an educational prototype that lets users interact with a displayed medical image through webcam-based hand gestures. It combines computer vision with a desktop interface to explore a more natural, touchless interaction experience.

## ✦ Features

| Gesture | Action | Description |
|---|---|---|
| 🤏 Pinch | Zoom | Zoom the image using thumb and index finger |
| ☝️ Index finger | Pan | Navigate around the image |
| 🖐️ Open palm | Rotate | Rotate the displayed image |
| ✊ Closed fist | Pause | Pause gesture-based interaction |
| 🖼️ Select Image | Load image | Choose a supported image from your computer |
| ↺ Reset View | Reset | Restore the original view |

## ◈ Technology Stack

- **Python** — application logic
- **OpenCV** — image processing and webcam frames
- **MediaPipe** — hand landmark detection
- **NumPy** — numerical operations
- **CustomTkinter** — desktop user interface
- **Pillow** — image handling

## ⌘ Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/jayashreekumaresan2007-sudo/Touchless-Medical-Image-Viewer.git
cd Touchless-Medical-Image-Viewer
```

### 2. Create and activate a virtual environment

**Windows PowerShell:**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Launch the application

```bash
python ui.py
```

Allow camera access when prompted. The hand landmark model file `hand_landmarker.task` must be present in the project directory.

## ▦ Project Structure

```text
Touchless-Medical-Image-Viewer/
├── images/
│   └── sample_xray.png
├── hand_landmarker.task
├── index.html
├── main.py
├── ui.py
├── requirements.txt
└── README.md
```

## ◈ Links

- **Live Portfolio:** [Open Medi Vision](https://jayashreekumaresan2007-sudo.github.io/Touchless-Medical-Image-Viewer/)
- **GitHub Repository:** [View Source Code](https://github.com/jayashreekumaresan2007-sudo/Touchless-Medical-Image-Viewer)

## ⚠️ Disclaimer

This project is intended for educational and demonstration purposes only. It is not a validated medical device and must not be used for clinical diagnosis or medical decision-making.

---

<div align="center">

**Medi Vision · Exploring Human–Computer Interaction**

</div>
