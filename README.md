# 💙 Medi Vision — Touchless Medical Image Viewer

A touchless medical image viewing prototype that uses hand gestures to interact with medical images, providing an alternative to traditional mouse-based controls.

## ✨ Features

- 🤏 **Zoom:** Use a thumb-and-index-finger pinch gesture.
- ✋ **Pan:** Move the index finger to navigate the image.
- 🔄 **Rotate:** Use an open-palm gesture.
- ✊ **Pause:** Use a closed-fist gesture.
- 🖼️ **Image Selection:** Load an image from your computer.
- ♻️ **Reset View:** Return the image to its default view.

## 🛠️ Technologies Used

- Python
- OpenCV
- MediaPipe
- NumPy
- CustomTkinter
- Pillow

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/jayashreekumaresan2007-sudo/Touchless-Medical-Image-Viewer.git
cd Touchless-Medical-Image-Viewer
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the application

```bash
python main.py
```

Allow webcam access when prompted.

## 🌐 Project Website

[Visit Medi Vision](https://jayashreekumaresan2007-sudo.github.io/Touchless-Medical-Image-Viewer/)

## 📁 Project Structure

```text
Touchless-Medical-Image-Viewer/
├── images/
│   └── sample_xray.png
├── hand_landmarker.task
├── index.html
├── main.py
├── requirements.txt
├── README.md
├── sample_xray.png
└── ui.py
```

## 🎯 Project Goal

To explore gesture-based interaction with medical images using computer vision and create a touchless image-viewing experience.

## ⚠️ Disclaimer

This is an educational prototype, not a certified medical device. It is not intended for diagnosis or clinical decision-making.

## 👩‍💻 Author

Developed as a computer science project to explore Python, computer vision, and human-computer interaction.
