
import os
import cv2
import math
import time
import numpy as np
import mediapipe as mp
import customtkinter as ctk

from PIL import Image
from tkinter import filedialog, messagebox

# =========================================
# 1. PROJECT PATHS
# =========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

XRAY_PATH = os.path.join(
    BASE_DIR, "images", "sample_xray.png"
)

MODEL_PATH = os.path.join(
    BASE_DIR, "hand_landmarker.task"
)

# =========================================
# 2. MEDIAPIPE SETUP
# =========================================
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

# =========================================
# 3. APPLICATION SETTINGS
# =========================================
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

app = ctk.CTk()
app.title("Medi Vision | Touchless Medical Image Viewer")
app.geometry("1200x800")
app.minsize(950, 680)

app.grid_columnconfigure(1, weight=1)
app.grid_rowconfigure(0, weight=1)

# =========================================
# 4. VIEWER STATE
# =========================================
zoom = 1.0
pan_x = 0.0
pan_y = 0.0

rotation_angle = 0.0
rotation_target = 0.0

ZOOM_SMOOTHING = 0.20
ROTATION_SMOOTHING = 0.06
PAN_SMOOTHING = 0.25
ROTATION_DEAD_ZONE = 0.008

last_index_x = None
last_index_y = None
last_wrist_x = None

last_time_ms = 0

current_gesture = "NONE"
gesture_candidate = "NONE"
gesture_frame_count = 0
GESTURE_STABLE_FRAMES = 4

camera = None
landmarker = None
camera_running = False
camera_job = None

original_image = None
current_image_path = None
viewer_photo = None
camera_photo = None

# =========================================
# 5. IMAGE AND GESTURE HELPERS
# =========================================
def point_distance(p1, p2):
    return math.hypot(
        p1.x - p2.x,
        p1.y - p2.y
    )


def rotate_image(img, angle):
    height, width = img.shape[:2]

    matrix = cv2.getRotationMatrix2D(
        (width // 2, height // 2),
        angle,
        1.0
    )

    return cv2.warpAffine(
        img,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0)
    )


def finger_is_up(hand, tip_id, pip_id):
    return hand[tip_id].y < hand[pip_id].y


def detect_gesture(hand):
    # Pinch: thumb and index finger close together
    if point_distance(hand[4], hand[8]) < 0.075:
        return "ZOOM"

    index_up = finger_is_up(hand, 8, 6)
    middle_up = finger_is_up(hand, 12, 10)
    ring_up = finger_is_up(hand, 16, 14)
    pinky_up = finger_is_up(hand, 20, 18)

    # Closed fist
    if not any([
        index_up, middle_up, ring_up, pinky_up
    ]):
        return "PAUSE"

    # Open palm
    if all([
        index_up, middle_up, ring_up, pinky_up
    ]):
        return "ROTATE"

    # Index finger only
    if index_up and not any([
        middle_up, ring_up, pinky_up
    ]):
        return "PAN"

    return "NONE"


# =========================================
# 6. LOAD INITIAL IMAGE
# =========================================
def load_image_from_path(path):
    global original_image, current_image_path

    loaded = cv2.imread(path)

    if loaded is None:
        messagebox.showerror(
            "Image Error",
            "Could not open this image. Please choose a valid image file."
        )
        return False

    original_image = loaded
    current_image_path = path

    reset_view()
    render_view()

    viewer_status.configure(
        text="Image Loaded",
        text_color="#34D399"
    )

    image_name_label.configure(
        text=os.path.basename(path)
    )

    print("Loaded image:", path)
    return True


# =========================================
# 7. RESET AND RENDER IMAGE
# =========================================
def reset_view():
    global zoom, pan_x, pan_y
    global rotation_angle, rotation_target
    global last_index_x, last_index_y, last_wrist_x

    zoom = 1.0
    pan_x = 0.0
    pan_y = 0.0

    rotation_angle = 0.0
    rotation_target = 0.0

    last_index_x = None
    last_index_y = None
    last_wrist_x = None

    render_view()


def render_view():
    global viewer_photo

    if original_image is None:
        return

    # Rotate first, then zoom
    rotated = rotate_image(
        original_image,
        rotation_angle
    )

    height, width = rotated.shape[:2]

    new_width = max(1, int(width * zoom))
    new_height = max(1, int(height * zoom))

    display = cv2.resize(
        rotated,
        (new_width, new_height),
        interpolation=cv2.INTER_LINEAR
    )

    # Black viewer canvas
    canvas = np.zeros(
        (480, 640, 3),
        dtype=np.uint8
    )

    center_x = int(320 + pan_x)
    center_y = int(240 + pan_y)

    x1 = center_x - new_width // 2
    y1 = center_y - new_height // 2
    x2 = x1 + new_width
    y2 = y1 + new_height

    # Clip to canvas
    dx1 = max(0, x1)
    dy1 = max(0, y1)
    dx2 = min(640, x2)
    dy2 = min(480, y2)

    if dx1 < dx2 and dy1 < dy2:
        sx1 = dx1 - x1
        sy1 = dy1 - y1
        sx2 = sx1 + (dx2 - dx1)
        sy2 = sy1 + (dy2 - dy1)

        canvas[dy1:dy2, dx1:dx2] = display[
            sy1:sy2, sx1:sx2
        ]

    canvas_rgb = cv2.cvtColor(
        canvas, cv2.COLOR_BGR2RGB
    )

    pil_image = Image.fromarray(canvas_rgb)

    viewer_photo = ctk.CTkImage(
        light_image=pil_image,
        dark_image=pil_image,
        size=(640, 480)
    )

    image_label.configure(
        image=viewer_photo,
        text=""
    )

    image_label.image = viewer_photo

    zoom_label.configure(
        text=f"Zoom: {zoom:.2f}x"
    )

    rotation_label.configure(
        text=f"Rotation: {rotation_angle:.1f}°"
    )


# =========================================
# 8. SELECT A DIFFERENT IMAGE
# =========================================
def choose_medical_image():
    file_path = filedialog.askopenfilename(
        parent=app,
        title="Select a Medical Image",
        filetypes=[
            (
                "Medical image files",
                "*.png *.jpg *.jpeg *.bmp"
            ),
            ("All files", "*.*")
        ]
    )

    if file_path:
        load_image_from_path(file_path)


# =========================================
# 9. BUILD DASHBOARD
# =========================================
sidebar = ctk.CTkFrame(
    app,
    width=220,
    corner_radius=0,
    fg_color="#111827"
)

sidebar.grid(row=0, column=0, sticky="nsew")
sidebar.grid_propagate(False)

ctk.CTkLabel(
    sidebar,
    text="MEDI VISION",
    font=("Segoe UI", 24, "bold"),
    text_color="#60A5FA"
).pack(padx=18, pady=(28, 5), anchor="w")

ctk.CTkLabel(
    sidebar,
    text="Touchless Imaging System",
    font=("Segoe UI", 12),
    text_color="#9CA3AF"
).pack(padx=18, pady=(0, 28), anchor="w")

ctk.CTkLabel(
    sidebar,
    text="WORKSPACE",
    font=("Segoe UI", 11, "bold"),
    text_color="#6B7280"
).pack(padx=18, pady=(0, 8), anchor="w")

for item in [
    "Medical Image Viewer",
    "Gesture Controls",
    "About Project"
]:
    ctk.CTkLabel(
        sidebar,
        text=item,
        anchor="w",
        font=("Segoe UI", 13),
        text_color="#E5E7EB"
    ).pack(fill="x", padx=15, pady=10)

ctk.CTkLabel(
    sidebar,
    text=(
        "Gestures\n"
        "Pinch: Zoom\n"
        "Index: Pan\n"
        "Open palm: Rotate\n"
        "Fist: Pause\n\n"
        "Educational prototype\n"
        "Not for clinical diagnosis"
    ),
    font=("Segoe UI", 11),
    text_color="#9CA3AF",
    justify="left"
).pack(side="bottom", padx=18, pady=24, anchor="w")

content = ctk.CTkFrame(
    app,
    corner_radius=0,
    fg_color="#0B1120"
)

content.grid(row=0, column=1, sticky="nsew")
content.grid_columnconfigure(0, weight=1)
content.grid_rowconfigure(2, weight=1)

header = ctk.CTkFrame(
    content,
    fg_color="transparent"
)

header.grid(
    row=0, column=0,
    sticky="ew",
    padx=24, pady=(20, 10)
)

ctk.CTkLabel(
    header,
    text="Medical Imaging Dashboard",
    font=("Segoe UI", 25, "bold"),
    text_color="#F9FAFB"
).pack(anchor="w")

ctk.CTkLabel(
    header,
    text="TOUCHLESS IMAGE VIEWING",
    font=("Segoe UI", 11, "bold"),
    text_color="#34D399"
).pack(anchor="w", pady=(4, 0))

# Status cards
stats = ctk.CTkFrame(
    content,
    fg_color="transparent"
)

stats.grid(
    row=1, column=0,
    sticky="ew", padx=18, pady=8
)

for col in range(3):
    stats.grid_columnconfigure(col, weight=1)


def create_status_card(parent, col, heading, value, color):
    card = ctk.CTkFrame(
        parent,
        corner_radius=12,
        fg_color="#172033"
    )

    card.grid(
        row=0, column=col,
        sticky="ew", padx=5, pady=4
    )

    ctk.CTkLabel(
        card,
        text=heading,
        font=("Segoe UI", 12),
        text_color="#9CA3AF"
    ).pack(anchor="w", padx=12, pady=(10, 3))

    value_label = ctk.CTkLabel(
        card,
        text=value,
        font=("Segoe UI", 16, "bold"),
        text_color=color
    )

    value_label.pack(
        anchor="w", padx=12, pady=(0, 10)
    )

    return value_label


camera_status = create_status_card(
    stats, 0, "Camera Status",
    "Stopped", "#FBBF24"
)

gesture_status = create_status_card(
    stats, 1, "Current Gesture",
    "Waiting", "#60A5FA"
)

viewer_status = create_status_card(
    stats, 2, "Viewer Status",
    "Loading...", "#34D399"
)

# Main image viewer
viewer = ctk.CTkFrame(
    content,
    corner_radius=16,
    fg_color="#172033"
)

viewer.grid(
    row=2, column=0,
    sticky="nsew", padx=24, pady=10
)

viewer.grid_columnconfigure(0, weight=1)
viewer.grid_rowconfigure(1, weight=1)

ctk.CTkLabel(
    viewer,
    text="X-RAY IMAGE VIEWER",
    font=("Segoe UI", 14, "bold"),
    text_color="#D1D5DB"
).grid(
    row=0, column=0,
    sticky="w", padx=16, pady=(12, 4)
)

image_name_label = ctk.CTkLabel(
    viewer,
    text="No image selected",
    font=("Segoe UI", 11),
    text_color="#9CA3AF"
)

image_name_label.grid(
    row=0, column=0,
    sticky="e", padx=16, pady=(12, 4)
)

image_area = ctk.CTkFrame(
    viewer,
    fg_color="#080D18",
    corner_radius=12
)

image_area.grid(
    row=1, column=0,
    sticky="nsew", padx=12, pady=8
)

image_area.grid_columnconfigure(0, weight=1)
image_area.grid_rowconfigure(0, weight=1)

image_label = ctk.CTkLabel(
    image_area,
    text="Loading image..."
)

image_label.grid(
    row=0, column=0,
    padx=8, pady=8
)

# Live camera preview
camera_area = ctk.CTkFrame(
    content,
    fg_color="#172033",
    corner_radius=12
)

camera_area.grid(
    row=3, column=0,
    sticky="ew", padx=24, pady=(0, 8)
)

camera_area.grid_columnconfigure(0, weight=1)

ctk.CTkLabel(
    camera_area,
    text="LIVE HAND CAMERA",
    font=("Segoe UI", 12, "bold"),
    text_color="#D1D5DB"
).grid(
    row=0, column=0,
    sticky="w", padx=12, pady=(8, 0)
)

camera_label = ctk.CTkLabel(
    camera_area,
    text="Camera is stopped",
    width=240,
    height=100,
    text_color="#9CA3AF"
)

camera_label.grid(
    row=1, column=0,
    padx=10, pady=8
)

# Footer controls
footer = ctk.CTkFrame(
    content,
    fg_color="transparent"
)

footer.grid(
    row=4, column=0,
    sticky="ew", padx=24, pady=(2, 12)
)

zoom_label = ctk.CTkLabel(
    footer,
    text="Zoom: 1.00x",
    font=("Segoe UI", 12),
    text_color="#9CA3AF"
)

zoom_label.pack(side="left", padx=(0, 12))

rotation_label = ctk.CTkLabel(
    footer,
    text="Rotation: 0.0°",
    font=("Segoe UI", 12),
    text_color="#9CA3AF"
)

rotation_label.pack(side="left")

# =========================================
# 10. CAMERA CONTROL
# =========================================
def start_camera():
    global camera, landmarker, camera_running
    global last_time_ms

    if camera_running:
        return

    if not os.path.exists(MODEL_PATH):
        messagebox.showerror(
            "Model Missing",
            f"Could not find:\n{MODEL_PATH}"
        )
        return

    try:
        camera = cv2.VideoCapture(0)

        if not camera.isOpened():
            camera.release()
            camera = None

            camera_status.configure(
                text="Unavailable",
                text_color="#F87171"
            )
            return

        options = HandLandmarkerOptions(
            base_options=BaseOptions(
                model_asset_path=MODEL_PATH
            ),
            running_mode=RunningMode.VIDEO,
            num_hands=1
        )

        landmarker = HandLandmarker.create_from_options(
            options
        )

        last_time_ms = 0
        camera_running = True

        camera_status.configure(
            text="Connected",
            text_color="#34D399"
        )

        camera_button.configure(
            text="Stop Camera",
            command=stop_camera
        )

        update_camera()

    except Exception as error:
        print("Camera start error:", error)

        camera_status.configure(
            text="Error",
            text_color="#F87171"
        )

        if camera is not None:
            camera.release()
            camera = None


def stop_camera():
    global camera, landmarker, camera_running
    global camera_job, current_gesture
    global last_index_x, last_index_y, last_wrist_x

    camera_running = False

    if camera_job is not None:
        try:
            app.after_cancel(camera_job)
        except Exception:
            pass

        camera_job = None

    if camera is not None:
        camera.release()
        camera = None

    if landmarker is not None:
        landmarker.close()
        landmarker = None

    current_gesture = "NONE"

    last_index_x = None
    last_index_y = None
    last_wrist_x = None

    camera_status.configure(
        text="Stopped",
        text_color="#FBBF24"
    )

    gesture_status.configure(
        text="Waiting",
        text_color="#60A5FA"
    )

    camera_label.configure(
        image=None,
        text="Camera is stopped"
    )

    camera_label.image = None
    camera_button.configure(
        text="Start Camera",
        command=start_camera
    )


# =========================================
# 11. LIVE CAMERA + GESTURE LOOP
# =========================================
def update_camera():
    global camera_job, last_time_ms
    global zoom, pan_x, pan_y
    global rotation_angle, rotation_target
    global last_index_x, last_index_y, last_wrist_x
    global current_gesture, gesture_candidate
    global gesture_frame_count, camera_photo

    if (
        not camera_running
        or camera is None
        or landmarker is None
    ):
        return

    success, frame = camera.read()

    if not success:
        camera_job = app.after(50, update_camera)
        return

    frame = cv2.flip(frame, 1)

    rgb_frame = cv2.cvtColor(
        frame, cv2.COLOR_BGR2RGB
    )

    timestamp_ms = max(
        int(time.monotonic() * 1000),
        last_time_ms + 1
    )

    last_time_ms = timestamp_ms

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    result = landmarker.detect_for_video(
        mp_image, timestamp_ms
    )

    hand = (
        result.hand_landmarks[0]
        if result.hand_landmarks
        else None
    )

    detected_gesture = (
        detect_gesture(hand)
        if hand is not None
        else "NONE"
    )

    # Stabilize gesture recognition
    if detected_gesture == gesture_candidate:
        gesture_frame_count += 1
    else:
        gesture_candidate = detected_gesture
        gesture_frame_count = 1

    if gesture_frame_count >= GESTURE_STABLE_FRAMES:
        if current_gesture != gesture_candidate:
            current_gesture = gesture_candidate

            last_index_x = None
            last_index_y = None
            last_wrist_x = None

    if hand is not None:
        for point in hand:
            px = int(point.x * frame.shape[1])
            py = int(point.y * frame.shape[0])

            cv2.circle(
                frame, (px, py), 4,
                (0, 255, 0), -1
            )

        if current_gesture == "ZOOM":
            finger_distance = point_distance(
                hand[4], hand[8]
            )

            target_zoom = float(
                np.clip(
                    np.interp(
                        finger_distance,
                        [0.025, 0.22],
                        [0.7, 2.5]
                    ),
                    0.7, 2.5
                )
            )

            zoom += ZOOM_SMOOTHING * (
                target_zoom - zoom
            )

        elif current_gesture == "PAN":
            index_x = hand[8].x
            index_y = hand[8].y

            if last_index_x is not None:
                dx = index_x - last_index_x
                dy = index_y - last_index_y

                pan_x -= dx * 500 * PAN_SMOOTHING
                pan_y += dy * 500 * PAN_SMOOTHING

            last_index_x = index_x
            last_index_y = index_y

        elif current_gesture == "ROTATE":
            wrist_x = hand[0].x

            if last_wrist_x is not None:
                delta = wrist_x - last_wrist_x

                if abs(delta) > ROTATION_DEAD_ZONE:
                    rotation_target -= delta * 180

                    rotation_target = (
                        (rotation_target + 180) % 360
                    ) - 180

            last_wrist_x = wrist_x

        # PAUSE and NONE intentionally do not move the image.

    else:
        current_gesture = "NONE"
        gesture_candidate = "NONE"
        gesture_frame_count = 0

        last_index_x = None
        last_index_y = None
        last_wrist_x = None

    # Smooth rotation
    angle_difference = (
        rotation_target - rotation_angle + 180
    ) % 360 - 180

    rotation_angle += (
        ROTATION_SMOOTHING * angle_difference
    )

    # Update X-ray view
    render_view()

    # Update camera preview
    preview = cv2.resize(frame, (240, 135))

    preview_rgb = cv2.cvtColor(
        preview, cv2.COLOR_BGR2RGB
    )

    preview_pil = Image.fromarray(preview_rgb)

    camera_photo = ctk.CTkImage(
        light_image=preview_pil,
        dark_image=preview_pil,
        size=(240, 135)
    )

    camera_label.configure(
        image=camera_photo,
        text=""
    )

    camera_label.image = camera_photo

    gesture_status.configure(
        text=current_gesture,
        text_color=(
            "#34D399"
            if current_gesture not in ["NONE", "PAUSE"]
            else "#60A5FA"
        )
    )

    camera_job = app.after(
        20, update_camera
    )


# =========================================
# 12. BUTTONS AND APP SHUTDOWN
# =========================================
select_image_button = ctk.CTkButton(
    footer,
    text="Select Image",
    command=choose_medical_image,
    width=110
)

select_image_button.pack(
    side="right", padx=(8, 0)
)

camera_button = ctk.CTkButton(
    footer,
    text="Start Camera",
    command=start_camera,
    width=110
)

camera_button.pack(
    side="right", padx=(8, 0)
)

reset_button = ctk.CTkButton(
    footer,
    text="Reset View",
    command=reset_view,
    width=100,
    fg_color="#374151",
    hover_color="#4B5563"
)

reset_button.pack(
    side="right", padx=(8, 0)
)


def close_app():
    stop_camera()
    app.destroy()


app.protocol(
    "WM_DELETE_WINDOW",
    close_app
)

# Load the original sample image
if os.path.exists(XRAY_PATH):
    load_image_from_path(XRAY_PATH)
else:
    viewer_status.configure(
        text="No Image",
        text_color="#F87171"
    )

    image_label.configure(
        text="Click Select Image to load an image."
    )

app.mainloop()
