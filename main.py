
import cv2
import mediapipe as mp
import math
import numpy as np
import time

# -----------------------------
# 1. Load X-ray image
# -----------------------------
image = cv2.imread("images/sample_xray.png")

if image is None:
    print("Error: X-ray image not found!")
    raise SystemExit

image = cv2.resize(image, (640, 480))
original_image = image.copy()

# -----------------------------
# 2. Start webcam
# -----------------------------
webcam = cv2.VideoCapture(0)

if not webcam.isOpened():
    print("Error: Could not open webcam!")
    raise SystemExit

# -----------------------------
# 3. MediaPipe setup
# -----------------------------
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path="hand_landmarker.task"
    ),
    running_mode=RunningMode.VIDEO,
    num_hands=1
)

# -----------------------------
# 4. Viewer settings
# -----------------------------
zoom = 1.0
pan_x = 0
pan_y = 0

rotation_angle = 0.0
rotation_target = 0.0

ZOOM_SMOOTHING = 0.20
ROTATION_SMOOTHING = 0.06
PAN_SMOOTHING = 0.25

last_index_x = None
last_index_y = None
last_wrist_x = None
last_time_ms = 0

current_gesture = "NONE"
GESTURE_STABLE_FRAMES = 4
gesture_candidate = "NONE"
gesture_frame_count = 0

ROTATION_DEAD_ZONE = 0.008

# -----------------------------
# 5. Helper functions
# -----------------------------
def distance(p1, p2):
    return math.hypot(p1.x - p2.x, p1.y - p2.y)


def rotate_image(img, angle):
    height, width = img.shape[:2]
    center = (width // 2, height // 2)

    matrix = cv2.getRotationMatrix2D(
        center, angle, 1.0
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
    # For fingers other than the thumb, compare fingertip
    # and middle-joint vertical positions.
    return hand[tip_id].y < hand[pip_id].y


def detect_gesture(hand):
    thumb_tip = hand[4]
    index_tip = hand[8]
    middle_tip = hand[12]
    ring_tip = hand[16]
    pinky_tip = hand[20]

    # Pinch: thumb and index finger are close together.
    pinch_distance = distance(thumb_tip, index_tip)

    if pinch_distance < 0.075:
        return "ZOOM"

    index_up = finger_is_up(hand, 8, 6)
    middle_up = finger_is_up(hand, 12, 10)
    ring_up = finger_is_up(hand, 16, 14)
    pinky_up = finger_is_up(hand, 20, 18)

    # Fist: all four fingertips are folded.
    if not any([index_up, middle_up, ring_up, pinky_up]):
        return "PAUSE"

    # Open palm: all four fingers are raised.
    if all([index_up, middle_up, ring_up, pinky_up]):
        return "ROTATE"

    # Index finger only: pan the image.
    if index_up and not any([
        middle_up, ring_up, pinky_up
    ]):
        return "PAN"

    return "NONE"


# -----------------------------
# 6. Main program
# -----------------------------
try:
    with HandLandmarker.create_from_options(options) as landmarker:

        while True:
            success, frame = webcam.read()

            if not success:
                print("Could not read webcam frame.")
                break

            frame = cv2.flip(frame, 1)

            rgb_frame = cv2.cvtColor(
                frame, cv2.COLOR_BGR2RGB
            )

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_frame
            )

            timestamp_ms = int(time.monotonic() * 1000)
            timestamp_ms = max(timestamp_ms, last_time_ms + 1)
            last_time_ms = timestamp_ms

            result = landmarker.detect_for_video(
                mp_image, timestamp_ms
            )

            detected_gesture = "NONE"
            hand = None

            if result.hand_landmarks:
                hand = result.hand_landmarks[0]

                for point in hand:
                    px = int(point.x * frame.shape[1])
                    py = int(point.y * frame.shape[0])

                    cv2.circle(
                        frame, (px, py), 4,
                        (0, 255, 0), -1
                    )

                detected_gesture = detect_gesture(hand)

            # -----------------------------
            # 7. Stabilize gesture changes
            # -----------------------------
            if detected_gesture == gesture_candidate:
                gesture_frame_count += 1
            else:
                gesture_candidate = detected_gesture
                gesture_frame_count = 1

            if gesture_frame_count >= GESTURE_STABLE_FRAMES:
                if current_gesture != gesture_candidate:
                    current_gesture = gesture_candidate

                    # Avoid jumps when switching gesture modes.
                    last_index_x = None
                    last_index_y = None
                    last_wrist_x = None

            # -----------------------------
            # 8. Perform selected gesture
            # -----------------------------
            if hand is not None:

                if current_gesture == "ZOOM":
                    # Pinch controls zoom.
                    pinch_distance = distance(
                        hand[4], hand[8]
                    )

                    target_zoom = np.interp(
                        pinch_distance,
                        [0.025, 0.22],
                        [0.7, 2.5]
                    )

                    target_zoom = float(
                        np.clip(target_zoom, 0.7, 2.5)
                    )

                    zoom += ZOOM_SMOOTHING * (
                        target_zoom - zoom
                    )

                elif current_gesture == "PAN":
                    # Index-finger movement controls pan.
                    index_tip = hand[8]

                    current_index_x = index_tip.x
                    current_index_y = index_tip.y

                    if last_index_x is not None:
                        dx = current_index_x - last_index_x
                        dy = current_index_y - last_index_y

                        pan_x -= int(
                            dx * 500 * PAN_SMOOTHING
                        )
                        pan_y += int(
                            dy * 500 * PAN_SMOOTHING
                        )

                    last_index_x = current_index_x
                    last_index_y = current_index_y

                elif current_gesture == "ROTATE":
                    # Open-palm wrist movement controls rotation.
                    wrist_x = hand[0].x

                    if last_wrist_x is not None:
                        wrist_delta = wrist_x - last_wrist_x

                        if abs(wrist_delta) > ROTATION_DEAD_ZONE:
                            rotation_target -= wrist_delta * 180

                            rotation_target = (
                                rotation_target + 180
                            ) % 360 - 180

                    last_wrist_x = wrist_x

                # PAUSE and NONE deliberately perform no movement.

            else:
                current_gesture = "NONE"
                gesture_candidate = "NONE"
                gesture_frame_count = 0

                last_index_x = None
                last_index_y = None
                last_wrist_x = None

            # -----------------------------
            # 9. Smooth rotation
            # -----------------------------
            angle_difference = (
                rotation_target - rotation_angle + 180
            ) % 360 - 180

            rotation_angle += (
                ROTATION_SMOOTHING * angle_difference
            )

            # -----------------------------
            # 10. Prepare viewer image
            # -----------------------------
            display_image = rotate_image(
                original_image, rotation_angle
            )

            display_image = cv2.resize(
                display_image,
                None,
                fx=zoom,
                fy=zoom,
                interpolation=cv2.INTER_LINEAR
            )

            canvas = np.zeros(
                (600, 800, 3), dtype=np.uint8
            )

            img_h, img_w = display_image.shape[:2]

            center_x = 400 + pan_x
            center_y = 300 + pan_y

            x1 = center_x - img_w // 2
            y1 = center_y - img_h // 2
            x2 = x1 + img_w
            y2 = y1 + img_h

            dst_x1 = max(0, x1)
            dst_y1 = max(0, y1)
            dst_x2 = min(800, x2)
            dst_y2 = min(600, y2)

            if dst_x1 < dst_x2 and dst_y1 < dst_y2:
                src_x1 = dst_x1 - x1
                src_y1 = dst_y1 - y1
                src_x2 = src_x1 + (dst_x2 - dst_x1)
                src_y2 = src_y1 + (dst_y2 - dst_y1)

                canvas[dst_y1:dst_y2, dst_x1:dst_x2] = (
                    display_image[
                        src_y1:src_y2,
                        src_x1:src_x2
                    ]
                )

            # -----------------------------
            # 11. Show gesture status
            # -----------------------------
            cv2.putText(
                canvas,
                f"Gesture: {current_gesture}",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 255, 0), 2
            )

            cv2.putText(
                canvas,
                f"Zoom: {zoom:.2f}x",
                (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 255, 0), 2
            )

            cv2.putText(
                canvas,
                f"Rotation: {rotation_angle:.1f} deg",
                (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 255, 0), 2
            )

            cv2.putText(
                canvas,
                "Pinch: Zoom | Index only: Pan",
                (20, 530),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (255, 255, 255), 1
            )

            cv2.putText(
                canvas,
                "Open palm: Rotate | Fist: Pause",
                (20, 555),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (255, 255, 255), 1
            )

            cv2.putText(
                canvas,
                "R: Reset | Q: Quit",
                (20, 585),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 255, 255), 1
            )

            cv2.imshow("Hand Gesture", frame)
            cv2.imshow("Touchless Medical Image Viewer", canvas)

            # -----------------------------
            # 12. Keyboard controls
            # -----------------------------
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            elif key == ord("r"):
                zoom = 1.0
                pan_x = 0
                pan_y = 0

                rotation_angle = 0.0
                rotation_target = 0.0

                last_index_x = None
                last_index_y = None
                last_wrist_x = None

                current_gesture = "NONE"
                gesture_candidate = "NONE"
                gesture_frame_count = 0

                print("View reset!")

finally:
    webcam.release()
    cv2.destroyAllWindows()
