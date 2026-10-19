#!/usr/bin/env python3
"""
Real-Time 5-Keypoint Hammer 3D Pose Tracker (Roll, Pitch, Yaw)
--------------------------------------------------------------
Tracks:
- 5 Keypoints: Face, Claw, Head Center, Grip, Tail
- Yaw (ψ): In-plane heading angle (0° to 360°)
- Pitch (θ): Out-of-plane elevation/tilt relative to floor (0° to 90°)
- Roll (φ): Axial rotation / 3D flip state (0° Normal vs 180° Flipped)
- Claw Yaw: Target angle for parallel gripper (0° to 180°)
- Target Grip (X, Y) for Linear Cartesian Arm
- 3D Coordinate Axis Tripod at the Grip Point
"""

import sys
import time
import math
import cv2
import numpy as np
from ultralytics import YOLO

# Try loading ZED SDK
HAS_PYZED = False
try:
    import pyzed.sl as sl
    HAS_PYZED = True
except ImportError:
    HAS_PYZED = False


class Hammer3DPoseTracker:
    def __init__(self, weights_path="weights/hammer_flip_5pt_best.pt", conf=0.35):
        if not os.path.exists(weights_path) and os.path.exists("runs/pose/runs_flip/hammer_flip_best/weights/best.pt"):
            weights_path = "runs/pose/runs_flip/hammer_flip_best/weights/best.pt"
        print(f"[INFO] Loading 5-Keypoint YOLOv8-pose model: {weights_path}")
        self.model = YOLO(weights_path)
        self.conf = conf
        self.mode = None
        self.zed = None
        self.cap = None

        # Temporal smoothing state (EMA across 5 keypoints)
        self.smooth_kpts = None
        self.alpha = 0.65
        self.lost_counter = 0
        self.max_lost_frames = 4

        # Nominal aspect ratio: Head Crossbar Length / Handle Length (~0.48 for this hammer)
        self.nominal_ratio = 0.48

        self._init_camera()

    def _init_camera(self):
        if HAS_PYZED:
            self.zed = sl.Camera()
            init = sl.InitParameters()
            init.camera_resolution = sl.RESOLUTION.HD720
            init.camera_fps = 30
            init.depth_mode = sl.DEPTH_MODE.NONE
            if self.zed.open(init) == sl.ERROR_CODE.SUCCESS:
                print("[SUCCESS] Connected to ZED 2i via ZED SDK (Left Eye)")
                self.mode = "ZED_SDK"
                self.image_mat = sl.Mat()
                self.runtime = sl.RuntimeParameters()
                return

        print("[INFO] Scanning standard camera indices...")
        for idx in [0, 1, 2]:
            cap = cv2.VideoCapture(idx)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    print(f"[SUCCESS] Camera opened on device index {idx}")
                    self.cap = cap
                    self.mode = "OPENCV"
                    return
                cap.release()

        print("[ERROR] No camera detected! Check connection.")
        sys.exit(1)

    def get_frame(self):
        if self.mode == "ZED_SDK":
            if self.zed.grab(self.runtime) == sl.ERROR_CODE.SUCCESS:
                self.zed.retrieve_image(self.image_mat, sl.VIEW.LEFT)
                bgra = self.image_mat.get_data()
                return cv2.cvtColor(bgra, cv2.COLOR_BGRA2BGR)
            return None
        elif self.mode == "OPENCV":
            ret, frame = self.cap.read()
            if not ret or frame is None:
                return None
            h, w = frame.shape[:2]
            if w > 2000:
                return frame[:, : w // 2]
            return frame
        return None

    def calculate_rpy(self, face, claw, head_c, tail):
        """
        Calculates 3D Roll, Pitch, and Yaw from 2D keypoints.
        """
        # Handle Vector (Tail -> Head)
        dx_h = float(head_c[0] - tail[0])
        dy_h = float(head_c[1] - tail[1])
        len_h = math.hypot(dx_h, dy_h)

        # Head Crossbar Vector (Claw -> Face)
        dx_head = float(face[0] - claw[0])
        dy_head = float(face[1] - claw[1])
        len_head = math.hypot(dx_head, dy_head)

        # 1. YAW (ψ): In-plane rotation angle (0° to 360°)
        yaw_deg = math.degrees(math.atan2(dy_h, dx_h)) % 360
        claw_yaw = yaw_deg % 180

        # 2. ROLL (φ) & FLIP DIRECTION (LEFT vs. RIGHT)
        # 2D cross product: v_handle x v_head
        # Looking from Tail -> Head:
        # If cross_val >= 0: Claw is on the LEFT side of the handle
        # If cross_val <  0: Claw is on the RIGHT side of the handle
        cross_val = (dx_h * dy_head) - (dy_h * dx_head)
        if cross_val >= 0:
            roll_deg = 0.0
            flip_side = "LEFT"
            flip_text = "FLIP: LEFT  (Claw on Left)"
            status_color = (0, 255, 0)       # Bright Green
        else:
            roll_deg = 180.0
            flip_side = "RIGHT"
            flip_text = "FLIP: RIGHT (Claw on Right)"
            status_color = (0, 140, 255)     # Bright Orange

        # 3. PITCH (θ): Out-of-plane elevation/tilt relative to floor
        # Derived from foreshortening ratio between handle and crossbar
        if len_head > 10:
            expected_handle_len = len_head / self.nominal_ratio
            ratio = min(1.0, max(0.1, len_h / max(1.0, expected_handle_len)))
            pitch_deg = math.degrees(math.acos(ratio))
        else:
            pitch_deg = 0.0

        return yaw_deg, pitch_deg, roll_deg, claw_yaw, flip_side, flip_text, status_color, (dx_h, dy_h), (dx_head, dy_head)

    def draw_3d_axes(self, img, origin, v_handle, v_head, cross_val, axis_len=55):
        """
        Renders a 3D coordinate tripod at the grip point:
        - Green Arrow: Y-Axis (along handle towards head)
        - Red Arrow:   X-Axis (along crossbar towards face)
        - Blue Arrow:  Z-Axis (normal out of ground)
        """
        ox, oy = origin

        # Normalize handle vector (Y-axis)
        mag_h = math.hypot(v_handle[0], v_handle[1])
        if mag_h > 1e-4:
            uy = (v_handle[0] / mag_h, v_handle[1] / mag_h)
        else:
            uy = (0, -1)

        # Normalize head vector (X-axis)
        mag_head = math.hypot(v_head[0], v_head[1])
        if mag_head > 1e-4:
            ux = (v_head[0] / mag_head, v_head[1] / mag_head)
        else:
            ux = (1, 0)

        # Z-axis (apparent normal)
        z_sign = 1.0 if cross_val >= 0 else -1.0
        uz = (-uy[1] * 0.4 * z_sign, uy[0] * 0.4 * z_sign - 0.7)
        mag_z = math.hypot(uz[0], uz[1])
        if mag_z > 1e-4:
            uz = (uz[0] / mag_z, uz[1] / mag_z)

        # Endpoints
        pt_x = (int(ox + ux[0] * axis_len), int(oy + ux[1] * axis_len))
        pt_y = (int(ox + uy[0] * axis_len), int(oy + uy[1] * axis_len))
        pt_z = (int(ox + uz[0] * (axis_len * 0.7)), int(oy + uz[1] * (axis_len * 0.7)))

        # Draw arrows: Red = X, Green = Y, Blue = Z
        cv2.arrowedLine(img, origin, pt_x, (0, 0, 255), 3, tipLength=0.25)  # X (Face)
        cv2.arrowedLine(img, origin, pt_y, (0, 255, 0), 3, tipLength=0.25)  # Y (Handle)
        cv2.arrowedLine(img, origin, pt_z, (255, 100, 0), 3, tipLength=0.25)  # Z (Normal)

        cv2.putText(img, "X", (pt_x[0] + 5, pt_x[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.putText(img, "Y", (pt_y[0] + 5, pt_y[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(img, "Z", (pt_z[0] + 5, pt_z[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 0), 2)

    def run(self):
        window_name = "URC 3D Hammer Pose (Roll, Pitch, Yaw)"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1280, 720)

        prev_time = time.time()
        fps = 0.0

        print("\n" + "=" * 60)
        print("  HAMMER 3D POSE ESTIMATION (ROLL, PITCH, YAW)")
        print("  Outputs:")
        print("    - Roll  (φ) : 3D Flip State (0.0° vs 180.0°)")
        print("    - Pitch (θ) : Tilt relative to ground (0.0° to 90.0°)")
        print("    - Yaw   (ψ) : Heading in image plane (0.0° to 360.0°)")
        print("    - Claw Yaw  : Gripper orientation angle")
        print("  Press 'Q' or ESC to Exit")
        print("=" * 60 + "\n")

        while True:
            frame = self.get_frame()
            if frame is None:
                continue

            curr_time = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / max(1e-4, curr_time - prev_time))
            prev_time = curr_time

            h, w = frame.shape[:2]

            results = self.model(frame, conf=self.conf, device=0, verbose=False)[0]

            detected_this_frame = False
            hud = frame.copy()

            if len(results.keypoints) > 0 and len(results.boxes) > 0:
                kpts = results.keypoints.xy[0].cpu().numpy()

                if self.smooth_kpts is None:
                    self.smooth_kpts = kpts
                else:
                    self.smooth_kpts = self.alpha * kpts + (1.0 - self.alpha) * self.smooth_kpts

                self.lost_counter = 0
                detected_this_frame = True
            else:
                self.lost_counter += 1
                if self.lost_counter > self.max_lost_frames:
                    self.smooth_kpts = None

            if self.smooth_kpts is not None:
                face = tuple(self.smooth_kpts[0].astype(int))
                claw = tuple(self.smooth_kpts[1].astype(int))
                head_c = tuple(self.smooth_kpts[2].astype(int))
                grip = tuple(self.smooth_kpts[3].astype(int))
                tail = tuple(self.smooth_kpts[4].astype(int))

                # Calculate Roll, Pitch, Yaw, and Flip Direction (LEFT vs. RIGHT)
                yaw, pitch, roll, claw_yaw, flip_side, flip_text, status_color, v_h, v_head = self.calculate_rpy(
                    face, claw, head_c, tail
                )

                # Visuals: Handle & Crossbar lines
                cv2.line(hud, tail, head_c, (0, 255, 255), 3)
                cv2.line(hud, claw, face, (0, 255, 255), 3)

                # Directional Arrow pointing to Claw side
                cv2.arrowedLine(hud, head_c, claw, status_color, 3, tipLength=0.25)

                # Keypoint Circles
                # Face (Red)
                cv2.circle(hud, face, 12, (255, 255, 255), 2)
                cv2.circle(hud, face, 8, (0, 0, 255), -1)
                cv2.putText(hud, "FACE", (face[0] + 14, face[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

                # Claw (Blue)
                cv2.circle(hud, claw, 12, (255, 255, 255), 2)
                cv2.circle(hud, claw, 8, (255, 0, 0), -1)
                cv2.putText(hud, f"CLAW ({flip_side})", (claw[0] - 110 if flip_side == 'LEFT' else claw[0] + 15, claw[1] + 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 0), 2)

                # Head Center (Yellow)
                cv2.circle(hud, head_c, 8, (0, 255, 255), -1)

                # Tail (Blue)
                cv2.circle(hud, tail, 12, (255, 255, 255), 2)
                cv2.circle(hud, tail, 8, (255, 0, 0), -1)
                cv2.putText(hud, "TAIL", (tail[0] + 14, tail[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 0), 2)

                # Grip (Yellow)
                cv2.circle(hud, grip, 13, (255, 255, 255), 2)
                cv2.circle(hud, grip, 9, (0, 255, 255), -1)

                # Draw 3D Axis Tripod at the Grip Point
                cross_val = v_h[0] * v_head[1] - v_h[1] * v_head[0]
                self.draw_3d_axes(hud, grip, v_h, v_head, cross_val, axis_len=50)

                # -------------------------------------------------------------
                # 1. BIG PROMINENT TOP-CENTER FLIP BANNER (LIVE LEFT vs. RIGHT)
                # -------------------------------------------------------------
                banner_w = 400
                bx1 = (w - banner_w) // 2
                bx2 = (w + banner_w) // 2
                cv2.rectangle(hud, (bx1, 15), (bx2, 90), (15, 15, 15), -1)
                cv2.rectangle(hud, (bx1, 15), (bx2, 90), status_color, 3)

                flip_title = f"FLIP: {flip_side}"
                detail_str = "CLAW: LEFT  <--  |  --> FACE: RIGHT" if flip_side == "LEFT" else "FACE: LEFT <--  |  --> CLAW: RIGHT"
                cv2.putText(hud, flip_title, (bx1 + 80, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.2, status_color, 3)
                cv2.putText(hud, detail_str, (bx1 + 25, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)

                # -------------------------------------------------------------
                # 2. TOP-LEFT HUD CARD: 3D POSE DETAILS
                # -------------------------------------------------------------
                cv2.rectangle(hud, (15, 15), (460, 205), (20, 20, 20), -1)
                cv2.rectangle(hud, (15, 15), (460, 205), status_color, 2)

                cv2.putText(hud, "3D POSE & ORIENTATION", (30, 42),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                # Flip State
                cv2.putText(hud, f"FLIP ORIENTATION : {flip_side}", (30, 72),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

                # Roll
                cv2.putText(hud, f"Roll (φ)         : {roll:5.1f} deg", (30, 102),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 1)

                # Pitch (Elevation Tilt)
                pitch_label = "Flat on floor" if pitch < 12.0 else "Propped / Tilted"
                cv2.putText(hud, f"Pitch (Tilt)     : {pitch:5.1f} deg [{pitch_label}]", (30, 130),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 200, 0), 1)

                # Yaw (Heading in camera plane)
                cv2.putText(hud, f"Yaw (Heading)    : {yaw:5.1f} deg | Claw: {claw_yaw:.1f} deg", (30, 158),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

                # Grip Target
                cv2.putText(hud, f"Target (Grip)    : X={grip[0]}px, Y={grip[1]}px", (30, 185),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1)

                status_str = "LOCKED" if detected_this_frame else "COASTING"
                cv2.putText(hud, f"Tracker: {status_str}", (30, 185),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 160), 1)
            else:
                cv2.rectangle(hud, (15, 15), (350, 60), (20, 20, 20), -1)
                cv2.putText(hud, "SEARCHING FOR HAMMER...", (30, 45),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 165, 255), 2)

            # Bottom FPS Bar
            cv2.putText(hud, f"FPS: {fps:.1f} | Sensor: {self.mode}", (w - 280, h - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 2)

            cv2.imshow(window_name, hud)
            key = cv2.waitKey(1) & 0xFF
            if key in [ord("q"), ord("Q"), 27]:
                break

        if self.mode == "ZED_SDK" and self.zed:
            self.zed.close()
        elif self.cap:
            self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    weights = "weights/hammer_flip_5pt_best.pt"
    tracker = Hammer3DPoseTracker(weights_path=weights, conf=0.35)
    tracker.run()
