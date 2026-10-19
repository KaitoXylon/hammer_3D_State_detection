#!/usr/bin/env python3
"""
Stable Real-Time Hammer Pose & Claw Orientation Tracker
-------------------------------------------------------
Features:
- Sub-pixel tracking using YOLOv8-pose
- Exponential Moving Average (EMA) temporal smoothing for rock-solid keypoints
- 3-frame persistence filter (eliminates single-frame flickers / dropouts)
- Outputs Grip Target (X, Y) and Claw Yaw Angle (θ)
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


class StableHammerTracker:
    def __init__(self, weights_path="runs/pose/runs/hammer_pose_best-2/weights/best.pt", conf=0.35):
        print(f"[INFO] Loading YOLOv8-pose model: {weights_path}")
        self.model = YOLO(weights_path)
        self.conf = conf
        self.mode = None
        self.zed = None
        self.cap = None

        # Temporal smoothing state (EMA)
        self.smooth_head = None
        self.smooth_tail = None
        self.alpha = 0.65  # 0.65 current + 0.35 history for responsiveness + stability
        self.lost_counter = 0
        self.max_lost_frames = 4

        self._init_camera()

    def _init_camera(self):
        # 1. ZED SDK
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

        # 2. Standard OpenCV
        print("[INFO] Scanning standard camera devices...")
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

    def run(self):
        window_name = "URC Stable Autonomous Hammer Tracker"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1280, 720)

        prev_time = time.time()
        fps = 0.0

        print("\n" + "=" * 60)
        print("  STABLE HAMMER TRACKER RUNNING")
        print("  Controls:")
        print("    [Q] / ESC : Exit")
        print("=" * 60 + "\n")

        while True:
            frame = self.get_frame()
            if frame is None:
                continue

            curr_time = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / max(1e-4, curr_time - prev_time))
            prev_time = curr_time

            h, w = frame.shape[:2]

            # Fast inference
            results = self.model(frame, conf=self.conf, device=0, verbose=False)[0]

            detected_this_frame = False
            hud = frame.copy()

            if len(results.keypoints) > 0 and len(results.boxes) > 0:
                box = results.boxes.xyxy[0].cpu().numpy()
                conf_val = float(results.boxes.conf[0])
                kpts = results.keypoints.xy[0].cpu().numpy()

                raw_head = kpts[0]
                raw_tail = kpts[1]

                # Temporal Exponential Moving Average (EMA) to prevent jitter
                if self.smooth_head is None:
                    self.smooth_head = raw_head
                    self.smooth_tail = raw_tail
                else:
                    self.smooth_head = self.alpha * raw_head + (1.0 - self.alpha) * self.smooth_head
                    self.smooth_tail = self.alpha * raw_tail + (1.0 - self.alpha) * self.smooth_tail

                self.lost_counter = 0
                detected_this_frame = True

            else:
                self.lost_counter += 1
                if self.lost_counter > self.max_lost_frames:
                    self.smooth_head = None
                    self.smooth_tail = None

            # Render tracked target if available (even through 1-3 frame dropouts)
            if self.smooth_head is not None and self.smooth_tail is not None:
                hx, hy = int(self.smooth_head[0]), int(self.smooth_head[1])
                tx, ty = int(self.smooth_tail[0]), int(self.smooth_tail[1])

                dx = hx - tx
                dy = hy - ty
                angle_deg = math.degrees(math.atan2(dy, dx)) % 360
                claw_yaw = angle_deg % 180

                # Grip target (40% up from handle tail)
                gx = int(tx + 0.4 * dx)
                gy = int(ty + 0.4 * dy)

                # Centerline
                cv2.line(hud, (tx, ty), (hx, hy), (0, 255, 255), 3)

                # Head: White ring + Red center
                cv2.circle(hud, (hx, hy), 12, (255, 255, 255), 2)
                cv2.circle(hud, (hx, hy), 8, (0, 0, 255), -1)
                cv2.putText(hud, "HEAD", (hx + 15, hy + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)

                # Tail: White ring + Blue center
                cv2.circle(hud, (tx, ty), 12, (255, 255, 255), 2)
                cv2.circle(hud, (tx, ty), 8, (255, 0, 0), -1)
                cv2.putText(hud, "TAIL", (tx + 15, ty + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 0, 0), 2)

                # Grip: Yellow circle
                cv2.circle(hud, (gx, gy), 14, (255, 255, 255), 2)
                cv2.circle(hud, (gx, gy), 9, (0, 255, 255), -1)
                cv2.putText(hud, f"GRIP ({gx}, {gy})", (gx + 16, gy + 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)

                # HUD Info Card
                status_color = (0, 255, 0) if detected_this_frame else (0, 165, 255)
                status_text = "LOCKED" if detected_this_frame else "TRACKING (COASTING)"

                cv2.rectangle(hud, (15, 15), (460, 130), (20, 20, 20), -1)
                cv2.rectangle(hud, (15, 15), (460, 130), status_color, 2)
                cv2.putText(hud, f"STATUS: {status_text}", (30, 45),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
                cv2.putText(hud, f"Target: X={gx}px, Y={gy}px", (30, 75),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
                cv2.putText(hud, f"Claw Yaw Angle: {claw_yaw:.1f} deg", (30, 108),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)
            else:
                cv2.rectangle(hud, (15, 15), (350, 60), (20, 20, 20), -1)
                cv2.putText(hud, "SEARCHING FOR HAMMER...", (30, 45),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 165, 255), 2)

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
    weights = "runs/pose/runs/hammer_pose_best-2/weights/best.pt"
    tracker = StableHammerTracker(weights_path=weights, conf=0.35)
    tracker.run()
