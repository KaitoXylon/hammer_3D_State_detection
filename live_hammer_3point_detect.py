#!/usr/bin/env python3
"""
================================================================================
3-KEYPOINT LIVE HAMMER TRACKER: [HEAD, GRAB, TAIL]
================================================================================
Autonomous Robotic Tool Detection & Pose Estimation Framework

Live real-time tracker for the 3-point model:
  K1: Head Center (Striking head T-junction)
  K2: Grab Point  (Handle grasp target)
  K3: Tail Base   (Bottom of handle)

Features:
  - EMA Temporal Smoothing (α = 0.65)
  - 4-Frame Persistence Filter (coasts through blur)
  - Live Grip Target (X, Y) in pixels and millimeters
  - Live Claw Yaw Servo Angle: θ_claw = Yaw % 180°
  - ZED 2i SDK support with OpenCV UVC fallback

Usage:
  python3 live_hammer_3point_detect.py
  python3 live_hammer_3point_detect.py --weights runs_3pt/hammer_3pt_best/weights/best.pt
================================================================================
"""

import sys
import time
import math
import argparse
import cv2
import numpy as np
from ultralytics import YOLO

HAS_PYZED = False
try:
    import pyzed.sl as sl
    HAS_PYZED = True
except ImportError:
    HAS_PYZED = False


class Hammer3PointTracker:
    def __init__(self, weights_path="weights/hammer_flip_5pt_best.pt", conf=0.35):
        if not os.path.exists(weights_path):
            if os.path.exists("runs_3pt/hammer_3pt_best/weights/best.pt"):
                weights_path = "runs_3pt/hammer_3pt_best/weights/best.pt"
            elif os.path.exists("weights/hammer_pose_2pt_best.pt"):
                weights_path = "weights/hammer_pose_2pt_best.pt"
        print(f"[INFO] Loading YOLOv8-pose model: {weights_path}")
        self.model = YOLO(weights_path)
        self.conf = conf
        self.mode = None
        self.zed = None
        self.cap = None

        # EMA Temporal Smoothing
        self.smooth_head = None
        self.smooth_grab = None
        self.smooth_tail = None
        self.alpha = 0.65
        self.lost_counter = 0
        self.max_lost_frames = 4

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

        for idx in [0, 2, 4, 1]:
            cap = cv2.VideoCapture(idx)
            if cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                    cap.set(cv2.CAP_PROP_FPS, 30)
                    self.cap = cap
                    self.mode = f"UVC_CAM_{idx}"
                    print(f"[SUCCESS] Connected via OpenCV VideoCapture({idx})")
                    return
                cap.release()

        print("[ERROR] No camera stream found!")
        sys.exit(1)

    def grab_frame(self):
        if self.mode == "ZED_SDK":
            if self.zed.grab(self.runtime) == sl.ERROR_CODE.SUCCESS:
                self.zed.retrieve_image(self.image_mat, sl.VIEW.LEFT)
                return cv2.cvtColor(self.image_mat.get_data(), cv2.COLOR_BGRA2BGR)
            return None
        elif self.cap:
            ret, frame = self.cap.read()
            return frame if ret else None
        return None

    def run(self):
        print("\n" + "=" * 60)
        print("   3-KEYPOINT LIVE TRACKER ACTIVE [HEAD, GRAB, TAIL]")
        print("   Press 'q' or ESC in window to exit")
        print("=" * 60 + "\n")

        fps_timer = time.time()
        frame_count = 0
        fps = 0.0

        while True:
            frame = self.grab_frame()
            if frame is None:
                continue

            frame_count += 1
            if time.time() - fps_timer >= 1.0:
                fps = frame_count / (time.time() - fps_timer)
                fps_timer = time.time()
                frame_count = 0

            # YOLO inference
            results = self.model(frame, conf=self.conf, verbose=False)[0]

            detected = False
            raw_head, raw_grab, raw_tail = None, None, None

            if len(results.boxes) > 0 and results.keypoints is not None:
                kpts_data = results.keypoints.xy.cpu().numpy()
                confs = results.boxes.conf.cpu().numpy()
                best_idx = int(np.argmax(confs))
                k = kpts_data[best_idx]

                if len(k) == 5:
                    # Head, Grab, Tail from 5-keypoint model [Face, Claw, HeadCenter, Grab, Tail]
                    raw_head = k[2]
                    raw_grab = k[3]
                    raw_tail = k[4]
                    detected = True
                elif len(k) == 3 and k[0][0] > 0 and k[2][0] > 0:
                    raw_head = k[0]
                    raw_grab = k[1]
                    raw_tail = k[2]
                    detected = True
                elif len(k) == 2 and k[0][0] > 0 and k[1][0] > 0:
                    raw_head = k[0]
                    raw_tail = k[1]
                    raw_grab = raw_tail + 0.40 * (raw_head - raw_tail)
                    detected = True

            # Temporal EMA & Persistence Coasting
            if detected:
                self.lost_counter = 0
                if self.smooth_head is None:
                    self.smooth_head = raw_head.copy()
                    self.smooth_grab = raw_grab.copy()
                    self.smooth_tail = raw_tail.copy()
                else:
                    self.smooth_head = self.alpha * raw_head + (1.0 - self.alpha) * self.smooth_head
                    self.smooth_grab = self.alpha * raw_grab + (1.0 - self.alpha) * self.smooth_grab
                    self.smooth_tail = self.alpha * raw_tail + (1.0 - self.alpha) * self.smooth_tail
            else:
                self.lost_counter += 1
                if self.lost_counter > self.max_lost_frames:
                    self.smooth_head = None
                    self.smooth_grab = None
                    self.smooth_tail = None

            # Render Tracking Overlay
            out = frame.copy()
            h, w = out.shape[:2]

            if self.smooth_head is not None and self.smooth_tail is not None and self.smooth_grab is not None:
                p_head = tuple(self.smooth_head.astype(int))
                p_grab = tuple(self.smooth_grab.astype(int))
                p_tail = tuple(self.smooth_tail.astype(int))

                # Handle line
                cv2.line(out, p_tail, p_head, (0, 255, 255), 3)

                # Head (Red)
                cv2.circle(out, p_head, 12, (255, 255, 255), 2)
                cv2.circle(out, p_head, 8, (0, 0, 255), -1)
                cv2.putText(out, "HEAD", (p_head[0] + 12, p_head[1] + 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                # Grab Point (Yellow)
                cv2.circle(out, p_grab, 11, (255, 255, 255), 2)
                cv2.circle(out, p_grab, 7, (0, 230, 255), -1)
                cv2.putText(out, "GRAB", (p_grab[0] + 12, p_grab[1] + 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 230, 255), 2)

                # Tail (Blue)
                cv2.circle(out, p_tail, 12, (255, 255, 255), 2)
                cv2.circle(out, p_tail, 8, (255, 0, 0), -1)
                cv2.putText(out, "TAIL", (p_tail[0] + 12, p_tail[1] + 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

                # Angles
                v = self.smooth_head - self.smooth_tail
                yaw = (math.degrees(math.atan2(v[1], v[0])) + 360) % 360
                claw_angle = yaw % 180.0

                # HUD
                cv2.rectangle(out, (15, 15), (420, 110), (20, 20, 20), -1)
                cv2.putText(out, f"3-PT TRACKER: LOCKED", (25, 42),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(out, f"Grip (X, Y): ({p_grab[0]}, {p_grab[1]}) px", (25, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
                cv2.putText(out, f"Yaw: {yaw:5.1f} deg | Claw: {claw_angle:5.1f} deg", (25, 95),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
            else:
                cv2.rectangle(out, (15, 15), (320, 60), (20, 20, 20), -1)
                cv2.putText(out, "SEARCHING HAMMER...", (25, 45),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

            cv2.putText(out, f"FPS: {fps:.1f}", (w - 140, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow("3-Keypoint Hammer Pose Tracker", out)
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):
                break

        if self.mode == "ZED_SDK":
            self.zed.close()
        elif self.cap:
            self.cap.release()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Live 3-Keypoint Hammer Tracker")
    parser.add_argument("--weights", type=str, default="weights/hammer_flip_5pt_best.pt",
                        help="Path to trained YOLO weights (defaults to weights/hammer_flip_5pt_best.pt)")
    parser.add_argument("--conf", type=float, default=0.35, help="Confidence threshold")
    args = parser.parse_args()

    tracker = Hammer3PointTracker(weights_path=args.weights, conf=args.conf)
    tracker.run()


if __name__ == "__main__":
    main()
