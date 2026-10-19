#!/usr/bin/env python3
"""
ZED 2i Left Camera Video Recorder
---------------------------------
Records RGB video exclusively from the LEFT eye of a Stereolabs ZED 2i camera.
Features:
- Primary: Uses native ZED SDK (pyzed) with DEPTH_MODE=NONE for maximum performance.
- Fallback: Uses standard OpenCV UVC capture (crops left half of side-by-side stream).
- Interactive OpenCV window with live HUD (Recording timer, frame counter, status).
- Key controls:
    [SPACE] : Start / Pause Recording
    [R]     : Restart / Clear current recording
    [Q]/ESC : Save and Exit
"""

import sys
import time
import cv2
import numpy as np

# Try importing pyzed
HAS_PYZED = False
try:
    import pyzed.sl as sl
    HAS_PYZED = True
except ImportError:
    HAS_PYZED = False


class ZedRecorder:
    def __init__(self, output_path="hammer_video.mp4", fps=30, resolution="HD720"):
        self.output_path = output_path
        self.fps = fps
        self.resolution_str = resolution
        self.mode = None
        self.zed = None
        self.cap = None
        self.writer = None
        self.is_recording = False
        self.frames_recorded = 0
        self.start_time = 0
        self.elapsed_time = 0

        self.width = 1280
        self.height = 720

        self._init_camera()

    def _init_camera(self):
        # Attempt 1: Native ZED SDK
        if HAS_PYZED:
            print("[INFO] Attempting to connect via ZED SDK (pyzed)...")
            self.zed = sl.Camera()
            init_params = sl.InitParameters()

            if self.resolution_str == "HD1080":
                init_params.camera_resolution = sl.RESOLUTION.HD1080
                self.width, self.height = 1920, 1080
            elif self.resolution_str == "HD2K":
                init_params.camera_resolution = sl.RESOLUTION.HD2K
                self.width, self.height = 2208, 1242
            else:
                init_params.camera_resolution = sl.RESOLUTION.HD720
                self.width, self.height = 1280, 720

            init_params.camera_fps = self.fps
            init_params.depth_mode = sl.DEPTH_MODE.NONE  # Disable depth to save GPU resources
            init_params.coordinate_units = sl.UNIT.MILLIMETER

            err = self.zed.open(init_params)
            if err == sl.ERROR_CODE.SUCCESS:
                print(f"[SUCCESS] ZED 2i opened via ZED SDK ({self.width}x{self.height} @ {self.fps} FPS)")
                self.mode = "ZED_SDK"
                self.image_left_mat = sl.Mat()
                self.runtime_params = sl.RuntimeParameters()
                return
            else:
                print(f"[WARNING] ZED SDK open failed: {err}. Falling back to OpenCV UVC capture...")

        # Attempt 2: Standard OpenCV UVC Fallback
        print("[INFO] Scanning for UVC camera devices...")
        for dev_idx in [0, 1, 2, 3]:
            cap = cv2.VideoCapture(dev_idx)
            if cap.isOpened():
                # ZED transmits side-by-side frames (e.g. 2560x720 for HD720, 3840x1080 for HD1080)
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width * 2)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                cap.set(cv2.CAP_PROP_FPS, self.fps)
                ret, frame = cap.read()
                if ret and frame is not None:
                    print(f"[SUCCESS] Camera found on /dev/video{dev_idx} via OpenCV UVC")
                    self.cap = cap
                    self.mode = "UVC"
                    return
                cap.release()

        print("[ERROR] Could not connect to ZED 2i via ZED SDK or OpenCV UVC.")
        print("Please check that the USB 3.0 cable is securely plugged in.")
        sys.exit(1)

    def read_left_frame(self):
        """Fetches only the LEFT camera frame in standard BGR format."""
        if self.mode == "ZED_SDK":
            if self.zed.grab(self.runtime_params) == sl.ERROR_CODE.SUCCESS:
                self.zed.retrieve_image(self.image_left_mat, sl.VIEW.LEFT)
                # ZED SDK stores 4-channel images in BGRA format by default
                bgra = self.image_left_mat.get_data()
                return cv2.cvtColor(bgra, cv2.COLOR_BGRA2BGR)
            return None

        elif self.mode == "UVC":
            ret, full_frame = self.cap.read()
            if not ret or full_frame is None:
                return None
            # ZED side-by-side format: Left camera is the left half
            h, w = full_frame.shape[:2]
            left_frame = full_frame[:, : w // 2]
            return cv2.resize(left_frame, (self.width, self.height))

        return None

    def start_recording(self):
        if not self.is_recording:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.writer = cv2.VideoWriter(self.output_path, fourcc, self.fps, (self.width, self.height))
            self.is_recording = True
            self.start_time = time.time() - self.elapsed_time
            print(f"\n[REC] Recording STARTED -> Saving to '{self.output_path}'")

    def pause_recording(self):
        if self.is_recording:
            self.is_recording = False
            self.elapsed_time = time.time() - self.start_time
            print(f"\n[PAUSE] Recording PAUSED ({self.frames_recorded} frames captured)")

    def restart_recording(self):
        if self.writer:
            self.writer.release()
            self.writer = None
        self.is_recording = False
        self.frames_recorded = 0
        self.elapsed_time = 0
        print("\n[RESET] Recording reset to 0.")

    def run(self):
        window_name = "ZED 2i - Left RGB Camera Recorder"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1024, 576)

        prev_time = time.time()
        fps_display = 0.0

        print("\n" + "=" * 60)
        print("  ZED 2i LEFT CAMERA RECORDER READY")
        print("  Controls:")
        print("    [SPACE]   : Start / Pause Recording")
        print("    [R]       : Reset Recording")
        print("    [Q] / ESC : Save and Exit")
        print("=" * 60 + "\n")

        while True:
            frame = self.read_left_frame()
            if frame is None:
                continue

            curr_time = time.time()
            fps_display = 0.9 * fps_display + 0.1 * (1.0 / max(1e-4, curr_time - prev_time))
            prev_time = curr_time

            # Handle video writing
            if self.is_recording:
                self.writer.write(frame)
                self.frames_recorded += 1
                self.elapsed_time = curr_time - self.start_time

            # Create HUD overlay
            hud = frame.copy()
            h, w = frame.shape[:2]

            # Top bar background
            cv2.rectangle(hud, (0, 0), (w, 60), (20, 20, 20), -1)

            # Left side: Mode & Resolution & FPS
            status_text = f"Mode: {self.mode} | {w}x{h} @ {fps_display:.1f} FPS"
            cv2.putText(hud, status_text, (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

            # Center/Right: Recording status
            mins = int(self.elapsed_time) // 60
            secs = int(self.elapsed_time) % 60
            millis = int((self.elapsed_time - int(self.elapsed_time)) * 10)

            if self.is_recording:
                # Blinking red dot every 500ms
                if int(curr_time * 2) % 2 == 0:
                    cv2.circle(hud, (w - 280, 32), 10, (0, 0, 255), -1)
                cv2.putText(hud, f"REC  {mins:02d}:{secs:02d}.{millis} ({self.frames_recorded}f)", 
                            (w - 260, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            else:
                if self.frames_recorded > 0:
                    cv2.putText(hud, f"PAUSED {mins:02d}:{secs:02d}", (w - 240, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 200, 255), 2)
                else:
                    cv2.putText(hud, "STANDBY (Press SPACE)", (w - 320, 40), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

            # Bottom instructions bar
            cv2.rectangle(hud, (0, h - 35), (w, h), (15, 15, 15), -1)
            cv2.putText(hud, "SPACE: Start/Pause | R: Reset | Q / ESC: Finish & Save", 
                        (20, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)

            # Display
            cv2.imshow(window_name, hud)
            key = cv2.waitKey(1) & 0xFF

            if key == 32:  # SPACE
                if self.is_recording:
                    self.pause_recording()
                else:
                    self.start_recording()
            elif key in [ord('r'), ord('R')]:
                self.restart_recording()
            elif key in [27, ord('q'), ord('Q')]:  # ESC or Q
                print("\n[INFO] Stopping recorder...")
                break

        # Cleanup
        if self.writer:
            self.writer.release()
            print(f"[DONE] Video saved successfully: '{self.output_path}'")
            print(f"       Total Frames: {self.frames_recorded}")
            print(f"       Total Time: {mins:02d}:{secs:02d}")

        if self.mode == "ZED_SDK" and self.zed:
            self.zed.close()
        elif self.cap:
            self.cap.release()

        cv2.destroyAllWindows()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="ZED 2i Left Camera Recorder")
    parser.add_argument("--output", type=str, default="hammer_video.mp4", help="Output file path (default: hammer_video.mp4)")
    parser.add_argument("--fps", type=int, default=30, help="Camera frame rate (default: 30)")
    parser.add_argument("--res", type=str, default="HD720", choices=["HD720", "HD1080", "HD2K"], help="Resolution")
    args = parser.parse_args()

    recorder = ZedRecorder(output_path=args.output, fps=args.fps, resolution=args.res)
    recorder.run()
