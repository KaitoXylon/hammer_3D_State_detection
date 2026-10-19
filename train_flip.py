#!/usr/bin/env python3
"""
Train YOLOv8-Pose on 5-Keypoint Hammer Dataset (Flip Detection)
--------------------------------------------------------------
Model: YOLOv8n-pose
Dataset: hammer_pose_flip.yaml (5 keypoints)
Augmentations:
- degrees=180.0 : Full 360-degree rotation invariance
- fliplr=0.0    : Preserves Face vs. Claw asymmetry
"""

from ultralytics import YOLO

def main():
    print("[INFO] Loading YOLOv8n-pose pre-trained model for 5-keypoint training...")
    model = YOLO("yolov8n-pose.pt")

    print("[INFO] Starting training on 5-keypoint flip dataset (872 images)...")
    model.train(
        data="hammer_pose_flip.yaml",
        epochs=60,
        imgsz=640,
        batch=16,
        device=0,          # RTX 5050 GPU
        workers=4,
        degrees=180.0,     # Full 360-degree rotation augmentation
        fliplr=0.0,        # No horizontal flip to preserve Face vs. Claw orientation
        flipud=0.0,
        scale=0.3,
        save=True,
        project="runs_flip",
        name="hammer_flip_best",
        plots=True
    )
    print("\n[DONE] 5-Keypoint Training finished! Model weights saved to: runs_flip/hammer_flip_best/weights/best.pt")

if __name__ == "__main__":
    main()
