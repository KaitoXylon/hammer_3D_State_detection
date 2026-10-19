#!/usr/bin/env python3
"""
Train YOLOv8-Pose on Hammer Dataset
-----------------------------------
Model: YOLOv8n-pose
Augmentations:
- degrees=180.0 : Learns all 360-degree orientations for top-down picking.
- fliplr=0.0    : Prevents mirroring so Head and Tail identities stay distinct.
"""

from ultralytics import YOLO

def main():
    print("[INFO] Loading YOLOv8n-pose pre-trained model...")
    model = YOLO("yolov8n-pose.pt")

    print("[INFO] Starting training on hammer dataset...")
    model.train(
        data="hammer_pose.yaml",
        epochs=60,
        imgsz=640,
        batch=16,
        device=0,          # RTX 5050 GPU
        workers=4,
        degrees=180.0,     # Full 360-degree rotation augmentation
        fliplr=0.0,        # No mirroring (preserves Head vs. Tail orientation)
        flipud=0.0,
        scale=0.3,
        save=True,
        project="runs",
        name="hammer_pose_best",
        plots=True
    )
    print("\n[DONE] Training complete! Best weights saved to: runs/pose/hammer_pose_best/weights/best.pt")

if __name__ == "__main__":
    main()
