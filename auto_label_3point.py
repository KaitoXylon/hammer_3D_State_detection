#!/usr/bin/env python3
"""
================================================================================
3-KEYPOINT AUTO-LABELING PIPELINE: [HEAD, GRAB, TAIL]
================================================================================
University Rover Challenge (URC) - Autonomous Hammer Detection

Dedicated standalone auto-labeler for the 3-point handle model:
  K1: Head Center (Striking head T-junction)
  K2: Grab Point  (Handle grasp target, ~40% from base)
  K3: Tail Base   (Bottom of handle)

Outputs:
  - Dataset: 'dataset_3pt/images/' and 'dataset_3pt/labels/'
  - YAML Config: 'hammer_pose_3pt.yaml'
  - Verification Previews: 'dataset_3pt/previews/'

Usage:
  python3 auto_label_3point.py                     # Auto-label from default video
  python3 auto_label_3point.py --video my_vid.mp4  # Auto-label from custom video
  python3 auto_label_3point.py --train             # Auto-label and immediately train
================================================================================
"""

import os
import sys
import time
import shutil
import argparse
import glob
import cv2
import numpy as np
import torch
from ultralytics import FastSAM


def compute_sharpness(gray_img):
    return cv2.Laplacian(gray_img, cv2.CV_64F).var()


def select_hammer_mask(res, img_w, img_h, min_area=15000, max_area=250000, min_aspect=1.2, max_aspect=6.5):
    if res.masks is None or len(res.boxes) == 0:
        return None

    boxes = res.boxes.xyxy.cpu().numpy()
    masks = res.masks.data.cpu().numpy()
    center_img = np.array([img_w / 2.0, img_h / 2.0])

    best_mask = None
    min_dist = float("inf")

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        bw = x2 - x1
        bh = y2 - y1
        area = bw * bh
        aspect = max(bw, bh) / max(1.0, min(bw, bh))

        if area < min_area or area > max_area:
            continue
        if aspect < min_aspect or aspect > max_aspect:
            continue
        if x1 < 5 or y1 < 5 or x2 > (img_w - 5) or y2 > (img_h - 5):
            continue

        box_center = np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0])
        dist = np.linalg.norm(box_center - center_img)
        if dist < min_dist:
            min_dist = dist
            m = (masks[i] * 255).astype(np.uint8)
            best_mask = cv2.resize(m, (img_w, img_h), interpolation=cv2.INTER_NEAREST)

    return best_mask


def snap_to_medial_spine(pt, dist_map, img_w, img_h, search_radius=25):
    x, y = int(pt[0]), int(pt[1])
    y1, y2 = max(0, y - search_radius), min(img_h, y + search_radius)
    x1, x2 = max(0, x - search_radius), min(img_w, x + search_radius)

    roi = dist_map[y1:y2, x1:x2]
    if roi.size == 0 or np.max(roi) == 0:
        return pt

    loc = np.unravel_index(np.argmax(roi), roi.shape)
    return np.array([x1 + loc[1], y1 + loc[0]], dtype=float)


def extract_3_keypoints(mask, img_w, img_h):
    """
    Extracts 3 keypoints:
      K1: Head Center
      K2: Grab Point (40% from Tail along handle spine)
      K3: Tail Center
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    c = max(contours, key=cv2.contourArea)
    if cv2.contourArea(c) < 10000:
        return None

    dist_map = cv2.distanceTransform(mask, cv2.DIST_L2, 5)

    y_idx, x_idx = np.where(mask > 0)
    coords = np.column_stack((x_idx, y_idx)).astype(np.float32)
    mean, eigenvectors = cv2.PCACompute(coords, mean=None)
    center = mean[0]
    axis_handle = eigenvectors[0]

    projs = np.dot(coords - center, axis_handle)
    p_min = center + np.min(projs) * axis_handle
    p_max = center + np.max(projs) * axis_handle

    length = np.linalg.norm(p_min - p_max)
    if length < 80:
        return None

    # Density test to find Head vs Tail
    r = int(length * 0.15)
    d_min = np.sum(mask[max(0, int(p_min[1]-r)):min(img_h, int(p_min[1]+r)),
                        max(0, int(p_min[0]-r)):min(img_w, int(p_min[0]+r))] > 0)
    d_max = np.sum(mask[max(0, int(p_max[1]-r)):min(img_h, int(p_max[1]+r)),
                        max(0, int(p_max[0]-r)):min(img_w, int(p_max[0]+r))] > 0)

    if d_min > d_max:
        head_raw, tail_raw = p_min, p_max
    else:
        head_raw, tail_raw = p_max, p_min

    head = snap_to_medial_spine(head_raw, dist_map, img_w, img_h, 25)
    tail = snap_to_medial_spine(tail_raw, dist_map, img_w, img_h, 25)
    grab_raw = tail + 0.40 * (head - tail)
    grab = snap_to_medial_spine(grab_raw, dist_map, img_w, img_h, 20)

    bx, by, bw, bh = cv2.boundingRect(c)
    pad_w = int(bw * 0.05)
    pad_h = int(bh * 0.05)
    x1 = max(0, bx - pad_w)
    y1 = max(0, by - pad_h)
    x2 = min(img_w, bx + bw + pad_w)
    y2 = min(img_h, by + bh + pad_h)
    bbox = (x1, y1, x2 - x1, y2 - y1)

    return [head, grab, tail], bbox


def render_preview_3pt(frame, kpts):
    dbg = frame.copy()
    head, grab, tail = kpts
    p_head = tuple(head.astype(int))
    p_grab = tuple(grab.astype(int))
    p_tail = tuple(tail.astype(int))

    # Spine line
    cv2.line(dbg, p_tail, p_head, (0, 255, 255), 3)

    # Head (Red)
    cv2.circle(dbg, p_head, 12, (255, 255, 255), 2)
    cv2.circle(dbg, p_head, 8, (0, 0, 255), -1)
    cv2.putText(dbg, "HEAD", (p_head[0] + 12, p_head[1] + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # Grab Point (Yellow)
    cv2.circle(dbg, p_grab, 11, (255, 255, 255), 2)
    cv2.circle(dbg, p_grab, 7, (0, 230, 255), -1)
    cv2.putText(dbg, "GRAB", (p_grab[0] + 12, p_grab[1] + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 230, 255), 2)

    # Tail (Blue)
    cv2.circle(dbg, p_tail, 12, (255, 255, 255), 2)
    cv2.circle(dbg, p_tail, 8, (255, 0, 0), -1)
    cv2.putText(dbg, "TAIL", (p_tail[0] + 12, p_tail[1] + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    # Info banner
    v = np.array(p_head) - np.array(p_tail)
    yaw = (np.degrees(np.arctan2(v[1], v[0])) + 360) % 360
    cv2.rectangle(dbg, (15, 15), (380, 60), (20, 20, 20), -1)
    cv2.putText(dbg, f"3-PT POSE | YAW: {yaw:.1f} deg", (25, 46),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)

    return dbg


def run_pipeline(
    video_path="hammer_video.mp4",
    output_dir="dataset_3pt",
    step=8,
    blur_threshold=8.0,
    val_ratio=0.20,
    previews=30,
    fastsam_weights="FastSAM-s.pt",
    conf=0.35,
    device="cuda" if torch.cuda.is_available() else "cpu"
):
    print("\n" + "=" * 70)
    print("   3-KEYPOINT AUTO-LABELING: [HEAD, GRAB, TAIL]")
    print("=" * 70)

    if not os.path.exists(video_path):
        fallbacks = ["edited_video.mp4", "/home/mt/2026-10-17 05-28-25_edited.mp4"]
        for fb in fallbacks:
            if os.path.exists(fb):
                video_path = fb
                break

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    for split in ["train", "val"]:
        os.makedirs(f"{output_dir}/images/{split}", exist_ok=True)
        os.makedirs(f"{output_dir}/labels/{split}", exist_ok=True)
    preview_dir = f"{output_dir}/previews"
    os.makedirs(preview_dir, exist_ok=True)

    print(f"[INFO] Video       : {video_path}")
    print(f"[INFO] Output Dir  : {output_dir}/")
    print(f"[INFO] Keypoints   : 3 Points [Head, Grab, Tail]")
    print(f"[INFO] Device      : {device.upper()}")

    sam = FastSAM(fastsam_weights)
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    img_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    img_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frame_idx = 0
    saved_count = 0
    preview_count = 0
    start_time = time.time()
    val_interval = int(round(1.0 / val_ratio))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % step == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if compute_sharpness(gray) >= blur_threshold:
                res = sam(frame, device=device, retina_masks=True, conf=conf, verbose=False)[0]
                mask = select_hammer_mask(res, img_w, img_h)

                if mask is not None:
                    res_kpts = extract_3_keypoints(mask, img_w, img_h)
                    if res_kpts is not None:
                        kpts, bbox = res_kpts
                        bx, by, bw, bh = bbox

                        # Normalization
                        xc = (bx + bw / 2.0) / img_w
                        yc = (by + bh / 2.0) / img_h
                        nw = bw / img_w
                        nh = bh / img_h

                        kpt_strs = []
                        for pt in kpts:
                            kpt_strs.append(f"{pt[0]/img_w:.6f} {pt[1]/img_h:.6f} 2")

                        label_line = f"0 {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f} " + " ".join(kpt_strs) + "\n"
                        split = "val" if (saved_count % val_interval == 0) else "train"

                        img_name = f"hammer_3pt_{saved_count:05d}.jpg"
                        lbl_name = f"hammer_3pt_{saved_count:05d}.txt"

                        cv2.imwrite(f"{output_dir}/images/{split}/{img_name}", frame)
                        with open(f"{output_dir}/labels/{split}/{lbl_name}", "w") as f:
                            f.write(label_line)

                        if preview_count < previews:
                            dbg = render_preview_3pt(frame, kpts)
                            cv2.imwrite(f"{preview_dir}/preview_{preview_count:02d}.jpg", dbg)
                            preview_count += 1

                        saved_count += 1

            if frame_idx % (step * 25) == 0:
                pct = (frame_idx / max(1, total_frames)) * 100
                print(f"  [{pct:5.1f}%] Frame {frame_idx:5d}/{total_frames} | Saved: {saved_count:4d} labels")

        frame_idx += 1

    cap.release()

    # YAML config
    yaml_path = os.path.join(os.getcwd(), "hammer_pose_3pt.yaml")
    yaml_content = f"""# YOLO-Pose 3-Keypoint Hammer Dataset: [Head, Grab, Tail]
path: {os.path.abspath(output_dir)}
train: images/train
val: images/val

# 3 Keypoints: [Head, Grab, Tail]
kpt_shape: [3, 3]
flip_idx: []

names:
  0: hammer
"""
    with open(yaml_path, "w") as yf:
        yf.write(yaml_content)

    print("\n" + "=" * 70)
    print("   3-KEYPOINT AUTO-LABELING FINISHED!")
    print(f"   Saved Images : {saved_count}")
    print(f"   Dataset Dir  : {output_dir}/")
    print(f"   Config YAML  : {yaml_path}")
    print(f"   Previews     : {preview_dir}/ ({preview_count} images)")
    print("=" * 70 + "\n")

    return yaml_path


def train_3pt(yaml_path, epochs=60):
    from ultralytics import YOLO
    print("\n[INFO] Starting training YOLOv8n-pose on 3-Keypoint dataset...")
    model = YOLO("yolov8n-pose.pt")
    model.train(
        data=yaml_path,
        epochs=epochs,
        imgsz=640,
        batch=16,
        device=0,
        workers=4,
        degrees=180.0,
        fliplr=0.0,
        flipud=0.0,
        scale=0.3,
        save=True,
        project="runs_3pt",
        name="hammer_3pt_best",
        plots=True
    )
    print("\n[DONE] 3-Keypoint Model trained! Weights: runs_3pt/hammer_3pt_best/weights/best.pt")


def main():
    parser = argparse.ArgumentParser(description="Dedicated 3-Keypoint Auto-Labeler: [Head, Grab, Tail]")
    parser.add_argument("--video", type=str, default="hammer_video.mp4")
    parser.add_argument("--output_dir", type=str, default="dataset_3pt")
    parser.add_argument("--step", type=int, default=8)
    parser.add_argument("--blur_thresh", type=float, default=8.0)
    parser.add_argument("--previews", type=int, default=30)
    parser.add_argument("--train", action="store_true", help="Train model immediately after labeling")
    parser.add_argument("--epochs", type=int, default=60)
    args = parser.parse_args()

    yaml_path = run_pipeline(
        video_path=args.video,
        output_dir=args.output_dir,
        step=args.step,
        blur_threshold=args.blur_thresh,
        previews=args.previews
    )

    if args.train:
        train_3pt(yaml_path, epochs=args.epochs)


if __name__ == "__main__":
    main()
