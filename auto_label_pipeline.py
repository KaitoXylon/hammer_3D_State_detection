#!/usr/bin/env python3
"""
================================================================================
AUTONOMOUS AUTO-LABELING PIPELINE FOR ROBOTIC TOOL POSE ESTIMATION
================================================================================
University Rover Challenge (URC) - Delivery Mission / Astronaut Assistance
Author: Team Autonomous Systems

A unified, end-to-end auto-labeling pipeline that transforms raw video into a
fully-annotated, high-precision YOLO-Pose dataset without any manual clicking:

  Stage 1: Frame Ingestion & Sharpness Filtering (Laplacian Variance)
  Stage 2: Foundation Model Instance Segmentation (FastSAM-s on GPU)
  Stage 3: Medial Spine Fitting (PCA + Euclidean Distance Transform)
  Stage 4: Asymmetry & Topological Disambiguation (Face vs. Claw, Head vs. Tail)
  Stage 5: YOLO-Pose Dataset Assembly & YAML Configuration
  Stage 6: Visual Verification Previews (Overlay with Live Flip/Roll State)
  Stage 7: (Optional) Direct Neural Training (YOLOv8n-pose with optimal augs)

Usage:
  # Run full 5-keypoint auto-labeling on default video:
  python3 auto_label_pipeline.py

  # Run on a custom video and save to custom directory:
  python3 auto_label_pipeline.py --video my_video.mp4 --output_dir my_dataset

  # Auto-label AND immediately train the YOLOv8-pose model:
  python3 auto_label_pipeline.py --train --epochs 60

  # Run in 2-keypoint mode (Head, Tail only):
  python3 auto_label_pipeline.py --mode 2
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


# ==============================================================================
# 1. Image Quality & Geometric Filtering
# ==============================================================================

def compute_sharpness(gray_img):
    """
    Computes frame sharpness score using variance of the 2D Laplacian operator:
      Var(Laplacian) = 1/N * sum((Laplacian(I) - mean)^2)
    Higher values indicate sharp edges; low values indicate motion blur.
    """
    return cv2.Laplacian(gray_img, cv2.CV_64F).var()


def select_best_mask(res, img_w, img_h, min_area=15000, max_area=250000, min_aspect=1.2, max_aspect=6.5):
    """
    Filters FastSAM segmentation candidates using physical geometric priors:
      1. Area constraint: filters out tiny dirt specs and full-frame wall masks.
      2. Aspect ratio: ensures elongated tool profile.
      3. Boundary margin: discards masks truncated by the image frame borders.
      4. Optical center proximity: ranks by distance to camera principal point.
    """
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

        # Size and aspect ratio filters
        if area < min_area or area > max_area:
            continue
        if aspect < min_aspect or aspect > max_aspect:
            continue

        # Inset margin check (reject border clipping)
        if x1 < 5 or y1 < 5 or x2 > (img_w - 5) or y2 > (img_h - 5):
            continue

        # Distance to optical center
        box_center = np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0])
        dist = np.linalg.norm(box_center - center_img)
        if dist < min_dist:
            min_dist = dist
            m = (masks[i] * 255).astype(np.uint8)
            best_mask = cv2.resize(m, (img_w, img_h), interpolation=cv2.INTER_NEAREST)

    return best_mask


# ==============================================================================
# 2. Keypoint Extraction: PCA + Medial Distance Transform
# ==============================================================================

def snap_to_medial_spine(pt, dist_map, img_w, img_h, search_radius=25):
    """
    Refines a candidate coordinate by snapping it to the local maximum ridge
    of the Euclidean Distance Transform (medial axis of the mask).
    """
    x, y = int(pt[0]), int(pt[1])
    y1, y2 = max(0, y - search_radius), min(img_h, y + search_radius)
    x1, x2 = max(0, x - search_radius), min(img_w, x + search_radius)

    roi = dist_map[y1:y2, x1:x2]
    if roi.size == 0 or np.max(roi) == 0:
        return pt

    loc = np.unravel_index(np.argmax(roi), roi.shape)
    return np.array([x1 + loc[1], y1 + loc[0]], dtype=float)


def extract_keypoints_5pt(mask, img_w, img_h):
    """
    Extracts 5 topologically anchored keypoints:
      K1: Face (Striking Head)
      K2: Claw (Curved Tip)
      K3: Head Center (T-Junction)
      K4: Grip Point (Handle Balance, ~40% from Tail)
      K5: Handle Tail (Base)
    Also computes tight bounding box with 5% padding.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    c = max(contours, key=cv2.contourArea)
    if cv2.contourArea(c) < 10000:
        return None

    # Compute Euclidean Distance Transform
    dist_map = cv2.distanceTransform(mask, cv2.DIST_L2, 5)

    # 1. PCA along handle axis
    y_idx, x_idx = np.where(mask > 0)
    coords = np.column_stack((x_idx, y_idx)).astype(np.float32)
    mean, eigenvectors = cv2.PCACompute(coords, mean=None)
    center = mean[0]
    axis_handle = eigenvectors[0]

    # Project along handle axis
    projs = np.dot(coords - center, axis_handle)
    p_min = center + np.min(projs) * axis_handle
    p_max = center + np.max(projs) * axis_handle

    length = np.linalg.norm(p_min - p_max)
    if length < 80:
        return None

    # 2. Disambiguate Head vs. Tail by mass density
    r = int(length * 0.15)
    d_min = np.sum(mask[max(0, int(p_min[1]-r)):min(img_h, int(p_min[1]+r)),
                        max(0, int(p_min[0]-r)):min(img_w, int(p_min[0]+r))] > 0)
    d_max = np.sum(mask[max(0, int(p_max[1]-r)):min(img_h, int(p_max[1]+r)),
                        max(0, int(p_max[0]-r)):min(img_w, int(p_max[0]+r))] > 0)

    if d_min > d_max:
        head_raw, tail_raw = p_min, p_max
        handle_dir = -axis_handle
    else:
        head_raw, tail_raw = p_max, p_min
        handle_dir = axis_handle

    # 3. Head Crossbar axis (perpendicular to handle)
    axis_head = np.array([-handle_dir[1], handle_dir[0]])

    # Isolate head region points
    dist_to_head = np.dot(coords - head_raw, handle_dir)
    head_indices = np.where(dist_to_head > -r * 1.5)[0]
    if len(head_indices) < 50:
        return None
    head_coords = coords[head_indices]

    projs_head = np.dot(head_coords - head_raw, axis_head)
    t1, t2 = np.min(projs_head), np.max(projs_head)
    pt1 = head_raw + t1 * axis_head
    pt2 = head_raw + t2 * axis_head

    # 4. Disambiguate Face (blunt/denser) vs Claw (sharp/less dense)
    r_tip = 25
    d1 = np.sum(mask[max(0, int(pt1[1]-r_tip)):min(img_h, int(pt1[1]+r_tip)),
                        max(0, int(pt1[0]-r_tip)):min(img_w, int(pt1[0]+r_tip))] > 0)
    d2 = np.sum(mask[max(0, int(pt2[1]-r_tip)):min(img_h, int(pt2[1]+r_tip)),
                        max(0, int(pt2[0]-r_tip)):min(img_w, int(pt2[0]+r_tip))] > 0)

    if d1 > d2:
        face_raw, claw_raw = pt1, pt2
    else:
        face_raw, claw_raw = pt2, pt1

    # 5. Snap to medial distance transform maxima
    face = snap_to_medial_spine(face_raw, dist_map, img_w, img_h, 20)
    claw = snap_to_medial_spine(claw_raw, dist_map, img_w, img_h, 20)
    head_center = snap_to_medial_spine(head_raw, dist_map, img_w, img_h, 25)
    tail = snap_to_medial_spine(tail_raw, dist_map, img_w, img_h, 25)
    grip_raw = tail + 0.40 * (head_center - tail)
    grip = snap_to_medial_spine(grip_raw, dist_map, img_w, img_h, 20)

    # 6. Tight bounding box with 5% margin
    bx, by, bw, bh = cv2.boundingRect(c)
    pad_w = int(bw * 0.05)
    pad_h = int(bh * 0.05)
    x1 = max(0, bx - pad_w)
    y1 = max(0, by - pad_h)
    x2 = min(img_w, bx + bw + pad_w)
    y2 = min(img_h, by + bh + pad_h)
    bbox = (x1, y1, x2 - x1, y2 - y1)

    kpts = [face, claw, head_center, grip, tail]
    return kpts, bbox


def extract_keypoints_2pt(mask, img_w, img_h):
    """
    Extracts 2 keypoints:
      K1: Head Center (Middle of striking head)
      K2: Tail Center (Base of handle)
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

    bx, by, bw, bh = cv2.boundingRect(c)
    pad_w = int(bw * 0.05)
    pad_h = int(bh * 0.05)
    x1 = max(0, bx - pad_w)
    y1 = max(0, by - pad_h)
    x2 = min(img_w, bx + bw + pad_w)
    y2 = min(img_h, by + bh + pad_h)
    bbox = (x1, y1, x2 - x1, y2 - y1)

    return [head, tail], bbox


# ==============================================================================
# 3. Annotation Formatter & Preview Renderer
# ==============================================================================

def format_yolo_pose_label(bbox, kpts, img_w, img_h):
    """
    Formats bounding box and keypoints into standard YOLO-Pose format:
      0 <xc> <yc> <w> <h> <k1_x> <k1_y> <v1> ...
    Normalized to [0, 1]. Visibility = 2 (labeled and visible).
    """
    bx, by, bw, bh = bbox
    xc = (bx + bw / 2.0) / img_w
    yc = (by + bh / 2.0) / img_h
    nw = bw / img_w
    nh = bh / img_h

    kpt_strs = []
    for pt in kpts:
        nx = pt[0] / img_w
        ny = pt[1] / img_h
        kpt_strs.append(f"{nx:.6f} {ny:.6f} 2")

    return f"0 {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f} " + " ".join(kpt_strs) + "\n"


def render_preview_image(frame, kpts, mode=5):
    """
    Renders high-contrast visualization overlay with skeleton lines,
    labeled joints, and computed 3D Flip state.
    """
    dbg = frame.copy()

    if mode == 5:
        face, claw, head_center, grip, tail = kpts
        p_face = tuple(face.astype(int))
        p_claw = tuple(claw.astype(int))
        p_head = tuple(head_center.astype(int))
        p_grip = tuple(grip.astype(int))
        p_tail = tuple(tail.astype(int))

        # Skeleton lines
        cv2.line(dbg, p_tail, p_head, (0, 255, 255), 3)  # Yellow handle
        cv2.line(dbg, p_claw, p_face, (0, 255, 255), 3)  # Yellow crossbar

        # K1: Face (Striking head - Red)
        cv2.circle(dbg, p_face, 12, (255, 255, 255), 2)
        cv2.circle(dbg, p_face, 8, (0, 0, 255), -1)
        cv2.putText(dbg, "FACE", (p_face[0] + 12, p_face[1] + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

        # K2: Claw (Curved tip - Blue)
        cv2.circle(dbg, p_claw, 12, (255, 255, 255), 2)
        cv2.circle(dbg, p_claw, 8, (255, 0, 0), -1)
        cv2.putText(dbg, "CLAW", (p_claw[0] - 65, p_claw[1] + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 0), 2)

        # K3: Head Center (Yellow)
        cv2.circle(dbg, p_head, 9, (255, 255, 255), 2)
        cv2.circle(dbg, p_head, 6, (0, 255, 255), -1)

        # K4: Grip Target (Yellow/Orange)
        cv2.circle(dbg, p_grip, 9, (255, 255, 255), 2)
        cv2.circle(dbg, p_grip, 6, (0, 215, 255), -1)
        cv2.putText(dbg, "GRIP", (p_grip[0] + 12, p_grip[1] + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 215, 255), 2)

        # K5: Tail Base (Blue)
        cv2.circle(dbg, p_tail, 12, (255, 255, 255), 2)
        cv2.circle(dbg, p_tail, 8, (255, 0, 0), -1)
        cv2.putText(dbg, "TAIL", (p_tail[0] + 12, p_tail[1] + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 0), 2)

        # Compute Flip State via signed cross product
        v_handle = np.array(p_head) - np.array(p_tail)
        v_head = np.array(p_face) - np.array(p_claw)
        cross_val = v_handle[0] * v_head[1] - v_handle[1] * v_head[0]
        flip_text = "FLIP: LEFT (Normal)" if cross_val >= 0 else "FLIP: RIGHT (Flipped)"
        flip_color = (0, 255, 0) if cross_val >= 0 else (0, 165, 255)

        cv2.rectangle(dbg, (15, 15), (420, 60), (20, 20, 20), -1)
        cv2.putText(dbg, flip_text, (25, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.7, flip_color, 2)

    else:  # 2-point mode
        head, tail = kpts
        p_head = tuple(head.astype(int))
        p_tail = tuple(tail.astype(int))

        cv2.line(dbg, p_tail, p_head, (0, 255, 255), 3)
        cv2.circle(dbg, p_head, 12, (255, 255, 255), 2)
        cv2.circle(dbg, p_head, 8, (0, 0, 255), -1)
        cv2.putText(dbg, "HEAD", (p_head[0] + 12, p_head[1] + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv2.circle(dbg, p_tail, 12, (255, 255, 255), 2)
        cv2.circle(dbg, p_tail, 8, (255, 0, 0), -1)
        cv2.putText(dbg, "TAIL", (p_tail[0] + 12, p_tail[1] + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    return dbg


# ==============================================================================
# 4. Pipeline Execution Engine
# ==============================================================================

def run_auto_label_pipeline(
    video_path="hammer_video.mp4",
    output_dir="dataset_flip",
    mode=5,
    step=8,
    blur_threshold=8.0,
    val_ratio=0.20,
    num_previews=30,
    fastsam_weights="FastSAM-s.pt",
    conf_thresh=0.35,
    device=None,
    clean_existing=True,
    custom_yaml=None
):
    """
    Executes the entire auto-labeling pipeline from input video to YOLO dataset.
    """
    print("\n" + "=" * 70)
    print("   AUTONOMOUS AUTO-LABELING PIPELINE (URC 2027)")
    print("=" * 70)

    # 1. Verify Video Existence
    if not os.path.exists(video_path):
        # Check fallback candidates
        fallbacks = ["edited_video.mp4", "/home/mt/2026-10-17 05-28-25_edited.mp4"]
        found = False
        for fb in fallbacks:
            if os.path.exists(fb):
                print(f"[WARN] '{video_path}' not found. Using fallback: '{fb}'")
                video_path = fb
                found = True
                break
        if not found:
            print(f"[ERROR] Video file '{video_path}' does not exist!")
            sys.exit(1)

    # 2. Check FastSAM model
    if not os.path.exists(fastsam_weights):
        print(f"[ERROR] FastSAM weights file '{fastsam_weights}' not found in working directory!")
        sys.exit(1)

    # 3. Setup Compute Device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Compute Device       : {device.upper()}")
    if device == "cuda" or (isinstance(device, int) and device >= 0):
        gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "GPU"
        print(f"[INFO] GPU Accelerator      : {gpu_name}")

    print(f"[INFO] Source Video         : {video_path}")
    print(f"[INFO] Target Output Dir    : {output_dir}/")
    print(f"[INFO] Annotation Mode      : {mode}-Keypoint Model")
    print(f"[INFO] Frame Step Interval  : Every {step} frames")
    print(f"[INFO] Sharpness Threshold  : Laplacian Var >= {blur_threshold}")
    print(f"[INFO] Train/Val Split      : {int((1-val_ratio)*100)}% Train / {int(val_ratio*100)}% Val")

    # 4. Prepare Directory Structure
    if clean_existing and os.path.exists(output_dir):
        print(f"[INFO] Cleaning existing output directory '{output_dir}'...")
        shutil.rmtree(output_dir)

    for split in ["train", "val"]:
        os.makedirs(f"{output_dir}/images/{split}", exist_ok=True)
        os.makedirs(f"{output_dir}/labels/{split}", exist_ok=True)
    preview_dir = f"{output_dir}/previews"
    os.makedirs(preview_dir, exist_ok=True)

    # 5. Initialize FastSAM
    print(f"[INFO] Loading FastSAM model ({fastsam_weights})...")
    from ultralytics import FastSAM
    sam = FastSAM(fastsam_weights)

    # 6. Open Video Stream
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    img_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    img_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    duration_sec = total_frames / max(1.0, fps)

    print(f"[INFO] Video Details        : {total_frames} frames ({img_w}x{img_h} @ {fps:.1f} FPS, {duration_sec:.1f}s)")
    print(f"[INFO] Starting frame processing...\n")

    frame_idx = 0
    saved_count = 0
    preview_count = 0
    blurred_count = 0
    rejected_mask_count = 0
    start_time = time.time()

    val_interval = int(round(1.0 / val_ratio)) if val_ratio > 0 else 999999

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % step == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            sharpness = compute_sharpness(gray)

            if sharpness < blur_threshold:
                blurred_count += 1
            else:
                # Run FastSAM
                res = sam(frame, device=device, retina_masks=True, conf=conf_thresh, verbose=False)[0]
                mask = select_best_mask(res, img_w, img_h)

                if mask is None:
                    rejected_mask_count += 1
                else:
                    # Keypoint Extraction
                    if mode == 5:
                        res_kpts = extract_keypoints_5pt(mask, img_w, img_h)
                    else:
                        res_kpts = extract_keypoints_2pt(mask, img_w, img_h)

                    if res_kpts is not None:
                        kpts, bbox = res_kpts
                        label_line = format_yolo_pose_label(bbox, kpts, img_w, img_h)

                        # Determine Split
                        split = "val" if (saved_count % val_interval == 0) else "train"
                        img_filename = f"hammer_{saved_count:05d}.jpg"
                        lbl_filename = f"hammer_{saved_count:05d}.txt"

                        cv2.imwrite(f"{output_dir}/images/{split}/{img_filename}", frame)
                        with open(f"{output_dir}/labels/{split}/{lbl_filename}", "w") as f:
                            f.write(label_line)

                        # Save Previews
                        if preview_count < num_previews:
                            dbg_img = render_preview_image(frame, kpts, mode=mode)
                            cv2.imwrite(f"{preview_dir}/preview_{preview_count:02d}.jpg", dbg_img)
                            preview_count += 1

                        saved_count += 1

            # Progress Logging
            if frame_idx % (step * 25) == 0 or frame_idx == total_frames - 1:
                elapsed = time.time() - start_time
                percent = (frame_idx / max(1, total_frames)) * 100.0
                fps_proc = (frame_idx / max(0.001, elapsed))
                print(f"  [{percent:5.1f}%] Frame {frame_idx:5d}/{total_frames:5d} | Saved: {saved_count:4d} labels | Speed: {fps_proc:5.1f} fps")

        frame_idx += 1

    cap.release()
    elapsed_total = time.time() - start_time

    # 7. Generate Dataset YAML Configuration
    abs_output_dir = os.path.abspath(output_dir)
    if custom_yaml:
        yaml_path = os.path.abspath(custom_yaml)
    elif output_dir == "dataset_flip":
        yaml_path = os.path.join(os.getcwd(), "hammer_pose_flip.yaml")
    elif output_dir == "dataset":
        yaml_path = os.path.join(os.getcwd(), "hammer_pose.yaml")
    else:
        yaml_path = os.path.join(os.getcwd(), f"{os.path.basename(os.path.normpath(output_dir))}.yaml")

    if mode == 5:
        yaml_content = f"""# YOLO-Pose 5-Keypoint Hammer Dataset (Flip Detection)
path: {abs_output_dir}
train: images/train
val: images/val

# 5 Keypoints: [Face, Claw, HeadCenter, Grip, Tail]
kpt_shape: [5, 3]
flip_idx: []

names:
  0: hammer
"""
    else:
        yaml_content = f"""# YOLO-Pose 2-Keypoint Hammer Dataset
path: {abs_output_dir}
train: images/train
val: images/val

# 2 Keypoints: [HeadCenter, Tail]
kpt_shape: [2, 3]
flip_idx: []

names:
  0: hammer
"""

    with open(yaml_path, "w") as yf:
        yf.write(yaml_content)

    # 8. Print Executive Summary
    train_count = len(glob.glob(f"{output_dir}/images/train/*.jpg"))
    val_count = len(glob.glob(f"{output_dir}/images/val/*.jpg"))

    print("\n" + "=" * 70)
    print("   AUTO-LABELING PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    print(f"  Total Video Frames Read     : {frame_idx}")
    print(f"  Frames Filtered (Blurry)    : {blurred_count}")
    print(f"  Frames Filtered (Mask Miss) : {rejected_mask_count}")
    print(f"  Successfully Labeled Frames : {saved_count}")
    print(f"  Training Split (Images)     : {train_count}")
    print(f"  Validation Split (Images)   : {val_count}")
    print(f"  Dataset Directory           : {abs_output_dir}")
    print(f"  YAML Configuration File     : {yaml_path}")
    print(f"  Visual Verification Samples : {preview_dir}/ ({preview_count} images)")
    print(f"  Total Elapsed Time          : {elapsed_total:.1f} seconds")
    print("=" * 70 + "\n")

    return yaml_path, saved_count


# ==============================================================================
# 5. Direct Training Integration (Optional)
# ==============================================================================

def train_yolo_model(yaml_path, epochs=60, batch=16, device=0, weights="yolov8n-pose.pt"):
    """
    Launches YOLOv8-pose training directly using the auto-labeled dataset.
    Configured with rotation-invariant augmentations specifically for top-down tools.
    """
    print("\n" + "=" * 70)
    print("   LAUNCHING YOLOV8-POSE TRAINING")
    print("=" * 70)
    print(f"[INFO] Base Weights   : {weights}")
    print(f"[INFO] Dataset Config : {yaml_path}")
    print(f"[INFO] Epochs         : {epochs}")
    print(f"[INFO] Batch Size     : {batch}")
    print(f"[INFO] Device         : {device}")

    from ultralytics import YOLO
    model = YOLO(weights)

    train_results = model.train(
        data=yaml_path,
        epochs=epochs,
        imgsz=640,
        batch=batch,
        device=device,
        workers=4,
        degrees=180.0,  # Full 360-degree rotation invariance
        fliplr=0.0,     # Preserves Face vs. Claw asymmetry
        flipud=0.0,
        scale=0.3,
        save=True,
        project="runs_auto",
        name="hammer_model_best",
        plots=True
    )

    best_pt = os.path.join("runs_auto", "hammer_model_best", "weights", "best.pt")
    print("\n" + "=" * 70)
    print(f"[DONE] Training finished! Best weights saved to: {best_pt}")
    print("=" * 70)
    return best_pt


# ==============================================================================
# 6. Main CLI Interface
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Unified Auto-Labeling Pipeline for Top-Down Tool Pose Estimation (URC 2027)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--video", type=str, default="hammer_video.mp4",
                        help="Path to input video file")
    parser.add_argument("--output_dir", type=str, default="dataset_flip",
                        help="Output directory to store images and YOLO labels")
    parser.add_argument("--mode", type=int, choices=[2, 5], default=5,
                        help="Keypoint mode: 5 (Face, Claw, HeadCenter, Grip, Tail) or 2 (Head, Tail)")
    parser.add_argument("--step", type=int, default=8,
                        help="Frame sampling step (e.g. 8 extracts every 8th frame)")
    parser.add_argument("--blur_thresh", type=float, default=8.0,
                        help="Laplacian variance threshold for sharpness filtering")
    parser.add_argument("--val_ratio", type=float, default=0.20,
                        help="Validation set fraction (e.g. 0.20 = 80/20 train/val split)")
    parser.add_argument("--previews", type=int, default=30,
                        help="Number of visual verification preview images to render")
    parser.add_argument("--fastsam", type=str, default="FastSAM-s.pt",
                        help="Path to FastSAM model checkpoint")
    parser.add_argument("--conf", type=float, default=0.35,
                        help="FastSAM segmentation confidence threshold")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Compute device ('cuda', 'cpu', or GPU index like '0')")
    parser.add_argument("--yaml", type=str, default=None,
                        help="Custom path or filename for the generated dataset YAML file")
    parser.add_argument("--no_clean", action="store_true",
                        help="Do not clean existing output directory before generating")

    # Training flags
    parser.add_argument("--train", action="store_true",
                        help="Automatically start YOLOv8-pose training after dataset generation")
    parser.add_argument("--epochs", type=int, default=60,
                        help="Training epochs if --train is set")
    parser.add_argument("--batch", type=int, default=16,
                        help="Batch size if --train is set")
    parser.add_argument("--weights", type=str, default="yolov8n-pose.pt",
                        help="Pre-trained base weights if --train is set")

    args = parser.parse_args()

    # Run auto-labeling
    yaml_path, count = run_auto_label_pipeline(
        video_path=args.video,
        output_dir=args.output_dir,
        mode=args.mode,
        step=args.step,
        blur_threshold=args.blur_thresh,
        val_ratio=args.val_ratio,
        num_previews=args.previews,
        fastsam_weights=args.fastsam,
        conf_thresh=args.conf,
        device=args.device,
        clean_existing=(not args.no_clean),
        custom_yaml=args.yaml
    )

    # Optional training
    if args.train:
        dev = int(args.device) if args.device.isdigit() else args.device
        train_yolo_model(
            yaml_path=yaml_path,
            epochs=args.epochs,
            batch=args.batch,
            device=dev,
            weights=args.weights
        )


if __name__ == "__main__":
    main()
