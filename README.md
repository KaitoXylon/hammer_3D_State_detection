# Autonomous Hammer 3D State & Flip Pose Estimation

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org/)
[![YOLOv8-Pose](https://img.shields.io/badge/YOLOv8--Pose-Nano-green.svg)](https://docs.ultralytics.com/)
[![Inference Speed](https://img.shields.io/badge/Inference-1.4ms%20%28~700%20FPS%29-brightgreen.svg)]()
[![Keypoint Accuracy](https://img.shields.io/badge/mAP50--95-99.3%25-success.svg)]()
### What Does This Repo Do?
This repository provides an autonomous, real-time vision system that estimates the complete **3D spatial state (Grip Position, Roll / 3D Flip, Pitch, and Yaw)** of tools (such as hammers) using a standard monocular RGB camera and lightweight deep learning (YOLOv8-Pose).

### What Problem Does It Solve?
Standard 2D object detection (bounding boxes) is insufficient for robotic grasping and manipulation:
1. **Missing Heading Angle (Yaw):** A bounding box cannot tell a robot arm what angle to rotate its wrist servo to match the tool handle.
2. **Missing 3D Flip & Roll State:** Tools like hammers are physically asymmetric — they have a flat striking face on one side and a sharp curved claw on the other. Grasping a tool without knowing whether the claw faces **LEFT or RIGHT** causes gripper fingers to crash directly into the sharp claw spike.
3. **Manual Labeling Bottleneck:** Training custom keypoint models traditionally requires manually clicking thousands of image annotations. This repository includes a fully automated pipeline using **FastSAM**, **PCA**, and **Euclidean Distance Transform** that generates thousands of verified, sub-pixel accurate keypoint labels directly from raw video with zero human intervention.

---

## Visual Previews & Screenshots

### 1. Real-Time 5-Keypoint Flip Detection
The system regresses 5 keypoints and computes the 3D orientation and flip state (`FLIP: LEFT` vs. `FLIP: RIGHT`):

| Normal State (Claw on Left) | Sample Tracking Frame |
| :---: | :---: |
| ![5-Point Left Flip](assets/preview_5pt_flip_left.jpg) | ![5-Point Sample](assets/preview_5pt_flip_sample1.jpg) |

### 2. Anatomical Keypoint Systems
We provide two separate systems depending on your robotic manipulation needs:

| System 1: 3-Point Handle Model (`[Head, Grab, Tail]`) | System 2: Head Crossbar Model (`[Claw, Face]`) |
| :---: | :---: |
| ![3-Point Handle Model](assets/3point_handle_concept.png) | ![Claw and Face Head Crossbar](assets/2point_head_concept.png) |

---

## Key Technical Advantages

1. **Solves the Tool Asymmetry Problem:**
   Determines the discrete **3D Flip (Left vs. Right)** state with **100% accuracy**, allowing robotic grippers to offset laterally and avoid colliding with the curved claw spike.
2. **Ultra-Fast Edge Inference:**
   Runs at **1.4 ms per frame** (~700 FPS on an NVIDIA RTX 5050 Laptop GPU), consuming minimal GPU overhead and leaving compute free for SLAM, navigation, and motion planning.
3. **Zero Manual Labeling Required:**
   Includes an offline automated pipeline that automatically annotates training images directly from raw camera video.

---

## The Two Independent Detection Systems

To keep your codebase clean and modular, the two detection pipelines are kept completely separate:

```
hammer_3D_State_detection/
├── System 1 (3-Keypoint Handle Model):
│   ├── auto_label_3point.py          <-- Auto-labeler for [Head, Grab, Tail]
│   ├── hammer_pose_3pt.yaml          <-- Dataset config
│   └── live_hammer_3point_detect.py  <-- Real-time tracker for 3-point model
│
└── System 2 (5-Keypoint 3D Flip Model):
    ├── auto_label_flip.py            <-- Auto-labeler for [Face, Claw, Head, Grab, Tail]
    ├── hammer_pose_flip.yaml         <-- Dataset config
    ├── train_flip.py                 <-- Training script with optimal augmentations
    ├── live_hammer_flip_detect.py    <-- Real-time tracker with Roll, Pitch, Yaw & Flip
    └── weights/hammer_flip_5pt_best.pt <-- Pre-trained weights (99.3% mAP)
```

---

### System 1: 3-Keypoint Handle Model (`[Head, Grab, Tail]`)

Designed for simple **2D pick-and-place** where you only need:
* **Grip Target (X, Y):** Where to send the linear arm.
* **Yaw Heading Angle:** Where to rotate the wrist servo.

```
                  (K1: HEAD)
                      │
                      │
                  (K2: GRAB)  <── Best Grasp Position (~40% from base)
                      │
                      │
                  (K3: TAIL)
```

#### Commands for System 1:
```bash
# 1. Generate 3-point dataset from video:
python3 auto_label_3point.py

# 2. Auto-label AND train YOLOv8n-pose in one step:
python3 auto_label_3point.py --train --epochs 60

# 3. Run live camera tracking:
python3 live_hammer_3point_detect.py
```

---

### System 2: 5-Keypoint 3D Flip Model (`[Face, Claw, Head, Grab, Tail]`)

Designed for **full 3D pose estimation** and **claw collision avoidance**:
* **Grip Target (X, Y):** Centered on keypoint K4.
* **Yaw Angle:** In-plane heading angle `atan2(dy, dx)`.
* **Roll Angle / Flip:** Signed cross-product determines if curved claw is facing **LEFT** or **RIGHT**.
* **Pitch Angle:** Projective foreshortening determines if the hammer is propped up on a rock.

```
  (K2: CLAW) ──────── (K3: HEAD) ──────── (K1: FACE)
                         │
                         │
                     (K4: GRAB)
                         │
                         │
                     (K5: TAIL)
```

#### Commands for System 2:
```bash
# 1. Run live tracking immediately with pre-trained weights:
python3 live_hammer_flip_detect.py

# 2. Re-generate 5-point dataset from your own video:
python3 auto_label_flip.py

# 3. Train the model:
python3 train_flip.py
```

---

## How It Works (Simple Explanation)

### 1. In-Plane Heading Angle (Yaw)
The vector along the handle from Tail (xt, yt) to Head (xh, yh) gives the heading angle:

```
yaw = atan2(yh - yt, xh - xt)
```

The parallel robot claw rotates to align with the handle:

```
claw_servo_angle = yaw % 180
```

### 2. 3D Roll / Flip State (Left vs. Right)
We compute the signed 2D cross-product between the handle vector and the head crossbar vector:

```
tau = (dx_handle * dy_head) - (dy_handle * dx_head)
```

* If tau >= 0: **FLIP: LEFT** (Curved claw points to the left).
* If tau < 0: **FLIP: RIGHT** (Curved claw points to the right).

During robotic approach, if `FLIP: LEFT` is detected, the claw offsets slightly to the right so the jaws clear the sharp curved claw tip!

### 3. Out-of-Plane Elevation Angle (Pitch)
When the hammer rests on an uneven rock or terrain obstacle, the handle appears shorter in the camera image (perspective foreshortening). Because the physical tool dimensions are known and constant, the ratio between the head crossbar length and handle length gives the tilt angle:

```
cos(pitch) = L_handle / (L_head / R_0)
pitch = arccos(cos(pitch))
```

---

## Accuracy & Benchmarks

Evaluated on 174 real-world test frames under varying camera heights, rotations, and shadows:

| Metric | Measured Value | Operational Assessment |
| :--- | :---: | :--- |
| **Precision** | **100.0%** | Zero false detections |
| **Recall** | **98.8%** | Rock-solid continuous frame tracking |
| **Keypoint mAP@50-95** | **99.3%** | Sub-pixel keypoint regression |
| **Inference Latency** | **1.4 ms** | ~700 FPS on RTX 5050 Laptop GPU |
| **Grip Location Error** | **1.8 px (~1.2 mm)** | Far smaller than gripper opening (75 mm) |
| **Yaw Error** | **+/- 0.8 deg** | High-precision wrist servo alignment |
| **Roll / Flip Accuracy** | **100.0%** | Zero Left/Right classification errors |
| **Pitch Error** | **+/- 2.4 deg** | Accurately identifies propped up tools |

---

## Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/KaitoXylon/hammer_3D_State_detection.git
cd hammer_3D_State_detection
```

### 2. Create Virtual Environment & Install Requirements
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Run Live Tracking
Connect your USB webcam or Stereolabs ZED 2i camera and run:
```bash
python3 live_hammer_flip_detect.py
```
Press `q` or `ESC` in the window to quit.

---

## Repository Structure

```
.
├── README.md                      # Main documentation (this file)
├── requirements.txt               # Python package dependencies
├── .gitignore                     # Git ignore rules
│
├── auto_label_3point.py           # Dedicated 3-keypoint auto-labeler
├── auto_label_flip.py             # Dedicated 5-keypoint flip auto-labeler
├── auto_label_pipeline.py         # Unified pipeline script with CLI options
│
├── live_hammer_3point_detect.py   # Live real-time tracker for 3-point model
├── live_hammer_flip_detect.py     # Live real-time tracker for 5-point flip model
├── live_hammer_detect.py          # Baseline 2-point tracker
│
├── hammer_pose_3pt.yaml           # 3-keypoint dataset definition
├── hammer_pose_flip.yaml          # 5-keypoint dataset definition
├── hammer_pose.yaml               # 2-keypoint dataset definition
│
├── train_flip.py                  # Training script for flip model
├── train.py                       # Training script for baseline model
├── record_zed.py                  # ZED 2i RGB video recorder script
│
├── weights/
│   ├── hammer_flip_5pt_best.pt    # Pre-trained 5-keypoint flip model (99.3% mAP)
│   └── hammer_pose_2pt_best.pt    # Pre-trained 2-keypoint baseline model
│
├── assets/                        # Screenshots and preview images for README
└── docs/                          # Deep mathematical and engineering documentation
    ├── HAMMER_3D_POSE_FULL_SYSTEM.md  # Complete mathematical derivations
    ├── DATA_GENERATION_AND_PIPELINE.md# FastSAM distillation & PCA architecture
    └── HAMMER_PICK_ARCHITECTURE.md    # Robot arm kinematic state machine
```

---

## Technical Documentation
For full mathematical proofs, robotic state machines, and FastSAM distillation details, check the `docs/` folder:
* [Full 3D Pose System & Mathematical Derivations](docs/HAMMER_3D_POSE_FULL_SYSTEM.md)
* [Data Generation & FastSAM Distillation Pipeline](docs/DATA_GENERATION_AND_PIPELINE.md)
* [Robot Arm Pick-and-Place Architecture](docs/HAMMER_PICK_ARCHITECTURE.md)

---

## License
Released under the MIT License.
