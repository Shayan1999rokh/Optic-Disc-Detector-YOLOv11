# Optic Disk Detector using YOLOv11

## Overview

This project presents an **Optic Disk Detection system** built using Ultralytics YOLO and the lightweight YOLO11 architecture. The model is trained to automatically localize optic disk and optic cup regions in retinal fundus images using bounding box annotations in YOLO format.

The project includes:

* Dataset analysis and visualization
* YOLO annotation parsing
* Training with YOLOv11
* Evaluation using confusion matrices and precision-recall curves
* Inference on unseen test images
* Visualization of prediction results

This implementation is suitable for:

* Medical image analysis research
* Computer vision projects
* Ophthalmology-related AI systems
* Retinal disease screening pipelines

---

# Features

* ✅ YOLOv11-based object detector
* ✅ Support for train / validation / test splits
* ✅ Bounding box visualization
* ✅ Dataset imbalance analysis
* ✅ Multi-GPU training support
* ✅ Automatic learning curve generation
* ✅ Inference visualization pipeline
* ✅ Easy integration with custom datasets

---

# Project Structure

```bash
project/
│
├── dataset/
│   ├── dataset.yaml
│   │
│   ├── images/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   │
│   └── labels/
│       ├── train/
│       ├── val/
│       └── test/
│
├── runs/
│   └── detect/
│       └── train/
│           ├── confusion_matrix.png
│           ├── results.png
│           ├── PR_curve.png
│           ├── P_curve.png
│           └── weights/
│
├── train.py
├── inference.py
└── README.md
```

---

# Installation

## Clone Repository

```bash
git clone <repository-url>
cd <repository-name>
```

---

## Install Dependencies

Install the required libraries:

```bash
pip install ultralytics
```

Additional dependencies:

```bash
pip install opencv-python matplotlib pillow pyyaml torchvision
```

---

# Model Initialization

Load the pretrained YOLOv11 nano model:

```python
from ultralytics import YOLO

model = YOLO("yolo11n.pt")
```

The pretrained weights are automatically downloaded from the official [Ultralytics Website](https://ultralytics.com?utm_source=chatgpt.com) if they do not already exist.

---

# Dataset Format

The dataset must follow YOLO detection format.

## Directory Structure

```bash
dataset/
│
├── images/
│   ├── train/
│   ├── val/
│   └── test/
│
└── labels/
    ├── train/
    ├── val/
    └── test/
```

---

## Annotation Format

Each label file should contain:

```txt
class_id x_center y_center width height
```

Example:

```txt
0 0.532 0.418 0.210 0.185
```

Where:

* `class_id` → object category index
* `x_center` → normalized center x-coordinate
* `y_center` → normalized center y-coordinate
* `width` → normalized bounding box width
* `height` → normalized bounding box height

---

# Dataset Configuration

Example `dataset.yaml`:

```yaml
path: /dataset

train: images/train
val: images/val
test: images/test

names:
  0: optic_disk
  1: optic_cup
```

---

# Dataset Analysis and Visualization

The project includes a custom `Visualization` class for:

* Reading YOLO annotations
* Plotting bounding boxes
* Displaying image samples
* Analyzing class distribution

## Example Usage

```python
vis = Visualization(
    root=root,
    data_types=["train", "val", "test"],
    n_ims=20,
    rows=5,
    cmap="rgb"
)

vis.analysis()
vis.visualization()
```

---

# Training

## Training Configuration

```python
train_results = model.train(
    data=f"{root}/dataset.yaml",
    epochs=10,
    imgsz=480,
    device=[0, 1]
)
```

---

## Training Parameters

| Parameter | Description                   |
| --------- | ----------------------------- |
| `data`    | Dataset YAML file             |
| `epochs`  | Number of training epochs     |
| `imgsz`   | Input image size              |
| `device`  | GPU devices used for training |

---

# Training Output

After training, YOLO automatically generates:

* Confusion matrix
* Precision curve
* Recall curve
* PR curve
* Validation prediction samples
* Model weights

Output directory:

```bash
runs/detect/train/
```

---

# Evaluation Results

Generated evaluation files include:

| File                   | Description                           |
| ---------------------- | ------------------------------------- |
| `confusion_matrix.png` | Classification confusion matrix       |
| `results.png`          | Training & validation learning curves |
| `P_curve.png`          | Precision curve                       |
| `PR_curve.png`         | Precision-Recall curve                |
| `val_batch0_pred.jpg`  | Validation prediction visualization   |

---

# Inference

Run inference on test images:

```python
inference_results = model(
    f"{root}/images/test",
    device=0,
    verbose=False
)
```

---

# Inference Visualization

The project includes a helper function for visualizing predictions with bounding boxes.

## Example

```python
inference_vis(
    results=inference_results,
    num_images=15,
    rows=3
)
```

Bounding boxes are displayed on retinal fundus images using OpenCV.

---

# Sample Workflow

## 1. Install Dependencies

```bash
pip install ultralytics
```

---

## 2. Load Model

```python
from ultralytics import YOLO

model = YOLO("yolo11n.pt")
```

---

## 3. Train Model

```python
model.train(
    data="dataset.yaml",
    epochs=10,
    imgsz=480
)
```

---

## 4. Run Inference

```python
results = model("test_images/")
```

---

## 5. Visualize Predictions

```python
inference_vis(results, num_images=10, rows=2)
```

---

# Model Architecture

This project uses:

* YOLO11 Nano (`yolo11n.pt`)
* Pretrained weights from COCO dataset
* Transfer learning for retinal object detection

YOLO11n characteristics:

* Lightweight architecture
* Fast inference speed
* Low computational cost
* Suitable for edge devices and medical applications

---

# Multi-GPU Training

Training supports multiple GPUs using Distributed Data Parallel (DDP):

```python
device=[0, 1]
```

This significantly accelerates training on large datasets.

---

# Applications

This project can be extended for:

* Glaucoma screening
* Optic cup-to-disc ratio estimation
* Retinal abnormality localization
* Ophthalmology diagnostic assistance
* Medical AI research

---

# Future Improvements

Potential enhancements include:

* Segmentation-based optic disk localization
* Attention mechanisms (CBAM, SE blocks)
* Ensemble learning
* Hyperparameter optimization
* YOLO11s / YOLO11m experiments
* Cross-validation evaluation
* Explainable AI (Grad-CAM)

---

# Requirements

## Python Version

```bash
Python >= 3.10
```

---

## Main Libraries

* ultralytics
* torch
* torchvision
* opencv-python
* matplotlib
* pillow
* numpy
* pyyaml

---

# Conclusion

This project demonstrates an end-to-end pipeline for **optic disk detection using YOLOv11**, including:

* Data preprocessing
* Annotation visualization
* Model training
* Evaluation
* Inference


---
# ReadMe is generated by GPT. Check the important info
The lightweight YOLO11 architecture enables efficient and accurate retinal object detection, making the system suitable for both research and real-world medical imaging applications.
