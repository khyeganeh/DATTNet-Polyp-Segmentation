# DATTNet for Polyp Segmentation

## Overview

This project implements DATTNet for automatic polyp segmentation on the Kvasir-SEG dataset using PyTorch.

## Features

- DATTNet Architecture
- BCE + Dice Loss
- Training & Validation
- Inference
- Evaluation
- Dice Score
- IoU Score
- Precision
- Recall
- F1 Score
- Accuracy
- Visualization
- Training History

---

## Dataset

Kvasir-SEG

1000 colonoscopy images with expert annotations.

---

## Training

```bash
python train.py
```

## Evaluation

```bash
python evaluate.py
```

## Inference

```bash
python inference.py
```

## Comparison

```bash
python compare.py
```

## Plot Training Curve

```bash
python plot.py
```

---

## Results

| Metric | Score |
|--------|-------|
| Dice | 0.8808 |
| IoU | 0.8118 |

---

## Project Structure

```
DATTNet/
│
├── models/
├── datasets/
├── checkpoints/
├── results/
├── train.py
├── evaluate.py
├── inference.py
├── compare.py
├── plot.py
├── dataset.py
├── metrics.py
├── loss.py
├── history.csv
├── requirements.txt
└── README.md
```

## Framework

- Python
- PyTorch
- OpenCV
- NumPy
- Matplotlib