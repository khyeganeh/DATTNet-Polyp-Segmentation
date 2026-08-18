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

## Reference

This project is an implementation and adaptation of the DATTNet architecture proposed by Zhang et al. for polyp segmentation using the Kvasir-SEG dataset.

**Original Paper:**

M. Zhang, Y. Zhang, S. Liu, Y. Han, H. Cao, and B. Qiao,  
"Dual-attention transformer-based hybrid network for multi-modal medical image segmentation,"  
Scientific Reports, 2024.

DOI: https://doi.org/10.1038/s41598-024-76234-y

**Original DATTNet Repository:**  
https://github.com/MhZhang123/DATTNet

### Acknowledgement

The architecture of this project is based on the DATTNet model proposed by Zhang et al. The original paper and official implementation are acknowledged and cited above.

This repository contains my implementation and adaptation of DATTNet for polyp segmentation using the Kvasir-SEG dataset, including training, evaluation, and inference experiments.