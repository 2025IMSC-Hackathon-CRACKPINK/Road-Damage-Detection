# 🧠 YOLO Object Detection - Performance Analysis Report

This repository documents the performance evaluation of a YOLOv12 object detection model trained on custom road surface data.

---

## 📁 Experimental Setup

### 📌 Model Configuration
- **YOLO Version**: YOLOv12 (Extra-Large variant)
- **Input Image Size**: 640 × 640 pixels
- **Epochs**: 150
- **Optimizer**: SGD (Stochastic Gradient Descent)
- **Learning Rate**: 0.001
- **Confidence Threshold**: 0.10
- **IoU Threshold**: 0.15

---

### 🧪 Data Augmentation
Data augmentation was enabled to improve generalization and robustness. Techniques used include:
- Random **scaling**
- **Shearing** operations
- **Gaussian noise** injection
- **Random cropping**
- **Rotation** (±10 degrees)

---

### 🧬 Ensemble Learning
- ❌ **Not applied** in this experiment

---

## 📸 Sample Predictions
> ✅ The image below illustrates sample detection outputs from the best-performing YOLOv12 model.

> *(Insert sample image or link here)*

---

## 📊 Observations & Insights

### 🔍 Key Hyperparameters
- **IoU Threshold (0.15)** offered a good trade-off between **Precision** and **Recall**.
- **Training Epochs (150)** helped achieve convergence without significant overfitting.
- **Train/Validation Split Ratio** affected model generalization notably.

### 🚀 YOLOv12 Advantages
YOLOv12 outperformed previous versions thanks to:
- Advanced **feature extraction** layers
- Improved **neck architecture** for multiscale feature fusion
- Optimized **anchor-free detection head** reducing complexity while maintaining accuracy

### 🎯 Data Augmentation Benefits
The aggressive augmentation pipeline played a critical role by:
- Improving detection robustness across various lighting and geometric conditions
- Simulating diverse real-world environments for better generalization

---

## 📌 Notes
- This report summarizes training conducted without ensemble strategies.
- Future work may explore test-time augmentation and ensemble methods for further performance gains.

---

## 📂 Directory Structure (optional)
