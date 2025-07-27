# 🩷 CRACKPINK: Road Defect Detection with YOLOv12 🩷

## 🧠 YOLO Object Detection - Performance Analysis Report

This repository presents the performance evaluation of a YOLOv12 object detection model trained on custom-labeled road surface data for identifying various crack types.

---

## 📁 Experimental Setup

### 📌 Model Configuration
- **Model**: YOLOv12 (Extra-Large variant)
- **Input Size**: 640 × 640 pixels
- **Epochs**: 150
- **Optimizer**: SGD (Stochastic Gradient Descent)
- **Learning Rate**: 0.001
- **Confidence Threshold**: 0.10
- **IoU Threshold**: 0.15

---

### 🧪 Data Augmentation
To enhance robustness and generalization, the following augmentation techniques were applied:
- Random **scaling**
- **Shearing** transformations
- **Gaussian noise** addition
- **Random cropping**
- **Rotation** (±10°)

---

### 🧬 Ensemble Learning
- **Status**: ❌ Not applied

---

## 📸 Sample Predictions

> ✅ Below is a sample output from the best-performing YOLOv12 model.

> *(Insert image or provide a link here)*

---

## 📊 Observations & Insights

### 🔍 Key Hyperparameter Findings
- **IoU Threshold (0.15)** provided a balanced trade-off between precision and recall.
- **150 training epochs** allowed for sufficient convergence without overfitting.
- Proper **train/validation split** significantly influenced model generalization.

### 🚀 YOLOv12 Advantages
Compared to previous versions, YOLOv12 showed improved performance due to:
- Enhanced **feature extraction** capability
- Refined **neck architecture** enabling better multi-scale feature fusion
- **Anchor-free detection head**: reduced computational load with high accuracy

### 🎯 Augmentation Impact
The augmentation strategy notably boosted model robustness:
- Enabled the model to generalize across different lighting and geometric distortions
- Improved detection stability in real-world road conditions

---

## 🗒️ Notes
- This report summarizes results **without** any ensemble techniques.
- Future directions include:
  - Applying **test-time augmentation**
  - Exploring **model ensembling** to further enhance performance

---

## 📂 Directory Structure (Optional)

