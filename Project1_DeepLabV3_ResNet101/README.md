# Off-Road Segmentation - DeepLabV3+ Optimized

This project implements a high-performance semantic segmentation model for off-road environments using DeepLabV3+ with a ResNet101 encoder.

## Key Features
- **Model**: DeepLabV3+ (segmentation-models-pytorch)
- **Encoder**: ResNet101 (Pretrained on ImageNet)
- **Optimization**:
  - Mixed Precision Training (AMP) for 2x faster training.
  - AdamW Optimizer with Weight Decay (1e-4).
  - Combined Loss: CrossEntropy + Dice + Focal.
  - Learning Rate Scheduling: ReduceLROnPlateau.
  - Early Stopping based on Validation Dice Score.
- **Data Augmentation**: Robust pipeline using Albumentations (Flip, Rotate, Blur, Dropout, etc.).
- **Metrics**: Real-time tracking of Loss, Dice Score, and IoU.

## 📂 Project Structure
- `app.py`: Interactive Streamlit dashboard for real-time inference.
- `train.py`: Main training script with AMP, early stopping, and class weighting.
- `evaluate.py`: Quantitative performance evaluation script (Mean Dice/IoU).
- `inference.py`: Batch script for qualitative visualization on test images.
- `compute_weights.py`: Statistical analysis tool for resolving class imbalance.
- `check_leakage.py`: Dataset integrity verification script.
- `dataset/`: Custom data loading and advanced augmentation logic.
- `models/`: DeepLabV3+ model architecture definitions.
- `losses/`: Optimized loss functions (Tversky, Focal, Weighted CE).

## 🚀 Interactive Dashboard
A Streamlit dashboard is included for real-time inference and visualization:

![Dashboard Screenshot](assets/dashboard.png)

```bash
# Run the dashboard
streamlit run app.py
```

**Dashboard Features:**
- Upload off-road images for instant segmentation.
- Adjustable confidence threshold slider.
- Real-time visualization of original image, predicted mask, and overlay.
- Class distribution statistics and color-coded legend.

---

## 🛠 How to Run the Project

### 1️⃣ Environment Setup
**Python version required:** Python 3.10

```bash
# Create and activate environment
python -m venv venv
.\venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2️⃣ Dataset Preparation
Due to GitHub size limits, the dataset is not included. Ensure your `data/` folder follows this structure:
```text
data/
├── train/
│   ├── Color_Images/
│   └── Segmentation/
├── val/
└── testImages/
```

### 3️⃣ Training the Model
To initiate retraining with optimized class weights and Tversky loss:
```bash
python train.py
```
- **Saved model:** `checkpoints/best_model.pth`

### 4️⃣ Running Inference (Visualization)
To generate batch results on test images:
```bash
python inference.py
```
**Output directory:** `inference_results/`

### 5️⃣ Evaluating Metrics
To compute final validation performance:
```bash
python evaluate.py
```

---

## 📈 Results
- **Training Duration**: 14 Epochs (Early Stopping)
- **Best Validation Dice Score**: 0.60
- **Primary Optimization**: Mitigated Class Collapse using Tversky Loss and inverse-frequency weighting.
