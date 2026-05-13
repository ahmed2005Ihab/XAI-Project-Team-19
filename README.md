# XAI Final Phase — Pneumonia Classification with Explainable AI

**Author:** Habiba Ayman (202202088 · Team 19)

This project applies three deep learning models to chest X-ray pneumonia classification, each paired with multiple Explainable AI (XAI) techniques. The pipeline is split across four Jupyter notebooks.

---

## Project Structure

```
XAI Final phase Habiba/
├── preprocessing/
│   └── preprocessing/
│       └── habiba-ayman-202202088-team-19-preprocessing-eda (1).ipynb   # Notebook 1
│
├── XG-Boost/
│   └── XG-Boost/
│       └── habiba-ayman-202202088-team-19-model-1-xgboost (1).ipynb     # Notebook 2
│
├── Mobinet/
│   └── Mobinet/
│       └── model-2-habiba-ayman-202202088-team-19 (2).ipynb             # Notebook 3
│
└── YOLOV11/
    └── model-3-habiba-ayman-yolov11.ipynb                               # Notebook 4
```

---

## Datasets

All notebooks download their data automatically via `kagglehub`. You will need:

| Dataset | Source | Used By |
|---|---|---|
| RSNA Pneumonia Detection Challenge | Kaggle competition: `rsna-pneumonia-detection-challenge` | All notebooks |
| COVID-19 Radiography Database | Kaggle dataset (MobileNet notebook) | Notebook 3 (MobileNetV2) |

**Before running,** make sure your Kaggle API credentials are configured:

```bash
# Place your kaggle.json in the default location
mkdir -p ~/.kaggle
cp kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

---

## Environment Setup

### Option A — Kaggle (Recommended)

Upload each notebook directly to [Kaggle](https://www.kaggle.com/) and run it there. The datasets will already be available, and GPU acceleration is included for free.

### Option B — Local / Colab

**Python version:** 3.9 or higher

Install shared dependencies:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn opencv-python pydicom kagglehub tqdm albumentations
```

Then install notebook-specific packages (see per-notebook sections below).

**GPU note:** TensorFlow notebooks (Notebooks 2 & 3) benefit significantly from a CUDA-compatible GPU. The YOLOv11 notebook (Notebook 4) uses PyTorch and also benefits from GPU.

---

## Notebooks — How to Run

### Notebook 1 — Preprocessing & EDA

**File:** `preprocessing/preprocessing/habiba-ayman-202202088-team-19-preprocessing-eda (1).ipynb`

**What it does:** Downloads the RSNA dataset, performs exploratory data analysis (class distribution, bounding box analysis, pixel intensity), engineers features, and produces a balanced 12,000-patient sample split into train/validation sets.

**Extra dependencies:**

```bash
pip install pydicom kagglehub
```

**Run order:** Run this notebook **first**. It lays the groundwork for understanding the data used by the model notebooks.

**Key outputs:** EDA plots saved to `New folder/` (bounding box heatmap, correlation matrix, feature distributions, sample X-ray images).

---

### Notebook 2 — CNN + XGBoost (Model 1)

**File:** `XG-Boost/XG-Boost/habiba-ayman-202202088-team-19-model-1-xgboost (1).ipynb`

**What it does:** Trains a custom CNN to extract features from chest X-rays, then feeds those features into an XGBoost classifier. Applies SHAP, LIME, GradCAM, and Guided GradCAM for explainability.

**Extra dependencies:**

```bash
pip install xgboost shap lime grad-cam albumentations kagglehub tensorflow
```

**Key hyperparameters (Cell 3):**

| Parameter | Default | Description |
|---|---|---|
| `IMG_SIZE` | 64 | Input image size (px) |
| `N_NORMAL` | 6000 | Normal samples |
| `N_PNEUMONIA` | 6000 | Pneumonia samples |
| `BATCH_SIZE` | 128 | Training batch size |
| `EPOCHS` | 40 | Max training epochs |
| `LEARNING_RATE` | 0.0001 | Adam learning rate |

**Key outputs:** Confusion matrix, SHAP summary plot, LIME explanation, GradCAM + Guided GradCAM overlays, CNN training curves — all saved to `out.img/`.

---

### Notebook 3 — MobileNetV2 (Model 2)

**File:** `Mobinet/Mobinet/model-2-habiba-ayman-202202088-team-19 (2).ipynb`

**What it does:** Trains a MobileNetV2 model (no transfer learning, trained from scratch) on a three-class problem: Normal, Pneumonia, and COVID-19. Applies PDP, LIME, Integrated Gradients, and Occlusion Sensitivity for explainability.

**Extra dependencies:**

```bash
pip install shap lime albumentations kagglehub tensorflow
```

**Key hyperparameters (Cell 3):**

| Parameter | Default | Description |
|---|---|---|
| `IMG_SIZE` | 224 | Input image size (px) — MobileNetV2 default |
| `N_NORMAL` | 949 | Normal samples (matches paper) |
| `N_PNEUMONIA` | 2563 | Pneumonia samples (matches paper) |
| `N_COVID` | 2169 | COVID-19 samples (matches paper) |
| `BATCH_SIZE` | 32 | Training batch size |

**Datasets required:** This notebook downloads both the RSNA dataset (Normal + Pneumonia) and the COVID-19 Radiography Database (COVID images) automatically via `kagglehub`.

**Key outputs:** Training curves, confusion matrix, PDP (XAI-1), LIME (XAI-2), Integrated Gradients (XAI-3), Occlusion Sensitivity (XAI-4) — all saved to `img.out/`.

---

### Notebook 4 — YOLOv11 (Model 3)

**File:** `YOLOV11/model-3-habiba-ayman-yolov11.ipynb`

**What it does:** Uses a YOLOv11 backbone (via Ultralytics) for feature extraction, paired with a classification head. Applies SHAP, LIME, GradCAM, and Guided GradCAM for explainability.

**Extra dependencies:**

```bash
pip install ultralytics shap lime grad-cam albumentations kagglehub tensorflow
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

> For GPU support, replace `cpu` with `cu118` (or your CUDA version) in the PyTorch install URL.

**Key hyperparameters (Cell 3):**

| Parameter | Default | Description |
|---|---|---|
| `IMG_SIZE` | 64 | Input image size (px) |
| `N_NORMAL` | 6000 | Normal samples |
| `N_PNEUMONIA` | 6000 | Pneumonia samples |
| `BATCH_SIZE` | 128 | Training batch size |
| `EPOCHS` | 40 | Max training epochs |

**Key outputs:** Training curves, confusion matrix, SHAP, LIME, GradCAM + Guided GradCAM — all saved to `img.out/`.

---

## Recommended Run Order

```
Notebook 1 (Preprocessing & EDA)
        ↓
Notebook 2 (CNN + XGBoost)
        ↓
Notebook 3 (MobileNetV2)
        ↓
Notebook 4 (YOLOv11)
```

Notebooks 2, 3, and 4 each download the dataset independently, so they can also be run in any order after Notebook 1.

---

## XAI Methods Summary

| Method | Notebook 2 (XGBoost) | Notebook 3 (MobileNetV2) | Notebook 4 (YOLOv11) |
|---|:---:|:---:|:---:|
| SHAP | ✓ | | ✓ |
| LIME | ✓ | ✓ | ✓ |
| GradCAM | ✓ | | ✓ |
| Guided GradCAM | ✓ | | ✓ |
| PDP | | ✓ | |
| Integrated Gradients | | ✓ | |
| Occlusion Sensitivity | | ✓ | |

---

## References

1. Kolonne, S., Fernando, C., Kumarasinghe, H., & Meedeniya, D. (2021). MobileNetV2 based Chest X-Rays Classification. *DASA 2021*. https://doi.org/10.1109/dasa53625.2021.9682248

2. Hedhoud, Y., Mekhaznia, T., & Amroune, M. (2023). An improvement of the CNN-XGboost model for pneumonia disease classification. *Polish Journal of Radiology, 88*, 483–493. https://doi.org/10.5114/pjr.2023.132533

3. Xie, Y., Zhu, B., Jiang, Y., Zhao, B., & Yu, H. (2025). Diagnosis of pneumonia from chest X-ray images using YOLO deep learning. *Frontiers in Neurorobotics, 19*, 1576438. https://doi.org/10.3389/fnbot.2025.1576438
