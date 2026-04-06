# RoadScan — CNN-Based Road Surface Damage Detection

> Machine Vision · 2025–26  
> Mihika 

A custom Convolutional Neural Network trained **from scratch** to classify road surface images into 4 severity levels, with confidence scoring, Grad-CAM explainability, severity index computation, and GPS-based location alerting — deployed as an interactive Streamlit web app.

---


## Project Structure

```
roadscan/
├── app.py                        # Streamlit web application
├── model_utils.py                # CNN architecture (RoadDamageCNN)
├── style.css                     # Custom UI styling
├── mv_final_roadmodel.ipynb      # Training notebook (Google Colab)
├── requirements.txt              # Python dependencies
└── README.md
```

> The trained model file (`road_damage_modeltest1.pth`) is not included in this repo due to file size. 
> **Download it here:** "https://drive.google.com/file/d/1YAh8F4kR38plGgGySS-4gG03EYj37KQQ/view?usp=sharing"
> Place it in the root directory before running the app.

---

## Model Architecture

```
Input (3 × 224 × 224)
    │
    ├── ConvBlock(3 → 32)    224×224 → 112×112   edges & gradients
    ├── ConvBlock(32 → 64)   112×112 → 56×56     textures
    ├── ConvBlock(64 → 128)   56×56  → 28×28     crack patterns
    ├── ConvBlock(128 → 256)  28×28  → 14×14     damage structures
    ├── Conv2d(256 → 256)     14×14  → 14×14     feature refinement
    │
    ├── AdaptiveAvgPool2d → (256,)
    │
    ├── FC(256 → 512) → ReLU → Dropout(0.5)
    ├── FC(512 → 256) → ReLU → Dropout(0.3)
    └── FC(256 → 4)   → logits
```

Each `ConvBlock` = Conv2d → BatchNorm → ReLU → Conv2d → BatchNorm → ReLU → MaxPool2d  
Weights initialized with **Kaiming Normal** (conv) and **Xavier Normal** (linear).

---

## Results

| Class        | Precision | Recall | F1    |
|--------------|-----------|--------|-------|
| Good         | 1.00      | 1.00   | 1.00  |
| Satisfactory | ~0.85     | 0.90   | ~0.87 |
| Poor         | ~0.88     | 0.75   | ~0.81 |
| Very Poor    | 1.00      | 1.00   | 1.00  |
| **Overall**  |           |        | **~92%** |

- Good and Very Poor — perfectly separated (0 misclassifications)
- Poor ↔ Satisfactory — expected inter-class ambiguity (visually similar)

---

## Training Configuration

| Hyperparameter   | Value                        |
|------------------|------------------------------|
| Image size       | 224 × 224                    |
| Batch size       | 32                           |
| Epochs           | 30 (early stopping, patience=7) |
| Optimizer        | AdamW (lr=1e-3, wd=1e-4)    |
| Scheduler        | CosineAnnealingLR            |
| Loss             | CrossEntropy + label smoothing (0.05) |
| Dropout          | 0.5 / 0.3                   |
| Augmentations    | Flip, Rotation, ColorJitter, Affine |
| Imbalance fix    | WeightedRandomSampler        |
| Dataset split    | 70 / 15 / 15 (stratified)   |

---

## Severity Index & Priority

Beyond the predicted class, RoadScan computes a **continuous Severity Index**:

```
SI = Σ wᵢ × pᵢ
```

where weights are: Good=0, Satisfactory=1, Poor=2, Very Poor=3

**Maintenance Priority** = f(SI × confidence):

| Score     | Priority |
|-----------|----------|
| < 0.5     | LOW      |
| 0.5 – 1.0 | MEDIUM   |
| 1.0 – 2.0 | HIGH     |
| > 2.0     | CRITICAL |

---

## Location Alerting

The app reads **GPS EXIF metadata** from the uploaded image to pin the damage location on an interactive Folium map. If no EXIF data is present (e.g., dataset images), it falls back to a simulated coordinate. In production with dashcam or field-captured images, the map populates automatically with real coordinates.

---

## Running the App

**1. Clone the repo**
```bash
git clone https://github.com/yourusername/roadscan.git
cd roadscan
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Download the model weights**  
Download `road_damage_modeltest1.pth` from [Google Drive Link] and place it in the root directory.

**4. Run**
```bash
streamlit run app.py
```

---

## Training Notebook

The full training pipeline is in `mv_final_roadmodel.ipynb`, designed for **Google Colab** with GPU.  
It covers: dataset download → preprocessing visualization → data loaders → model definition → training loop → evaluation → Grad-CAM → severity analysis.

To retrain:
1. Open in Google Colab
2. Run all cells — dataset downloads automatically via `kagglehub`
3. The best checkpoint saves to `/content/best_model.pth` and can be downloaded at the end

---

## Tech Stack

| Component       | Library              |
|-----------------|----------------------|
| Deep Learning   | PyTorch              |
| Data Pipeline   | torchvision, OpenCV  |
| Explainability  | Grad-CAM (custom)    |
| Web App         | Streamlit            |
| Maps            | Folium               |
| Evaluation      | scikit-learn         |
| Visualization   | matplotlib, seaborn  |

---

## Demo

![Demo Image](goodroaddemo.jpeg)
![Demo Image](goodroaddemo2.jpeg)
![Demo Image](poordemo1.jpeg)
![Demo Image](poordemo2.jpeg)


> Upload a road image → get damage class, confidence, severity index, Grad-CAM heatmap, and a GPS-pinned map alert.

---
## License

This project is submitted as academic course project for Machine Vision.
