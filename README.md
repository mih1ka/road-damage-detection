# RoadScan — CNN-Based Road Surface Damage Detection

**Machine Vision Project — 2025–26**  
Made by: Mihika

---

A custom Convolutional Neural Network trained from scratch to classify road surface images into four severity levels — Good, Satisfactory, Poor, and Very Poor. The system produces a continuous Severity Index, a maintenance priority alert, Grad-CAM explainability with damage localization, and GPS-based location reporting — all deployed as an interactive Streamlit web application.

---

## Project Structure

```
road-damage-detection/
├── app.py                      # Streamlit web application
├── model_utils.py              # CNN architecture (RoadDamageCNN)
├── style.css                   # Custom UI styling
├── mv_final_roadmodel.ipynb    # Training notebook (Google Colab)
├── inject_gps.py               # Utility to embed GPS into image EXIF
├── checkgps.py                 # Utility to verify EXIF GPS data
├── requirements.txt            # Python dependencies
└── README.md
```

> The trained model weights (`road_damage_test2_model.pth`) are not included due to file size limits.  
> Download from Google Drive and place in the root directory before running.  
> **Model weights:** https://drive.google.com/file/d/1YAh8F4kR38plGgGySS-4gG03EYj37KQQ/view?usp=sharing

---

## System Overview

The pipeline takes a single road image as input and produces six outputs:

1. **Class label** — Good, Satisfactory, Poor, or Very Poor
2. **Confidence score** — model certainty for the predicted class
3. **Severity Index** — continuous weighted damage score (0 to 3)
4. **Maintenance priority** — LOW, MEDIUM, HIGH, or CRITICAL
5. **Grad-CAM heatmap with damage localization** — visual explanation of the prediction with bounding boxes around detected damage regions and extracted close-up crops
6. **GPS-tagged map marker** — damage report pinned to the exact capture location using image EXIF metadata

---

## Model Architecture

```
Input (3 x 224 x 224)
    |
    ├── ConvBlock(3 -> 32)     224x224 -> 112x112    edges and gradients
    ├── ConvBlock(32 -> 64)    112x112 -> 56x56      textures
    ├── ConvBlock(64 -> 128)    56x56  -> 28x28      crack patterns
    ├── ConvBlock(128 -> 256)   28x28  -> 14x14      damage structures
    ├── Conv2d(256 -> 256)      14x14  -> 14x14      feature refinement
    |
    ├── AdaptiveAvgPool2d -> (256,)
    |
    ├── FC(256 -> 512) -> ReLU -> Dropout(0.5)
    ├── FC(512 -> 256) -> ReLU -> Dropout(0.3)
    └── FC(256 -> 4)   -> logits
```

Each ConvBlock contains: Conv2d -> BatchNorm2d -> ReLU -> Conv2d -> BatchNorm2d -> ReLU -> MaxPool2d  
Weight initialisation: Kaiming Normal for conv layers, Xavier Normal for linear layers.

The model is trained from scratch rather than fine-tuned from a pretrained backbone. Road damage is a texture and structural degradation problem; ImageNet pretrained features are optimised for object recognition and are not well suited to this domain.

---

## Results

| Class        | Precision | Recall | F1-Score |
|--------------|-----------|--------|----------|
| Good         | 1.00      | 1.00   | 1.00     |
| Satisfactory | ~0.85     | 0.90   | ~0.87    |
| Poor         | ~0.88     | 0.75   | ~0.81    |
| Very Poor    | 1.00      | 1.00   | 1.00     |
| **Overall**  |           |        | **~92%** |

Good and Very Poor classes are perfectly separated. The primary source of error is inter-class ambiguity between Poor and Satisfactory, which are visually similar at borderline severity levels.

---

## Training Configuration

| Hyperparameter   | Value                                        |
|------------------|----------------------------------------------|
| Image size       | 224 x 224                                    |
| Batch size       | 32                                           |
| Epochs           | 30 with early stopping (patience = 7)        |
| Optimizer        | AdamW (lr = 1e-3, weight decay = 1e-4)       |
| Scheduler        | CosineAnnealingLR (eta_min = 1e-6)           |
| Loss             | CrossEntropyLoss with label smoothing = 0.05 |
| Dropout          | 0.5 (FC1), 0.3 (FC2)                        |
| Augmentations    | RandomHorizontalFlip, RandomVerticalFlip, RandomRotation, ColorJitter, RandomAffine |
| Imbalance        | WeightedRandomSampler (inverse class frequency) |
| Dataset split    | 70 / 15 / 15 stratified train / val / test   |

---

## Severity Index and Maintenance Priority

Beyond the predicted class label, the system computes a continuous Severity Index that preserves the full probability distribution rather than collapsing to a single class:

```
SI = (p_good x 0) + (p_satisfactory x 1) + (p_poor x 2) + (p_very_poor x 3)
```

Maintenance priority is derived from the product of SI and confidence, so that uncertain high-severity predictions are weighted differently from confident ones:

```
Priority score = SI x confidence
```

| Score         | Priority |
|---------------|----------|
| < 0.5         | LOW      |
| 0.5 to 1.0    | MEDIUM   |
| 1.0 to 2.0    | HIGH     |
| > 2.0         | CRITICAL |

---

## Grad-CAM Explainability and Damage Localization

The system uses Gradient-weighted Class Activation Mapping (Grad-CAM) to produce a spatial heatmap showing which regions of the image the model focused on when making its prediction.

**How it works:**
1. A forward hook captures feature maps at the last Conv2d layer (model.features[-3], shape 256 x 14 x 14)
2. A backward hook captures gradients for the target class score at the same layer
3. Gradients are global average pooled to produce one importance weight per channel
4. Weighted sum of feature maps across all 256 channels produces a 14 x 14 heatmap
5. ReLU removes negative activations; heatmap is normalised to 0-1 and resized to 224 x 224

**Damage localization** (applied for Poor and Very Poor predictions):

The Grad-CAM heatmap is thresholded at 60% of its peak activation value to produce a binary mask. OpenCV `findContours` identifies individual damage regions from this mask. A bounding box is drawn around each contour with area greater than 100 pixels. Each detected region is also extracted as a close-up crop with 10 pixels of padding, giving a zoomed view of each distinct damaged patch.

The display layout for Poor and Very Poor images is:

```
Row 1:  [ Input image ]       [ Grad-CAM heatmap ]
Row 2:  [ Damage + boxes ]    [ Extracted damage crop ]
```

---

## GPS-Based Location Alerting

The app reads GPS coordinates embedded in the EXIF metadata of the uploaded image. Smartphones embed GPS automatically in JPEG files when location access is enabled. Coordinates are stored in degrees-minutes-seconds format and converted to decimal degrees for map rendering.

```python
decimal_degrees = degrees + (minutes / 60) + (seconds / 3600)
```

If EXIF GPS data is present, the app displays "Live GPS from image EXIF" and pins the damage report to the real capture location on a Folium map. If no EXIF data is found (for example, dataset images downloaded from the web), the system falls back to a default coordinate and clearly labels it as simulated.

The map marker displays the predicted class, Severity Index, confidence, and priority level in a popup.

**GPS utility scripts:**

`inject_gps.py` — embeds GPS coordinates into a JPEG's EXIF block for testing. Useful for demonstrating the live GPS feature with dataset images that do not have location metadata.

`checkgps.py` — verifies whether a given image contains GPS EXIF data and prints the parsed coordinates.

---

## Running the App

**1. Clone the repository**
```bash
git clone https://github.com/mih1ka/road-damage-detection.git
cd road-damage-detection
```

**2. Create a virtual environment and install dependencies**
```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**3. Download model weights**  
Download `road_damage_test2_model.pth` from the Google Drive link above and place it in the root directory.

**4. Run the application**
```bash
streamlit run app.py
```

---

## Training Notebook

The full training pipeline is in `mv_final_roadmodel.ipynb`, designed to run on Google Colab with GPU.

Sections covered: dataset download via kagglehub, preprocessing pipeline visualisation, data augmentation and DataLoaders, model architecture, training loop with early stopping, evaluation metrics, Grad-CAM on test set, severity index analysis.

To retrain:
1. Open `mv_final_roadmodel.ipynb` in Google Colab
2. Run all cells — the dataset downloads automatically
3. The best checkpoint is saved to `/content/best_model.pth` and downloaded at the end of the notebook

---

## Tech Stack

| Component         | Library / Tool                  |
|-------------------|---------------------------------|
| Deep learning     | PyTorch                         |
| Data pipeline     | torchvision, OpenCV, Pillow     |
| Explainability    | Grad-CAM (implemented from scratch) |
| Damage localization | OpenCV contour detection      |
| Web application   | Streamlit                       |
| Maps              | Folium, streamlit-folium        |
| GPS parsing       | Pillow EXIF, piexif             |
| Evaluation        | scikit-learn                    |
| Visualisation     | matplotlib, seaborn             |

---

## Academic Context

This project was submitted as a course project for the Machine Vision module. The objective was to design and implement a complete computer vision pipeline — from dataset preparation and model training through to deployment — addressing a real-world infrastructure problem.

The system addresses three gaps common in existing road damage classification work: the absence of a continuous damage score that captures prediction uncertainty, the lack of visual explainability for model decisions, and the absence of a spatial component that connects predictions to physical locations.
