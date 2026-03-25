# CNN Chest X-Ray Pneumonia Detection

A Convolutional Neural Network that detects pneumonia from chest X-ray images with 93.11% accuracy. Built with TensorFlow/Keras using transfer learning with VGG16.

## Results

| Model | Accuracy | Precision | Recall | F1-Score | AUC |
|-------|----------|-----------|--------|----------|-----|
| Baseline CNN | 70.51% | 0.6807 | 0.9949 | 0.8083 | 0.9019 |
| ResNet50 Transfer | 80.13% | 0.7639 | 0.9872 | 0.8613 | 0.9411 |
| **VGG16 Transfer (best)** | **93.11%** | **0.9201** | **0.9744** | **0.9465** | **0.9745** |

Key findings:
- VGG16 at 150×150 outperformed both baseline and ResNet50 at 224×224
- Smaller images reduced noise and improved generalization
- Recall of 97.44% means almost no pneumonia cases are missed
- 22.60% improvement over baseline, 14.26% over ResNet50

## Objective

Develop a CNN that can classify chest X-ray images as either **Normal** or **Pneumonia**, assisting medical professionals in faster and more accurate screening.

## Dataset

**Chest X-Ray Images (Pneumonia)** from Kaggle
- **Download:** [Chest X-Ray Images (Pneumonia) on Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
- **Total images:** ~5,856
- **Classes:** 2 (Normal, Pneumonia)
- **Split:** Pre-split into train, validation, and test sets
- **Image type:** Grayscale chest X-ray images
- **Format:** JPEG

| Split | Normal | Pneumonia | Total |
|-------|--------|-----------|-------|
| Train | ~1,139 | ~3,295 | ~4,434 |
| Val | ~202 | ~596 | ~798 |
| Test | 234 | 390 | 624 |

> Note: The original Kaggle validation set had only 16 images. We created a proper stratified validation split (15%) from the training data to ensure balanced class ratios in both sets.

## Approach

### 1. Baseline CNN
- Custom CNN built from scratch (4 Conv blocks + Dense layers)
- Establishes baseline accuracy (70.51%)
- Demonstrates understanding of CNN architecture

### 2. Transfer Learning — ResNet50
- Pre-trained on ImageNet (1.4M images)
- Two-phase training: frozen base → fine-tune top 20 layers
- Improved to 80.13% accuracy

### 3. Transfer Learning — VGG16 (best)
- Pre-trained on ImageNet, simpler architecture than ResNet50
- 150×150 images outperformed 224×224 (less noise, better generalization)
- Two-phase training: frozen base → fine-tune top 8 layers
- Best accuracy: 93.11%

### 4. Handling Class Imbalance
- Dataset has ~2.89:1 Pneumonia to Normal ratio
- Applied class weights during training to prevent bias toward majority class
- Stratified validation split preserves class ratios

### 5. Data Augmentation
- Random rotation (15°), zoom (10%), horizontal flip, shift (10%)
- Increases effective dataset size and reduces overfitting
- Applied only to training data — validation and test remain unmodified

### 6. Evaluation
- Accuracy, Precision, Recall, F1-Score
- Confusion Matrix for all three models
- ROC Curve and AUC Score comparison
- Grad-CAM visualization (heatmap showing where the model focuses)

## Project Structure

```
cnn_pneumonia_detection/
├── data/
│   └── chest_xray/
│       ├── train/
│       │   ├── NORMAL/
│       │   └── PNEUMONIA/
│       ├── val/
│       │   ├── NORMAL/
│       │   └── PNEUMONIA/
│       └── test/
│           ├── NORMAL/
│           └── PNEUMONIA/
├── notebooks/
│   ├── 01_exploration.ipynb                    # Explore dataset, visualize samples
│   ├── 02_preprocessing.ipynb                  # Clean, resize, augment images
│   ├── 03_baseline_model.ipynb                 # Train custom CNN from scratch
│   ├── 04_1_transfer_learning_resnet50.ipynb   # Fine-tune ResNet50
│   ├── 04_2_transfer_learning_vgg16.ipynb      # Fine-tune VGG16 (best model)
│   ├── 05_evaluation.ipynb                     # Full evaluation — all 3 models
│   └── 06_demo.ipynb                           # Predict on new images, Grad-CAM
├── src/
│   ├── __init__.py                             # Package exports
│   ├── data_loader.py                          # Load datasets, stratified split, class weights
│   ├── preprocessing.py                        # Resize, normalize, augment images
│   ├── baseline_model.py                       # Custom CNN architecture
│   ├── transfer_model.py                       # Transfer learning (ResNet50/VGG16)
│   ├── train.py                                # Training with callbacks
│   ├── evaluate.py                             # Metrics, confusion matrix, ROC curve
│   └── gradcam.py                              # Grad-CAM heatmap visualization
├── models/
│   └── (saved .keras models)
└── README.md
```

## Source Files

| File | Purpose |
|------|---------|
| `data_loader.py` | Load images from directories, create stratified train/val splits, calculate class weights for imbalanced data |
| `preprocessing.py` | Resize images to target size, normalize pixel values (0-1), apply data augmentation (rotation, flip, zoom, shift) |
| `baseline_model.py` | Custom CNN with 4 Conv2D blocks, BatchNormalization, MaxPooling, Dense layers, and Dropout |
| `transfer_model.py` | Pre-trained ResNet50/VGG16 with frozen base and custom classification head, supports fine-tuning |
| `train.py` | Training with early stopping, learning rate reduction, model checkpointing, and class weights |
| `evaluate.py` | Generate accuracy, precision, recall, F1-score, confusion matrix, ROC curve, and AUC |
| `gradcam.py` | Grad-CAM visualization — highlights which regions of the X-ray the model focuses on for its prediction |

## Notebooks

| Notebook | Purpose |
|----------|---------|
| `01_exploration` | Visualize sample X-rays, check class distribution, analyze image sizes and pixel intensities |
| `02_preprocessing` | Create stratified validation split, set up data augmentation, verify image shapes |
| `03_baseline_model` | Train custom CNN from scratch, establish baseline (70.51%) |
| `04_1_transfer_learning_resnet50` | Fine-tune ResNet50 in two phases (80.13%) |
| `04_2_transfer_learning_vgg16` | Fine-tune VGG16 at 150×150, best model (93.11%) |
| `05_evaluation` | Side-by-side comparison of all 3 models — ROC curves, bar charts, error analysis |
| `06_demo` | Load best model, predict on new X-rays, Grad-CAM heatmaps, final dashboard |

## Key Metrics

- **Accuracy** — overall correct predictions (93.11%)
- **Precision** — of all predicted pneumonia cases, how many actually have it (0.9201)
- **Recall** — of all actual pneumonia cases, how many were detected (0.9744)
- **F1-Score** — harmonic mean of precision and recall (0.9465)
- **AUC** — area under the ROC curve (0.9745)

> In medical contexts, **Recall is critical** — missing a pneumonia case (false negative) is more dangerous than a false alarm. Our best model achieves 97.44% recall.

## Technologies

- **Python 3.12**
- **TensorFlow / Keras** — deep learning framework
- **VGG16 / ResNet50** — pre-trained models for transfer learning
- **OpenCV / Pillow** — image processing
- **scikit-learn** — metrics, evaluation, stratified splitting
- **matplotlib / seaborn** — visualization

## Setup

### 1. Download the dataset
- **Download:** [Chest X-Ray Images (Pneumonia) on Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
- You'll need a free Kaggle account to download
- Extract the zip and copy the inner `chest_xray` folder into `cnn_pneumonia_detection/data/`
- Final path should be: `cnn_pneumonia_detection/data/chest_xray/train/`, `val/`, `test/`

### 2. Use the same conda environment
```bash
conda activate dev0
```

All required packages are already installed from the LSTM project.

### 3. Run notebooks in order
```bash
cd notebooks
jupyter notebook
```

Run 01 → 06 sequentially. Each notebook builds on the previous one's outputs.

## Limitations

- Model is trained on a single dataset — may not generalize to all X-ray machines or populations
- Binary classification only (Normal vs Pneumonia) — does not distinguish between bacterial and viral pneumonia
- Class imbalance (2.89:1) is handled with class weights but still influences model behavior
- This is a demonstration project, not a clinical diagnostic tool