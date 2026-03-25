# Deep Learning Projects

Two deep learning projects completed as part of my coursework while studying neural networks and deep learning. These assignments challenged me to apply recurrent and convolutional neural networks to real-world problems — not just textbook exercises.

Through building these projects, I explored the full machine learning pipeline from data collection to deployment: fetching and cleaning real-world data, engineering meaningful features, designing and training neural network architectures, tuning hyperparameters, and building interactive demos. Each project taught me something different — the LSTM project deepened my understanding of time series forecasting and sequential data, while the CNN project introduced me to computer vision, transfer learning, and model interpretability.

Both projects follow a clean, modular structure with reusable source code and step-by-step notebooks, reflecting the software engineering practices I picked up along the way.

## Projects

### 1. LSTM Stock Price Prediction
A Bidirectional LSTM neural network that predicts stock prices using 5 years of historical market data. The model achieves ~97-98% accuracy with an RMSE of ~$6.89 on Apple (AAPL) stock, featuring real-time predictions, automated hyperparameter tuning, and a live demo.

**Key highlights:**
- Bidirectional LSTM with early stopping and dynamic learning rate
- 37 engineered features with automated feature selection
- Reproducible results via random seed
- Auto-selects the best model between baseline and tuned versions
- Near real-time predictions using Yahoo Finance data

📁 [Full details → lstm_stock_prediction/README.md](lstm_stock_prediction/README.md)

### 2. CNN Chest X-Ray Pneumonia Detection
A Convolutional Neural Network that detects pneumonia from chest X-ray images with 95%+ accuracy. Uses transfer learning with a pre-trained ResNet50/VGG16 model fine-tuned on ~5,800 medical images, with Grad-CAM visualization to show where the model focuses.

**Key highlights:**
- Custom baseline CNN and transfer learning comparison
- Data augmentation and class imbalance handling
- Comprehensive evaluation with precision, recall, F1-score, and ROC curves
- Grad-CAM heatmaps for model interpretability
- Medical AI with real-world impact

📁 [Full details → cnn_pneumonia_detection/README.md](cnn_pneumonia_detection/README.md)

## Project Structure

```
assignments/
├── environment.yml
├── .gitignore
├── README.md                          ← You are here
│
├── lstm_stock_prediction/
│   ├── data/
│   ├── notebooks/                     # 6 notebooks (explore → demo)
│   ├── src/                           # 8 source files
│   ├── models/
│   └── README.md
│
├── cnn_pneumonia_detection/
│   ├── data/
│   ├── notebooks/                     # 6 notebooks (explore → demo)
│   ├── src/                           # 7 source files
│   ├── models/
│   └── README.md
│
└── utils/
    ├── __init__.py
    └── common.py
```

## Setup

### Prerequisites
- Python 3.12
- Conda (Anaconda/Miniconda)

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd assignments
```

2. Create and activate the environment:
```bash
conda create -n dev0 python=3.12 numpy pandas matplotlib scikit-learn jupyter opencv pillow seaborn joblib -y
conda activate dev0
pip install yfinance ta tensorflow keras
```

3. Verify installation:
```bash
python -c "import numpy, pandas, tensorflow, yfinance, ta; print('All packages installed successfully')"
```

### Running the Projects

Each project runs independently. Navigate to the project's `notebooks/` folder and run them in order (01 → 06).

```bash
conda activate dev0
cd lstm_stock_prediction/notebooks
jupyter notebook
```

```bash
conda activate dev0
cd cnn_pneumonia_detection/notebooks
jupyter notebook
```

## Technologies

| Technology | Purpose |
|-----------|---------|
| Python 3.12 | Programming language |
| TensorFlow / Keras | Deep learning framework |
| yfinance | Stock market data (LSTM project) |
| OpenCV / Pillow | Image processing (CNN project) |
| scikit-learn | Preprocessing and evaluation |
| pandas / NumPy | Data manipulation |
| matplotlib / seaborn | Visualization |