# LSTM Stock Price Prediction

A deep learning project that uses a Bidirectional LSTM neural network to predict stock prices with ~97% accuracy. Built with TensorFlow/Keras and Yahoo Finance data.

## Results

| Metric | Value |
|--------|-------|
| RMSE | ~$6.89 |
| MAE | ~$5.38 |
| Accuracy | ~97-98% |
| Data | 5 years of AAPL daily prices (1,256 data points) |

## Project Structure

```
lstm_stock_prediction/
├── data/
│   └── raw/                          # Cleaned and engineered data
├── notebooks/
│   ├── 01_exploration.ipynb          # Data exploration and visualization
│   ├── 02_data_cleaning.ipynb        # Handle missing values, outliers
│   ├── 03_feature_engineering.ipynb   # Create and select features
│   ├── 04_training.ipynb             # Train the LSTM model
│   ├── 05_tuning.ipynb               # Hyperparameter tuning
│   └── 06_demo.ipynb                 # Real-time predictions and demo
├── src/
│   ├── __init__.py                   # Package exports
│   ├── data_fetcher.py               # Fetch data from Yahoo Finance
│   ├── data_cleaning.py              # Data quality and cleaning
│   ├── feature_engineering.py        # Technical indicators and features
│   ├── feature_selection.py          # Select best features
│   ├── preprocessing.py              # Scale data and create sequences
│   ├── model.py                      # Bidirectional LSTM architecture
│   ├── train.py                      # Training with early stopping
│   └── predict.py                    # Predictions and real-time demo
├── models/                           # Saved models and configs
└── README.md
```

## Features

- **Bidirectional LSTM** — reads price sequences forwards and backwards for better pattern recognition
- **Early Stopping** — automatically stops training when the model stops improving
- **Learning Rate Reduction** — dynamically adjusts learning rate for better convergence
- **Reproducible Results** — random seed ensures consistent results across runs
- **Auto Model Selection** — automatically saves the best performing model (baseline vs tuned)
- **Real-Time Predictions** — fetches latest market data and predicts next price
- **5-Day Forecasting** — generates multi-day price forecasts with confidence bands

## How It Works

### Data Pipeline
1. **Fetch** — 5 years of daily OHLCV data from Yahoo Finance
2. **Clean** — handle missing values, detect outliers and anomalous returns
3. **Engineer** — create 37 features including moving averages, RSI, MACD, Bollinger Bands, volume indicators, lag features, and date features
4. **Select** — automatically select the best 4 non-redundant features using correlation analysis

### Selected Features
| Feature | Type | Description |
|---------|------|-------------|
| Close | Price | Daily closing price |
| Volume_SMA_20 | Volume | 20-day average trading volume |
| Volume | Volume | Daily trading volume |
| MACD_Signal | Trend | MACD signal line for trend detection |

### Model Architecture
```
Input (60 days × 4 features)
    ↓
Bidirectional LSTM (50 units) + Dropout (0.2)
    ↓
Bidirectional LSTM (50 units) + Dropout (0.2)
    ↓
Bidirectional LSTM (50 units) + Dropout (0.2)
    ↓
Dense (25 units, ReLU)
    ↓
Dense (1 unit) → Predicted Price
```

### Training
- **Optimizer:** Adam with dynamic learning rate
- **Loss:** Mean Squared Error
- **Early Stopping:** Patience of 10 epochs, restores best weights
- **LR Reducer:** Halves learning rate after 5 epochs of no improvement

## Setup

### Prerequisites
- Python 3.12
- Conda (Anaconda/Miniconda)

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd lstm_stock_prediction
```

2. Create and activate the conda environment:
```bash
conda create -n dev0 python=3.12 numpy pandas matplotlib scikit-learn jupyter opencv pillow seaborn joblib -y
conda activate dev0
```

3. Install pip packages:
```bash
pip install yfinance ta tensorflow keras
```

4. Verify installation:
```bash
python -c "import numpy, pandas, tensorflow, yfinance, ta; print('All packages installed successfully')"
```

## Usage

### Run notebooks in order:

```bash
cd notebooks
jupyter notebook
```

1. **01_exploration.ipynb** — explore and visualize the stock data
2. **02_data_cleaning.ipynb** — clean the data and save to CSV
3. **03_feature_engineering.ipynb** — engineer features and select the best ones
4. **04_training.ipynb** — train the baseline LSTM model
5. **05_tuning.ipynb** — tune hyperparameters and auto-save the best model
6. **06_demo.ipynb** — run predictions and real-time demo

### Quick Demo
If the model is already trained, just run `06_demo.ipynb` to see predictions immediately.

### Change Stock Ticker
To predict a different stock, update the `ticker` variable in the notebooks:
```python
ticker = "MSFT"  # or "GOOGL", "TSLA", "NVDA", etc.
```
Then re-run all notebooks from 01 to 06.

## Key Metrics Explained

- **RMSE (Root Mean Squared Error):** Average prediction error in dollars. Lower is better. Penalizes large errors more heavily.
- **MAE (Mean Absolute Error):** Average absolute prediction error in dollars. Lower is better. More intuitive than RMSE.
- **Training Loss vs Validation Loss:** When these are close together, the model is generalizing well. A large gap indicates overfitting.

## Limitations

- Data has a ~15 minute delay from actual market prices (Yahoo Finance free tier)
- Real-time predictions work best during US market hours (9:30 AM - 4:00 PM EST)
- Model should be retrained periodically as market conditions change
- This is a demonstration project, not a trading system

## Technologies

- **Python 3.12**
- **TensorFlow / Keras** — deep learning framework
- **yfinance** — Yahoo Finance data API
- **ta** — technical analysis indicators
- **scikit-learn** — data preprocessing
- **pandas / NumPy** — data manipulation
- **matplotlib / seaborn** — visualization