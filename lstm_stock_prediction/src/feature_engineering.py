import pandas as pd
import numpy as np


def add_moving_averages(df, windows=[7, 14, 21, 50, 200]):
    """
    Add simple and exponential moving averages.

    Args:
        df: DataFrame with 'Close' column
        windows: List of window sizes

    Returns:
        DataFrame with moving average columns added
    """
    df = df.copy()
    for w in windows:
        if len(df) > w:
            df[f'SMA_{w}'] = df['Close'].rolling(window=w).mean()
            df[f'EMA_{w}'] = df['Close'].ewm(span=w, adjust=False).mean()

    print(f"Added SMA and EMA for windows: {windows}")
    return df


def add_rsi(df, window=14):
    """
    Add Relative Strength Index.

    Args:
        df: DataFrame with 'Close' column
        window: RSI period

    Returns:
        DataFrame with RSI column added
    """
    df = df.copy()
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    print(f"Added RSI (window={window})")
    return df


def add_macd(df, fast=12, slow=26, signal=9):
    """
    Add MACD, Signal line, and Histogram.

    Args:
        df: DataFrame with 'Close' column
        fast: Fast EMA period
        slow: Slow EMA period
        signal: Signal line period

    Returns:
        DataFrame with MACD columns added
    """
    df = df.copy()
    ema_fast = df['Close'].ewm(span=fast, adjust=False).mean()
    ema_slow = df['Close'].ewm(span=slow, adjust=False).mean()
    df['MACD'] = ema_fast - ema_slow
    df['MACD_Signal'] = df['MACD'].ewm(span=signal, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']

    print(f"Added MACD (fast={fast}, slow={slow}, signal={signal})")
    return df


def add_bollinger_bands(df, window=20, num_std=2):
    """
    Add Bollinger Bands.

    Args:
        df: DataFrame with 'Close' column
        window: Moving average window
        num_std: Number of standard deviations

    Returns:
        DataFrame with Bollinger Band columns added
    """
    df = df.copy()
    sma = df['Close'].rolling(window=window).mean()
    std = df['Close'].rolling(window=window).std()
    df['BB_Upper'] = sma + (std * num_std)
    df['BB_Middle'] = sma
    df['BB_Lower'] = sma - (std * num_std)
    df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']

    print(f"Added Bollinger Bands (window={window}, std={num_std})")
    return df


def add_volume_features(df):
    """
    Add volume-based features.

    Args:
        df: DataFrame with 'Close' and 'Volume' columns

    Returns:
        DataFrame with volume feature columns added
    """
    df = df.copy()
    df['Volume_Change'] = df['Volume'].pct_change()
    df['Volume_SMA_20'] = df['Volume'].rolling(window=20).mean()
    df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA_20']

    print("Added volume features (Volume_Change, Volume_SMA_20, Volume_Ratio)")
    return df


def add_lag_features(df, column="Close", lags=[1, 2, 3, 5, 7]):
    """
    Add lagged price features.

    Args:
        df: DataFrame with stock data
        column: Column to create lags for
        lags: List of lag periods

    Returns:
        DataFrame with lag columns added
    """
    df = df.copy()
    for lag in lags:
        df[f'{column}_Lag_{lag}'] = df[column].shift(lag)

    print(f"Added lag features for '{column}': {lags}")
    return df


def add_price_features(df):
    """
    Add price-derived features.

    Args:
        df: DataFrame with OHLC data

    Returns:
        DataFrame with price feature columns added
    """
    df = df.copy()
    df['Daily_Return'] = df['Close'].pct_change()
    df['High_Low_Range'] = (df['High'] - df['Low']) / df['Close']
    df['Open_Close_Range'] = (df['Close'] - df['Open']) / df['Open']
    df['Price_Momentum'] = df['Close'] / df['Close'].shift(10) - 1

    print("Added price features (Daily_Return, High_Low_Range, Open_Close_Range, Price_Momentum)")
    return df


def add_date_features(df):
    """
    Add date-based features.

    Args:
        df: DataFrame with DateTimeIndex

    Returns:
        DataFrame with date feature columns added
    """
    df = df.copy()
    df['Day_of_Week'] = df.index.dayofweek
    df['Month'] = df.index.month
    df['Quarter'] = df.index.quarter
    df['Week_of_Year'] = df.index.isocalendar().week.astype(int)

    print("Added date features (Day_of_Week, Month, Quarter, Week_of_Year)")
    return df


def feature_engineering_pipeline(df, include_ta=True):
    """
    Run the full feature engineering pipeline.

    Args:
        df: Cleaned DataFrame with OHLCV data
        include_ta: Whether to include all technical indicators

    Returns:
        DataFrame with all engineered features
    """
    print("Starting feature engineering pipeline...")
    print("-" * 40)

    df = add_moving_averages(df, windows=[7, 14, 21, 50])
    df = add_rsi(df)
    df = add_macd(df)
    df = add_bollinger_bands(df)
    df = add_volume_features(df)
    df = add_lag_features(df)
    df = add_price_features(df)
    df = add_date_features(df)

    # Drop rows with NaN from feature calculations
    before = len(df)
    df = df.dropna()
    after = len(df)

    print("-" * 40)
    print(f"Rows dropped (NaN from calculations): {before - after}")
    print(f"Final shape: {df.shape}")
    print(f"Total features: {df.shape[1]}")
    return df