import numpy as np
from sklearn.preprocessing import MinMaxScaler


def prepare_data(df, feature_columns=None, target_column="Close", test_size=0.2):
    """
    Scale data and split into train/test sets.

    Args:
        df: DataFrame with stock data
        feature_columns: List of columns to use as features (None = Close only)
        target_column: Column to predict
        test_size: Fraction of data for testing

    Returns:
        train_data, test_data, scaler
    """
    if feature_columns is None:
        feature_columns = [target_column]

    data = df[feature_columns].values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    split_idx = int(len(scaled_data) * (1 - test_size))
    train_data = scaled_data[:split_idx]
    test_data = scaled_data[split_idx:]

    return train_data, test_data, scaler


def create_sequences(data, seq_length=60):
    """
    Create input sequences and labels for LSTM.

    Args:
        data: Scaled numpy array
        seq_length: Number of time steps to look back

    Returns:
        X (sequences), y (next value targets)
    """
    X, y = [], []
    for i in range(seq_length, len(data)):
        X.append(data[i - seq_length:i])
        y.append(data[i, 0])

    return np.array(X), np.array(y)