import numpy as np
import time
from .data_fetcher import fetch_realtime_data


def predict_next_price(model, last_sequence, scaler):
    """
    Predict the next price given a sequence.

    Args:
        model: Trained LSTM model
        last_sequence: Last seq_length data points (scaled)
        scaler: Fitted scaler for inverse transform

    Returns:
        Predicted price (unscaled)
    """
    input_seq = last_sequence.reshape(1, last_sequence.shape[0], last_sequence.shape[1])
    prediction = model.predict(input_seq, verbose=0)

    n_features = scaler.n_features_in_
    pred_padded = np.zeros((1, n_features))
    pred_padded[0, 0] = prediction[0, 0]
    predicted_price = scaler.inverse_transform(pred_padded)[0, 0]

    return predicted_price


def predict_realtime(model, scaler, ticker="AAPL", seq_length=60, interval=60):
    """
    Continuously fetch latest data and predict next price.

    Args:
        model: Trained LSTM model
        scaler: Fitted scaler
        ticker: Stock symbol
        seq_length: Lookback window
        interval: Seconds between predictions

    Returns:
        Generator yielding (timestamp, actual_price, predicted_price)
    """
    while True:
        try:
            df = fetch_realtime_data(ticker)
            if len(df) < seq_length:
                print(f"Not enough data points. Need {seq_length}, got {len(df)}")
                time.sleep(interval)
                continue

            recent_data = df[["Close"]].values[-seq_length:]
            scaled_data = scaler.transform(
                np.column_stack([recent_data] + [np.zeros((len(recent_data), scaler.n_features_in_ - 1))])
            ) if scaler.n_features_in_ > 1 else scaler.transform(recent_data)

            predicted_price = predict_next_price(model, scaled_data, scaler)
            actual_price = df["Close"].iloc[-1]
            timestamp = df.index[-1]

            yield timestamp, actual_price, predicted_price

            time.sleep(interval)

        except Exception as e:
            print(f"Error: {e}")
            time.sleep(interval)