import os
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Bidirectional


def set_random_seed(seed=42):
    """
    Set random seed for reproducibility across all libraries.

    This ensures the same results every time you run the model.
    Without this, weights are initialized randomly each run,
    causing different RMSE values every time.

    Args:
        seed: Seed value (any integer, 42 is convention)
    """
    # Python's built-in random
    random.seed(seed)

    # NumPy random (used for data shuffling, splitting)
    np.random.seed(seed)

    # TensorFlow random (used for weight initialization, dropout)
    tf.random.set_seed(seed)

    # Ensure deterministic operations in TensorFlow
    # This forces TF to use deterministic algorithms where possible
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['PYTHONHASHSEED'] = str(seed)

    print(f"Random seed set to {seed} for reproducible results")


def build_lstm_model(seq_length, n_features, units=50, dropout=0.2):
    """
    Build and compile a Bidirectional LSTM model.

    Bidirectional LSTM reads the sequence both forwards and backwards,
    allowing it to capture patterns from both directions. For example:
    - Forward: learns how past prices lead to current price
    - Backward: learns what future context tells about current price

    This typically improves accuracy over a standard LSTM.

    Args:
        seq_length: Number of time steps (lookback window)
        n_features: Number of input features
        units: Number of LSTM units per layer
        dropout: Dropout rate for regularization

    Returns:
        Compiled Keras Bidirectional LSTM model
    """
    # Set seed before building to ensure same weight initialization
    set_random_seed(42)

    model = Sequential([
        # Input layer: defines the shape of incoming data
        # shape = (seq_length, n_features) e.g., (60 days, 4 features)
        Input(shape=(seq_length, n_features)),

        # First Bidirectional LSTM layer:
        # - Wraps the LSTM to read sequence both forwards and backwards
        # - Output units = units * 2 (forward + backward combined)
        # - return_sequences=True passes full sequence to next layer
        Bidirectional(LSTM(units=units, return_sequences=True)),
        # Dropout: randomly turns off 20% of neurons to prevent overfitting
        Dropout(dropout),

        # Second Bidirectional LSTM layer:
        # - Learns more complex patterns from the first layer's output
        # - return_sequences=True because another LSTM follows
        Bidirectional(LSTM(units=units, return_sequences=True)),
        Dropout(dropout),

        # Third Bidirectional LSTM layer:
        # - Learns the most abstract patterns
        # - No return_sequences: outputs single vector for Dense layers
        Bidirectional(LSTM(units=units)),
        Dropout(dropout),

        # Dense layer: compresses LSTM output into 25 values
        # Acts as a bridge between LSTM patterns and final prediction
        Dense(units=25, activation='relu'),

        # Output layer: single value — the predicted stock price
        # No activation function because we're predicting a continuous value
        Dense(units=1)
    ])

    # Compile the model:
    # - adam optimizer: auto-adjusts learning rate during training
    # - mean_squared_error: penalizes larger errors more (good for prices)
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model