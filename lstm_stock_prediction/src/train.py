import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


def get_callbacks(patience_stop=10, patience_lr=5, min_lr=1e-6):
    """
    Create training callbacks for better and more stable training.

    Callbacks monitor training progress and take action automatically:

    1. EarlyStopping: Stops training when the model stops improving.
       - Monitors validation loss
       - If val_loss doesn't improve for 'patience' epochs, training stops
       - Restores the best weights (not the last, which might be worse)
       - Prevents overfitting by not training too long

    2. ReduceLROnPlateau: Reduces learning rate when stuck.
       - If val_loss stops improving for 'patience' epochs, learning rate drops
       - Smaller learning rate = finer adjustments = better convergence
       - factor=0.5 means learning rate is halved each time
       - Helps the model escape local minima and find better solutions

    Args:
        patience_stop: Epochs to wait before stopping (default 10)
        patience_lr: Epochs to wait before reducing LR (default 5)
        min_lr: Minimum learning rate floor (default 1e-6)

    Returns:
        List of Keras callbacks
    """
    # Stop training if validation loss doesn't improve for 10 epochs
    # restore_best_weights=True ensures we keep the best model, not the last
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=patience_stop,
        restore_best_weights=True,
        verbose=1
    )

    # Halve the learning rate if validation loss doesn't improve for 5 epochs
    # This helps the model make finer adjustments when it's close to optimal
    lr_reducer = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=patience_lr,
        min_lr=min_lr,
        verbose=1
    )

    return [early_stopping, lr_reducer]


def train_model(model, X_train, y_train, epochs=100, batch_size=32, validation_split=0.1):
    """
    Train the LSTM model with early stopping and learning rate reduction.

    Training process:
    1. Model trains on data for up to 'epochs' epochs
    2. After each epoch, checks validation loss
    3. If val_loss stops improving for 5 epochs → learning rate is halved
    4. If val_loss stops improving for 10 epochs → training stops early
    5. Best weights (lowest val_loss) are automatically restored

    This means the model might train for 30 epochs instead of 100 if it
    converges early, saving time and preventing overfitting.

    Args:
        model: Compiled Keras model
        X_train: Training sequences
        y_train: Training targets
        epochs: Maximum number of training epochs (will stop early if needed)
        batch_size: Batch size
        validation_split: Fraction of training data for validation

    Returns:
        Training history
    """
    # Get callbacks for early stopping and learning rate reduction
    callbacks = get_callbacks()

    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        shuffle=False,  # Don't shuffle time series data (order matters)
        callbacks=callbacks,  # Apply early stopping + LR reducer
        verbose=1
    )
    return history


def evaluate_model(model, X_test, y_test, scaler):
    """
    Evaluate model performance on test data.

    Process:
    1. Model predicts on test sequences
    2. Predictions are inverse-transformed back to real dollar values
       (since we scaled data to 0-1 range before training)
    3. Calculates RMSE and MAE in actual dollars

    RMSE (Root Mean Squared Error): Average error, penalizes big misses
    MAE (Mean Absolute Error): Average absolute error in dollars

    Args:
        model: Trained Keras model
        X_test: Test sequences
        y_test: Test targets
        scaler: Fitted scaler for inverse transform

    Returns:
        Dictionary with predictions and metrics (RMSE, MAE)
    """
    predictions = model.predict(X_test)

    # Inverse transform predictions back to real dollar values
    # We need to pad with zeros because scaler expects all features,
    # but we only have the prediction (first column)
    n_features = scaler.n_features_in_
    pred_padded = np.zeros((len(predictions), n_features))
    pred_padded[:, 0] = predictions.flatten()
    predictions_inv = scaler.inverse_transform(pred_padded)[:, 0]

    # Same inverse transform for actual values
    actual_padded = np.zeros((len(y_test), n_features))
    actual_padded[:, 0] = y_test
    actual_inv = scaler.inverse_transform(actual_padded)[:, 0]

    # Calculate error metrics in real dollar values
    rmse = np.sqrt(mean_squared_error(actual_inv, predictions_inv))
    mae = mean_absolute_error(actual_inv, predictions_inv)

    return {
        "predictions": predictions_inv,
        "actual": actual_inv,
        "rmse": rmse,
        "mae": mae
    }