import os
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
)


def set_random_seed(seed=42):
    """
    Set random seed for reproducible results.

    Args:
        seed: Seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"Random seed set to {seed}")


def get_callbacks(model_save_path='../models/best_model.keras',
                  patience_stop=10, patience_lr=5, min_lr=1e-6):
    """
    Create training callbacks.

    1. EarlyStopping: Stops when validation loss stops improving
       - Restores best weights automatically
    2. ReduceLROnPlateau: Halves learning rate when stuck
       - Helps the model make finer adjustments
    3. ModelCheckpoint: Saves the best model during training
       - Monitors validation accuracy
       - Only saves when accuracy improves

    Args:
        model_save_path: Where to save the best model
        patience_stop: Epochs to wait before stopping
        patience_lr: Epochs to wait before reducing LR
        min_lr: Minimum learning rate floor

    Returns:
        List of Keras callbacks
    """
    # Stop training if val_loss doesn't improve for 10 epochs
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=patience_stop,
        restore_best_weights=True,
        verbose=1
    )

    # Halve learning rate if val_loss doesn't improve for 5 epochs
    lr_reducer = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=patience_lr,
        min_lr=min_lr,
        verbose=1
    )

    # Save the best model based on validation accuracy
    checkpoint = ModelCheckpoint(
        model_save_path,
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )

    return [early_stopping, lr_reducer, checkpoint]


def train_model(model, train_gen, val_gen, epochs=50, class_weights=None,
                model_save_path='../models/best_model.keras'):
    """
    Train the CNN model.

    Args:
        model: Compiled Keras model
        train_gen: Training data generator
        val_gen: Validation data generator
        epochs: Maximum number of epochs
        class_weights: Dictionary of class weights for imbalanced data
        model_save_path: Where to save the best model

    Returns:
        Training history
    """
    # Set seed for reproducibility
    set_random_seed(42)

    # Get callbacks
    callbacks = get_callbacks(model_save_path=model_save_path)

    # Train the model
    # steps_per_epoch: how many batches per epoch (all training data)
    # validation_steps: how many batches for validation
    history = model.fit(
        train_gen,
        steps_per_epoch=train_gen.samples // train_gen.batch_size,
        epochs=epochs,
        validation_data=val_gen,
        validation_steps=val_gen.samples // val_gen.batch_size,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1
    )

    return history