from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Input
)


def build_baseline_cnn(input_shape=(224, 224, 3)):
    """
    Build a custom CNN from scratch for binary classification.

    Architecture follows a standard pattern:
    Conv2D → BatchNorm → MaxPool → repeated, then Flatten → Dense → Output

    Each convolutional block:
    1. Conv2D: Learns to detect features (edges, textures, shapes)
       - Filters increase (32→64→128) to learn increasingly complex features
    2. BatchNormalization: Stabilizes training by normalizing layer outputs
    3. MaxPooling: Reduces image size by half, keeping strongest features

    Args:
        input_shape: Shape of input images (height, width, channels)

    Returns:
        Compiled Keras model
    """
    model = Sequential([
        Input(shape=input_shape),

        # Block 1: Detect basic features (edges, lines)
        # 32 filters of size 3x3 scan across the image
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),  # 224x224 → 112x112

        # Block 2: Detect intermediate features (textures, curves)
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),  # 112x112 → 56x56

        # Block 3: Detect complex features (shapes, patterns)
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),  # 56x56 → 28x28

        # Block 4: Detect high-level features (lung opacity, consolidation)
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),  # 28x28 → 14x14

        # Flatten: Convert 2D feature maps to 1D vector
        # 14x14x128 = 25,088 values
        Flatten(),

        # Dense layers: Learn to classify based on extracted features
        Dense(256, activation='relu'),
        Dropout(0.5),  # Drop 50% of neurons to prevent overfitting

        Dense(128, activation='relu'),
        Dropout(0.3),

        # Output: Single neuron with sigmoid activation
        # Outputs probability between 0 (Normal) and 1 (Pneumonia)
        Dense(1, activation='sigmoid')
    ])

    # Binary crossentropy: standard loss for 2-class classification
    # Measures how far predictions are from actual labels (0 or 1)
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model