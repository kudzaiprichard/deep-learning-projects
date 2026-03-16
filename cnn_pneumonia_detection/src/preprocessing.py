import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
import os


def create_data_generators(data_dir, target_size=(224, 224), batch_size=32, augment=True):
    """
    Create data generators with optional augmentation for training.

    Data augmentation artificially increases the training set by applying
    random transformations to images. This helps the model generalize
    better and reduces overfitting.

    Augmentation only applies to training data — validation and test
    data are only rescaled (no transformations).

    Args:
        data_dir: Path to root data directory
        target_size: Resize images to this size
        batch_size: Number of images per batch
        augment: Whether to apply augmentation to training data

    Returns:
        train_gen, val_gen, test_gen
    """
    if augment:
        # Training data: rescale + augmentation
        # These transformations create variations of each image:
        train_datagen = ImageDataGenerator(
            rescale=1.0 / 255,          # Normalize pixels to 0-1
            rotation_range=15,           # Randomly rotate up to 15 degrees
            width_shift_range=0.1,       # Randomly shift horizontally by 10%
            height_shift_range=0.1,      # Randomly shift vertically by 10%
            shear_range=0.1,             # Randomly apply shearing
            zoom_range=0.1,              # Randomly zoom in/out by 10%
            horizontal_flip=True,        # Randomly flip horizontally
            fill_mode='nearest'          # Fill empty pixels after transformation
        )
    else:
        # No augmentation — just rescale
        train_datagen = ImageDataGenerator(rescale=1.0 / 255)

    # Validation and test: only rescale, no augmentation
    # We want to evaluate on unmodified images
    val_datagen = ImageDataGenerator(rescale=1.0 / 255)
    test_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_gen = train_datagen.flow_from_directory(
        os.path.join(data_dir, 'train'),
        target_size=target_size,
        batch_size=batch_size,
        class_mode='binary',
        shuffle=True
    )

    val_gen = val_datagen.flow_from_directory(
        os.path.join(data_dir, 'val'),
        target_size=target_size,
        batch_size=batch_size,
        class_mode='binary',
        shuffle=False
    )

    test_gen = test_datagen.flow_from_directory(
        os.path.join(data_dir, 'test'),
        target_size=target_size,
        batch_size=batch_size,
        class_mode='binary',
        shuffle=False
    )

    print(f"Training samples: {train_gen.samples} (augmentation: {augment})")
    print(f"Validation samples: {val_gen.samples}")
    print(f"Test samples: {test_gen.samples}")

    return train_gen, val_gen, test_gen


def preprocess_single_image(img_path, target_size=(224, 224)):
    """
    Preprocess a single image for prediction.

    Loads an image, resizes it to match the model's expected input,
    and normalizes pixel values to 0-1.

    Args:
        img_path: Path to the image file
        target_size: Size to resize the image to

    Returns:
        Preprocessed image array with shape (1, height, width, 3)
        Original image for display
    """
    # Load image and resize
    img = image.load_img(img_path, target_size=target_size)

    # Convert to numpy array: shape (224, 224, 3)
    img_array = image.img_to_array(img)

    # Normalize pixel values from 0-255 to 0-1
    img_normalized = img_array / 255.0

    # Add batch dimension: shape (1, 224, 224, 3)
    # Model expects a batch of images, so we add a dimension
    img_batch = np.expand_dims(img_normalized, axis=0)

    return img_batch, img_array