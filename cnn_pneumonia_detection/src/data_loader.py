import os
import shutil
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def load_dataset(data_dir, target_size=(224, 224), batch_size=32):
    """
    Load image dataset from directory structure.

    Expects folder structure:
        data_dir/
            train/
                NORMAL/
                PNEUMONIA/
            val/
                NORMAL/
                PNEUMONIA/
            test/
                NORMAL/
                PNEUMONIA/

    Args:
        data_dir: Path to root data directory
        target_size: Resize all images to this size (height, width)
        batch_size: Number of images per batch

    Returns:
        train_gen, val_gen, test_gen (Keras ImageDataGenerator iterators)
    """
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')
    test_dir = os.path.join(data_dir, 'test')

    # Training data: rescale pixel values from 0-255 to 0-1
    train_datagen = ImageDataGenerator(rescale=1.0 / 255)
    val_datagen = ImageDataGenerator(rescale=1.0 / 255)
    test_datagen = ImageDataGenerator(rescale=1.0 / 255)

    # Load images from directories
    # class_mode='binary' because we have 2 classes (Normal vs Pneumonia)
    train_gen = train_datagen.flow_from_directory(
        train_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='binary',
        shuffle=True
    )

    val_gen = val_datagen.flow_from_directory(
        val_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='binary',
        shuffle=False
    )

    test_gen = test_datagen.flow_from_directory(
        test_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='binary',
        shuffle=False
    )

    # Print dataset info
    print(f"Training samples: {train_gen.samples}")
    print(f"Validation samples: {val_gen.samples}")
    print(f"Test samples: {test_gen.samples}")
    print(f"Classes: {train_gen.class_indices}")

    return train_gen, val_gen, test_gen


def get_class_weights(train_gen):
    """
    Calculate class weights to handle imbalanced dataset.

    The dataset has more Pneumonia images than Normal images.
    Class weights tell the model to pay more attention to the
    underrepresented class (Normal) during training.

    Args:
        train_gen: Training data generator

    Returns:
        Dictionary mapping class index to weight
    """
    # Get all labels from the training set
    labels = train_gen.classes

    # Compute balanced weights
    weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(labels),
        y=labels
    )

    class_weights = dict(enumerate(weights))

    print("Class weights:")
    for cls, weight in class_weights.items():
        class_name = list(train_gen.class_indices.keys())[cls]
        count = np.sum(labels == cls)
        print(f"  {class_name}: {weight:.3f} ({count} images)")

    return class_weights


def create_validation_split(data_dir, val_ratio=0.15):
    """
    Create a proper validation set from training data.

    The Kaggle dataset has only 16 validation images which is too few.
    This function moves a portion of training images to the validation folder.

    Args:
        data_dir: Path to root data directory
        val_ratio: Fraction of training data to use for validation

    Returns:
        None (moves files on disk)
    """
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')

    for class_name in ['NORMAL', 'PNEUMONIA']:
        train_class_dir = os.path.join(train_dir, class_name)
        val_class_dir = os.path.join(val_dir, class_name)

        # Create validation class directory if it doesn't exist
        os.makedirs(val_class_dir, exist_ok=True)

        # Get all image files
        images = os.listdir(train_class_dir)
        images = [f for f in images if f.lower().endswith(('.jpeg', '.jpg', '.png'))]

        # Calculate number to move
        n_val = int(len(images) * val_ratio)

        # Randomly select images to move
        np.random.seed(42)
        val_images = np.random.choice(images, size=n_val, replace=False)

        # Move images from train to val
        moved = 0
        for img in val_images:
            src = os.path.join(train_class_dir, img)
            dst = os.path.join(val_class_dir, img)
            if not os.path.exists(dst):
                shutil.move(src, dst)
                moved += 1

        print(f"{class_name}: moved {moved} images to validation set")

    print("Validation split complete")