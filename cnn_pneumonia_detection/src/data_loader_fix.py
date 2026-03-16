import os
import shutil
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
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
    Create a proper validation set from training data using stratified sampling.

    Stratified split ensures the same Normal:Pneumonia ratio in both
    train and validation sets. This prevents the val set from being
    accidentally skewed toward one class, which would give unreliable
    validation metrics during training.

    The Kaggle dataset has only 16 validation images which is too few.
    This function moves a portion of training images to the validation folder
    while preserving the class distribution.

    Args:
        data_dir: Path to root data directory
        val_ratio: Fraction of training data to use for validation

    Returns:
        None (moves files on disk)
    """
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')

    # Collect all images and their labels
    all_images = []
    all_labels = []

    for class_name in ['NORMAL', 'PNEUMONIA']:
        class_path = os.path.join(train_dir, class_name)
        images = [f for f in os.listdir(class_path)
                  if f.lower().endswith(('.jpeg', '.jpg', '.png'))]
        all_images.extend([(class_name, img) for img in images])
        all_labels.extend([class_name] * len(images))

    print(f"Total training images before split: {len(all_images)}")
    print(f"  NORMAL: {all_labels.count('NORMAL')}")
    print(f"  PNEUMONIA: {all_labels.count('PNEUMONIA')}")

    # Stratified split — preserves the class ratio in both sets
    # E.g., if train has 75% Pneumonia, val will also have ~75% Pneumonia
    _, val_set, _, val_labels = train_test_split(
        all_images,
        all_labels,
        test_size=val_ratio,
        stratify=all_labels,  # This ensures balanced class ratios
        random_state=42
    )

    # Move selected images from train to val
    moved = {'NORMAL': 0, 'PNEUMONIA': 0}
    for (class_name, img_name), _ in zip(val_set, val_labels):
        src = os.path.join(train_dir, class_name, img_name)
        dst_dir = os.path.join(val_dir, class_name)
        dst = os.path.join(dst_dir, img_name)

        os.makedirs(dst_dir, exist_ok=True)

        if os.path.exists(src) and not os.path.exists(dst):
            shutil.move(src, dst)
            moved[class_name] += 1

    print(f"\nStratified validation split complete:")
    print(f"  NORMAL moved to val: {moved['NORMAL']}")
    print(f"  PNEUMONIA moved to val: {moved['PNEUMONIA']}")

    # Verify the ratio is preserved
    train_after = {c: len([f for f in os.listdir(os.path.join(train_dir, c))
                           if f.lower().endswith(('.jpeg', '.jpg', '.png'))])
                   for c in ['NORMAL', 'PNEUMONIA']}
    val_after = {c: len([f for f in os.listdir(os.path.join(val_dir, c))
                         if f.lower().endswith(('.jpeg', '.jpg', '.png'))])
                 for c in ['NORMAL', 'PNEUMONIA']}

    train_ratio = train_after['PNEUMONIA'] / max(train_after['NORMAL'], 1)
    val_ratio_actual = val_after['PNEUMONIA'] / max(val_after['NORMAL'], 1)

    print(f"\nClass ratio verification:")
    print(f"  Train — NORMAL: {train_after['NORMAL']}, PNEUMONIA: {train_after['PNEUMONIA']} (ratio: {train_ratio:.2f}:1)")
    print(f"  Val   — NORMAL: {val_after['NORMAL']}, PNEUMONIA: {val_after['PNEUMONIA']} (ratio: {val_ratio_actual:.2f}:1)")
    print(f"  Ratios match: {'Yes' if abs(train_ratio - val_ratio_actual) < 0.3 else 'No'}")