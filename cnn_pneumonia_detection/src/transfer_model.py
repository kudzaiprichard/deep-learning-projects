from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, Input
from tensorflow.keras.applications import ResNet50, VGG16
from tensorflow.keras.optimizers import Adam


def build_transfer_model(input_shape=(224, 224, 3), base_model_name="resnet50",
                         learning_rate=0.0001, fine_tune_layers=0):
    """
    Build a transfer learning model using a pre-trained base.

    Transfer learning: Instead of training from scratch, we use a model
    that was already trained on ImageNet (1.4M images, 1000 classes).
    This model already knows how to detect edges, textures, shapes, etc.
    We only need to teach it the final step: is this Normal or Pneumonia?

    Process:
    1. Load pre-trained model WITHOUT its classification head
    2. Freeze the pre-trained layers (don't change what it already learned)
    3. Add our own classification head (Dense layers for our 2 classes)
    4. Optionally unfreeze some top layers for fine-tuning

    Args:
        input_shape: Shape of input images (height, width, channels)
        base_model_name: 'resnet50' or 'vgg16'
        learning_rate: Learning rate for optimizer
        fine_tune_layers: Number of top layers to unfreeze for fine-tuning
                          (0 = freeze all base layers)

    Returns:
        Compiled Keras model
    """
    input_tensor = Input(shape=input_shape)

    # Load pre-trained model without the top classification layers
    # weights='imagenet' loads weights trained on 1.4M images
    # include_top=False removes the original 1000-class output layer
    if base_model_name == "resnet50":
        base_model = ResNet50(
            weights='imagenet',
            include_top=False,
            input_tensor=input_tensor
        )
    elif base_model_name == "vgg16":
        base_model = VGG16(
            weights='imagenet',
            include_top=False,
            input_tensor=input_tensor
        )
    else:
        raise ValueError(f"Unknown base model: {base_model_name}. Use 'resnet50' or 'vgg16'")

    # Freeze base model layers — keep the pre-trained knowledge intact
    # These layers already know how to extract features from images
    base_model.trainable = False

    # Optionally unfreeze the top N layers for fine-tuning
    # This lets the model slightly adjust the pre-trained features
    # to better fit our specific task (chest X-rays)
    if fine_tune_layers > 0:
        base_model.trainable = True
        for layer in base_model.layers[:-fine_tune_layers]:
            layer.trainable = False
        print(f"Fine-tuning last {fine_tune_layers} layers of {base_model_name}")
    else:
        print(f"All {base_model_name} layers frozen (feature extraction only)")

    # Add our custom classification head
    x = base_model.output

    # GlobalAveragePooling: Reduces each feature map to a single number
    # E.g., from (7, 7, 2048) to (2048,) — much smaller than Flatten
    x = GlobalAveragePooling2D()(x)

    # Dense layers to learn classification
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)

    # Output: probability of Pneumonia (0 = Normal, 1 = Pneumonia)
    output = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=input_tensor, outputs=output)

    # Use a smaller learning rate for transfer learning
    # We don't want to destroy the pre-trained weights with large updates
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    # Print trainable vs frozen layer counts
    trainable = sum(1 for layer in model.layers if layer.trainable)
    frozen = sum(1 for layer in model.layers if not layer.trainable)
    print(f"Total layers: {len(model.layers)} (trainable: {trainable}, frozen: {frozen})")

    return model, base_model