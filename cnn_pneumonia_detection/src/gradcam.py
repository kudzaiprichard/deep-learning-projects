import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import tensorflow as tf


def make_gradcam_heatmap(img_array, model, last_conv_layer_name=None):
    """
    Generate a Grad-CAM heatmap for an image.

    Grad-CAM (Gradient-weighted Class Activation Mapping) shows which
    regions of the image the model focused on to make its prediction.

    How it works:
    1. Feed the image through the model
    2. Get the gradients of the prediction with respect to the last
       convolutional layer's output
    3. Weight each feature map by its average gradient (importance)
    4. Combine weighted feature maps into a heatmap
    5. The brighter areas = where the model looked most

    For chest X-rays, you'd expect the model to focus on the lung
    regions, especially areas with opacity (white patches) for pneumonia.

    Args:
        img_array: Preprocessed image array with shape (1, H, W, 3)
        model: Trained Keras model
        last_conv_layer_name: Name of the last conv layer to visualize
                              (auto-detected if None)

    Returns:
        Heatmap as numpy array (H, W) with values 0-1
    """
    # Auto-detect last convolutional layer if not specified
    if last_conv_layer_name is None:
        for layer in reversed(model.layers):
            if 'conv' in layer.name.lower():
                last_conv_layer_name = layer.name
                break
        if last_conv_layer_name is None:
            raise ValueError("No convolutional layer found in model")

    # Create a model that outputs both the conv layer output and the prediction
    grad_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=[
            model.get_layer(last_conv_layer_name).output,
            model.output
        ]
    )

    # Record operations for automatic differentiation
    with tf.GradientTape() as tape:
        # Forward pass: get conv outputs and prediction
        conv_outputs, predictions = grad_model(img_array)

        # For binary classification, use the single output neuron
        predicted_class = predictions[:, 0]

    # Get gradients of the prediction with respect to conv layer output
    # This tells us how much each feature map contributed to the prediction
    grads = tape.gradient(predicted_class, conv_outputs)

    # Average gradients across spatial dimensions (global average pooling)
    # Result: one importance weight per feature map
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Weight each feature map by its importance
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Normalize heatmap to 0-1 range using ReLU (only positive contributions)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)

    return heatmap.numpy()


def display_gradcam(img_array, heatmap, prediction, class_names, alpha=0.4):
    """
    Display the original image with Grad-CAM heatmap overlay.

    Args:
        img_array: Original image array (H, W, 3) with values 0-255
        heatmap: Grad-CAM heatmap from make_gradcam_heatmap
        prediction: Model's prediction probability
        class_names: List of class names ['NORMAL', 'PNEUMONIA']
        alpha: Transparency of heatmap overlay (0 = invisible, 1 = opaque)
    """
    # Resize heatmap to match original image size
    heatmap_resized = np.uint8(255 * heatmap)
    heatmap_resized = tf.image.resize(
        heatmap_resized[..., np.newaxis],
        (img_array.shape[0], img_array.shape[1])
    ).numpy().squeeze()

    # Apply colormap (jet: blue=cold/low, red=hot/high attention)
    colormap = cm.get_cmap('jet')
    heatmap_colored = colormap(np.uint8(255 * heatmap_resized / (heatmap_resized.max() + 1e-8)))
    heatmap_colored = np.uint8(255 * heatmap_colored[:, :, :3])

    # Normalize original image to 0-255 if needed
    if img_array.max() <= 1.0:
        img_display = np.uint8(img_array * 255)
    else:
        img_display = np.uint8(img_array)

    # Overlay heatmap on original image
    superimposed = np.uint8(heatmap_colored * alpha + img_display * (1 - alpha))

    # Determine prediction label
    pred_class = 1 if prediction >= 0.5 else 0
    pred_label = class_names[pred_class]
    confidence = prediction if pred_class == 1 else 1 - prediction

    # Plot original, heatmap, and overlay side by side
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(img_display)
    axes[0].set_title('Original X-Ray', fontsize=12)
    axes[0].axis('off')

    axes[1].imshow(heatmap_resized, cmap='jet')
    axes[1].set_title('Grad-CAM Heatmap', fontsize=12)
    axes[1].axis('off')

    axes[2].imshow(superimposed)
    axes[2].set_title(f'Prediction: {pred_label} ({confidence:.1%})', fontsize=12)
    axes[2].axis('off')

    plt.tight_layout()
    plt.show()