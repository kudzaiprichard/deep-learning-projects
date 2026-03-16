import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, auc,
    accuracy_score, precision_score, recall_score, f1_score
)


def evaluate_model(model, test_gen):
    """
    Run predictions on test data and return predictions + true labels.

    Args:
        model: Trained Keras model
        test_gen: Test data generator

    Returns:
        Dictionary with predictions, probabilities, true labels, and class names
    """
    # Reset generator to start from the beginning
    test_gen.reset()

    # Get predictions (probabilities between 0 and 1)
    y_prob = model.predict(test_gen, steps=int(np.ceil(test_gen.samples / test_gen.batch_size)))
    y_prob = y_prob.flatten()

    # Convert probabilities to class predictions (0 or 1)
    # Threshold of 0.5: above = Pneumonia, below = Normal
    y_pred = (y_prob >= 0.5).astype(int)

    # True labels
    y_true = test_gen.classes

    # Ensure same length (last batch might be partial)
    min_len = min(len(y_true), len(y_pred))
    y_true = y_true[:min_len]
    y_pred = y_pred[:min_len]
    y_prob = y_prob[:min_len]

    # Class names
    class_names = list(test_gen.class_indices.keys())

    # Print summary metrics
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    print("=" * 50)
    print("MODEL EVALUATION")
    print("=" * 50)
    print(f"Accuracy:  {acc:.4f} ({acc * 100:.2f}%)")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print("=" * 50)

    return {
        "y_true": y_true,
        "y_pred": y_pred,
        "y_prob": y_prob,
        "class_names": class_names,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1
    }


def classification_report_dict(y_true, y_pred, class_names):
    """
    Generate and print a detailed classification report.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names

    Returns:
        Classification report as dictionary
    """
    report = classification_report(y_true, y_pred, target_names=class_names)
    print("\nDetailed Classification Report:")
    print(report)
    return classification_report(y_true, y_pred, target_names=class_names, output_dict=True)


def plot_confusion_matrix(y_true, y_pred, class_names, title="Confusion Matrix"):
    """
    Plot a confusion matrix heatmap.

    The confusion matrix shows:
    - True Positives: Correctly predicted Pneumonia
    - True Negatives: Correctly predicted Normal
    - False Positives: Normal predicted as Pneumonia (false alarm)
    - False Negatives: Pneumonia predicted as Normal (missed case — dangerous!)

    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        title: Plot title
    """
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                annot_kws={"size": 16})
    plt.title(title, fontsize=14)
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('Actual', fontsize=12)
    plt.tight_layout()
    plt.show()

    # Print interpretation
    tn, fp, fn, tp = cm.ravel()
    print(f"\nTrue Negatives (correct Normal): {tn}")
    print(f"False Positives (false alarm): {fp}")
    print(f"False Negatives (missed Pneumonia): {fn}")
    print(f"True Positives (correct Pneumonia): {tp}")


def plot_roc_curve(y_true, y_prob, title="ROC Curve"):
    """
    Plot the ROC curve and calculate AUC score.

    ROC curve shows the tradeoff between:
    - True Positive Rate (Recall): catching real Pneumonia cases
    - False Positive Rate: incorrectly flagging Normal as Pneumonia

    AUC (Area Under Curve): 1.0 = perfect, 0.5 = random guessing

    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        title: Plot title
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', linewidth=2, label=f'ROC Curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--', label='Random (AUC = 0.5)')
    plt.fill_between(fpr, tpr, alpha=0.1, color='blue')
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate (Recall)', fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.show()

    print(f"AUC Score: {roc_auc:.4f}")
    return roc_auc


def plot_training_history(history, title="Training History"):
    """
    Plot training and validation accuracy/loss over epochs.

    Args:
        history: Keras training history object
        title: Plot title
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy plot
    axes[0].plot(history.history['accuracy'], label='Train Accuracy')
    axes[0].plot(history.history['val_accuracy'], label='Val Accuracy')
    axes[0].set_title(f'{title} - Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()

    # Loss plot
    axes[1].plot(history.history['loss'], label='Train Loss')
    axes[1].plot(history.history['val_loss'], label='Val Loss')
    axes[1].set_title(f'{title} - Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()

    plt.tight_layout()
    plt.show()