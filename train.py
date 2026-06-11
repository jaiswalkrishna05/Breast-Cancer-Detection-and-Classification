"""
train.py - Cancer Detection and Classification using CNN

This script implements a complete end‑to‑end pipeline for training and
evaluating convolutional neural network (CNN) models to distinguish
between benign and malignant cancer images.  Two models are provided:

1. **Baseline CNN** – a relatively small network trained from scratch.
2. **Transfer learning model** – based on ResNet50 pretrained on
   ImageNet.  Only the last layers are fine‑tuned.

The script performs the following steps:

* Loads image file paths and labels from a folder structure of the form:

    ``dataset/
        benign/
        malignant/``

* Splits the images into training, validation and test sets in a
  stratified manner so both classes are represented proportionally.
* Preprocesses images (resizes to 224×224 and normalizes pixel values).
* Optionally augments the training data with random flips, rotations,
  zooming and contrast changes to reduce overfitting.
* Builds and trains the baseline CNN and the transfer learning model.
* Plots training/validation accuracy and loss curves and saves them to
  the ``outputs/`` directory.
* Evaluates the models on the test set, generating confusion matrices
  and classification reports (precision, recall, F1‑score, accuracy).
* Saves the trained models to the ``models/`` directory for later use.

Run this script from the root of your project:

    python train.py

Ensure that you have installed the dependencies listed in
``requirements.txt`` before running the script.
"""

import os
import random
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# -----------------------------------------------------------------------------
# Reproducibility
# -----------------------------------------------------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

# Path to the dataset directory.  The directory should contain two
# sub‑directories named ``benign`` and ``malignant``, each containing
# images belonging to that class.
DATASET_DIR = Path('dataset')

# Image size to which all images will be resized.  224×224 is a common
# choice for CNNs and is required by many pretrained networks such as
# ResNet50 and VGG16.
IMG_SIZE = (224, 224)

# Batch size used for training and evaluation.
BATCH_SIZE = 32

# Number of epochs to train the baseline and transfer models.  These
# values can be increased for larger datasets.
EPOCHS_BASELINE = 15
EPOCHS_TRANSFER = 12

# Output directories for saving models and plots.  They will be created
# automatically if they do not exist.
OUTPUT_DIR = Path('outputs')
MODEL_DIR = Path('models')
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
MODEL_DIR.mkdir(exist_ok=True, parents=True)

# Class names in the order in which they will be encoded.  ``benign`` is
# assigned label 0 and ``malignant`` label 1.
CLASS_NAMES = ['benign', 'malignant']


# -----------------------------------------------------------------------------
# Utility Functions
# -----------------------------------------------------------------------------
def collect_image_paths(dataset_dir: Path):
    """Collects image file paths and corresponding labels.

    Args:
        dataset_dir: Path to the dataset directory containing class
                     subdirectories.

    Returns:
        Tuple of numpy arrays (filepaths, labels).

    Raises:
        FileNotFoundError: if required class subdirectories are missing.
        ValueError: if no images are found.
    """
    filepaths = []
    labels = []

    for class_name in CLASS_NAMES:
        class_dir = dataset_dir / class_name
        if not class_dir.exists():
            raise FileNotFoundError(f"Missing folder: {class_dir}")

        # Supported image extensions
        extensions = ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff', '*.webp')
        class_files = []
        for ext in extensions:
            class_files.extend(class_dir.glob(ext))

        for filepath in class_files:
            filepaths.append(str(filepath))
            labels.append(0 if class_name == 'benign' else 1)

    if not filepaths:
        raise ValueError('No images found. Check the dataset structure and file extensions.')

    return np.array(filepaths), np.array(labels)


def split_data(filepaths, labels, test_size=0.15, val_size=0.15):
    """Splits data into stratified train, validation and test sets.

    The dataset is first divided into train+validation and test portions,
    preserving class distribution.  The remaining portion is further split
    into training and validation sets.

    Args:
        filepaths: array of file paths to images.
        labels: array of integer labels.
        test_size: fraction of data to reserve for the test set.
        val_size: fraction of data to reserve for the validation set.

    Returns:
        (X_train, y_train), (X_val, y_val), (X_test, y_test)
    """
    # Split out the test set
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        filepaths, labels, test_size=test_size, random_state=SEED, stratify=labels
    )

    # Compute the fraction of the remaining data to use as validation
    val_fraction_of_trainval = val_size / (1.0 - test_size)

    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval,
        test_size=val_fraction_of_trainval,
        random_state=SEED,
        stratify=y_trainval
    )
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def decode_and_resize(path, label):
    """Reads an image from disk, decodes it into a tensor and resizes it."""
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.cast(img, tf.float32) / 255.0  # normalize to [0, 1]
    return img, tf.cast(label, tf.float32)


def build_dataset(paths, labels, training=False):
    """Builds a tf.data Dataset for training, validation or testing.

    Applies decoding, resizing, normalization and optional data augmentation.
    """
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        ds = ds.shuffle(buffer_size=len(paths), seed=SEED, reshuffle_each_iteration=True)
    ds = ds.map(decode_and_resize, num_parallel_calls=tf.data.AUTOTUNE)
    if training:
        ds = ds.map(lambda img, lbl: (data_augmentation(img, training=True), lbl),
                    num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds


# Define a simple data augmentation pipeline.  This pipeline applies random
# horizontal flips, small rotations, zoom and contrast changes.  These
# operations simulate natural variations in imaging conditions and have been
# shown to improve generalization by reducing overfitting【559546313050346†L109-L123】.

# Data augmentation pipeline
# We define this outside of the dataset function so the same augmentation
# object can be reused; this speeds up graph construction.
data_augmentation = keras.Sequential([
    layers.RandomFlip('horizontal'),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    layers.RandomContrast(0.1),
], name='augmentation')


# -----------------------------------------------------------------------------
# Model Architectures
# -----------------------------------------------------------------------------

def build_baseline_cnn():
    """Builds a simple baseline CNN model from scratch.

    The architecture consists of a series of convolutional layers with
    increasing filter counts followed by global average pooling and a final
    dense layer with sigmoid activation to produce a probability.  A small
    dropout layer is inserted to reduce overfitting.
    """
    inputs = keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    x = layers.Conv2D(32, 3, padding='same', activation='relu')(inputs)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(128, 3, padding='same', activation='relu')(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(256, 3, padding='same', activation='relu')(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    model = keras.Model(inputs, outputs, name='baseline_cnn')
    return model


def build_transfer_resnet50(fine_tune_last_n: int = 30):
    """Builds a transfer learning model based on ResNet50.

    Args:
        fine_tune_last_n: Number of layers at the end of ResNet50 to unfreeze
                          for fine‑tuning.  Setting this to 0 keeps the
                          pretrained base frozen.  Fine‑tuning a subset of
                          layers can improve performance on the target task
                          by adapting high‑level features【624872716780368†L166-L179】.

    Returns:
        A compiled Keras model ready for training.
    """
    base_model = keras.applications.ResNet50(
        include_top=False,
        weights='imagenet',
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)
    )
    # Freeze all layers initially
    base_model.trainable = False

    # Input layer
    inputs = keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    # Rescale to [0, 255] and apply ResNet preprocessing
    x = inputs * 255.0
    x = keras.applications.resnet.preprocess_input(x)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    model = keras.Model(inputs, outputs, name='transfer_resnet50')

    # Optionally unfreeze the last N layers for fine‑tuning
    if fine_tune_last_n > 0:
        base_model.trainable = True
        for layer in base_model.layers[:-fine_tune_last_n]:
            layer.trainable = False
    return model


def compile_model(model, lr=1e-4):
    """Compiles a Keras model with Binary Cross‑Entropy and Adam optimizer.

    Binary cross‑entropy is appropriate for binary classification problems
    because it compares the predicted probabilities with the true labels and
    penalizes deviations【713855461409482†L149-L167】.  The Adam optimizer adapts
    learning rates for each parameter and often performs well in practice.
    """
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss=keras.losses.BinaryCrossentropy(),
        metrics=[keras.metrics.BinaryAccuracy(name='accuracy')]
    )
    return model


def plot_history(history: keras.callbacks.History, title: str, save_dir: Path):
    """Plots training and validation loss/accuracy curves.

    Saves two PNG files: one for loss and one for accuracy.
    """
    loss_path = save_dir / f"{title}_loss.png"
    acc_path = save_dir / f"{title}_accuracy.png"

    # Loss plot
    plt.figure(figsize=(6, 4))
    plt.plot(history.history['loss'], label='train_loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Binary Cross‑Entropy Loss')
    plt.title(f'{title} – Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(loss_path, dpi=200)
    plt.close()

    # Accuracy plot
    plt.figure(figsize=(6, 4))
    plt.plot(history.history['accuracy'], label='train_accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(f'{title} – Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.savefig(acc_path, dpi=200)
    plt.close()


def evaluate_model(model, dataset, true_labels, title: str, save_dir: Path):
    """Evaluates a trained model on the test dataset and reports metrics.

    Generates a confusion matrix and prints a classification report.
    """
    # Predict probabilities and convert to class labels
    probs = model.predict(dataset, verbose=0).ravel()
    preds = (probs >= 0.5).astype(int)

    # Confusion matrix
    cm = confusion_matrix(true_labels, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
    fig, ax = plt.subplots(figsize=(4, 4))
    disp.plot(ax=ax, cmap='Blues', colorbar=False)
    plt.title(f'{title} – Confusion Matrix')
    plt.tight_layout()
    fig_path = save_dir / f'{title}_confusion_matrix.png'
    plt.savefig(fig_path, dpi=200)
    plt.close()

    # Classification report
    print('\n' + '=' * 70)
    print(f'{title} – Classification Report (Test Set)')
    print('=' * 70)
    print(classification_report(true_labels, preds, target_names=CLASS_NAMES, digits=4))


# -----------------------------------------------------------------------------
# Main Routine
# -----------------------------------------------------------------------------
def main():
    print(f'TensorFlow version: {tf.__version__}')
    filepaths, labels = collect_image_paths(DATASET_DIR)
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = split_data(filepaths, labels)

    print(f'Total images: {len(filepaths)}')
    print(f'Train: {len(X_train)} | Validation: {len(X_val)} | Test: {len(X_test)}')

    # Build datasets
    train_ds = build_dataset(X_train, y_train, training=True)
    val_ds = build_dataset(X_val, y_val, training=False)
    test_ds = build_dataset(X_test, y_test, training=False)

    # Early stopping to prevent overfitting
    early_stop = keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=4, restore_best_weights=True
    )

    # -------------------------------------------------------------------------
    # Train baseline CNN
    # -------------------------------------------------------------------------
    baseline_model = build_baseline_cnn()
    compile_model(baseline_model, lr=1e-4)
    baseline_model.summary()
    history_baseline = baseline_model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS_BASELINE,
        callbacks=[early_stop],
        verbose=1
    )
    # Save baseline model
    baseline_model.save(MODEL_DIR / 'baseline_cnn.keras')
    # Plot training curves
    plot_history(history_baseline, 'baseline_cnn', OUTPUT_DIR)
    # Evaluate baseline model
    evaluate_model(baseline_model, test_ds, y_test, 'baseline_cnn', OUTPUT_DIR)

    # -------------------------------------------------------------------------
    # Train transfer learning model (ResNet50)
    # -------------------------------------------------------------------------
    transfer_model = build_transfer_resnet50(fine_tune_last_n=30)
    compile_model(transfer_model, lr=1e-5)  # smaller LR for fine‑tuning
    transfer_model.summary()
    history_transfer = transfer_model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS_TRANSFER,
        callbacks=[early_stop],
        verbose=1
    )
    # Save transfer learning model
    transfer_model.save(MODEL_DIR / 'transfer_resnet50.keras')
    # Plot training curves
    plot_history(history_transfer, 'transfer_resnet50', OUTPUT_DIR)
    # Evaluate transfer learning model
    evaluate_model(transfer_model, test_ds, y_test, 'transfer_resnet50', OUTPUT_DIR)

    print('\nTraining complete. Models and plots are saved under the respective directories.')


if __name__ == '__main__':
    main()

