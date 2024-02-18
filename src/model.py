import tensorflow as tf
from tensorflow.keras import layers
from typing import Tuple


def build_model(image_size: Tuple[int, int]):
    """
    Builds a CNN model for keypoint detection.

    Args:
        image_size: A tuple specifying the input image dimensions (height, width).

    Returns:
        A TensorFlow model constructed with Keras functional API.
    """

    # Input layer
    inputs = layers.Input(shape=(image_size[0], image_size[1], 1))

    # First convolutional block
    x = layers.Conv2D(32, kernel_size=(5, 5))(inputs)
    x = layers.ReLU()(x)
    x = layers.MaxPool2D((2, 2))(x)

    # Second convolutional block
    x = layers.Conv2D(64, kernel_size=(3, 3))(x)
    x = layers.ReLU()(x)
    x = layers.MaxPool2D((2, 2))(x)

    # Third convolutional block
    x = layers.Conv2D(128, kernel_size=(3, 3))(x)
    x = layers.ReLU()(x)
    x = layers.MaxPool2D((2, 2))(x)

    # Fourth convolutional block
    x = layers.Conv2D(256, kernel_size=(3, 3))(x)
    x = layers.ReLU()(x)
    x = layers.MaxPool2D((2, 2))(x)

    # Fifth convolutional block
    x = layers.Conv2D(512, kernel_size=(3, 3))(x)
    x = layers.ReLU()(x)
    x = layers.MaxPool2D((2, 2))(x)

    # Global Average Pooling
    x = layers.GlobalAveragePooling2D()(x)

    # Dense layers for prediction
    x = layers.Dense(units=256)(x)
    x = layers.ReLU()(x)
    x = layers.Dense(units=30)(x)

    # Construct the model
    model = tf.keras.Model(inputs, outputs=x)

    return model
