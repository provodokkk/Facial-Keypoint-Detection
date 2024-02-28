import matplotlib.pyplot as plt
import numpy as np
import os

from typing import Any, Tuple


def evaluation_keypoints_plot(image: np.ndarray, outputs: np.ndarray, orig_keypoints: np.ndarray, save_path: str,
                              image_shape: Tuple[int, int] = (96, 96)) -> None:
    """
    Plot and save an image with original and regressed (predicted) keypoints.
    The original keypoints are green and the predicted keypoints are red.

    Args:
        image: The image as a numpy array, expected shape is (height*width,).
        outputs: Predicted keypoints as a flat array.
        orig_keypoints: Original keypoints as a flat array.
        save_path: Path where the plot will be saved.
        image_shape: A tuple indicating the shape of the image for reshaping, default is (96, 96).

    Returns:
        None
    """

    assert image.size == image_shape[0] * image_shape[1], "Image size does not match the provided shape."

    output_keypoint = outputs.reshape(-1, 2)
    orig_keypoint = orig_keypoints.reshape(-1, 2)
    image = image.reshape(*image_shape)
    plt.style.use('default')
    plt.imshow(image, cmap='gray')
    plt.axis('off')

    for p, (predicted, original) in enumerate(zip(output_keypoint, orig_keypoint)):
        plt.plot(predicted[0], predicted[1], 'r.')
        plt.text(predicted[0], predicted[1], f"{p}")
        plt.plot(original[0], original[1], 'g.')
        plt.text(original[0], original[1], f"{p}")

    plt.savefig(save_path)
    plt.close()


def test_keypoints_plot(image: np.ndarray, outputs: np.ndarray, save_path: str,
                        image_shape: Tuple[int, int] = (96, 96)) -> None:
    """
    Plot keypoints from the test dataset and save the image.

    Args:
        image: The image as a numpy array, expected shape is (height*width,).
        outputs: Predicted keypoints as a flat array.
        save_path: Path where the plot will be saved.
        image_shape: A tuple indicating the shape of the image for reshaping, default is (96, 96).

    Returns:
        None
    """

    assert image.size == image_shape[0] * image_shape[1], "Image size does not match the provided shape."

    output_keypoint = outputs.reshape(-1, 2)
    image = image.reshape(*image_shape)
    plt.style.use('default')
    plt.imshow(image, cmap='gray')
    plt.axis('off')

    for p, point in enumerate(output_keypoint):
        plt.plot(point[0], point[1], 'r.')
        plt.text(point[0], point[1], f'{p}')

    plt.savefig(save_path)
    plt.close()


def dataset_keypoints_plot(data: np.ndarray) -> None:
    """
    The function plots the first 30 images and their keypoints from the dataset.

    Args:
        data: An array containing images and their keypoints.
        Structure is [images, keypoints],
        where images should be reshaped to 96x96 pixels and keypoints reshaped to (-1, 2).

    Returns:
        None
    """

    plt.figure(figsize=(20, 40))
    for i in range(30):
        img = np.array(data[0][0][i], dtype='float32').reshape(96, 96)
        plt.subplot(5, 6, i + 1)
        plt.imshow(img, cmap='gray')
        plt.axis('off')

        keypoints = data[0][1][i].reshape(-1, 2)
        for x, y in keypoints:
            plt.plot(x, y, 'r.')

    plt.show()
    plt.close()


def save_plots(history: Any) -> None:
    """
    Accepts the training history object and saves to disk a plot of training and validation loss.

    Args:
        history: An object containing the training history of a model.
        It includes metrics such as loss and accuracy for each epoch, both for training and validation.

    Returns:
        None
    """
    train_loss = history.history['loss']
    valid_loss = history.history['val_loss']

    plt.figure(figsize=(12, 9))
    plt.plot(train_loss, color='green', linestyle='-', label='Train Loss')
    plt.plot(valid_loss, color='red', linestyle='-', label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.savefig(os.path.join('..', 'outputs', 'loss.png'))
    plt.show()
