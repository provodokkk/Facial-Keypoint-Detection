import cv2
import pandas as pd
import numpy as np
import config
import utils

from tensorflow.keras.utils import Sequence
from tqdm import tqdm
from typing import Tuple

resize = config.IMG_RESIZE


def train_test_split(csv_path: str, split: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Cleans up the data by dropping all the rows with missing values.
    Splits the data into a train and validation set.

    Args:
        csv_path: The file path to the CSV file containing the dataset.
        split: The fraction of the data to be used for the validation set.

    Returns:
        A tuple containing two DataFrames: (training_samples, valid_samples).
    """

    if not 0 < split < 1:
        raise ValueError("split must be a float between 0 and 1.")

    df_data = pd.read_csv(csv_path)
    # Drop all rows with missing values
    df_data = df_data.dropna()

    # Calculate indices for splitting data
    valid_split_index = int(len(df_data) * (1 - split))

    # Split data
    training_samples = df_data.iloc[:valid_split_index]
    valid_samples = df_data.iloc[valid_split_index:]

    print(f"Training sample instances: {len(training_samples)}")
    print(f"Validation sample instances: {len(valid_samples)}")

    return training_samples, valid_samples


class FaceKeypointDataset(Sequence):
    def __init__(self, samples: pd.DataFrame, batch_size: int):
        """
        Initializes the dataset object.

        Args:
            samples (pd.DataFrame): DataFrame containing the samples and their keypoints.
            batch_size (int): Size of the batch to be used during training/validation.
        """
        self.batch_size = batch_size
        self.data = samples

        # Pre-process and store image pixels
        self.images = np.array(
            [np.fromstring(img, sep=' ', dtype='float32').reshape(96, 96) for img in tqdm(self.data['Image'])])

    def __len__(self) -> int:
        """Returns the number of batches per epoch."""
        return len(self.data) // self.batch_size

    def __getitem__(self, index: int) -> tuple:
        """
        Generates one batch of data.

        Args:
            index (int): Batch index.

        Returns:
            tuple: Tuple containing batch images and keypoints.
        """
        start = index * self.batch_size
        end = (index + 1) * self.batch_size
        batch_images = self.images[start:end]
        batch_keypoints = self.data.iloc[start:end, :-1].to_numpy(dtype='float32')

        # Process and prepare the batch of images for the model
        final_images = np.array(
            # Resize each image in the batch to (resize)x(resize) pixels
            # Reshape to add a singleton dimension for grayscale channel
            # Normalize pixel values to the range [0, 1]
            [cv2.resize(img, (resize, resize)).reshape(resize, resize, 1) / 255.0 for img in batch_images])

        # Adjust keypoints for the corresponding batch and prepare for the model
        final_keypoints = (
            # Reshape keypoints into pairs (x, y) for each point
            # Scale keypoints coordinates
            np.array([kp.reshape(-1, 2) * [resize / 96, resize / 96] for kp in batch_keypoints])
            # Reshape keypoints array to match the number of images in the batch
            # Flatten keypoints for each image into a single array for model input
            .reshape(len(batch_images), -1))

        return final_images, final_keypoints


def get_data():
    # Get the training and validation data samples.
    training_samples, valid_samples = train_test_split(f"{config.INPUT_PATH}/training.csv", config.TEST_SPLIT)

    train_ds = FaceKeypointDataset(training_samples, batch_size=config.BATCH_SIZE)
    valid_ds = FaceKeypointDataset(valid_samples, batch_size=config.BATCH_SIZE)

    return train_ds, valid_ds


if __name__ == '__main__':
    # Show dataset keypoint plots
    if config.SHOW_DATASET_PLOT:
        _, valid_ds = get_data()
        utils.dataset_keypoints_plot(valid_ds)
