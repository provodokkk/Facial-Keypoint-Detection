"""
Script to evaluate the model on the validation dataset and
test it on the test dataset. Also, plot all the results and save them to disk.
"""
import os
import tensorflow as tf
import pandas as pd
import numpy as np

from tqdm import tqdm
from dataset import get_data
from utils import evaluation_keypoints_plot, test_keypoints_plot


# Create directories for saving validation and test results
validation_result_path = './outputs/validation_results'
test_result_path = './outputs/test_results'
os.makedirs(validation_result_path, exist_ok=True)
os.makedirs(test_result_path, exist_ok=True)

# Load the pre-trained model
model = tf.keras.models.load_model('./outputs/saved_model')
print(model.summary())


def evaluate(valid_ds):
    """Evaluate the model on the validation dataset and save result plot to disk."""

    results = model.predict(valid_ds)
    counter = 0

    for batch in tqdm(valid_ds, total=len(valid_ds)):
        images, keypoints = batch
        for image, true_keypoints in zip(images, keypoints):
            evaluation_keypoints_plot(
                image, results[counter], true_keypoints,
                save_path=os.path.join(validation_result_path, f'{counter}.png')
            )
            counter += 1


def test(test_csv_path):
    """Predict keypoints for images in `test.csv` and save the plots."""

    test_df = pd.read_csv(test_csv_path)
    images = test_df.Image
    for i in tqdm(range(len(images)), total=len(images)):
        image = images.iloc[i].split(' ')
        image = np.array(image, dtype=np.float32) / 255.
        image = image.reshape(96, 96)
        image = image.reshape(96, 96, 1)
        image_batch = np.expand_dims(image, axis=0)
        image_tensor = tf.convert_to_tensor(image_batch)
        outputs = model.predict(image_tensor)
        test_keypoints_plot(
            image, outputs,
            save_path=os.path.join(test_result_path, f'{i}.png')
        )


if __name__ == '__main__':
    print('Evaluating...')
    _, valid_ds = get_data()
    evaluate(valid_ds)

    print('Testing...')
    test(test_csv_path='./input/test.csv')
