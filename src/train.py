import config
import tensorflow as tf

from model import build_model
from dataset import get_data
from utils import save_plots


def train_and_evaluate():
    # Model checkpoint callback.
    model_ckpt = tf.keras.callbacks.ModelCheckpoint(
        filepath='./outputs/saved_model',
        monitor='val_loss',
        mode='auto',
        save_best_only=True
    )

    # Load the training and validation data
    train_ds, valid_ds = get_data()

    # Build and compile the model
    model = build_model((config.IMG_RESIZE, config.IMG_RESIZE))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE),
        loss=tf.keras.losses.MeanSquaredError(),
    )
    print(model.summary())

    # Train the model
    history = model.fit(
        train_ds,
        validation_data=valid_ds,
        epochs=config.EPOCHS,
        callbacks=[model_ckpt],
    )

    # Save training and validation loss plots
    save_plots(history)


if __name__ == '__main__':
    train_and_evaluate()
