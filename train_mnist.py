"""Trains a neural network on MNIST."""

import sys
from typing import List
from argparse import ArgumentParser, Namespace
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten

MNIST_IMAGE_SHAPE = (28, 28)
MAX_PIXEL_VALUE = 255


def configure_gpu(memory_limit: int) -> None:
    """Configures the GPU on the Jetson for training.

    :param memory_limit: The memory limit for the GPU, in MiB.
    """
    devices = tf.config.list_physical_devices("GPU")
    if devices:
        tf.config.experimental.set_memory_growth(devices[0], True)
        tf.config.experimental.set_virtual_device_configuration(
            devices[0],
            [
                tf.config.experimental.VirtualDeviceConfiguration(
                    memory_limit=memory_limit
                )
            ],
        )
    else:
        raise ValueError("No GPU found.")


def parse_args(args: List[str]) -> Namespace:
    """Returns the processed command line arguments.

    :param args: The command line arguments; sys.argv[1:].
    :return: The processed command line arguments.
    """
    parser = ArgumentParser(description="Trains a neural network on MNIST.")
    parser.add_argument(
        "--memory_limit",
        help="The memory limit for the GPU, in MiB. Some memory is required "
        "for performing operations on data, so the device's physical "
        "size cannot necessarily be used. Moderator posts on the NVIDIA "
        "forums (see issue #1) state that Jetson Nano 2 GB should use "
        "1024 MiB.",
        type=int,
        default=1024,
    )
    parser.add_argument(
        "--batch_size",
        help="The number of samples to use in each batch.",
        type=int,
        default=512,
    )
    return parser.parse_args(args)


def main() -> None:
    """Trains a neural network on MNIST."""
    args = parse_args(sys.argv[1:])
    devices = tf.config.get_visible_devices()
    print(f"Found the following devices: {devices}")
    configure_gpu(args.memory_limit)
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = tf.cast(X_train, tf.float32) / MAX_PIXEL_VALUE
    X_test = tf.cast(X_test, tf.float32) / MAX_PIXEL_VALUE
    print(X_train.shape)
    print(y_train.shape)
    print(X_test.shape)
    print(y_test.shape)
    model = Sequential(
        [
            Flatten(input_shape=MNIST_IMAGE_SHAPE, name="flatten"),
            Dense(128, activation="relu"),
            Dense(10, activation="softmax"),
        ]
    )
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["sparse_categorical_accuracy"],
    )
    model.fit(X_train, y_train, epochs=10, batch_size=args.batch_size)
    metrics = model.evaluate(X_test, y_test, batch_size=args.batch_size)
    print(metrics)


if __name__ == "__main__":
    main()
