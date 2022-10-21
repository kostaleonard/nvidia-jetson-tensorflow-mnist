"""Trains a neural network on MNIST."""

import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten

MNIST_IMAGE_SHAPE = (28, 28)
MAX_PIXEL_VALUE = 255
BATCH_SIZE = 512

# TODO explain this in issue
# https://forums.developer.nvidia.com/t/jetson-nano-running-out-of-memory-resourceexhaustederror-oom-when-allocating-tensor-with-shape-3-3-512-1024/154513/5
devices = tf.config.list_physical_devices('GPU')
if devices:
    tf.config.experimental.set_memory_growth(devices[0], True)
    tf.config.experimental.set_virtual_device_configuration(devices[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])

# TODO log GPU stats if possible
# TODO use a TF dataset or something to repeat the dataset and show speedups? Or is that too complicated for an MNIST example?


def main() -> None:
    """Trains a neural network on MNIST."""
    # TODO clean this up
    devices = tf.config.get_visible_devices()
    print(f"Found the following devices: {devices}")
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = tf.cast(X_train, tf.float32) / MAX_PIXEL_VALUE
    X_test = tf.cast(X_test, tf.float32) / MAX_PIXEL_VALUE
    print(X_train.shape)
    print(y_train.shape)
    print(X_test.shape)
    print(y_test.shape)
    model = Sequential([
        Flatten(input_shape=MNIST_IMAGE_SHAPE, name='flatten'),
        Dense(128, activation='relu'),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['sparse_categorical_accuracy'])
    model.fit(X_train, y_train, epochs=10, batch_size=BATCH_SIZE)
    metrics = model.evaluate(X_test, y_test, batch_size=BATCH_SIZE)
    print(metrics)


if __name__ == "__main__":
    main()
