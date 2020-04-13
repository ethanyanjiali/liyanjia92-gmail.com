import os
import time

import tensorflow as tf
from tensorflow.keras.layers import (
    GlobalAveragePooling2D,
    Dense,
    Softmax,
)

from model import ResNet50
from dataset import create_dataset


def train_keras():
    """
    Distributed strategy with Keras API
    """
    epochs = 2
    strategy = tf.distribute.MirroredStrategy()
    global_batch_size = strategy.num_replicas_in_sync * 32
    train_dataset = create_dataset(global_batch_size)

    with strategy.scope():
        model = ResNet50(input_shape=(224, 224, 3), num_classes=1000)
        model.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
                      optimizer=tf.keras.optimizers.Adam(),
                      metrics=['accuracy'])
        model.fit(train_dataset, epochs=epochs)


if __name__ == "__main__":
    t0 = time.time()
    train_keras()
    t1 = time.time()

    print('Keras API loop took {} seconds'.format(t1 - t0))
