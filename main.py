import os
import time

import tensorflow as tf
from tensorflow.keras.layers import (
    GlobalAveragePooling2D,
    Dense,
    Softmax,
)


def ResNet50(input_shape, num_classes):
    base_model = tf.keras.applications.ResNet50(input_shape=input_shape,
                                                include_top=False)
    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(num_classes),
        Softmax(),
    ])

    return model


def preprocess(number):
    """
    Fake preprocess function for tf.data.Dataset
    """
    label = tf.zeros([1000])
    image = tf.zeros([256, 256, 3])
    image = tf.image.random_flip_left_right(image)
    image = tf.image.resize(image, (224, 224))
    return image, label


def create_dataset():
    """
    Create a tf.data.Dataset
    """
    dataset = tf.data.Dataset.range(10000)
    dataset = dataset.map(preprocess,
                          num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.shuffle(256)
    dataset = dataset.batch(32)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset


def train_keras():
    """
    Distributed strategy with Keras API
    """
    epochs = 2
    strategy = tf.distribute.MirroredStrategy()
    global_batch_size = strategy.num_replicas_in_sync * batch_size
    train_dataset = create_dataset()

    with strategy.scope():
        model = ResNet50(input_shape=(224, 224, 3), num_classes=1000)
        model.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
                      optimizer=tf.keras.optimizers.Adam(),
                      metrics=['accuracy'])
        model.fit(train_dataset, epochs=epochs)


def train_custom():
    """
    Distributed strategy with custom training loop
    """
    epochs = 2
    strategy = tf.distribute.MirroredStrategy()
    global_batch_size = strategy.num_replicas_in_sync * batch_size
    train_dataset = create_dataset()

    with strategy.scope():
        model = ResNet50(input_shape=(224, 224, 3), num_classes=1000)
        loss_object = tf.keras.losses.CategoricalCrossentropy()

        def train_step(self, inputs):
            images, labels = inputs
            with tf.GradientTape() as tape:
                outputs = model(images, training=True)
                loss = loss_object(labels, outputs)

            grads = tape.gradient(target=loss,
                                  sources=model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            return loss

        @tf.function
        def distributed_train_epoch(dataset):
            t0 = tf.timestamp()
            for one_batch in dataset:
                per_replica_loss = strategy.experimental_run_v2(
                    train_step, args=(one_batch, ))
                strategy.reduce(tf.distribute.ReduceOp.SUM,
                                per_replica_loss,
                                axis=None)
                tf.print((tf.timestamp() - t0) / 1000, 'ms/step')
                t0 = tf.timestamp()

        for i in range(0, epochs):
            distributed_train_epoch(train_distributed_dataset)


if __name__ == "__main__":
    t0 = time.time()
    train_custom()
    t1 = time.time()
    train_keras()
    t2 = time.time()

    print('Custom loop took {} seconds'.format(t1 - t0))
    print('Keras API loop took {} seconds'.format(t2 - t1))
