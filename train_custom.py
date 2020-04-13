import os
import time

import tensorflow as tf

from model improt ResNet50
from dataset import create_dataset


def train_custom():
    """
    Distributed strategy with custom training loop
    """
    epochs = 2
    strategy = tf.distribute.MirroredStrategy()
    global_batch_size = strategy.num_replicas_in_sync * 32
    train_dataset = create_dataset(global_batch_size)
    train_distribute_dataset = strategy.experimental_distribute_dataset(
        train_dataset)

    with strategy.scope():
        model = ResNet50(input_shape=(224, 224, 3), num_classes=1000)
        loss_object = tf.keras.losses.CategoricalCrossentropy(
            reduction=tf.keras.losses.Reduction.NONE)
        optimizer = tf.keras.optimizers.Adam()

        def train_step(inputs):
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
                delta_t = tf.strings.as_string((tf.timestamp() - t0) / 1000,
                                               precision=1)
                tf.print(delta_t, 'ms/step')
                t0 = tf.timestamp()

        for i in range(0, epochs):
            distributed_train_epoch(train_distribute_dataset)


if __name__ == "__main__":
    t0 = time.time()
    train_custom()
    t1 = time.time()

    print('Custom loop took {} seconds'.format(t1 - t0))
