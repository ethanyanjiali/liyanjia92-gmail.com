import tensorflow as tf


def preprocess(number):
    """
    Fake preprocess function for tf.data.Dataset
    """
    label = tf.zeros([1000])
    image = tf.zeros([256, 256, 3])
    image = tf.image.random_flip_left_right(image)
    image = tf.image.resize(image, (224, 224))
    return image, label


def create_dataset(global_batch_size):
    """
    Create a tf.data.Dataset
    """
    dataset = tf.data.Dataset.range(1000)
    dataset = dataset.map(preprocess,
                          num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.shuffle(256)
    dataset = dataset.batch(global_batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset
