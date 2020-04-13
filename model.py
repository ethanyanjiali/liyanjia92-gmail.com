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