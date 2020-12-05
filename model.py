"""


"""
################################################################################
# Imports
from parameters import *


################################################################################
# Discriminator
def build_discriminator():
    def block(t, filters):
        t = tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=(3, 3),
            strides=2,
            padding="same"
        )(t)
        t = tf.keras.layers.BatchNormalization()(t)
        t = tf.keras.layers.LeakyReLU(alpha=LEAKY_ALPHA)(t)

        return t

    # Input
    inputs = tf.keras.layers.Input(
        shape=(64, 64, 3)
    )
    x = inputs

    x = block(x, filters=64)  # 64 32x32
    x = block(x, filters=128)  # 128 16x16
    x = tf.keras.layers.Dropout(rate=DROPOUT_RATE)(x)
    x = block(x, filters=256)  # 256 8x8
    x = block(x, filters=512)  # 512 4x4
    x = tf.keras.layers.Dropout(rate=DROPOUT_RATE)(x)
    x = tf.keras.layers.Flatten()(x)

    # Output
    x = tf.keras.layers.Dense(
        units=1
    )(x)
    outputs = x

    # define model
    model = tf.keras.Model(
        inputs=inputs,
        outputs=outputs
    )

    return model


################################################################################
# VGG16 Discriminator
def build_discriminator_vgg16():
    vgg16 = tf.keras.applications.vgg16.VGG16(
        input_shape=(64, 64, 3),
        include_top=False
    )

    vgg16.trainable = False

    model = tf.keras.Sequential()
    model.add(vgg16)
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(
        units=512
    ))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU(alpha=LEAKY_ALPHA))

    model.add(tf.keras.layers.Dense(
        units=1,
        activation=tf.keras.activations.sigmoid
    ))

    return model


################################################################################
# Generator
def build_generator():
    def block(t, filters):
        t = tf.keras.layers.Conv2DTranspose(
            filters=filters,
            kernel_size=(3, 3),
            strides=2,
            padding="same"
        )(t)
        t = tf.keras.layers.BatchNormalization()(t)
        t = tf.keras.layers.LeakyReLU(alpha=LEAKY_ALPHA)(t)

        return t

    # Input
    inputs = tf.keras.layers.Input(
        shape=(NOISE_DIM, )
    )
    x = inputs

    x = tf.keras.layers.Dense(
        units=4*4*512
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha=LEAKY_ALPHA)(x)

    x = tf.keras.layers.Reshape(
        target_shape=(4, 4, 512)
    )(x)

    x = tf.keras.layers.Conv2D(
        filters=512,
        kernel_size=(3, 3),
        strides=1,
        padding="same"
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha=LEAKY_ALPHA)(x)

    x = block(x, filters=256)  # 256 8x8
    x = block(x, filters=128)  # 128 16x16
    x = block(x, filters=64)  # 64 32x32
    x = block(x, filters=64)  # 64 64x64

    # Output
    x = tf.keras.layers.Conv2DTranspose(
        filters=3,  # RGB
        kernel_size=(3, 3),
        strides=1,
        padding="same",
        activation=tf.keras.activations.tanh
    )(x)
    outputs = x

    # define model
    model = tf.keras.Model(
        inputs=inputs,
        outputs=outputs
    )

    return model
