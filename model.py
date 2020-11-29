"""
Michael Patel
April 2020

Project description:
    Build a GAN to create basketball shoe designs

File description:
    For model definitions
"""
################################################################################
# Imports
from parameters import *


################################################################################
# Discriminator
def build_discriminator():
    # Input
    inputs = tf.keras.layers.Input(
        shape=(32, 32, 3)
    )

    x = inputs

    x = tf.keras.layers.Conv2D(
        filters=64,
        kernel_size=(3, 3),
        strides=2,
        padding="same"
    )(x)
    x = tf.keras.layers.LeakyReLU()(x)

    x = tf.keras.layers.Conv2D(
        filters=128,
        kernel_size=(3, 3),
        strides=2,
        padding="same"
    )(x)
    x = tf.keras.layers.LeakyReLU()(x)

    x = tf.keras.layers.Flatten()(x)

    # Output
    x = tf.keras.layers.Dense(
        units=1,
        activation=tf.keras.activations.sigmoid
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
        input_shape=(32, 32, 3),
        include_top=False
    )

    vgg16.trainable = False

    model = tf.keras.Sequential()
    model.add(vgg16)
    model.add(tf.keras.layers.Flatten())

    model.add(tf.keras.layers.Dense(
        units=1024
    ))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU(alpha=LEAKY_ALPHA))

    # Output
    model.add(tf.keras.layers.Dense(
        units=1,
        activation=tf.keras.activations.sigmoid
    ))

    return model


################################################################################
# Generator
def build_generator():
    # Input
    inputs = tf.keras.layers.Input(
        shape=(NOISE_DIM, )
    )
    x = inputs

    x = tf.keras.layers.Dense(
        units=8*8*256
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)

    x = tf.keras.layers.Reshape(
        target_shape=(8, 8, 256)
    )(x)

    x = tf.keras.layers.Conv2DTranspose(
        filters=128,
        kernel_size=(3, 3),
        strides=1,
        padding="same"
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)

    x = tf.keras.layers.Conv2DTranspose(
        filters=128,
        kernel_size=(3, 3),
        strides=2,
        padding="same"
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)

    x = tf.keras.layers.Conv2DTranspose(
        filters=128,
        kernel_size=(3, 3),
        strides=2,
        padding="same"
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)

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
