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
    model = tf.keras.Sequential()

    # Input
    model.add(tf.keras.layers.Input(
        shape=(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS)
    ))

    # Conv 64 128x128
    model.add(tf.keras.layers.Conv2D(
        filters=64,
        kernel_size=(3, 3),
        strides=1,
        padding="same"
    ))

    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU(alpha=LEAKY_ALPHA))

    # Conv 64 64x64
    model.add(tf.keras.layers.Conv2D(
        filters=64,
        kernel_size=(3, 3),
        strides=2,
        padding="same"
    ))

    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU(alpha=LEAKY_ALPHA))

    # Conv 64 64x64
    model.add(tf.keras.layers.Conv2D(
        filters=64,
        kernel_size=(3, 3),
        strides=1,
        padding="same"
    ))

    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU(alpha=LEAKY_ALPHA))

    # Conv 64 32x32
    model.add(tf.keras.layers.Conv2D(
        filters=64,
        kernel_size=(3, 3),
        strides=2,
        padding="same"
    ))

    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU(alpha=LEAKY_ALPHA))

    # Conv 128 32x32
    model.add(tf.keras.layers.Conv2D(
        filters=128,
        kernel_size=(3, 3),
        strides=1,
        padding="same"
    ))

    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU(alpha=LEAKY_ALPHA))

    # Conv 128 16x16
    model.add(tf.keras.layers.Conv2D(
        filters=128,
        kernel_size=(3, 3),
        strides=2,
        padding="same"
    ))

    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU(alpha=LEAKY_ALPHA))

    # Conv 256 8x8
    model.add(tf.keras.layers.Conv2D(
        filters=256,
        kernel_size=(3, 3),
        strides=2,
        padding="same"
    ))

    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU(alpha=LEAKY_ALPHA))

    # Conv 512 4x4
    model.add(tf.keras.layers.Conv2D(
        filters=512,
        kernel_size=(3, 3),
        strides=2,
        padding="same"
    ))

    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU(alpha=LEAKY_ALPHA))

    # Conv 512 2x2
    model.add(tf.keras.layers.Conv2D(
        filters=512,
        kernel_size=(3, 3),
        strides=2,
        padding="same"
    ))

    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU(alpha=LEAKY_ALPHA))

    # Flatten
    model.add(tf.keras.layers.Flatten())

    # Output
    model.add(tf.keras.layers.Dense(
        units=2
    ))

    return model


################################################################################
# Generator
# will use skip connections
def build_generator():
    # will try Functional API of Model first, otherwise will sub-class Model

    # Input
    inputs = tf.keras.layers.Input(
        shape=(64, 64, 3)
    )

    x = inputs

    # Output
    outputs = tf.keras.layers.Conv2D(
        filters=3,  # RGB
        kernel_size=(3, 3),
        strides=1,
        padding="same",
        activation=tf.keras.activations.tanh
    )(x)

    # create model
    model = tf.keras.Model(
        inputs=inputs,
        outputs=outputs
    )

    return model
