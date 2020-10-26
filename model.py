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
        shape=(256, 256, 3)  # high resolution
    ))

    # Conv 64 256x256
    model.add(tf.keras.layers.Conv2D(
        filters=64,
        kernel_size=(3, 3),
        strides=1,
        padding="same"
    ))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU(alpha=LEAKY_ALPHA))

    # Conv 64 128x128
    model.add(tf.keras.layers.Conv2D(
        filters=64,
        kernel_size=(3, 3),
        strides=2,
        padding="same"
    ))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU(alpha=LEAKY_ALPHA))

    # Conv 128 128x128
    model.add(tf.keras.layers.Conv2D(
        filters=128,
        kernel_size=(3, 3),
        strides=1,
        padding="same"
    ))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU(alpha=LEAKY_ALPHA))

    # Conv 128 64x64
    model.add(tf.keras.layers.Conv2D(
        filters=128,
        kernel_size=(3, 3),
        strides=2,
        padding="same"
    ))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU(alpha=LEAKY_ALPHA))

    # Conv 256 64x64
    model.add(tf.keras.layers.Conv2D(
        filters=256,
        kernel_size=(3, 3),
        strides=1,
        padding="same"
    ))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU(alpha=LEAKY_ALPHA))

    # Conv 256 32x32
    model.add(tf.keras.layers.Conv2D(
        filters=256,
        kernel_size=(3, 3),
        strides=2,
        padding="same"
    ))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU(alpha=LEAKY_ALPHA))

    # Conv 512 32x32
    model.add(tf.keras.layers.Conv2D(
        filters=512,
        kernel_size=(3, 3),
        strides=1,
        padding="same"
    ))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU(alpha=LEAKY_ALPHA))

    # Conv 512 16x16
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

    # Fully connected
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
# will use skip connections
def build_generator():
    # residual block
    def build_residual_block(t):
        residual = t  # skip connection
        t = tf.keras.layers.Conv2D(
            filters=64,
            kernel_size=(3, 3),
            strides=1,
            padding="same"
        )(t)
        t = tf.keras.layers.BatchNormalization()(t)
        t = tf.keras.layers.ReLU()(t)
        t = tf.keras.layers.Conv2D(
            filters=64,
            kernel_size=(3, 3),
            strides=1,
            padding="same"
        )(t)
        t = tf.keras.layers.BatchNormalization()(t)
        t = t + residual

        return t

    # will try Functional API of Model first, otherwise will sub-class Model

    # Input
    inputs = tf.keras.layers.Input(
        shape=(64, 64, 3)  # low resolution
    )

    # Conv

    # skip connection

    # residual blocks

    # Conv
    # add skip

    # Conv / Upsample

    # Conv / Upsample

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
