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
import tensorflow as tf

from parameters import *


################################################################################
# Discriminator
def build_discriminator():
    m = tf.keras.Sequential()

    # Layer 1: Conv: 32x32x64
    m.add(tf.keras.layers.Conv2D(
        filters=64,
        kernel_size=(3, 3),
        strides=2,
        input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, NUM_CHANNELS),
        padding="same"
    ))
    m.add(tf.keras.layers.BatchNormalization())
    m.add(tf.keras.layers.LeakyReLU(alpha=LEAKY_ALPHA))

    # Layer 2: Conv 16x16x128
    m.add(tf.keras.layers.Conv2D(
        filters=128,
        kernel_size=(3, 3),
        strides=2,
        padding="same"
    ))
    m.add(tf.keras.layers.BatchNormalization())
    m.add(tf.keras.layers.LeakyReLU(alpha=LEAKY_ALPHA))

    # Layer 3: Dropout
    m.add(tf.keras.layers.Dropout(rate=DROPOUT_RATE))

    # Layer 4: Conv: 8x8x256
    m.add(tf.keras.layers.Conv2D(
        filters=256,
        kernel_size=(3, 3),
        strides=2,
        padding="same"
    ))
    m.add(tf.keras.layers.BatchNormalization())
    m.add(tf.keras.layers.LeakyReLU(alpha=LEAKY_ALPHA))

    # Layer 5: Conv: 4x4x512
    m.add(tf.keras.layers.Conv2D(
        filters=512,
        kernel_size=(3, 3),
        strides=2,
        padding="same"
    ))
    m.add(tf.keras.layers.BatchNormalization())
    m.add(tf.keras.layers.LeakyReLU(alpha=LEAKY_ALPHA))

    # Layer 6: Dropout
    m.add(tf.keras.layers.Dropout(rate=DROPOUT_RATE))

    # Layer 7: Flatten
    m.add(tf.keras.layers.Flatten())

    # Layer 8: Output
    m.add(tf.keras.layers.Dense(
        units=1,
    ))

    return m


def build_generator():
    m = tf.keras.Sequential()

    # Layer 1: Fully connected
    m.add(tf.keras.layers.Dense(
        units=4*4*512,
        input_shape=(NOISE_DIM, ),
        use_bias=False
    ))
    m.add(tf.keras.layers.BatchNormalization())
    m.add(tf.keras.layers.LeakyReLU(alpha=LEAKY_ALPHA))

    # Layer 2: Reshape
    m.add(tf.keras.layers.Reshape(
        target_shape=(4, 4, 512)
    ))

    # Layer 3: Conv: 8x8x256
    m.add(tf.keras.layers.Conv2DTranspose(
        filters=256,
        kernel_size=(3, 3),
        strides=2,
        padding="same"
    ))
    m.add(tf.keras.layers.BatchNormalization())
    m.add(tf.keras.layers.LeakyReLU(alpha=LEAKY_ALPHA))

    # Layer 4: Conv: 16x16x128
    m.add(tf.keras.layers.Conv2DTranspose(
        filters=128,
        kernel_size=(3, 3),
        strides=2,
        padding="same"
    ))
    m.add(tf.keras.layers.BatchNormalization())
    m.add(tf.keras.layers.LeakyReLU(alpha=LEAKY_ALPHA))

    # Layer 5: Conv: 32x32x64
    m.add(tf.keras.layers.Conv2DTranspose(
        filters=64,
        kernel_size=(3, 3),
        strides=2,
        padding="same"
    ))
    m.add(tf.keras.layers.BatchNormalization())
    m.add(tf.keras.layers.LeakyReLU(alpha=LEAKY_ALPHA))

    # Layer 6: Conv: 64x64x3
    m.add(tf.keras.layers.Conv2DTranspose(
        filters=3,  # RGB
        kernel_size=(3, 3),
        strides=2,
        padding="same",
        activation=tf.keras.activations.tanh
    ))

    return m
