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
def build_discriminator(input_shape):
    m = tf.keras.Sequential()

    # Layer 1
    m.add(tf.keras.layers.Conv2D(
        filters=32,
        kernel_size=(4, 4),
        strides=2,
        input_shape=input_shape,
        padding="same",
        activation=tf.keras.activations.relu
    ))

    m.add(tf.keras.layers.BatchNormalization())

    # Layer 2
    m.add(tf.keras.layers.Conv2D(
        filters=64,
        kernel_size=(4, 4),
        strides=2,
        padding="same",
        activation=tf.keras.activations.relu
    ))

    m.add(tf.keras.layers.BatchNormalization())

    # Layer 3
    m.add(tf.keras.layers.Conv2D(
        filters=128,
        kernel_size=(4, 4),
        strides=2,
        padding="same",
        activation=tf.keras.activations.relu
    ))

    m.add(tf.keras.layers.BatchNormalization())

    m.add(tf.keras.layers.Flatten())

    m.add(tf.keras.layers.Dense(
        units=1,
    ))

    # model inputs and outputs
    image = tf.keras.layers.Input(shape=input_shape)
    output = m(image)
    return tf.keras.Model(inputs=image, outputs=output)


# Generator
def build_generator(noise_dim):
    m = tf.keras.Sequential()

    # Layer 1
    m.add(tf.keras.layers.Dense(
        units=64*64*64,
        input_dim=noise_dim,
        activation=tf.keras.activations.relu
    ))

    m.add(tf.keras.layers.BatchNormalization())

    # Layer 2
    m.add(tf.keras.layers.Reshape(
        target_shape=(64, 64, 64)
    ))

    # Layer 3
    m.add(tf.keras.layers.Conv2DTranspose(
        filters=512,
        kernel_size=(4, 4),
        strides=2,
        padding="same",
        activation=tf.keras.activations.relu
    ))

    m.add(tf.keras.layers.BatchNormalization())

    # Layer 4
    m.add(tf.keras.layers.Conv2DTranspose(
        filters=128,
        kernel_size=(4, 4),
        strides=2,
        padding="same",
        activation=tf.keras.activations.relu
    ))

    m.add(tf.keras.layers.BatchNormalization())

    # Layer 5
    m.add(tf.keras.layers.Conv2DTranspose(
        filters=3,
        kernel_size=(4, 4),
        strides=2,
        padding="same",
        activation=tf.keras.activations.tanh
    ))

    # model inputs and outputs
    z = tf.keras.layers.Input(shape=(noise_dim, ))
    image = m(z)

    return tf.keras.Model(inputs=z, outputs=image)
