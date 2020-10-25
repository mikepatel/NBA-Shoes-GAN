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
        shape=(, 3)
    ))

    # Output
    model.add(tf.keras.layers.Dense(
        activation=tf.keras.activations.sigmoid
    ))

    return model


################################################################################
# Generator
# will use skip connections
def build_generator():
    # residual block
    def build_residual_block():


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
