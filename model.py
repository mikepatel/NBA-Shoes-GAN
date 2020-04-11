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
"""
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
"""


# Discriminator
class Discriminator(tf.keras.Model):
    def __init__(self):
        super(Discriminator, self).__init__()

        # Conv 1
        self.conv1 = tf.keras.layers.Conv2D(
            filters=32,
            kernel_size=(5, 5),
            strides=2,
            padding="same"
        )

        # Batchnorm 1
        self.batchnorm1 = tf.keras.layers.BatchNormalization()

        # LeakyReLU 1
        self.leaky1 = tf.keras.layers.LeakyReLU()

        # Conv 2
        self.conv2 = tf.keras.layers.Conv2D(
            filters=64,
            kernel_size=(5, 5),
            strides=2,
            padding="Same"
        )

        # Batchnorm 2
        self.batchnorm2 = tf.keras.layers.BatchNormalization()

        # LeakyReLU 2
        self.leaky2 = tf.keras.layers.LeakyReLU()

        # Conv 3
        self.conv3 = tf.keras.layers.Conv2D(
            filters=128,
            kernel_size=(5, 5),
            strides=2,
            padding="same"
        )

        # Batchnorm 3
        self.batchnorm3 = tf.keras.layers.BatchNormalization()

        # LeakyReLU 3
        self.leaky3 = tf.keras.layers.LeakyReLU()

        # Flatten
        self.flatten = tf.keras.layers.Flatten()

        # Fully connected
        self.fc = tf.keras.layers.Dense(
            units=1
        )

    # forward call
    def call(self, x, training=True):
        # Layer 1: Conv 1
        x = self.conv1(x)
        x = self.batchnorm1(x, training=training)
        x = self.leaky1(x)

        # Layer 2: Conv 2
        x = self.conv2(x)
        x = self.batchnorm2(x, training=training)
        x = self.leaky2(x)

        # Layer 3: Conv 3
        x = self.conv3(x)
        x = self.batchnorm3(x, training=training)
        x = self.leaky3(x)

        # Layer 4: Flatten
        x = self.flatten(x)

        # Layer 5: Output
        x = self.fc(x)

        return x


# Generator
class Generator(tf.keras.Model):
    def __init__(self):
        super(Generator, self).__init__()

        # Fully connected
        self.fc = tf.keras.layers.Dense(
            units=64*64*3
        )

        # Batchnorm 1
        self.batchnorm1 = tf.keras.layers.BatchNormalization()

        # LeakyReLU 1
        self.leaky1 = tf.keras.layers.LeakyReLU()

        # Reshape
        self.reshape = tf.keras.layers.Reshape(
            target_shape=(64, 64, 3)
        )

        # Conv 1
        self.conv1 = tf.keras.layers.Conv2DTranspose(
            filters=512,
            kernel_size=(5, 5),
            strides=2,
            padding="same"
        )

        # Batchnorm 2
        self.batchnorm2 = tf.keras.layers.BatchNormalization()

        # LeakyReLU 2
        self.leaky2 = tf.keras.layers.LeakyReLU()

        # Conv 2
        self.conv2 = tf.keras.layers.Conv2DTranspose(
            filters=128,
            kernel_size=(5, 5),
            strides=2,
            padding="same"
        )

        # Batchnorm 3
        self.batchnorm3 = tf.keras.layers.BatchNormalization()

        # LeakyReLU 3
        self.leaky3 = tf.keras.layers.LeakyReLU()

        # Conv 3
        self.conv3 = tf.keras.layers.Conv2DTranspose(
            filters=3,
            kernel_size=(5, 5),
            strides=2,
            padding="same"
        )

        # Batchnorm 4
        self.batchnorm4 = tf.keras.layers.BatchNormalization()

    # forward call
    def call(self, x, training=True):
        # Layer 1: Fully connected
        x = self.fc(x)
        x = self.batchnorm1(x, training=training)
        x = self.leaky1(x)

        # Layer 2: Reshape
        x = self.reshape(x)

        # Layer 3: Convolution 1
        x = self.conv1(x)
        x = self.batchnorm2(x, training=training)
        x = self.leaky2(x)

        # Layer 4: Convolution 2
        x = self.conv2(x)
        x = self.batchnorm3(x, training=training)
        x = self.leaky3(x)

        # Layer 5: Convolution 3
        x = self.conv3(x)
        x = self.batchnorm4(x, training=training)
        x = tf.nn.tanh(x)

        return x


"""
def build_generator(noise_dim):
    m = tf.keras.Sequential()

    # Layer 1
    m.add(tf.keras.layers.Dense(
        units=64*64*3,
        input_dim=noise_dim,
        activation=tf.keras.activations.relu
    ))

    #m.add(tf.keras.layers.BatchNormalization())

    # Layer 2
    m.add(tf.keras.layers.Reshape(
        target_shape=(64, 64, 3)
    ))

    # Layer 3
    m.add(tf.keras.layers.Conv2DTranspose(
        filters=512,
        kernel_size=(4, 4),
        strides=2,
        padding="same",
        activation=tf.keras.activations.relu
    ))

    #m.add(tf.keras.layers.BatchNormalization())

    # Layer 4
    m.add(tf.keras.layers.Conv2DTranspose(
        filters=128,
        kernel_size=(4, 4),
        strides=2,
        padding="same",
        activation=tf.keras.activations.relu
    ))

    #m.add(tf.keras.layers.BatchNormalization())

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
"""
