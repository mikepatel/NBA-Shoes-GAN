"""
Michael Patel
April 2020

Project description:
    Build a GAN to create basketball shoe designs

File description:
    For model preprocessing and training
"""
################################################################################
# Imports
import os
import numpy as np
from datetime import datetime
import glob
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
import matplotlib.pyplot as plt

import tensorflow as tf

from parameters import *
from model import build_discriminator, build_generator
#from model import Discriminator, Generator


################################################################################
# get data generator
def get_data_gen():
    # augment dataset using tf.keras.preprocessing.image.ImageDataGenerator
    image_generator = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=30,  # degrees
        horizontal_flip=True,
        rescale=1. / 255,
    )

    data_gen = image_generator.flow_from_directory(
        directory=os.path.join(os.getcwd(), "data"),
        batch_size=BATCH_SIZE,
        shuffle=True,
        target_size=(IMAGE_WIDTH, IMAGE_HEIGHT),
        class_mode=None
    )

    return data_gen


# plot images in a 1x5 grid
def plotImages(images_arr):
    fig, axes = plt.subplots(1, 5, figsize=(20, 20))
    axes = axes.flatten()
    for img, ax in zip( images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()


# discriminator loss function
def discriminator_loss(real_output, generated_output):
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    real_loss = cross_entropy(
        tf.ones_like(real_output),
        real_output
    )

    fake_loss = cross_entropy(
        tf.zeros_like(generated_output),
        generated_output
    )

    total_loss = real_loss + fake_loss
    return total_loss


# generator loss function
def generator_loss(generated_output):
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    generated_loss = cross_entropy(
        tf.ones_like(generated_output),
        generated_output
    )

    return generated_loss


# generate and save images
def generate_images(model, epoch, save_dir):
    predictions = model(GEN_INPUT, training=False)
    predictions = predictions[:16]  # generate 16 images

    # rescale from [-1, 1] to [0, 1]
    predictions = (predictions + 1) / 2

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i+1)
        plt.imshow(predictions[i])
        plt.axis("off")

    fig_name = os.path.join(save_dir, f'Epoch {epoch:06d}')
    plt.savefig(fig_name)
    plt.close()


# training loop
def train(train_data_gen, discriminator, generator, d_optimizer, g_optimizer, save_dir):
    print()


################################################################################
# Main
if __name__ == "__main__":
    # print TF version
    print(f'TF version: {tf.__version__}')

    # create output directory for results
    output_dir = "results\\" + datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # ----- ETL ----- #
    # ETL = Extraction, Transformation, Load
    train_data_gen = get_data_gen()

    """
    x = next(train_data_gen)
    print(len(x))
    plotImages(x[:5])
    """

    # ----- MODEL ----- #
    # discriminator
    discriminator = build_discriminator()
    discriminator_optimizer = tf.keras.optimizers.Adam(
        learning_rate=LEARNING_RATE,
        beta_1=BETA_1
    )
    discriminator.summary()

    # generator
    generator = build_generator()
    generator_optimizer = tf.keras.optimizers.Adam(
        learning_rate=LEARNING_RATE,
        beta_1=BETA_1
    )
    generator.summary()

    # ----- TRAINING ----- #
    z_input_gen = tf.random.normal(shape=(BATCH_SIZE, NOISE_DIM))
