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


################################################################################
# get images
def get_images(dataset="train"):
    image_files_pattern = os.path.join(os.getcwd(), "data\\") + str(dataset) + "\\*.png"
    filenames = glob.glob(image_files_pattern)

    images = []

    for f in filenames:
        image = Image.open(f)

        # resize image: 64x64
        resized_image = image.resize((64, 64))

        # images as arrays
        images.append(np.array(resized_image))

    return images


# get data generator
def get_data_gen(dataset):
    # augment dataset using tf.keras.preprocessing.image.ImageDataGenerator
    image_generator = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=30,  # degrees
        horizontal_flip=True,
        rescale=1. / 255,
    )

    data_gen = image_generator.flow_from_directory(
        directory=os.path.join(os.getcwd(), "data\\") + str(dataset),
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

    train_images = []
    val_images = []
    test_images = []

    train_labels = []
    val_labels = []
    test_labels = []

    # get images
    #train_images = get_images(dataset="train")
    #test_images = get_images(dataset="test")

    # normalize images
    #train_images = np.array(train_images).astype(np.float32) / 255.0
    #test_images = np.array(test_images).astype(np.float32) / 255.0

    # create validation set
    #midpoint = int(len(test_images) / 2)
    #val_images = test_images[:midpoint]
    #test_images = test_images[midpoint:]

    train_data_gen = get_data_gen(dataset="train")
    val_data_gen = get_data_gen(dataset="validation")
    test_data_gen = get_data_gen(dataset="test")

    #x = next(train_data_gen)
    #plotImages(x[:5])

    # ----- MODEL ----- #
    discriminator = build_discriminator(input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, NUM_CHANNELS))
    discriminator.compile(
        loss=tf.keras.losses.binary_crossentropy,
        optimizer=tf.keras.optimizers.Adam(),
        metrics=["accuracy"]
    )

    generator = build_generator()

    # ----- TRAINING ----- #

    # ----- GENERATION ----- #
