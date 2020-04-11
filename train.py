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
#from model import build_discriminator, build_generator
from model import Discriminator, Generator


################################################################################
GEN_INPUT = tf.random.normal(shape=[16, 100])

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

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i+1)
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5)
        plt.axis("off")

    fig_name = os.path.join(save_dir, f'Epoch {epoch:06d}')
    plt.savefig(fig_name)
    plt.close()


# training loop
def train(train_data_gen, discriminator, generator, d_optimizer, g_optimizer, save_dir):
    # adversarial ground truths
    valid = np.ones((BATCH_SIZE, 1))
    fake = np.zeros((BATCH_SIZE, 1))

    for e in range(NUM_EPOCHS+1):
        real_images = next(train_data_gen)

        # generate noise
        noise = tf.random.normal(shape=[BATCH_SIZE, 100])

        # GradientTape
        with tf.GradientTape() as g_tape, tf.GradientTape() as d_tape:
            # generator
            generated_image = generator(noise)

            # discriminator
            real_output = discriminator(real_images)
            fake_output = discriminator(generated_image)

            # loss functions
            g_loss = generator_loss(fake_output)
            d_loss = discriminator_loss(real_output, fake_output)

        # compute gradients recorded on "tape"
        g_gradients = g_tape.gradient(g_loss, generator.variables)
        d_gradients = d_tape.gradient(d_loss, discriminator.variables)

        # apply gradients to model variables to minimize loss function
        g_optimizer.apply_gradients(zip(g_gradients, generator.variables))
        d_optimizer.apply_gradients(zip(d_gradients, discriminator.variables))

        generate_images(generator, e, save_dir)

        """
        # input for discriminator
        real_images = next(train_data_gen)

        # input for generator
        noise_input = np.random.normal(0, 1, size=[BATCH_SIZE, 100])
        fake_images = generator.predict(noise_input)

        # train discriminator
        discriminator.trainable = True
        d_loss_real = discriminator.train_on_batch(real_images, valid)
        d_loss_fake = discriminator.train_on_batch(fake_images, fake)
        d_loss = np.add(d_loss_real, d_loss_fake) * 0.5

        # train generator
        noise_input = np.random.normal(0, 1, size=[BATCH_SIZE, 100])
        discriminator.trainable = False
        g_loss = gan.train_on_batch(noise_input, valid)

        print(f'Epoch: {e}')
        print(f'D real: {d_loss_real}')
        print(f'D fake: {d_loss_fake}')
        print(f'D: {d_loss}')
        print(f'G: {g_loss}')
        print()

        generate_images(generator, e, save_dir)
        """


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
    #print(len(x))
    #plotImages(x[:5])

    # ----- MODEL ----- #
    #discriminator = build_discriminator(input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, NUM_CHANNELS))
    """
    discriminator.compile(
        loss=tf.keras.losses.binary_crossentropy,
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.5),
        metrics=["accuracy"]
    )
    """

    #generator = build_generator(100)
    """
    z = tf.keras.layers.Input(shape=(100, ))
    image = generator(z)

    discriminator.trainable = False
    prediction = discriminator(image)

    gan = tf.keras.Model(inputs=z, outputs=prediction)
    gan.compile(
        loss=tf.keras.losses.binary_crossentropy,
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.5),
        metrics=["accuracy"]
    )
    """
    d = Discriminator()
    g = Generator()
    d_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.5)
    g_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.5)

    # ----- TRAINING ----- #
    train(train_data_gen, d, g, d_optimizer, g_optimizer, output_dir)

    # ----- GENERATION ----- #
