"""
Michael Patel
April 2020

Project description:
    Build a GAN to create basketball shoe designs

File description:
    For model and training parameters
"""
################################################################################
NUM_EPOCHS = 200
BATCH_SIZE = 20

NOISE_DIM = 100
DROP_RATE = 0.2

LEARNING_RATE = 0.0001
BETA_1 = 0.9  # 0.5

LEAKY_ALPHA = 0.3  # default is 0.3
DROPOUT_RATE = 0.3

IMAGE_WIDTH = 64
IMAGE_HEIGHT = 64
NUM_CHANNELS = 3
