"""
Michael Patel
April 2020

Project description:
    Build a GAN to create basketball shoe designs

File description:
    For model and training parameters
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
# GPU allocation
gpus = tf.config.experimental.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(gpus[0], True)


################################################################################
# training parameters
NUM_EPOCHS = 10000
BATCH_SIZE = 1  # 35
LEARNING_RATE = 0.0001
BETA_1 = 0.9  # 0.5
GEN_EPOCH = 100

# model parameters
NOISE_DIM = 100
LEAKY_ALPHA = 0.3  # default is 0.3
NUM_G_RESIDUAL_BLOCKS = 10  # 16

# image dimensions
#IMAGE_WIDTH = 128
#IMAGE_HEIGHT = 128
#IMAGE_CHANNELS = 3

# directories
DATA_DIR = os.path.join(os.getcwd(), "data")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
RESULTS_DIR = os.path.join(os.getcwd(), "results")
TEMP_DIR = os.path.join(os.getcwd(), "temp")
