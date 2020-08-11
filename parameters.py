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


################################################################################
# training parameters
NUM_EPOCHS = 1
BATCH_SIZE = 20  # 35
LEARNING_RATE = 0.0002
BETA_1 = 0.9  # 0.5

# model parameters
NOISE_DIM = 100
LEAKY_ALPHA = 0.3  # default is 0.3
DROPOUT_RATE = 0.3

# image dimensions
IMAGE_WIDTH = 64
IMAGE_HEIGHT = 64
NUM_CHANNELS = 3

# directories
DATA_DIR = os.path.join(os.getcwd(), "data")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
RESULTS_DIR = os.path.join(os.getcwd(), "results")
TEMP_DIR = os.path.join(os.getcwd(), "temp")
