"""


"""
################################################################################
# Imports
import os
import numpy as np
from datetime import datetime
from PIL import Image
import matplotlib.pyplot as plt

import tensorflow as tf


################################################################################
# model parameters
NOISE_DIM = 100
LEAKY_ALPHA = 0.1
DROPOUT_RATE = 0.5

# training parameters
NUM_EPOCHS = 1
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
BETA_1 = 0.9

NUM_GEN_IMAGES = 16

# directories
DATA_DIR = os.path.join(os.getcwd(), "data")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
