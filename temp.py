"""

"""
################################################################################
# Imports
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import tensorflow as tf
import tensorflow_hub as hub


################################################################################
# Main
if __name__ == "__main__":
    latent_dim = 512

    gan = hub.load("https://tfhub.dev/google/progan-128/1").signatures["default"]
    output = gan(tf.random.normal(shape=[1, latent_dim]))
    output = output["default"]
    output = tf.squeeze(output, axis=0)

    output = tf.keras.preprocessing.image.array_to_img(output)

    output.show()
