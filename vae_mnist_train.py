from model.VAE import VAE
from loaders.load_mnist import load_mnist

from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf

from datetime import datetime
from packaging import version

print("TensorFlow version: ", tf.__version__)
assert version.parse(tf.__version__).release[0] >= 2, \
    "This notebook requires TensorFlow 2.0 or above."

import tensorboard
tensorboard.__version__

# Define the Keras TensorBoard callback.
logdir="logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)


BATCH_SIZE = 128
EPOCHS = 500

mode = "save"

vae = VAE(
    input_dim = (28, 28, 1),
    z_dim = 2,
    encoder_conv_filters = [32, 64, 64, 64],
    encoder_conv_kernel_size = [3, 3, 3, 3],
    encoder_conv_strides = [1, 2, 2, 1],
    decoder_conv_t_filters = [64, 64, 32, 1],
    decoder_conv_t_kernel_size = [3, 3, 3, 3],
    decoder_conv_t_strides = [1, 2, 2, 1]
)


(x_train, y_train), (x_test, y_test) = load_mnist()

vae.compile(optimizer=keras.optimizers.Adam())
vae.fit(x_train, epochs= EPOCHS, batch_size= BATCH_SIZE, callbacks=[tensorboard_callback])

if mode == "save":
    vae.save("./save")