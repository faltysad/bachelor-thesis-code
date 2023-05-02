from model.VAE import VAE
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

LEARNING_RATE = 0.0005
R_LOSS_FACTOR = 1000
BATCH_SIZE = 32
EPOCHS = 200
PRINT_EVERY_N_BATCHES = 100
INITIAL_EPOCH = 0

# mode = "load"
mode = "save"

vae = VAE(
    input_dim = (28, 28, 1),
    latent_dim = 2,
    encoder_conv_filters = [32, 64, 64, 64],
    encoder_conv_kernel_size = [3, 3, 3, 3],
    encoder_conv_strides = [1, 2, 2, 1],
    decoder_conv_t_filters = [64, 64, 32, 1],
    decoder_conv_t_kernel_size = [3, 3, 3, 3],
    decoder_conv_t_strides = [1, 2, 2, 1]
)

if mode == "load":
    vae.load_weights("./weights/mnist")

# TODO: move to loaders util
(x_train, _), (x_test, _) = keras.datasets.mnist.load_data()
mnist_digits = np.concatenate([x_train, x_test], axis=0)
mnist_digits = np.expand_dims(mnist_digits, -1).astype("float32") / 255

print(vae.model.summary())

vae.compile(optimizer=keras.optimizers.Adam(), run_eagerly=True)
vae.fit(mnist_digits, epochs=1, batch_size=128)

if mode == "save":
    vae.save("./save/mnist")