import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras import backend as K



import numpy as np

class VAE(keras.Model):
    def __init__(self, input_dim, latent_dim, encoder_conv_filters, encoder_conv_kernel_size, encoder_conv_strides, decoder_conv_t_filters, decoder_conv_t_kernel_size, decoder_conv_t_strides, **kwargs):
        super().__init__(**kwargs)
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        self.encoder_conv_filters = encoder_conv_filters
        self.encoder_conv_kernel_size = encoder_conv_kernel_size
        self.encoder_conv_strides = encoder_conv_strides 
        self.n_layers_encoder = len(encoder_conv_filters)

        self.decoder_conv_t_filters = decoder_conv_t_filters
        self.decoder_conv_t_kernel_size = decoder_conv_t_kernel_size
        self.decoder_conv_t_strides = decoder_conv_t_strides
        self.n_layers_decoder = len(decoder_conv_t_filters)


        # Metrics to keep track of loss averages over each epoch
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

        self.sampler = Sampler()
        self._build()

    @property
    def metrics(self):
        return [self.total_loss_tracker, self.reconstruction_loss_tracker, self.kl_loss_tracker]

    def _build(self):
        # Encoder
        encoder_inputs = keras.Input(shape = self.input_dim)
        x = encoder_inputs

        for i in range(self.n_layers_encoder):
            conv_layer = layers.Conv2D(
                filters = self.encoder_conv_filters[i],
                kernel_size = self.encoder_conv_kernel_size[i],
                strides = self.encoder_conv_strides[i],
                padding = "same",
                # activation="relu"
            )
            
            # x = layers.BatchNormalization()(x)
            x = conv_layer(x)
            x = layers.LeakyReLU()(x)
            # x = layers.Dropout(rate = 0.25)(x)

        shape_before_flattening = K.int_shape(x)[1:]

        x = layers.Flatten()(x)
        self.mu = layers.Dense(self.latent_dim, name="mu")(x)
        self.log_var = layers.Dense(self.latent_dim, name="log_var")(x)
        self.encoder = keras.Model(encoder_inputs, [ self.mu, self.log_var ], name="encoder")


        # Sampler TODO: odstranit
        def sampling(args):
            mu, log_var = args
            epsilon = K.random_normal(shape = K.shape(mu), mean = 0, stddev = 1)

            return mu * K.exp(log_var / 2) * epsilon
        
        encoder_output = layers.Lambda(sampling)([ self.mu, self.log_var ])
        self.encoder_sampling = keras.Model(encoder_inputs, encoder_output, name="encoder_sampling")

        # Decoder
        decoder_inputs = keras.Input(shape = (self.latent_dim))

        x = layers.Dense(np.prod(shape_before_flattening))(decoder_inputs)
        x = layers.Reshape(shape_before_flattening)(x)

        for i in range(self.n_layers_decoder):
            conv_t_layer = layers.Conv2DTranspose(
                filters = self.decoder_conv_t_filters[i],
                kernel_size = self.decoder_conv_t_kernel_size[i],
                strides = self.decoder_conv_t_strides[i],
                padding = "same"
            )

            x = conv_t_layer(x)
            x = layers.Activation("sigmoid")(x)

        decoder_output = x
        self.decoder = keras.Model(decoder_inputs, decoder_output)


        # VAE
        model_input = encoder_inputs
        model_output = self.decoder(encoder_output)

        self.model = keras.Model(model_input, model_output)

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var = self.encoder(data) # TODO: take it from encoder
            # z_mean = self.mu
            # z_log_var = self.log_var

            z = self.sampler(z_mean, z_log_var)
            reconstruction = self.decoder(z)

            reconstruction_loss = tf.reduce_mean(tf.reduce_sum(
                keras.losses.binary_crossentropy(data, reconstruction), axis=(1, 2)))
            kl_loss = -0.5 * (1 + z_log_var -
                              tf.square(z_mean) - tf.exp(z_log_var))  # Kullback-Leibler divergence (regularization term)
            total_loss = reconstruction_loss + kl_loss

            grads = tape.gradient(total_loss, self.trainable_weights)
            self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

            self.total_loss_tracker.update_state(total_loss)
            self.reconstruction_loss_tracker.update_state(reconstruction_loss)
            self.kl_loss_tracker.update_state(kl_loss)

            return {
                "total_loss": self.total_loss_tracker.result(),
                "reconstruction_loss": self.reconstruction_loss_tracker.result(),
                "kl_loss": self.kl_loss_tracker.result(),
            }
        
    # def compile(self, learning_rate):
    #     optimizer = keras.optimizers.Adam(learning_rate = learning_rate)

    #     self.compile(optimizer = optimizer, run_eagerly = True)

    def save(self, weights_file_name):
        self.model.save(weights_file_name)

    def load_weights(self, weights_file_path):
        self.model.load_weights(weights_file_path)

class Sampler(layers.Layer):
    def call(self, mu, log_var):
        batch_size = tf.shape(mu)[0]
        z_size = tf.shape(mu)[1]

        epsilon = tf.random.normal(shape = (batch_size, z_size))

        return mu + tf.exp(0.5 * log_var) * epsilon

