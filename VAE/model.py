import numpy as np

import tensorflow as tf
from tensorflow import keras as K

class VAE(K.Model):

    def __init__(self, image_size, channels, latent_dim, **kwargs):
        super().__init__(**kwargs)

        self.IMAGE_SIZE = image_size
        self.channels = channels
        self.latent_dim = latent_dim

        class Sampling(K.layers.Layer):
            """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.seed_generator = K.random.SeedGenerator(1337)

            def call(self, inputs):
                z_mean, z_log_var = inputs
                batch = K.backend.shape(z_mean)[0]
                dim = K.backend.shape(z_mean)[1]
                epsilon = K.random.normal(shape=(batch, dim), seed=self.seed_generator)
                return z_mean + K.backend.exp(0.5 * z_log_var) * epsilon

        # Encoder
        encoder_inputs = K.layers.Input(shape=(self.IMAGE_SIZE, self.IMAGE_SIZE, self.channels))
        x = K.layers.Conv2D(32, 3, activation='relu', strides=2, padding='same')(encoder_inputs)
        x = K.layers.Conv2D(64, 3, activation='relu', strides=2, padding='same')(x)
        x = K.layers.Flatten()(x)
        x = K.layers.Dense(16, activation='relu')(x)
        x = K.layers.Dense(512, activation='relu')(x)
        z_mean = K.layers.Dense(latent_dim, name="z_mean")(x)
        z_log_var = K.layers.Dense(latent_dim, name="z_log_var")(x)
        z = Sampling()([z_mean, z_log_var])
        self.encoder = K.Model(encoder_inputs, [z_mean, z_log_var, z], name='encoder')

        # Decoder
        latent_inputs = K.layers.Input(shape=(self.latent_dim,))
        x = K.layers.Dense(7 * 7 * 64, activation="relu")(latent_inputs)
        x = K.layers.Reshape((7, 7, 64))(x)
        x = K.layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(x)
        x = K.layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)
        decoder_outputs = K.layers.Conv2DTranspose(self.channels, 3, activation="sigmoid", padding="same")(x)
        self.decoder = K.Model(latent_inputs, decoder_outputs, name="decoder")

        self.total_loss_tracker = K.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = K.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = K.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            # Clip para evitar explosión numérica
            z_log_var = tf.clip_by_value(z_log_var, -10, 10)
            reconstruction = self.decoder(z)
            reconstruction_loss = K.ops.mean(
                K.ops.sum(
                    K.losses.binary_crossentropy(data, reconstruction),
                    axis=(1, 2),
                )
            )
            kl_loss = -0.5 * (1 + z_log_var - K.ops.square(z_mean) - K.ops.exp(z_log_var))
            kl_loss = K.ops.mean(K.ops.sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }
    
    def get_callbacks(self, log_dir="logs", patience=5, recon_data=None, recon_every_n_epochs=5, recon_max_images=8):
        early_stop = K.callbacks.EarlyStopping(
            monitor="loss", patience=patience, restore_best_weights=True
        )
        tensorboard = K.callbacks.TensorBoard(log_dir=log_dir)
        callbacks = [early_stop, tensorboard]

        class ImageReconstructionCallback(K.callbacks.Callback):
            def __init__(self, data, log_dir, every_n_epochs=5, max_images=8):
                super().__init__()
                self.data = data  
                self.log_dir = log_dir
                self.every_n_epochs = every_n_epochs
                self.max_images = max_images
                self.file_writer = tf.summary.create_file_writer(log_dir)

            def on_epoch_end(self, epoch, logs=None):
                if epoch == 0 or (epoch + 1) % self.every_n_epochs != 0:
                    return
                x = self.data[:self.max_images]
                # Reconstrucción
                z_mean, z_log_var, z = self.encoder.predict(x)
                x_hat = self.decoder.predict(z)
                # Concatenar originales y reconstruidas
                comparison = np.concatenate([x, x_hat])
                comparison = np.clip(comparison, 0, 1)
                with self.file_writer.as_default():
                    tf.summary.image("Reconstructions", comparison, step=epoch, max_outputs=2*self.max_images)
        if recon_data is not None:
            recon_cb = ImageReconstructionCallback(
                data=recon_data,
                log_dir=log_dir+"/images",
                every_n_epochs=recon_every_n_epochs,
                max_images=recon_max_images
            )
            callbacks.append(recon_cb)
        return callbacks
