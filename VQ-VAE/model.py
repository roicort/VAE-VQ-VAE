import numpy as np

import tensorflow as tf
from tensorflow import keras as K

class VQVAE(K.Model):

    def __init__(self, image_size, channels, latent_dim, train_variance, num_embeddings=64, beta=0.25, decoder=None, **kwargs):
        super().__init__(**kwargs)
        self.IMAGE_SIZE = image_size
        self.channels = channels
        self.latent_dim = latent_dim
        self.num_embeddings = num_embeddings
        self.beta = beta
        self.train_variance = train_variance

        class VectorQuantizer(K.layers.Layer):
            """
            VectorQuantizer as in https://keras.io/examples/generative/vq_vae/
            """
            def __init__(self, num_embeddings, embedding_dim, beta=0.25, **kwargs):
                super().__init__(**kwargs)
                self.embedding_dim = embedding_dim
                self.num_embeddings = num_embeddings

                # The `beta` parameter is best kept between [0.25, 2] as per the paper.
                self.beta = beta

                # Initialize the embeddings which we will quantize.
                w_init = tf.random_uniform_initializer()
                self.embeddings = tf.Variable(
                    initial_value=w_init(
                        shape=(self.embedding_dim, self.num_embeddings), dtype="float32"
                    ),
                    trainable=True,
                    name="embeddings_vqvae",
                )

            def call(self, x):
                # Calculate the input shape of the inputs and
                # then flatten the inputs keeping `embedding_dim` intact.
                input_shape = tf.shape(x)
                flattened = tf.reshape(x, [-1, self.embedding_dim])

                # Quantization.
                encoding_indices = self.get_code_indices(flattened)
                encodings = tf.one_hot(encoding_indices, self.num_embeddings)
                quantized = tf.matmul(encodings, self.embeddings, transpose_b=True)

                # Reshape the quantized values back to the original input shape
                quantized = tf.reshape(quantized, input_shape)

                # Calculate vector quantization loss and add that to the layer. You can learn more
                # about adding losses to different layers here:
                # https://keras.io/guides/making_new_layers_and_models_via_subclassing/. Check
                # the original paper to get a handle on the formulation of the loss function.
                commitment_loss = tf.reduce_mean((tf.stop_gradient(quantized) - x) ** 2)
                codebook_loss = tf.reduce_mean((quantized - tf.stop_gradient(x)) ** 2)
                self.add_loss(self.beta * commitment_loss + codebook_loss)

                # Straight-through estimator.
                quantized = x + tf.stop_gradient(quantized - x)
                return quantized

            def get_code_indices(self, flattened_inputs):
                # Calculate L2-normalized distance between the inputs and the codes.
                similarity = tf.matmul(flattened_inputs, self.embeddings)
                distances = (
                    tf.reduce_sum(flattened_inputs ** 2, axis=1, keepdims=True)
                    + tf.reduce_sum(self.embeddings ** 2, axis=0)
                    - 2 * similarity
                )

                # Derive the indices for minimum distances.
                encoding_indices = tf.argmin(distances, axis=1)
                return encoding_indices
    
        # Encoder
        encoder_inputs = K.layers.Input(shape=(self.IMAGE_SIZE, self.IMAGE_SIZE, self.channels))
        x = K.layers.Conv2D(32, 3, activation="relu", strides=2, padding="same")(encoder_inputs)
        x = K.layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(x)
        encoder_outputs = K.layers.Conv2D(self.latent_dim, 1, padding="same")(x)
        self.encoder = K.Model(encoder_inputs, encoder_outputs, name="encoder")

        # Decoder
        if decoder is not None:
            self.decoder = decoder
        else:
            latent_inputs = K.layers.Input(shape=self.encoder.output.shape[1:])
            x = K.layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(latent_inputs)
            x = K.layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)
            decoder_outputs = K.layers.Conv2DTranspose(self.channels, 3, padding="same", activation="sigmoid")(x)
            self.decoder = K.Model(latent_inputs, decoder_outputs, name="decoder")

        self.total_loss_tracker = K.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = K.metrics.Mean(name="reconstruction_loss")
        self.vq_loss_tracker = K.metrics.Mean(name="vq_loss")

        vq_layer = VectorQuantizer(num_embeddings=self.num_embeddings, embedding_dim=self.latent_dim, beta=self.beta, name="vector_quantizer")
        inputs = K.Input(shape=(self.IMAGE_SIZE, self.IMAGE_SIZE, self.channels))
        encoder_outputs = self.encoder(inputs)
        quantized_latents = vq_layer(encoder_outputs)
        reconstructions = self.decoder(quantized_latents)
        self.vqvae = K.Model(inputs, reconstructions, name="vq_vae")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.vq_loss_tracker,
        ]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            reconstructions = self.vqvae(data)
            # Pérdida de reconstrucción (MSE normalizada)
            reconstruction_loss = (
                tf.reduce_mean((data - reconstructions) ** 2) / self.train_variance
            )
            total_loss = reconstruction_loss + sum(self.vqvae.losses)

        # Backpropagation.
        grads = tape.gradient(total_loss, self.vqvae.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.vqvae.trainable_variables))

        # Loss tracking.
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.vq_loss_tracker.update_state(sum(self.vqvae.losses))
        
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "vqvae_loss": self.vq_loss_tracker.result(),
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
                x_hat = self.vqvae.predict(x)
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
