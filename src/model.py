"""
model.py
--------
Defines the Autoencoder (AE) and Variational Autoencoder (VAE)
architectures using TensorFlow / Keras.
"""

# Standard library imports
from typing import List, Tuple

# Third-party imports
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


IMG_HEIGHT: int = 64
IMG_WIDTH: int = 64
IMG_CHANNELS: int = 1


# AUTOENCODER (AE)

def build_encoder_ae(latent_dim: int = 2) -> keras.Model:

    inputs = keras.Input(
        shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS),
        name="ae_encoder_input"
    )
    x = layers.Conv2D(32, 3, strides=2, padding="same", activation="relu")(inputs)
    x = layers.Conv2D(64, 3, strides=2, padding="same", activation="relu")(x)
    x = layers.Conv2D(128, 3, strides=2, padding="same", activation="relu")(x)
    x = layers.Flatten()(x)
    x = layers.Dense(256, activation="relu")(x)
    z = layers.Dense(latent_dim, name="z")(x)
    return keras.Model(inputs, z, name="AE_Encoder")


def build_decoder_ae(latent_dim: int = 2) -> keras.Model:

    inputs = keras.Input(shape=(latent_dim,), name="ae_decoder_input")
    x = layers.Dense(8 * 8 * 128, activation="relu")(inputs)
    x = layers.Reshape((8, 8, 128))(x)
    x = layers.Conv2DTranspose(128, 3, strides=2, padding="same", activation="relu")(x)
    x = layers.Conv2DTranspose(64, 3, strides=2, padding="same", activation="relu")(x)
    x = layers.Conv2DTranspose(32, 3, strides=2, padding="same", activation="relu")(x)
    outputs = layers.Conv2DTranspose(
        IMG_CHANNELS, 3, padding="same",
        activation="sigmoid", name="ae_reconstruction"
    )(x)
    return keras.Model(inputs, outputs, name="AE_Decoder")


def build_autoencoder(
    latent_dim: int = 2,
    learning_rate: float = 1e-3,
) -> Tuple[keras.Model, keras.Model, keras.Model]:

    encoder = build_encoder_ae(latent_dim)
    decoder = build_decoder_ae(latent_dim)

    inputs = keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    z = encoder(inputs)
    reconstructed = decoder(z)

    autoencoder = keras.Model(inputs, reconstructed, name="Autoencoder")
    autoencoder.compile(
        optimizer=keras.optimizers.Adam(learning_rate),
        loss="mse",
        metrics=["mae"],
    )
    return autoencoder, encoder, decoder



# VARIATIONAL AUTOENCODER (VAE)

class Sampling(layers.Layer):
 

    def call(self, inputs: Tuple[tf.Tensor, tf.Tensor]) -> tf.Tensor:

        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


def build_encoder_vae(latent_dim: int = 2) -> keras.Model:

    inputs = keras.Input(
        shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS),
        name="vae_encoder_input"
    )
    x = layers.Conv2D(32, 3, strides=2, padding="same", activation="relu")(inputs)
    x = layers.Conv2D(64, 3, strides=2, padding="same", activation="relu")(x)
    x = layers.Conv2D(128, 3, strides=2, padding="same", activation="relu")(x)
    x = layers.Flatten()(x)
    x = layers.Dense(256, activation="relu")(x)

    z_mean    = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
    z         = Sampling(name="z_sampling")([z_mean, z_log_var])

    return keras.Model(inputs, [z_mean, z_log_var, z], name="VAE_Encoder")


def build_decoder_vae(latent_dim: int = 2) -> keras.Model:

    inputs = keras.Input(shape=(latent_dim,), name="vae_decoder_input")
    x = layers.Dense(8 * 8 * 128, activation="relu")(inputs)
    x = layers.Reshape((8, 8, 128))(x)
    x = layers.Conv2DTranspose(128, 3, strides=2, padding="same", activation="relu")(x)
    x = layers.Conv2DTranspose(64, 3, strides=2, padding="same", activation="relu")(x)
    x = layers.Conv2DTranspose(32, 3, strides=2, padding="same", activation="relu")(x)
    outputs = layers.Conv2DTranspose(
        IMG_CHANNELS, 3, padding="same",
        activation="sigmoid", name="vae_reconstruction"
    )(x)
    return keras.Model(inputs, outputs, name="VAE_Decoder")


class VAE(keras.Model):

    def __init__(
        self,
        encoder: keras.Model,
        decoder: keras.Model,
        **kwargs,
    ) -> None:

        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

        # Metric trackers
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.recon_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker    = keras.metrics.Mean(name="kl_loss")
        self.val_total_tracker  = keras.metrics.Mean(name="val_total_loss")
        self.val_recon_tracker  = keras.metrics.Mean(name="val_reconstruction_loss")
        self.val_kl_tracker     = keras.metrics.Mean(name="val_kl_loss")

    @property
    def metrics(self) -> List[keras.metrics.Metric]:
        """Return all tracked metrics."""
        return [
            self.total_loss_tracker,
            self.recon_loss_tracker,
            self.kl_loss_tracker,
            self.val_total_tracker,
            self.val_recon_tracker,
            self.val_kl_tracker,
        ]

    def _compute_losses(
        self,
        x: tf.Tensor,
        training: bool,
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:

        z_mean, z_log_var, z = self.encoder(x, training=training)
        reconstruction = self.decoder(z, training=training)

        recon_loss = tf.reduce_mean(
            tf.reduce_sum(
                keras.losses.binary_crossentropy(x, reconstruction),
                axis=(1, 2),
            )
        )
        kl_loss = -0.5 * tf.reduce_mean(
            tf.reduce_sum(
                1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var),
                axis=1,
            )
        )
        return recon_loss + kl_loss, recon_loss, kl_loss

    def train_step(self, data: Tuple) -> dict:

        x, _ = data
        with tf.GradientTape() as tape:
            total_loss, recon_loss, kl_loss = self._compute_losses(x, training=True)

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        self.total_loss_tracker.update_state(total_loss)
        self.recon_loss_tracker.update_state(recon_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        return {m.name: m.result() for m in [
            self.total_loss_tracker,
            self.recon_loss_tracker,
            self.kl_loss_tracker,
        ]}

    def test_step(self, data: Tuple) -> dict:

        x, _ = data
        total_loss, recon_loss, kl_loss = self._compute_losses(x, training=False)

        self.val_total_tracker.update_state(total_loss)
        self.val_recon_tracker.update_state(recon_loss)
        self.val_kl_tracker.update_state(kl_loss)

        return {m.name: m.result() for m in [
            self.val_total_tracker,
            self.val_recon_tracker,
            self.val_kl_tracker,
        ]}

    def call(self, inputs: tf.Tensor) -> tf.Tensor:

        z_mean, z_log_var, z = self.encoder(inputs)
        return self.decoder(z)


def build_vae(
    latent_dim: int = 2,
    learning_rate: float = 1e-3,
) -> Tuple["VAE", keras.Model, keras.Model]:

    encoder = build_encoder_vae(latent_dim)
    decoder = build_decoder_vae(latent_dim)
    vae = VAE(encoder, decoder, name="VAE")
    vae.compile(optimizer=keras.optimizers.Adam(learning_rate))
    return vae, encoder, decoder