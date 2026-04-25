"""
test_model.py
-------------
Unit tests for model.py — AE and VAE architectures.
"""


import unittest
import numpy as np
import tensorflow as tf
from src.model import (
    build_encoder_ae,
    build_decoder_ae,
    build_autoencoder,
    build_encoder_vae,
    build_decoder_vae,
    build_vae,
    Sampling,
    VAE,
)


class TestAEEncoder(unittest.TestCase):
    """Tests for the AE encoder."""

    def test_output_shape(self) -> None:
        """Encoder should output (batch, latent_dim)."""
        encoder = build_encoder_ae(latent_dim=2)
        x = tf.random.normal((4, 64, 64, 1))
        z = encoder(x)
        self.assertEqual(z.shape, (4, 2))

    def test_latent_dim_32(self) -> None:
        """Encoder should respect custom latent_dim."""
        encoder = build_encoder_ae(latent_dim=32)
        x = tf.random.normal((2, 64, 64, 1))
        self.assertEqual(encoder(x).shape, (2, 32))


class TestAEDecoder(unittest.TestCase):
    """Tests for the AE decoder."""

    def test_output_shape(self) -> None:
        """Decoder should output (batch, 64, 64, 1)."""
        decoder = build_decoder_ae(latent_dim=2)
        z = tf.random.normal((4, 2))
        out = decoder(z)
        self.assertEqual(out.shape, (4, 64, 64, 1))

    def test_output_range(self) -> None:
        """Decoder output should be in [0, 1] due to sigmoid."""
        decoder = build_decoder_ae(latent_dim=2)
        z = tf.random.normal((4, 2))
        out = decoder(z)
        self.assertGreaterEqual(float(tf.reduce_min(out)), 0.0)
        self.assertLessEqual(float(tf.reduce_max(out)), 1.0)


class TestAutoencoder(unittest.TestCase):
    """Tests for the full AE pipeline."""

    def test_reconstruction_shape(self) -> None:
        """AE output should match input shape."""
        ae, _, _ = build_autoencoder(latent_dim=2)
        x = tf.random.uniform((4, 64, 64, 1))
        out = ae(x)
        self.assertEqual(out.shape, x.shape)

    def test_forward_pass(self) -> None:
        """AE forward pass should not raise."""
        ae, _, _ = build_autoencoder(latent_dim=2)
        x = tf.random.uniform((2, 64, 64, 1))
        self.assertIsNotNone(ae(x))


class TestSampling(unittest.TestCase):
    """Tests for the Sampling reparameterization layer."""

    def test_output_shape(self) -> None:
        """Sampling output should match z_mean shape."""
        layer = Sampling()
        z_mean    = tf.zeros((4, 2))
        z_log_var = tf.zeros((4, 2))
        z = layer([z_mean, z_log_var])
        self.assertEqual(z.shape, (4, 2))

    def test_zero_variance(self) -> None:
        """With log_var=very_negative, output should be close to z_mean."""
        layer = Sampling()
        z_mean    = tf.ones((100, 2)) * 5.0
        z_log_var = tf.ones((100, 2)) * -20.0   # exp(-10) ≈ 0
        z = layer([z_mean, z_log_var])
        np.testing.assert_allclose(z.numpy(), z_mean.numpy(), atol=0.01)


class TestVAE(unittest.TestCase):
    """Tests for the full VAE model."""

    def test_encoder_outputs(self) -> None:
        """VAE encoder should return 3 outputs: z_mean, z_log_var, z."""
        encoder = build_encoder_vae(latent_dim=2)
        x = tf.random.normal((4, 64, 64, 1))
        outputs = encoder(x)
        self.assertEqual(len(outputs), 3)
        for o in outputs:
            self.assertEqual(o.shape, (4, 2))

    def test_decoder_output_shape(self) -> None:
        """VAE decoder should output (batch, 64, 64, 1)."""
        decoder = build_decoder_vae(latent_dim=2)
        z = tf.random.normal((4, 2))
        out = decoder(z)
        self.assertEqual(out.shape, (4, 64, 64, 1))

    def test_vae_call(self) -> None:
        """VAE call (encode+decode) should return input shape."""
        vae, _, _ = build_vae(latent_dim=2)
        x = tf.random.uniform((4, 64, 64, 1))
        out = vae(x)
        self.assertEqual(out.shape, x.shape)


if __name__ == "__main__":
    unittest.main()