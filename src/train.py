"""
train.py
--------
Training pipeline for AE and VAE models across all anatomical regions.
"""

import os
from typing import Dict, List, Optional
import tensorflow as tf
from tensorflow import keras

from src.data_processing import build_dataset, find_region_path
from src.model import build_autoencoder, build_vae


# Config
ANATOMICAL_REGIONS: List[str] = [
    "AbdomenCT",
    "BreastMRI",
    "ChestCT",
    "CXR",
    "Hand",
    "HeadCT",
]

MODELS_DIR: str = "models"
LATENT_DIM: int = 2
EPOCHS: int = 25   
BATCH_SIZE: int = 32
LEARNING_RATE: float = 1e-3


# Callbacks
def get_ae_callbacks() -> List[keras.callbacks.Callback]:
    return [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=7,
            restore_best_weights=True,
            mode="min",
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=4,
            mode="min",
            verbose=1,
        ),
    ]


def get_vae_callbacks() -> List[keras.callbacks.Callback]:
    return [
        keras.callbacks.EarlyStopping(
            monitor="val_val_total_loss",  # ✅ FIXED
            patience=7,
            restore_best_weights=True,
            mode="min",
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_val_total_loss",  # ✅ FIXED
            factor=0.5,
            patience=4,
            mode="min",
            verbose=1,
        ),
    ]


# -----------------------------
# Train single region
# -----------------------------
def train_region(
    region: str,
    dataset_path: str,
    epochs: int = EPOCHS,
    latent_dim: int = LATENT_DIM,
    batch_size: int = BATCH_SIZE,
    learning_rate: float = LEARNING_RATE,
    models_dir: str = MODELS_DIR,
) -> Dict:

    print(f"\n{'='*60}")
    print(f"  Training region: {region}")
    print(f"{'='*60}")

    os.makedirs(models_dir, exist_ok=True)

    # Data
    
    region_dir = find_region_path(dataset_path, region)
    train_ds, val_ds = build_dataset(region_dir, batch_size=batch_size)

    # AE
    print(f"\n  🔵 Training AE for {region}...")
    ae_model, ae_encoder, ae_decoder = build_autoencoder(latent_dim, learning_rate)

    ae_history = ae_model.fit(
        train_ds,
        epochs=epochs,
        validation_data=val_ds,
        callbacks=get_ae_callbacks(),
        verbose=1,
    )

    ae_path = os.path.join(models_dir, f"ae_{region.lower()}_v1.keras")
    ae_model.save(ae_path)
    print(f"  ✅ AE saved → {ae_path}")

    # -----------------------------
    # VAE
    # -----------------------------
    print(f"\n  🟣 Training VAE for {region}...")
    vae_model, vae_encoder, vae_decoder = build_vae(latent_dim, learning_rate)

    vae_history = vae_model.fit(
        train_ds,
        epochs=epochs,
        validation_data=val_ds,
        callbacks=get_vae_callbacks(),
        verbose=1,
    )

    # ✅ Build model before saving (fix warning)
    dummy_input = tf.zeros((1, 64, 64, 1))
    vae_model.predict(dummy_input)

    # ✅ Save weights instead of full model (fix serialization error)
    vae_path = os.path.join(models_dir, f"vae_{region.lower()}_v1.weights.h5")
    vae_model.save_weights(vae_path)

    print(f"  ✅ VAE weights saved → {vae_path}")

    return {
        "ae": {
            "model": ae_model,
            "encoder": ae_encoder,
            "decoder": ae_decoder,
            "history": ae_history,
        },
        "vae": {
            "model": vae_model,
            "encoder": vae_encoder,
            "decoder": vae_decoder,
            "history": vae_history,
        },
        "train_ds": train_ds,
        "val_ds": val_ds,
    }


# Train all regions
def train_all(
    dataset_path: str,
    regions: Optional[List[str]] = None,
    epochs: int = EPOCHS,
    latent_dim: int = LATENT_DIM,
    batch_size: int = BATCH_SIZE,
    learning_rate: float = LEARNING_RATE,
    models_dir: str = MODELS_DIR,
) -> Dict[str, Dict]:

    regions = regions or ANATOMICAL_REGIONS
    results: Dict[str, Dict] = {}

    for region in regions:
        results[region] = train_region(
            region=region,
            dataset_path=dataset_path,
            epochs=epochs,
            latent_dim=latent_dim,
            batch_size=batch_size,
            learning_rate=learning_rate,
            models_dir=models_dir,
        )

    print("\n🎉 All regions trained successfully!")
    return results


# Main
if __name__ == "__main__":
    import kagglehub

    os.environ["KAGGLEHUB_VERIFY_SSL"] = "0"

    DATASET_PATH = kagglehub.dataset_download("andrewmvd/medical-mnist")
    print(f"Dataset path: {DATASET_PATH}")

    train_all(dataset_path=DATASET_PATH)