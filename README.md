# Medical MNIST AE & VAE Representation Learning Analysis

A complete deep learning project for **medical image reconstruction, latent representation learning, denoising, and generative modeling** using **Autoencoders (AE)** and **Variational Autoencoders (VAE)** on the **Medical MNIST** dataset.

This project was developed for **DSAI 490 - Assignment 1: Representation Learning with Autoencoders (AE & VAE)**.

---

## 📌 Project Overview

Modern machine learning systems rely heavily on the ability to learn compact, meaningful, and useful data representations. Autoencoders and Variational Autoencoders are powerful unsupervised deep learning models used for:

- Image reconstruction
- Dimensionality reduction
- Latent space learning
- Denoising
- Generative modeling
- Medical image representation analysis

This project implements and compares:

1. **Autoencoder (AE)**
2. **Variational Autoencoder (VAE)**

using convolutional neural network architectures on grayscale medical images.

---

## 🎯 Main Objectives

The main goals of this project are:

- Build a convolutional Autoencoder for image reconstruction.
- Build a convolutional Variational Autoencoder with probabilistic latent space.
- Apply the reparameterization trick in the VAE.
- Train models on Medical MNIST anatomical image categories.
- Compare AE and VAE behavior.
- Analyze reconstruction performance.
- Analyze latent space behavior.
- Generate new medical-like samples using the VAE decoder.
- Test denoising capability.
- Maintain a clean, modular, and professional machine learning codebase.

---

## 🧠 Key Concepts

### Autoencoder

An Autoencoder is a neural network that learns to reconstruct its input.

It consists of:

```text
Input Image → Encoder → Latent Vector → Decoder → Reconstructed Image
```

The Autoencoder learns a compressed representation of the input data by minimizing reconstruction error.

For an input image \(x\), the Autoencoder reconstructs \(\hat{x}\):

\[
x \rightarrow z \rightarrow \hat{x}
\]

The training objective is typically:

\[
\mathcal{L}_{AE} = \text{ReconstructionLoss}(x, \hat{x})
\]

In this project, the AE is trained using Mean Squared Error:

\[
\mathcal{L}_{AE} = \frac{1}{N}\sum_{i=1}^{N}(x_i - \hat{x}_i)^2
\]

---

### Variational Autoencoder

A Variational Autoencoder extends the Autoencoder by learning a probability distribution in latent space.

Instead of mapping an input directly to one latent vector, the encoder predicts:

\[
\mu
\]

and

\[
\log(\sigma^2)
\]

Then a latent vector is sampled using the reparameterization trick:

\[
z = \mu + \sigma \cdot \epsilon
\]

where:

\[
\epsilon \sim \mathcal{N}(0, I)
\]

The VAE loss consists of two terms:

\[
\mathcal{L}_{VAE} = \mathcal{L}_{reconstruction} + \mathcal{L}_{KL}
\]

The KL divergence term is:

\[
\mathcal{L}_{KL} =
-\frac{1}{2}
\sum
\left(
1 + \log(\sigma^2) - \mu^2 - \sigma^2
\right)
\]

This encourages the learned latent distribution to be close to a standard normal distribution:

\[
z \sim \mathcal{N}(0, I)
\]

---

## 📊 AE vs VAE Comparison

| Aspect | Autoencoder | Variational Autoencoder |
|---|---|---|
| Latent Space Type | Deterministic | Probabilistic |
| Encoder Output | Single latent vector | Mean, log variance, sampled latent vector |
| Main Goal | Reconstruction | Reconstruction + generation |
| Loss | Reconstruction loss | Reconstruction loss + KL divergence |
| Reconstruction Quality | Usually sharper | Usually smoother |
| Generation Ability | Limited | Stronger |
| Latent Space | Less structured | More continuous and regularized |
| Best Use Case | Compression and reconstruction | Generative modeling and smooth interpolation |

---

## 🏥 Dataset

This project uses the **Medical MNIST** dataset.

Dataset source:

```text
https://www.kaggle.com/datasets/andrewmvd/medical-mnist
```

The dataset contains grayscale medical images divided into six anatomical categories.

---

## 🧬 Dataset Classes

| Class ID | Class Name |
|---:|---|
| 0 | AbdomenCT |
| 1 | BreastMRI |
| 2 | ChestCT |
| 3 | CXR |
| 4 | Hand |
| 5 | HeadCT |

---

## 🖼️ Image Preprocessing

All images are processed using a `tf.data` pipeline.

Preprocessing steps:

1. Read image file from disk.
2. Decode image as grayscale.
3. Resize image to `64 × 64`.
4. Normalize pixel values to `[0, 1]`.
5. Use image as both input and target for reconstruction.

```python
img = tf.io.read_file(file_path)
img = tf.image.decode_jpeg(img, channels=1)
img = tf.image.resize(img, [64, 64])
img = tf.cast(img, tf.float32) / 255.0
```

---

## 📁 Project Structure

```text
project/
│
├── data/
│   ├── raw/
│   │   └── Medical MNIST dataset folders go here
│   │
│   └── processed/
│       └── Optional folder for processed data
│
├── models/
│   ├── ae_abdomenct_v1.keras
│   ├── vae_abdomenct_v1.weights.h5
│   ├── ae_breastmri_v1.keras
│   ├── vae_breastmri_v1.weights.h5
│   ├── ae_chestct_v1.keras
│   ├── vae_chestct_v1.weights.h5
│   ├── ae_cxr_v1.keras
│   ├── vae_cxr_v1.weights.h5
│   ├── ae_hand_v1.keras
│   ├── vae_hand_v1.weights.h5
│   ├── ae_headct_v1.keras
│   └── vae_headct_v1.weights.h5
│
├── notebooks/
│   ├── dsai490-assignment-1-*.ipynb
│   ├── loss_<region>.png
│   ├── reconstruction_<region>.png
│   ├── latent_space_<region>.png
│   ├── generated_samples_<region>.png
│   ├── denoising_<region>.png
│   └── comparison_ae_vs_vae.png
│
├── src/
│   ├── __init__.py
│   ├── data_processing.py
│   ├── model.py
│   └── train.py
│
├── tests/
│   ├── test_data_processing.py
│   └── test_model.py
│
├── README.md
└── requirements.txt
```

---

## 📂 Folder Explanation

| Folder | Purpose |
|---|---|
| `data/raw/` | Stores the original Medical MNIST dataset |
| `data/processed/` | Optional folder for processed data |
| `models/` | Stores trained AE models and VAE weights |
| `notebooks/` | Stores experiment notebooks and output visualizations |
| `src/` | Source code for data loading, model definitions, and training |
| `tests/` | Unit tests for data processing and model architecture |
| `requirements.txt` | Python package dependencies |

---

## ⚠️ Note About `data/processed/`

The `data/processed/` folder is included for standard project organization.

However, in this project, preprocessing is performed dynamically using the `tf.data` pipeline.

Therefore, `data/processed/` may remain empty.

This is intentional because:

- Images are resized during loading.
- Images are normalized during loading.
- No duplicate processed dataset is saved to disk.
- The pipeline remains memory-efficient.

---

## 🧱 Codebase Design

The project follows a modular structure.

### `src/data_processing.py`

Responsible for:

- Loading images
- Parsing image files
- Normalizing pixel values
- Creating TensorFlow datasets
- Finding anatomical region folders

Main functions:

| Function | Description |
|---|---|
| `parse_image(file_path)` | Loads and preprocesses one image |
| `build_dataset(region_path, batch_size, val_split, seed)` | Builds train and validation datasets |
| `find_region_path(dataset_root, region_name)` | Searches recursively for a region folder |

---

### `src/model.py`

Responsible for:

- Building AE encoder
- Building AE decoder
- Building full AE
- Building VAE encoder
- Building VAE decoder
- Implementing the sampling layer
- Implementing custom VAE training logic

Main components:

| Component | Description |
|---|---|
| `build_encoder_ae()` | Builds Autoencoder encoder |
| `build_decoder_ae()` | Builds Autoencoder decoder |
| `build_autoencoder()` | Builds and compiles full AE |
| `Sampling` | Reparameterization trick layer |
| `build_encoder_vae()` | Builds VAE encoder |
| `build_decoder_vae()` | Builds VAE decoder |
| `VAE` | Custom Keras model subclass |
| `build_vae()` | Builds and compiles full VAE |

---

### `src/train.py`

Responsible for:

- Training AE and VAE models
- Training across anatomical regions
- Saving AE full models
- Saving VAE weights
- Using callbacks for stable training

Main functions:

| Function | Description |
|---|---|
| `get_ae_callbacks()` | Returns callbacks for AE training |
| `get_vae_callbacks()` | Returns callbacks for VAE training |
| `train_region()` | Trains AE and VAE for one anatomical region |
| `train_all()` | Trains models for all anatomical regions |

---

## 🧠 Model Architecture

## Autoencoder Architecture

### Encoder

```text
Input: 64 × 64 × 1

Conv2D(32, kernel=3, stride=2, relu)
↓
Conv2D(64, kernel=3, stride=2, relu)
↓
Conv2D(128, kernel=3, stride=2, relu)
↓
Flatten
↓
Dense(256, relu)
↓
Dense(latent_dim)

Output: latent vector
```

### Decoder

```text
Input: latent vector

Dense(8 × 8 × 128, relu)
↓
Reshape(8, 8, 128)
↓
Conv2DTranspose(128, kernel=3, stride=2, relu)
↓
Conv2DTranspose(64, kernel=3, stride=2, relu)
↓
Conv2DTranspose(32, kernel=3, stride=2, relu)
↓
Conv2DTranspose(1, kernel=3, sigmoid)

Output: 64 × 64 × 1 reconstructed image
```

---

## Variational Autoencoder Architecture

### VAE Encoder

```text
Input: 64 × 64 × 1

Conv2D(32, kernel=3, stride=2, relu)
↓
Conv2D(64, kernel=3, stride=2, relu)
↓
Conv2D(128, kernel=3, stride=2, relu)
↓
Flatten
↓
Dense(256, relu)
↓
Dense(latent_dim) → z_mean
Dense(latent_dim) → z_log_var
↓
Sampling Layer
↓
z
```

### VAE Decoder

```text
Input: sampled latent vector z

Dense(8 × 8 × 128, relu)
↓
Reshape(8, 8, 128)
↓
Conv2DTranspose(128, kernel=3, stride=2, relu)
↓
Conv2DTranspose(64, kernel=3, stride=2, relu)
↓
Conv2DTranspose(32, kernel=3, stride=2, relu)
↓
Conv2DTranspose(1, kernel=3, sigmoid)

Output: 64 × 64 × 1 reconstructed image
```

---

## ⚙️ Training Configuration

| Parameter | Value |
|---|---:|
| Image size | `64 × 64` |
| Channels | `1` |
| Batch size | `32` |
| Latent dimension | `2` |
| Epochs | `25` |
| Optimizer | Adam |
| Learning rate | `1e-3` |
| Validation split | `0.2` |
| AE loss | MSE |
| VAE loss | Reconstruction loss + KL divergence |
| AE saved format | `.keras` |
| VAE saved format | `.weights.h5` |

---

## 🧪 Training Strategy

The models are trained separately for each anatomical region.

Regions:

```python
ANATOMICAL_REGIONS = [
    "AbdomenCT",
    "BreastMRI",
    "ChestCT",
    "CXR",
    "Hand",
    "HeadCT",
]
```

For each region:

1. Locate region folder.
2. Build train and validation datasets.
3. Train AE.
4. Save AE model.
5. Train VAE.
6. Save VAE weights.

---

## 🔁 Training Callbacks

### Autoencoder Callbacks

```python
keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=7,
    restore_best_weights=True,
    mode="min",
)

keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=4,
    mode="min",
    verbose=1,
)
```

### VAE Callbacks

```python
keras.callbacks.EarlyStopping(
    monitor="val_val_total_loss",
    patience=7,
    restore_best_weights=True,
    mode="min",
)

keras.callbacks.ReduceLROnPlateau(
    monitor="val_val_total_loss",
    factor=0.5,
    patience=4,
    mode="min",
    verbose=1,
)
```

---

## 💾 Model Saving

### Autoencoder

The Autoencoder is a standard Keras functional model, so it is saved as:

```python
ae_model.save("models/ae_<region>_v1.keras")
```

Example:

```text
models/ae_abdomenct_v1.keras
```

---

### Variational Autoencoder

The VAE is implemented as a custom subclassed Keras model.

Because custom subclassed models may require explicit serialization logic, VAE weights are saved instead of saving the full model object.

```python
vae_model.save_weights("models/vae_<region>_v1.weights.h5")
```

Example:

```text
models/vae_abdomenct_v1.weights.h5
```

This avoids serialization errors while preserving trained parameters.

---

## 🧪 Unit Testing

Unit tests are included in the `tests/` directory.

### Test Files

| Test File | Purpose |
|---|---|
| `test_data_processing.py` | Tests image parsing, normalization, dataset construction, and folder search |
| `test_model.py` | Tests AE encoder, AE decoder, AE model, Sampling layer, VAE encoder, VAE decoder, and VAE forward pass |

---

## ✅ Test Results

The test suite was executed successfully.

```text
Ran 20 tests in 2.425s

OK
```

This confirms:

- Data preprocessing functions work correctly.
- Output image shapes are correct.
- Pixel values are normalized to `[0, 1]`.
- Dataset returns image-target pairs.
- AE encoder and decoder shapes are correct.
- AE forward pass works.
- VAE encoder outputs three tensors.
- Sampling layer works correctly.
- VAE forward pass works.

---

## 🧪 Running Tests

### Using `pytest`

```bash
python -m pytest tests/ -v
```

### Using `unittest`

```bash
python -m unittest discover tests
```

---

## 🚀 Running Training

To train all models:

```bash
python -m src.train
```

The script will:

1. Download the Medical MNIST dataset using `kagglehub`.
2. Locate each anatomical region folder.
3. Train AE and VAE models.
4. Save trained models and weights in the `models/` folder.

---

## 📦 Requirements

Install all dependencies using:

```bash
pip install -r requirements.txt
```

---

## 🧰 Main Dependencies

| Package | Purpose |
|---|---|
| TensorFlow | Deep learning framework |
| Keras | Neural network API |
| NumPy | Numerical operations |
| Matplotlib | Visualization |
| Seaborn | Visualization |
| Scikit-learn | Latent space visualization tools |
| Pillow | Image handling |
| KaggleHub | Dataset download |
| Pytest | Unit testing |

---

## 📄 Example `requirements.txt`

```text
tensorflow>=2.15.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.3.0
kagglehub>=0.2.0
pytest>=7.2.0
pillow
```

---

## 🖥️ Environment Notes

This project was tested on:

```text
Python 3.11
TensorFlow 2.21.0
Windows
CPU execution
```

TensorFlow produced the following warning on native Windows:

```text
TensorFlow GPU support is not available on native Windows for TensorFlow >= 2.11.
```

This does not affect correctness.

It only means training runs on CPU unless using WSL2 or TensorFlow DirectML.

---

## 📈 Training Results Summary

### AbdomenCT

#### Autoencoder

| Metric | Initial | Final |
|---|---:|---:|
| Training loss | ~0.0076 | ~0.0038 |
| Validation loss | ~0.0062 | ~0.0038 |
| Training MAE | ~0.0504 | ~0.0317 |
| Validation MAE | ~0.0425 | ~0.0317 |

Interpretation:

- The AE showed stable convergence.
- Reconstruction loss decreased significantly.
- Validation loss followed training loss closely.
- No major overfitting was observed.

#### VAE

| Metric | Initial | Final |
|---|---:|---:|
| KL loss | ~0.1088 | ~5.2943 |
| Reconstruction loss | ~2792.18 | ~2756.17 |
| Total loss | ~2792.28 | ~2761.46 |
| Validation total loss | ~2786.73 | ~2762.14 |

Interpretation:

- Reconstruction loss decreased gradually.
- KL loss increased as the latent distribution became more structured.
- This is expected VAE behavior.

---

### BreastMRI

#### Autoencoder

| Metric | Initial | Final |
|---|---:|---:|
| Training loss | ~0.0685 | ~0.0587 |
| Validation loss | ~0.0577 | ~0.0586 |
| Training MAE | ~0.1306 | ~0.1057 |
| Validation MAE | ~0.1044 | ~0.1058 |

Interpretation:

- AE training showed limited improvement.
- BreastMRI appears more challenging.
- The model plateaued early compared with CT image categories.
- This may be due to higher image complexity and lower structural uniformity.

#### VAE

| Metric | Initial | Final |
|---|---:|---:|
| KL loss | ~9.9931 | ~7.0990 |
| Reconstruction loss | ~1186.39 | ~804.93 |
| Total loss | ~1196.39 | ~812.03 |
| Validation total loss | ~1024.22 | ~810.92 |

Interpretation:

- VAE showed strong improvement on BreastMRI.
- Reconstruction loss decreased significantly.
- VAE learned a useful latent representation for this region.

---

### ChestCT

#### Autoencoder

| Metric | Initial | Final |
|---|---:|---:|
| Training loss | ~0.0034 | ~0.0027 |
| Validation loss | ~0.0033 | ~0.0027 |
| Training MAE | ~0.0283 | ~0.0233 |
| Validation MAE | ~0.0275 | ~0.0231 |

Interpretation:

- ChestCT was one of the easiest categories for reconstruction.
- AE achieved very low loss.
- Training was stable and consistent.

#### VAE

| Metric | Initial | Final |
|---|---:|---:|
| KL loss | ~0.0021 | ~1.0752 |
| Reconstruction loss | ~2835.26 | ~2833.13 |
| Total loss | ~2835.26 | ~2834.21 |
| Validation total loss | ~2834.91 | ~2834.19 |

Interpretation:

- VAE reconstruction loss changed only slightly.
- KL loss increased gradually.
- This suggests stronger latent regularization and limited reconstruction improvement.

---

### CXR

#### Autoencoder

| Metric | Initial | Final |
|---|---:|---:|
| Training loss | ~0.0431 | ~0.0245 |
| Validation loss | ~0.0344 | ~0.0243 |
| Training MAE | ~0.1576 | ~0.1128 |
| Validation MAE | ~0.1387 | ~0.1118 |

Interpretation:

- AE showed clear improvement.
- CXR images were harder than CT categories.
- Loss decreased steadily across epochs.

---

## 📌 General Observations

### AE Behavior

Autoencoders generally achieved better reconstruction quality because they optimize directly for image reconstruction.

They tend to preserve sharper details because there is no KL divergence regularization.

### VAE Behavior

VAEs generally produced smoother reconstructions because the latent space is constrained to follow a distribution close to a standard normal distribution.

This introduces a trade-off:

```text
Better latent structure and generation ability
vs.
Lower reconstruction sharpness
```

### Dataset Difficulty

| Dataset | Relative Difficulty | Observation |
|---|---|---|
| ChestCT | Easy | Very low AE reconstruction loss |
| AbdomenCT | Medium | Stable AE and VAE training |
| BreastMRI | Hard | AE plateaued, VAE improved strongly |
| CXR | Hard | Higher loss due to image complexity |
| Hand | Expected Medium | Structural anatomy may support reconstruction |
| HeadCT | Expected Medium | CT structure likely supports stable reconstruction |

---

## 📊 Required Output Visualizations

The following visualizations are expected to be generated and stored in the `notebooks/` directory.

| Output File | Description |
|---|---|
| `loss_<region>.png` | Training and validation loss curves |
| `reconstruction_<region>.png` | Original vs reconstructed images |
| `latent_space_<region>.png` | 2D latent space visualization |
| `generated_samples_<region>.png` | VAE-generated samples |
| `latent_grid_<region>.png` | VAE latent grid walk |
| `denoising_<region>.png` | Noisy input and reconstructed output |
| `comparison_ae_vs_vae.png` | Quantitative AE vs VAE comparison |

---

## 🖼️ Reconstruction Analysis

The reconstruction task compares:

```text
Original Image vs Reconstructed Image
```

For AE:

```text
Input → Encoder → Latent Vector → Decoder → Reconstruction
```

For VAE:

```text
Input → Encoder → Mean and Log Variance → Sampling → Decoder → Reconstruction
```

Expected behavior:

| Model | Reconstruction Behavior |
|---|---|
| AE | Sharper and more faithful reconstructions |
| VAE | Smoother reconstructions with more regularized structure |

---

## 🌌 Latent Space Analysis

The latent dimension is set to:

```python
LATENT_DIM = 2
```

This allows direct visualization of the learned latent space.

### AE Latent Space

The AE latent space may form useful clusters but is not explicitly regularized.

### VAE Latent Space

The VAE latent space is expected to be smoother and more continuous due to the KL divergence term.

The VAE objective encourages:

\[
q(z|x) \approx \mathcal{N}(0, I)
\]

This makes the latent space better suited for generation and interpolation.

---

## 🧪 Denoising Experiment

The denoising experiment tests model robustness.

Noise is added to an input image:

\[
x_{noisy} = x + \alpha \cdot \epsilon
\]

where:

\[
\epsilon \sim \mathcal{N}(0, I)
\]

and:

\[
\alpha = 0.3
\]

The model then reconstructs the noisy input.

Expected result:

```text
Original Image → Noisy Image → Denoised Reconstruction
```

This shows whether the model learned meaningful structure rather than simply memorizing pixels.

---

## 🎨 VAE Sample Generation

The VAE can generate new samples by sampling from the latent prior:

\[
z \sim \mathcal{N}(0, I)
\]

Then decoding:

\[
\hat{x} = Decoder(z)
\]

This allows the model to create new medical-like images from random latent vectors.

---

## 🔍 Key Findings

1. Autoencoders generally produce better reconstruction quality.
2. Variational Autoencoders learn smoother and more structured latent spaces.
3. VAEs are better suited for sample generation.
4. Reconstruction difficulty varies by anatomical region.
5. ChestCT and AbdomenCT are easier to reconstruct than BreastMRI and CXR.
6. BreastMRI showed slower AE convergence, likely due to higher visual complexity.
7. VAE loss behavior reflects the trade-off between reconstruction and KL regularization.
8. A latent dimension of 2 enables direct visualization but limits reconstruction capacity.
9. Training with callbacks improves stability.
10. Modular project structure improves reproducibility and maintainability.

---

## ⚠️ Challenges Encountered

### 1. Dataset Path Issues

Initial dataset loading failed because the dataset was not attached or the path was incorrect.

This was resolved by verifying the dataset location and using `kagglehub`.

---

### 2. TensorFlow Environment Setup

TensorFlow and pytest were missing locally.

This was resolved by installing dependencies from `requirements.txt`.

---

### 3. VAE Serialization Issue

Saving the custom VAE model using:

```python
vae_model.save(vae_path)
```

caused a serialization error because the VAE is a subclassed custom Keras model.

The solution was to save weights instead:

```python
vae_model.save_weights(vae_path)
```

---

### 4. VAE Callback Metric Name

The VAE validation metrics appeared with names such as:

```text
val_val_total_loss
```

Therefore, callbacks were updated to monitor:

```python
monitor="val_val_total_loss"
```

---

### 5. CPU Training

Training was performed on CPU in the local Windows environment.

TensorFlow warned that GPU support is not available on native Windows for TensorFlow versions greater than or equal to 2.11.

Training still completed successfully.

---

## 🧪 Example Usage

### Train All Models

```bash
python -m src.train
```

### Run Unit Tests

```bash
python -m unittest discover tests
```

or:

```bash
python -m pytest tests/ -v
```

---

## 🧩 Loading a Saved AE Model

```python
import tensorflow as tf

ae_model = tf.keras.models.load_model("models/ae_abdomenct_v1.keras")
```

---

## 🧩 Loading a Saved VAE Weights File

Because the VAE is a custom subclassed model, rebuild the architecture first:

```python
from src.model import build_vae

vae_model, vae_encoder, vae_decoder = build_vae(latent_dim=2)

vae_model.load_weights("models/vae_abdomenct_v1.weights.h5")
```

---

## 📌 Recommended GitHub Upload Policy

The repository should include:

```text
src/
tests/
models/
notebooks/
README.md
requirements.txt
.gitignore
```

The repository should not include large raw datasets.

Recommended `.gitignore`:

```text
data/raw/
.cache/
__pycache__/
*.pyc
.ipynb_checkpoints/
.DS_Store
```

---

## 🧾 Assignment Deliverables Covered

| Requirement | Status |
|---|---|
| Autoencoder implementation | Completed |
| VAE implementation | Completed |
| Reparameterization trick | Completed |
| tf.data pipeline | Completed |
| Modular codebase | Completed |
| GitHub repository | Completed |
| Unit tests | Completed |
| Training pipeline | Completed |
| Reconstruction analysis | Included |
| Latent space visualization | Planned / notebook output |
| Sample generation | Planned / notebook output |
| Denoising capability | Planned / notebook output |
| Technical report support | Included |
| Video presentation support | Included |

---

## 📚 Technical Report Summary

The report can be based on the following points:

### Autoencoder Summary

The Autoencoder uses a convolutional encoder-decoder architecture to compress medical images into a 2-dimensional latent representation and reconstruct them. It achieved stable reconstruction performance, especially on CT-based datasets.

### VAE Summary

The Variational Autoencoder extends the AE by learning a probabilistic latent space. It uses mean and log variance vectors, followed by the reparameterization trick. The VAE objective combines reconstruction loss and KL divergence, allowing both reconstruction and generation.

### Key Difference

The AE focuses on reconstruction accuracy, while the VAE balances reconstruction and latent space regularization. As a result, AE reconstructions tend to be sharper, while VAE outputs tend to be smoother but more suitable for generation.

---

## 🎤 Presentation Summary

Recommended presentation sections:

1. Title
2. Introduction
3. Dataset
4. Problem Statement
5. Methodology
6. AE Architecture
7. VAE Architecture
8. Training Setup
9. AE Results
10. VAE Results
11. Reconstruction Comparison
12. Latent Space Visualization
13. Generated Samples
14. Denoising Results
15. Challenges
16. Conclusion
17. Future Work

---

## 🔮 Future Work

Possible improvements:

- Train one unified AE and VAE on all six classes together.
- Increase latent dimension from 2 to 16 or 32 for better reconstruction.
- Use deeper convolutional architectures.
- Add batch normalization.
- Add dropout for regularization.
- Use perceptual loss for better reconstruction quality.
- Perform quantitative comparison using SSIM or PSNR.
- Use UMAP or t-SNE for latent space visualization.
- Train longer on GPU.
- Add class-conditioned VAE.
- Implement Conditional VAE.
- Compare with GAN-based generation models.

---

## 🧠 Final Conclusion

This project demonstrates the use of Autoencoders and Variational Autoencoders for representation learning on medical image data.

The Autoencoder was effective for image reconstruction and achieved low reconstruction losses on structured datasets such as ChestCT and AbdomenCT.

The Variational Autoencoder learned probabilistic latent representations and enabled generative modeling through sampling from the latent space.

Overall, the project shows that:

- AE is better for reconstruction-focused tasks.
- VAE is better for generative and latent-space-focused tasks.
- Medical image structure strongly affects reconstruction difficulty.
- Clean code organization and reproducible pipelines are essential for deep learning experiments.

---

## 👤 Author

```text
Mohamed Ehab Yousri
DSAI 490 - Assignment 1
Medical MNIST AE & VAE Representation Learning Analysis
```

---

## 🔗 Repository

```text
https://github.com/mohamed1232005/Medical-Mnist-AE-VAE-Representation-Learning-Analysis
```
