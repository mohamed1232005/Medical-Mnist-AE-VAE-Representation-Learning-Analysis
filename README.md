# 🧠 Medical MNIST AE & VAE Representation Learning Analysis

A complete deep learning project for **medical image reconstruction, latent representation learning, denoising, and generative modeling** using **Autoencoders (AE)** and **Variational Autoencoders (VAE)** on the **Medical MNIST dataset**.

This project was developed for:

> **DSAI 490 — Assignment 1: Representation Learning with Autoencoders**

---

# 📌 Project Overview

This project explores **unsupervised representation learning** using:

- Autoencoders (AE)
- Variational Autoencoders (VAE)

on grayscale medical images.

The goal is to learn **compact latent representations** that enable:

- Image reconstruction
- Latent space analysis
- Denoising
- Generative modeling

---

# 🎯 Objectives

- Build convolutional AE and VAE models
- Implement the reparameterization trick
- Compare reconstruction performance
- Analyze latent space behavior
- Generate new samples using VAE
- Evaluate denoising capability
- Maintain clean, modular ML code

---

# 🧠 Theory

## 🔹 Autoencoder (AE)

An Autoencoder learns a compressed representation of input data.

### Mapping

\[
z = f_{\theta}(x)
\]

\[
\hat{x} = g_{\phi}(z)
\]

\[
\hat{x} = g_{\phi}(f_{\theta}(x))
\]

---

### Loss Function (MSE)

\[
\mathcal{L}_{AE} =
\frac{1}{N} \sum_{i=1}^{N} \| x_i - \hat{x}_i \|^2
\]

or:

\[
\mathcal{L}_{AE} =
\mathbb{E}_{x \sim p_{data}(x)} \left[ \| x - \hat{x} \|^2 \right]
\]

---

## 🔹 Variational Autoencoder (VAE)

The VAE learns a **probabilistic latent space**.

### Encoder Distribution

\[
q_{\phi}(z|x) = \mathcal{N}(z; \mu(x), \sigma^2(x))
\]

---

### Reparameterization Trick

\[
z = \mu + \sigma \odot \epsilon
\]

\[
\epsilon \sim \mathcal{N}(0, I)
\]

---

### VAE Loss (ELBO)

\[
\mathcal{L}_{VAE} =
\mathbb{E}_{q_{\phi}(z|x)} \left[ \| x - \hat{x} \|^2 \right]
+
D_{KL} \left( q_{\phi}(z|x) \;||\; p(z) \right)
\]

---

### KL Divergence

\[
D_{KL} =
-\frac{1}{2}
\sum_{j=1}^{d}
\left(
1 + \log(\sigma_j^2) - \mu_j^2 - \sigma_j^2
\right)
\]

---

### Prior

\[
p(z) = \mathcal{N}(0, I)
\]

---

# ⚖️ AE vs VAE

| Aspect | AE | VAE |
|------|----|-----|
| Latent Space | Deterministic | Probabilistic |
| Output | Single vector | Distribution |
| Loss | Reconstruction | Reconstruction + KL |
| Reconstruction | Sharp | Smooth |
| Generation | Weak | Strong |

---

# 🏥 Dataset

Dataset:  
https://www.kaggle.com/datasets/andrewmvd/medical-mnist

---

## Classes

| ID | Class |
|----|------|
| 0 | AbdomenCT |
| 1 | BreastMRI |
| 2 | ChestCT |
| 3 | CXR |
| 4 | Hand |
| 5 | HeadCT |

---

# 🖼️ Preprocessing

```python
img = tf.io.read_file(path)
img = tf.image.decode_jpeg(img, channels=1)
img = tf.image.resize(img, [64, 64])
img = tf.cast(img, tf.float32) / 255.0
```

---

# 📁 Project Structure

```
project/
├── data/
│   ├── raw/
│   └── processed/
├── models/
├── notebooks/
├── src/
├── tests/
├── README.md
└── requirements.txt
```

---

# 🧱 Architecture

## AE Encoder

```
64×64×1 → Conv → Conv → Conv → Flatten → Dense → Latent
```

## AE Decoder

```
Latent → Dense → Reshape → Deconv → Output
```

---

## VAE Encoder

Outputs:

\[
\mu, \log \sigma^2
\]

---

## VAE Decoder

\[
\hat{x} = g_{\phi}(z)
\]

---

# ⚙️ Training Config

| Parameter | Value |
|----------|------|
| Image Size | 64×64 |
| Batch Size | 32 |
| Latent Dim | 2 |
| Epochs | 25 |
| LR | 1e-3 |

---

# 🔁 Denoising

\[
x_{noisy} = x + \alpha \cdot \epsilon
\]

\[
\epsilon \sim \mathcal{N}(0, I), \quad \alpha = 0.3
\]

---

# 🎨 Generation

\[
z \sim \mathcal{N}(0, I)
\]

\[
\hat{x} = g_{\phi}(z)
\]

---

# 📊 Results Summary

### AE
- Better reconstruction
- Sharp outputs

### VAE
- Smooth outputs
- Structured latent space
- Better generation

---

# 🧪 Tests

```
Ran 20 tests in ~2.4s
OK
```

---

# 🚀 Run

### Install

```bash
pip install -r requirements.txt
```

### Train

```bash
python -m src.train
```

### Test

```bash
python -m unittest discover tests
```

---

# 📦 Dependencies

- TensorFlow
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Pillow

---

# ⚠️ Notes

- CPU training (no GPU on Windows TF ≥2.11)
- VAE saved using weights only

---

# 🔍 Key Insights

- AE → best for reconstruction  
- VAE → best for generation  
- CT images easier than MRI/CXR  
- Latent space regularization improves structure  

---

# 🔮 Future Work

- Increase latent dimension
- Train unified model
- Add conditional VAE
- Use SSIM / PSNR
- Train on GPU

---

# 👤 Author

Mohamed Ehab Yousri  
DSAI 490 — Assignment 1  

---

# 🔗 Repo

https://github.com/mohamed1232005/Medical-Mnist-AE-VAE-Representation-Learning-Analysis
