# UV Wireless Sensor Network Anomaly Detection

This repository contains the implementation of an Intelligent Anomaly Detection framework for ultraviolet (UV) wireless sensor networks.

## Features

- UV-C wireless channel modeling with LOS and NLOS propagation
- Synthetic Channel Impulse Response (CIR) dataset generation
- Environmental impairment modeling
- Baseline Convolutional Autoencoder (CAE)
- Transformer-enhanced CAE
- Multi-task learning for:
  - CIR signal reconstruction
  - LOS/NLOS classification
  - RMS delay spread prediction
  - Link Gain Ratio (LGR) prediction
- Reconstruction-error-based unsupervised anomaly detection
- Statistical anomaly thresholding using:
  - Three-sigma
  - Percentile-based thresholding
  - Interquartile Range (IQR)
- Model evaluation under different channel conditions
- Reconstruction and SNR performance analysis

## Environmental Impairments

The UV channel model considers:

- Fog
- Temperature
- Ozone
- Humidity
- Mixed environmental conditions

## Models

### Baseline CAE

A convolutional autoencoder for joint CIR reconstruction, LOS/NLOS classification, RMS delay spread prediction, and LGR prediction.

### Transformer-based CAE

A Transformer-enhanced convolutional autoencoder incorporating multi-head self-attention to capture long-range temporal dependencies in UV channel impulse responses.

## Anomaly Detection

Anomalies are detected using the reconstruction error between the original and reconstructed CIR signals.

The framework uses **Normalized Mean Squared Error (NMSE)** and statistical thresholds based on:

- Three-sigma
- Percentile-based thresholding
- Interquartile Range (IQR)

## Workflow

```text
UV Channel Simulation
        ↓
Environmental Impairment Modeling
        ↓
Synthetic CIR Dataset Generation
        ↓
CAE / Transformer-based CAE
        ↓
Multi-Task Learning
        ↓
CIR Reconstruction
LOS/NLOS Classification
Delay Spread Prediction
LGR Prediction
        ↓
Reconstruction Error (NMSE)
        ↓
Statistical Thresholding
        ↓
Anomaly / No Anomaly

```

## Layout

- `src/` — channel simulation, impairments, data generation, models, training, evaluation, and inference.
- `demo/` — working Flask demos for the baseline and Transformer models.
- `saved_models/` — trained Keras models and scaler parameters.
- `results/` and `figures/` — generated validation artefacts.
- `old_code/` — unmodified research-prototype source retained for traceability.

## Quick start

Install dependencies with `pip install -r requirements.txt`, then train a baseline model:

```bash
python -m src.train --model baseline --epochs 75
```

Add `--impairment fog` (or `temperature`, `ozone`, `mixed`) to train with an environmental condition. The supplied demos retain their original assets and can be launched from their respective folders with `python app.py` or `python app2.py`.
