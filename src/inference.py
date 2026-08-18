"""Loading and prediction helpers for saved UV anomaly models."""

import numpy as np
from tensorflow.keras.models import load_model

from .model import TransformerBlock


def load_uv_model(path):
    """Load either project model architecture, including Transformer layers."""
    return load_model(path, custom_objects={"TransformerBlock": TransformerBlock})


def predict(model, cir_batch):
    """Return named outputs for a (batch, samples, 1) CIR batch."""
    cir_batch = np.asarray(cir_batch)
    if cir_batch.ndim == 2:
        cir_batch = cir_batch[..., np.newaxis]
    delay, lgr, anomaly, reconstruction = model.predict(cir_batch, verbose=0)
    return {"delay_spread": delay, "lgr": lgr, "anomaly_probability": anomaly,
            "reconstruction": reconstruction}
