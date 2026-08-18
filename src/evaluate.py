"""Evaluation utilities for multi-task anomaly-detection models."""

from pathlib import Path
import json
import numpy as np
from sklearn.metrics import accuracy_score, mean_absolute_error

global_results = {}


def evaluate_model(model, x, y_delay, y_lgr, y_class, y_reconstruction, output_path=None):
    """Compute concise regression, classification, and reconstruction metrics."""
    pred_delay, pred_lgr, pred_class, pred_reconstruction = model.predict(x, verbose=0)
    metrics = {
        "delay_mae": float(mean_absolute_error(y_delay, pred_delay)),
        "lgr_mae": float(mean_absolute_error(y_lgr, pred_lgr)),
        "classification_accuracy": float(accuracy_score(y_class.ravel(), (pred_class.ravel() >= .5).astype(int))),
        "reconstruction_mse": float(np.mean((y_reconstruction - pred_reconstruction) ** 2)),
    }
    if output_path:
        path = Path(output_path); path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def evaluate_and_log(model, model_name, dataset_name, X_val, Ycls_val, Yds_val,
                     Ylgr_val, Yrec_val, lgr_scaler, save_dir="results", save_json=True):
    """Notebook-compatible evaluation with JSON metrics and diagnostic plots."""
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_score, recall_score, f1_score, r2_score, ConfusionMatrixDisplay, confusion_matrix

    output_dir = Path(save_dir) / model_name / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = dict(zip(model.output_names, model.predict(X_val, verbose=0)))
    predicted_class = (outputs["class_output"].ravel() >= .5).astype(int)
    true_class = Ycls_val.ravel().astype(int)
    true_lgr = lgr_scaler.inverse_transform(Ylgr_val).ravel()
    predicted_lgr = lgr_scaler.inverse_transform(outputs["lgr_output"]).ravel()
    true_delay, predicted_delay = Yds_val.ravel(), outputs["delay_output"].ravel()
    reconstruction = outputs["reconstruction"]
    nmse = np.sum((X_val - reconstruction) ** 2, axis=(1, 2)) / np.maximum(np.sum(X_val ** 2, axis=(1, 2)), 1e-30)
    metrics = {"classification": {"accuracy": float(accuracy_score(true_class, predicted_class)),
        "precision": float(precision_score(true_class, predicted_class, zero_division=0)),
        "recall": float(recall_score(true_class, predicted_class, zero_division=0)),
        "f1": float(f1_score(true_class, predicted_class, zero_division=0))},
        "delay": {"r2": float(r2_score(true_delay, predicted_delay))},
        "lgr": {"mse": float(np.mean((true_lgr - predicted_lgr) ** 2))},
        "reconstruction": {"avg_mse": float(np.mean((X_val - reconstruction) ** 2)), "avg_nmse": float(np.mean(nmse))}}
    ConfusionMatrixDisplay(confusion_matrix(true_class, predicted_class), display_labels=["LOS", "NLOS"]).plot(cmap="Blues")
    plt.savefig(output_dir / "confusion_matrix.png", dpi=300, bbox_inches="tight"); plt.close()
    for actual, predicted, title, filename in ((true_delay, predicted_delay, "Delay Spread", "delay_spread.png"),
                                                 (true_lgr, predicted_lgr, "LGR (dB)", "lgr.png")):
        plt.scatter(actual, predicted, alpha=.6); bounds = [min(actual.min(), predicted.min()), max(actual.max(), predicted.max())]
        plt.plot(bounds, bounds, "r--"); plt.xlabel(f"True {title}"); plt.ylabel(f"Predicted {title}")
        plt.savefig(output_dir / filename, dpi=300, bbox_inches="tight"); plt.close()
    global_results.setdefault(model_name, {})[dataset_name] = metrics
    if save_json:
        (Path(save_dir) / "evaluation_results.json").write_text(json.dumps(global_results, indent=2), encoding="utf-8")
    return global_results


def plot_best_reconstruction(X_val, pred_reconstruction, save_path):
    """Save the notebook's best (lowest-NMSE) CIR reconstruction plot."""
    import matplotlib.pyplot as plt
    nmse = np.sum((X_val - pred_reconstruction) ** 2, axis=(1, 2)) / np.maximum(
        np.sum(X_val ** 2, axis=(1, 2)), 1e-30)
    index = int(np.argmin(nmse))
    actual, predicted = X_val[index, :, 0], pred_reconstruction[index, :, 0]
    plt.figure(figsize=(12, 4))
    plt.plot(actual, label="True CIR", linewidth=2)
    plt.plot(predicted, "--", label="Reconstruction", linewidth=2)
    plt.xlim(0, len(actual) - 1); plt.xlabel("Time Index / Tap"); plt.ylabel("Amplitude")
    plt.grid(True, linestyle="--", alpha=.7); plt.legend(); plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight"); plt.close()
    return index
