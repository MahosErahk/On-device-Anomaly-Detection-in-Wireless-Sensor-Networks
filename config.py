"""Project-wide defaults for UV channel anomaly detection."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SEED = 42
SAMPLE_RATE_POINTS = 1_000
TIME_WINDOW_SECONDS = 1_000e-9
SPEED_OF_LIGHT = 3e8
SCALER_RANGE = (-1, 1)

SRC_DIR = PROJECT_ROOT / "src"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"
MODELS_DIR = PROJECT_ROOT / "saved_models"
