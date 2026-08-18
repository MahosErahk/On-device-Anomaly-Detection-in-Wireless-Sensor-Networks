"""Command-line model training entry point."""

import argparse
from pathlib import Path

from .dataset import generate_dataset
from .model import build_baseline_model, build_transformer_model, compile_model
from .utils import save_scaler_params


def train(model_type="baseline", impairment=None, epochs=75, batch_size=128, output_dir="saved_models"):
    data = generate_dataset(impairment=impairment)
    builder = build_transformer_model if model_type == "transformer" else build_baseline_model
    model = compile_model(builder(seq_len=data["x_train"].shape[1]))
    history = model.fit(data["x_train"], {"delay_output": data["yds_train"], "lgr_output": data["ylgr_train"],
        "class_output": data["ycls_train"].astype("float32"), "reconstruction": data["yrec_train"]},
        validation_data=(data["x_val"], {"delay_output": data["yds_val"], "lgr_output": data["ylgr_val"],
        "class_output": data["ycls_val"].astype("float32"), "reconstruction": data["yrec_val"]}),
        epochs=epochs, batch_size=batch_size)
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    stem = f"{model_type}_{impairment or 'no_impairment'}"
    model.save(output / f"{stem}.keras")
    save_scaler_params(output / f"{stem}_scaler.npz", data["cir_scaler"])
    return model, history, data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("baseline", "transformer"), default="baseline")
    parser.add_argument("--impairment", choices=("temperature", "fog", "ozone", "mixed"))
    parser.add_argument("--epochs", type=int, default=75)
    args = parser.parse_args()
    train(args.model, args.impairment, args.epochs)
