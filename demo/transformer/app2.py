# app.py — Transformer-based CIR inference server (pickle-free, TF 2.17 style)
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# ---- TF 2.17-compatible imports ----
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import register_keras_serializable
# ------------------------------------

import os, glob, werkzeug, io

app = Flask(__name__)
CORS(app)

# ---------- Config ----------
SEQ_LEN = 1000
MODEL_PATH = "no_impairement_transformer_model.keras"   # note the filename typo
DEFAULT_BATCH = "test_batch.npz"
SCALER_PARAMS_PATH = "scaler_params.npz"
DATA_DIR = "."
ALLOWED_UPLOAD_EXT = {".txt", ".npy", ".npz"}

# ---------- Define & register the custom TransformerBlock ----------
@register_keras_serializable(package="Custom", name="TransformerBlock")
class TransformerBlock(layers.Layer):
    def __init__(self, d_model, num_heads, d_ff, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.attn = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)
        self.ffn = models.Sequential([layers.Dense(d_ff, activation="relu"),
                                      layers.Dense(d_model)])
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(dropout_rate)
        self.dropout2 = layers.Dropout(dropout_rate)

    def call(self, x, training=False, mask=None):
        attn_out = self.attn(x, x, x, attention_mask=mask)
        attn_out = self.dropout1(attn_out, training=training)
        out1 = self.norm1(x + attn_out)
        ffn_out = self.ffn(out1)
        ffn_out = self.dropout2(ffn_out, training=training)
        return self.norm2(out1 + ffn_out)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            "d_model": getattr(self.attn, "_key_dim", None),
            "num_heads": getattr(self.attn, "num_heads", None),
            "d_ff": self.ffn.layers[0].units if self.ffn.layers else None,
            "dropout_rate": float(getattr(self.dropout1, "rate", 0.1)),
        })
        return cfg

# ---------- Load model (with custom_objects) ----------
model = load_model(MODEL_PATH, compile=False,
                   custom_objects={"TransformerBlock": TransformerBlock})

# ---------- Helpers ----------
def rebuild_minmax_from_minmax(feature_min, feature_max, feature_range=(-1, 1)):
    feature_min = np.asarray(feature_min)
    feature_max = np.asarray(feature_max)
    eq = (feature_max == feature_min)
    if np.any(eq):
        feature_max = feature_max.copy()
        feature_max[eq] = feature_min[eq] + 1e-9
    fake = np.vstack([feature_min, feature_max])
    sc = MinMaxScaler(feature_range=feature_range)
    sc.fit(fake)
    return sc

# SAFE key selector (replaces "a or b" on NumPy arrays)
def first_present(d, *names):
    for n in names:
        if n in d and d[n] is not None:
            return d[n]
    return None

params_npz = np.load(SCALER_PARAMS_PATH)
cir_scaler = rebuild_minmax_from_minmax(params_npz["cir_min"], params_npz["cir_max"], feature_range=(-1, 1))
lgr_scaler = rebuild_minmax_from_minmax(params_npz["lgr_min"], params_npz["lgr_max"], feature_range=(-1, 1))

UNITS = {
    "delay_spread_ns": "ns",
    "lgr_scaled": "unitless (−1…1)",
    "lgr_db": "dB",
    "p_LOS": "probability (0–1)",
    "p_LOS_percent": "%",
    "p_NLOS": "probability (0–1)",
    "p_NLOS_percent": "%"
}

# ---------- Batch management ----------
CURRENT_BATCH = DEFAULT_BATCH

def list_npz_files():
    return sorted(os.path.basename(path) for path in glob.glob(os.path.join(DATA_DIR, "*.npz")))

def _load_npz_arrays(path):
    d = np.load(path)
    if "X" not in d.files:
        raise KeyError("NPZ must contain key 'X' with shape (N, 1000, 1)")
    X = d["X"].astype(np.float32)
    if X.ndim != 3 or X.shape[1:] != (SEQ_LEN, 1):
        raise ValueError(f"'X' must have shape (N,{SEQ_LEN},1), got {X.shape}")
    t = d["t"].astype(np.float32) if "t" in d.files else None
    return X, t

X_batch, T_batch = _load_npz_arrays(DEFAULT_BATCH)

def set_current_batch(filename):
    global CURRENT_BATCH, X_batch, T_batch
    if ("/" in filename) or ("\\" in filename):
        raise ValueError("Invalid filename")
    path = os.path.join(DATA_DIR, filename)
    if not (os.path.exists(path) and path.endswith(".npz")):
        raise FileNotFoundError("File not found or not a .npz")
    X, t = _load_npz_arrays(path)
    CURRENT_BATCH = filename
    X_batch, T_batch = X, t

# ---------- Inference ----------
def _predict_outputs(x):
    preds = model.predict(x, verbose=0)
    names = getattr(model, "output_names", None)
    out = {}
    if isinstance(preds, (list, tuple)) and names and len(names) == len(preds):
        out = {n: p for n, p in zip(names, preds)}
    elif isinstance(preds, (list, tuple)):
        for arr in preds:
            shp = arr.shape
            if len(shp) == 3 and shp[-1] == 1 and shp[1] == SEQ_LEN:
                out["reconstruction"] = arr
            elif shp[-1] == 2:
                out["classification"] = arr
            elif shp[-1] == 1:
                if "delay_spread" not in out:
                    out["delay_spread"] = arr
                else:
                    out["lgr"] = arr
    else:
        raise ValueError("Unexpected model outputs")
    return out

def predict_transformer(x):
    out = _predict_outputs(x)

    # Delay
    delay = first_present(out, "delay_spread", "delay_output")
    if delay is None:
        raise ValueError("Delay output not found in model outputs.")
    delay_spread_ns = float(delay[0, 0])

    # LGR
    lgr = first_present(out, "lgr", "lgr_output")
    if lgr is None:
        raise ValueError("LGR output not found in model outputs.")
    lgr_scaled = float(lgr[0, 0])
    lgr_db = float(lgr_scaler.inverse_transform([[lgr_scaled]])[0, 0])

    # Classification (softmax 2 → [p_LOS, p_NLOS], or sigmoid 1 → p_NLOS)
    cls = first_present(out, "classification", "class_output")
    if cls is None:
        raise ValueError("Classification output not found in model outputs.")
    cls = cls[0]
    if cls.ndim == 1 and cls.shape[0] == 2:
        p_nlos = float(cls[1])
    else:
        p_nlos = float(cls.squeeze())
    p_los = 1.0 - p_nlos

    # Reconstruction
    recon = first_present(out, "reconstruction")
    if recon is None:
        raise ValueError("Reconstruction output not found.")
    recon_1d = recon[0, :, 0].astype(np.float32).tolist()

    return {
        "delay_spread_ns": delay_spread_ns,
        "lgr_scaled": lgr_scaled,
        "lgr_db": lgr_db,
        "p_LOS": p_los,
        "p_LOS_percent": p_los * 100.0,
        "p_NLOS": p_nlos,
        "p_NLOS_percent": p_nlos * 100.0,
        "reconstruction": recon_1d,
        "active_batch": CURRENT_BATCH,
        "units": UNITS
    }

def scale_if_needed(vec1d, scaled_flag=True):
    x = np.asarray(vec1d, dtype=np.float32).reshape(-1)
    if x.size != SEQ_LEN:
        raise ValueError(f"CIR length must be {SEQ_LEN}, got {x.size}")
    if not scaled_flag:
        x = cir_scaler.transform(x.reshape(1, -1)).astype(np.float32).reshape(-1)
    return x.reshape(1, SEQ_LEN, 1)

# ---------- Routes ----------
@app.route("/", methods=["GET"])
def index():
    return send_from_directory(".", "index.html")

@app.route("/batches", methods=["GET"])
def batches():
    return jsonify({"files": list_npz_files(), "active": CURRENT_BATCH})

@app.route("/set_batch", methods=["POST"])
def set_batch():
    data = request.get_json(force=True)
    fname = data.get("file")
    if not fname:
        return jsonify({"error": "Missing 'file'"}), 400
    try:
        set_current_batch(fname)
        return jsonify({"ok": True, "active": CURRENT_BATCH})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/sample", methods=["GET"])
def sample():
    i = int(request.args.get("i", 0))
    n = X_batch.shape[0]
    if i < 0: i = 0
    if i >= n: i = n - 1
    x = X_batch[i].reshape(-1).tolist()
    t_out = T_batch.tolist() if T_batch is not None else None
    return jsonify({
        "cir": x,
        "t_ns": t_out,
        "scaled": True,
        "active_batch": CURRENT_BATCH,
        "index": i,
        "count": n
    })

@app.route("/smoke", methods=["GET"])
def smoke():
    x0 = X_batch[:1]
    return jsonify(predict_transformer(x0))

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    cir = data.get("cir")
    if cir is None:
        return jsonify({"error": "Missing 'cir' array"}), 400
    scaled = bool(data.get("scaled", True))
    try:
        x = scale_if_needed(cir, scaled_flag=scaled)
        return jsonify(predict_transformer(x))
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# ---------- Optional: Windows-safe upload ----------
@app.route("/upload_predict", methods=["POST"])
def upload_predict():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400
    f = request.files["file"]
    filename = werkzeug.utils.secure_filename(f.filename or "")
    ext = os.path.splitext(filename)[1].lower()
    if ext not in ALLOWED_UPLOAD_EXT:
        return jsonify({"error": f"Unsupported file type: {ext}"}), 400

    scaled_flag = request.form.get("scaled", "true").lower() != "false"

    try:
        if ext == ".txt":
            arr = np.loadtxt(f.stream, dtype=np.float32)
            x = scale_if_needed(arr, scaled_flag=scaled_flag)
            out = predict_transformer(x); out.update({"source": "upload_txt"})
            return jsonify(out)

        if ext == ".npy":
            arr = np.load(io.BytesIO(f.read()))
            if arr.ndim != 1:
                return jsonify({"error": "For .npy, provide a 1D CIR of length 1000"}), 400
            x = scale_if_needed(arr, scaled_flag=scaled_flag)
            out = predict_transformer(x); out.update({"source": "upload_npy"})
            return jsonify(out)

        if ext == ".npz":
            file_bytes = f.read()
            with np.load(io.BytesIO(file_bytes)) as npz:
                if "X" in npz.files:
                    save_path = os.path.join(DATA_DIR, filename)
                    with open(save_path, "wb") as out_f:
                        out_f.write(file_bytes)
                    set_current_batch(filename)
                    return jsonify({"batch_loaded": True, "active": CURRENT_BATCH, "count": int(X_batch.shape[0])})
                elif "cir" in npz.files:
                    arr = npz["cir"]
                    x = scale_if_needed(arr, scaled_flag=scaled_flag)
                    out = predict_transformer(x); out.update({"source": "upload_npz_cir"})
                    return jsonify(out)
                else:
                    return jsonify({"error": "NPZ must contain 'X' (N,1000,1) or 'cir' 1D"}), 400

        return jsonify({"error": "Unhandled file type"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# ---------- Dataset stats ----------
@app.route("/dataset_stats", methods=["GET"])
def dataset_stats():
    path = os.path.join(DATA_DIR, CURRENT_BATCH)
    d = np.load(path)
    if "Y_delay_spread" in d.files:
        y = d["Y_delay_spread"].astype(np.float32).reshape(-1)
        return jsonify({
            "delay_spread_ns": y.tolist(),
            "count": int(y.size),
            "units": {"delay_spread_ns": "ns"},
            "source": "ground_truth",
            "active_batch": CURRENT_BATCH
        })

    out = _predict_outputs(X_batch)
    delay = first_present(out, "delay_spread", "delay_output")
    if delay is None:
        return jsonify({"error": "Delay output not found in model outputs."}), 500
    delay_pred = delay.reshape(-1).astype(np.float32)
    return jsonify({
        "delay_spread_ns": delay_pred.tolist(),
        "count": int(delay_pred.size),
        "units": {"delay_spread_ns": "ns"},
        "source": "predicted",
        "active_batch": CURRENT_BATCH
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)

