# app.py — Baseline CAE inference server (pickle-free, uploads + histogram)

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import os, glob, werkzeug

app = Flask(__name__)
CORS(app)

# ---------- Config ----------
SEQ_LEN = 1000
MODEL_PATH = "no_impairment_base_model.keras"
DEFAULT_BATCH = "test_batch.npz"
SCALER_PARAMS_PATH = "scaler_params.npz"
DATA_DIR = "."  # folder to list/save .npz

ALLOWED_UPLOAD_EXT = {".txt", ".npy", ".npz"}

# ---------- Load model ----------
model = load_model(MODEL_PATH, compile=False)

# ---------- Helpers: rebuild MinMaxScaler from saved min/max ----------
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

# ---------- Load scalers ----------
p = np.load(SCALER_PARAMS_PATH)
cir_scaler = rebuild_minmax_from_minmax(p["cir_min"], p["cir_max"], feature_range=(-1, 1))
lgr_scaler = rebuild_minmax_from_minmax(p["lgr_min"], p["lgr_max"], feature_range=(-1, 1))

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
    return sorted(os.path.basename(p) for p in glob.glob(os.path.join(DATA_DIR, "*.npz")))

def _load_npz_arrays(path):
    d = np.load(path)
    if "X" not in d.files:
        raise KeyError("NPZ must contain key 'X' with shape (N, 1000, 1)")
    X = d["X"].astype(np.float32)
    if X.ndim != 3 or X.shape[1:] != (SEQ_LEN, 1):
        raise ValueError(f"'X' must have shape (N,{SEQ_LEN},1), got {X.shape}")
    t = d["t"].astype(np.float32) if "t" in d.files else None
    return X, t

# Load default batch
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
def predict_cae(x):
    # x: (1, 1000, 1), scaled
    lgr, delay, cls_sig, recon = model.predict(x, verbose=0)
    lgr_scaled = float(lgr[0, 0])
    lgr_db = float(lgr_scaler.inverse_transform([[lgr_scaled]])[0, 0])
    delay_spread_ns = float(delay[0, 0])
    p_nlos = float(cls_sig[0, 0])
    p_los = 1.0 - p_nlos
    return {
        "delay_spread_ns": delay_spread_ns,
        "lgr_scaled": lgr_scaled,
        "lgr_db": lgr_db,
        "p_LOS": p_los,
        "p_LOS_percent": p_los * 100.0,
        "p_NLOS": p_nlos,
        "p_NLOS_percent": p_nlos * 100.0,
        "reconstruction": recon[0, :, 0].astype(np.float32).tolist(),
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
    return jsonify(predict_cae(x0))

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    cir = data.get("cir")
    if cir is None:
        return jsonify({"error": "Missing 'cir' array"}), 400
    scaled = bool(data.get("scaled", True))
    try:
        x = scale_if_needed(cir, scaled_flag=scaled)
        return jsonify(predict_cae(x))
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# ---------- NEW: Upload & Predict ----------
@app.route("/upload_predict", methods=["POST"])
def upload_predict():
    """
    Accepts multipart/form-data with:
      file: .txt | .npy | .npz
      scaled: "true"/"false" (optional, default true)
    Behaviors:
      - .txt/.npy with a single 1D CIR (len==1000): predict immediately
      - .npz with 'X' array (N,1000,1): save file, set as active batch, return meta
      - .npz with single 'cir' 1D: predict immediately
    """
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400
    f = request.files["file"]
    filename = werkzeug.utils.secure_filename(f.filename)
    ext = os.path.splitext(filename)[1].lower()
    if ext not in ALLOWED_UPLOAD_EXT:
        return jsonify({"error": f"Unsupported file type: {ext}"}), 400

    scaled_flag = request.form.get("scaled", "true").lower() != "false"

    # Save to a temp path first
    tmp_path = os.path.join(DATA_DIR, f"__upload_tmp{ext}")
    f.save(tmp_path)

    try:
        if ext == ".txt":
            arr = np.loadtxt(tmp_path, dtype=np.float32)
            x = scale_if_needed(arr, scaled_flag=scaled_flag)
            out = predict_cae(x)
            out.update({"source": "upload_txt"})
            return jsonify(out)

        elif ext == ".npy":
            arr = np.load(tmp_path)
            if arr.ndim == 1:
                x = scale_if_needed(arr, scaled_flag=scaled_flag)
                out = predict_cae(x)
                out.update({"source": "upload_npy"})
                return jsonify(out)
            else:
                return jsonify({"error": "For .npy, provide a 1D CIR of length 1000"}), 400

        elif ext == ".npz":
            data = np.load(tmp_path)
            if "X" in data.files:
                # treat as dataset; save permanently with original name
                save_path = os.path.join(DATA_DIR, filename)
                os.replace(tmp_path, save_path)  # move tmp -> final
                set_current_batch(filename)
                return jsonify({
                    "batch_loaded": True,
                    "active": CURRENT_BATCH,
                    "count": int(X_batch.shape[0]),
                    "message": f"Dataset '{filename}' loaded as active batch"
                })
            elif "cir" in data.files:
                arr = data["cir"]
                x = scale_if_needed(arr, scaled_flag=scaled_flag)
                out = predict_cae(x)
                out.update({"source": "upload_npz_cir"})
                return jsonify(out)
            else:
                return jsonify({"error": "NPZ must contain 'X' (N,1000,1) or a single 'cir' 1D array"}), 400

    finally:
        # Clean up tmp if it still exists
        if os.path.exists(tmp_path) and not tmp_path.endswith(filename):
            try:
                os.remove(tmp_path)
            except Exception:
                pass

# ---------- NEW: dataset delay-spread stats ----------
@app.route("/dataset_stats", methods=["GET"])
def dataset_stats():
    """
    Returns delay_spread_ns for all samples in the active batch.
    If 'Y_delay_spread' present in NPZ -> returns ground truth.
    Else -> uses model predictions.
    """
    # Try GT first
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
    # Predict all (may be small batch)
    delay_pred = model.predict(X_batch, verbose=0)[1].reshape(-1).astype(np.float32)
    return jsonify({
        "delay_spread_ns": delay_pred.tolist(),
        "count": int(delay_pred.size),
        "units": {"delay_spread_ns": "ns"},
        "source": "predicted",
        "active_batch": CURRENT_BATCH
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

