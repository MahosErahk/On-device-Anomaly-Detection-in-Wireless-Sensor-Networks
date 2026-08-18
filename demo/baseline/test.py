import os, joblib, numpy as np

for fn in ["cir_scaler.pkl", "lgr_scaler.pkl", "test_batch.npz"]:
    print(fn, os.path.exists(fn), os.path.getsize(fn) if os.path.exists(fn) else 0)

# sanity: test_batch should be a valid npz
np.load("test_batch.npz")

# sanity: try loading pickles directly
joblib.load("cir_scaler.pkl")
joblib.load("lgr_scaler.pkl")
