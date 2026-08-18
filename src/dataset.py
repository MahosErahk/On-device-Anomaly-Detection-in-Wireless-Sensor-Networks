"""Synthetic datasets for UV LOS/NLOS anomaly detection."""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from .channel import calculate_channel_metrics, channel_impulse_response, simple_channel_impulse_response
from .impairments import apply_impairment


def _single_cir(rng, los):
    t = np.linspace(0, 1_000e-9, 1_000)
    distance = rng.uniform(30, 70) if los else rng.uniform(700, 1_200)
    rays = max(1, rng.poisson(4 if los else 15))
    max_order = 8
    orders = np.clip(rng.geometric(0.6 if los else 0.4, rays), 1, max_order)
    r_beta = np.ones((rays, max_order)); area = np.ones((rays, max_order))
    d_kr = np.zeros((rays, max_order)); d_j1j = np.zeros((rays, max_order))
    for ray, order in enumerate(orders):
        r_beta[ray, :order] = rng.uniform(0.3, 0.7, order)
        area[ray, :order] = rng.uniform(0.3, 0.7, order)
        d_kr[ray, :order] = rng.exponential(5 if los else 15, order) + 1
        d_j1j[ray, :order] = rng.exponential(3 if los else 10, order)
    strength = 0.9 if los else 0.05
    total, h_los, h_nlos = channel_impulse_response(
        t, 3.0, strength, strength, distance, 3e8, rays, r_beta, area, d_kr, d_j1j)
    lgr, delay, _ = calculate_channel_metrics(total, h_los, h_nlos, t)
    return total / max(np.max(np.abs(total)), 1e-12), delay, lgr, distance


def generate_dataset(impairment=None, num_los=500, num_nlos=500, seed=42, test_size=0.2):
    """Generate, scale, and split a labelled multi-task CIR dataset.

    Returns a dictionary with train/validation arrays and fitted scalers.
    """
    rng = np.random.default_rng(seed)
    rows, delays, lgrs, labels = [], [], [], []
    for los, count in ((True, num_los), (False, num_nlos)):
        for _ in range(count):
            cir, delay, lgr, distance = _single_cir(rng, los)
            rows.append(apply_impairment(cir, distance, impairment, rng))
            delays.append(delay); lgrs.append(lgr); labels.append(int(not los))
    cir_scaler = MinMaxScaler(feature_range=(-1, 1))
    lgr_scaler = MinMaxScaler(feature_range=(-1, 1))
    x = cir_scaler.fit_transform(np.asarray(rows))[..., np.newaxis]
    y_lgr = lgr_scaler.fit_transform(np.asarray(lgrs).reshape(-1, 1))
    split = train_test_split(x, x.copy(), np.asarray(delays).reshape(-1, 1), y_lgr,
                             np.asarray(labels).reshape(-1, 1), test_size=test_size,
                             random_state=seed, stratify=labels)
    keys = ("x_train", "x_val", "yrec_train", "yrec_val", "yds_train", "yds_val",
            "ylgr_train", "ylgr_val", "ycls_train", "ycls_val")
    data = dict(zip(keys, split))
    data.update(cir_scaler=cir_scaler, lgr_scaler=lgr_scaler)
    return data


def generate_dataset_imp(dataset_type="temperature", num_los=500, num_nlos=500,
                         seed=42, use_impairments=True):
    """Callable version of the Colab impairment-dataset generator.

    The tuple order matches the original notebook so existing experiments can be
    moved over without changing their unpacking code.
    """
    t = np.linspace(0, 5e-6, 5000)
    t = t[:np.argmin(np.abs(t - 1_000e-9))]
    rng = np.random.default_rng(seed)
    cir_data = np.zeros((num_los + num_nlos, len(t)))
    delays, lgr_values, labels = (np.zeros(num_los + num_nlos) for _ in range(3))

    def sample(los):
        distance = rng.uniform(30, 70) if los else 1000.0
        rays = max(1, rng.poisson(1 if los else 50))
        order = 1 if los else max(1, rng.poisson(8))
        r_betak = rng.uniform(.3, .7, (rays, order))
        area = rng.uniform(.3, .7, (rays, order))
        d_kr = rng.exponential(5 if los else 15, (rays, order)) + 1
        d_j1j = rng.exponential(5 if los else 10, (rays, order))
        cir, h_los, h_nlos = simple_channel_impulse_response(
            t, 3., .9 if los else .01, .9 if los else .01, distance, 3e8,
            rays, order, r_betak, area, d_kr, d_j1j, nlos_scale=10 if los else 100)
        if use_impairments:
            cir = apply_impairment(cir, distance, dataset_type, rng)
        cir /= max(np.max(np.abs(cir)), 1e-12)
        lgr, delay, los_dominant = calculate_channel_metrics(cir, h_los, h_nlos, t)
        return cir, delay, lgr, los_dominant

    # The notebook accepts only samples whose LGR agrees with their intended class.
    for target_los, start, count in ((True, 0, num_los), (False, num_los, num_nlos)):
        accepted = 0
        while accepted < count:
            cir, delay, lgr, los_dominant = sample(target_los)
            if los_dominant != target_los:
                continue
            index = start + accepted
            cir_data[index], delays[index], lgr_values[index], labels[index] = cir, delay, lgr, int(not target_los)
            accepted += 1
    cir_scaler = MinMaxScaler(feature_range=(-1, 1))
    lgr_scaler = MinMaxScaler(feature_range=(-1, 1))
    x = cir_scaler.fit_transform(cir_data)[..., np.newaxis]
    y_lgr = lgr_scaler.fit_transform(lgr_values.reshape(-1, 1))
    split = train_test_split(x, x.copy(), delays.reshape(-1, 1), y_lgr, labels.astype(int),
        test_size=.2, random_state=42, stratify=labels)
    return (*split, lgr_scaler, cir_scaler)
