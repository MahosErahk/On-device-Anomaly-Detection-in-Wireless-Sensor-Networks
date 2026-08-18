"""UV optical channel impulse-response simulation."""

import numpy as np


def channel_impulse_response(t, v_d, r_beta, a_eff_zeta, d_los, c,
                             n_rays, r_betak, a_eff_zetak, d_kr, d_j1j,
                             alpha=0.02, sigma_t_ns=2.0):
    """Return total, LOS, and NLOS Gaussian-pulse channel responses."""
    h_los = np.zeros_like(t, dtype=float)
    h_nlos = np.zeros_like(t, dtype=float)
    sigma_t = sigma_t_ns * 1e-9

    def add_pulse(target, delay, amplitude):
        if sigma_t <= 0:
            target[np.argmin(np.abs(t - delay))] += amplitude
            return
        pulse = np.exp(-0.5 * ((t - delay) / sigma_t) ** 2)
        target += amplitude * pulse / (sigma_t * np.sqrt(2 * np.pi))

    add_pulse(h_los, d_los / c,
              (v_d * r_beta * a_eff_zeta / d_los ** 2) * np.exp(-alpha * d_los))
    for ray in range(n_rays):
        valid = d_kr[ray] > 0
        if not np.any(valid):
            continue
        length = np.sum(d_j1j[ray, valid]) + d_kr[ray, valid][-1]
        scatter_loss = np.prod(r_betak[ray, valid] * a_eff_zetak[ray, valid])
        add_pulse(h_nlos, length / c, scatter_loss / length ** 2 * np.exp(-alpha * length))
    return h_los + h_nlos, h_los, h_nlos


def calculate_channel_metrics(h_total, h_los, h_nlos, t):
    """Return compressed LGR (dB), RMS delay spread (ns), and LOS status."""
    eps = 1e-30
    lgr_raw = 10 * np.log10(max(np.sum(h_los ** 2), eps) / max(np.sum(h_nlos ** 2), eps))
    lgr_compressed = 40 * np.tanh(lgr_raw / 40)
    pdp = h_total ** 2
    if not np.any(pdp):
        return lgr_compressed, 0.0, False
    t_ns = t * 1e9
    mean_delay = np.sum(pdp * t_ns) / np.sum(pdp)
    rms_delay = np.sqrt(np.sum(pdp * (t_ns - mean_delay) ** 2) / np.sum(pdp))
    return lgr_compressed, rms_delay, lgr_raw > 10


def simple_channel_impulse_response(t, v_d, r_beta, a_eff_zeta, d_los, c,
                                    n_rays, scatter_order, r_betak, a_eff_zetak,
                                    d_kr, d_j1j, alpha=0.02, nlos_scale=100):
    """Notebook-compatible spike-based CIR used for impairment experiments."""
    h_los = np.zeros_like(t, dtype=float)
    h_nlos = np.zeros_like(t, dtype=float)
    h_los[np.argmin(np.abs(t - d_los / c))] = (
        v_d * r_beta * a_eff_zeta / d_los ** 2 * np.exp(-alpha * d_los))
    for ray in range(n_rays):
        length = np.sum(d_j1j[ray, :scatter_order]) + d_kr[ray, scatter_order - 1]
        index = np.argmin(np.abs(t - length / c))
        loss = np.prod(r_betak[ray, :scatter_order] * a_eff_zetak[ray, :scatter_order])
        h_nlos[index] += nlos_scale * loss / length ** 2 * np.exp(-alpha * length)
    return h_los + h_nlos, h_los, h_nlos
