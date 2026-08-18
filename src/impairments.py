"""Environmental impairments for simulated UV channel responses."""

import numpy as np


def apply_impairment(cir, distance_m, impairment, rng):
    """Apply stochastic amplitude extinction for a named environment."""
    distance_km = distance_m / 1000.0
    if impairment == "temperature":
        extinction = 0.02 * abs(rng.uniform(-10, 45) - 20)
    elif impairment == "fog":
        extinction = 3.912 / rng.uniform(0.2, 20.0)
    elif impairment == "ozone":
        extinction = rng.uniform(0.0, 0.5)
    elif impairment == "mixed":
        extinction = (3.912 / rng.uniform(0.2, 10.0) + rng.uniform(0.1, 0.5)
                      + 0.02 * abs(rng.uniform(-10, 45) - 20)
                      + 0.01 * rng.uniform(20, 100) / 100)
    elif impairment in (None, "none"):
        return cir.copy()
    else:
        raise ValueError("impairment must be temperature, fog, ozone, mixed, or none")
    return cir * np.exp(-0.5 * extinction * distance_km)
