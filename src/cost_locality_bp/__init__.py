"""Cost locality versus noise-induced barren plateaus: gradient statistics and shot cost."""

from .circuit import depth_for, cz_sign_vectors
from .costs import global_cost, local_cost, shot_variances
from .noisy_sim import simulate_probs

__all__ = [
    "depth_for",
    "cz_sign_vectors",
    "global_cost",
    "local_cost",
    "shot_variances",
    "simulate_probs",
]
__version__ = "1.0.0"
