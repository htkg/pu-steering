"""Fixed sampler configs for the three Ours variants (same values as run_score.py)."""

from kernel_model import KernelPULogistic

ESS_TARGET = 0.6

MODE_KW = {
    "OFF": {"grad_clip": 0.5, "tilt_max": 3.0, "tilt_start": 0.6},
    "ON": {"proposal": "dng", "dng_lambda0": 0.6, "dng_tau": 1.0, "dng_lam_max": 4.0, "lam_fixed": 2.0, "tilt_max": 0.0, "resample_at": "40"},
    "ON_TILT": {"proposal": "dng", "dng_lambda0": 0.6, "dng_tau": 1.0, "dng_lam_max": 4.0, "wq_ratio": False, "tilt_max": 3.0, "resample_at": "40"},
    "ON_GRAD40": {
        "grad_clip": 0.5,
        "lam_mode": "cos",
        "lam_end": 2.0,
        "lam_ramp": 0.5,
        "tilt_max": 0.0,
        "ess_resample": 0.9,
        "resample_stop": 40,
        "resample_cooldown": 40,
    },
    "ON_GRAD48": {
        "grad_clip": 0.5,
        "lam_mode": "cos",
        "lam_end": 2.0,
        "lam_ramp": 0.5,
        "tilt_max": 0.0,
        "ess_resample": 0.9,
        "resample_stop": 38,
        "resample_cooldown": 48,
    },
}

MODE_DESC = {
    "OFF": "grad proposal, no resample, w_q = IS ratio x ESS-adaptive tilt (tilt_max=3, start=0.6) — weighted metrics",
    "ON": "DNG(l0=0.6) proposal, w_q = IS ratio x tilt (lam_fixed=2.0), resample at t=40 — raw metrics",
    "ON_TILT": "DNG(l0=0.6) proposal, w_q = tilt only (ESS-adaptive, tilt_max=3), resample at t=40 — raw metrics",
    "ON_GRAD40": "grad proposal, w_q = IS ratio x tilt (cos 0->2, plateau 0.5T), ESS-fired resample (er=0.9, cd=40, stop=40) — raw metrics",
    "ON_GRAD48": "grad proposal, w_q = IS ratio x tilt (cos 0->2, plateau 0.5T), ESS-fired resample (er=0.9, cd=48, stop=38) — raw metrics",
}


def make_pu():
    return KernelPULogistic(sigma=None, sigma_scale=0.2, lam=1e-4, n_centers_pos=128, n_centers_u=64, max_iter=100)
