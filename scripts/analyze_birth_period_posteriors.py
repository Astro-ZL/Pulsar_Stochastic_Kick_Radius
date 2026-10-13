#!/usr/bin/env python3
"""Derive parameterization-independent birth-period summaries from MCMC files."""

from __future__ import annotations
import sys as _archive_sys
from pathlib import Path as _ArchivePath
_archive_root = next(p for p in _ArchivePath(__file__).resolve().parents if (p / "config.json").is_file())
_archive_sys.path.insert(0, str(_archive_root / "scripts"))
from archive_paths import ROOT, DATA, MCMC, EVIDENCE, run_dir


import csv
import pickle
from pathlib import Path

import numpy as np
from scipy.optimize import brentq
from scipy.special import gamma


FILES = [
    ("47", "LN", "fixed", "posterior_young_LN_47_9param.pkl"),
    ("47", "LN", "marg", "posterior_young_47_11param.pkl"),
    ("47", "G", "fixed", "posterior_young_47_G_9param.pkl"),
    ("47", "G", "marg", "posterior_young_47_G_11param.pkl"),
    ("47", "M", "fixed", "posterior_young_M_47_8param.pkl"),
    ("47", "M", "marg", "posterior_young_47_10param.pkl"),
    ("77", "LN", "fixed", "posterior_all_LN_77_9param.pkl"),
    ("77", "LN", "marg", "posterior_all_77_11param.pkl"),
    ("77", "G", "fixed", "posterior_all_77_G_9param.pkl"),
    ("77", "G", "marg", "posterior_all_77_G_11param.pkl"),
    ("77", "M", "fixed", "posterior_all_M_77_8param.pkl"),
    ("77", "M", "marg", "posterior_all_77_10param.pkl"),
]


def posterior_quantiles(x: np.ndarray) -> tuple[float, float, float]:
    return tuple(np.quantile(x, [0.16, 0.50, 0.84]))


def mixture_quantile(scale: np.ndarray, shape: np.ndarray, probability: float) -> float:
    """Exact quantile of an equal-weight posterior mixture of Weibulls."""

    def cdf_minus_target(x: float) -> float:
        z = np.power(x / scale, shape)
        return float(np.mean(-np.expm1(-z)) - probability)

    hi = float(np.max(scale * np.power(-np.log1p(-probability), 1.0 / shape)))
    while cdf_minus_target(hi) < 0.0:
        hi *= 2.0
    return float(brentq(cdf_minus_target, 0.0, hi, xtol=1e-12, rtol=1e-12))


def analyze(path: Path, kick: str) -> dict[str, float | int | str]:
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    samples = np.asarray(payload["samples"], dtype=float)
    columns = list(payload["columns"])
    col = {name: samples[:, i] for i, name in enumerate(columns)}

    if kick == "LN":
        # E[v_trans^2] = E[sin^2(i)] E[v_3d^2], with E[sin^2(i)] = 2/3.
        vtrans2_km2_s2 = (2.0 / 3.0) * np.exp(2.0 * col["mu_v"] + 2.0 * col["sigma_v"] ** 2)
    elif kick == "G":
        vtrans2_km2_s2 = (
            (2.0 / 3.0)
            * col["theta_kick"] ** 2
            * col["k_kick"]
            * (col["k_kick"] + 1.0)
        )
    elif kick == "M":
        vtrans2_km2_s2 = 2.0 * col["sigma_v"] ** 2
    else:
        raise ValueError(kick)

    m_sol = col.get("M_sol", np.full(samples.shape[0], 1.4))
    i_45 = col.get("I_45", np.ones(samples.shape[0]))
    m_ns = m_sol * 1.989e33
    i_ns = i_45 * 1.0e45
    pb_s = col["Pb_ms"] * 1.0e-3
    s_b = 2.0 * np.pi * i_ns / pb_s
    coupling2 = (
        col["zeta"] ** 2
        * (
            m_ns
            * (col["R_eff"] * 1.0e5)
            * col["sin_alpha"]
            / (col["sin_beta"] + 1.0e-10)
        )
        ** 2
        * (vtrans2_km2_s2 * 1.0e10)
    )
    p0_rms_ms = (2.0 * np.pi * i_ns / np.sqrt(s_b**2 + coupling2)) * 1.0e3

    shape = col["k_w"]
    scale_ms = p0_rms_ms / np.sqrt(gamma(1.0 + 2.0 / shape))
    pop_median_ms = scale_ms * np.power(np.log(2.0), 1.0 / shape)
    pop_p90_ms = scale_ms * np.power(np.log(10.0), 1.0 / shape)

    if not all(np.all(np.isfinite(x)) and np.all(x > 0.0) for x in (shape, scale_ms, p0_rms_ms)):
        raise ValueError(f"Non-finite or non-positive derived values in {path}")

    med_q = posterior_quantiles(pop_median_ms)
    p90_q = posterior_quantiles(pop_p90_ms)
    rms_q = posterior_quantiles(p0_rms_ms)
    mixture = {
        f"mixture_p{int(100 * p):02d}_ms": mixture_quantile(scale_ms, shape, p)
        for p in (0.16, 0.50, 0.84, 0.90)
    }
    mixture["mixture_rms_ms"] = float(np.sqrt(np.mean(p0_rms_ms**2)))

    return {
        "model_name": payload["model_name"],
        "filename": path.name,
        "n_samples": samples.shape[0],
        "conditional_median_p16_ms": med_q[0],
        "conditional_median_p50_ms": med_q[1],
        "conditional_median_p84_ms": med_q[2],
        "conditional_p90_p16_ms": p90_q[0],
        "conditional_p90_p50_ms": p90_q[1],
        "conditional_p90_p84_ms": p90_q[2],
        "conditional_rms_p16_ms": rms_q[0],
        "conditional_rms_p50_ms": rms_q[1],
        "conditional_rms_p84_ms": rms_q[2],
        **mixture,
    }


def main() -> None:
    input_dir = MCMC
    output_path = run_dir("summaries") / "birth_period_posterior_summary.csv"
    rows = []
    for sample, kick, structure, filename in FILES:
        row = analyze(input_dir / filename, kick)
        rows.append({"sample": sample, "kick": kick, "structure": structure, **row})

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(output_path)
    for row in rows:
        print(
            row["sample"], row["kick"], row["structure"],
            f'mix median={row["mixture_p50_ms"]:.2f} ms',
            f'mix p90={row["mixture_p90_ms"]:.2f} ms',
            f'mix rms={row["mixture_rms_ms"]:.2f} ms',
        )


if __name__ == "__main__":
    main()
