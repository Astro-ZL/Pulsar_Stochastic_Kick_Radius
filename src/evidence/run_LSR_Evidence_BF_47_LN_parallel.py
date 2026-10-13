import sys as _archive_sys
from pathlib import Path as _ArchivePath
_archive_root = next(p for p in _ArchivePath(__file__).resolve().parents if (p / "config.json").is_file())
_archive_sys.path.insert(0, str(_archive_root / "scripts"))
from archive_paths import ROOT, DATA, MCMC, EVIDENCE, run_dir

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

# 避免 500 个 Python 进程里每个进程又调用多线程 BLAS，造成过度并行
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pathlib import Path
from scipy.stats import gaussian_kde
from scipy.special import logsumexp, gamma, expit
import dynesty
from dynesty import NestedSampler
from dynesty.utils import resample_equal
import pickle
import time
import argparse
import multiprocessing as mp


# ============================================================
# 0. Global settings
# ============================================================

BASE_PATH = DATA
OUTDIR = run_dir("evidence/lognormal_fixed")
OUTDIR.mkdir(parents=True, exist_ok=True)

# synthetic population size
N_SIM = 20000

# observational velocity-cloud samples per pulsar
N_OBS_SAMPLES = 100

# dynesty settings
NLIVE_FULL = 1000
NLIVE_ZERO = 800
DLOGZ = 0.05

# random seeds
RANDOM_SEED_SIM = 12345
RANDOM_SEED_DYNASTY_FULL = 202601
RANDOM_SEED_DYNASTY_ZERO = 202602


# ============================================================
# 1. Load observed data: LSR-corrected velocity clouds
# ============================================================

def load_obs_data():
    clean_periods = np.loadtxt(BASE_PATH / "period_2_new_47.txt")
    obs_velocity_samples = np.loadtxt(BASE_PATH / "vpec_lsr_samples_matrix_47x100.txt")

    n_obs = len(clean_periods)

    if obs_velocity_samples.shape != (n_obs, N_OBS_SAMPLES):
        raise ValueError(
            f"Expected obs_velocity_samples shape {(n_obs, N_OBS_SAMPLES)}, "
            f"but got {obs_velocity_samples.shape}."
        )

    if not np.all(np.isfinite(obs_velocity_samples)):
        raise ValueError("obs_velocity_samples contains NaN or inf.")

    if np.any(obs_velocity_samples <= 0):
        raise ValueError("obs_velocity_samples contains non-positive velocities.")

    obs_logV_cloud = np.log10(obs_velocity_samples * 1e5)
    obs_logP_cloud = np.tile(
        np.log10(clean_periods)[:, None],
        (1, N_OBS_SAMPLES),
    )

    return obs_logP_cloud, obs_logV_cloud, clean_periods, obs_velocity_samples


OBS_LOGP_CLOUD, OBS_LOGV_CLOUD, CLEAN_PERIODS, OBS_VSAMPLES = load_obs_data()
OBS_DATA = (OBS_LOGP_CLOUD, OBS_LOGV_CLOUD)


# ============================================================
# 2. Common random numbers: deterministic likelihood
# ============================================================

def make_common_random_numbers(n_sim, seed=12345):
    rng = np.random.default_rng(seed)

    lognorm_z = rng.normal(size=n_sim)
    u_los = rng.uniform(-1.0, 1.0, n_sim)
    sin_proj = np.sqrt(1.0 - u_los**2)

    u_weib = rng.uniform(1e-12, 1.0 - 1e-12, n_sim)
    weibull_base = -np.log1p(-u_weib)

    u_detect = rng.uniform(0.0, 1.0, n_sim)

    return {
        "lognorm_z": lognorm_z,
        "sin_proj": sin_proj,
        "weibull_base": weibull_base,
        "u_detect": u_detect,
    }


COMMON = make_common_random_numbers(N_SIM, RANDOM_SEED_SIM)


# ============================================================
# 3. Population synthesis
# ============================================================

def simulate_population(theta, model="full"):
    """
    full model theta:
        [sigma_v, R_eff_km, zeta, sin_alpha, sin_beta, Pb_ms, k_w, f_evo]

    zero model theta:
        [sigma_v, Pb_ms, k_w, f_evo]
    """

    if model == "full":
        mu_log, sigma_log, R_eff_km, zeta, sin_alpha, sin_beta, Pb_ms, k_w, f_evo = theta

    elif model == "zero":
        mu_log, sigma_log, Pb_ms, k_w, f_evo = theta
        R_eff_km = 0.0
        zeta = 0.0
        sin_alpha = 0.5
        sin_beta = 0.5

    else:
        raise ValueError("model must be 'full' or 'zero'.")

    M_ns = 1.4 * 1.989e33
    I_ns = 1.0e45

    P_b_s = Pb_ms * 1e-3
    S_b_star = (2.0 * np.pi * I_ns) / P_b_s

    v_3d_kms = np.exp(mu_log + sigma_log * COMMON["lognorm_z"])
    v_3d_cgs = v_3d_kms * 1e5
    v_trans = v_3d_cgs * COMMON["sin_proj"]

    if model == "full":
        R_cgs = R_eff_km * 1e5
        delta_S_dist = (M_ns * R_cgs * sin_alpha * v_trans) / (sin_beta + 1e-10)
        delta_S_sq_mean = (zeta**2) * np.mean(delta_S_dist**2)
    else:
        delta_S_sq_mean = 0.0

    S0_rms_val = np.sqrt(S_b_star**2 + delta_S_sq_mean)
    P0_rms = (2.0 * np.pi * I_ns) / S0_rms_val

    gamma_factor = np.sqrt(gamma(1.0 + 2.0 / k_w))
    lambda_w = P0_rms / gamma_factor

    P_birth = lambda_w * COMMON["weibull_base"]**(1.0 / k_w)
    P_obs = P_birth * f_evo

    p_detect = expit((P_obs - 0.1) / 0.01)
    mask = (COMMON["u_detect"] < p_detect) & (P_obs < 100.0) & (v_trans > 0.0)

    if np.count_nonzero(mask) < 50:
        return np.array([]), np.array([])

    return np.log10(P_obs[mask]), np.log10(v_trans[mask])


# ============================================================
# 4. Prior transforms
# ============================================================

def prior_transform_full(u):
    mu_log = 4.0 + u[0] * (7.5 - 4.0)
    sigma_log = 0.1 + u[1] * (2.0 - 0.1)
    R_eff = 0.1 + u[2] * (50.0 - 0.1)
    zeta = u[3]
    sin_alpha = 0.01 + u[4] * (1.0 - 0.01)
    sin_beta = 0.01 + u[5] * (1.0 - 0.01)
    Pb_ms = 10.0 + u[6] * (5000.0 - 10.0)
    k_w = 0.1 + u[7] * (5.0 - 0.1)
    f_evo = 1.0 + u[8] * (50.0 - 1.0)

    return np.array([
        mu_log,
        sigma_log,
        R_eff,
        zeta,
        sin_alpha,
        sin_beta,
        Pb_ms,
        k_w,
        f_evo,
    ])


def prior_transform_zero(u):
    mu_log = 4.0 + u[0] * (7.5 - 4.0)
    sigma_log = 0.1 + u[1] * (2.0 - 0.1)
    Pb_ms = 10.0 + u[2] * (5000.0 - 10.0)
    k_w = 0.1 + u[3] * (5.0 - 0.1)
    f_evo = 1.0 + u[4] * (50.0 - 1.0)

    return np.array([
        mu_log,
        sigma_log,
        Pb_ms,
        k_w,
        f_evo,
    ])


# ============================================================
# 5. KDE likelihood
# ============================================================

def log_likelihood(theta, model="full"):
    try:
        sim_logP, sim_logV = simulate_population(theta, model=model)

        if len(sim_logP) < 50:
            return -np.inf

        values = np.vstack([sim_logP, sim_logV])

        if not np.all(np.isfinite(values)):
            return -np.inf

        kernel = gaussian_kde(values, bw_method="scott")

        obs_P_mat, obs_V_mat = OBS_DATA
        n_obs, n_samples = obs_P_mat.shape

        check_points = np.vstack([
            obs_P_mat.ravel(),
            obs_V_mat.ravel(),
        ])

        log_pdf_all = kernel.logpdf(check_points)
        log_pdf_all = np.maximum(log_pdf_all, -100.0)

        log_pdf_mat = log_pdf_all.reshape(n_obs, n_samples)
        log_likelihood_per_star = logsumexp(log_pdf_mat, axis=1) - np.log(n_samples)

        total = np.sum(log_likelihood_per_star)

        if not np.isfinite(total):
            return -np.inf

        return total

    except Exception:
        return -np.inf


def log_likelihood_full(theta):
    return log_likelihood(theta, model="full")


def log_likelihood_zero(theta):
    return log_likelihood(theta, model="zero")


# ============================================================
# 6. dynesty runner with multiprocessing
# ============================================================

def run_dynesty_model(model="full", ncores=500):
    if model == "full":
        ndim = 9
        loglike = log_likelihood_full
        ptform = prior_transform_full
        nlive = NLIVE_FULL
        seed = RANDOM_SEED_DYNASTY_FULL
        label = "full"
        columns = [
            "mu_log",
            "sigma_log",
            "R_eff",
            "zeta",
            "sin_alpha",
            "sin_beta",
            "Pb_ms",
            "k_w",
            "f_evo",
        ]


    elif model == "zero":
        ndim = 5
        loglike = log_likelihood_zero
        ptform = prior_transform_zero
        nlive = NLIVE_ZERO
        seed = RANDOM_SEED_DYNASTY_ZERO
        label = "zero"
        columns = [
            "mu_log",
            "sigma_log",
            "Pb_ms",
            "k_w",
            "f_evo",
        ]

    else:
        raise ValueError("model must be 'full' or 'zero'.")

    print(f"\n>>> Running dynesty: {label} model")
    print(f">>> ndim={ndim}, nlive={nlive}, N_SIM={N_SIM}, dlogz={DLOGZ}")
    print(f">>> ncores={ncores}")
    print(">>> multiprocessing enabled")
    print(">>> deterministic common-random-number likelihood enabled")
    print("", flush=True)

    rstate = np.random.default_rng(seed)
    t0 = time.time()

    if ncores <= 1:
        sampler = NestedSampler(
            loglike,
            ptform,
            ndim,
            nlive=nlive,
            bound="multi",
            sample="rwalk",
            walks=50,
            rstate=rstate,
        )

        sampler.run_nested(
            dlogz=DLOGZ,
            print_progress=False,
        )

    else:
        with mp.Pool(processes=ncores) as pool:
            sampler = NestedSampler(
                loglike,
                ptform,
                ndim,
                nlive=nlive,
                bound="multi",
                sample="rwalk",
                walks=50,
                rstate=rstate,
                pool=pool,
                queue_size=ncores,
            )

            sampler.run_nested(
                dlogz=DLOGZ,
                print_progress=False,
            )

    dt = time.time() - t0

    res = sampler.results
    logz = res.logz[-1]
    logzerr = res.logzerr[-1]

    print(f"\n>>> {label} model finished in {dt / 60.0:.2f} min")
    print(f">>> logZ({label}) = {logz:.6f} +/- {logzerr:.6f}")

    result_file = OUTDIR / f"dynesty_results_{label}.pkl"
    with open(result_file, "wb") as f:
        pickle.dump(res, f)

    print(f">>> Saved dynesty result: {result_file}")

    try:
        weights = np.exp(res.logwt - res.logz[-1])
        samples_equal = resample_equal(res.samples, weights)

        posterior_file = OUTDIR / f"posterior_equal_{label}.pkl"
        save_dict = {
            "model_name": label,
            "columns": columns,
            "samples_equal": samples_equal,
            "logz": logz,
            "logzerr": logzerr,
            "nlive": nlive,
            "n_sim": N_SIM,
            "dlogz": DLOGZ,
        }

        with open(posterior_file, "wb") as f:
            pickle.dump(save_dict, f)

        print(f">>> Saved equal-weight posterior: {posterior_file}")

    except Exception as e:
        print(f">>> Warning: failed to save equal-weight posterior for {label}: {e}")

    return res


# ============================================================
# 7. Summary
# ============================================================

def summarize_results(res_full, res_zero):
    logz_full = res_full.logz[-1]
    logz_zero = res_zero.logz[-1]

    err_full = res_full.logzerr[-1]
    err_zero = res_zero.logzerr[-1]

    log_bf = logz_full - logz_zero
    log_bf_err = np.sqrt(err_full**2 + err_zero**2)

    print("\n" + "=" * 70)
    print("Bayesian evidence comparison")
    print("=" * 70)
    print(f"logZ_full = {logz_full:.8f} +/- {err_full:.8f}")
    print(f"logZ_zero = {logz_zero:.8f} +/- {err_zero:.8f}")
    print(f"logBF = logZ_full - logZ_zero = {log_bf:.8f} +/- {log_bf_err:.8f}")
    print("=" * 70)

    summary_file = OUTDIR / "bayes_factor_summary.txt"
    with open(summary_file, "w") as f:
        f.write("Bayesian evidence comparison\n")
        f.write("=" * 70 + "\n")
        f.write(f"logZ_full = {logz_full:.10f} +/- {err_full:.10f}\n")
        f.write(f"logZ_zero = {logz_zero:.10f} +/- {err_zero:.10f}\n")
        f.write(f"logBF = logZ_full - logZ_zero = {log_bf:.10f} +/- {log_bf_err:.10f}\n")
        f.write("=" * 70 + "\n")

    print(f">>> Saved summary: {summary_file}")


def summarize_from_saved():
    full_file = EVIDENCE / "lognormal_fixed/dynesty_results_full.pkl"
    zero_file = EVIDENCE / "lognormal_fixed/dynesty_results_zero.pkl"

    if not full_file.exists():
        raise FileNotFoundError(f"Cannot find {full_file}")

    if not zero_file.exists():
        raise FileNotFoundError(f"Cannot find {zero_file}")

    with open(full_file, "rb") as f:
        res_full = pickle.load(f)

    with open(zero_file, "rb") as f:
        res_zero = pickle.load(f)

    summarize_results(res_full, res_zero)


# ============================================================
# 8. Main
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Parallel dynesty evidence calculation for full vs zero pulsar kick-spin models."
    )

    parser.add_argument(
        "mode",
        choices=["full", "zero", "both", "summary"],
        help="Run full model, zero model, both models, or summarize saved results.",
    )

    parser.add_argument(
        "--ncores",
        type=int,
        default=min(8, mp.cpu_count()),
        help="Number of multiprocessing cores. Default: min(8, available CPUs).",
    )

    parser.add_argument("--execute-fit", action="store_true", help="Explicitly start nested sampling")
    args = parser.parse_args()
    if args.mode != "summary" and not args.execute_fit:
        parser.error("Fitting requires --execute-fit. Use summary to read archived evidence.")

    if args.mode == "full":
        run_dynesty_model("full", ncores=args.ncores)

    elif args.mode == "zero":
        run_dynesty_model("zero", ncores=args.ncores)

    elif args.mode == "both":
        res_full = run_dynesty_model("full", ncores=args.ncores)
        res_zero = run_dynesty_model("zero", ncores=args.ncores)
        summarize_results(res_full, res_zero)

    elif args.mode == "summary":
        summarize_from_saved()
