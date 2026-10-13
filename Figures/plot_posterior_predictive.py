import sys as _archive_sys
from pathlib import Path as _ArchivePath
_archive_root = next(p for p in _ArchivePath(__file__).resolve().parents if (p / "config.json").is_file())
_archive_sys.path.insert(0, str(_archive_root / "scripts"))
from archive_paths import ROOT, DATA, MCMC, EVIDENCE, run_dir

"""
47 个年轻脉冲星：single-kick、stochastic full 和 stochastic zero 对比。

本脚本不重新进行拟合，而是直接读取已经完成的 dynesty 等权后验，
再用独立随机数生成 posterior-predictive population。
"""

# ============================================================
# 模块 1：导入依赖并设置文件路径
# ============================================================
import pickle
from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter
from scipy.special import expit, gamma
from scipy.stats import gaussian_kde

BASE = DATA
FIGOUT = run_dir("figures")
RUN = ROOT / "results"
OUT = run_dir("figures")
DYN = EVIDENCE / "maxwell_fixed"
DYN_MI = EVIDENCE / "maxwell_marginalized"

OUT.mkdir(parents=True, exist_ok=True)

# posterior predictive 的计算量；只影响绘图精度，不改变已经完成的拟合。
N_POSTERIOR_DRAWS = 300
N_POPULATION_PER_DRAW = 2000
N_KDE_MAX = 20000


# ============================================================
# 模块 2：读取不同格式的 posterior 文件
# ============================================================
def load_pickle_samples(path):
    """兼容裸 ndarray、samples 字典和 samples_equal 字典。"""
    with open(path, "rb") as file:
        data = pickle.load(file)

    if isinstance(data, np.ndarray):
        return data, None, {}

    if "samples_equal" in data:
        samples = np.asarray(data["samples_equal"])
    elif "samples" in data:
        samples = np.asarray(data["samples"])
    else:
        raise KeyError(f"{path} 中没有 samples 或 samples_equal。")

    return samples, data.get("columns"), data


# 47 样本 single-kick dynesty posterior
single_samples, single_columns, _ = load_pickle_samples(
    ROOT / "results/single_kick/posterior_LSR_47_samples.pkl"
)

# Maxwellian fixed-M,I：与 Bayes factor=8.86 对应的 full/zero dynesty posterior
full_samples, full_columns, full_meta = load_pickle_samples(
    DYN / "posterior_equal_full.pkl"
)
zero_samples, zero_columns, zero_meta = load_pickle_samples(
    DYN / "posterior_equal_zero.pkl"
)

expected_full = [
    "sigma_v", "R_eff", "zeta", "sin_alpha",
    "sin_beta", "Pb_ms", "k_w", "f_evo",
]
expected_zero = ["sigma_v", "Pb_ms", "k_w", "f_evo"]

if full_columns != expected_full:
    raise ValueError(f"full posterior 参数顺序不符合预期：{full_columns}")
if zero_columns != expected_zero:
    raise ValueError(f"zero posterior 参数顺序不符合预期：{zero_columns}")
if single_samples.shape[1] != 6:
    raise ValueError(f"single-kick posterior 应有 6 列，实际为 {single_samples.shape}")


# ============================================================
# 模块 3：读取同一组 47 个 LSR 修正观测数据
# ============================================================
period_obs = np.loadtxt(BASE / "period_2_new_47.txt")
velocity_obs = np.loadtxt(BASE / "vpec_lsr_median_47.txt")
velocity_err_low = np.loadtxt(BASE / "vpec_lsr_err_low_47.txt")
velocity_err_up = np.loadtxt(BASE / "vpec_lsr_err_up_47.txt")


# ============================================================
# 模块 4：single-kick posterior predictive band
# ============================================================
def single_kick_periods(velocity_kms, theta):
    """由 single-kick 参数计算给定速度下的预测周期。"""
    r_kick, alpha, beta, gamma_angle, p_birth, tau_tilde = theta
    mass = 1.4
    inertia = 50.0

    delta_omega = (
        mass * r_kick * velocity_kms * np.sin(alpha)
        / (inertia * np.sin(beta))
    )
    omega_birth = 2.0 * np.pi / p_birth
    omega_0 = np.sqrt(
        (omega_birth + delta_omega * np.cos(gamma_angle)) ** 2
        + (delta_omega * np.sin(gamma_angle)) ** 2
    )
    return 2.0 * np.pi / (omega_0 * np.sqrt(1.0 - tau_tilde))


rng_single = np.random.default_rng(4101)
single_idx = rng_single.choice(
    len(single_samples), size=min(4000, len(single_samples)), replace=False
)
velocity_grid = np.logspace(0.8, np.log10(1500.0), 260)
single_curves = np.array([
    single_kick_periods(velocity_grid, single_samples[index])
    for index in single_idx
])
single_q = np.percentile(
    single_curves,
    [0.135, 2.5, 16, 50, 84, 97.5, 99.865],
    axis=0,
)


# ============================================================
# 模块 5：从 dynesty posterior 生成真正的总体后验预测
# ============================================================
def simulate_maxwell_population(theta, model, n_sim, rng):
    """使用与证据计算相同的 Maxwellian stochastic-kick 物理模型。"""
    if model == "full":
        sigma_v, r_eff_km, zeta, sin_alpha, sin_beta, pb_ms, k_w, f_evo = theta
    elif model == "zero":
        sigma_v, pb_ms, k_w, f_evo = theta
        r_eff_km = 0.0
        zeta = 0.0
        sin_alpha = 0.5
        sin_beta = 0.5
    else:
        raise ValueError("model 必须是 full 或 zero。")

    # 每个 posterior draw 使用独立总体随机数；不重复使用拟合时的 COMMON cache。
    xyz = rng.normal(size=(n_sim, 3))
    maxwell_unit = np.sqrt(np.sum(xyz**2, axis=1))
    sin_projection = np.sqrt(1.0 - rng.uniform(-1.0, 1.0, n_sim) ** 2)
    weibull_base = -np.log1p(-rng.uniform(1e-12, 1.0 - 1e-12, n_sim))
    u_detect = rng.uniform(0.0, 1.0, n_sim)

    mass_cgs = 1.4 * 1.989e33
    inertia_cgs = 1.0e45

    velocity_3d_kms = sigma_v * maxwell_unit
    velocity_trans_kms = velocity_3d_kms * sin_projection
    velocity_trans_cgs = velocity_trans_kms * 1e5

    spin_birth = (2.0 * np.pi * inertia_cgs) / (pb_ms * 1e-3)

    if model == "full":
        delta_spin = (
            mass_cgs * (r_eff_km * 1e5) * sin_alpha * velocity_trans_cgs
            / (sin_beta + 1e-10)
        )
        delta_spin_sq_mean = zeta**2 * np.mean(delta_spin**2)
    else:
        delta_spin_sq_mean = 0.0

    spin0_rms = np.sqrt(spin_birth**2 + delta_spin_sq_mean)
    p0_rms = (2.0 * np.pi * inertia_cgs) / spin0_rms
    weibull_scale = p0_rms / np.sqrt(gamma(1.0 + 2.0 / k_w))
    period_birth = weibull_scale * weibull_base ** (1.0 / k_w)
    period_model = period_birth * f_evo

    p_detect = expit((period_model - 0.1) / 0.01)
    mask = (
        (u_detect < p_detect)
        & (period_model < 100.0)
        & (velocity_trans_kms > 0.0)
    )
    return period_model[mask], velocity_trans_kms[mask]


def posterior_predictive(samples, model, seed):
    """抽取参数后验并合并多个模拟总体。"""
    rng = np.random.default_rng(seed)
    indices = rng.choice(
        len(samples),
        size=min(N_POSTERIOR_DRAWS, len(samples)),
        replace=False,
    )
    periods = []
    velocities = []
    for index in indices:
        p_one, v_one = simulate_maxwell_population(
            samples[index], model, N_POPULATION_PER_DRAW, rng
        )
        periods.append(p_one)
        velocities.append(v_one)
    return np.concatenate(periods), np.concatenate(velocities)


period_full, velocity_full = posterior_predictive(full_samples, "full", 5201)
period_zero, velocity_zero = posterior_predictive(zero_samples, "zero", 5202)


# ============================================================
# 模块 6：在 log10(P)-log10(V) 空间计算 68%/95% HDR
# ============================================================
def highest_density_levels(density, probabilities=(0.997, 0.95, 0.68)):
    """返回包含指定概率质量的最高密度区域边界。"""
    flat = density.ravel()
    order = np.argsort(flat)[::-1]
    cumulative = np.cumsum(flat[order])
    cumulative /= cumulative[-1]
    levels = [
        flat[order[np.searchsorted(cumulative, probability)]]
        for probability in probabilities
    ]
    return sorted(levels)


def make_density_grid(periods, velocities, seed):
    """为提高速度，KDE 最多使用 N_KDE_MAX 个后验预测点。"""
    log_samples = np.column_stack([np.log10(periods), np.log10(velocities)])
    rng = np.random.default_rng(seed)
    if len(log_samples) > N_KDE_MAX:
        index = rng.choice(len(log_samples), N_KDE_MAX, replace=False)
        log_samples = log_samples[index]

    x_grid = np.linspace(-1.3, 1.05, 240)
    y_grid = np.linspace(0.75, np.log10(1500.0), 240)
    xx, yy = np.meshgrid(x_grid, y_grid)
    kde = gaussian_kde(log_samples.T, bw_method="scott")
    density = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
    return xx, yy, density


full_grid = make_density_grid(period_full, velocity_full, 6101)
zero_grid = make_density_grid(period_zero, velocity_zero, 6102)


# ============================================================
# 模块 7：绘制正文三面板图
# ============================================================
mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["STIX Two Text", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "pdf.fonttype": 42,
    "axes.linewidth": 1.3,
    "axes.labelsize": 14,
    "axes.titlesize": 14,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "legend.frameon": False,
})

red = "#B54845"
blue = "#356FA3"
orange = "#B16B3B"

fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.4), sharex=True, sharey=True)

# (a) single-kick：95%浅色、68%深色和后验中位线
axes[0].fill_betweenx(
    velocity_grid, single_q[1], single_q[5], color=red, alpha=0.14
)
axes[0].fill_betweenx(
    velocity_grid, single_q[2], single_q[4], color=red, alpha=0.32
)
axes[0].plot(single_q[3], velocity_grid, color=red, lw=2.2)
axes[0].plot(single_q[0], velocity_grid, color=red, lw=1.0, ls=":")
axes[0].plot(single_q[6], velocity_grid, color=red, lw=1.0, ls=":")

# (b,c) stochastic full/zero：posterior-predictive HDR
for ax, grid, color in [
    (axes[1], full_grid, blue),
    (axes[2], zero_grid, orange),
]:
    xx, yy, density = grid
    levels = highest_density_levels(density)
    ax.contourf(
        10**xx,
        10**yy,
        density,
        levels=[levels[1], levels[2], density.max()],
        colors=[
            mpl.colors.to_rgba(color, 0.13),
            mpl.colors.to_rgba(color, 0.30),
        ],
        extend="max",
    )
    ax.contour(
        10**xx,
        10**yy,
        density,
        levels=levels,
        colors=color,
        linewidths=[1.0, 1.3, 2.2],
        linestyles=[":", "-", "-"],
    )

# 三个面板叠加相同观测点
for ax in axes:
    ax.errorbar(
        period_obs,
        velocity_obs,
        yerr=np.vstack([velocity_err_low, velocity_err_up]),
        fmt="o",
        ms=3.6,
        mfc="white",
        mec="black",
        mew=0.8,
        ecolor="0.25",
        elinewidth=0.7,
        capsize=1.5,
        alpha=0.93,
        zorder=5,
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(10**-1.25, 10.0)
    ax.set_ylim(10**0.8, 1500.0)
    ax.set_xlabel(r"$P$ (s)")
    ax.xaxis.set_major_locator(LogLocator(base=10))
    ax.yaxis.set_major_locator(LogLocator(base=10))
    ax.xaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 0.1))
    ax.yaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 0.1))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.grid(color="0.92", lw=0.7)
    ax.set_axisbelow(True)

axes[0].set_title("(a) Single-kick model", pad=8)
axes[1].set_title("(b) Stochastic full model", pad=8)
axes[2].set_title(
    r"(c) Stochastic null: $\Delta\Omega_{\rm rms}=0$", pad=8
)
axes[0].set_ylabel(r"$V_{\perp,\rm pec}^{\rm LSR}$ (km s$^{-1}$)")

fig.subplots_adjust(left=0.09, right=0.995, top=0.93, bottom=0.16, wspace=0.08)

main_output = OUT / "posterior_predictive"
fig.savefig(str(main_output) + ".pdf", bbox_inches="tight")
fig.savefig(str(main_output) + ".png", dpi=300, bbox_inches="tight")
plt.close("all")


# ============================================================
# 模块 8：打印证据度，确认图和 Bayes factor 使用同一批 dynesty 结果
# ============================================================
logz_full = float(full_meta["logz"])
logz_zero = float(zero_meta["logz"])
log_bf = logz_full - logz_zero
print(f"logZ(full) = {logz_full:.6f} +/- {float(full_meta['logzerr']):.6f}")
print(f"logZ(zero) = {logz_zero:.6f} +/- {float(zero_meta['logzerr']):.6f}")
print(f"log Bayes factor = {log_bf:.6f}")
print(f"Bayes factor full/zero = {np.exp(log_bf):.3f}")
print(f"正文图已保存：{main_output}.pdf")
