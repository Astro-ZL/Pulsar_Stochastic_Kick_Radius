import sys as _archive_sys
from pathlib import Path as _ArchivePath
_archive_root = next(p for p in _ArchivePath(__file__).resolve().parents if (p / "config.json").is_file())
_archive_sys.path.insert(0, str(_archive_root / "scripts"))
from archive_paths import ROOT, DATA, MCMC, EVIDENCE, run_dir

"""改进版日心横向速度与 LSR 修正速度比较图。"""

# ============================================================
# 模块 1：导入依赖和设置路径
# ============================================================
from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import LogLocator, NullFormatter

BASE = DATA
FIGOUT = run_dir("figures")


# ============================================================
# 模块 2：读取并配对77个脉冲星的修正前后速度
# ============================================================
raw_velocity_all = np.loadtxt(BASE / "vtrans_87_data.txt")
raw_period_all = np.loadtxt(BASE / "period_87_data.txt")
raw_err_low_all = np.loadtxt(BASE / "v_err_low_87_data.txt")
raw_err_up_all = np.loadtxt(BASE / "v_err_up_87_data.txt")

mask = (
    (raw_period_all > 0.1)
    & (raw_velocity_all > 10.0)
    & (raw_velocity_all <= 1000.0)
)

period_raw = raw_period_all[mask]
velocity_raw = raw_velocity_all[mask]
raw_err_low = raw_err_low_all[mask]
raw_err_up = raw_err_up_all[mask]

period_lsr = np.loadtxt(BASE / "period_77_data.txt")
velocity_lsr = np.loadtxt(BASE / "vpec_lsr_median_77.txt")
lsr_err_low = np.loadtxt(BASE / "vpec_lsr_err_low_77.txt")
lsr_err_up = np.loadtxt(BASE / "vpec_lsr_err_up_77.txt")

if len(period_raw) != 77 or len(period_lsr) != 77:
    raise ValueError("修正前后样本都应包含77个源。")
if not np.allclose(period_raw, period_lsr):
    raise ValueError("修正前后数据的源顺序不一致。")

ratio = velocity_lsr / velocity_raw
increased = ratio >= 1.0
decreased = ~increased


# ============================================================
# 模块 3：期刊风格设置
# ============================================================
mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["STIX Two Text", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "pdf.fonttype": 42,
    "axes.linewidth": 1.0,
    "axes.labelsize": 10.5,
    "axes.titlesize": 10.5,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "legend.frameon": True,
    "legend.fancybox": False,
    "legend.framealpha": 1.0,
    "legend.facecolor": "white",
    "legend.edgecolor": "0.35",
    "legend.borderpad": 0.5,
})

grey = "#858585"
orange = "#D58A19"
blue = "#3B75AF"
dark = "#252525"


# ============================================================
# 模块 4：绘制双面板比较图
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.25))

# (a) P-V 平面：连接线明确表示同一个源修正前后的移动
ax = axes[0]
for p_value, v_before, v_after in zip(period_raw, velocity_raw, velocity_lsr):
    ax.plot(
        [p_value, p_value],
        [v_before, v_after],
        color="0.72",
        lw=0.65,
        alpha=0.75,
        zorder=1,
    )

helio_eb = ax.errorbar(
    period_raw,
    velocity_raw,
    yerr=np.vstack([raw_err_low, raw_err_up]),
    fmt="o",
    ms=3.0,
    mfc="white",
    mec=grey,
    mew=0.75,
    ecolor="0.70",
    elinewidth=0.55,
    capsize=1.2,
    alpha=0.85,
    zorder=2,
)
lsr_eb = ax.errorbar(
    period_lsr,
    velocity_lsr,
    yerr=np.vstack([lsr_err_low, lsr_err_up]),
    fmt="o",
    ms=3.1,
    mfc=orange,
    mec=orange,
    mew=0.6,
    ecolor=orange,
    elinewidth=0.55,
    capsize=1.2,
    alpha=0.90,
    zorder=3,
)

ax.set_xlabel(r"$P$ (s)")
ax.set_ylabel(r"$V_\perp$ (km s$^{-1}$)")
ax.legend(
    handles=[helio_eb, lsr_eb],
    labels=["Heliocentric", "LSR-corrected"],
    loc="lower right",
    fontsize=8,
)

# (b) 直接比较：1:1线以上表示修正后速度增加，以下表示降低
ax = axes[1]
limits = (7.0, 1500.0)
ax.plot(limits, limits, color="0.25", lw=1.0, ls="--", zorder=1)

for selection, color, label in [
    (increased, orange, "Increased after correction"),
    (decreased, blue, "Decreased after correction"),
]:
    ax.errorbar(
        velocity_raw[selection],
        velocity_lsr[selection],
        xerr=np.vstack([raw_err_low[selection], raw_err_up[selection]]),
        yerr=np.vstack([lsr_err_low[selection], lsr_err_up[selection]]),
        fmt="o",
        ms=3.1,
        mfc=color,
        mec="white",
        mew=0.35,
        ecolor=mpl.colors.to_rgba(color, 0.45),
        elinewidth=0.45,
        capsize=0,
        alpha=0.82,
        label=label,
        zorder=2,
    )

ax.text(
    0.05,
    0.95,
    rf"median $V_{{\rm LSR}}/V_{{\rm helio}}={np.median(ratio):.2f}$"
    + "\n"
    + rf"$N_\uparrow={np.sum(increased)},\ N_\downarrow={np.sum(decreased)}$",
    transform=ax.transAxes,
    ha="left",
    va="top",
    fontsize=8.2,
)
ax.set_xlabel(r"$V_{\perp,\rm helio}$ (km s$^{-1}$)")
ax.set_ylabel(r"$V_{\perp,\rm LSR}$ (km s$^{-1}$)")
ax.legend(loc="lower right", fontsize=7.6)


# ============================================================
# 模块 5：统一对数坐标和导出
# ============================================================
for ax in axes:
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.xaxis.set_major_locator(LogLocator(base=10))
    ax.yaxis.set_major_locator(LogLocator(base=10))
    ax.xaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 0.1))
    ax.yaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 0.1))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.grid(color="0.92", lw=0.55)
    ax.set_axisbelow(True)

axes[0].set_xlim(0.1, 10.0)
axes[0].set_ylim(7.0, 1500.0)
axes[1].set_xlim(limits)
axes[1].set_ylim(limits)

fig.subplots_adjust(left=0.09, right=0.99, top=0.90, bottom=0.16, wspace=0.24)

output = FIGOUT / "heliocentric_vs_lsr_velocity"
fig.savefig(str(output) + ".pdf", bbox_inches="tight")
fig.savefig(str(output) + ".png", dpi=300, bbox_inches="tight")
plt.close("all")
print(f"Saved: {output}.pdf")
