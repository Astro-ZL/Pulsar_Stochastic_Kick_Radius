import sys as _archive_sys
from pathlib import Path as _ArchivePath
_archive_root = next(p for p in _ArchivePath(__file__).resolve().parents if (p / "config.json").is_file())
_archive_sys.path.insert(0, str(_archive_root / "scripts"))
from archive_paths import ROOT, DATA, MCMC, EVIDENCE, run_dir

"""Reproduce the 8-parameter Maxwellian corner plot.

This is the standalone version of the plot code recovered from the earlier
notebook.  It does not refit the model; it only reads the completed posterior
pickle files and overlays the young and full samples.
"""

import argparse
import pickle
from pathlib import Path

import corner
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import AutoMinorLocator


BASE = DATA
FIGOUT = run_dir("figures")
OUTDIR = MCMC
PATH_YOUNG = OUTDIR / "posterior_young_M_47_8param.pkl"
PATH_FULL = OUTDIR / "posterior_all_M_77_8param.pkl"

TARGET_COLUMNS = [
    "sigma_v",
    "R_eff",
    "zeta",
    "sin_alpha",
    "sin_beta",
    "Pb_ms",
    "k_w",
    "f_evo",
]

LABELS = [
    r"$\sigma_{v}$ ($\mathrm{km}\,\mathrm{s}^{-1}$)",
    r"$R_{\mathrm{eff}}^{\mathrm{kick}}$ ($\mathrm{km}$)",
    r"$\mathrm{\zeta}_{J}$",
    r"$\sin \alpha_{\mathrm{eff}}$",
    r"$\sin \beta_{\mathrm{eff}}$",
    r"$P_{\mathrm{b},*}$ ($\mathrm{ms}$)",
    r"$k_{w}$",
    r"$f_{\mathrm{evo}}$",
]

PARAM_NAMES = [
    r"\sigma_{v}",
    r"R_{\mathrm{eff}}^{\mathrm{kick}}",
    r"\mathrm{\zeta}_{J}",
    r"\sin \alpha_{\mathrm{eff}}",
    r"\sin \beta_{\mathrm{eff}}",
    r"P_{\mathrm{b},*}",
    r"k_{w}",
    r"f_{\mathrm{evo}}",
]

RANGES = [
    (170, 300),
    (0, 18),
    (0, 1),
    (0, 0.8),
    (0.2, 1),
    (0, 1200),
    (0.4, 1.2),
    (0, 20),
]

COLOR_FULL = "#E67E22"
COLOR_YOUNG = "#2C7BB6"


def load_samples(path: Path) -> np.ndarray:
    with path.open("rb") as handle:
        data = pickle.load(handle)
    columns = data["columns"]
    samples = np.asarray(data["samples"])
    indices = [columns.index(column) for column in TARGET_COLUMNS]
    return samples[:, indices]


def median_err(values):
    q16, q50, q84 = np.percentile(values, [16, 50, 84])
    return q50, q50 - q16, q84 - q50


def make_figure(output_path: Path) -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["STIX Two Text", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "axes.labelsize": 20,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "axes.linewidth": 1.5,
            "xtick.major.size": 6,
            "ytick.major.size": 6,
            "xtick.minor.size": 3,
            "ytick.minor.size": 3,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
        }
    )

    samples_young = load_samples(PATH_YOUNG)
    samples_full = load_samples(PATH_FULL)

    fig = corner.corner(
        samples_young,
        labels=LABELS,
        color=COLOR_YOUNG,
        fill_contours=False,
        plot_density=True,
        plot_datapoints=False,
        show_titles=False,
        smooth=2,
        range=RANGES,
        quantiles=[0.16, 0.84],
        hist_kwargs={
            "density": True,
            "color": COLOR_YOUNG,
            "alpha": 1,
            "linewidth": 1.5,
        },
        contour_kwargs={
            "colors": COLOR_YOUNG,
            "linewidths": 1.5,
            "linestyles": "solid",
        },
    )

    corner.corner(
        samples_full,
        fig=fig,
        color=COLOR_FULL,
        fill_contours=False,
        plot_density=True,
        plot_datapoints=False,
        show_titles=False,
        smooth=2,
        range=RANGES,
        quantiles=[0.16, 0.84],
        hist_kwargs={
            "density": True,
            "color": COLOR_FULL,
            "alpha": 1,
            "linewidth": 1.5,
        },
        contour_kwargs={
            "colors": COLOR_FULL,
            "linewidths": 1.5,
            "linestyles": "solid",
        },
    )

    n_params = len(LABELS)
    axes = np.asarray(fig.axes).reshape(n_params, n_params)
    for index in range(n_params):
        ax = axes[index, index]
        ax.set_ylim(ax.get_ylim()[0], ax.get_ylim()[1] * 1.5)

        median_young, low_young, high_young = median_err(samples_young[:, index])
        median_full, low_full, high_full = median_err(samples_full[:, index])
        text_young = (
            fr"${PARAM_NAMES[index]} = {median_young:.2f}^{{+{high_young:.2f}}}"
            fr"_{{-{low_young:.2f}}}$"
        )
        text_full = (
            fr"${PARAM_NAMES[index]} = {median_full:.2f}^{{+{high_full:.2f}}}"
            fr"_{{-{low_full:.2f}}}$"
        )

        ax.text(
            0.5,
            1.15,
            text_young,
            color=COLOR_YOUNG,
            fontsize=15,
            ha="center",
            va="bottom",
            transform=ax.transAxes,
        )
        ax.text(
            0.5,
            1.0,
            text_full,
            color=COLOR_FULL,
            fontsize=15,
            ha="center",
            va="bottom",
            transform=ax.transAxes,
        )

    for ax in fig.axes:
        ax.xaxis.set_minor_locator(AutoMinorLocator(5))
        ax.yaxis.set_minor_locator(AutoMinorLocator(5))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"8-parameter historical corner plot saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=FIGOUT / "corner_8param.pdf",
        help="Output PDF path (default: reproduced/figures/corner_8param.pdf)",
    )
    make_figure(parser.parse_args().output)
