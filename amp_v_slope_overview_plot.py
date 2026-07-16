#!/usr/bin/env python3
"""
Make a 4-panel overview plot of MLE amplitude vs slope.

Panels:
  top left:     MLE simulations, Blue sample
  top right:    resolution models, Blue sample
  bottom left:  MLE simulations, Green sample
  bottom right: resolution models, Green sample

This script reproduces the two relevant single-panel plots from the original
script:
  ./Plots/amp_v_slope_mle_{iz}.*
  ./Plots/amp_v_slope_mle_{iz}_resolution.*

and combines them into one overview figure.

Example:
  python amp_v_slope_overview_plot.py L1000N1800 0 --output-format pdf
  python amp_v_slope_overview_plot.py L1000N1800 0 --output-format png --dpi 400
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=8)
plt.rcParams['font.size'] = 8


# -----------------------------------------------------------------------------
# Simulation-series settings copied from the original plotting script
# -----------------------------------------------------------------------------
SIMS = [
    "HYDRO_LOW_SIGMA8_STRONGEST_AGN",
    "HYDRO_LOW_SIGMA8",
    "HYDRO_PLANCK_LARGE_NU_FIXED",
    "HYDRO_PLANCK_LARGE_NU_VARY",
    "HYDRO_PLANCK",
    "HYDRO_STRONG_JETS_published",
    "HYDRO_JETS_published",
    "HYDRO_STRONG_SUPERNOVA",
    "HYDRO_STRONGEST_AGN",
    "HYDRO_STRONGER_AGN",
    "HYDRO_STRONG_AGN",
    "HYDRO_WEAK_AGN",
    "HYDRO_FIDUCIAL",
][::-1]

SIM_NAMES = [
    r"LS8\_fgas$-8\sigma$",
    "LS8",
    "PlanckNu0p24Fix",
    "PlanckNu0p24Var",
    "Planck",
    r"Jet\_fgas$-4\sigma$",
    "Jet",
    r"M*-$\sigma$",
    r"fgas$-8\sigma$",
    r"fgas$-4\sigma$",
    r"fgas$-2\sigma$",
    r"fgas+2$\sigma$",
    "L1\_m9",
][::-1]

SIM_COLOURS = [
    "#7B68EE",
    "#882255",
    "#999933",
    "#AA4499",
    "#44AA99",
    "#55E18E",
    "#7EFF4B",
    "#FF8C40",
    "#105ba4",
    "#3787c0",
    "#6aaed6",
    "#abd0e6",
    "#117733",
][::-1]


# -----------------------------------------------------------------------------
# Resolution-series settings copied from the original plotting script
# -----------------------------------------------------------------------------
RES = [
    "L2800N5040",
    "L1000N3600",
    "L1000N1800",
][::-1]

RES_NAMES = [
    "L2p8\_m9",
    "L1\_m8",
    "L1\_m9",
][::-1]

RES_COLOURS = [
    "#332288",
    "#CC6677",
    "#117733",
][::-1]

FIDUCIAL_SIM = "HYDRO_FIDUCIAL"
SAMPLES = ("Blue", "Green")


# -----------------------------------------------------------------------------
# I/O helpers
# -----------------------------------------------------------------------------
def read_mle_values(path: Path) -> tuple[float, float, float, float, float, float]:
    """
    Read AMP, SLOPE, and their lower/upper errors from mle_values.txt.

    This assumes the same file layout as the original script:
      skiprows=6  -> AMP
      skiprows=7  -> SLOPE
      skiprows=8  -> AMP lower error
      skiprows=9  -> AMP upper error
      skiprows=10 -> SLOPE lower error
      skiprows=11 -> SLOPE upper error
    """
    if not path.exists():
        raise FileNotFoundError(f"Missing MLE file: {path}")

    amp = np.loadtxt(path, usecols=1, skiprows=6, max_rows=1, delimiter="=")
    slope = np.loadtxt(path, usecols=1, skiprows=7, max_rows=1, delimiter="=")
    amp_lower = np.loadtxt(path, usecols=1, skiprows=8, max_rows=1, delimiter="=")
    amp_upper = np.loadtxt(path, usecols=1, skiprows=9, max_rows=1, delimiter="=")
    slope_lower = np.loadtxt(path, usecols=1, skiprows=10, max_rows=1, delimiter="=")
    slope_upper = np.loadtxt(path, usecols=1, skiprows=11, max_rows=1, delimiter="=")

    return (
        float(amp),
        float(slope),
        float(amp_lower),
        float(amp_upper),
        float(slope_lower),
        float(slope_upper),
    )


def mle_path(data_root: Path, box: str, isim: str, iz: str, lc: int | str) -> Path:
    return data_root / "mle_parameters" / box / isim / iz / f"lightcone{lc}" / "mle_values.txt"

        


def load_sim_series(data_root: Path, box: str, iz: str, lc: int | str):
    rows = []
    for sim, label, colour in zip(SIMS, SIM_NAMES, SIM_COLOURS):
        vals = read_mle_values(mle_path(data_root, box, sim, iz, lc))
        rows.append({"kind": "sim", "name": sim, "label": label, "colour": colour, "values": vals})
    return rows


def load_resolution_series(data_root: Path, iz: str, lc: int | str):
    rows = []
    tab10 = plt.get_cmap("tab10")

    for res, label, colour in zip(RES, RES_NAMES, RES_COLOURS):
        if res == "L2800N5040":
            # Original script plots all 8 L2p8 lightcones separately.
            for lc_2p8 in range(8):
                vals = read_mle_values(mle_path(data_root, res, FIDUCIAL_SIM, iz, lc_2p8))
                rows.append(
                    {
                        "kind": "resolution",
                        "name": res,
                        "label": f"L2p8_m9 (lc={lc_2p8})",
                        "edge_colour": "#332288",
                        "face_colour": tab10(lc_2p8),
                        "values": vals,
                    }
                )
        else:
            vals = read_mle_values(mle_path(data_root, res, FIDUCIAL_SIM, iz, lc))
            rows.append(
                {
                    "kind": "resolution",
                    "name": res,
                    "label": label,
                    "edge_colour": colour,
                    "face_colour": colour,
                    "values": vals,
                }
            )

    return rows


# -----------------------------------------------------------------------------
# Plot helpers
# -----------------------------------------------------------------------------
def plot_sim_panel(ax, rows, title: str, show_legend: bool = True):
    for row in rows:
        amp, slope, amp_lower, amp_upper, slope_lower, slope_upper = row["values"]
        colour = row["colour"]

        ax.errorbar(
            amp,
            slope,
            xerr=np.array([[amp_lower], [amp_upper]]),
            yerr=np.array([[slope_lower], [slope_upper]]),
            fmt="*",
            capsize=3,
            color=colour,
            zorder=1,
        )
        ax.scatter(
            amp,
            slope,
            marker="*",
            s=50,
            color=colour,
            label=row["label"],
            zorder=2,
        )

    # ax.set_title(title)
    ax.grid(which="major", linestyle="--", linewidth=0.5, alpha=0.7)
    if show_legend:
        ax.legend(loc="upper right", fontsize=8, title=title, frameon=False)


def plot_resolution_panel(ax, rows, title: str, show_legend: bool = True):
    for row in rows:
        amp, slope, amp_lower, amp_upper, slope_lower, slope_upper = row["values"]
        face_colour = row["face_colour"]
        edge_colour = row["edge_colour"]

        ax.errorbar(
            amp,
            slope,
            xerr=np.array([[amp_lower], [amp_upper]]),
            yerr=np.array([[slope_lower], [slope_upper]]),
            fmt="none",
            capsize=3,
            ecolor=face_colour,
            zorder=1,
        )
        ax.scatter(
            amp,
            slope,
            marker="*",
            s=50,
            facecolor=face_colour,
            edgecolor=edge_colour,
            label=row["label"],
            zorder=2,
        )

    # ax.set_title(title)
    ax.grid(which="major", linestyle="--", linewidth=0.5, alpha=0.7)
    if show_legend:
        ax.legend(loc="upper right", fontsize=8, title=title, frameon=False)


def make_plot(args):
    data_root = Path(args.data_root)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    fig, axs = plt.subplots(
        2,
        2,
        figsize=(11, 8.5),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )

    # Top left: MLE sims, Blue
    plot_sim_panel(
        axs[0, 0],
        load_sim_series(data_root, args.box, "Blue", args.lightcone),
        title="Blue sample",
    )

    # Top right: resolution models, Blue
    plot_resolution_panel(
        axs[0, 1],
        load_resolution_series(data_root, "Blue", args.lightcone),
        title="Blue sample",
    )

    # Bottom left: MLE sims, Green
    plot_sim_panel(
        axs[1, 0],
        load_sim_series(data_root, args.box, "Green", args.lightcone),
        title="Green sample",
    )

    # Bottom right: resolution models, Green
    plot_resolution_panel(
        axs[1, 1],
        load_resolution_series(data_root, "Green", args.lightcone),
        title="Green sample",
    )

    for ax in axs.flat:
        ax.set_xlim(args.xmin, args.xmax)
        ax.set_ylim(args.ymin, args.ymax)

    for ax in axs[1, :]:
        ax.set_xlabel("Amplitude")
    for ax in axs[:, 0]:
        ax.set_ylabel("Slope")

    # fig.suptitle("MLE amplitude vs slope overview", y=1.02)

    output_path = outdir / f"amp_v_slope_overview_plot.{args.output_format}"
    fig.savefig(output_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Make 4-panel amp-v-slope overview plot.")
    parser.add_argument("box", help="Box used for the MLE simulation panels, e.g. L1000N1800")
    parser.add_argument("lightcone", help="Lightcone number for non-L2p8 data, e.g. 0")
    parser.add_argument("--data-root", default="./data_files", help="Root data directory.")
    parser.add_argument("--outdir", default="./Plots", help="Output plot directory.")
    parser.add_argument("--output-format", choices=("png", "pdf"), default="pdf", help="Output file format.")
    parser.add_argument("--dpi", type=int, default=400, help="DPI for saved figure.")
    parser.add_argument("--xmin", type=float, default=10.3, help="Minimum x-axis amplitude.")
    parser.add_argument("--xmax", type=float, default=11.3, help="Maximum x-axis amplitude.")
    parser.add_argument("--ymin", type=float, default=0.0, help="Minimum y-axis slope.")
    parser.add_argument("--ymax", type=float, default=1.2, help="Maximum y-axis slope.")
    return parser.parse_args()


if __name__ == "__main__":
    make_plot(parse_args())
