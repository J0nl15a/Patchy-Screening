#!/usr/bin/env python3
"""
Make a 4-panel overview plot for halo-mass and stellar-mass distributions.

Panels:
  top left:     halo-mass counts, Blue sample
  top right:    halo-mass counts, Green sample
  bottom left:  stellar-mass counts, Blue sample
  bottom right: stellar-mass counts, Green sample

Two modes are supported:

1) population mode
   Plot Total, Centrals, and Satellites for one simulation/box.

2) simulation mode
   Plot the same population type from several simulations, using the same
   simulation labels/colours as amp_v_slope_overview_plot.py.

Examples
--------
python dndm_overview_plot.py 8 L1000N1800 HYDRO_FIDUCIAL 0 --mode population --output-format pdf
python dndm_overview_plot.py 8 L1000N1800 HYDRO_FIDUCIAL 0 --mode simulations --population total --output-format png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from imp_patchy_screening import patchyScreening

plt.rc('text', usetex=True)


# Match your catalogue convention
SATELLITE_FLAG = 0
CENTRAL_FLAG = 1
SAMPLES = ("Blue", "Green")

# Copied from amp_v_slope_overview_plot.py / amp-v-slope plotting code
SIMS = [
    "HYDRO_LOW_SIGMA8_STRONGEST_AGN",
    "HYDRO_LOW_SIGMA8",
    # "HYDRO_PLANCK_LARGE_NU_FIXED",
    # "HYDRO_PLANCK_LARGE_NU_VARY",
    # "HYDRO_PLANCK",
    # "HYDRO_STRONG_JETS_published",
    # "HYDRO_JETS_published",
    # "HYDRO_STRONG_SUPERNOVA",
    "HYDRO_STRONGEST_AGN",
    # "HYDRO_STRONGER_AGN",
    # "HYDRO_STRONG_AGN",
    # "HYDRO_WEAK_AGN",
    "HYDRO_FIDUCIAL",
][::-1]

SIM_NAMES = [
    r"LS8\_fgas$-8\sigma$",
    "LS8",
    # "PlanckNu0p24Fix",
    # "PlanckNu0p24Var",
    # "Planck",
    # r"Jet\_fgas$-4\sigma$",
    # "Jet",
    # r"M*$-\sigma$",
    r"fgas$-8\sigma$",
    # r"fgas$-4\sigma$",
    # r"fgas$-2\sigma$",
    # r"fgas$+2\sigma$",
    "L1\_m9",
][::-1]

SIM_COLOURS = [
    "#7B68EE",
    "#882255",
    # "#999933",
    # "#AA4499",
    # "#44AA99",
    # "#55E18E",
    # "#7EFF4B",
    # "#FF8C40",
    "#105ba4",
    # "#3787c0",
    # "#6aaed6",
    # "#abd0e6",
    "#117733",
][::-1]

RESOLUTION_RUNS = [
    {
        "box": "L1000N3600",
        "isim": "HYDRO_FIDUCIAL",
        "label": r"L1\_m8",
        "colour": "#CC6677",
        "lightcone": None,   # use args.lightcone
    },
    {
        "box": "L2800N5040",
        "isim": "HYDRO_FIDUCIAL",
        "label": r"L2p8\_m9",
        "colour": "#332288",
        "lightcone": 0,      # force lightcone 0
    },
]

POPULATION_STYLES = {
    "total": {"label": "Total", "colour": "black"},
    "central": {"label": "Centrals", "colour": "red"},
    "satellite": {"label": "Satellites", "colour": "blue"},
}


def set_plot_style(fontsize: int = 8) -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": fontsize,
        "axes.labelsize": fontsize,
        "xtick.labelsize": fontsize,
        "ytick.labelsize": fontsize,
        "legend.fontsize": fontsize,
        "legend.title_fontsize": fontsize,
    })


def read_mle_cut(data_root: Path, box: str, isim: str, iz: str, lc: int) -> tuple[float, float]:
    path = data_root / "mle_parameters" / box / isim / iz / f"lightcone{lc}" / "mle_values.txt"
    if not path.exists():
        raise FileNotFoundError(f"Missing MLE file: {path}")

    m_cut = float(np.loadtxt(path, usecols=1, skiprows=6, max_rows=1, delimiter="="))
    s_cut = float(np.loadtxt(path, usecols=1, skiprows=7, max_rows=1, delimiter="="))
    return m_cut, s_cut


def load_catalogue(data_root: Path, box: str, isim: str, iz: str, lc: int, ncpu: int):
    """Load and stellar-mass-filter the mock catalogue exactly as in the single-panel script."""
    m_cut, s_cut = read_mle_cut(data_root, box, isim, iz, lc)
    ps = patchyScreening(box, isim, iz, m_cut, s_cut, ncpu=ncpu, lightcone=lc, mle=True)
    ps.filter_stellar_mass()
    return ps.merge


def select_population(cat, population: str):
    if population == "total":
        return cat
    if population == "central":
        return cat[cat["Structuretype"] == CENTRAL_FLAG]
    if population == "satellite":
        return cat[cat["Structuretype"] == SATELLITE_FLAG]
    raise ValueError(f"Unknown population: {population}")


def finite_log10(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    mask = np.isfinite(values) & (values > 0)
    return np.log10(values[mask])


def histogram_log_mass(cat, mass_column: str, bins: int | np.ndarray):
    logm = finite_log10(cat[mass_column].to_numpy())
    return np.histogram(logm, bins=bins)


def plot_hist_with_errors(ax, counts: np.ndarray, bins: np.ndarray, colour: str, label: str) -> None:
    centres = 0.5 * (bins[1:] + bins[:-1])
    ax.stairs(counts, bins, color=colour, label=label)
    ax.errorbar(
        centres,
        counts,
        yerr=np.sqrt(counts),
        fmt="none",
        ecolor=colour,
        elinewidth=0.8,
        alpha=0.9,
    )


def plot_population_panel(ax, cat, mass_column: str, n_bins: int) -> None:
    total = select_population(cat, "total")
    counts_total, bins = histogram_log_mass(total, mass_column, n_bins)

    histograms = {"total": counts_total}
    for population in ("central", "satellite"):
        sub = select_population(cat, population)
        histograms[population], _ = histogram_log_mass(sub, mass_column, bins)

    for population in ("total", "central", "satellite"):
        style = POPULATION_STYLES[population]
        plot_hist_with_errors(ax, histograms[population], bins, style["colour"], style["label"])


# def plot_simulations_panel(ax, data_root: Path, box: str, iz: str, lc: int, ncpu: int, mass_column: str, population: str, n_bins: int) -> None:


#     # Use shared bins set by the fiducial simulation, then draw every simulation on those bins.
#     fid_cat = load_catalogue(data_root, box, "HYDRO_FIDUCIAL", iz, lc, ncpu)
#     _, bins = histogram_log_mass(select_population(fid_cat, population), mass_column, n_bins)

#     for sim, label, colour in zip(SIMS, SIM_NAMES, SIM_COLOURS):
#         try:
#             cat = fid_cat if sim == "HYDRO_FIDUCIAL" else load_catalogue(data_root, box, sim, iz, lc, ncpu)
#         except FileNotFoundError as err:
#             print(f"[WARN] {err}. Skipping {sim}/{iz}.")
#             continue

#         counts, _ = histogram_log_mass(select_population(cat, population), mass_column, bins)
#         plot_hist_with_errors(ax, counts, bins, colour, label)

def plot_simulations_panel(ax, data_root: Path, box: str, iz: str, lc: int, ncpu: int, mass_column: str, population: str, n_bins: int, include_resolutions: bool = False) -> None:

    fid_cat = load_catalogue(data_root, box, "HYDRO_FIDUCIAL", iz, lc, ncpu)
    _, bins = histogram_log_mass(select_population(fid_cat, population), mass_column, n_bins)

    plot_entries = []

    for sim, label, colour in zip(SIMS, SIM_NAMES, SIM_COLOURS):
        plot_entries.append({
            "box": box,
            "isim": sim,
            "label": label,
            "colour": colour,
            "lightcone": lc,
        })

    if include_resolutions:
        for run in RESOLUTION_RUNS:
            plot_entries.append({
                "box": run["box"],
                "isim": run["isim"],
                "label": run["label"],
                "colour": run["colour"],
                "lightcone": lc if run["lightcone"] is None else run["lightcone"],
            })

    for entry in plot_entries:
        try:
            if entry["box"] == box and entry["isim"] == "HYDRO_FIDUCIAL":
                cat = fid_cat
            else:
                cat = load_catalogue(data_root, entry["box"], entry["isim"], iz, entry["lightcone"], ncpu)
        except FileNotFoundError as err:
            print(f"[WARN] {err}. Skipping {entry['label']}/{iz}.")
            continue

        counts, _ = histogram_log_mass(select_population(cat, population), mass_column, bins)

        plot_hist_with_errors(ax, counts, bins, entry["colour"], entry["label"])


def configure_axes(axs) -> None:
    halo_axes = (axs[0, 0], axs[0, 1])
    stellar_axes = (axs[1, 0], axs[1, 1])

    for ax in axs.flat:
        ax.set_yscale("log")
        ax.tick_params(axis="both", which="both", labelsize=8)

    halo_axes[0].set_ylabel("Counts")
    stellar_axes[0].set_ylabel("Counts")
    halo_axes[0].set_xlabel(r"Halo Mass [$\log_{10}(M / M_\odot)$]")
    halo_axes[1].set_xlabel(r"Halo Mass [$\log_{10}(M / M_\odot)$]")
    stellar_axes[0].set_xlabel(r"Stellar Mass [$\log_{10}(M / M_\odot)$]")
    stellar_axes[1].set_xlabel(r"Stellar Mass [$\log_{10}(M / M_\odot)$]")

    # Same limits as your single-panel plots.
    for ax in halo_axes:
        ax.set_xlim(11.0, 16.2)
        ax.set_ylim(1e3, 1e8)
    for ax in stellar_axes:
        ax.set_xlim(10.5, 12.5)
        ax.set_ylim(1e3, 1e8)


def make_plot(args) -> Path:
    data_root = Path(args.data_root)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    set_plot_style(7)

    fig, axs = plt.subplots(2, 2, figsize=(7.2, 5.8), sharey=True)

    for col, iz in enumerate(SAMPLES):
        if args.mode == "population":
            cat = load_catalogue(data_root, args.box, args.isim, iz, args.lightcone, args.ncpu)
            plot_population_panel(axs[0, col], cat, "mvir", args.n_bins)
            plot_population_panel(axs[1, col], cat, "mstar", args.n_bins)
        elif args.mode == "simulations":
            plot_simulations_panel(axs[0, col], data_root, args.box, iz, args.lightcone, args.ncpu, "mvir", args.population, args.n_bins, include_resolutions=args.include_resolutions)
            plot_simulations_panel(axs[1, col], data_root, args.box, iz, args.lightcone, args.ncpu, "mstar", args.population, args.n_bins, include_resolutions=args.include_resolutions)
        else:
            raise ValueError(f"Unknown mode: {args.mode}")

    configure_axes(axs)

    # Legends: sample-labelled, no frame. Put only one legend per panel because the user requested sample labels.
    axs[0, 0].legend(title="Blue sample", loc="upper right", frameon=False)
    axs[0, 1].legend(title="Green sample", loc="upper right", frameon=False)
    axs[1, 0].legend(title="Blue sample", loc="upper right", frameon=False)
    axs[1, 1].legend(title="Green sample", loc="upper right", frameon=False)

    # Keep no titles/subplot titles.
    for ax in axs.flat:
        ax.set_title("")

    suffix = args.output_format.lower().lstrip(".")
    mode_tag = args.mode if args.mode == "population" else f"simulations_{args.population}"
    outpath = outdir / f"dndm_overview_{mode_tag}.{suffix}"
    fig.savefig(outpath, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    return outpath


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create 4-panel dN/dM overview plot.")
    parser.add_argument("ncpu", type=int, help="Number of CPUs passed to patchyScreening.")
    parser.add_argument("box", help="Box name, e.g. L1000N1800.")
    parser.add_argument("isim", help="Simulation name used in population mode, e.g. HYDRO_FIDUCIAL.")
    parser.add_argument("lightcone", type=int, help="Lightcone number.")
    parser.add_argument("--mode", choices=("population", "simulations"), default="population")
    parser.add_argument("--population", choices=("total", "central", "satellite"), default="total", help="Population to plot in simulations mode.")
    parser.add_argument("--include-resolutions", action="store_true", help="In simulations mode, also plot L1_m8 and L2p8_m9. L2p8_m9 uses lightcone 0.")
    parser.add_argument("--n-bins", type=int, default=100)
    parser.add_argument("--output-format", choices=("png", "pdf"), default="pdf")
    parser.add_argument("--dpi", type=int, default=400)
    parser.add_argument("--data-root", default="./data_files")
    parser.add_argument("--outdir", default="./Plots")
    return parser.parse_args()


if __name__ == "__main__":
    output = make_plot(parse_args())
    print(f"Saved {output}")
