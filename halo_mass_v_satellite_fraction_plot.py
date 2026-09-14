#!/usr/bin/env python3

"""
Plot mean halo M200crit against total satellite fraction
for FLAMINGO mock galaxy catalogues.

x-axis:
    mean M200crit, with +/- 1-sigma scatter in halo mass.

y-axis:
    total satellite fraction N_sat / N_total,
    with binomial 1-sigma uncertainty.

Optional observed unWISE points can be hard-coded in OBSERVED_DATA.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np, pandas as pd, pylab as pb


# -----------------------------------------------------------------------------
# Plot style
# -----------------------------------------------------------------------------

pb.rc("text", usetex=True)
pb.rc("font", family="serif", size=8)
pb.rcParams["font.size"] = 8


# -----------------------------------------------------------------------------
# FLAMINGO simulation scheme
# -----------------------------------------------------------------------------

SIMS = [
    "HYDRO_FIDUCIAL",
    "HYDRO_WEAK_AGN",
    "HYDRO_STRONG_AGN",
    "HYDRO_STRONGER_AGN",
    "HYDRO_STRONGEST_AGN",
    "HYDRO_STRONG_SUPERNOVA",
    "HYDRO_JETS_published",
    "HYDRO_STRONG_JETS_published",
    "HYDRO_PLANCK",
    "HYDRO_PLANCK_LARGE_NU_VARY",
    "HYDRO_PLANCK_LARGE_NU_FIXED",
    "HYDRO_LOW_SIGMA8",
    "HYDRO_LOW_SIGMA8_STRONGEST_AGN",
]

SIM_LABELS = {
    "HYDRO_FIDUCIAL":                 r"L1\_m9",
    "HYDRO_WEAK_AGN":                 r"fgas$+2\sigma$",
    "HYDRO_STRONG_AGN":               r"fgas$-2\sigma$",
    "HYDRO_STRONGER_AGN":             r"fgas$-4\sigma$",
    "HYDRO_STRONGEST_AGN":            r"fgas$-8\sigma$",
    "HYDRO_STRONG_SUPERNOVA":         r"M*-$\sigma$",
    "HYDRO_JETS_published":           r"Jet",
    "HYDRO_STRONG_JETS_published":    r"Jet\_fgas$-4\sigma$",
    "HYDRO_PLANCK":                   r"Planck",
    "HYDRO_PLANCK_LARGE_NU_VARY":     r"PlanckNu0p24Var",
    "HYDRO_PLANCK_LARGE_NU_FIXED":    r"PlanckNu0p24Fix",
    "HYDRO_LOW_SIGMA8":               r"LS8",
    "HYDRO_LOW_SIGMA8_STRONGEST_AGN": r"LS8\_fgas$-8\sigma$",
}

SIM_COLOURS = {
    "HYDRO_FIDUCIAL":                 "#117733",
    "HYDRO_WEAK_AGN":                 "#abd0e6",
    "HYDRO_STRONG_AGN":               "#6aaed6",
    "HYDRO_STRONGER_AGN":             "#3787c0",
    "HYDRO_STRONGEST_AGN":            "#105ba4",
    "HYDRO_STRONG_SUPERNOVA":         "#FF8C40",
    "HYDRO_JETS_published":           "#7EFF4B",
    "HYDRO_STRONG_JETS_published":    "#55E18E",
    "HYDRO_PLANCK":                   "#44AA99",
    "HYDRO_PLANCK_LARGE_NU_VARY":     "#AA4499",
    "HYDRO_PLANCK_LARGE_NU_FIXED":    "#999933",
    "HYDRO_LOW_SIGMA8":               "#882255",
    "HYDRO_LOW_SIGMA8_STRONGEST_AGN": "#7B68EE",
}


# -----------------------------------------------------------------------------
# Hard-coded observed data
# -----------------------------------------------------------------------------
#
# Fill these in later.
#
# mass:
#     observed mean halo M200crit
#
# mass_err:
#     1-sigma uncertainty on observed mean halo mass
#
# sat_fraction:
#     observed total satellite fraction
#
# sat_fraction_err:
#     uncertainty on observed satellite fraction
#
# Set an entry to None if you do not want to plot it yet.
#

OBSERVED_DATA = {
    # Example:
    "Blue": {
        "mass": 1.88e13 / 0.6766,
        "mass_err": None,
        "sat_fraction": 0.30,
        "sat_fraction_err": None,
    },

    # Example:
    "Green": {
        "mass": 1.66e13 / 0.6766,
        "mass_err": None,
        "sat_fraction": 0.16,
        "sat_fraction_err": None,
    },
}


# -----------------------------------------------------------------------------
# Catalogue I/O
# -----------------------------------------------------------------------------

def name_float(x, mle=False):
    if mle:
        return f"{float(x):.3f}".replace(".", "p")
    else:
        return f"{float(x):.1f}".replace(".", "p")


def catalogue_path(data_root, box, sim, sample, lightcone, amp, slope, mle=True):
    amp_name = name_float(amp, mle=mle)
    slope_name = name_float(slope, mle=mle)

    directory = (Path(data_root) / box / sim / sample / f"lightcone{lightcone}")

    base_name = (f"sampled_halo_data_{amp_name}_{slope_name}")

    # If you used --overwrite when adding m200crit, it will instead
    # be stored under the original filename.
    overwritten_path = directory / f"{base_name}.parquet"

    if overwritten_path.exists():
        return overwritten_path

    raise FileNotFoundError(f"Could not find mock catalogue. Tried: {overwritten_path}")


def load_catalogue(path):

    print(f"Reading: {path}")

    catalogue = pd.read_parquet(path)

    required = ["m200crit", "Structuretype"]

    missing = [col for col in required if col not in catalogue.columns]

    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")

    m200crit = catalogue["m200crit"].to_numpy(dtype=float)

    # SOAP-HBT IsCentral convention:
    #     1 = central
    #     0 = satellite
    is_satellite = (catalogue["Structuretype"].to_numpy() == 0)

    return m200crit, is_satellite


def load_mle_parameters(mle_root, box, sim, sample, lightcone):
    path = (Path(mle_root) / box / sim / sample / f"lightcone{lightcone}" / "mle_values.txt")

    if not path.exists():
        raise FileNotFoundError(f"Missing MLE file: {path}")

    amp = np.loadtxt(path, usecols=1, skiprows=6, max_rows=1, delimiter="=")
    slope = np.loadtxt(path, usecols=1, skiprows=7, max_rows=1, delimiter="=")

    return float(amp), float(slope)


# -----------------------------------------------------------------------------
# Statistics
# -----------------------------------------------------------------------------

def catalogue_statistics(m200crit, is_satellite):
    """
    Return:

        mean halo mass
        1-sigma halo-mass scatter
        satellite fraction
        binomial satellite-fraction error
        total galaxy count
        satellite count
    """

    m200crit = np.asarray(m200crit, dtype=float)
    is_satellite = np.asarray(is_satellite, dtype=bool)

    valid = (np.isfinite(m200crit) & (m200crit > 0))

    m200crit = m200crit[valid]
    is_satellite = is_satellite[valid]

    if len(m200crit) == 0:
        raise ValueError("No valid galaxies remain after filtering M200crit.")

    n_total = len(is_satellite)
    n_sat = np.count_nonzero(is_satellite)

    mean_mass = np.mean(m200crit)

    # Population standard deviation of the halo-mass distribution.
    mass_sigma = np.std(m200crit, ddof=0) / np.sqrt(len(m200crit))

    sat_fraction = n_sat / n_total

    # Standard binomial error.
    sat_fraction_error = np.sqrt(sat_fraction * (1.0 - sat_fraction) / n_total)

    return {
        "mean_mass": mean_mass,
        "mass_sigma": mass_sigma,
        "sat_fraction": sat_fraction,
        "sat_fraction_error": sat_fraction_error,
        "n_total": n_total,
        "n_sat": n_sat,
    }


# -----------------------------------------------------------------------------
# Load FLAMINGO series
# -----------------------------------------------------------------------------


def load_simulation_series(args):

    rows = []

    for sim in SIMS:

        amp, slope = load_mle_parameters(args.mle_root, args.box, sim, args.sample, args.lightcone)

        try:
            path = catalogue_path(data_root=args.data_root, box=args.box, sim=sim, sample=args.sample, 
                                  lightcone=args.lightcone, amp=amp, slope=slope, mle=True)

            m200crit, is_satellite = load_catalogue(path)

        except FileNotFoundError:
            if args.skip_missing:
                print(f"Skipping {sim}: catalogue missing.")
                continue

            raise

        stats = catalogue_statistics(m200crit, is_satellite)

        rows.append(
            {
                "sim": sim,
                "label": SIM_LABELS[sim],
                "colour": SIM_COLOURS[sim],
                "amp": amp,
                "slope": slope,
                **stats,
            }
        )

    return rows


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------

def plot_simulations(ax, rows):

    for row in rows:

        ax.errorbar(row["mean_mass"], row["sat_fraction"], xerr=row["mass_sigma"], yerr=row["sat_fraction_error"], 
                    fmt="none", ecolor=row["colour"], elinewidth=1.0, capsize=3, zorder=1)

        ax.scatter(row["mean_mass"], row["sat_fraction"], marker="*", s=55, color=row["colour"], label=row["label"], zorder=2)


def plot_observation(ax, sample):

    obs = OBSERVED_DATA.get(sample)

    if obs is None:
        return

    ax.errorbar(obs["mass"], obs["sat_fraction"], xerr=obs["mass_err"], yerr=obs["sat_fraction_err"], 
                fmt="X", markersize=5, color="k", ecolor="k", elinewidth=1.0, capsize=3, label=rf"Kusiak et al. (2022)", zorder=5)


def make_plot(args):

    rows = load_simulation_series(args)

    fig, ax = pb.subplots(figsize=(6.0, 4.5))

    plot_simulations(ax, rows)

    if args.observed:
        plot_observation(ax, args.sample)

    ax.set_xlabel(r"Mean halo mass, $\langle M_{200{\rm c}}\rangle\ [{\rm M_\odot}]$")
    ax.set_ylabel(r"Satellite fraction, $f_{\rm sat}$")

    if args.log_mass:
        ax.set_xscale("log")

    if args.xmin is not None:
        ax.set_xlim(left=args.xmin)

    if args.xmax is not None:
        ax.set_xlim(right=args.xmax)

    if args.ymin is not None:
        ax.set_ylim(bottom=args.ymin)

    if args.ymax is not None:
        ax.set_ylim(top=args.ymax)

    if args.grid:
        ax.grid(which="major", linestyle="--", linewidth=0.5, alpha=0.7)

    ax.legend(loc=args.legend_loc, fontsize=8, title=f"{args.sample} sample", frameon=False)

    fig.tight_layout()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    outfile = (outdir / f"halo_mass_vs_satellite_fraction_{args.sample.lower()}.{args.output_format}")

    fig.savefig(outfile, dpi=args.dpi, bbox_inches="tight")

    pb.close(fig)

    print(f"Saved: {outfile}")

    print()
    print(f"{args.sample} catalogue statistics:")
    print()

    for row in rows:
        print(
            f"{row['label']:25s} "
            f"<M200c> = {row['mean_mass']:.4e}, "
            f"sigma_M = {row['mass_sigma']:.4e}, "
            f"f_sat = {row['sat_fraction']:.5f} +/- "
            f"{row['sat_fraction_error']:.5f}, "
            f"N = {row['n_total']}, "
            f"N_sat = {row['n_sat']}"
        )


# -----------------------------------------------------------------------------
# Command line
# -----------------------------------------------------------------------------

def parse_args():

    parser = argparse.ArgumentParser(description=("Plot mean halo M200crit against total satellite fraction for FLAMINGO mock catalogues."))

    parser.add_argument("--sample", choices=("Blue", "Green"), default="Blue")
    parser.add_argument("--box", default="L1000N1800",)
    parser.add_argument("--lightcone", type=int, default=0)
    parser.add_argument("--data-root", default="./data_files/mock_halo_catalogs", help="Root directory containing sampled halo catalogues.")
    parser.add_argument("--mle-root", default="./data_files/mle_parameters", help="Root directory containing MLE parameter files.")
    parser.add_argument("--outdir", default="./Plots")
    parser.add_argument("--observed", action="store_true", help="Plot hard-coded observed unWISE point.")
    parser.add_argument("--grid", action="store_true", help="Enable major grid lines.")
    parser.add_argument("--log-mass", action="store_true", help="Use a logarithmic halo-mass x-axis.")
    parser.add_argument("--skip-missing", action="store_true", help="Skip simulations for which the catalogue is missing.")
    parser.add_argument("--xmin", type=float, default=None)
    parser.add_argument("--xmax", type=float, default=None)
    parser.add_argument("--ymin", type=float, default=0.0)
    parser.add_argument("--ymax", type=float, default=None)
    parser.add_argument("--legend-loc", default="best")
    parser.add_argument("--output-format", choices=("png", "pdf"), default="pdf")
    parser.add_argument("--dpi", type=int, default=400)

    return parser.parse_args()


if __name__ == "__main__":
    make_plot(parse_args())