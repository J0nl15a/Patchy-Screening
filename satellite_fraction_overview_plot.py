#!/usr/bin/env python3
"""
Make a 4-panel overview plot for satellite fractions.

Default panels:
  top left:     satellite fraction vs halo mass, Blue sample
  top right:    satellite fraction vs halo mass, Green sample
  bottom left:  satellite fraction vs redshift, Blue sample
  bottom right: satellite fraction vs redshift, Green sample

Use --mass-axis stellar to switch the top row to stellar mass instead of halo mass.

Examples
--------
python satellite_fraction_overview_plot.py 8 L1000N1800 HYDRO_FIDUCIAL 0 --output-format pdf
python satellite_fraction_overview_plot.py 8 L1000N1800 HYDRO_FIDUCIAL 0 --mass-axis stellar --output-format png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from imp_patchy_screening import patchyScreening

plt.rc('text', usetex=True)


SATELLITE_FLAG = 0
CENTRAL_FLAG = 1
SAMPLES = ("Blue", "Green")

SIMS = [
    "HYDRO_LOW_SIGMA8_STRONGEST_AGN",
    "HYDRO_LOW_SIGMA8",
    # "HYDRO_PLANCK",
    # "HYDRO_JETS_published",
    "HYDRO_STRONGEST_AGN",
    "HYDRO_FIDUCIAL",
][::-1]

SIM_NAMES = [
    r"LS8\_fgas$-8\sigma$",
    "LS8",
    # "Planck",
    # r"Jet",
    r"fgas$-8\sigma$",
    r"L1\_m9",
][::-1]

SIM_COLOURS = [
    "#7B68EE",
    "#882255",
    # "#44AA99",
    # "#7EFF4B",
    "#105ba4",
    "#117733",
][::-1]

RESOLUTION_RUNS = [
    {
        "box": "L1000N3600",
        "isim": "HYDRO_FIDUCIAL",
        "label": r"L1\_m8",
        "colour": "#CC6677",
        "lightcone": None,
    },
    {
        "box": "L2800N5040",
        "isim": "HYDRO_FIDUCIAL",
        "label": r"L2p8\_m9",
        "colour": "#332288",
        "lightcone": 0,
    },
]


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
    m_cut, s_cut = read_mle_cut(data_root, box, isim, iz, lc)
    ps = patchyScreening(box, isim, iz, m_cut, s_cut, ncpu=ncpu, lightcone=lc, mle=False)
    ps.filter_stellar_mass()
    return ps.merge


def finite_log10(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    mask = np.isfinite(values) & (values > 0)
    return np.log10(values[mask])


def fraction_and_error(n_sat: np.ndarray, n_total: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n_sat = np.asarray(n_sat, dtype=float)
    n_total = np.asarray(n_total, dtype=float)

    frac = np.full_like(n_total, np.nan, dtype=float)
    err = np.full_like(n_total, np.nan, dtype=float)

    valid = n_total > 0
    frac[valid] = n_sat[valid] / n_total[valid]
    err[valid] = np.sqrt(frac[valid] * (1.0 - frac[valid]) / n_total[valid])
    return frac, err


def mass_satellite_fraction(cat, mass_column: str, n_bins: int):
    total = cat
    satellites = cat[cat["Structuretype"] == SATELLITE_FLAG]

    total_logm = finite_log10(total[mass_column].to_numpy())
    sat_logm = finite_log10(satellites[mass_column].to_numpy())

    total_hist, bins = np.histogram(total_logm, bins=n_bins)
    sat_hist, _ = np.histogram(sat_logm, bins=bins)

    frac, err = fraction_and_error(sat_hist, total_hist)
    x = bins[:-1]
    return x, frac, err


def redshift_midpoints_path(redshift_root: Path, box: str, isim: str, lc: int) -> Path:
    return redshift_root / box / isim / f"lightcone{lc}" / "FLAMINGO_halo_redshift_values.txt"


def redshift_satellite_fraction(cat, redshift_root: Path, box: str, isim: str, lc: int):
    path = redshift_midpoints_path(redshift_root, box, isim, lc)
    redshift_bins = np.loadtxt(path, usecols=(0, 1, 2, 3)) if box != 'L1000N3600' else np.loadtxt(path, usecols=(0, 1, 2, 3))[:-1]
    redshift_midpoints = redshift_bins[:, 2]

    redshift_hist = []
    redshift_hist_satellites = []

    for z in redshift_bins[:, 0]:
        snapnum = len(redshift_bins[:, 0]) - z
        total_z = cat[cat["SnapNum"] == snapnum]
        satellites_z = total_z[total_z["Structuretype"] == SATELLITE_FLAG]

        if len(total_z) == 0:
            break

        redshift_hist.append(len(total_z))
        redshift_hist_satellites.append(len(satellites_z))

    total = np.asarray(redshift_hist, dtype=float)
    sat = np.asarray(redshift_hist_satellites, dtype=float)
    frac, err = fraction_and_error(sat, total)
    return redshift_midpoints[:len(frac)], frac, err


def plot_fraction(ax, x: np.ndarray, frac: np.ndarray, err: np.ndarray, label: str, colour: str) -> None:
    ax.errorbar(x, frac, yerr=err, color=colour, fmt="o", markersize=2, linewidth=0.8, capsize=0, label=label)


def simulation_entries(box: str, lc: int, include_resolutions: bool) -> list[dict]:
    entries = []

    for sim, label, colour in zip(SIMS, SIM_NAMES, SIM_COLOURS):
        entries.append({
            "box": box,
            "isim": sim,
            "label": label,
            "colour": colour,
            "lightcone": lc,
        })

    if include_resolutions:
        for run in RESOLUTION_RUNS:
            entries.append({
                "box": run["box"],
                "isim": run["isim"],
                "label": run["label"],
                "colour": run["colour"],
                "lightcone": lc if run["lightcone"] is None else run["lightcone"],
            })

    return entries


def plot_simulation_fraction_panels(axs, data_root: Path, redshift_root: Path, box: str, iz: str, lc: int, ncpu: int, mass_column: str, n_bins: int, include_resolutions: bool, col: int) -> None:
    
    entries = simulation_entries(box=box, lc=lc, include_resolutions=include_resolutions)

    for entry in entries:

        try:
            cat = load_catalogue(data_root, entry["box"], entry["isim"], iz, entry["lightcone"], ncpu)
            x_mass, frac_mass, err_mass = mass_satellite_fraction(cat, mass_column, n_bins)
            plot_fraction(axs[0, col], x_mass, frac_mass, err_mass, entry["label"], colour=entry["colour"])
            x_z, frac_z, err_z = redshift_satellite_fraction(cat, redshift_root, entry["box"], entry["isim"], entry["lightcone"])
            plot_fraction(axs[1, col], x_z, frac_z, err_z, entry["label"], colour=entry["colour"])

        except FileNotFoundError as err:
            print(f"[WARN] {err}. Skipping {entry['label']}/{iz}.")
            continue


def configure_axes(axs, mass_axis: str) -> None:

    for ax in axs.flat:
        ax.tick_params(axis="both", which="both", labelsize=8)
    #     ax.set_title("")

    mass_label = "Halo Mass" if mass_axis == "halo" else "Stellar Mass"
    axs[0, 0].set_ylabel("Satellite Fraction")
    axs[1, 0].set_ylabel("Satellite Fraction")
    axs[0, 0].set_xlabel(rf"{mass_label} [$\log_{{10}}M_\odot$]")
    axs[0, 1].set_xlabel(rf"{mass_label} [$\log_{{10}}M_\odot$]")
    axs[1, 0].set_xlabel("$z$")
    axs[1, 1].set_xlabel("$z$")

    if mass_axis == "halo":
        axs[0, 0].set_xlim(left=12.0)
        axs[0, 1].set_xlim(left=12.0)
        axs[0, 0].set_ylim(0.0, 1.0)
        axs[0, 1].set_ylim(0.0, 1.0)
    else:
        axs[0, 0].set_xlim(10.5, 12.5)
        axs[0, 1].set_xlim(10.5, 12.5)
        axs[0, 0].set_ylim(0.0, 0.45)
        axs[0, 1].set_ylim(0.0, 0.45)

    axs[1, 0].set_ylim(0.0, 0.55)
    axs[1, 1].set_ylim(0.0, 0.55)


def make_plot(args) -> Path:
    data_root = Path(args.data_root)
    redshift_root = Path(args.redshift_root)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    set_plot_style(8)

    fig, axs = plt.subplots(2, 2, figsize=(7.2, 5.8))

    reference_halo_mass = np.loadtxt("./data_files/satellite_fraction_full_lightcone/L1000N1800/HYDRO_FIDUCIAL/lightcone0/satellite_fraction_vs_halo_mass_mass_cut.txt")
    reference_stellar_mass = np.loadtxt("./data_files/satellite_fraction_full_lightcone/L1000N1800/HYDRO_FIDUCIAL/lightcone0/satellite_fraction_vs_stellar_mass_mass_cut.txt")
    reference_redshift = np.loadtxt("./data_files/satellite_fraction_full_lightcone/L1000N1800/HYDRO_FIDUCIAL/lightcone0/satellite_fraction_vs_redshift_mass_cut.txt")

    # Columns:
    # 0 = left bin edge
    # 1 = right bin edge
    # 2 = bin centre
    # 3 = N_total
    # 4 = N_satellite
    # 5 = satellite fraction
    # 6 = binomial error

    if args.mass_axis == "halo":
        axs[0, 0].plot(reference_halo_mass[:,2], reference_halo_mass[:,5], color="0.5", alpha=0.8, label="Full lightcone")
        axs[0, 1].plot(reference_halo_mass[:,2], reference_halo_mass[:,5], color="0.5", alpha=0.8, label="Full lightcone")
    if args.mass_axis == "stellar":
        axs[0, 0].plot(reference_stellar_mass[:,2], reference_stellar_mass[:,5], color="0.5", alpha=0.8, label="Full lightcone")
        axs[0, 1].plot(reference_stellar_mass[:,2], reference_stellar_mass[:,5], color="0.5", alpha=0.8, label="Full lightcone")
    axs[1, 0].plot(reference_redshift[:,2], reference_redshift[:,5], color="0.5", alpha=0.8, label="Full lightcone")
    axs[1, 1].plot(reference_redshift[:,2], reference_redshift[:,5], color="0.5", alpha=0.8, label="Full lightcone")

    # axs[0, 0].fill_between(reference_halo_mass[:,2], reference_halo_mass[:,5] - reference_halo_mass[:,6], reference_halo_mass[:,5] + reference_halo_mass[:,6], color="0.8", alpha=0.3, linewidth=0)
    # axs[0 ,1].fill_between(reference_halo_mass[:,2], reference_halo_mass[:,5] - reference_halo_mass[:,6], reference_halo_mass[:,5] + reference_halo_mass[:,6], color="0.8", alpha=0.3, linewidth=0)
    # axs[1, 0].fill_between(reference_redshift[:,2], reference_redshift[:,5] - reference_redshift[:,6], reference_redshift[:,5] + reference_redshift[:,6], color="0.8", alpha=0.3, linewidth=0)
    # axs[1, 1].fill_between(reference_redshift[:,2], reference_redshift[:,5] - reference_redshift[:,6], reference_redshift[:,5] + reference_redshift[:,6], color="0.8", alpha=0.3, linewidth=0)


    mass_column = "mvir" if args.mass_axis == "halo" else "mstar"
    mass_label = "Halo mass" if args.mass_axis == "halo" else "Stellar mass"

    for col, iz in enumerate(SAMPLES):

        if args.mode == "single":
            cat = load_catalogue(data_root, args.box, args.isim, iz, args.lightcone, args.ncpu)
            x_mass, frac_mass, err_mass = mass_satellite_fraction(cat, mass_column, args.n_bins)
            plot_fraction(axs[0, col], x_mass, frac_mass, err_mass, SIM_NAMES[int(np.where((s == args.isim) for s in SIMS)[0])], colour=f"{iz.lower()}")
            x_z, frac_z, err_z = redshift_satellite_fraction(cat, redshift_root, args.box, args.isim, args.lightcone)
            plot_fraction(axs[1, col], x_z, frac_z, err_z, SIM_NAMES[int(np.where((s == args.isim) for s in SIMS)[0])], colour=f"{iz.lower()}")

        elif args.mode == "simulations":
            plot_simulation_fraction_panels(axs=axs, data_root=data_root, redshift_root=redshift_root, box=args.box, iz=iz, lc=args.lightcone, ncpu=args.ncpu, mass_column=mass_column, n_bins=args.n_bins, include_resolutions=args.include_resolutions, col=col)
        
        else:
            raise ValueError(f"Unknown mode: {args.mode}")

    configure_axes(axs, args.mass_axis)

    axs[0, 0].legend(title="Blue sample", loc="best", frameon=False)
    axs[0, 1].legend(title="Green sample", loc="best", frameon=False)
    axs[1, 0].legend(title="Blue sample", loc="best", frameon=False)
    axs[1, 1].legend(title="Green sample", loc="best", frameon=False)

    suffix = args.output_format.lower().lstrip(".")
    mode_tag = args.mode
    if args.mode == "simulations" and args.include_resolutions:
        mode_tag = "simulations_with_resolutions"

    outpath = outdir / f"satellite_fraction_overview_{args.mass_axis}_mass_{mode_tag}.{suffix}"
    fig.savefig(outpath, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    return outpath


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create 4-panel satellite-fraction overview plot.")
    parser.add_argument("ncpu", type=int, help="Number of CPUs passed to patchyScreening.")
    parser.add_argument("box", help="Box name, e.g. L1000N1800.")
    parser.add_argument("isim", help="Simulation name, e.g. HYDRO_FIDUCIAL.")
    parser.add_argument("lightcone", type=int, help="Lightcone number.")
    parser.add_argument("--mode", choices=("single", "simulations"), default="single", help="single: plot one simulation; simulations: compare multiple simulations/resolutions.")
    parser.add_argument("--include-resolutions", action="store_true", help="In simulations mode, also plot L1_m8 and L2p8_m9. L2p8_m9 uses lightcone 0.")
    parser.add_argument("--mass-axis", choices=("halo", "stellar"), default="halo", help="Top-row mass variable.")
    parser.add_argument("--n-bins", type=int, default=100)
    parser.add_argument("--output-format", choices=("png", "pdf"), default="pdf")
    parser.add_argument("--dpi", type=int, default=400)
    parser.add_argument("--data-root", default="./data_files")
    parser.add_argument(
        "--redshift-root",
        default="/cosma8/data/dp004/dc-conl1/FLAMINGO/patchy_screening/data_files/halo_redshifts",
    )
    parser.add_argument("--outdir", default="./Plots")
    return parser.parse_args()


if __name__ == "__main__":
    output = make_plot(parse_args())
    print(f"Saved {output}")
