#!/usr/bin/env python3
import argparse
import pickle
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

plt.rc("text", usetex=True)
plt.rc("font", family="serif", size=8)
plt.rcParams["font.size"] = 8

GROUPS = {
    "resolution": {"title": "Resolution"},
    "cosmology": {
        "title": "Cosmology", "box": "L1000N1800",
        "sims": ["HYDRO_FIDUCIAL", "HYDRO_PLANCK", "HYDRO_PLANCK_LARGE_NU_VARY", "HYDRO_PLANCK_LARGE_NU_FIXED", "HYDRO_LOW_SIGMA8", "HYDRO_LOW_SIGMA8_STRONGEST_AGN"],
        "names": [r"L1\_m9", "Planck", "PlanckNu0p24Var", "PlanckNu0p24Fix", "LS8", r"LS8\_fgas$-8\sigma$"],
        "colors": ["#117733", "#44AA99", "#AA4499", "#999933", "#882255", "#7B68EE"],
    },
    "agn_feedback": {
        "title": "AGN feedback", "box": "L1000N1800",
        "sims": ["HYDRO_FIDUCIAL", "HYDRO_WEAK_AGN", "HYDRO_STRONG_AGN", "HYDRO_STRONGER_AGN", "HYDRO_STRONGEST_AGN"],
        "names": [r"L1\_m9", r"fgas$+2\sigma$", r"fgas$-2\sigma$", r"fgas$-4\sigma$", r"fgas$-8\sigma$"],
        "colors": ["#117733", "#abd0e6", "#6aaed6", "#3787c0", "#105ba4"],
    },
    "other_feedback": {
        "title": "Other feedback", "box": "L1000N1800",
        "sims": ["HYDRO_FIDUCIAL", "HYDRO_STRONG_SUPERNOVA", "HYDRO_JETS_published", "HYDRO_STRONG_JETS_published"],
        "names": [r"L1\_m9", r"$M^*$-$\sigma$", "Jet", r"Jet\_fgas$-4\sigma$"],
        "colors": ["#117733", "#FF8C40", "#7EFF4B", "#55E18E"],
    },
}

def tau_path(base_dir, box, sim, sample, lightcone, nside, primary_method, file_method, no_ps):
    suffix = "_no_ps" if no_ps else ""
    if box == 'L2800N5040':
        nside = 4096
    filename = f"tau_mle_catalogue_nside{nside}_{primary_method}_{file_method}{suffix}.pickle"
    return Path(base_dir) / box / sim / sample / f"lightcone{lightcone}" / filename

def load_tau_profile(path):
    if not path.exists():
        raise FileNotFoundError(f"Tau profile not found: {path}")
    with path.open("rb") as f:
        data = pickle.load(f)
    theta = np.asarray(data[0], dtype=float)
    tau = np.asarray(data[1], dtype=float)
    distance = np.asarray(data[2], dtype=float) if len(data) > 2 else None
    if theta.shape != tau.shape:
        raise ValueError(f"theta shape {theta.shape} != tau shape {tau.shape} in {path}")
    return theta, tau, distance

def load_named_profile(args, box, sim, lightcone=0):
    path = tau_path(args.base_dir, box, sim, args.sample, lightcone, args.nside, args.primary_method, args.file_method, args.no_ps)
    theta, tau, distance = load_tau_profile(path)
    return {"theta": theta, "tau": tau, "distance": distance, "path": path}

def put_on_reference_grid(theta_ref, theta, values):
    if theta.shape == theta_ref.shape and np.allclose(theta, theta_ref):
        return values

    raise ValueError("Tau profiles use different radial-bin grids.")

def stable_fractional_difference(numerator, reference, absolute_threshold=1e-12):
    numerator = np.asarray(numerator, dtype=float)
    reference = np.asarray(reference, dtype=float)

    scale = np.nanmax(np.abs(reference))

    if not np.isfinite(scale) or scale <= absolute_threshold:
        raise ValueError("Reference tau profile has no usable amplitude.")

    result = (numerator - reference) / scale

    result[~np.isfinite(numerator) | ~np.isfinite(reference)] = np.nan

    return result

def symmetric_ratio(numerator, reference, threshold=1e-12):
    numerator = np.asarray(numerator, dtype=float)
    reference = np.asarray(reference, dtype=float)

    denominator = (np.abs(numerator) + np.abs(reference))

    valid = (np.isfinite(numerator) & np.isfinite(reference) & (denominator > threshold))

    result = np.full_like(numerator, np.nan)

    result[valid] = (2.0 * (numerator[valid] - reference[valid]) / denominator[valid])

    return result

def plot_resolution(ax, ax_ratio, args):
    fid = load_named_profile(args, "L1000N1800", "HYDRO_FIDUCIAL", 0)
    hires = load_named_profile(args, "L1000N3600", "HYDRO_FIDUCIAL", 0)
    l2800_profiles = [load_named_profile(args, "L2800N5040", "HYDRO_FIDUCIAL", lc) for lc in range(args.lc_count)]

    theta = fid["theta"]
    tau_fid = fid["tau"]
    tau_hires = put_on_reference_grid(theta, hires["theta"], hires["tau"])
    l2800_stack = np.asarray([put_on_reference_grid(theta, p["theta"], p["tau"]) for p in l2800_profiles])
    # for i in range(8):
    #     print(f"lc= {i}, {l2800_profiles[i]}")
    #     print("\n")
    # quit()
    mean = np.mean(l2800_stack, axis=0)
    lo = np.min(l2800_stack, axis=0)
    hi = np.max(l2800_stack, axis=0)

    ax.plot(theta, tau_fid, color="#117733", label=r"L1\_m9")
    ax.plot(theta, tau_hires, color="#CC6677", label=r"L1\_m8")
    ax.plot(theta, mean, color="#332288", label=r"L2p8\_m9")
    ax.fill_between(theta, lo, hi, color="#332288", alpha=0.25, linewidth=0)

    ax_ratio.axhline(0.0, color="k", linestyle="--", linewidth=0.8, alpha=0.7)

    fid_difference = stable_fractional_difference(tau_fid, tau_fid, args.ratio_threshold)
    hires_difference = stable_fractional_difference(tau_hires, tau_fid, args.ratio_threshold)
    mean_difference = stable_fractional_difference(mean, tau_fid, args.ratio_threshold)
    lo_difference = stable_fractional_difference(lo, tau_fid, args.ratio_threshold)
    hi_difference = stable_fractional_difference(hi, tau_fid, args.ratio_threshold)

    ax_ratio.plot(theta, fid_difference, color="#117733")
    ax_ratio.plot(theta, hires_difference, color="#CC6677")
    ax_ratio.plot(theta, mean_difference, color="#332288")
    ax_ratio.fill_between(theta, lo_difference, hi_difference, where=(np.isfinite(lo_difference) & np.isfinite(hi_difference)), 
                          color="#332288", alpha=0.25, linewidth=0)
    
    return fid

def plot_standard_group(ax, ax_ratio, args, group_key):
    group = GROUPS[group_key]
    profiles = []

    for sim, name, color in zip(
        group["sims"],
        group["names"],
        group["colors"],
    ):
        profile = load_named_profile(args, group["box"], sim, 0)
        profiles.append((profile, name, color))

    # Use the first profile in each group as the reference.
    reference = profiles[0][0]
    theta_ref = reference["theta"]
    tau_ref = reference["tau"]

    ax_ratio.axhline(0.0, color="k", linestyle="--", linewidth=0.8, alpha=0.7)

    for profile, name, color in profiles:
        tau = put_on_reference_grid(
            theta_ref,
            profile["theta"],
            profile["tau"],
        )

        ax.plot(theta_ref, tau, color=color, label=name, alpha=0.9)

        difference = stable_fractional_difference(tau, tau_ref, args.ratio_threshold)
        ax_ratio.plot(theta_ref, difference, color=color, alpha=0.9)
        
    return reference

def configure_axis(ax, title, args, show_ylabel=True):
    ax.axhline(0.0, color="k", linewidth=0.8, alpha=0.7)
    ax.set_xlim(args.xmin, args.xmax)
    if args.ymin is not None or args.ymax is not None:
        ax.set_ylim(bottom=args.ymin, top=args.ymax)
    # ax.set_title(title, fontsize=9)
    if show_ylabel:
        ax.set_ylabel(r"Filtered $\tau$")
    else:
        ax.tick_params(axis="y", labelleft=False, left=False)
        ax.yaxis.get_offset_text().set_visible(False)
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((-4, -4))
    ax.yaxis.set_major_formatter(formatter)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(-4, -4))
    ax.legend(fontsize=7, loc="best", frameon=False, title=f"{args.sample} sample")

def add_distance_axis(ax, theta, distance, show_label=True, show_ticklabels=True):
    if distance is None or len(distance) != len(theta):
        return None

    if not (np.all(np.diff(theta) > 0) and np.all(np.diff(distance) > 0)):
        return None

    sec = ax.secondary_xaxis("top", functions=(lambda x: np.interp(x, theta, distance), lambda x: np.interp(x, distance, theta)))

    if show_label:
        sec.set_xlabel(r"$r\,[{\rm Mpc}/h]$")
    else:
        sec.set_xlabel("")

    if not show_ticklabels:
        sec.tick_params(axis="x", which="both", top=False, labeltop=False)

    return sec

def make_overview(args):
    fig, axes = plt.subplots(4, 2, figsize=(10, 9.0), sharex="col", gridspec_kw={"height_ratios": [3.0, 1.0, 3.0, 1.0], "hspace": 0.10, "wspace": 0.08})
    panel_map = {"resolution": axes[0, 0], "cosmology": axes[0, 1], "agn_feedback": axes[2, 0], "other_feedback": axes[2, 1]}
    ratio_map = {"resolution": axes[1, 0], "cosmology": axes[1, 1], "agn_feedback": axes[3, 0], "other_feedback": axes[3, 1]}

    refs = {"resolution": plot_resolution(panel_map["resolution"], ratio_map["resolution"], args)}

    for key in ("cosmology", "agn_feedback", "other_feedback"):
        refs[key] = plot_standard_group(panel_map[key], ratio_map[key], args, key)

    if args.ymin is None:
        shared_ymin = min(ax.dataLim.ymin for ax in panel_map.values())
    else:
        shared_ymin = args.ymin

    if args.ymax is None:
        shared_ymax = max(ax.dataLim.ymax for ax in panel_map.values())
    else:
        shared_ymax = args.ymax

    y_padding = 0.05 * (shared_ymax - shared_ymin)

    if args.ymin is None:
        shared_ymin -= y_padding

    if args.ymax is None:
        shared_ymax += y_padding

    for key, ax in panel_map.items():
        configure_axis(ax, GROUPS[key]["title"], args, show_ylabel=key in ("resolution", "agn_feedback"))
        ax.set_ylim(shared_ymin, shared_ymax)
        if args.distance_axis:
            is_top_panel = key in ("resolution", "cosmology")

            add_distance_axis(ax, refs[key]["theta"], refs[key]["distance"], show_label=is_top_panel, show_ticklabels=is_top_panel)

    fig.canvas.draw()

    for key in ("cosmology", "other_feedback"):
        panel_map[key].yaxis.get_offset_text().set_visible(False)

    for ax in panel_map.values():
        offset_text = ax.yaxis.get_offset_text()
        offset_text.set_x(-0.03)
        offset_text.set_ha("right")

    for key, ax_ratio in ratio_map.items():
        ax_ratio.set_xlim(args.xmin, args.xmax)
        ax_ratio.set_ylim(args.ratio_ymin, args.ratio_ymax)
        ax_ratio.set_ylabel(r"$\Delta\tau/\max|\tau_{\rm fid}|$")
        # ax_ratio.set_ylabel(r"$2(\tau-\tau_{\rm fid})/(|\tau|+|\tau_{\rm fid}|)$")
        ax_ratio.grid(alpha=0.25)

    ratio_map["cosmology"].tick_params(axis="y", labelleft=False)
    ratio_map["other_feedback"].tick_params(axis="y", labelleft=False)

    ratio_map["cosmology"].set_ylabel("")
    ratio_map["other_feedback"].set_ylabel("")

    axes[3, 0].set_xlabel("Annulus centre [arcmin]")
    axes[3, 1].set_xlabel("Annulus centre [arcmin]")

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    suffix = "_no_ps" if args.no_ps else ""
    output = Path(args.output_dir) / f"tau_1D_profile_overview_{args.sample}_nside{args.nside}_{args.primary_method}_{args.file_method}{suffix}.{args.file}"
    fig.savefig(output, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output}")

def main():
    parser = argparse.ArgumentParser(description="Create a four-panel overview of mock-catalogue tau profiles.")
    parser.add_argument("sample", choices=["Blue", "Green"])
    parser.add_argument("--base-dir", default="./data_files/tau_profiles")
    parser.add_argument("--lc-count", type=int, default=8)
    parser.add_argument("--nside", type=int, default=8192)
    parser.add_argument("--primary-method", default="FITS")
    parser.add_argument("--file-method", default="unlensed")
    parser.add_argument("--no-ps", action="store_true")
    parser.add_argument("--distance-axis", action="store_true")
    parser.add_argument("--xmin", type=float, default=0.0)
    parser.add_argument("--xmax", type=float, default=11.0)
    parser.add_argument("--ymin", type=float, default=None)
    parser.add_argument("--ymax", type=float, default=None)
    parser.add_argument("--ratio-ymin", type=float, default=-0.7)
    parser.add_argument("--ratio-ymax", type=float, default=0.7)
    parser.add_argument("--ratio-threshold", type=float, default=1e-24, help=("Do not calculate ratios where the absolute reference tau profile is below this value."))
    parser.add_argument("--file", choices=["png", "pdf"], default="png")
    parser.add_argument("--dpi", type=int, default=400)
    parser.add_argument("--output-dir", default="./Plots")
    make_overview(parser.parse_args())

if __name__ == "__main__":
    main()
