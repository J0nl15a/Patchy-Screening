#!/usr/bin/env python3
"""
Flexible 2-column tau-profile comparison plots.

Two modes are supported:

1. sims
   Left  = Blue sample
   Right = Green sample

   Plot a user-selected set of FLAMINGO simulations, all using the
   same amplitude A and slope s.

2. params
   Plot one selected galaxy sample (Blue OR Green) for the
   L1000N1800 / HYDRO_FIDUCIAL simulation.

   Left  = varying amplitude at fixed slope
   Right = varying slope at fixed amplitude

Each column has a main tau-profile panel and a difference panel
underneath:

    (tau - tau_ref) x 10^4

Observed unWISE tau profiles are also plotted.

The top x-axis shows transverse distance r [Mpc/h].
"""

import argparse, os, pickle
from pathlib import Path

import numpy as np, pylab as pb


# ============================================================
# Plotting style
# ============================================================

pb.rc("text", usetex=True)
pb.rc("font", family="serif", size=11)
pb.rcParams["font.size"] = 11


# ============================================================
# FLAMINGO simulation labels / colours
# ============================================================

SIM_LABELS = {
    "HYDRO_FIDUCIAL": r"L1\_m9",
    "HYDRO_LOW_SIGMA8_STRONGEST_AGN": r"LS8\_fgas$-8\sigma$",
    "HYDRO_LOW_SIGMA8": r"LS8",
    "HYDRO_PLANCK_LARGE_NU_FIXED": r"PlanckNu0p24Fix",
    "HYDRO_PLANCK_LARGE_NU_VARY": r"PlanckNu0p24Var",
    "HYDRO_PLANCK": r"Planck",
    "HYDRO_STRONGEST_AGN": r"fgas$-8\sigma$",
    "HYDRO_STRONGER_AGN": r"fgas$-4\sigma$",
    "HYDRO_STRONG_AGN": r"fgas$-2\sigma$",
    "HYDRO_WEAK_AGN": r"fgas$+2\sigma$",
    "HYDRO_STRONG_JETS_published": r"Jet\_fgas$-4\sigma$",
    "HYDRO_JETS_published": r"Jet",
    "HYDRO_STRONG_SUPERNOVA": r"M*-$\sigma$",
}

SIM_COLORS = {
    "HYDRO_FIDUCIAL": "#117733",
    "HYDRO_LOW_SIGMA8_STRONGEST_AGN": "#7B68EE",
    "HYDRO_LOW_SIGMA8": "#882255",
    "HYDRO_PLANCK_LARGE_NU_FIXED": "#999933",
    "HYDRO_PLANCK_LARGE_NU_VARY": "#AA4499",
    "HYDRO_PLANCK": "#44AA99",
    "HYDRO_STRONGEST_AGN": "#105ba4",
    "HYDRO_STRONGER_AGN": "#3787c0",
    "HYDRO_STRONG_AGN": "#6aaed6",
    "HYDRO_WEAK_AGN": "#abd0e6",
    "HYDRO_STRONG_JETS_published": "#55E18E",
    "HYDRO_JETS_published": "#7EFF4B",
    "HYDRO_STRONG_SUPERNOVA": "#FF8C40",
}


# ============================================================
# General helpers
# ============================================================

def name_float(x, mle=False):
    """Convert 10.8 -> 10p8 for filenames."""
    if mle:
        return f"{float(x):.3f}".replace(".", "p")
    else:
        return f"{float(x):.1f}".replace(".", "p")


def parse_csv(value):
    if value is None or str(value).strip() == "":
        return []

    return [v.strip() for v in str(value).split(",") if v.strip()]


def parse_float_spec(value, default=None):
    """
    Accept either:

        10.5,10.6,10.7

    or an inclusive range:

        10.5:11.0:0.1
    """

    if value is None:
        return default if default is not None else []

    value = str(value).strip()

    if ":" in value:
        start, stop, step = map(float, value.split(":"))

        n = int(np.floor((stop - start) / step + 0.5)) + 1

        return [round(start + i * step, 10) for i in range(n)]

    return [float(v) for v in parse_csv(value)]


# ============================================================
# Tau-profile paths / loading
# ============================================================

def tau_path(box, sim, sample, lightcone, amp, slope, nside=8192, cmb="unlensed", no_ps=False):
    amp_name = name_float(amp)
    slope_name = name_float(slope)

    ps_suffix = "_no_ps" if no_ps else ""

    filename = (f"tau_Mstar_bin{amp_name}_{slope_name}_nside{nside}_FITS_{cmb}{ps_suffix}.pickle")

    return Path("./data_files/tau_profiles") / box / sim / sample / f"lightcone{lightcone}" / filename


def load_tau_profile(box, sim, sample, lightcone, amp, slope, nside=8192, cmb="unlensed", no_ps=False):
    path = tau_path(box=box, sim=sim, sample=sample, lightcone=lightcone, amp=amp, slope=slope, nside=nside, cmb=cmb, no_ps=no_ps)

    if not path.exists():
        raise FileNotFoundError(f"Tau profile does not exist:\n{path}")

    print(f"Loading: {path}")

    with open(path, "rb") as f:
        data = pickle.load(f)

    theta = np.asarray(data[0], dtype=float)
    tau = np.asarray(data[1], dtype=float)
    distance = np.asarray(data[2], dtype=float)

    return {"theta": theta, "tau": tau, "distance": distance, "path": path}


# ============================================================
# Observed tau profiles
# ============================================================

def load_observations(sample):

    sample_lower = sample.lower()

    path = Path(f"./data_files/tau_profiles/digitized_obs_data_{sample_lower}.txt")

    data = np.loadtxt(path)

    theta = data[:, 0]
    tau = data[:, 1]

    # These are absolute upper/lower y-values,
    # not error magnitudes.
    upper = data[:, 2]
    lower = data[:, 3]

    error = np.vstack((tau - lower, upper - tau))

    return {"theta": theta, "tau": tau, "error": error}


def add_observations(ax, ax_ratio, sample, reference, show=True):
    if not show:
        return None

    obs = load_observations(sample)

    # Observed tau data are already stored at the
    # plotted tau x 10^4 scale.
    handle = ax.errorbar(obs["theta"], obs["tau"], yerr=obs["error"], fmt="o", markersize=4, color="k", ecolor="k", 
                         capsize=2, linewidth=1.0, label="Coulton et al. 2025", zorder=10)

    # Reference profile needs converting to tau x 10^4.
    fid_at_obs = np.interp(obs["theta"], reference["theta"], reference["tau"] * 1.0e4)

    ax_ratio.errorbar(obs["theta"], obs["tau"] - fid_at_obs, yerr=obs["error"], fmt="o", markersize=4, color="k", ecolor="k", 
                      capsize=2, linewidth=1.0, zorder=10)

    return handle


# ============================================================
# Top distance axis
# ============================================================

def add_distance_axis(ax, theta, distance):
    """
    Add r [Mpc/h] as a top x-axis.

    The distance array is taken from the corresponding tau
    profile pickle.
    """

    ax_distance = ax.twiny()

    theta_limits = ax.get_xlim()

    distance_limits = np.interp(theta_limits, theta, distance)

    ax_distance.set_xlim(distance_limits)

    ax_distance.set_xlabel(r"$r\ [{\rm Mpc}/h]$")

    return ax_distance


# ============================================================
# Plot one column
# ============================================================

def plot_panel(ax, ax_ratio, entries, reference, sample, legend_title, args):
    model_handles = []

    ax.axhline(0.0, color="k", linewidth=0.8)
    ax_ratio.axhline(0.0, color="k", linestyle="--", linewidth=0.8, alpha=0.6)

    # --------------------------------------------------------
    # Profiles
    # --------------------------------------------------------

    for entry in entries:

        profile = entry["profile"]

        theta = profile["theta"]
        tau = profile["tau"]

        line, = ax.plot(theta, tau * 1.0e4, color=entry["color"], lw=1.3, alpha=0.9, label=entry["label"])

        model_handles.append(line)

        reference_at_theta = np.interp(theta, reference["theta"], reference["tau"])

        ax_ratio.plot(theta, (tau - reference_at_theta) * 1.0e4, color=entry["color"], lw=1.3, alpha=0.9)

    # --------------------------------------------------------
    # Observed data
    # --------------------------------------------------------

    obs_handle = add_observations(ax=ax, ax_ratio=ax_ratio, sample=sample, reference=reference, show=not args.no_observations)

    # --------------------------------------------------------
    # Axes
    # --------------------------------------------------------

    ax.set_xlim(args.xmin, args.xmax)
    ax.set_ylim(args.ymin, args.ymax)

    ax_ratio.set_xlim(args.xmin, args.xmax,)
    ax_ratio.set_ylim(args.ratio_ymin, args.ratio_ymax)

    if args.grid:
        ax.grid(alpha=0.25)
        ax_ratio.grid(alpha=0.25)

    # --------------------------------------------------------
    # Legend
    # --------------------------------------------------------

    # model_legend = ax.legend(handles=model_handles, loc=args.legend_loc, title=legend_title, frameon=False, 
    #                          fontsize=args.legend_fontsize, title_fontsize=args.legend_fontsize, ncol=args.legend_ncol)

    # # Keep observations separate from model legend.
    # if obs_handle is not None:

    #     obs_legend = ax.legend(handles=[obs_handle], labels=["Coulton et al. 2025"], loc=args.obs_legend_loc, frameon=False, 
    #                            fontsize=args.legend_fontsize)

    #     ax.add_artist(model_legend)

    legend_handles = model_handles.copy()

    if obs_handle is not None:
        legend_handles.append(obs_handle)

    ax.legend(handles=legend_handles, loc=args.legend_loc, title=legend_title, frameon=False, fontsize=args.legend_fontsize, title_fontsize=args.legend_fontsize, ncol=args.legend_ncol)

    # --------------------------------------------------------
    # Distance axis
    # --------------------------------------------------------

    add_distance_axis(ax, reference["theta"], reference["distance"])


# ============================================================
# SIMULATION MODE
# ============================================================

def make_sim_entries(args, sample):
    sims = parse_csv(args.sims)

    if len(sims) == 0:
        raise ValueError("--sims must contain at least one simulation when --mode sims.")

    # Sample-dependent catalogue parameters.
    if sample == "Blue":
        amp = args.blue_amp
        slope = args.blue_slope

    elif sample == "Green":
        amp = args.green_amp
        slope = args.green_slope

    else:
        raise ValueError("sample must be 'Blue' or 'Green'")

    entries = []

    for sim in sims:

        profile = load_tau_profile(box=args.box, sim=sim, sample=sample, lightcone=args.lc, amp=amp, slope=slope, nside=args.nside, 
                                   cmb=args.cmb, no_ps=args.no_ps)

        entries.append({"sim": sim,
                        "label": SIM_LABELS.get(sim, sim),
                        "color": SIM_COLORS.get(sim, None),
                        "profile": profile})

    return entries


def find_sim_reference(entries, args):
    """
    Reference for the difference panel.

    Default: L1000N1800 / HYDRO_FIDUCIAL if present.
    Otherwise use --ref-index.
    """

    if args.ref_index is not None:
        return entries[args.ref_index]["profile"]

    for entry in entries:
        if entry["sim"] == "HYDRO_FIDUCIAL":
            return entry["profile"]

    print("WARNING: HYDRO_FIDUCIAL is not in --sims. "
        "Using the first simulation as the reference.")

    return entries[0]["profile"]


def make_sims_figure(args):

    fig, axes = pb.subplots(2, 2, figsize=(9.0, 5.3), sharex="col", sharey="row", 
                            gridspec_kw={"height_ratios": [3.0, 1.0], "hspace": 0.06, "wspace": 0.08}, dpi=args.dpi)

    # --------------------------------------------------------
    # Blue
    # --------------------------------------------------------

    blue_entries = make_sim_entries(args, "Blue")
    blue_reference = find_sim_reference(blue_entries, args,)

    blue_legend_title = (rf"Blue sample, $A={args.blue_amp:.1f}$, $s={args.blue_slope:.1f}$")

    plot_panel(ax=axes[0, 0], ax_ratio=axes[1, 0], entries=blue_entries, reference=blue_reference, sample="Blue", 
               legend_title=blue_legend_title, args=args)

    # --------------------------------------------------------
    # Green
    # --------------------------------------------------------

    green_entries = make_sim_entries(args, "Green")
    green_reference = find_sim_reference(green_entries, args)

    green_legend_title = (rf"Green sample, $A={args.green_amp:.1f}$, $s={args.green_slope:.1f}$")

    plot_panel(ax=axes[0, 1], ax_ratio=axes[1, 1], entries=green_entries, reference=green_reference, sample="Green", 
               legend_title=green_legend_title, args=args)

    # --------------------------------------------------------
    # Shared labels
    # --------------------------------------------------------

    axes[0, 0].set_ylabel(r"$\tau \times 10^4$")
    axes[1, 0].set_ylabel(r"$(\tau-\tau_{\rm ref})\times10^4$")
    axes[1, 0].set_xlabel("Annulus centre (arcmin)")
    axes[1, 1].set_xlabel("Annulus centre (arcmin)")

    # No duplicate y labels/tick labels on right.
    axes[0, 1].tick_params(axis="y", labelleft=False)
    axes[1, 1].tick_params(axis="y", labelleft=False,)

    save_figure(fig, args, default_name=(f"tau_1D_profile_custom_sims_Blue_Green"))


# ============================================================
# PARAMETER MODE
# ============================================================

def make_parameter_entries(args, sample, amps, slopes, vary):
    entries = []

    if vary == "amp":

        values = amps

        cmap = pb.get_cmap(args.cmap)

        for i, amp in enumerate(values):

            profile = load_tau_profile(box="L1000N1800", sim="HYDRO_FIDUCIAL", sample=sample, lightcone=args.lc, amp=amp, 
                                       slope=args.slope, nside=args.nside, cmb=args.cmb, no_ps=args.no_ps)

            entries.append({"label": rf"$A={amp:.1f}$",
                            "color": cmap(i / max(1, len(values) - 1)),
                            "amp": amp,
                            "slope": args.slope,
                            "profile": profile})

    elif vary == "slope":

        values = slopes

        cmap = pb.get_cmap(args.cmap)

        for i, slope in enumerate(values):

            profile = load_tau_profile(box="L1000N1800", sim="HYDRO_FIDUCIAL", sample=sample, lightcone=args.lc, amp=args.amp, 
                                       slope=slope, nside=args.nside, cmb=args.cmb, no_ps=args.no_ps)

            entries.append({"label": rf"$s={slope:.1f}$",
                            "color": cmap(i / max(1, len(values) - 1)),
                            "amp": args.amp,
                            "slope": slope,
                            "profile": profile})

    else:
        raise ValueError("vary must be 'amp' or 'slope'")

    return entries


def load_parameter_reference(args, sample):
    """
    Common reference profile for BOTH parameter panels.

    By default this is explicitly selected with
    --ref-amp and --ref-slope.
    """

    return load_tau_profile(box="L1000N1800", sim="HYDRO_FIDUCIAL", sample=sample, lightcone=args.lc, amp=args.ref_amp, 
                            slope=args.ref_slope, nside=args.nside, cmb=args.cmb, no_ps=args.no_ps)


def make_params_figure(args):

    sample = args.sample

    amps = parse_float_spec(args.amps, default=[args.amp])
    slopes = parse_float_spec(args.slopes, default=[args.slope])

    fig, axes = pb.subplots(2, 2, figsize=(9.0, 5.3), sharex="col", sharey="row", 
                            gridspec_kw={"height_ratios": [3.0, 1.0], "hspace": 0.06, "wspace": 0.08}, dpi=args.dpi)

    # Common reference for both ratio panels.
    reference = load_parameter_reference(args, sample)

    # --------------------------------------------------------
    # Left: varying amplitude
    # --------------------------------------------------------

    amp_entries = make_parameter_entries(args=args, sample=sample, amps=amps, slopes=slopes, vary="amp")
    amp_legend_title = (rf"{sample} sample, $s={args.slope:.1f}$")

    plot_panel(ax=axes[0, 0], ax_ratio=axes[1, 0], entries=amp_entries, reference=reference, sample=sample, 
               legend_title=amp_legend_title, args=args)

    # --------------------------------------------------------
    # Right: varying slope
    # --------------------------------------------------------

    slope_entries = make_parameter_entries(args=args, sample=sample, amps=amps, slopes=slopes, vary="slope")
    slope_legend_title = (rf"{sample} sample, $A={args.amp:.1f}$")

    plot_panel(ax=axes[0, 1], ax_ratio=axes[1, 1], entries=slope_entries, reference=reference, sample=sample, 
               legend_title=slope_legend_title, args=args)

    # --------------------------------------------------------
    # Shared labels
    # --------------------------------------------------------

    axes[0, 0].set_ylabel(r"$\tau \times 10^4$")
    axes[1, 0].set_ylabel(r"$(\tau-\tau_{\rm ref})\times10^4$")
    axes[1, 0].set_xlabel("Annulus centre (arcmin)")
    axes[1, 1].set_xlabel("Annulus centre (arcmin)")

    axes[0, 1].tick_params(axis="y", labelleft=False)
    axes[1, 1].tick_params(axis="y", labelleft=False)

    save_figure(fig, args, default_name=(f"tau_1D_profile_params_{sample}_amps_slopes"))


# ============================================================
# Saving
# ============================================================

def save_figure(fig, args, default_name):

    Path("./Plots").mkdir(exist_ok=True)

    if args.output is None:
        out = Path("./Plots") / f"{default_name}.{args.file}"

    else:
        out = Path(args.output)

    pb.savefig(out, dpi=args.dpi, bbox_inches="tight",)
    pb.close(fig)

    print(f"Saved {out}")


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser(description=("Flexible two-column FLAMINGO tau-profile plotter."))

    # --------------------------------------------------------
    # Plot mode
    # --------------------------------------------------------

    parser.add_argument("--mode", choices=["sims", "params"], required=True, help=("sims: Blue vs Green for selected simulations. "
                                                                                   "params: amplitude/slope scans for one sample."))
    parser.add_argument("--sample", choices=["Blue", "Green"], default="Blue", help=("Galaxy sample used in --mode params. "
                                                                                     "--mode sims always plots both Blue and Green."))

    # --------------------------------------------------------
    # Simulation selection
    # --------------------------------------------------------

    parser.add_argument("--box", default="L1000N1800")
    parser.add_argument("--sims", default="HYDRO_FIDUCIAL", help=("Comma-separated simulation list used in --mode sims."))
    parser.add_argument("--lc", type=int, default=0)

    # --------------------------------------------------------
    # Catalogue parameters
    # --------------------------------------------------------

    parser.add_argument("--amp", type=float, default=10.8, 
                        help=("Fixed amplitude. In sims mode this is used for all simulations. In params mode it is the fixed amplitude in the right-hand slope panel."))
    parser.add_argument("--slope", type=float, default=0.6, 
                        help=("Fixed slope. In sims mode this is used for all simulations. In params mode it is the fixed slope in the left-hand amplitude panel."))
    parser.add_argument("--amps", default="10.4:11.2:0.2", 
                        help=("Amplitude values for the left panel of params mode. Comma list or start:stop:step."))
    parser.add_argument("--slopes", default="0.0:1.2:0.2", 
                        help=("Slope values for the right panel of params mode. Comma list or start:stop:step."))
    parser.add_argument("--blue-amp", type=float, default=10.8, help="Amplitude used for the Blue sample in sims mode.")
    parser.add_argument("--blue-slope", type=float, default=0.6, help="Slope used for the Blue sample in sims mode.")
    parser.add_argument("--green-amp", type=float, default=10.8, help="Amplitude used for the Green sample in sims mode.")
    parser.add_argument("--green-slope", type=float, default=0.6, help="Slope used for the Green sample in sims mode.")

    # Explicit ratio-panel reference.
    parser.add_argument("--ref-amp", type=float, default=10.8, help=("Amplitude of the reference tau profile in params mode."))
    parser.add_argument("--ref-slope", type=float, default=0.6, help=("Slope of the reference tau profile in params mode."))
    parser.add_argument("--ref-index", type=int, default=None, 
                        help=("Reference simulation index in sims mode. By default HYDRO_FIDUCIAL is used when present."))

    # --------------------------------------------------------
    # Tau-profile options
    # --------------------------------------------------------

    parser.add_argument("--nside", type=int, default=8192)
    parser.add_argument("--cmb", choices=["unlensed", "lensed_z2", "lensed_z3"], default="unlensed")
    parser.add_argument("--no-ps", action="store_true", help="Load the _no_ps tau profiles.")
    parser.add_argument("--no-observations", action="store_true")

    # --------------------------------------------------------
    # Parameter-scan colours
    # --------------------------------------------------------

    parser.add_argument("--cmap", default="viridis", help=("Colour map used for amplitude/slope scans."))

    # --------------------------------------------------------
    # Axis settings
    # --------------------------------------------------------

    parser.add_argument("--xmin", type=float, default=0.0)
    parser.add_argument("--xmax", type=float, default=11.0)
    parser.add_argument("--ymin", type=float, default=None,)
    parser.add_argument("--ymax", type=float, default=None)
    parser.add_argument("--ratio-ymin", type=float, default=None)
    parser.add_argument("--ratio-ymax", type=float, default=None)
    parser.add_argument("--grid", action="store_true", help="Show grid lines.")

    # --------------------------------------------------------
    # Legend settings
    # --------------------------------------------------------

    parser.add_argument("--legend-loc", default="best")
    parser.add_argument("--legend-fontsize", type=float, default=9)
    parser.add_argument("--legend-ncol", type=int, default=1)

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    parser.add_argument("--file", choices=["png", "pdf"], default="pdf")
    parser.add_argument("--output", default=None)
    parser.add_argument("--dpi", type=int, default=400)
    args = parser.parse_args()

    # --------------------------------------------------------
    # Make requested plot
    # --------------------------------------------------------

    if args.mode == "sims":
        make_sims_figure(args)

    elif args.mode == "params":
        make_params_figure(args)


if __name__ == "__main__":
    main()