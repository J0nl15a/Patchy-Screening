#!/usr/bin/env python3
"""
Flexible 2-column power-spectrum comparison plots for mock catalogues.

Two modes are supported:
  1. samples : left=Blue, right=Green for one selected spectrum type.
  2. spectra : left=gg, right=kg for one selected sample.

Each column has a main spectrum panel and a ratio panel underneath.
"""

import argparse
from pathlib import Path

import numpy as np
import pylab as pb
import yaml
import pymaster as nmt
from scipy.signal import savgol_filter

pb.rc("text", usetex=True)
pb.rc("font", family="serif", size=8)
pb.rcParams["font.size"] = 8

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

SPECTRA_MODE_SIMS = [
    "HYDRO_FIDUCIAL",
    "HYDRO_STRONG_SUPERNOVA",
    "HYDRO_STRONGEST_AGN",
    "HYDRO_PLANCK",
    "HYDRO_LOW_SIGMA8",
    "HYDRO_LOW_SIGMA8_STRONGEST_AGN",
    # "HYDRO_PLANCK_LARGE_NU_FIXED",
    # "HYDRO_PLANCK_LARGE_NU_VARY",
    # "HYDRO_STRONG_AGN",
    # "HYDRO_WEAK_AGN",
]

SPECTRA_MODE_BOXES = [
    "L1000N1800",
] * len(SPECTRA_MODE_SIMS)

SPECTRA_MODE_LIGHTCONES = [
    0,
] * len(SPECTRA_MODE_SIMS)


def name_float(x):
    return f"{float(x):.1f}".replace(".", "p")


def parse_csv(value):
    if value is None or str(value).strip() == "":
        return []
    return [v.strip() for v in str(value).split(",") if v.strip()]


def parse_float_spec(value, default=None):
    """Accept comma lists, or inclusive range 'start:stop:step'."""
    if value is None:
        return default if default is not None else []

    value = str(value).strip()
    if ":" in value:
        start, stop, step = map(float, value.split(":"))
        n = int(np.floor((stop - start) / step + 0.5)) + 1
        return [round(start + i * step, 10) for i in range(n)]

    return [float(v) for v in parse_csv(value)]


def expand_to_length(values, n, name):
    if len(values) == 0:
        raise ValueError(f"No values supplied for {name}.")
    if len(values) == 1:
        return values * n
    if len(values) != n:
        raise ValueError(f"{name} must have length 1 or {n}; got {len(values)}")
    return values


def load_mle(box, sim, iz, lc):
    path = f"./data_files/mle_parameters/{box}/{sim}/{iz}/lightcone{lc}/mle_values.txt"
    amp = np.loadtxt(path, usecols=1, skiprows=6, max_rows=1, delimiter="=")
    slope = np.loadtxt(path, usecols=1, skiprows=7, max_rows=1, delimiter="=")
    return float(amp), float(slope)


def load_fiducial_std(iz, spectra_key):
    path = f"./data_files/mle_parameters/L1000N1800/HYDRO_FIDUCIAL/{iz}/lightcone0/output_std.npz"
    data = np.load(path)
    return np.asarray(data["auto_std" if spectra_key == "auto" else "cross_std"], dtype=float)


def load_binning_and_data(iz):
    bin_setup = yaml.safe_load(open("./unWISExLens_lklh/unWISExLens_lklh/config_files/binning_setup.yaml"))

    if iz == "Blue":
        bin_edges = np.array(bin_setup["Blue_ACT"]["ell_bin_edges"])
    elif iz == "Green":
        bin_edges = np.array(bin_setup["Green_ACT"]["ell_bin_edges"])
    else:
        raise ValueError("iz must be Blue or Green")

    edges_int = np.rint(bin_edges).astype(int)
    b = nmt.NmtBin.from_edges(edges_int[:-1], edges_int[1:])
    ells = b.get_effective_ells()
    ell_200_mask = np.where(ells > 200)
    ell = ells[ell_200_mask]

    obs = np.loadtxt(
        f"./unWISExLens_lklh/data/v1.0/bandpowers/unWISExACT-DR6_{iz.lower()}_baseline_Clgg+Clkk+Clkg.dat",
        usecols=(0, 1, 2, 3),
    )[ell_200_mask, :].reshape(-1, 4)

    planck_raw = np.loadtxt(
        f"./unWISExLens_lklh/data/v1.0/bandpowers/unWISExPlanck-PR4_{iz.lower()}_baseline_Clgg+Clkk+Clkg.dat",
        usecols=(0, 1, 2, 3),
    )
    planck_mask = np.where(planck_raw[:, 0] > 200)[0]
    planck = planck_raw[planck_mask, :].reshape(-1, 4)

    cov = np.loadtxt(
        f"./unWISExLens_lklh/data/v1.0/covariances/covmat_Clgg+Clkg_unWISExACT-DR6_{iz.lower()}_baseline.dat"
    )
    var = np.diag(cov)
    obs_std_auto = np.sqrt(var[: len(var) // 2])[ell_200_mask]
    obs_std_cross = np.sqrt(var[len(var) // 2 :])[ell_200_mask]

    planck_cov = np.loadtxt(
        f"./unWISExLens_lklh/data/v1.0/covariances/covmat_Clgg+Clkg_unWISExPlanck-PR4_{iz.lower()}_baseline.dat"
    )
    planck_var = np.diag(planck_cov)
    planck_std_auto = np.sqrt(planck_var[: len(planck_var) // 2])[planck_mask]
    planck_std_cross = np.sqrt(planck_var[len(planck_var) // 2 :])[planck_mask]

    return {
        "ell": ell,
        "obs": obs,
        "planck": planck,
        "obs_std_auto": obs_std_auto,
        "obs_std_cross": obs_std_cross,
        "planck_std_auto": planck_std_auto,
        "planck_std_cross": planck_std_cross,
    }


def load_spectrum(box, sim, iz, lc, amp, slope, spectra_key, ell, shot_noise=False, smooth=True):
    amp_name = name_float(amp)
    slope_name = name_float(slope)

    if spectra_key == "auto":
        usecol = 1 if shot_noise else 2
        path = (
            f"./data_files/power_spectra/galaxy_galaxy/{box}/{sim}/{iz}/lightcone{lc}/"
            f"galaxy_galaxy_power_spectrum_{amp_name}_{slope_name}.txt"
        )
    elif spectra_key == "cross":
        usecol = 1
        path = (
            f"./data_files/power_spectra/kappa_galaxy/{box}/{sim}/{iz}/lightcone{lc}/"
            f"kappa_galaxy_power_spectrum_{amp_name}_{slope_name}.txt"
        )
    else:
        raise ValueError("spectra_key must be 'auto' or 'cross'")

    y = np.loadtxt(path, skiprows=1, usecols=usecol)

    if not smooth:
        return y

    ell_1000_mask = np.where(ell > 1000)
    y_low = y[np.where(ell <= 1000)]
    y_high = savgol_filter(y[ell_1000_mask], window_length=5, polyorder=2)
    return np.concatenate((y_low, y_high))


def make_entries(args, iz, ell, spectra_key):
    if args.mode == "spectra":
        sims = SPECTRA_MODE_SIMS
        boxes = SPECTRA_MODE_BOXES
        lightcones = SPECTRA_MODE_LIGHTCONES
    else:
        sims = [args.sim]
        boxes = [args.box]
        lightcones = [args.lc]

    amps = parse_float_spec(args.amps, default=[args.amp])
    slopes = parse_float_spec(args.slopes, default=[args.slope])

    # If --use-mle is set, make one entry per sim. Otherwise make all amp/slope combinations for every sim.
    entries = []
    if args.use_mle:
        for box, sim, lc in zip(boxes, sims, lightcones):
            amp, slope = load_mle(box, sim, iz, lc)
            label = SIM_LABELS.get(sim, sim)
            color = SIM_COLORS.get(sim, None)
            entries.append({"box": box, "sim": sim, "lc": lc, "amp": amp, "slope": slope, "label": label, "color": color})
    else:
        for box, sim, lc in zip(boxes, sims, lightcones):
            for amp in amps:
                for slope in slopes:
                    if args.mode == "spectra":
                        label = SIM_LABELS.get(sim, sim)
                        color = SIM_COLORS.get(sim, None)
                    else:
                        if len(amps) > 1 and len(slopes) == 1:
                            label = rf"$A={amp:.1f}$"
                        elif len(slopes) > 1 and len(amps) == 1:
                            label = rf"$s={slope:.1f}$"
                        else:
                            label = rf"$A={amp:.1f}$, $s={slope:.1f}$"
                        color = None

                    entries.append({
                        "box": box,
                        "sim": sim,
                        "lc": lc,
                        "amp": amp,
                        "slope": slope,
                        "label": label,
                        "color": color,
                    })

    labels_override = parse_csv(args.labels)
    colors_override = parse_csv(args.colors)
    if labels_override:
        labels_override = expand_to_length(labels_override, len(entries), "labels")
    if colors_override:
        colors_override = expand_to_length(colors_override, len(entries), "colors")

    cmap = pb.get_cmap(args.cmap)
    for i, entry in enumerate(entries):
        if labels_override:
            entry["label"] = labels_override[i]
        if colors_override:
            entry["color"] = colors_override[i]
        if entry["color"] is None:
            entry["color"] = cmap(i / max(1, len(entries) - 1))

        entry["spectrum"] = load_spectrum(
            entry["box"], entry["sim"], iz, entry["lc"], entry["amp"], entry["slope"],
            spectra_key, ell, shot_noise=args.shot_noise, smooth=not args.no_smooth,
        )

    return entries


def choose_reference(entries, requested):
    if requested is not None:
        return int(requested)

    for i, e in enumerate(entries):
        if e["box"] == "L1000N1800" and e["sim"] == "HYDRO_FIDUCIAL" and e["lc"] == 0:
            return i
    return 0


def add_observations(ax, ax_ratio, data, fid, spectra_key, show=True):
    if not show:
        return

    obs_col = 1 if spectra_key == "auto" else 3
    obs_std_key = "obs_std_auto" if spectra_key == "auto" else "obs_std_cross"
    planck_std_key = "planck_std_auto" if spectra_key == "auto" else "planck_std_cross"

    obs = data["obs"]
    planck = data["planck"]
    obs_std = data[obs_std_key]
    planck_std = data[planck_std_key]
    ell = data["ell"]

    y_obs = obs[:, obs_col] * 1e5
    y_obs_err = obs_std * 1e5
    act_line, = ax.plot(obs[:, 0], y_obs, color="k", marker=".", markersize=4, linewidth=0, label=r"ACT $\times$ unWISE")
    ax.fill_between(obs[:, 0], y_obs + y_obs_err, y_obs - y_obs_err, color="k", alpha=0.2, linewidth=0)

    if ax_ratio is not None:
        fid_obs = fid
        ax_ratio.plot(obs[:, 0], y_obs / fid_obs, color="k", marker=".", markersize=4, linewidth=0, alpha=0.8)
        ax_ratio.fill_between(obs[:, 0], (y_obs + y_obs_err) / fid_obs, (y_obs - y_obs_err) / fid_obs,
                              color="k", alpha=0.2, linewidth=0)

    y_planck = planck[:, obs_col] * 1e5
    y_planck_err = planck_std * 1e5
    planck_line, = ax.plot(planck[:, 0], y_planck, color="r", marker=".", markersize=4, linewidth=0, label=r"Planck $\times$ unWISE")
    ax.fill_between(planck[:, 0], y_planck + y_planck_err, y_planck - y_planck_err,
                    color="r", alpha=0.2, linewidth=0)

    if ax_ratio is not None:
        fid_planck = np.interp(planck[:, 0], ell, fid)
        ax_ratio.plot(planck[:, 0], y_planck / fid_planck, color="r", marker=".", markersize=4, linewidth=0, alpha=0.8)
        ax_ratio.fill_between(planck[:, 0], (y_planck + y_planck_err) / fid_planck,
                            (y_planck - y_planck_err) / fid_planck, color="r", alpha=0.2, linewidth=0)
        
    return act_line, planck_line


def plot_panel(ax, ax_ratio, data, entries, spectra_key, iz, args):
    model_handles = []
    ell = data["ell"]
    ref_idx = choose_reference(entries, args.ref_index)
    fid = np.asarray(entries[ref_idx]["spectrum"], dtype=float)

    if ax_ratio is not None:
        ax_ratio.axhline(1.0, color="k", linestyle="--", alpha=0.6)

    for i, entry in enumerate(entries):
        y = np.asarray(entry["spectrum"], dtype=float)
        line, = ax.plot(ell, y, color=entry["color"], lw=1.3 if i != ref_idx else 1.8,
                alpha=0.9, label=entry["label"])
        model_handles.append(line)
        if ax_ratio is not None:
            ax_ratio.plot(ell, y / fid, color=entry["color"], alpha=0.9)

    if args.fiducial_std and entries[ref_idx]["box"] == "L1000N1800" and entries[ref_idx]["sim"] == "HYDRO_FIDUCIAL":
        std = load_fiducial_std(iz, spectra_key)
        if std.shape == fid.shape:
            ax.fill_between(ell, fid - std, fid + std, color=entries[ref_idx]["color"], alpha=0.25, linewidth=0)
            if ax_ratio is not None:
                ax_ratio.fill_between(ell, (fid - std) / fid, (fid + std) / fid,
                                      color=entries[ref_idx]["color"], alpha=0.25, linewidth=0)
        else:
            print(f"Skipping fiducial std for {iz}/{spectra_key}: std shape {std.shape} != spectrum shape {fid.shape}")

    act_line, planck_line = add_observations(ax, ax_ratio, data, fid, spectra_key, show=not args.no_observations)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(args.xmin, args.xmax)
    ax.set_ylim(bottom=args.auto_bottom if spectra_key == "auto" else args.cross_bottom,
                top=args.auto_top if spectra_key == "auto" else args.cross_top)
    if ax_ratio is not None:
        ax_ratio.set_xscale("log")
        ax_ratio.set_xlim(args.xmin, args.xmax)
        ax_ratio.set_ylim(*(args.auto_ratio_ylim if spectra_key == "auto" else args.cross_ratio_ylim))
        ax_ratio.grid(alpha=0.25)

    return fid, model_handles, act_line, planck_line


def make_figure(panel_specs, args):
    if args.mode == "samples":
        fig, axes = pb.subplots(
            1, 2,
            figsize=(9.0, 3.6),
            sharex=True,
            sharey=True,
            gridspec_kw={"wspace": 0.08},
        )
        axes = np.asarray([axes, [None, None]])
    else:
        fig, axes = pb.subplots(
            2, 2,
            figsize=(9.0, 4.8),
            sharex="col",
            gridspec_kw={"height_ratios": [3.0, 1.0], "hspace": 0.06, "wspace": 0.20},
        )

    for col, spec in enumerate(panel_specs):
        iz = spec["iz"]
        spectra_key = spec["spectra_key"]
        data = load_binning_and_data(iz)
        old_amps, old_slopes = args.amps, args.slopes

        if "amps" in spec:
            args.amps = spec["amps"]
        if "slopes" in spec:
            args.slopes = spec["slopes"]

        entries = make_entries(args, iz, data["ell"], spectra_key)

        args.amps, args.slopes = old_amps, old_slopes

        ax = axes[0, col]
        ax_ratio = axes[1, col]

        _, model_handles, act_line, planck_line = plot_panel(
            ax,
            ax_ratio,
            data,
            entries,
            spectra_key,
            iz,
            args,
        )

        ylabel = r"$C_\ell^{\rm gg} \times 10^5$" if spectra_key == "auto" else r"$C_\ell^{\kappa \rm g} \times 10^5$"

        if args.mode == "samples":
            if col == 0:
                ax.set_ylabel(ylabel)
            else:
                ax.tick_params(axis="y", labelleft=False)
        elif args.mode == "spectra":
            ax.set_ylabel(ylabel)

        ax.set_xlabel(r"Multipole moment $\ell$")
        legend_title = spec.get("legend_title", None)

        if args.mode == "spectra":
            legend_title = f"{iz} sample, $A={args.amp:.1f}$, $s={args.slope:.1f}$"

        model_legend = ax.legend(
            handles=model_handles,
            ncol=2 if args.mode == "samples" else 1,
            loc="upper right",
            title=legend_title,
            frameon=False,
        )

        obs_legend = ax.legend(
            handles=[act_line, planck_line],
            loc="lower left",
            fontsize=7,
            frameon=False,
        )

        ax.add_artist(model_legend)

        if args.mode == "spectra":
            ax_ratio.set_ylabel("Ratio" if col == 0 else None)
            ax_ratio.set_xlabel(r"Multipole moment $\ell$")

    Path("./Plots").mkdir(exist_ok=True)
    out = Path(args.output) if args.output else Path("./Plots") / default_output_name(args)
    pb.savefig(out, dpi=args.dpi, bbox_inches="tight")
    pb.close(fig)
    print(f"Saved {out}")


def default_output_name(args):
    if args.mode == "samples":
        spec = "gg" if args.spectrum == "auto" else "kg"
        return f"custom_{spec}_{args.sample}_amps_slopes.{args.file}"
    else:
        return f"custom_gg_kg_{args.sample}_sims.{args.file}"


def main():
    parser = argparse.ArgumentParser(description="Flexible two-column mock-catalogue power-spectrum plotter.")

    parser.add_argument("--mode", choices=["samples", "spectra"], required=True,
                        help="samples: Blue vs Green for one spectrum. spectra: gg vs kg for one sample.")
    parser.add_argument("--spectrum", choices=["auto", "cross"], default="auto",
                        help="Used in --mode samples.")
    parser.add_argument("--sample", choices=["Blue", "Green"], default="Blue",
                        help="Used in --mode spectra.")

    parser.add_argument("--box", default="L1000N1800")
    parser.add_argument("--sim", default="HYDRO_FIDUCIAL")
    parser.add_argument("--lc", type=int, default=0)
    parser.add_argument("--boxes", default=None, help="Comma list. Length 1 or same length as --sims.")
    parser.add_argument("--sims", default=None, help="Comma list of simulations. Defaults to --sim.")
    parser.add_argument("--lightcones", default=None, help="Comma list. Length 1 or same length as --sims.")

    parser.add_argument("--amps", default=None, help="Comma list or start:stop:step. Ignored with --use-mle.")
    parser.add_argument("--slopes", default=None, help="Comma list or start:stop:step. Ignored with --use-mle.")
    parser.add_argument("--amp", type=float, default=10.8)
    parser.add_argument("--slope", type=float, default=0.5)
    parser.add_argument("--use-mle", action="store_true", help="Use one MLE amp/slope per sim.")

    parser.add_argument("--labels", default=None, help="Optional comma list matching final plotted entries.")
    parser.add_argument("--colors", default=None, help="Optional comma list matching final plotted entries.")
    parser.add_argument("--cmap", default="viridis")
    parser.add_argument("--ref-index", type=int, default=None)

    parser.add_argument("--shot-noise", action="store_true")
    parser.add_argument("--no-smooth", action="store_true")
    parser.add_argument("--no-observations", action="store_true")
    parser.add_argument("--fiducial-std", action="store_true")

    parser.add_argument("--xmin", type=float, default=200)
    parser.add_argument("--xmax", type=float, default=4000)
    parser.add_argument("--auto-bottom", type=float, default=1e-2)
    parser.add_argument("--cross-bottom", type=float, default=1e-4)
    parser.add_argument("--auto-top", type=float, default=None)
    parser.add_argument("--cross-top", type=float, default=4e-2)
    parser.add_argument("--auto-ratio-ylim", type=float, nargs=2, default=(0.9, 1.1))
    parser.add_argument("--cross-ratio-ylim", type=float, nargs=2, default=(0.5, 1.5))

    parser.add_argument("--file", choices=["png", "pdf"], default="png")
    parser.add_argument("--output", default=None)
    parser.add_argument("--dpi", type=int, default=400)

    args = parser.parse_args()

    if args.mode == "samples":
        panel_specs = [
            {
                "iz": args.sample,
                "spectra_key": args.spectrum,
                "title": "amplitudes",
                "amps": args.amps,
                "slopes": str(args.slope),
                "legend_title": f"{args.sample} sample",
            },
            {
                "iz": args.sample,
                "spectra_key": args.spectrum,
                "title": "slopes",
                "amps": str(args.amp),
                "slopes": args.slopes,
                "legend_title": f"{args.sample} sample",
            },
        ]
    else:
        args.amps = str(args.amp)
        args.slopes = str(args.slope)

        panel_specs = [
            {
                "iz": args.sample,
                "spectra_key": "auto",
                "title": None,
                "legend_title": f"{args.sample} sample, $A={args.amp:.1f}$, $s={args.slope:.1f}$",
            },
            {
                "iz": args.sample,
                "spectra_key": "cross",
                "title": None,
                "legend_title": f"{args.sample} sample, $A={args.amp:.1f}$, $s={args.slope:.1f}$",
            },
        ]

    make_figure(panel_specs, args)


if __name__ == "__main__":
    main()
