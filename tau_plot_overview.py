#!/usr/bin/env python3
import argparse
import pickle
from pathlib import Path
from scipy.interpolate import interp1d
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
    # if box == 'L2800N5040':
    #     nside = 4096
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


def load_observed_tau(args):
    """
    Load digitised observed tau profile.

    Expected columns:
        0: theta [arcmin]
        1: median tau
        2: upper error
        3: lower error
    """
    path = (Path(args.base_dir)/f"digitized_obs_data_{args.sample.lower()}.txt")

    if not path.exists():
        raise FileNotFoundError(f"Observed tau profile not found: {path}")

    data = np.loadtxt(path, comments="#", skiprows=1)

    if data.ndim == 1:
        data = data.reshape(1, -1)

    if data.shape[1] < 4:
        raise ValueError(f"Expected at least four columns in {path}, but found {data.shape[1]}.")

    theta = data[:, 0]
    tau = data[:, 1]
    upper_error = data[:, 2]
    lower_error = data[:, 3]

    return {
        "theta": theta,
        "tau": tau,
        "upper_error": upper_error,
        "lower_error": lower_error,
        "path": path,
    }


def plot_observed_tau(ax, observed):
    theta = observed["theta"]
    tau = observed["tau"]
    upper_error = observed["upper_error"]
    lower_error = observed["lower_error"]

    # Matplotlib expects yerr as:
    # [[distance below the central value],
    #  [distance above the central value]]

    upper_error = (observed["upper_error"] - observed["tau"])
    lower_error = (observed["tau"] - observed["lower_error"])

    yerr = np.vstack((lower_error, upper_error))

    observed_handle = ax.errorbar(theta, tau, yerr=yerr, fmt="o", markersize=3.5, color="k", ecolor="k", elinewidth=0.8, capsize=2, linewidth=0, label="Coulton et al. 2025", zorder=10)

    return observed_handle


def plot_observed_tau_difference(ax_ratio, observed, theta_fid, tau_fid):
    """
    Interpolate the fiducial model onto the observed theta values and plot
    observed tau minus fiducial tau, including asymmetric observed errors.

    Parameters
    ----------
    observed_scaled
        True when the observed file already stores tau * 1e4.
        The simulated tau profiles are assumed to be unscaled.
    """
    theta_obs = np.asarray(observed["theta"], dtype=float)
    tau_obs = np.asarray(observed["tau"], dtype=float)
    upper_bound = np.asarray(observed["upper_error"], dtype=float)
    lower_bound = np.asarray(observed["lower_error"], dtype=float)

    theta_fid = np.asarray(theta_fid, dtype=float)
    tau_fid = np.asarray(tau_fid, dtype=float)

    if not np.all(np.diff(theta_fid) > 0):
        raise ValueError("Fiducial theta values must be strictly increasing.")

    # Do not extrapolate beyond the model's radial range.
    # valid = ((theta_obs >= theta_fid.min()) & (theta_obs <= theta_fid.max()))
    theta_tolerance = 0.1  # arcmin
    valid = ((theta_obs >= theta_fid.min() - theta_tolerance) & (theta_obs <= theta_fid.max() + theta_tolerance))

    outside = ((theta_obs < theta_fid.min()) | (theta_obs > theta_fid.max()))
    if np.any(outside & valid):
        print("Warning: extrapolating fiducial tau for observed theta:", theta_obs[outside & valid])

    fid_interpolator = interp1d(theta_fid, tau_fid, kind="linear", bounds_error=False, fill_value="extrapolate", assume_sorted=True)

    tau_fid_at_obs = fid_interpolator(theta_obs[valid])

    # Observations are already tau * 1e4, so scale the model.
    tau_fid_at_obs = tau_fid_at_obs * 1e4

    tau_obs_plot = tau_obs[valid]
    upper_bound_plot = upper_bound[valid]
    lower_bound_plot = lower_bound[valid]

    difference = tau_obs_plot - tau_fid_at_obs

    # Since the observed columns are absolute upper/lower bounds:
    upper_error = upper_bound_plot - tau_obs_plot
    lower_error = tau_obs_plot - lower_bound_plot

    if np.any(upper_error < 0) or np.any(lower_error < 0):
        raise ValueError("Observed upper/lower bounds do not bracket the median.")

    yerr = np.vstack((lower_error, upper_error))

    ax_ratio.errorbar(theta_obs[valid], difference, yerr=yerr, fmt="o", markersize=3.5, color="k", ecolor="k", elinewidth=0.8, capsize=2, linewidth=0, zorder=10)


def calculate_chi_squared(theta_sim, tau_sim, observed, scale=1e4, theta_tolerance=0.1):
    """
    Calculate chi^2 between a simulated tau profile and the observed data.

    The simulated profile is interpolated onto the observed theta values.
    Observed tau values and bounds are assumed to already be in tau * 1e4,
    while the simulated profile is in unscaled tau units.

    For asymmetric observational uncertainties:
        model > observation -> use upper uncertainty
        model < observation -> use lower uncertainty
    """

    theta_sim = np.asarray(theta_sim, dtype=float)
    tau_sim = np.asarray(tau_sim, dtype=float)

    theta_obs = np.asarray(observed["theta"], dtype=float)
    tau_obs = np.asarray(observed["tau"], dtype=float)
    upper_bound = np.asarray(observed["upper_error"], dtype=float)
    lower_bound = np.asarray(observed["lower_error"], dtype=float)

    if not np.all(np.diff(theta_sim) > 0):
        raise ValueError("Simulated theta values must be strictly increasing.")

    # Observational 1-sigma errors.
    sigma_upper = upper_bound - tau_obs
    sigma_lower = tau_obs - lower_bound

    if np.any(sigma_upper <= 0) or np.any(sigma_lower <= 0):
        raise ValueError("Observed upper/lower bounds must give positive uncertainties.")

    # Allow only a very small extrapolation beyond the simulated range.
    valid = ((theta_obs >= theta_sim.min() - theta_tolerance) & (theta_obs <= theta_sim.max() + theta_tolerance) 
             & np.isfinite(theta_obs) & np.isfinite(tau_obs) & np.isfinite(sigma_upper) & np.isfinite(sigma_lower))

    if not np.any(valid):
        raise ValueError("No observed points overlap the simulated theta range.")

    interpolator = interp1d(theta_sim, tau_sim, kind="linear", bounds_error=False, fill_value="extrapolate", assume_sorted=True)

    # Simulations are unscaled; observations are already tau * 1e4.
    tau_sim_at_obs = (interpolator(theta_obs[valid]) * scale)

    tau_obs_valid = tau_obs[valid]
    sigma_upper_valid = sigma_upper[valid]
    sigma_lower_valid = sigma_lower[valid]

    residual = tau_sim_at_obs - tau_obs_valid

    # Choose the uncertainty in the direction of the simulation.
    sigma = np.where(residual >= 0.0, sigma_upper_valid, sigma_lower_valid)

    chi2 = np.sum((residual / sigma) ** 2)

    return chi2, np.count_nonzero(valid)


def add_chi_squared_text(ax, chi2_values, x=0.45, y=0.90, line_spacing=0.055):
    """
    Display chi^2 values in the same order as the simulation legend.

    chi2_values should contain:
        [(name, chi2, color), ...]
    """

    if len(chi2_values) == 0:
        return

    chi2_fid = chi2_values[0][1]

    for i, (name, chi2, color) in enumerate(chi2_values):

        if i == 0:
            text = rf"{chi2:.1f}"
        else:
            delta_chi2 = chi2 - chi2_fid
            text = rf"({delta_chi2:.1f})"

        ax.text(x, y - i * line_spacing, text, transform=ax.transAxes, color=color, fontsize=8, ha="left", va="top")


def put_on_reference_grid(theta_ref, theta, values):
    if theta.shape == theta_ref.shape and np.allclose(theta, theta_ref):
        return values

    raise ValueError("Tau profiles use different radial-bin grids.")


def plot_shifted_ratios(ax_ratio, theta, profiles, reference, colors, shift_padding=0.05, alpha=0.9):
    """
    Shift all profiles by the same numerical value so they are positive,
    divide each shifted profile by the shifted reference, and plot the ratios.

    Parameters
    ----------
    ax_ratio : matplotlib.axes.Axes
        Axis on which to plot the ratios.

    theta : array-like
        Common radial grid.

    profiles : sequence of array-like
        Profiles to compare with the reference.

    reference : array-like
        Fiducial/reference profile.

    colors : sequence
        Plot colour for each profile.

    shift_padding : float
        Additional shift as a fraction of the full profile amplitude range.
        This keeps the shifted denominator safely away from zero.

    alpha : float
        Line transparency.

    Returns
    -------
    ratios : list of ndarray
        Shifted ratios that were plotted.

    shift : float
        Common additive shift applied to every profile.
    """
    reference = np.asarray(reference, dtype=float)
    profiles = [np.asarray(profile, dtype=float) for profile in profiles]

    if len(profiles) != len(colors):
        raise ValueError("profiles and colors must have the same length.")

    all_values = np.concatenate([reference.ravel()] + [profile.ravel() for profile in profiles])

    finite_values = all_values[np.isfinite(all_values)]

    if finite_values.size == 0:
        raise ValueError("No finite profile values were supplied.")

    global_min = np.min(finite_values)
    global_max = np.max(finite_values)
    amplitude_range = global_max - global_min

    # A fallback is needed if all values happen to be identical.
    if amplitude_range == 0.0:
        amplitude_range = max(np.abs(global_max), 1.0)

    # Make the smallest shifted value equal to
    # shift_padding * amplitude_range, rather than exactly zero.
    minimum_allowed = shift_padding * amplitude_range
    shift = max(0.0, -global_min + minimum_allowed)

    shifted_reference = reference + shift

    if np.any(~np.isfinite(shifted_reference) | (shifted_reference <= 0.0)):
        raise ValueError("The shifted reference profile is not strictly positive.")

    ax_ratio.axhline(1.0, color="k", linestyle="--", linewidth=0.8, alpha=0.7)

    ratios = []

    for profile, color in zip(profiles, colors):
        shifted_profile = profile + shift

        ratio = np.full_like(shifted_profile, np.nan, dtype=float)

        valid = (np.isfinite(shifted_profile) & np.isfinite(shifted_reference) & (shifted_profile > 0.0) & (shifted_reference > 0.0))

        np.divide(shifted_profile, shifted_reference, out=ratio, where=valid)

        ax_ratio.plot(theta, ratio, color=color, alpha=alpha)

        ratios.append(ratio)

    return ratios, shift


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


def plot_resolution(ax, ax_ratio, args, observed=None):
    fid = load_named_profile(args, "L1000N1800", "HYDRO_FIDUCIAL", 0)
    hires = load_named_profile(args, "L1000N3600", "HYDRO_FIDUCIAL", 0)
    l2800_profiles = [load_named_profile(args, "L2800N5040", "HYDRO_FIDUCIAL", lc) for lc in range(args.lc_count)]

    # l2800_cluster_profiles = []
    # for lc in range(args.lc_count):
    #     path = Path(f"{args.base_dir}/L2800N5040/HYDRO_FIDUCIAL/{args.sample}/lightcone{lc}/tau_mle_catalogue_nside{args.nside}_{args.primary_method}_{args.file_method}_high_mass_clusters.pickle")
    #     data = pickle.load(path.open("rb"))
    #     l2800_cluster_profiles.append({"theta": np.asarray(data[0], dtype=float), "tau": np.asarray(data[1], dtype=float), "distance": np.asarray(data[2], dtype=float), "path": path})

    theta = fid["theta"]
    tau_fid = fid["tau"]
    tau_hires = put_on_reference_grid(theta, hires["theta"], hires["tau"])
    l2800_stack = np.asarray([put_on_reference_grid(theta, p["theta"], p["tau"]) for p in l2800_profiles])
    # l2800_cluster_stack = np.asarray([put_on_reference_grid(theta, p["theta"], p["tau"]) for p in l2800_cluster_profiles])
    # for i in range(8):
    #     print(f"lc= {i}, {l2800_profiles[i]}")
    #     print("\n")
    # quit()
    mean = np.mean(l2800_stack, axis=0)
    lo = np.min(l2800_stack, axis=0)
    hi = np.max(l2800_stack, axis=0)
    # mean_cluster = np.mean(l2800_cluster_stack, axis=0)
    # lo_cluster = np.min(l2800_cluster_stack, axis=0)
    # hi_cluster = np.max(l2800_cluster_stack, axis=0)

    chi2_values = []

    ax.plot(theta, (tau_fid)*1e4, color="#117733", label=r"L1\_m9")
    ax.plot(theta, (tau_hires)*1e4, color="#CC6677", label=r"L1\_m8")
    ax.plot(theta, (mean)*1e4, color="#332288", label=r"L2p8\_m9")
    # ax.plot(theta, (mean_cluster)*1e4, color="#4488AA", linestyle='dashed', label=r"L2p8\_m9\_clusters")
    ax.fill_between(theta, (lo)*1e4, (hi)*1e4, color="#332288", alpha=0.25, linewidth=0)
    # ax.fill_between(theta, (lo_cluster)*1e4, (hi_cluster)*1e4, color="#4488AA", alpha=0.25, linewidth=0)

    # ratios, shift = plot_shifted_ratios(ax_ratio=ax_ratio, theta=theta, profiles=[tau_fid, tau_hires, mean], reference=tau_fid, colors=["#117733", "#CC6677", "#332288"], shift_padding=args.shift_padding)

    # print(f"resolution: shifted ratio offset = {shift:.6e}")

    # shifted_reference = tau_fid + shift

    # lo_ratio = (lo + shift) / shifted_reference
    # hi_ratio = (hi + shift) / shifted_reference

    # envelope_lower = np.minimum(lo_ratio, hi_ratio)
    # envelope_upper = np.maximum(lo_ratio, hi_ratio)
    # envelope_valid = (np.isfinite(envelope_lower) & np.isfinite(envelope_upper) & (shifted_reference > 0.0))

    # ax_ratio.fill_between(theta, envelope_lower, envelope_upper, where=envelope_valid, color="#332288", alpha=0.25, linewidth=0)


    ax_ratio.axhline(0.0, color="k", linestyle="--", linewidth=0.8, alpha=0.7)

    ax_ratio.plot(theta, (tau_fid - tau_fid)*1e4, color="#117733")
    ax_ratio.plot(theta, (tau_hires - tau_fid)*1e4, color="#CC6677")
    ax_ratio.plot(theta, (mean - tau_fid)*1e4, color="#332288")
    ax_ratio.fill_between(theta, (lo - tau_fid)*1e4, (hi - tau_fid)*1e4, color="#332288", alpha=0.25, linewidth=0)
    # ax_ratio.plot(theta, (mean_cluster - tau_fid)*1e4, color="#4488AA", linestyle='dashed')
    # ax_ratio.fill_between(theta, (lo_cluster - tau_fid)*1e4, (hi_cluster - tau_fid)*1e4, color="#4488AA", alpha=0.25, linewidth=0)

    if observed is not None:
        resolution_profiles = [(r"L1\_m9", tau_fid, "#117733"), 
                               (r"L1\_m8", tau_hires, "#CC6677"), 
                               (r"L2p8\_m9", mean, "#332288"),
                            #    (r"L2p8\_m9\_clusters", mean_cluster, "#4488AA")
                            ]

        for name, tau, color in resolution_profiles:
            chi2, npoints = calculate_chi_squared(theta_sim=theta, tau_sim=tau, observed=observed)

            chi2_values.append((name, chi2, color))

            print(f"resolution: {name}: chi2 = {chi2:.3f}, N = {npoints}")


    # fid_difference = stable_fractional_difference(tau_fid, tau_fid, args.ratio_threshold)
    # hires_difference = stable_fractional_difference(tau_hires, tau_fid, args.ratio_threshold)
    # mean_difference = stable_fractional_difference(mean, tau_fid, args.ratio_threshold)
    # lo_difference = stable_fractional_difference(lo, tau_fid, args.ratio_threshold)
    # hi_difference = stable_fractional_difference(hi, tau_fid, args.ratio_threshold)

    # ax_ratio.plot(theta, fid_difference, color="#117733")
    # ax_ratio.plot(theta, hires_difference, color="#CC6677")
    # ax_ratio.plot(theta, mean_difference, color="#332288")
    # ax_ratio.fill_between(theta, lo_difference, hi_difference, where=(np.isfinite(lo_difference) & np.isfinite(hi_difference)), 
    #                       color="#332288", alpha=0.25, linewidth=0)
    
    return fid, chi2_values


# Use for stable fractional difference

# def plot_standard_group(ax, ax_ratio, args, group_key):
#     group = GROUPS[group_key]
#     profiles = []

#     for sim, name, color in zip(
#         group["sims"],
#         group["names"],
#         group["colors"],
#     ):
#         profile = load_named_profile(args, group["box"], sim, 0)
#         profiles.append((profile, name, color))

#     # Use the first profile in each group as the reference.
#     reference = profiles[0][0]
#     theta_ref = reference["theta"]
#     tau_ref = reference["tau"]

#     ax_ratio.axhline(0.0, color="k", linestyle="--", linewidth=0.8, alpha=0.7)

#     for profile, name, color in profiles:
#         tau = put_on_reference_grid(
#             theta_ref,
#             profile["theta"],
#             profile["tau"],
#         )

#         ax.plot(theta_ref, tau, color=color, label=name, alpha=0.9)

#         difference = stable_fractional_difference(tau, tau_ref, args.ratio_threshold)
#         ax_ratio.plot(theta_ref, difference, color=color, alpha=0.9)
        
#     return reference

def plot_standard_group(ax, ax_ratio, args, group_key, observed=None):
    group = GROUPS[group_key]
    profiles = []

    for sim, name, color in zip(group["sims"], group["names"], group["colors"]):
        profile = load_named_profile(args, group["box"], sim, 0)
        profiles.append((profile, name, color))

    reference = profiles[0][0]
    theta_ref = reference["theta"]
    tau_ref = reference["tau"]

    tau_profiles = []
    colors = []
    chi2_values = []

    ax_ratio.axhline(0.0, color="k", linestyle="--", linewidth=0.8, alpha=0.7)

    for profile, name, color in profiles:
        tau = put_on_reference_grid(theta_ref, profile["theta"], profile["tau"])

        ax.plot(theta_ref, (tau)*1e4, color=color, label=name, alpha=0.9)

        tau_profiles.append(tau)
        colors.append(color)

        ax_ratio.plot(theta_ref, (tau - tau_ref)*1e4, color=color, label=name, alpha=0.9)

        if observed is not None:
            chi2, npoints = calculate_chi_squared(theta_sim=theta_ref, tau_sim=tau, observed=observed)

            chi2_values.append((name, chi2, color))

            print(f"{group_key}: {name}: chi2 = {chi2:.3f}, N = {npoints}")

        # print(name, tau, tau_ref, tau - tau_ref)  

    # ratios, shift = plot_shifted_ratios(ax_ratio=ax_ratio, theta=theta_ref, profiles=tau_profiles, reference=tau_ref, colors=colors, shift_padding=args.shift_padding)

    # print(f"{group_key}: shifted ratio offset = {shift:.6e}")
    return reference, chi2_values


def configure_axis(ax, ax_ratio, args, show_ylabel=True):
    ax.axhline(0.0, color="k", linewidth=0.8, alpha=0.7)
    ax.set_xlim(args.xmin, args.xmax)
    if args.ymin is not None or args.ymax is not None:
        ax.set_ylim(bottom=args.ymin, top=args.ymax)
    # ax.set_title(title, fontsize=9)
    if show_ylabel:
        ax.set_ylabel(r"Filtered $\tau\,(\times 10^4)$")
    else:
        ax.tick_params(axis="y", labelleft=False, left=False)
        ax.yaxis.get_offset_text().set_visible(False)
    # formatter = ScalarFormatter(useMathText=True)
    # formatter.set_powerlimits((-4, -4))
    # ax.yaxis.set_major_formatter(formatter)
    # ax.ticklabel_format(axis="y", style="sci", scilimits=(-4, -4))
    # ax_ratio.yaxis.set_major_formatter(formatter)
    # ax_ratio.ticklabel_format(axis="y", style="sci", scilimits=(-4, -4))
    ax.legend(fontsize=8, loc="upper right", frameon=False, title=f"{args.sample} sample")


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

    observed = (None if args.no_observed else load_observed_tau(args))

    refs = {}
    chi2_map = {}

    refs["resolution"], chi2_map["resolution"] = (plot_resolution(panel_map["resolution"], ratio_map["resolution"], args, observed=observed))

    for key in ("cosmology", "agn_feedback", "other_feedback"):
        refs[key], chi2_map[key] = (plot_standard_group(panel_map[key], ratio_map[key], args, key, observed=observed))

    if args.ymin is None:
        shared_ymin = min(min(ax.dataLim.ymin for ax in panel_map.values()), np.nanmin(observed["lower_error"]) if observed is not None else np.nan)
        # shared_ymin = min(ax.dataLim.ymin for ax in panel_map.values())
    else:
        shared_ymin = args.ymin

    if args.ymax is None:
        shared_ymax = max(max(ax.dataLim.ymax for ax in panel_map.values()), np.nanmax(observed["upper_error"]) if observed is not None else np.nan)
        # shared_ymax = max(ax.dataLim.ymax for ax in panel_map.values())
    else:
        shared_ymax = args.ymax

    y_padding = 0.05 * (shared_ymax - shared_ymin)

    if args.ymin is None:
        shared_ymin -= y_padding

    if args.ymax is None:
        shared_ymax += y_padding

    if observed is not None:
        observed_handles = {}

        for key, ax in panel_map.items():
            observed_handles[key] = plot_observed_tau(ax, observed)

    for key, ax_ratio in ratio_map.items():
        plot_observed_tau_difference(ax_ratio=ax_ratio, observed=observed, theta_fid=refs[key]["theta"], tau_fid=refs[key]["tau"])

    for key, ax in panel_map.items():
        configure_axis(ax, ratio_map[key], args, show_ylabel=key in ("resolution", "agn_feedback"))
        ax.set_ylim(shared_ymin, shared_ymax)
        if args.distance_axis:
            is_top_panel = key in ("resolution", "cosmology")

            add_distance_axis(ax, refs[key]["theta"], refs[key]["distance"], show_label=is_top_panel, show_ticklabels=is_top_panel)

    if observed is not None:
        for key, ax in panel_map.items():
            add_chi_squared_text(ax, chi2_map[key])

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
        ax_ratio.set_ylabel(r"$(\tau - \tau_{\rm fid})\times 10^4$") # For difference
        # ax_ratio.set_ylabel(r"$(\tau+s)/(\tau_{\rm fid}+s)$") # For shifted ratio
        # ax_ratio.set_ylabel(r"$\Delta\tau/\max|\tau_{\rm fid}|$") # For stable fractional difference
        # ax_ratio.set_ylabel(r"$2(\tau-\tau_{\rm fid})/(|\tau|+|\tau_{\rm fid}|)$") # For symmetric rato
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
    parser.add_argument("--no-observed", action="store_true", help="Do not plot the digitised observed tau profile.")
    parser.add_argument("--xmin", type=float, default=0.0)
    parser.add_argument("--xmax", type=float, default=11.0)
    parser.add_argument("--ymin", type=float, default=None)
    parser.add_argument("--ymax", type=float, default=None)
    parser.add_argument("--ratio-ymin", type=float, default=0.1)
    parser.add_argument("--ratio-ymax", type=float, default=1.9)
    parser.add_argument("--ratio-threshold", type=float, default=1e-24, help=("Do not calculate ratios where the absolute reference tau profile is below this value."))
    parser.add_argument("--shift-padding", type=float, default=0.05, help=("Extra positive offset, expressed as a fraction of the full profile amplitude range, used when calculating shifted profile ratios."))
    parser.add_argument("--file", choices=["png", "pdf"], default="png")
    parser.add_argument("--dpi", type=int, default=400)
    parser.add_argument("--output-dir", default="./Plots")
    make_overview(parser.parse_args())


if __name__ == "__main__":
    main()
