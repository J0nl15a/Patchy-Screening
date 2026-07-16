import argparse
from pathlib import Path

import numpy as np
import pylab as pb
import yaml
import pymaster as nmt
from scipy.signal import savgol_filter

pb.rc('text', usetex=True)
pb.rc('font', family='serif', size=8)
pb.rcParams['font.size'] = 8


GROUPS = {
    "resolution": {
        "panel_title": "Resolution / lightcones",
        "box": "L2800N5040",
    },
    "cosmology": {
        "panel_title": "Cosmology",
        "box": "L1000N1800",
        "sims": [
            "HYDRO_LOW_SIGMA8_STRONGEST_AGN",
            "HYDRO_LOW_SIGMA8",
            "HYDRO_PLANCK_LARGE_NU_FIXED",
            "HYDRO_PLANCK_LARGE_NU_VARY",
            "HYDRO_PLANCK",
            "HYDRO_FIDUCIAL",
        ],
        "names": [
            r"LS8\_fgas$-8\sigma$",
            "LS8",
            "PlanckNu0p24Fix",
            "PlanckNu0p24Var",
            "Planck",
            "L1\_m9",
        ],
        "colors": [
            "#7B68EE",
            "#882255",
            "#999933",
            "#AA4499",
            "#44AA99",
            "#117733",
        ],
    },
    "agn_feedback": {
        "panel_title": "AGN feedback",
        "box": "L1000N1800",
        "sims": [
            "HYDRO_STRONGEST_AGN",
            "HYDRO_STRONGER_AGN",
            "HYDRO_STRONG_AGN",
            "HYDRO_WEAK_AGN",
            "HYDRO_FIDUCIAL",
        ],
        "names": [
            r"fgas$-8\sigma$",
            r"fgas$-4\sigma$",
            r"fgas$-2\sigma$",
            r"fgas$+2\sigma$",
            "L1\_m9",
        ],
        "colors": [
            "#105ba4",
            "#3787c0",
            "#6aaed6",
            "#abd0e6",
            "#117733",
        ],
    },
    "other_feedback": {
        "panel_title": "Other feedback",
        "box": "L1000N1800",
        "sims": [
            "HYDRO_STRONG_JETS_published",
            "HYDRO_JETS_published",
            "HYDRO_STRONG_SUPERNOVA",
            "HYDRO_FIDUCIAL",
        ],
        "names": [
            r"Jet\_fgas$-4\sigma$",
            "Jet",
            r"M*-$\sigma$",
            "L1\_m9",
        ],
        "colors": [
            "#55E18E",
            "#7EFF4B",
            "#FF8C40",
            "#117733",
        ],
    },
}


def name_float(x):
    return f"{float(x):.3f}".replace(".", "p")


def compute_chi2(model, obs, err):
    return float(np.sum(((obs - model) / err) ** 2))


def load_mle(box, sim, iz, lc):
    path = f"./data_files/mle_parameters/{box}/{sim}/{iz}/lightcone{lc}/mle_values.txt"
    amp = np.loadtxt(path, usecols=1, skiprows=6, max_rows=1, delimiter="=")
    slope = np.loadtxt(path, usecols=1, skiprows=7, max_rows=1, delimiter="=")
    return float(amp), float(slope)


def load_mle_chi2(box, sim, iz, lc):
    path = f"./data_files/mle_parameters/{box}/{sim}/{iz}/lightcone{lc}/mle_values.txt"
    chi2_auto = float(np.loadtxt(path, usecols=1, skiprows=3, max_rows=1, delimiter="="))
    chi2_cross = float(np.loadtxt(path, usecols=1, skiprows=5, max_rows=1, delimiter="="))
    return chi2_auto, chi2_cross


def load_fiducial_std(iz, spectra_key):
    path = f"./data_files/mle_parameters/L1000N1800/HYDRO_FIDUCIAL/{iz}/lightcone0/output_std.npz"
    data = np.load(path)

    if spectra_key == "auto":
        return np.asarray(data["auto_std"], dtype=float)
    elif spectra_key == "cross":
        return np.asarray(data["cross_std"], dtype=float)
    else:
        raise ValueError("spectra_key must be 'auto' or 'cross'")


def load_spectrum(box, sim, iz, lc, amp, slope, spectrum, ell_namaster, shot_noise=False):
    amp_name = name_float(amp)
    slope_name = name_float(slope)

    if spectrum == "auto":
        usecol = 1 if shot_noise else 2
        path = (
            f"./data_files/power_spectra/galaxy_galaxy/{box}/{sim}/{iz}/lightcone{lc}/"
            f"galaxy_galaxy_power_spectrum_{amp_name}_{slope_name}.txt"
        )
    elif spectrum == "cross":
        usecol = 1
        path = (
            f"./data_files/power_spectra/kappa_galaxy/{box}/{sim}/{iz}/lightcone{lc}/"
            f"kappa_galaxy_power_spectrum_{amp_name}_{slope_name}.txt"
        )
    else:
        raise ValueError("spectrum must be 'auto' or 'cross'")

    y = np.loadtxt(path, skiprows=1, usecols=usecol)

    ell_1000_mask = np.where(ell_namaster > 1000)
    y_low = y[np.where(ell_namaster <= 1000)]
    y_high = savgol_filter(y[ell_1000_mask], window_length=5, polyorder=2)

    return np.concatenate((y_low, y_high))


def load_group(group_key, iz, lc_count, ell_namaster, shot_noise=False):
    group = GROUPS[group_key]

    boxes, sims, lightcones, names, colors = [], [], [], [], []

    if group_key == "resolution":
        for lc in range(lc_count):
            boxes.append("L2800N5040")
            sims.append("HYDRO_FIDUCIAL")
            lightcones.append(lc)
            names.append("L2p8\_m9")
            colors.append("#332288")

        boxes.append("L1000N3600")
        sims.append("HYDRO_FIDUCIAL")
        lightcones.append(0)
        names.append("L1\_m8")
        colors.append("#CC6677")
        boxes.append("L1000N1800")
        sims.append("HYDRO_FIDUCIAL")
        lightcones.append(0)
        names.append("L1\_m9")
        colors.append("#117733")

    else:
        for sim, name, color in zip(group["sims"], group["names"], group["colors"]):
            boxes.append(group["box"])
            sims.append(sim)
            lightcones.append(0)
            names.append(name)
            colors.append(color)

    # Put fiducial first, then reverse the rest
    if group_key != "resolution":
        boxes = boxes[::-1]
        sims = sims[::-1]
        lightcones = lightcones[::-1]
        names = names[::-1]
        colors = colors[::-1]

    auto, cross = [], []
    amps, slopes = [], []
    chi2_auto, chi2_cross = [], []

    for box, sim, lc in zip(boxes, sims, lightcones):
        amp, slope = load_mle(box, sim, iz, lc)
        amps.append(amp)
        slopes.append(slope)

        c2a, c2c = load_mle_chi2(box, sim, iz, lc)
        chi2_auto.append(c2a)
        chi2_cross.append(c2c)

        auto.append(load_spectrum(box, sim, iz, lc, amp, slope, "auto", ell_namaster, shot_noise))
        cross.append(load_spectrum(box, sim, iz, lc, amp, slope, "cross", ell_namaster, shot_noise))

    return {
        "boxes": boxes,
        "sims": sims,
        "lightcones": lightcones,
        "names": names,
        "colors": colors,
        "auto": auto,
        "cross": cross,
        "chi2_auto": chi2_auto,
        "chi2_cross": chi2_cross,
        "title": group["panel_title"],
    }


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
    ell_namaster = ells[ell_200_mask]

    obs = np.loadtxt(
        f"./unWISExLens_lklh/data/v1.0/bandpowers/unWISExACT-DR6_{iz.lower()}_baseline_Clgg+Clkk+Clkg.dat",
        usecols=(0, 1, 2, 3),
    )[ell_200_mask, :].reshape(-1, 4)

    planck = np.loadtxt(
        f"./unWISExLens_lklh/data/v1.0/bandpowers/unWISExPlanck-PR4_{iz.lower()}_baseline_Clgg+Clkk+Clkg.dat",
        usecols=(0, 1, 2, 3),
    )
    planck = planck[np.where(planck[:, 0] > 200)[0], :].reshape(-1, 4)

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

    planck_ell_mask = np.where(
        np.loadtxt(
            f"./unWISExLens_lklh/data/v1.0/bandpowers/unWISExPlanck-PR4_{iz.lower()}_baseline_Clgg+Clkk+Clkg.dat",
            usecols=0,
        ) > 200
    )[0]

    planck_std_auto = np.sqrt(planck_var[: len(planck_var) // 2])[planck_ell_mask]
    planck_std_cross = np.sqrt(planck_var[len(planck_var) // 2 :])[planck_ell_mask]

    return ell_namaster, obs, planck, obs_std_auto, obs_std_cross, planck_std_auto, planck_std_cross


def get_fiducial(group, spectra_key):
    idx = [i for i, b in enumerate(group["boxes"]) if b == "L1000N1800"][0]
    return idx, np.asarray(group[spectra_key][idx], dtype=float)


def plot_standard_group(ax, ax_ratio, ell, group, spectra_key, ratio_ylim, iz):
    fid_idx, fid = get_fiducial(group, spectra_key)

    ax_ratio.axhline(1.0, color="k", linestyle="--", alpha=0.6)

    ax.plot(ell, fid, color=group["colors"][fid_idx], lw=1.8, label=group["names"][fid_idx])

    fid_std = load_fiducial_std(iz, spectra_key)
    ax.fill_between(ell, fid - fid_std, fid + fid_std, color=group["colors"][fid_idx], alpha=0.25, linewidth=0)

    ax_ratio.plot(ell, fid / fid, color=group["colors"][fid_idx], alpha=0.9)
    ax_ratio.fill_between(ell, (fid - fid_std) / fid, (fid + fid_std) / fid, color=group["colors"][fid_idx] if "colors" in group else "#117733", alpha=0.25, linewidth=0)

    for i, spec in enumerate(group[spectra_key]):
        if i == fid_idx:
            continue

        spec = np.asarray(spec, dtype=float)
        line, = ax.plot(ell, spec, color=group["colors"][i], alpha=0.9, label=group["names"][i])
        ax_ratio.plot(ell, spec / fid, color=line.get_color(), alpha=0.9)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(200, 4000)

    ax_ratio.set_xscale("log")
    ax_ratio.set_xlim(200, 4000)
    ax_ratio.set_ylim(*ratio_ylim)
    ax_ratio.grid(alpha=0.25)

    return fid


def plot_resolution_group(ax, ax_ratio, ell, group, spectra_key, ratio_ylim, iz):
    fid_idx, fid = get_fiducial(group, spectra_key)
    l1000m8_idx = [i for i, b in enumerate(group["boxes"]) if b == "L1000N3600"][0]
    l2800_idx = [i for i, b in enumerate(group["boxes"]) if b == "L2800N5040"]

    l1000m8 = np.asarray(group[spectra_key][l1000m8_idx], dtype=float)
    l2800 = np.asarray([group[spectra_key][i] for i in l2800_idx], dtype=float)

    mean = np.mean(l2800, axis=0)
    lo = np.min(l2800, axis=0)
    hi = np.max(l2800, axis=0)

    ax_ratio.axhline(1.0, color="k", linestyle="--", alpha=0.6)

    ax.plot(ell, fid, color="#117733", label="L1_m9")
    fid_std = load_fiducial_std(iz, spectra_key)
    ax.fill_between(ell, fid - fid_std, fid + fid_std, color="#117733", alpha=0.25, linewidth=0)

    ax.plot(ell, l1000m8, color="#CC6677", label="L1_m8")
    ax.plot(ell, mean, color="#332288", label="L2p8_m9")
    ax.fill_between(ell, lo, hi, color="#332288", alpha=0.25, linewidth=0)

    ax_ratio.plot(ell, fid / fid, color="#117733")#, lw=1.8)
    ax_ratio.plot(ell, l1000m8 / fid, color="#CC6677")#, lw=1.8)
    ax_ratio.plot(ell, mean / fid, color="#332288")#, lw=1.8)
    ax_ratio.fill_between(ell, (fid - fid_std) / fid, (fid + fid_std) / fid, color="#117733", alpha=0.25, linewidth=0)
    ax_ratio.fill_between(ell, lo / fid, hi / fid, color="#332288", alpha=0.25, linewidth=0)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(200, 4000)

    ax_ratio.set_xscale("log")
    ax_ratio.set_xlim(200, 4000)
    ax_ratio.set_ylim(*ratio_ylim)
    ax_ratio.grid(alpha=0.25)

    return fid, mean


def add_observations(ax, ax_ratio, ell, fid, obs, planck, obs_std, planck_std, spectra_key):
    if spectra_key == "auto":
        obs_col = 1
        obs_label = r"ACT $\times$ unWISE"
        planck_label = r"Planck $\times$ unWISE"
    else:
        obs_col = 3
        obs_label = r"ACT $\times$ unWISE"
        planck_label = r"Planck $\times$ unWISE"

    y_obs = obs[:, obs_col] * 1e5
    y_obs_err = obs_std * 1e5

    ax.plot(obs[:, 0], y_obs, color="k", marker=".", markersize=4, linewidth=0, label=obs_label)
    ax.fill_between(obs[:, 0], y_obs + y_obs_err, y_obs - y_obs_err, color="k", alpha=0.2, linewidth=0)

    y_planck = planck[:, obs_col] * 1e5
    ax.plot(planck[:, 0], y_planck, color="r", marker=".", markersize=4, linewidth=0, label=planck_label)

    y_planck_err = planck_std * 1e5

    ax.fill_between(planck[:, 0], y_planck + y_planck_err, y_planck - y_planck_err, color="r", alpha=0.2, linewidth=0)

    fid_obs = fid
    fid_planck = np.interp(planck[:, 0], ell, fid)

    ax_ratio.plot(obs[:, 0], y_obs / fid_obs, color="k", marker=".", markersize=4, linewidth=0, alpha=0.8)
    ax_ratio.fill_between(
        obs[:, 0],
        (y_obs + y_obs_err) / fid_obs,
        (y_obs - y_obs_err) / fid_obs,
        color="k",
        alpha=0.2,
        linewidth=0,
    )

    ax_ratio.plot(planck[:, 0], y_planck / fid_planck, color="r", marker=".", markersize=4, linewidth=0, alpha=0.8)
    ax_ratio.fill_between(
        planck[:, 0],
        (y_planck + y_planck_err) / fid_planck,
        (y_planck - y_planck_err) / fid_planck,
        color="r",
        alpha=0.2,
        linewidth=0,
    )


def annotate_chi2(ax, group, spectra_key, x=0.1, y0=0.3, dy=0.045):
    if "L2800N5040" in group["boxes"]:
        return  # skip 2.8 Gpc lightcone panel for now

    chi2_key = "chi2_auto" if spectra_key == "auto" else "chi2_cross"
    chi2_vals = np.asarray(group[chi2_key], dtype=float)

    fid_idx = [i for i, b in enumerate(group["boxes"]) if b == "L1000N1800"][0]
    fid_chi2 = chi2_vals[fid_idx]

    for i in range(len(chi2_vals)):
        if i == fid_idx:
            text = rf"{chi2_vals[i]:.1f}"
        else:
            text = rf"({chi2_vals[i] - fid_chi2:.1f})"

        ax.text(
            x,
            y0 - dy*i, #* (len(chi2_vals) - 1 - i),
            text,
            transform=ax.transAxes,
            color=group["colors"][i],
            fontsize=8,
            ha="left",
            va="bottom",
        )


def annotate_resolution_chi2(ax, fid_chi2, l1000m8_chi2, l2800_chi2, x=0.1, y0=0.3, dy=0.045):
    ax.text(
        x,
        y0,
        rf"{fid_chi2:.1f}",
        transform=ax.transAxes,
        color="#117733",
        fontsize=8,
        ha="left",
        va="bottom",
    )

    ax.text(
        x,
        y0 - dy,
        rf"({l1000m8_chi2 - fid_chi2:.1f})",
        transform=ax.transAxes,
        color="#CC6677",
        fontsize=8,
        ha="left",
        va="bottom",
    )

    ax.text(
        x,
        y0 - dy*2,
        rf"({l2800_chi2 - fid_chi2:.1f})",
        transform=ax.transAxes,
        color="#332288",
        fontsize=8,
        ha="left",
        va="bottom",
    )


def make_overview(groups, ell, obs, planck, obs_std, planck_std, spectra_key, iz, file_ext):
    fig, axes = pb.subplots(
        4,
        2,
        figsize=(12, 10),
        sharex="col",
        gridspec_kw={
            "height_ratios": [3.0, 1.0, 3.0, 1.0],
            "hspace": 0.08,
            "wspace": 0.1,
        },
    )

    panel_map = {
        "resolution": (axes[0, 0], axes[1, 0]),
        "cosmology": (axes[0, 1], axes[1, 1]),
        "agn_feedback": (axes[2, 0], axes[3, 0]),
        "other_feedback": (axes[2, 1], axes[3, 1]),
    }

    fig.align_ylabels()

    if spectra_key == "auto":
        ylabel = r"$C_\ell^{\rm gg} \times 10^5$"
        bottom = 1e-2
        top = None
        ratio_ylim = (0.9, 1.1)
        outname = f"./Plots/halo_map_gg_power_spectrum_{iz}_overview.{file_ext}"
    else:
        ylabel = r"$C_\ell^{\kappa \rm g} \times 10^5$"
        bottom = 1e-4
        top = 4e-2
        ratio_ylim = (0.5, 1.5)
        outname = f"./Plots/halo_map_kg_power_spectrum_{iz}_overview.{file_ext}"

    for key, (ax, ax_ratio) in panel_map.items():
        group = groups[key]

        if key == "resolution":
            fid, l2800_mean = plot_resolution_group(ax, ax_ratio, ell, group, spectra_key, ratio_ylim, iz)
        else:
            fid = plot_standard_group(ax, ax_ratio, ell, group, spectra_key, ratio_ylim, iz)
            l2800_mean = None

        add_observations(ax, ax_ratio, ell, fid, obs, planck, obs_std, planck_std, spectra_key)

        if key == "resolution":
            if spectra_key == "auto":
                obs_y = obs[:, 1] * 1e5
                obs_err_y = obs_std * 1e5
                chi2_key = "chi2_auto"
            else:
                obs_y = obs[:, 3] * 1e5
                obs_err_y = obs_std * 1e5
                chi2_key = "chi2_cross"

            l2800_chi2 = compute_chi2(l2800_mean, obs_y, obs_err_y)

            fid_idx = [i for i, b in enumerate(group["boxes"]) if b == "L1000N1800"][0]
            fid_chi2 = group[chi2_key][fid_idx]
            l1000m8_idx = [i for i, b in enumerate(group["boxes"]) if b == "L1000N3600"][0]
            l1000m8_chi2 = group[chi2_key][l1000m8_idx]

            annotate_resolution_chi2(
                ax,
                fid_chi2=fid_chi2,
                l1000m8_chi2=l1000m8_chi2,
                l2800_chi2=l2800_chi2,
            )
        else:
            annotate_chi2(ax, group, spectra_key)

        ax.set_ylabel(ylabel)
        ax.set_ylim(bottom=bottom, top=top)
        ax.legend(fontsize=8, loc="best", title=f"{iz} sample", frameon=False)

        ax_ratio.set_ylabel("Ratio")
        if key in ["agn_feedback", "other_feedback"]:
            ax_ratio.set_xlabel(r"Multipole moment $\ell$")
        else:
            ax_ratio.set_xlabel("")

        # Remove y-axis labels/ticks from right column
        if key in ["cosmology", "other_feedback"]:
            ax.set_ylabel("")
            ax_ratio.set_ylabel("")
            ax.tick_params(axis="y", labelleft=False)
            ax_ratio.tick_params(axis="y", labelleft=False)

    Path("./Plots").mkdir(exist_ok=True)
    pb.savefig(outname, dpi=400, bbox_inches="tight")
    pb.close(fig)

    print(f"Saved {outname}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("iz", choices=["Blue", "Green"])
    parser.add_argument("--lc-count", type=int, default=8, help="Number of L2800 lightcones.")
    parser.add_argument("--file", choices=["png", "pdf"], default="png")
    parser.add_argument("--shot-noise", action="store_true", help="Use auto spectra with shot noise included.")
    args = parser.parse_args()

    ell, obs, planck, obs_std_auto, obs_std_cross, planck_std_auto, planck_std_cross = load_binning_and_data(args.iz)

    groups = {
        key: load_group(
            key,
            iz=args.iz,
            lc_count=args.lc_count,
            ell_namaster=ell,
            shot_noise=args.shot_noise,
        )
        for key in ["resolution", "cosmology", "agn_feedback", "other_feedback"]
    }

    make_overview(groups, ell, obs, planck, obs_std_auto, planck_std_auto, "auto", args.iz, args.file)
    make_overview(groups, ell, obs, planck, obs_std_cross, planck_std_cross, "cross", args.iz, args.file)

if __name__ == "__main__":
    main()