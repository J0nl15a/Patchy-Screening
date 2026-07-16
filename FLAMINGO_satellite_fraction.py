#!/usr/bin/env python3
"""
Compute satellite fraction vs halo mass and redshift from the full HBT lightcone shells.

No dN/dz sampling.
No stellar-mass cut.
Only:
    box      = L1000N1800
    simname  = HYDRO_FIDUCIAL
    lightcone= 0
    HBT lightcones

Outputs:
    ./data_files/satellite_fraction_full_lightcone/L1000N1800/HYDRO_FIDUCIAL/lightcone0/
        satellite_fraction_vs_halo_mass.txt
        satellite_fraction_vs_redshift.txt
        satellite_fraction_full_lightcone.png
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import matplotlib.pyplot as plt


# -------------------------------------------------------------------
# User settings
# -------------------------------------------------------------------

BOX = "L1000N1800"
SIM = "HYDRO_FIDUCIAL"
LIGHTCONE = 0

# File with shell indices / redshift information.
# Expected columns follow your earlier convention:
#   col 0 = shell/snapshot-like index
#   col 1 = z_min
#   col 2 = z_mid
#   col 3 = z_max
#
# Edit this path:
REDSHIFT_SHELL_FILE = (
    "./data_files/halo_redshifts/"
    "L1000N1800/HYDRO_FIDUCIAL/lightcone0/"
    "FLAMINGO_halo_redshift_values.txt"
)

OUTDIR = Path(
    "./data_files/satellite_fraction_full_lightcone/"
    f"{BOX}/{SIM}/lightcone{LIGHTCONE}"
)

PLOTDIR = Path("./Plots")

ZMAX = 3.0
N_HALO_MASS_BINS = 100

SATELLITE_FLAG = 0
CENTRAL_FLAG = 1

# For L1000N1800 HBT lightcones from your load_halo_data logic
SNAP_MAX = 77
HALO_LC_DIR = "hbt_lightcone_halos"

LIGHTCONE_TEMPLATE = (
    "/cosma8/data/dp004/flamingo/Runs/"
    "{box}/{sim}/{halo_lc_dir}/lightcone{lightcone}/"
    "lightcone_halos_{shell_snap:04d}.hdf5"
)

SOAP_HBT_TEMPLATE = (
    "/cosma8/data/dp004/flamingo/Runs/"
    "{box}/{sim}/SOAP-HBT/halo_properties_{snap:04d}.hdf5"
)

APPLY_STELLAR_MASS_CUT = True

# Use log10 stellar mass threshold
LOG10_MSTAR_MIN = 9.92

# Or set to None to disable
# LOG10_MSTAR_MIN = None


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------

def binomial_fraction_error(n_sat: np.ndarray, n_total: np.ndarray):
    n_sat = np.asarray(n_sat, dtype=float)
    n_total = np.asarray(n_total, dtype=float)

    frac = np.full_like(n_total, np.nan, dtype=float)
    err = np.full_like(n_total, np.nan, dtype=float)

    valid = n_total > 0
    frac[valid] = n_sat[valid] / n_total[valid]
    err[valid] = np.sqrt(frac[valid] * (1.0 - frac[valid]) / n_total[valid])

    return frac, err


def read_shell_table(path: str | Path) -> np.ndarray:
    shell_table = np.loadtxt(path)

    if shell_table.ndim == 1:
        shell_table = shell_table.reshape(1, -1)

    if shell_table.shape[1] < 3:
        raise ValueError(
            "Expected redshift shell file to have at least 3 columns, "
            "with z_mid in column 2."
        )

    return shell_table


def load_lightcone_shell(shell_snap: int):
    path = LIGHTCONE_TEMPLATE.format(
        box=BOX,
        sim=SIM,
        halo_lc_dir=HALO_LC_DIR,
        lightcone=LIGHTCONE,
        shell_snap=shell_snap,
    )

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    with h5py.File(path, "r") as f:
        ids = f["InputHalos/HaloCatalogueIndex"][...]
        snapnums = f["Lightcone/SnapshotNumber"][...]
        redshifts = f["Lightcone/Redshift"][...]

    return ids, snapnums, redshifts


def load_soap_hbt_snapshot(snap: int):
    path = SOAP_HBT_TEMPLATE.format(
        box=BOX,
        sim=SIM,
        snap=snap,
    )

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    with h5py.File(path, "r") as f:
        ids = f["InputHalos/HaloCatalogueIndex"][...]
        structure_type = f["InputHalos/IsCentral"][...]
        mvir = f["SO/500_crit/TotalMass"][...] * 1e10
        mstar = f["ExclusiveSphere/50kpc/StellarMass"][...] * 1e10
        host_halo_index = f["SOAP/HostHaloIndex"][...]

    return ids, structure_type, mvir, mstar, host_halo_index


def make_snapshot_lookup(ids: np.ndarray) -> dict[int, int]:
    """
    Map InputHalos/HaloCatalogueIndex -> row index in SOAP-HBT arrays.
    """
    return {int(halo_id): i for i, halo_id in enumerate(ids)}


def host_fixed_mvir_for_lightcone_ids(
    lc_ids: np.ndarray,
    soap_ids: np.ndarray,
    structure_type: np.ndarray,
    mvir: np.ndarray,
    mstar: np.ndarray,
    host_halo_index: np.ndarray,
):
    """
    Return structure type, host-fixed mvir, and stellar mass for the lightcone objects.
    """
    id_to_row = make_snapshot_lookup(soap_ids)

    rows = np.array(
        [id_to_row.get(int(halo_id), -1) for halo_id in lc_ids],
        dtype=np.int64,
    )

    valid = rows >= 0

    lc_structure = np.full(len(lc_ids), -999, dtype=structure_type.dtype)
    lc_mvir = np.full(len(lc_ids), np.nan, dtype=float)
    lc_mstar = np.full(len(lc_ids), np.nan, dtype=float)

    lc_structure[valid] = structure_type[rows[valid]]
    lc_mvir[valid] = mvir[rows[valid]]
    lc_mstar[valid] = mstar[rows[valid]]

    sat_mask = valid & (lc_structure == SATELLITE_FLAG)

    if np.any(sat_mask):
        sat_rows = rows[sat_mask]
        host_rows = host_halo_index[sat_rows].astype(np.int64)

        good_host_index = (
            (host_rows >= 0)
            & (host_rows < len(mvir))
        )

        fixed_values = lc_mvir[sat_mask].copy()

        if np.any(good_host_index):
            candidate_host_rows = host_rows[good_host_index]
            host_is_central = structure_type[candidate_host_rows] == CENTRAL_FLAG

            fixed_values[good_host_index] = np.where(
                host_is_central,
                mvir[candidate_host_rows],
                fixed_values[good_host_index],
            )

        lc_mvir[sat_mask] = fixed_values

    return lc_structure, lc_mvir, lc_mstar, valid


# -------------------------------------------------------------------
# Main calculation
# -------------------------------------------------------------------

def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    # PLOTDIR.mkdir(parents=True, exist_ok=True)

    shell_table = read_shell_table(REDSHIFT_SHELL_FILE)

    all_mvir = []
    all_mstar = []
    all_is_satellite = []
    redshift_rows = []

    for row in shell_table:
        shell_index = int(row[0])
        z_mid = float(row[2])

        if z_mid > ZMAX:
            continue

        shell_snap = SNAP_MAX - shell_index

        try:
            lc_ids, lc_snapnums, lc_redshifts = load_lightcone_shell(shell_snap)
        except FileNotFoundError as err:
            print(f"[WARN] Missing lightcone shell: {err}. Skipping.")
            continue

        unique_snaps = np.unique(lc_snapnums)

        shell_total = 0
        shell_sat = 0

        for snap in unique_snaps:
            snap = int(snap)
            in_snap = lc_snapnums == snap

            try:
                soap_ids, structure_type, mvir, mstar, host_halo_index = load_soap_hbt_snapshot(snap)
            except FileNotFoundError as err:
                print(f"[WARN] Missing SOAP-HBT snapshot: {err}. Skipping snap={snap}.")
                continue

            lc_structure, lc_mvir, lc_mstar, valid = host_fixed_mvir_for_lightcone_ids(
                lc_ids=lc_ids[in_snap],
                soap_ids=soap_ids,
                structure_type=structure_type,
                mvir=mvir,
                mstar=mstar,
                host_halo_index=host_halo_index,
            )

            good = (
                valid
                & np.isfinite(lc_mvir)
                & (lc_mvir > 0)
                & np.isfinite(lc_mstar)
                & (lc_mstar > 0)
                & (
                    (lc_structure == CENTRAL_FLAG)
                    | (lc_structure == SATELLITE_FLAG)
                )
            )

            if APPLY_STELLAR_MASS_CUT and LOG10_MSTAR_MIN is not None:
                good &= np.log10(lc_mstar) >= LOG10_MSTAR_MIN

            if not np.any(good):
                continue

            is_sat = lc_structure[good] == SATELLITE_FLAG

            all_mvir.append(lc_mvir[good])
            all_mstar.append(lc_mstar[good])
            all_is_satellite.append(is_sat)

            shell_total += int(np.sum(good))
            shell_sat += int(np.sum(is_sat))

        frac_z, err_z = binomial_fraction_error(
            np.array([shell_sat]),
            np.array([shell_total]),
        )

        redshift_rows.append(
            [
                shell_index,
                shell_snap,
                z_mid,
                shell_total,
                shell_sat,
                frac_z[0],
                err_z[0],
            ]
        )

        print(
            f"shell_index={shell_index:3d}, "
            f"shell_snap={shell_snap:4d}, "
            f"z_mid={z_mid:.3f}, "
            f"N={shell_total}, "
            f"N_sat={shell_sat}, "
            f"f_sat={frac_z[0]:.4f}"
        )

    if len(all_mvir) == 0 or len(all_mstar) == 0:
        raise RuntimeError("No valid galaxies found. Check shell indices, paths, and z range.")


    all_mvir = np.concatenate(all_mvir)
    all_mstar = np.concatenate(all_mstar)
    all_is_satellite = np.concatenate(all_is_satellite)

    log_mvir = np.log10(all_mvir)
    log_mstar = np.log10(all_mstar)

    halo_mass_hist, halo_mass_bins = np.histogram(
        log_mvir,
        bins=N_HALO_MASS_BINS,
    )

    halo_mass_hist_sat, _ = np.histogram(
        log_mvir[all_is_satellite],
        bins=halo_mass_bins,
    )

    stellar_mass_hist, stellar_mass_bins = np.histogram(
        log_mstar,
        bins=N_HALO_MASS_BINS,
    )

    stellar_mass_hist_sat, _ = np.histogram(
        log_mstar[all_is_satellite],
        bins=stellar_mass_bins,
    )

    satellite_fraction_mass, satellite_fraction_mass_err = binomial_fraction_error(
        halo_mass_hist_sat,
        halo_mass_hist,
    )

    satellite_fraction_stellar, satellite_fraction_stellar_err = binomial_fraction_error(
        stellar_mass_hist_sat,
        stellar_mass_hist,
    )

    halo_mass_bin_centres = 0.5 * (halo_mass_bins[1:] + halo_mass_bins[:-1])
    stellar_mass_bin_centres = 0.5 * (stellar_mass_bins[1:] + stellar_mass_bins[:-1])

    halo_mass_rows = np.column_stack(
        [
            halo_mass_bins[:-1],
            halo_mass_bins[1:],
            halo_mass_bin_centres,
            halo_mass_hist,
            halo_mass_hist_sat,
            satellite_fraction_mass,
            satellite_fraction_mass_err,
        ]
    )

    stellar_mass_rows = np.column_stack(
        [
            stellar_mass_bins[:-1],
            stellar_mass_bins[1:],
            stellar_mass_bin_centres,
            stellar_mass_hist,
            stellar_mass_hist_sat,
            satellite_fraction_stellar,
            satellite_fraction_stellar_err,
        ]
    )

    redshift_rows = np.asarray(redshift_rows)

    halo_mass_out = OUTDIR / "satellite_fraction_vs_halo_mass_mass_cut.txt"
    stellar_mass_out = OUTDIR / "satellite_fraction_vs_stellar_mass_mass_cut.txt"
    redshift_out = OUTDIR / "satellite_fraction_vs_redshift_mass_cut.txt"

    np.savetxt(
        halo_mass_out,
        halo_mass_rows,
        header=(
            "log10_mvir_bin_left log10_mvir_bin_right log10_mvir_bin_centre "
            "n_total n_satellite satellite_fraction satellite_fraction_binomial_error"
        ),
    )

    np.savetxt(
        stellar_mass_out,
        stellar_mass_rows,
        header=(
            "log10_mstar_bin_left log10_mstar_bin_right log10_mstar_bin_centre "
            "n_total n_satellite satellite_fraction satellite_fraction_binomial_error"
        ),
    )

    np.savetxt(
        redshift_out,
        redshift_rows,
        header=(
            "shell_index shell_snap z_mid n_total n_satellite "
            "satellite_fraction satellite_fraction_binomial_error"
        ),
    )

    print(f"Saved {halo_mass_out}")
    print(f"Saved {stellar_mass_out}")
    print(f"Saved {redshift_out}")

    # fig, axs = plt.subplots(1, 2, figsize=(8.0, 3.4))

    # axs[0].errorbar(
    #     halo_mass_bin_centres,
    #     satellite_fraction_mass,
    #     yerr=satellite_fraction_mass_err,
    #     fmt="o",
    #     markersize=3,
    #     linewidth=0.8,
    #     color="black",
    # )

    # axs[0].set_xlabel(r"Halo Mass [$\log_{10}(M/M_\odot)$]")
    # axs[0].set_ylabel("Satellite Fraction")
    # axs[0].set_ylim(0.0, 1.0)
    # axs[0].grid(alpha=0.3)

    # z_mid = redshift_rows[:, 2]
    # f_sat_z = redshift_rows[:, 5]
    # f_sat_z_err = redshift_rows[:, 6]

    # order = np.argsort(z_mid)

    # axs[1].errorbar(
    #     z_mid[order],
    #     f_sat_z[order],
    #     yerr=f_sat_z_err[order],
    #     fmt="o",
    #     markersize=3,
    #     linewidth=0.8,
    #     color="black",
    # )

    # axs[1].set_xlabel("Redshift")
    # axs[1].set_ylabel("Satellite Fraction")
    # axs[1].set_ylim(0.0, 1.0)
    # axs[1].set_xlim(0.0, ZMAX)
    # axs[1].grid(alpha=0.3)

    # fig.tight_layout()

    # plot_out = PLOTDIR / "satellite_fraction_full_lightcone_L1000N1800_HYDRO_FIDUCIAL_lightcone0.png"
    # fig.savefig(plot_out, dpi=400, bbox_inches="tight")
    # plt.close(fig)

    # print(f"Saved {plot_out}")


if __name__ == "__main__":
    main()