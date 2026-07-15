import numpy as np, healpy as hp, matplotlib.pyplot as plt
from pathlib import Path
import sys


def plot_shell_DM_tau_relations_from_diagnostics(
    box,
    sim,
    lightcone=0,
    scale_factor=True
):

    base_path = Path(
        f'/cosma8/data/dp004/dc-conl1/FLAMINGO/patchy_screening/data_files/DM_maps/{box}/{sim}/lightcone{lightcone}/'
    )

    suffix = "_scale_factor" if scale_factor else ""

    shell_path = base_path / "shells"

    if box == 'L1000N1800':
        max_redshift = '3'
    elif box == 'L2800N5040' and sim == 'HYDRO_FIDUCIAL':
        max_redshift = '5'
    else:
        print("Lightcone redshift file not available for this box/simulation combination.")
        sys.exit()

    lightcone_shell_redshifts = np.loadtxt(
        f'/cosma8/data/dp004/flamingo/Runs/{box}/{sim}/shell_redshifts_z{max_redshift}.txt',
        skiprows=0,
        delimiter=','
    )

    redshifts = []
    DM_means = []
    tau_means = []

    DM = 0
    tau = 0

    for i in range(len(lightcone_shell_redshifts)):

        z_min = lightcone_shell_redshifts[i, 0]
        z_max = lightcone_shell_redshifts[i, 1]
        z_mid = 0.5 * (z_min + z_max)

        if z_max > 3.0:
            break

        DM_file = shell_path / f"DM_map_shell_{i}{suffix}.fits"
        tau_file = shell_path / f"tau_map_shell_{i}{suffix}.fits"

        if not DM_file.exists():
            print(f"Missing file: {DM_file}")
            continue

        if not tau_file.exists():
            print(f"Missing file: {tau_file}")
            continue

        DM += hp.read_map(DM_file, verbose=False)
        tau += hp.read_map(tau_file, verbose=False)

        redshifts.append(z_mid)

        DM_means.append(np.mean(DM))

        tau_means.append(np.mean(tau))

    redshifts = np.array(redshifts)
    DM_means = np.array(DM_means) / 3.0856775814913673e18
    tau_means = np.array(tau_means)

    diagnostics_file = base_path / f"shell_diagnostics{suffix}.txt"

    if not diagnostics_file.exists():
        print(f"Diagnostics file not found: {diagnostics_file}")
        sys.exit()

    diagnostics = np.loadtxt(diagnostics_file)

    shell_index = diagnostics[:, 0].astype(int)
    redshift = diagnostics[:, 1]

    mean_DM = diagnostics[:, 2]

    mean_DM /= 3.0856775814913673e18

    mean_tau = diagnostics[:, 4]

    z_PL = np.asarray([
        0.11, 0.40, 0.58, 0.72, 0.90, 1.0, 1.166, 1.487, 1.849, 2.13,
        2.531, 2.851, 3.173, 3.494, 3.815, 4.136, 4.376, 4.657, 5.218
    ])

    tau_PL = np.asarray([
        2.9e-4, 1.16e-3, 1.74e-3, 2.03e-3, 2.90e-3, 3.48e-3, 3.329e-3,
        5.305e-3, 7.284e-3, 9.259e-3, 1.157e-2, 1.42e-2, 1.617e-2,
        1.848e-2, 2.111e-2, 2.341e-2, 2.571e-2, 2.801e-2, 3.294e-2
    ])

    DM_z_TNG = np.loadtxt('/cosma8/data/dp004/dc-conl1/FLAMINGO/patchy_screening/data_files/DM_maps/TNG_DM_z.txt', delimiter=' ', usecols=(0, 1))

    plot_path = Path('/cosma8/data/dp004/dc-conl1/FLAMINGO/patchy_screening/Plots/')
    plot_path.mkdir(parents=True, exist_ok=True)

    plt.figure()
    plt.plot(DM_z_TNG[:, 0], DM_z_TNG[:, 1], label='TNG DM(z)', color='green')
    plt.plot(redshift, np.cumsum(mean_DM), label='FLAMINGO shell means', color='blue')
    plt.plot(redshifts, DM_means, label='FLAMINGO shells', color='orange')
    plt.xlabel("Redshift")
    plt.ylabel("Mean DM")
    plt.tight_layout()
    plt.legend()
    plt.savefig(plot_path / f"DM_mean_vs_redshift{suffix}.png", dpi=300)
    plt.close()


    plt.figure()
    plt.plot(z_PL, tau_PL, label='Planck 2018', color='red')
    plt.plot(redshift, np.cumsum(mean_tau), label='FLAMINGO shell means', color='blue')
    plt.plot(redshifts, tau_means, label='FLAMINGO shells', color='orange')
    plt.xlabel("Redshift")
    plt.ylabel("Mean tau")
    plt.tight_layout()
    plt.legend()
    plt.savefig(plot_path / f"tau_mean_vs_redshift{suffix}.png", dpi=300)
    plt.close()

    return shell_index, redshift, mean_DM, mean_tau


if __name__ == "__main__":

    results = plot_shell_DM_tau_relations_from_diagnostics(
        'L1000N1800',
        'HYDRO_FIDUCIAL',
        lightcone=0,
        scale_factor=False
    )