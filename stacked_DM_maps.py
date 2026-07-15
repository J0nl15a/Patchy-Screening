import h5py, numpy as np, healpy as hp
from pathlib import Path
import sys
from joblib import Parallel, delayed

sample_redshift_shell = {'Blue':11, 'Green':21, 'Red':29}

def read_one_shell(i, box, sim, lightcone, map_dir, scale_factor):

    map_lightcone = (
        f'/cosma8/data/dp004/flamingo/Runs/{box}/{sim}/{map_dir}/'
        f'lightcone{lightcone}_shells/shell_{i}/'
        f'lightcone{lightcone}.shell_{i}.0.hdf5'
    )

    with h5py.File(map_lightcone, "r") as g:
        DM = (
            g["DM"][...] *
            g["DM"].attrs[
                "Conversion factor to CGS (not including cosmological corrections)"
            ]
        )
        redshift = g["DM"].attrs["Central redshift assumed for correction"]

    if scale_factor:
        DM *= (1 + redshift)

    return i, redshift, DM

def stack_DM_maps_z3(ncpu, box, sim, lightcone=0, scale_factor=False, save_shells=True):

    sigma_T = 6.6524587321e-25 # cm^2

    if box == 'L1000N1800' or box == 'L1000N3600':
        max_redshift = '3'
        map_dir = 'neutrino_corrected_maps'
    elif box == 'L2800N5040':
        max_redshift = '5'
        map_dir = 'neutrino_corrected_maps_downsampled_4096'
    else:
        print("Lightcone map not available for this box/simulation combination.")
        sys.exit()
    
    lightcone_shell_redshifts = np.loadtxt(f'/cosma8/data/dp004/flamingo/Runs/{box}/{sim}/shell_redshifts_z{max_redshift}.txt' if box != 'L2800N5040' else f'/cosma8/data/dp004/flamingo/Runs/{box}/{sim}/shell_redshifts.txt', 
                                           skiprows=0, delimiter=',')
    redshift_mask = lightcone_shell_redshifts[:,1] <= 3.0
    lightcone_shell_redshifts = lightcone_shell_redshifts[redshift_mask]

    output_path = Path(f'/cosma8/data/dp004/dc-conl1/FLAMINGO/patchy_screening/data_files/DM_maps/{box}/{sim}/lightcone{lightcone}')
    output_path.mkdir(parents=True, exist_ok=True)

    shell_output_path = output_path / "shells"
    shell_output_path.mkdir(parents=True, exist_ok=True)

    suffix = "_scale_factor" if scale_factor else ""

    # DM_total = 0
    # tau_total = 0
    diagnostics = []

    # results = Parallel(n_jobs=ncpu, verbose=10)(
    #     delayed(read_one_shell)(i, box, sim, lightcone, map_dir, scale_factor)
    #     for i in range(len(lightcone_shell_redshifts))
    # )

    DM_total = None
    tau_total = None

    for i in range(len(lightcone_shell_redshifts)):
        i, redshift, DM = read_one_shell(i, box, sim, lightcone, map_dir, scale_factor)

        if redshift > 3.0:
            break

        if DM_total is None:
            DM_total = np.zeros_like(DM)

        if tau_total is None:
            tau_total = np.zeros_like(DM)

        DM_total += DM

        tau = DM * sigma_T * (1+redshift)
        tau_total += tau

        print(i, redshift, np.mean(DM), np.sum(DM), np.mean(tau), np.sum(tau))
        diagnostics.append([i, redshift, np.mean(DM), np.sum(DM), np.mean(tau), np.sum(tau)])

        if save_shells:
            hp.write_map(f"{shell_output_path}/DM_map_shell_{i}{suffix}.fits", DM, overwrite=True)
            hp.write_map(f"{shell_output_path}/tau_map_shell_{i}{suffix}.fits", tau, overwrite=True)

    # tau_total = sigma_T * DM_total
    
    hp.write_map(f"{output_path}/stacked_DM_map_z3p0{suffix}.fits", DM_total, overwrite=True)
    hp.write_map(f"{output_path}/stacked_tau_map_z3p0_scale_factor.fits", tau_total, overwrite=True)

    np.savetxt(
        output_path / f"shell_diagnostics{suffix}.txt",
        np.array(diagnostics),
        header="shell redshift mean_DM sum_DM mean_tau_scale_factor sum_tau_scale_factor "
    )

    return DM_total, tau_total

if __name__ == "__main__":
    ncpu = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    box = str(sys.argv[2]) if len(sys.argv) > 2 else 'L1000N1800'
    sim = str(sys.argv[3]) if len(sys.argv) > 3 else 'HYDRO_FIDUCIAL'
    scale_factor = sys.argv[4].lower() in ("true", "1", "yes", "y") if len(sys.argv) > 4 else False
    save_shells = sys.argv[5].lower() in ("true", "1", "yes", "y") if len(sys.argv) > 5 else True
    lightcone = int(sys.argv[6]) if len(sys.argv) > 6 else 0

    DM_total, tau_total = stack_DM_maps_z3(ncpu, box, sim, lightcone=lightcone, scale_factor=scale_factor, save_shells=save_shells)
    print(DM_total, np.mean(DM_total))
    print(tau_total, np.mean(tau_total))
