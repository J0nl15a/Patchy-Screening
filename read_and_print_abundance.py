#!/usr/bin/env python3

import argparse
from pathlib import Path
import numpy as np, pandas as pd



def name_float(x, mle=False):
    if mle:
        return f"{x:.3f}".replace(".", "p")
    return f"{x:.1f}".replace(".", "p")


def read_total_abundance(filename):
    with open(filename, "r") as f:
        first_line = f.readline().strip()

    prefix = "#Total number of sampled galaxies:"

    if not first_line.startswith(prefix):
        raise ValueError(f"Unexpected first line in {filename}:\n{first_line}")

    return int(first_line.split(":")[-1].strip())


def print_satellite_fraction(catalogue):

    if "Structuretype" not in catalogue.columns:
        raise ValueError("Catalogue does not contain 'Structuretype'")

    n_total = len(catalogue)
    n_sat = np.count_nonzero(
        catalogue["Structuretype"].to_numpy() == 0
    )
    n_cen = np.count_nonzero(
        catalogue["Structuretype"].to_numpy() == 1
    )

    if n_total == 0:
        raise ValueError("Catalogue is empty")

    f_sat = n_sat / n_total

    print(f"Total galaxies     : {n_total:,}")
    print(f"Central galaxies   : {n_cen:,}")
    print(f"Satellite galaxies : {n_sat:,}")
    print(f"Satellite fraction : {f_sat:.6f}")
    print(f"Satellite fraction : {100*f_sat:.3f}%")

    return f_sat


def main():
    parser = argparse.ArgumentParser(description="Print total abundance from MLE dndz files.")

    parser.add_argument("box", type=str)
    # parser.add_argument("sim", type=str)
    parser.add_argument("sample", type=str, choices=["Blue", "Green"])
    # parser.add_argument("--lightcone", type=int, default=0)
    args = parser.parse_args()

    sims = {'L1000N1800': ['HYDRO_FIDUCIAL',
                           'HYDRO_LOW_SIGMA8','HYDRO_LOW_SIGMA8_STRONGEST_AGN','HYDRO_PLANCK','HYDRO_PLANCK_LARGE_NU_FIXED','HYDRO_PLANCK_LARGE_NU_VARY',
                           'HYDRO_WEAK_AGN','HYDRO_STRONG_AGN','HYDRO_STRONGER_AGN','HYDRO_STRONGEST_AGN',
                           'HYDRO_STRONG_SUPERNOVA','HYDRO_JETS_published','HYDRO_STRONG_JETS_published'],
            'L1000N3600': ['HYDRO_FIDUCIAL'],
            'L2800N5040': ['HYDRO_FIDUCIAL']}

    for s in sims[args.box]:

        if args.box == 'L2800N5040':
            lightcones = 8
        else:
            lightcones = 1

        for lc in range(lightcones):

            mle_file = Path(f"./data_files/mle_parameters/{args.box}/{s}/{args.sample}/lightcone{lc}/mle_values.txt")

            if not mle_file.exists():
                raise FileNotFoundError(mle_file)

            amp = float(np.loadtxt(mle_file, usecols=1, skiprows=6, max_rows=1, delimiter="="))
            slope = float(np.loadtxt(mle_file, usecols=1, skiprows=7, max_rows=1, delimiter="="))
            likelihood = float(np.loadtxt(mle_file, usecols=1, skiprows=0, max_rows=1, delimiter="="))
            chi2 = float(np.loadtxt(mle_file, usecols=1, skiprows=1, max_rows=1, delimiter="="))
                        

            amp_name = name_float(amp, mle=True)
            slope_name = name_float(slope, mle=True)

            dndz_file = Path(f"./data_files/dndz_samples/{args.box}/{s}/{args.sample}/lightcone{lc}/dndz_galaxies_sampled_{amp_name}_{slope_name}.txt")

            if not dndz_file.exists():
                raise FileNotFoundError(dndz_file)

            abundance = read_total_abundance(dndz_file)

            print(f"Box       : {args.box}")
            print(f"Simulation: {s}")
            print(f"Sample    : {args.sample}")
            print(f"Lightcone : {lc}")
            print(f"MLE amp   : {amp:.3f}")
            print(f"MLE slope : {slope:.3f}")
            print(f"Abundance : {abundance:,}")
            print(f"log_10 Abundance : {np.log10(abundance):.3f}")
            print(f"Likelihood : {likelihood:.3f}")
            print(f"chi^2 : {chi2:.2f}")
            print(f"Reduced chi^2 : {(chi2/55):.2f}")

            input_path = Path(f"./data_files/mock_halo_catalogs/{args.box}/{s}/{args.sample}/lightcone{lc}/sampled_halo_data_{amp_name}_{slope_name}.parquet")
            catalogue = pd.read_parquet(input_path)
            print_satellite_fraction(catalogue)
            print("\n")


if __name__ == "__main__":
    main()