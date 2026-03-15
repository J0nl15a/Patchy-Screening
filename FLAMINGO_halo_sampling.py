import os
from joblib import Parallel, delayed
import numpy as np, polars as pl
from imp_patchy_screening import patchyScreening
from FLAMINGO_halo_redshifts import multiprocess_z_bins
from pathlib import Path
import time

def halo_sampling(boxname, simname, z_sample, mass_cut, n_cut, ncpu, lightcone=0):
    box_list = ['L1000N1800', 'L2800N5040']
    sim_list = ['HYDRO_FIDUCIAL','HYDRO_PLANCK','HYDRO_PLANCK_LARGE_NU_FIXED','HYDRO_PLANCK_LARGE_NU_VARY','HYDRO_STRONG_AGN','HYDRO_WEAK_AGN','HYDRO_LOW_SIGMA8','HYDRO_STRONGER_AGN','HYDRO_JETS','HYDRO_STRONGEST_AGN','HYDRO_STRONG_SUPERNOVA','HYDRO_STRONGER_AGN_STRONG_SUPERNOVA','HYDRO_STRONG_JETS']

    try:
        box = int(boxname)
        boxname = box_list[box]
    except (ValueError, IndexError):
        boxname = str(boxname)

    try:
        isim = int(simname)
        simname = sim_list[isim]
    except (ValueError, IndexError):
        simname = str(simname)
        
    z_sample = str(z_sample)
    im = float(mass_cut)
    if round(im, 1) == im:
        im_name = f"{float(mass_cut):.1f}".replace('.', 'p')
    else:
        im_name = f"{float(mass_cut):.3f}".replace('.', 'p')
    slope = float(n_cut)
    if round(slope, 1) == slope:
        slope_name = f"{float(n_cut):.1f}".replace('.', 'p')
    else:
        slope_name = f"{float(n_cut):.3f}".replace('.', 'p')
    if slope < 0.0:
        slope_name = f"{slope_name}".replace('-', 'minus')

    lightcone = int(lightcone)

    path = f'/cosma8/data/dp004/dc-conl1/FLAMINGO/patchy_screening/data_files/mock_halo_catalogs/{boxname}/{simname}/{z_sample}/lightcone{lightcone}/sampled_halo_data_{im_name}_{slope_name}.parquet'
    output_path = Path(path)
    # Create parent directories if they don't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not os.path.isfile(f'/cosma8/data/dp004/dc-conl1/FLAMINGO/patchy_screening/data_files/halo_redshifts/{boxname}/{simname}/lightcone{lightcone}/FLAMINGO_halo_redshift_values.txt'):
        multiprocess_z_bins(ncpu, boxname, simname, lightcone=lightcone)

    halo_z_bins = np.genfromtxt(
        f'/cosma8/data/dp004/dc-conl1/FLAMINGO/patchy_screening/data_files/halo_redshifts/{boxname}/{simname}/lightcone{lightcone}/FLAMINGO_halo_redshift_values.txt',
        dtype=[('i',   'i4'),
               ('z_min', 'f8'),
               ('mid_z', 'f8'),
               ('z_max', 'f8')],
        delimiter=None
    )

    z_stellar_cuts = np.loadtxt(f'/cosma8/data/dp004/dc-conl1/FLAMINGO/patchy_screening/data_files/z_dependant_stellar_cuts/{boxname}/{z_sample}/z_stellar_cut_data_{im_name}_{slope_name}.txt')

    dndz_sample = np.loadtxt(f'/cosma8/data/dp004/dc-conl1/FLAMINGO/patchy_screening/data_files/dndz_samples/{boxname}/{simname}/{z_sample}/lightcone{lightcone}/dndz_galaxies_sampled_{im_name}_{slope_name}.txt')

    job_start_time = time.time()

    # dispatch in parallel
    # results = Parallel(n_jobs=int(ncpu), prefer='processes', verbose=10,   # adjust to your cores
    #                    backend='loky')(
    #                        delayed(process_snapshot)(boxname, simname, int(i), z_stellar_cuts, halo_z_bins, dndz_sample, lightcone)
    #                        for i in halo_z_bins['i']
    #                    )

    results = []
    for iz in halo_z_bins['i']:
        result = process_snapshot(boxname, simname, int(iz), z_stellar_cuts, halo_z_bins, dndz_sample, lightcone)
        if result is not None:
            results.append(result)

    print(results)

    # concatenate once at the end
    dfs = [df for df in results if df is not None]
    sampled_halo_data = pl.concat(dfs)

    mvir = sampled_halo_data['mvir'].to_numpy()
    nhalo = mvir.size
    print(nhalo)

    print(f"Total job time: {time.time() - job_start_time:.2f} seconds")

    sampled_halo_data.to_parquet(output_path, compression='snappy')
    return

def process_snapshot(box, sim, iz, stellar_cuts, halo_z_bins, dndz_sample, lightcone):
    """Process a single redshift‐bin index i.  
       Returns subdf or None on failure/skip."""
    if halo_z_bins['mid_z'][iz] > 3.0:
        return None

    im = stellar_cuts[iz][1]

    # ps = patchyScreening(box, sim, iz, im,  # or however you pass
    #                      0, 0, 1,
    #                      lightcone_method=('FULL','shell'), lightcone=lightcone)
    # ps.filter_stellar_mass()

    df = pl.read_parquet(
        f"/cosma8/data/dp004/dc-conl1/FLAMINGO/patchy_screening/data_files/shell_caches/{box}/{sim}/lightcone{lightcone}/shell_{iz:03d}.parquet"
    )
    df = df.filter(pl.col("mstar") >= 10**(float(im)))

    nsamp = int(dndz_sample[iz][1])
    print(iz, nsamp)

    subdf = df.sample(n=nsamp, replace=False, random_state=1000)
    print(iz, subdf)

    return subdf


if __name__ == "__main__":
    import sys
    halo_sampling(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6], sys.argv[1], lightcone=int(sys.argv[7]))
