from joblib import Parallel, delayed
import numpy as np
import pandas as pd
from imp_patchy_screening import patchyScreening

def halo_sampling(simname, z_sample, mass_cut, ncpu):
    sim_list = ['HYDRO_FIDUCIAL','HYDRO_PLANCK','HYDRO_PLANCK_LARGE_NU_FIXED','HYDRO_PLANCK_LARGE_NU_VARY','HYDRO_STRONG_AGN','HYDRO_WEAK_AGN','HYDRO_LOW_SIGMA8','HYDRO_STRONGER_AGN','HYDRO_JETS','HYDRO_STRONGEST_AGN','HYDRO_STRONG_SUPERNOVA','HYDRO_STRONGER_AGN_STRONG_SUPERNOVA','HYDRO_STRONG_JETS']

    try:
        isim = int(simname)
        simname = sim_list[isim]
    except (ValueError, IndexError):
        simname = str(simname)
    z_sample = str(z_sample)
    im = float(mass_cut)
    im_name = f"{float(mass_cut):.1f}".replace('.', 'p')

    halo_z_bins = np.genfromtxt(
        '/cosma8/data/dp004/dc-conl1/FLAMINGO/patchy_screening/data_files/FLAMINGO_halo_redshift_values.txt',
        dtype=[('i',   'i4'),
               ('z1', 'f8'),
               ('z2', 'f8')],
        delimiter=None
    )

    dndz_sample = np.loadtxt(f'/cosma8/data/dp004/dc-conl1/FLAMINGO/patchy_screening/data_files/dndz_galaxies_sampled_{simname}_{z_sample}_{im_name}.txt')

    # dispatch in parallel
    results = Parallel(n_jobs=int(ncpu),   # adjust to your cores
                       backend='loky')(
    delayed(process_snapshot)(simname, int(i), im, halo_z_bins, dndz_sample)
    for i in halo_z_bins['i']
    )

    print(results)

    # concatenate once at the end
    dfs = [df for df in results if df is not None]
    sampled_halo_data = pd.concat(dfs, ignore_index=True)

    mvir=np.asarray(sampled_halo_data.m_vir)
    nhalo=len(mvir)
    print(nhalo)

    # write out
    sampled_halo_data.to_parquet(f'/cosma8/data/dp004/dc-conl1/FLAMINGO/patchy_screening/data_files/sampled_halo_data_{simname}_{z_sample}_{im_name}.parquet',  compression='snappy', index=False)
    return

def process_snapshot(sim, i, im, halo_z_bins, dndz_sample):
    """Process a single redshift‐bin index i.  
       Returns (subdf_blue, subdf_green) or None on failure/skip."""
    if halo_z_bins['z1'][i] > 3.0:
        return None

    ps = patchyScreening(sim, i,
                         10**np.array(im),  # or however you pass
                         0, 1, 0,
                         lightcone_method=('FULL','shell'))
    ps.filter_stellar_mass()

    nsamp = int(dndz_sample[i][2])
    print(i, nsamp)

    subdf = ps.merge.sample(n=nsamp, random_state=1000)
    print(i, subdf)

    return subdf


if __name__ == "__main__":
    import sys
    halo_sampling(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[1])
