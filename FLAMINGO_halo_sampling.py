from joblib import Parallel, delayed
import numpy as np
import pandas as pd
import h5py
from imp_patchy_screening import patchyScreening
import sys

halo_z_bins = np.genfromtxt(
    '/cosma8/data/dp004/dc-conl1/FLAMINGO/patchy_screening/data_files/FLAMINGO_halo_redshift_values.txt',
    dtype=[('i',   'i4'),
           ('z1', 'f8'),
           ('z2', 'f8')],
    delimiter=None
)

blue_sample = np.loadtxt('/cosma8/data/dp004/dc-conl1/FLAMINGO/patchy_screening/data_files/dndz_galaxies_sampled_Blue_ntotal.txt')
green_sample = np.loadtxt('/cosma8/data/dp004/dc-conl1/FLAMINGO/patchy_screening/data_files/dndz_galaxies_sampled_Green_ntotal.txt')

im = float(sys.argv[2])
im_name = f"{float(sys.argv[2]):.1f}".replace('.', 'p')

def process_snapshot(i):
    """Process a single redshift‐bin index i.  
       Returns (subdf_blue, subdf_green) or None on failure/skip."""
    if halo_z_bins['z1'][i] > 3.0:
        return None

    ps = patchyScreening(0, i,
                         10**np.array(im),  # or however you pass
                         0, 1, 0,
                         lightcone_method=('FULL','shell'))
    ps.filter_stellar_mass()
    #print(f"File /cosma8/data/dp004/flamingo/Runs/L1000N1800/HYDRO_FIDUCIAL/hbt_lightcone_halos/lightcone0/lightcone_halos_{77-i:04d}.hdf5 doesn't work.")

    # sample blue
    n_blue = int(blue_sample[i][2])
    subdf_blue = ps.merge.sample(n=n_blue, random_state=10)

    # sample green
    n_green = int(green_sample[i][2])
    subdf_green = ps.merge.sample(n=n_green, random_state=1000)

    return subdf_blue, subdf_green

# dispatch in parallel
results = Parallel(n_jobs=int(sys.argv[1]),   # adjust to your cores
                   backend='loky')(
    delayed(process_snapshot)(int(i))
    for i in halo_z_bins['i']
)

# filter out the None’s and unzip
blue_chunks  = [r[0] for r in results if r is not None]
green_chunks = [r[1] for r in results if r is not None]

# concatenate once at the end
sampled_halo_data_blue  = pd.concat(blue_chunks,  ignore_index=True)
sampled_halo_data_green = pd.concat(green_chunks, ignore_index=True)

mvir_blue=np.asarray(sampled_halo_data_blue.m_vir)
nhalo_blue=len(mvir_blue)
print(nhalo_blue)
mvir_green=np.asarray(sampled_halo_data_green.m_vir)
nhalo_green=len(mvir_green)
print(nhalo_green)

# write out
sampled_halo_data_blue.to_parquet(f'/cosma8/data/dp004/dc-conl1/FLAMINGO/patchy_screening/data_files/sampled_halo_data_Blue_ntotal_{im_name}.parquet',  compression='snappy', index=False)
sampled_halo_data_green.to_parquet(f'/cosma8/data/dp004/dc-conl1/FLAMINGO/patchy_screening/data_files/sampled_halo_data_Green_ntotal_{im_name}.parquet', compression='snappy', index=False)
