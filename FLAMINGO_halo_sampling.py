from joblib import Parallel, delayed
import numpy as np
import pandas as pd
import h5py
from imp_patchy_screening import patchyScreening
import sys

sample_z = sys.argv[2]
im = float(sys.argv[3])
im_name = f"{float(sys.argv[3]):.1f}".replace('.', 'p')

halo_z_bins = np.genfromtxt(
    '/cosma8/data/dp004/dc-conl1/FLAMINGO/patchy_screening/data_files/FLAMINGO_halo_redshift_values.txt',
    dtype=[('i',   'i4'),
           ('z1', 'f8'),
           ('z2', 'f8')],
    delimiter=None
)

dndz_sample = np.loadtxt(f'/cosma8/data/dp004/dc-conl1/FLAMINGO/patchy_screening/data_files/dndz_galaxies_sampled_{sample_z}_ntotal.txt')

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

    nsamp = int(dndz_sample[i][2])
    subdf = ps.merge.sample(n=nsamp, random_state=1000)
    
    return subdf

# dispatch in parallel
results = Parallel(n_jobs=int(sys.argv[1]),   # adjust to your cores
                   backend='loky')(
    delayed(process_snapshot)(int(i))
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
sampled_halo_data.to_parquet(f'/cosma8/data/dp004/dc-conl1/FLAMINGO/patchy_screening/data_files/sampled_halo_data_{sample_z}_ntotal_{im_name}.parquet',  compression='snappy', index=False)
