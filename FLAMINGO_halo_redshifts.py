import h5py
import numpy as np
import glob, os
from joblib import Parallel, delayed
import sys

def z_bins(i):

    halo_lightcone = f'/cosma8/data/dp004/flamingo/Runs/L1000N1800/HYDRO_FIDUCIAL/hbt_lightcone_halos/lightcone0/lightcone_halos_{77-i:04d}.hdf5'
    f = h5py.File(halo_lightcone, 'r')
    z = f['Lightcone/Redshift'][...]
    
    try:
        min_z = min(z)
        max_z = max(z)
        midpoint = (max_z - min_z)/2 + min_z
    except ValueError:
        return None
    print(i, min_z, midpoint, max_z)

    return i, min_z, midpoint, max_z


output_path = '/cosma8/data/dp004/dc-conl1/FLAMINGO/patchy_screening/data_files/FLAMINGO_halo_redshift_values.txt'

filelist = glob.glob('/cosma8/data/dp004/flamingo/Runs/L1000N1800/HYDRO_FIDUCIAL/hbt_lightcone_halos/lightcone0/lightcone_halos_*.hdf5')
print(len(filelist))

results = Parallel(n_jobs=int(sys.argv[1]),   # adjust to your cores
                   backend='loky')(
                       delayed(z_bins)(i)
                       for i in range(len(filelist))
                   )

print(results)
idx, min_vals, mid, max_vals = zip(*[r for r in results if r is not None])
print(idx)
print(min_vals)
print(mid)
print(max_vals)

idx = np.array(idx, dtype=int)
min_vals = np.array(min_vals, dtype=float)
mid = np.array(mid, dtype=float)
max_vals = np.array(max_vals, dtype=float)

print(idx)
print(min_vals)
print(mid)
print(max_vals)

out = np.column_stack((idx, min_vals, mid, max_vals))
out = out[out[:, 0].argsort()]
print(out)
#quit()
np.savetxt(output_path, out, fmt='%d %.18f %.18f %.18f', comments='')
