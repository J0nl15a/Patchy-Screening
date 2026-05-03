import h5py
import numpy as np
import glob, sys
from joblib import Parallel, delayed
from pathlib import Path

def z_bins(box: str, isim: str, i: int, lightcone: int = 0):

    if box == 'L1000N1800':
        snap_max = 77
        map_dir = 'hbt_lightcone_halos'
    elif box == 'L1000N3600':
        snap_max = 78
        map_dir = 'hbt_lightcone_halos'
    elif  box == 'L2800N5040':
        snap_max = 78
        map_dir = 'sorted_hbt_lightcone_halos'
    else:
        print("Halo lightcone not available for this box/simulation combination.")
        return None

    halo_lightcone = f'/cosma8/data/dp004/flamingo/Runs/{box}/{isim}/{map_dir}/lightcone{lightcone}/lightcone_halos_{snap_max-i:04d}.hdf5'
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


def multiprocess_z_bins(ncpu: int, box: str, isim: str, lightcone: int = 0):

    output_path = f'/cosma8/data/dp004/dc-conl1/FLAMINGO/patchy_screening/data_files/halo_redshifts/{box}/{isim}/lightcone{lightcone}/FLAMINGO_halo_redshift_values.txt'
    outfile = Path(output_path)
    outfile.parent.mkdir(parents=True, exist_ok=True)

    if box == 'L1000N1800' or box == 'L1000N3600':
        map_dir = 'hbt_lightcone_halos'
    elif box == 'L2800N5040':
        map_dir = 'sorted_hbt_lightcone_halos'
    else:
        print("Halo lightcone not available for this box/simulation combination.")
        return None

    filelist = glob.glob(f'/cosma8/data/dp004/flamingo/Runs/{box}/{isim}/{map_dir}/lightcone{lightcone}/lightcone_halos_*.hdf5')
    print(len(filelist))

    results = Parallel(n_jobs=int(ncpu),   # adjust to your cores
                       backend='loky')(
                           delayed(z_bins)(box, isim, i, lightcone=lightcone)
                           for i in range(len(filelist))
      
    )
    # results = []
    # for i in range(len(filelist)):
    #     results.append(z_bins(box, isim, i, lightcone=lightcone))

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

if __name__ == '__main__':

    multiprocess_z_bins(sys.argv[1], sys.argv[2], sys.argv[3], lightcone=int(sys.argv[4]))
