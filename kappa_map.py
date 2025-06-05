import numpy as np
import healpy as hp

kappa_map = 0
for i in range(60):
    print(i)
    kappa_shell = hp.read_map(f'/cosma8/data/dp004/dc-yang3/maps/L1000N1800/HYDRO_FIDUCIAL/lightcone0_shells/kappa_per_shell/kappa_map_shell_{i}.fits', dtype=np.float64, verbose=False)
    kappa_map += kappa_shell
    print(kappa_map[0])

map_write = hp.write_map('./data_files/kappa_map_z3.fits', kappa_map, overwrite=True)
