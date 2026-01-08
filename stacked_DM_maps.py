import numpy as np
import h5py
import healpy as hp
from pathlib import Path

sample_redshift_shell = {'Blue':11, 'Green':21, 'Red':29}

def stack_DM_maps_z3(box, sim):


    lightcone_shell_redshifts = np.loadtxt(f'/cosma8/data/dp004/flamingo/Runs/{box}/{sim}/shell_redshifts_z3.txt', skiprows=0, delimiter=',')

    for i in range(len(lightcone_shell_redshifts)):
        DM_total = 0
        print(lightcone_shell_redshifts[i,0], lightcone_shell_redshifts[i,1])
        if lightcone_shell_redshifts[i,1] > 3.0:
            break
        else:
            #for j in range(z+1):
            map_lightcone = f'/cosma8/data/dp004/flamingo/Runs/{box}/{sim}/neutrino_corrected_maps/lightcone0_shells/shell_{i}/lightcone0.shell_{i}.0.hdf5'
            g = h5py.File(map_lightcone,'r')
            DM = g['DM'][...]*g['DM'].attrs['Conversion factor to CGS (not including cosmological corrections)']*6.6524587321e-25
            redshift = g['DM'].attrs['Central redshift assumed for correction']
            DM *= (1+redshift)
            DM_total += DM
            print(redshift, DM)
            g.close()

    output_path = Path(f'/cosma8/data/dp004/dc-conl1/FLAMINGO/patchy_screening/data_files/DM_maps/{box}/{sim}/')
    output_path.mkdir(parents=True, exist_ok=True)
    
    map_write = hp.write_map(f'{output_path}stacked_DM_map_z3p0.fits', DM_total, overwrite=True)

    return DM_total

if __name__ == "__main__":
    DM_total = stack_DM_maps_z3('L1000N1800', 'HYDRO_FIDUCIAL')
    print(DM_total, np.mean(DM_total))
