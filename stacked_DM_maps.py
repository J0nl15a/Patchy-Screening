import h5py, numpy as np, healpy as hp
from pathlib import Path
import sys

sample_redshift_shell = {'Blue':11, 'Green':21, 'Red':29}

def stack_DM_maps_z3(box, sim, lightcone=0):

    if box == 'L1000N1800' and lightcone == 0:
        max_redshift = '3'
    elif box == 'L2800N5040' and sim == 'HYDRO_FIDUCIAL':
        max_redshift = '5'
    lightcone_shell_redshifts = np.loadtxt(f'/cosma8/data/dp004/flamingo/Runs/{box}/{sim}/shell_redshifts_z{max_redshift}.txt', skiprows=0, delimiter=',')

    for i in range(len(lightcone_shell_redshifts)):
        DM_total = 0
        print(lightcone_shell_redshifts[i,0], lightcone_shell_redshifts[i,1])
        if lightcone_shell_redshifts[i,1] > 3.0:
            break
        else:
            #for j in range(z+1):
            if box == 'L1000N1800':
                map_dir = 'neutrino_corrected_maps'
            elif box == 'L2800N5040' and sim == 'HYDRO_FIDUCIAL':
                map_dir = 'neutrino_corrected_maps_downsampled_4096'
            else:
                print("Lightcone map not available for this box/simulation combination.")
                sys.exit()
            map_lightcone = f'/cosma8/data/dp004/flamingo/Runs/{box}/{sim}/{map_dir}/lightcone{lightcone}_shells/shell_{i}/lightcone{lightcone}.shell_{i}.0.hdf5'
            g = h5py.File(map_lightcone,'r')
            DM = g['DM'][...]*g['DM'].attrs['Conversion factor to CGS (not including cosmological corrections)']*6.6524587321e-25
            redshift = g['DM'].attrs['Central redshift assumed for correction']
            DM *= (1+redshift)
            DM_total += DM
            print(redshift, DM)
            g.close()

    output_path = Path(f'/cosma8/data/dp004/dc-conl1/FLAMINGO/patchy_screening/data_files/DM_maps/{box}/{sim}/lightcone{lightcone}/')
    output_path.mkdir(parents=True, exist_ok=True)
    
    map_write = hp.write_map(f'{output_path}stacked_DM_map_z3p0.fits', DM_total, overwrite=True)

    return DM_total

if __name__ == "__main__":
    DM_total = stack_DM_maps_z3('L1000N1800', 'HYDRO_FIDUCIAL')
    print(DM_total, np.mean(DM_total))
