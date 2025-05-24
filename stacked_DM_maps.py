import numpy as np
import h5py
import healpy as hp

sample_redshift_shell = {'Blue':11, 'Green':21, 'Red':29}
lightcone_shell_redshifts = np.loadtxt('/cosma8/data/dp004/flamingo/Runs/L1000N1800/HYDRO_FIDUCIAL/shell_redshifts_z3.txt', skiprows=0, delimiter=',')

for i in range(len(lightcone_shell_redshifts)): #sample_redshift_shell.items():
    DM_total = 0
    print(lightcone_shell_redshifts[i,0], lightcone_shell_redshifts[i,1])
    if lightcone_shell_redshifts[i,1] > 3.0:
        break
    else:
        #for j in range(z+1):
        map_lightcone = f'/cosma8/data/dp004/flamingo/Runs/L1000N1800/HYDRO_FIDUCIAL/neutrino_corrected_maps/lightcone0_shells/shell_{i}/lightcone0.shell_{i}.0.hdf5'
        g = h5py.File(map_lightcone,'r')
        DM = g['DM'][...]*g['DM'].attrs['Conversion factor to CGS (not including cosmological corrections)']*6.6524587321e-25
        DM_total += DM
        redshift = g['DM'].attrs['Central redshift assumed for correction']
        print(redshift)
    map_write = hp.write_map(f'/cosma8/data/dp004/dc-conl1/FLAMINGO/patchy_screening/data_files/stacked_DM_map_z3p0.fits', DM_total, overwrite=True)
