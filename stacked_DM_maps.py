import h5py
import healpy as hp

sample_redshift_shell = {'Blue':11, 'Green':21, 'Red':29}

for i,z in sample_redshift_shell.items():
    DM_total = 0
    for j in range(z+1):
        map_lightcone = f'/cosma8/data/dp004/flamingo/Runs/L1000N1800/HYDRO_FIDUCIAL/neutrino_corrected_maps/lightcone0_shells/shell_{j}/lightcone0.shell_{j}.0.hdf5'
        g = h5py.File(map_lightcone,'r')
        DM = g['DM'][...]*g['DM'].attrs['Conversion factor to CGS (not including cosmological corrections)']*6.6524587321e-25
        DM_total += DM
    
    map_write = hp.write_map(f'./stacked_DM_map_{i}.fits', DM_total, overwrite=True)
