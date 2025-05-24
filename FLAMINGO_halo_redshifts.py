import h5py

with open('/cosma8/data/dp004/dc-conl1/FLAMINGO/patchy_screening/data_files/FLAMINGO_halo_redshift_values.txt', 'w') as f_out:
    for i in range(78):
        try:
            #halo_lightcone = f'/cosma8/data/dp004/jch/FLAMINGO/HBT/L1000N1800/HYDRO_FIDUCIAL/lightcone_halos/lightcone0/lightcone_halos_{77-i:04d}.hdf5'
            halo_lightcone = f'/cosma8/data/dp004/flamingo/Runs/L1000N1800/HYDRO_FIDUCIAL/hbt_lightcone_halos/lightcone0/lightcone_halos_{77-i:04d}.hdf5'
            f = h5py.File(halo_lightcone, 'r')
        except OSError:
            print(f'Error with filie: /cosma8/data/dp004/flamingo/Runs/L1000N1800/HYDRO_FIDUCIAL/lightcones/lightcone0_particles/lightcone0_{77-i:04d}.0.hdf5')
            continue

        z = f['Lightcone/Redshift'][...]
        print(i, min(z), max(z))
        f_out.write(f"{i} {min(z)} {max(z)}\n")
