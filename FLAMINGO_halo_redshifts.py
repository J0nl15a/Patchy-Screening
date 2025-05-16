import h5py

with open('FLAMINGO_halo_redshift_values.txt', 'w') as f_out:
    for i in range(78):
        try:
            halo_lightcone = f'/cosma8/data/dp004/jch/FLAMINGO/HBT/L1000N1800/HYDRO_FIDUCIAL/lightcone_halos/lightcone0/lightcone_halos_{77-i:04d}.hdf5'
            f = h5py.File(halo_lightcone, 'r')
        except OSError:
            continue

        z = f['Lightcone/Redshift'][...]
        print(i, min(z), max(z))
        f_out.write(f"{i} {min(z)} {max(z)}\n")
