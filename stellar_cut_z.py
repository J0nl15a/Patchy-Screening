import os
import numpy as np
from pathlib import Path
from FLAMINGO_halo_redshifts import multiprocess_z_bins

def stellar_cut_z(ncpu, boxname, simname, z_sample, mean_z_mass_cut, slope, lightcone=0):

    z_sample = str(z_sample)

    im = float(mean_z_mass_cut)
    if round(im, 1) == im:
        im_name = f"{im:.1f}".replace('.', 'p')
    else:
        im_name = f"{im:.3f}".replace('.', 'p')
    
    slope = float(slope)
    if round(slope, 1) == slope:
        slope_name = f"{float(slope):.1f}".replace('.', 'p')
    else:
        slope_name = f"{float(slope):.3f}".replace('.', 'p')
    if slope < 0.0:
        slope_name = f"{slope_name}".replace('-', 'minus')

    lightcone = int(lightcone)
    
    mass_cut_list = []
    z_mean = {'Blue':0.6, 'Green':1.1, 'Red':1.5}
    outfile_path = f'/cosma8/data/dp004/dc-conl1/FLAMINGO/patchy_screening/data_files/z_dependant_stellar_cuts/{boxname}/{z_sample}/z_stellar_cut_data_{im_name}_{slope_name}.txt'
    outfile_path = Path(outfile_path)
    outfile_path.parent.mkdir(parents=True, exist_ok=True)

    if not os.path.isfile(f'/cosma8/data/dp004/dc-conl1/FLAMINGO/patchy_screening/data_files/halo_redshifts/{boxname}/{simname}/lightcone{lightcone}/FLAMINGO_halo_redshift_values.txt'):
        multiprocess_z_bins(ncpu, boxname, simname, lightcone=lightcone)

    halo_z_bins = np.genfromtxt(
        f'/cosma8/data/dp004/dc-conl1/FLAMINGO/patchy_screening/data_files/halo_redshifts/{boxname}/{simname}/lightcone{lightcone}/FLAMINGO_halo_redshift_values.txt',
        dtype=[('i',   'i4'),
               ('z_min', 'f8'),
               ('mid_z', 'f8'),
               ('z_max', 'f8')],
        delimiter=None
    )

    for i,z in enumerate(halo_z_bins['mid_z']):
        mass_cut_z = float((slope * (z-z_mean[z_sample])) + (im))
        mass_cut_list.append(f"{i} {mass_cut_z}\n")

    print(f'mean_z_mass_cut = {im}, slope = {slope}, z_sample = {z_sample}')
    print(mass_cut_list)

    with open(outfile_path, 'w') as f_out:
        f_out.writelines(mass_cut_list)

    return

if __name__ == "__main__":
    import sys

    mean_z_mass_cut_values = [round(m,1) for m in np.arange(10.0, 11.6, 0.1)]
    slope_values = [round(s,1) for s in np.arange(-1.0, 1.1, 0.1)]

    '''for i in mean_z_mass_cut_values:
        for j in slope_values:
            stellar_cut_z(i, j, z_sample)'''

    stellar_cut_z(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6], lightcone=sys.argv[7])
