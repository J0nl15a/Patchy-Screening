import numpy as np

def stellar_cut_z(z_sample, mean_z_mass_cut, slope):

    z_sample = str(z_sample)
    im = float(mean_z_mass_cut)
    im_name = f"{float(mean_z_mass_cut):.1f}".replace('.', 'p')
    slope = float(slope)
    slope_name = f"{float(slope):.1f}".replace('.', 'p')
    if slope < 0.0:
        slope_name = f"{slope_name}".replace('-', 'minus')
    mass_cut_list = []
    z_mean = {'Blue':0.6, 'Green':1.1, 'Red':1.5}
    outfile_path = f'/cosma8/data/dp004/dc-conl1/FLAMINGO/patchy_screening/data_files/z_dependant_stellar_cuts/z_stellar_cut_data_{z_sample}_{im_name}_{slope_name}.txt'

    FLAMINGO_halo_bins = np.loadtxt('/cosma8/data/dp004/dc-conl1/FLAMINGO/patchy_screening/data_files/FLAMINGO_halo_redshift_values.txt', usecols=2)
    z_max_idx = np.where(FLAMINGO_halo_bins <= 3)[0]
    FLAMINGO_z_bins = np.zeros((len(z_max_idx)+2))
    FLAMINGO_z_bins[1:-1] = FLAMINGO_halo_bins[z_max_idx]
    FLAMINGO_z_bins[-1] = 3.0
    FLAMINGO_mid_point = ((FLAMINGO_z_bins[1:] - FLAMINGO_z_bins[:-1]) / 2) + FLAMINGO_z_bins[:-1]
    print(FLAMINGO_mid_point)
    
    for i,z in enumerate(FLAMINGO_mid_point):
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

    stellar_cut_z(sys.argv[1], sys.argv[2], sys.argv[3])
