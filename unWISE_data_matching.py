import numpy as np
import pylab as pb
from scipy.interpolate import interp1d
from scipy.integrate import simpson
import re
from io import StringIO
import math
import textwrap

def unWISE_data_matching(simname, z_sample, mass_cut, n_cut, nsamp='ntotal', plot=False):
    sim_list = ['HYDRO_FIDUCIAL','HYDRO_PLANCK','HYDRO_PLANCK_LARGE_NU_FIXED','HYDRO_PLANCK_LARGE_NU_VARY','HYDRO_STRONG_AGN','HYDRO_WEAK_AGN','HYDRO_LOW_SIGMA8','HYDRO_STRONGER_AGN','HYDRO_JETS_published','HYDRO_STRONGEST_AGN','HYDRO_STRONG_SUPERNOVA','HYDRO_STRONGER_AGN_STRONG_SUPERNOVA','HYDRO_STRONG_JETS']
    
    try:
        isim = int(simname)
        simname = sim_list[isim]
    except (ValueError, IndexError):
        simname = str(simname)
    z_sample = str(z_sample)
    im_name = f"{float(mass_cut):.1f}".replace('.', 'p')
    slope_name = f"{float(n_cut):.1f}".replace('.', 'p')
    if float(n_cut) < 0.0:
        slope_name = f"{slope_name}".replace('-', 'minus')

    try:
        nsamp = int(nsamp)
    except ValueError:
        nsamp = str(nsamp)

    if z_sample == 'Blue':
        dndz_match = np.loadtxt("/cosma8/data/dp004/dc-conl1/FLAMINGO/patchy_screening/unWISExLens_lklh/data/v1.0/aux_data/dndz/unWISE_blue_xmatch_dndz.txt", usecols=(0,1))
    elif z_sample == 'Green':
        dndz_match = np.loadtxt("/cosma8/data/dp004/dc-conl1/FLAMINGO/patchy_screening/unWISExLens_lklh/data/v1.0/aux_data/dndz/unWISE_green_xmatch_dndz.txt", usecols=(0,1))

    print(min(dndz_match[:,0]), max(dndz_match[:,0]))

    halo_z_bins = np.genfromtxt(
        '/cosma8/data/dp004/dc-conl1/FLAMINGO/patchy_screening/data_files/FLAMINGO_halo_redshift_values.txt',
        dtype=[('i',   'i4'),
               ('z_min', 'f8'),
               ('mid_z', 'f8'),
               ('z_max', 'f8')],
        delimiter=None
    )

    FLAMINGO_z_bins = np.zeros((len(halo_z_bins['z_max'][np.where(halo_z_bins['z_max'] <= 3.0)])+1))
    FLAMINGO_z_bins[:-1] = halo_z_bins['z_max'][np.where(halo_z_bins['z_max'] <= 3.0)]
    FLAMINGO_z_bins[-1] = 3.0
    FLAMINGO_mid_point = halo_z_bins['mid_z'][np.where(halo_z_bins['mid_z'] <= 3.0)]
    print(FLAMINGO_mid_point)
    print(FLAMINGO_z_bins)
    
    with open(f"/cosma8/data/dp004/dc-conl1/FLAMINGO/patchy_screening/data_files/halo_totals/FLAMINGO_halo_totals_{simname}_{z_sample}_{im_name}_{slope_name}.txt", "r") as f:
        first_line = f.readline().strip()
        remaining_lines = f.readlines()

    # Extract number from the first line
    match = re.search(r"(\d+)", first_line)
    if match:
        total_available_halos = int(match.group(1))
        print("Total number of suitable halos:", total_available_halos)
    else:
        raise ValueError("No number found in the first line")

    # Convert remaining lines to a NumPy array
    halo_lightcones_str = "".join(remaining_lines)
    halo_lightcones = np.loadtxt(StringIO(halo_lightcones_str), usecols=(0,1))
    print(halo_lightcones.shape)

    empty_shell_mask = np.where(halo_lightcones[:,1] == 0)
    print(empty_shell_mask)

    m = interp1d(dndz_match[:,0], dndz_match[:,1], kind='cubic')
    dndz_match_interpolated = m(FLAMINGO_z_bins)
    total_area = simpson(y=dndz_match_interpolated, x=FLAMINGO_z_bins)
    print(total_area)

    if plot==True:
        pb.plot(dndz_match[:,0], dndz_match[:,1], color='g', marker='.', label='Original curve')
        pb.plot(FLAMINGO_z_bins, dndz_match_interpolated, color='b', marker='.', label='Interpolated curve')
        pb.plot(FLAMINGO_mid_point, m(FLAMINGO_mid_point), color='r', marker='.', label='Midpoint')
        pb.xlim(left=0, right=3)
        pb.xlabel('z')
        pb.ylabel('dn/dz')
        pb.title("\n".join(textwrap.wrap(f'unWISE dndz ({z_sample} sample)', width=75)))
        pb.legend()
        pb.savefig(f'./Plots/unWISE_dndz_match_curve_{z_sample}_test.png', dpi=400)
        pb.clf()

    outfile_name = f'/cosma8/data/dp004/dc-conl1/FLAMINGO/patchy_screening/data_files/dndz_samples/dndz_galaxies_sampled_{simname}_{z_sample}_{im_name}_{slope_name}.txt'

    if nsamp == 'ntotal':
        if z_sample == 'Blue':
            kusiak_nbar_blue = 3409 #From Kusiak paper
            kusiak_ntotal_blue = kusiak_nbar_blue * 41253 #4pi steradians
            nsamp = kusiak_ntotal_blue
        elif z_sample == 'Green':
            kusiak_nbar_green = 1846 #From Kusiak paper
            kusiak_ntotal_green = kusiak_nbar_green * 41253
            nsamp = kusiak_ntotal_green
    
    if nsamp > total_available_halos:
        nsamp = total_available_halos
        
    print(nsamp)

    galaxies_required = compute_galaxies_required(FLAMINGO_mid_point, FLAMINGO_z_bins, m, nsamp, total_area, halo_lightcones[:,1])
    print(galaxies_required)
    conflict, difference = validate_required_vs_available(galaxies_required, halo_lightcones[:, 1])

    if plot==True:
        pb.plot(FLAMINGO_mid_point, galaxies_required, color='b', marker='.', label='Galaxies required')
        pb.plot(FLAMINGO_mid_point, halo_lightcones[:, 1], color='r', marker='.', label='Galaxies available')
        pb.xlim(left=0, right=3)
        pb.xlabel('z')
        pb.ylabel('dn/dz')
        pb.title("\n".join(textwrap.wrap(f'unWISE galaxy redshift distribution cross-match (rescaled, simname = {simname}, {z_sample} sample, stellar cut at mean z = {mass_cut}, stellar cut slope = {n_cut}, total number of galaxies = {nsamp})', width=75)))
        pb.legend()
        pb.savefig(f'./Plots/unWISE_dndz_match_rescaled_{simname}_{z_sample}_{im_name}_{slope_name}_0.png', dpi=400)
        pb.clf()

    count = 0
    while conflict == True:

        print("Rescaling required")
        count+=1
        print(difference)
        max_diff_index = np.argmax(difference)
        print(max_diff_index)
        print(halo_lightcones[max_diff_index, 0])
        print(halo_lightcones[max_diff_index, 1], galaxies_required[max_diff_index])
        ratio_nsamp = halo_lightcones[max_diff_index, 1]/galaxies_required[max_diff_index]
        print(ratio_nsamp)
        print(nsamp * ratio_nsamp, math.floor(nsamp*ratio_nsamp))
        nsamp = math.floor(nsamp*ratio_nsamp)
        galaxies_required = compute_galaxies_required(FLAMINGO_mid_point, FLAMINGO_z_bins, m, nsamp, total_area, halo_lightcones[:,1])
        conflict, difference = validate_required_vs_available(galaxies_required, halo_lightcones[:, 1])

        if plot==True:
            pb.plot(FLAMINGO_mid_point, galaxies_required, color='b', marker='.', label='Galaxies required')
            pb.plot(FLAMINGO_mid_point, halo_lightcones[:, 1], color='r', marker='.', label='Galaxies available')
            pb.xlim(left=0, right=3)
            pb.xlabel('z')
            pb.ylabel('dn/dz')
            pb.title("\n".join(textwrap.wrap(f'unWISE galaxy redshift distribution cross-match (rescaled, simname = {simname}, {z_sample} sample, stellar cut at mean z = {mass_cut}, stellar cut slope = {n_cut}, total number of galaxies = {nsamp})', width=75)))
            pb.legend()
            pb.savefig(f'./Plots/unWISE_dndz_match_rescaled_{simname}_{z_sample}_{im_name}_{slope_name}_{count}.png', dpi=400)
            pb.clf()

    print(simpson(y=dndz_match[:,1]*(nsamp/simpson(y=dndz_match[:,1], x=dndz_match[:,0])*0.05), x=dndz_match[:,0]), simpson(y=galaxies_required, x=FLAMINGO_mid_point))
    
    if plot==True:
        pb.plot(FLAMINGO_mid_point, galaxies_required, color='b', marker='.', label='Galaxies required')
        pb.plot(FLAMINGO_mid_point, halo_lightcones[:, 1], color='r', marker='.', label='Galaxies available')
        pb.plot(dndz_match[:,0], dndz_match[:,1]*(nsamp/simpson(y=dndz_match[:,1], x=dndz_match[:,0]))*0.05, color='g', marker='.', label='Original')
        pb.xlim(left=0, right=3)
        pb.xlabel('z')
        pb.ylabel('dn/dz')
        pb.title("\n".join(textwrap.wrap(f'unWISE galaxy redshift distribution cross-match (rescaled, simname = {simname}, {z_sample} sample, stellar cut at mean z = {mass_cut}, stellar cut slope = {n_cut}, total number of galaxies = {nsamp})', width=75)))
        pb.legend()
        pb.savefig(f'./Plots/unWISE_dndz_match_rescaled_{simname}_{z_sample}_{im_name}_{slope_name}_test.png', dpi=400)
        pb.clf()

        pb.plot(FLAMINGO_z_bins[1:], dndz_match_interpolated[1:]/galaxies_required, color='b', marker='.', label='Galaxies required')
        pb.xlim(left=0, right=3)
        pb.xlabel('z')
        pb.ylabel('dn/dz')
        pb.yscale('log')
        pb.title("\n".join(textwrap.wrap(f'unWISE galaxy redshift distribution cross-match (rescaled, simname = {simname}, {z_sample} sample, stellar cut at mean z = {mass_cut}, stellar cut slope = {n_cut}, total number of galaxies = {nsamp})', width=75)))
        pb.legend()
        pb.savefig(f'./Plots/unWISE_dndz_match_{simname}_{z_sample}_{im_name}_{slope_name}_ratio.png', dpi=400)
        pb.clf()

    write_sampled_galaxies_file(outfile_name, FLAMINGO_z_bins, galaxies_required, nsamp)

    return

# This refactored version separates the loop from file-writing so the
# user can inspect galaxy counts and rescale before final output

def compute_galaxies_required(FLAMINGO_mid_point, FLAMINGO_z_bins, dndz_func, nsamp, total_area, halo_lightcones):
    galaxies_required = []
    for i in range(len(FLAMINGO_mid_point)):
        if halo_lightcones[i] == 0:
            galaxies_required.append(0)
        else:
            delta_z = FLAMINGO_z_bins[i+1] - FLAMINGO_z_bins[i]
            required = dndz_func(FLAMINGO_mid_point[i]) * (nsamp/total_area) * delta_z
            galaxies_required.append(required)
    print(simpson(y=galaxies_required, x=FLAMINGO_mid_point))
    return np.array(galaxies_required)

def validate_required_vs_available(galaxies_required, available_counts):
    diff = galaxies_required - available_counts
    excess = diff > 0.0
    return excess.any(), diff

def write_sampled_galaxies_file(outfile_name, FLAMINGO_z_bins, galaxies_required, nsamp):
    with open(outfile_name, 'w') as outfile:
        outfile.write(f"#Total number of sampled galaxies: {int(round(nsamp))}\n")
        for i in range(1, len(FLAMINGO_z_bins)):
            zmin = FLAMINGO_z_bins[i-1]
            zmax = FLAMINGO_z_bins[i]
            n_gals = int(round(galaxies_required[i-1]))
            outfile.write(f"{zmin} {zmax} {n_gals}\n")
    print(f"Wrote file: {outfile_name}")
    return


if __name__ == '__main__':
    import sys

    plot=False
    try:
        unWISE_data_matching(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], plot=plot)
    except IndexError:
        unWISE_data_matching(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], plot=plot)

    if plot==True:
        pb.plot(dndz_blue_match[:,0], dndz_blue_match[:,1], color='tab:blue', label='Blue sample')
        pb.plot(dndz_green_match[:,0], dndz_green_match[:,1], color='tab:green', label='Green sample')
        pb.plot(FLAMINGO_z_bins, dndz_blue_match_interpolated, color='tab:cyan', marker='.', label='FLAMINGO bins')
        pb.plot(FLAMINGO_z_bins, dndz_green_match_interpolated, color='tab:olive', marker='.', label='FLAMINGO bins')
        pb.xlim(left=0, right=4)
        pb.xlabel('z')
        pb.ylabel('dn/dz')
        pb.title('unWISE galaxy redshift distribution cross-match')
        pb.legend()
        pb.savefig('./Plots/unWISE_dndz_match.png', dpi=1200)
        pb.clf()
