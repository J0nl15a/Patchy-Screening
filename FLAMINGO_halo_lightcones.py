import numpy as np
from imp_patchy_screening import patchyScreening

def halo_lightcones(simname, mass_cut, max_z=3.0):
    sim_list = ['HYDRO_FIDUCIAL','HYDRO_PLANCK','HYDRO_PLANCK_LARGE_NU_FIXED','HYDRO_PLANCK_LARGE_NU_VARY','HYDRO_STRONG_AGN','HYDRO_WEAK_AGN','HYDRO_LOW_SIGMA8','HYDRO_STRONGER_AGN','HYDRO_JETS_published','HYDRO_STRONGEST_AGN','HYDRO_STRONG_SUPERNOVA','HYDRO_STRONGER_AGN_STRONG_SUPERNOVA','HYDRO_STRONG_JETS']

    try:
        isim = int(simname)
        simname = sim_list[isim]
    except (ValueError, IndexError):
        simname = str(simname)
    print(simname, type(simname))
    
    im = 10**np.array(float(mass_cut))
    im_name = f"{float(mass_cut):.1f}".replace('.', 'p')
    total_nhalo = 0
    output_path = f'/cosma8/data/dp004/dc-conl1/FLAMINGO/patchy_screening/data_files/FLAMINGO_halo_totals_{simname}_{im_name}.txt'
    FLAMINGO_halo_bins = np.loadtxt('/cosma8/data/dp004/dc-conl1/FLAMINGO/patchy_screening/data_files/FLAMINGO_halo_redshift_values.txt', usecols=2)
    z_max_idx = np.where(FLAMINGO_halo_bins <= max_z)[0]

    # Store data lines separately
    data_lines = []
    
    for i in range(len(z_max_idx)+1):
        ps = patchyScreening(simname, i, im, 0, 1, 0, lightcone_method=('FULL', 'shell'))
        ps.filter_stellar_mass()
        data_lines.append(f"{i} {ps.nhalo}\n")
        total_nhalo+=ps.nhalo
        print(i, ps.nhalo)

    print(mass_cut, total_nhalo)
    with open(output_path, 'w') as f_out:
        f_out.write(f"Total number of suitable halos: {total_nhalo}\n")
        f_out.writelines(data_lines)
            
    return

if __name__ == '__main__':
    import sys
    import pylab as pb
    import re
    from io import StringIO
    import textwrap

    halo_lightcones(sys.argv[1], sys.argv[2])

    sim_list = ['HYDRO_FIDUCIAL','HYDRO_PLANCK','HYDRO_PLANCK_LARGE_NU_FIXED','HYDRO_PLANCK_LARGE_NU_VARY','HYDRO_STRONG_AGN','HYDRO_WEAK_AGN','HYDRO_LOW_SIGMA8','HYDRO_STRONGER_AGN','HYDRO_JETS_published','HYDRO_STRONGEST_AGN','HYDRO_STRONG_SUPERNOVA','HYDRO_STRONGER_AGN_STRONG_SUPERNOVA','HYDRO_STRONG_JETS']

    try:
        isim = int(sys.argv[1])
        simname = sim_list[isim]
    except (ValueError, IndexError):
        simname = str(sys.argv[1])
    im = float(sys.argv[2])
    im_name = f"{float(sys.argv[2]):.1f}".replace('.', 'p')
    FLAMINGO_halo_bins = np.loadtxt('/cosma8/data/dp004/dc-conl1/FLAMINGO/patchy_screening/data_files/FLAMINGO_halo_redshift_values.txt', usecols=2)
    z_max_idx = np.where(FLAMINGO_halo_bins <= 3)[0]
    FLAMINGO_z_bins = np.zeros((len(z_max_idx)+2))
    FLAMINGO_z_bins[1:-1] = FLAMINGO_halo_bins[z_max_idx]
    FLAMINGO_z_bins[-1] = 3.0
    FLAMINGO_mid_point = ((FLAMINGO_z_bins[1:] - FLAMINGO_z_bins[:-1]) / 2) + FLAMINGO_z_bins[:-1]
    print(FLAMINGO_mid_point)
    
    with open(f"/cosma8/data/dp004/dc-conl1/FLAMINGO/patchy_screening/data_files/FLAMINGO_halo_totals_{simname}_{im_name}.txt", "r") as f:
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
    halo_lightcones_values = np.loadtxt(StringIO(halo_lightcones_str), usecols=(0,1))

    pb.plot(FLAMINGO_mid_point, halo_lightcones_values[:,1], color='r', marker='.', label='Galaxies available')
    pb.xlim(left=0, right=3)
    pb.xlabel('z')
    pb.ylabel('dn/dz')
    pb.title("\n".join(textwrap.wrap(f'Galaxies in FLAMINGO (simname = {simname}, stellar cut = {im}, total number of galaxies = {total_available_halos})', width=75)))
    pb.legend()
    pb.savefig(f"./Plots/FLAMINGO_available_halos_{simname}_{im_name}.png", dpi=1200)
    pb.clf()
