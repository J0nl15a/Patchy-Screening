import numpy as np
from joblib import Parallel, delayed
from imp_patchy_screening import patchyScreening

def halo_lightcones(simname, z_sample, mass_cut, n_cut, ncpu, max_z=3.0):
    sim_list = ['HYDRO_FIDUCIAL','HYDRO_PLANCK','HYDRO_PLANCK_LARGE_NU_FIXED','HYDRO_PLANCK_LARGE_NU_VARY','HYDRO_STRONG_AGN','HYDRO_WEAK_AGN','HYDRO_LOW_SIGMA8','HYDRO_STRONGER_AGN','HYDRO_JETS_published','HYDRO_STRONGEST_AGN','HYDRO_STRONG_SUPERNOVA','HYDRO_STRONGER_AGN_STRONG_SUPERNOVA','HYDRO_STRONG_JETS']

    try:
        isim = int(simname)
        simname = sim_list[isim]
    except (ValueError, IndexError):
        simname = str(simname)
    print(simname, type(simname))

    z_sample = str(z_sample)
    im = float(mass_cut)
    im_name = f"{float(mass_cut):.1f}".replace('.', 'p')
    slope = float(n_cut)
    slope_name = f"{float(n_cut):.1f}".replace('.', 'p')
    if slope < 0.0:
        slope_name = f"{slope_name}".replace('-', 'minus')
    output_path = f'/cosma8/data/dp004/dc-conl1/FLAMINGO/patchy_screening/data_files/halo_totals/FLAMINGO_halo_totals_{simname}_{z_sample}_{im_name}_{slope_name}.txt'

    halo_z_bins = np.genfromtxt(
        '/cosma8/data/dp004/dc-conl1/FLAMINGO/patchy_screening/data_files/FLAMINGO_halo_redshift_values.txt',
        dtype=[('i',   'i4'),
               ('z_min', 'f8'),
               ('mid_z', 'f8'),
               ('z_max', 'f8')],
        delimiter=None
    )

    z_stellar_cuts = np.loadtxt(f'/cosma8/data/dp004/dc-conl1/FLAMINGO/patchy_screening/data_files/z_dependant_stellar_cuts/z_stellar_cut_data_{z_sample}_{im_name}_{slope_name}.txt')

    # dispatch in parallel
    results = Parallel(n_jobs=int(ncpu),   # adjust to your cores
                       backend='loky')(
                           delayed(process_snapshot)(simname, int(i), z_stellar_cuts, halo_z_bins)
                           for i in halo_z_bins['i']
                       )

    print(results)
    idx, data = zip(*[r for r in results if r is not None])
    total_nhalo = np.sum(data)

    idx = np.array(idx, dtype=int)
    data = np.array(data, dtype=int)
    
    out = np.column_stack((idx, data))
    
    print(idx, data)
    print(total_nhalo)
    
    np.savetxt(output_path, out, fmt='%d %d', header=f"Total number of suitable halos: {total_nhalo}", comments='')
            
    return

def process_snapshot(sim, iz, stellar_cuts, halo_z_bins):
    if halo_z_bins['mid_z'][iz] > 3.0:
        return None

    im = stellar_cuts[int(iz)][1]
    print(f'im={im}')

    ps = patchyScreening(sim, iz, im,  # or however you pass
                         0 ,0, 1,
                         lightcone_method=('FULL','shell'))
    ps.filter_stellar_mass()
    print(f'ps.im={ps.im}')

    return iz, ps.nhalo

if __name__ == '__main__':
    import sys
    import pylab as pb
    import re
    from io import StringIO
    import textwrap

    halo_lightcones(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[1])
    quit()
    sim_list = ['HYDRO_FIDUCIAL','HYDRO_PLANCK','HYDRO_PLANCK_LARGE_NU_FIXED','HYDRO_PLANCK_LARGE_NU_VARY','HYDRO_STRONG_AGN','HYDRO_WEAK_AGN','HYDRO_LOW_SIGMA8','HYDRO_STRONGER_AGN','HYDRO_JETS_published','HYDRO_STRONGEST_AGN','HYDRO_STRONG_SUPERNOVA','HYDRO_STRONGER_AGN_STRONG_SUPERNOVA','HYDRO_STRONG_JETS']

    try:
        isim = int(sys.argv[2])
        simname = sim_list[isim]
    except (ValueError, IndexError):
        simname = str(sys.argv[2])
    z_sample = str(sys.argv[3])
    im = float(sys.argv[4])
    im_name = f"{float(sys.argv[4]):.1f}".replace('.', 'p')
    slope = float(sys.argv[5])
    slope_name = f"{float(sys.argv[5]):.1f}".replace('.', 'p')
    if slope < 0.0:
        slope_name = f"{slope_name}".replace('-', 'minus')

    halo_z_bins = np.genfromtxt(
        '/cosma8/data/dp004/dc-conl1/FLAMINGO/patchy_screening/data_files/FLAMINGO_halo_redshift_values.txt',
        dtype=[('i',   'i4'),
               ('z_min', 'f8'),
               ('mid_z', 'f8'),
               ('z_max', 'f8')],
        delimiter=None
    )
    FLAMINGO_mid_point = halo_z_bins['mid_z'][np.where(halo_z_bins['mid_z'] <= 3.0)]
    print(FLAMINGO_mid_point)
    
    for s in [round(n, 2) for n in np.arange(-1.0,0.6,0.1)]:
        print(s)
        slope_name = f"{float(s):.1f}".replace('.', 'p')
        if s < 0.0:
            slope_name = f"{slope_name}".replace('-', 'minus')
            if s == -0.0:
                slope_name = "0p0"
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
        halo_lightcones_values = np.loadtxt(StringIO(halo_lightcones_str), usecols=(0,1))

        pb.plot(FLAMINGO_mid_point, halo_lightcones_values[:,1], marker='.', label=f'Galaxies available (slope = {s})') #color='r', marker='.', label=f'Galaxies available (slope = {s})')
    pb.xlim(left=0, right=3)
    pb.xlabel('z')
    pb.ylabel('dn/dz')
    pb.title("\n".join(textwrap.wrap(f'Galaxies in FLAMINGO (simname = {simname}, {z_sample} sample, stellar cut at mean z = {im}, slope of stellar cut = {slope}, total number of galaxies = {total_available_halos})', width=75)))
    pb.legend(fontsize=4)
    pb.savefig(f"./Plots/FLAMINGO_available_halos_{simname}_{z_sample}_{im_name}_{slope_name}.png", dpi=400)
    pb.clf()
