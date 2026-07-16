import os
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from imp_patchy_screening import patchyScreening
from FLAMINGO_halo_redshifts import multiprocess_z_bins
from pathlib import Path
import time

def halo_lightcones(boxname, simname, z_sample, mass_cut, n_cut, ncpu, max_z=3.0, lightcone=0):
    box_list = ['L1000N1800', 'L2800N5040']
    sim_list = ['HYDRO_FIDUCIAL','HYDRO_PLANCK','HYDRO_PLANCK_LARGE_NU_FIXED','HYDRO_PLANCK_LARGE_NU_VARY','HYDRO_STRONG_AGN','HYDRO_WEAK_AGN','HYDRO_LOW_SIGMA8','HYDRO_STRONGER_AGN','HYDRO_JETS_published','HYDRO_STRONGEST_AGN','HYDRO_STRONG_SUPERNOVA','HYDRO_STRONGER_AGN_STRONG_SUPERNOVA','HYDRO_STRONG_JETS']

    try:
        box = int(boxname)
        boxname = box_list[box]
    except (ValueError, IndexError):
        boxname = str(boxname)
    print(boxname, type(boxname))

    try:
        isim = int(simname)
        simname = sim_list[isim]
    except (ValueError, IndexError):
        simname = str(simname)
    print(simname, type(simname))

    z_sample = str(z_sample)
    lightcone = int(lightcone)

    im = float(mass_cut)
    if round(im, 1) == im:
        im_name = f"{float(mass_cut):.1f}".replace('.', 'p')
    else:
        im_name = f"{float(mass_cut):.3f}".replace('.', 'p')
    slope = float(n_cut)
    if round(slope, 1) == slope:
        slope_name = f"{float(n_cut):.1f}".replace('.', 'p')
    else:
        slope_name = f"{float(n_cut):.3f}".replace('.', 'p')
    if slope < 0.0:
        slope_name = f"{slope_name}".replace('-', 'minus')
        
    output_path = f'/cosma8/data/dp004/dc-conl1/FLAMINGO/patchy_screening/data_files/halo_totals/{boxname}/{simname}/{z_sample}/lightcone{lightcone}/FLAMINGO_halo_totals_{im_name}_{slope_name}.txt'
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

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

    # only shells where midpoint z < 3.0 (halo_lightcones-style)
    iz_list = sorted([iz for iz, z in zip(halo_z_bins['i'], halo_z_bins['mid_z']) if z < max_z])

    z_stellar_cuts = np.loadtxt(f'/cosma8/data/dp004/dc-conl1/FLAMINGO/patchy_screening/data_files/z_dependant_stellar_cuts/{boxname}/{z_sample}/z_stellar_cut_data_{im_name}_{slope_name}.txt')

    job_start_time = time.time()

    # 3+ hours
    # dispatch in parallel
    # results = Parallel(n_jobs=int(ncpu), prefer='processes', verbose=10,   # adjust to your cores
    #                    backend='loky')(
    #                        delayed(process_snapshot)(boxname, simname, int(i), z_stellar_cuts, lightcone)
    #                        for i in iz_list
    #                    )

    # 3827 seconds
    results = []
    for i in iz_list:
        result = process_snapshot(boxname, simname, int(i), z_stellar_cuts, lightcone)
        results.append(result)

    print(results)
    idx, data = zip(*[r for r in results if r is not None])
    total_nhalo = np.sum(data)

    idx = np.array(idx, dtype=int)
    data = np.array(data, dtype=int)
    
    out = np.column_stack((idx, data))
    
    print(idx, data)
    print(total_nhalo)
    
    np.savetxt(output_path, out, fmt='%d %d', header=f"Total number of suitable halos: {total_nhalo}", comments='')
    
    print(f'Finished processing halo lightcone data: {time.time() - job_start_time}s')
            
    return

def process_snapshot(box, sim, iz, stellar_cuts, lightcone):
    # if halo_z_bins['mid_z'][iz] > 3.0:
    #     return None
    print(f'Processing snapshot {iz}...')

    im = stellar_cuts[int(iz)][1]
    print(f'im={im}')

    # ps = patchyScreening(box, sim, iz, im,  # or however you pass
    #                      0, 1,
    #                      lightcone_method=('FULL','shell'), lightcone=lightcone)
    # ps.filter_stellar_mass()
    # print(f'ps.im={ps.im}')

    df = pd.read_parquet(
        f"/cosma8/data/dp004/dc-conl1/FLAMINGO/patchy_screening/data_files/shell_caches/{box}/{sim}/lightcone{lightcone}/shell_{iz:03d}.parquet"
    )
    df = df[df["mstar"] >= 10**(float(im))]
    nhalo = len(df)

    print(f'Processed snapshot {iz}, number of halos after stellar mass cut: {nhalo}')

    return iz, nhalo #ps.nhalo

if __name__ == '__main__':
    import sys, re, textwrap
    import pylab as pb
    from io import StringIO

    halo_lightcones(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6], sys.argv[1], lightcone=int(sys.argv[7]))
    quit()
    box_list = ['L1000N1800', 'L2800N5040']
    sim_list = ['HYDRO_FIDUCIAL','HYDRO_PLANCK','HYDRO_PLANCK_LARGE_NU_FIXED','HYDRO_PLANCK_LARGE_NU_VARY','HYDRO_STRONG_AGN','HYDRO_WEAK_AGN','HYDRO_LOW_SIGMA8','HYDRO_STRONGER_AGN','HYDRO_JETS_published','HYDRO_STRONGEST_AGN','HYDRO_STRONG_SUPERNOVA','HYDRO_STRONGER_AGN_STRONG_SUPERNOVA','HYDRO_STRONG_JETS']

    try:
        box = int(sys.argv[2])
        boxname = box_list[box]
    except (ValueError, IndexError):
        boxname = str(sys.argv[2])

    try:
        isim = int(sys.argv[3])
        simname = sim_list[isim]
    except (ValueError, IndexError):
        simname = str(sys.argv[3])

    z_sample = str(sys.argv[4])
    im = float(sys.argv[5])
    im_name = f"{float(sys.argv[5]):.1f}".replace('.', 'p')
    slope = float(sys.argv[6])
    slope_name = f"{float(sys.argv[6]):.1f}".replace('.', 'p')
    if slope < 0.0:
        slope_name = f"{slope_name}".replace('-', 'minus')
    lightcone = int(sys.argv[7])

    halo_z_bins = np.genfromtxt(
        f'/cosma8/data/dp004/dc-conl1/FLAMINGO/patchy_screening/data_files/halo_redshift_values/{boxname}/{simname}/lightcone{lightcone}/FLAMINGO_halo_redshift_values.txt',
        dtype=[('i',   'i4'),
               ('z_min', 'f8'),
               ('mid_z', 'f8'),
               ('z_max', 'f8')],
        delimiter=None
    )

    FLAMINGO_mid_point = halo_z_bins['mid_z'][np.where(halo_z_bins['mid_z'] <= 3.0)]
    print(FLAMINGO_mid_point)
    
    for s in [round(n, 2) for n in np.arange(-1.0,1.1,0.1)]:
        print(s)
        slope_name = f"{float(s):.1f}".replace('.', 'p')
        if s < 0.0:
            slope_name = f"{slope_name}".replace('-', 'minus')
        if s == 0.0:
            slope_name = "0p0"
        with open(f"/cosma8/data/dp004/dc-conl1/FLAMINGO/patchy_screening/data_files/halo_totals/{boxname}/{simname}/{z_sample}/lightcone{lightcone}/FLAMINGO_halo_totals_{im_name}_{slope_name}.txt", "r") as f:
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
    pb.title("\n".join(textwrap.wrap(f'Galaxies in FLAMINGO (boxname = {boxname}, simname = {simname}, {z_sample} sample, stellar cut at mean z = {im}, slope of stellar cut = {slope}, total number of galaxies = {total_available_halos})', width=75)))
    pb.legend(fontsize=4)
    pb.savefig(f"./Plots/FLAMINGO_available_halos_{boxname}_{simname}_{z_sample}_{im_name}_{slope_name}.png", dpi=400)
    pb.clf()
