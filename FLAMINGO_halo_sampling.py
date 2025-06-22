from joblib import Parallel, delayed
import numpy as np
import polars as pl
from imp_patchy_screening import patchyScreening

def halo_sampling(simname, z_sample, mass_cut, n_cut, ncpu):
    sim_list = ['HYDRO_FIDUCIAL','HYDRO_PLANCK','HYDRO_PLANCK_LARGE_NU_FIXED','HYDRO_PLANCK_LARGE_NU_VARY','HYDRO_STRONG_AGN','HYDRO_WEAK_AGN','HYDRO_LOW_SIGMA8','HYDRO_STRONGER_AGN','HYDRO_JETS','HYDRO_STRONGEST_AGN','HYDRO_STRONG_SUPERNOVA','HYDRO_STRONGER_AGN_STRONG_SUPERNOVA','HYDRO_STRONG_JETS']

    try:
        isim = int(simname)
        simname = sim_list[isim]
    except (ValueError, IndexError):
        simname = str(simname)
        
    z_sample = str(z_sample)
    im = float(mass_cut)
    im_name = f"{float(mass_cut):.1f}".replace('.', 'p')
    slope = float(n_cut)
    slope_name = f"{float(n_cut):.1f}".replace('.', 'p')
    if slope < 0.0:
        slope_name = f"{slope_name}".replace('-', 'minus')
    output_path = f'/cosma8/data/dp004/dc-conl1/FLAMINGO/patchy_screening/data_files/mock_halo_catalogs/sampled_halo_data_{simname}_{z_sample}_{im_name}_{slope_name}.parquet'

    halo_z_bins = np.genfromtxt(
        '/cosma8/data/dp004/dc-conl1/FLAMINGO/patchy_screening/data_files/FLAMINGO_halo_redshift_values.txt',
        dtype=[('i',   'i4'),
               ('z_min', 'f8'),
               ('mid_z', 'f8'),
               ('z_max', 'f8')],
        delimiter=None
    )

    z_stellar_cuts = np.loadtxt(f'/cosma8/data/dp004/dc-conl1/FLAMINGO/patchy_screening/data_files/z_dependant_stellar_cuts/z_stellar_cut_data_{z_sample}_{im_name}_{slope_name}.txt')

    dndz_sample = np.loadtxt(f'/cosma8/data/dp004/dc-conl1/FLAMINGO/patchy_screening/data_files/dndz_samples/dndz_galaxies_sampled_{simname}_{z_sample}_{im_name}_{slope_name}.txt')

    # dispatch in parallel
    results = Parallel(n_jobs=int(ncpu),   # adjust to your cores
                       backend='loky')(
                           delayed(process_snapshot)(simname, int(i), z_stellar_cuts, halo_z_bins, dndz_sample)
                           for i in halo_z_bins['i']
                       )

    print(results)

    # concatenate once at the end
    dfs = [df for df in results if df is not None]
    sampled_halo_data = pl.concat(dfs)

    mvir = sampled_halo_data['mvir'].to_numpy()
    nhalo = mvir.size
    print(nhalo)

    sampled_halo_data.write_parquet(output_path, compression='snappy')
    return

def process_snapshot(sim, iz, stellar_cuts, halo_z_bins, dndz_sample):
    """Process a single redshift‐bin index i.  
       Returns (subdf_blue, subdf_green) or None on failure/skip."""
    if halo_z_bins['mid_z'][iz] > 3.0:
        return None

    im = stellar_cuts[int(iz)][1]
    
    ps = patchyScreening(sim, iz, im,  # or however you pass
                         0, 0, 1,
                         lightcone_method=('FULL','shell'))
    ps.filter_stellar_mass()

    nsamp = int(dndz_sample[iz][2])
    print(iz, nsamp)

    try:
        subdf = ps.merge.sample(n=nsamp, with_replacement=False, seed=1000)
    except AttributeError:
        return None    
    print(iz, subdf)

    return subdf


if __name__ == "__main__":
    import sys
    halo_sampling(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[1])
