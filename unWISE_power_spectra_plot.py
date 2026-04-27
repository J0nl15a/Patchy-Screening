import numpy as np, pylab as pb
import sys, yaml, textwrap
import pymaster as nmt
import scipy.interpolate as interpolate
from scipy.signal import savgol_filter
pb.rcParams['font.family'] = 'serif'

def power_spectra_plot(box, isim, iz, im, slope, fits, single, paper_ready=False, 
                       template=False, multi_im = False, multi_slope = False, multi_sim=False, plot_type=None, mle=True, covariance=False, shot_noise=False, 
                       data=None, lc=0, file='png'):

    if round(im, 1) == im:
        im_name = f"{im:.1f}".replace('.', 'p')
    else:
        im_name = f"{im:.3f}".replace('.', 'p')

    if round(slope, 1) == slope:
        slope_name = f"{slope:.1f}".replace('.', 'p')
    else:
        slope_name = f"{slope:.3f}".replace('.', 'p')

    file_extension = str('.' + file) if file in ['png', 'pdf'] else '.png'

    if single:
        ims = [im]
        slopes = [slope]
        ims_name = [im_name]
        slopes_name = [slope_name]
        isim_names = [isim]
        variable_list = [slope]
        variable_name_list = [slope_name]
        min_val = 0.0
        max_val = 1.0
            
    elif not single or not multi_sim:

        if multi_slope:
            min_val = 0.0
            max_val = float(slope)
            slopes = [round(n, 2) for n in np.arange(min_val, max_val+0.1, 0.1)] #CHANGE BACK   
            if round(slope, 1) == slope:
                slopes_name = [f"{round(n, 2):.1f}".replace('.', 'p') for n in np.arange(min_val, max_val+0.1, 0.1)]
            else:
                slopes_name = [f"{round(n, 2)}".replace('.', 'p') for n in np.arange(min_val, max_val+0.1, 0.1)]
            variable_list = slopes
            variable_name_list = slopes_name

        elif multi_im:
            min_val = 10.3
            max_val = float(im)
            ims = [round(n, 2) for n in np.arange(min_val, max_val+0.1, 0.1)]
            if round(im, 1) == im:
                ims_name = [f"{round(n, 2):.1f}".replace('.', 'p') for n in np.arange(min_val, max_val+0.1, 0.1)]
            else:
                ims_name = [f"{round(n, 2)}".replace('.', 'p') for n in np.arange(min_val, max_val+0.1, 0.1)]
            variable_list = ims
            variable_name_list = ims_name

        if multi_sim:
            # Build a per-simulation list (box, isim_dir, lightcone_index) so we can
            # combine entries from different boxes (e.g. L1000 HYDRO_FIDUCIAL and
            # all L2800 lightcones) on the same plot.
            isim_dirs = []      # directory names under data_files (e.g. 'HYDRO_FIDUCIAL' or 'L2p8_m9 (lc=0)')
            isim_boxes = []     # which box each isim_dir belongs to
            sim_lc_idx = []     # which lightcone index to use for that isim
            FLAMINGO_names = []
            FLAMINGO_colors_sorted = []
            if plot_type == None:
                plot_type_extension = ''
            elif plot_type != None:
                plot_type_extension = plot_type + '_'

            # Interpret `lc` argument as number of lightcones for L2800 when plotting
            # multiple L2800 lightcones. L1000 always uses lightcone 0.
            lc_count = int(lc)+1

            if box == 'L1000N1800':
                if not paper_ready:
                    isim_dirs = ['HYDRO_FIDUCIAL', 'HYDRO_LOW_SIGMA8', 'HYDRO_PLANCK', 'HYDRO_PLANCK_LARGE_NU_FIXED', 'HYDRO_PLANCK_LARGE_NU_VARY', 'HYDRO_JETS_published', 'HYDRO_STRONG_JETS_published', 'HYDRO_STRONG_SUPERNOVA', 'HYDRO_WEAK_AGN', 'HYDRO_STRONG_AGN', 'HYDRO_STRONGER_AGN', 'HYDRO_STRONGEST_AGN']
                    FLAMINGO_names = ['L1_m9', 'LS8', 'Planck', 'PlanckNu0p24Fix', 'PlanckNu0p24Var', 'Jet', 'Jet_fgas-4$\sigma$', '$M^*$-$\sigma$', 'fgas+2$\sigma$', 'fgas-2$\sigma$', 'fgas-4$\sigma$', 'fgas-8$\sigma$']
                    FLAMINGO_colors_sorted = ['#117733', '#882255', '#44AA99', '#999933', '#AA4499', '#7EFF4B', '#55E18E', '#FF8C40', '#abd0e6', '#6aaed6', '#3787c0', '#105ba4']
                
                elif paper_ready:
                    if plot_type == 'cosmology':
                        isim_dirs = ['HYDRO_LOW_SIGMA8_STRONGEST_AGN', 'HYDRO_LOW_SIGMA8', 'HYDRO_PLANCK_LARGE_NU_FIXED', 'HYDRO_PLANCK_LARGE_NU_VARY', 'HYDRO_PLANCK']
                        FLAMINGO_names = ['LS8_fgas-8$\sigma$', 'LS8', 'PlanckNu0p24Fix', 'PlanckNu0p24Var', 'Planck']
                        FLAMINGO_colors_sorted = ['#7B68EE', '#882255', '#999933', '#AA4499', '#44AA99']
                    elif plot_type == 'agn_feedback':
                        isim_dirs = ['HYDRO_STRONGEST_AGN', 'HYDRO_STRONGER_AGN', 'HYDRO_STRONG_AGN', 'HYDRO_WEAK_AGN']
                        FLAMINGO_names = ['fgas-8$\sigma$', 'fgas-4$\sigma$', 'fgas-2$\sigma$', 'fgas+2$\sigma$']
                        FLAMINGO_colors_sorted = ['#105ba4', '#3787c0', '#6aaed6', '#abd0e6']
                    elif plot_type == 'other_feedback':
                        isim_dirs = ['HYDRO_STRONG_JETS_published', 'HYDRO_JETS_published', 'HYDRO_STRONG_SUPERNOVA']
                        FLAMINGO_names = ['Jet_fgas-4$\sigma$', 'Jet', '$M^*$-$\sigma$']
                        FLAMINGO_colors_sorted = ['#55E18E', '#7EFF4B', '#FF8C40']
                    isim_dirs += ['HYDRO_FIDUCIAL']
                    FLAMINGO_names += ['L1_m9']
                    FLAMINGO_colors_sorted += ['#117733']
            
                sim_lc_idx = [0] * len(isim_dirs)
                isim_boxes = ['L1000N1800'] * len(isim_dirs)

            elif box == 'L2800N5040':
                # Include the L1000 HYDRO_FIDUCIAL (lightcone 0) as a reference,
                # then include all L2800 lightcones (L2p8_m9 (lc=0..lc_count-1)).
                isim_dirs = ['HYDRO_FIDUCIAL' for l in range(0, lc_count)] + ['HYDRO_FIDUCIAL']
                isim_boxes = (['L2800N5040'] * lc_count) + ['L1000N1800']
                sim_lc_idx = [l for l in range(0, lc_count)] + [0]
                if paper_ready:
                    FLAMINGO_names = [f'L2p8_m9' for l in range(0, lc_count)] + ['L1_m9']
                else:
                    FLAMINGO_names = [f'L2p8_m9 (lc={l})' for l in range(0, lc_count)] + ['L1_m9']
                # simple color palette: keep fiducial green and the L2800 lightcones a bluish tone
                FLAMINGO_colors_sorted = (['#332288'] * lc_count) + ['#117733']

            ims = []
            slopes = []
            ims_name = []
            slopes_name = []
            # amp_vals = [10.798, 10.756, 10.811]
            # slope_vals = [0.659, 0.744, 0.635]
            # Unified handling: iterate over isim_dirs / isim_boxes / sim_lc_idx
            for i_sim in range(len(isim_dirs)):
                sim_box = isim_boxes[i_sim]
                sim_name = isim_dirs[i_sim]
                sim_lc = sim_lc_idx[i_sim]

                if mle:
                    amp = np.loadtxt(f"./data_files/mle_parameters/{sim_box}/{sim_name}/{iz}/lightcone{sim_lc}/mle_values.txt", usecols=1, skiprows=6, max_rows=1, delimiter='=')
                    slope = np.loadtxt(f"./data_files/mle_parameters/{sim_box}/{sim_name}/{iz}/lightcone{sim_lc}/mle_values.txt", usecols=1, skiprows=7, max_rows=1, delimiter='=')
                    ims_name.append(f"{amp:.3f}".replace('.', 'p'))
                    slopes_name.append(f"{slope:.3f}".replace('.', 'p'))
                elif not mle:
                    amp = float(im)
                    slope = float(slope)
                    ims_name.append(f"{amp:.1f}".replace('.', 'p'))
                    slopes_name.append(f"{slope:.1f}".replace('.', 'p'))
                print(amp, slope)
                ims.append(amp)
                slopes.append(slope)


            variable_list = FLAMINGO_names
            variable_name_list = FLAMINGO_names
            min_val = 0.0
            max_val = 1.0

    bin_setup = yaml.safe_load(open("./unWISExLens_lklh/unWISExLens_lklh/config_files/binning_setup.yaml"))
    if str(iz) == 'Blue':
        bin_edges = np.array(bin_setup["Blue_ACT"]["ell_bin_edges"])
    elif str(iz) == 'Green':
        bin_edges = np.array(bin_setup["Green_ACT"]["ell_bin_edges"])

    edges_int = np.rint(bin_edges).astype(int)  # 19.5->20, 51.5->52, ...
    l0 = edges_int[:-1]
    lf = edges_int[1:]

    b = nmt.NmtBin.from_edges(l0, lf)
    print(l0, lf)

    ells = b.get_effective_ells()
    ell_200_mask = np.where(ells > 200)
    ell_namaster = ells[ell_200_mask]
    ell_1000_mask = np.where(ell_namaster > 1000)

    w = 5
    p = 2

    obs_data = np.loadtxt(f'./unWISExLens_lklh/data/v1.0/bandpowers/unWISExACT-DR6_{str(iz).lower()}_baseline_Clgg+Clkk+Clkg.dat', usecols=(0,1,2,3))[ell_200_mask, :].reshape(-1,4)
    Planck_obs_data = np.loadtxt(f'./unWISExLens_lklh/data/v1.0/bandpowers/unWISExPlanck-PR4_{str(iz).lower()}_baseline_Clgg+Clkk+Clkg.dat', usecols=(0,1,2,3))
    Planck_ell_mask = np.where(Planck_obs_data[:,0] > 200)[0]
    Planck_obs_data = Planck_obs_data[Planck_ell_mask, :].reshape(-1,4)

    obs_data_covariance = np.loadtxt(f'./unWISExLens_lklh/data/v1.0/covariances/covmat_Clgg+Clkg_unWISExACT-DR6_{str(iz).lower()}_baseline.dat')
    obs_data_variance = np.diag(obs_data_covariance)
    obs_data_std = np.sqrt(obs_data_variance[:int(len(obs_data_variance)/2)])[ell_200_mask]
    obs_data_std_cross = np.sqrt(obs_data_variance[int(len(obs_data_variance)/2):])[ell_200_mask]

    Planck_obs_data_covariance = np.loadtxt(f'./unWISExLens_lklh/data/v1.0/covariances/covmat_Clgg+Clkg_unWISExPlanck-PR4_{str(iz).lower()}_baseline.dat')
    Planck_obs_data_variance = np.diag(Planck_obs_data_covariance)
    Planck_obs_data_std = np.sqrt(Planck_obs_data_variance[:int(len(Planck_obs_data_variance)/2)])[Planck_ell_mask]
    Planck_obs_data_std_cross = np.sqrt(Planck_obs_data_variance[int(len(Planck_obs_data_variance)/2):])[Planck_ell_mask]

    if template:
        initial_auto_spectra = np.loadtxt(f'/cosma8/data/dp004/dc-conl1/FLAMINGO/patchy_screening/data_files/power_spectra/galaxy_galaxy/{box}/{isim}/{iz}/lightcone{lc}/galaxy_galaxy_power_spectrum_10p8_0p5.txt', 
                                          delimiter=' ', skiprows=1, usecols=2)
        initial_cross_spectra = np.loadtxt(f'/cosma8/data/dp004/dc-conl1/FLAMINGO/patchy_screening/data_files/power_spectra/kappa_galaxy/{box}/{isim}/{iz}/lightcone{lc}/kappa_galaxy_power_spectrum_10p8_0p5.txt', 
                                           delimiter=' ', skiprows=1, usecols=1)
        obs_data[:,1] /=  initial_auto_spectra
        obs_data[:,3] /=  initial_cross_spectra
        func_auto = interpolate.interp1d(ell_namaster, initial_auto_spectra, kind='cubic', fill_value="extrapolate")
        func_cross = interpolate.interp1d(ell_namaster, initial_cross_spectra, kind='cubic', fill_value="extrapolate")
        Planck_obs_data[:,1] /= func_auto(Planck_obs_data[:,0])
        Planck_obs_data[:,3] /= func_cross(Planck_obs_data[:,0])
        template_prefix = '_template'
    else:
        template_prefix = ''

    if data == None:
        nhalos_list = []
        auto_spectra_list = []
        cross_spectra_list = []
        auto_spectra_covariance_list = []
        cross_spectra_covariance_list = []
        auto_spectra_shot_noise_list = []

        for i in range(len(variable_list)):

            if single:
                if shot_noise:
                    auto_power_spectra = np.loadtxt(f'./data_files/power_spectra/galaxy_galaxy/{box}/{isim}/{iz}/lightcone{lc}/galaxy_galaxy_power_spectrum_{im_name}_{slope_name}.txt', skiprows=1, usecols=(1,3) if covariance else 1)
                    auto_power_spectra_unsmoothed_component = auto_power_spectra[np.where(ell_namaster <= 1000)]
                    auto_power_spectra_smooth_component = savgol_filter(auto_power_spectra[ell_1000_mask], window_length=w, polyorder=p) if w > 0 and p > 0 else auto_power_spectra[ell_1000_mask]
                    auto_power_spectra = np.concatenate((auto_power_spectra_unsmoothed_component, auto_power_spectra_smooth_component))
                    with open(f"./data_files/dndz_samples/{box}/{isim}/{iz}/lightcone{lc}/dndz_galaxies_sampled_{im_name}_{slope_name}.txt", "r") as f:
                        first_line = f.readline().strip()
                        nhalos = int(first_line.split(":")[-1])
                        nhalos_list.append(nhalos)
                        auto_spectra_shot_noise_list.append((4*np.pi)/nhalos)
                elif not shot_noise:
                    auto_power_spectra = np.loadtxt(f'./data_files/power_spectra/galaxy_galaxy/{box}/{isim}/{iz}/lightcone{lc}/galaxy_galaxy_power_spectrum_{im_name}_{slope_name}.txt', skiprows=1, usecols=(2,3) if covariance else 2)
                    auto_power_spectra_unsmoothed_component = auto_power_spectra[np.where(ell_namaster <= 1000)]
                    auto_power_spectra_smooth_component = savgol_filter(auto_power_spectra[ell_1000_mask], window_length=w, polyorder=p) if w > 0 and p > 0 else auto_power_spectra[ell_1000_mask]
                    auto_power_spectra = np.concatenate((auto_power_spectra_unsmoothed_component, auto_power_spectra_smooth_component))
                    with open(f"./data_files/dndz_samples/{box}/{isim}/{iz}/lightcone{lc}/dndz_galaxies_sampled_{im_name}_{slope_name}.txt", "r") as f:
                        first_line = f.readline().strip()
                        nhalos = int(first_line.split(":")[-1])
                        nhalos_list.append(nhalos)
                cross_power_spectra = np.loadtxt(f'./data_files/power_spectra/kappa_galaxy/{box}/{isim}/{iz}/lightcone{lc}/kappa_galaxy_power_spectrum_{im_name}_{slope_name}.txt', skiprows=1, usecols=(1,2) if covariance else 1)
                cross_power_spectra_unsmoothed_component = cross_power_spectra[np.where(ell_namaster <= 1000)]
                cross_power_spectra_smooth_component = savgol_filter(cross_power_spectra[ell_1000_mask], window_length=w, polyorder=p) if w > 0 and p > 0 else cross_power_spectra[ell_1000_mask]
                cross_power_spectra = np.concatenate((cross_power_spectra_unsmoothed_component, cross_power_spectra_smooth_component))

            elif multi_im:
                if shot_noise:
                    auto_power_spectra = np.loadtxt(f'./data_files/power_spectra/galaxy_galaxy/{box}/{isim}/{iz}/lightcone{lc}/galaxy_galaxy_power_spectrum_{ims_name[i]}_{slope_name}.txt', skiprows=1, usecols=(1,3) if covariance else 1)
                    auto_power_spectra_unsmoothed_component = auto_power_spectra[np.where(ell_namaster <= 1000)]
                    auto_power_spectra_smooth_component = savgol_filter(auto_power_spectra[ell_1000_mask], window_length=w, polyorder=p) if w > 0 and p > 0 else auto_power_spectra[ell_1000_mask]
                    auto_power_spectra = np.concatenate((auto_power_spectra_unsmoothed_component, auto_power_spectra_smooth_component))
                    with open(f"./data_files/dndz_samples/{box}/{isim}/{iz}/lightcone{lc}/dndz_galaxies_sampled_{ims_name[i]}_{slope_name}.txt", "r") as f:
                        first_line = f.readline().strip()
                        nhalos = int(first_line.split(":")[-1])
                        nhalos_list.append(nhalos)
                        auto_spectra_shot_noise_list.append((4*np.pi)/nhalos)
                elif not shot_noise:
                    auto_power_spectra = np.loadtxt(f'./data_files/power_spectra/galaxy_galaxy/{box}/{isim}/{iz}/lightcone{lc}/galaxy_galaxy_power_spectrum_{ims_name[i]}_{slope_name}.txt', skiprows=1, usecols=(2,3) if covariance else 2)
                    auto_power_spectra_unsmoothed_component = auto_power_spectra[np.where(ell_namaster <= 1000)]
                    auto_power_spectra_smooth_component = savgol_filter(auto_power_spectra[ell_1000_mask], window_length=w, polyorder=p) if w > 0 and p > 0 else auto_power_spectra[ell_1000_mask]
                    auto_power_spectra = np.concatenate((auto_power_spectra_unsmoothed_component, auto_power_spectra_smooth_component))
                    with open(f"./data_files/dndz_samples/{box}/{isim}/{iz}/lightcone{lc}/dndz_galaxies_sampled_{ims_name[i]}_{slope_name}.txt", "r") as f:
                        first_line = f.readline().strip()
                        nhalos = int(first_line.split(":")[-1])
                        nhalos_list.append(nhalos)
                cross_power_spectra = np.loadtxt(f'./data_files/power_spectra/kappa_galaxy/{box}/{isim}/{iz}/lightcone{lc}/kappa_galaxy_power_spectrum_{ims_name[i]}_{slope_name}.txt', skiprows=1, usecols=(1,2) if covariance else 1)
                cross_power_spectra_unsmoothed_component = cross_power_spectra[np.where(ell_namaster <= 1000)]
                cross_power_spectra_smooth_component = savgol_filter(cross_power_spectra[ell_1000_mask], window_length=w, polyorder=p) if w > 0 and p > 0 else cross_power_spectra[ell_1000_mask]
                cross_power_spectra = np.concatenate((cross_power_spectra_unsmoothed_component, cross_power_spectra_smooth_component))

            elif multi_slope:
                if shot_noise:
                    auto_power_spectra = np.loadtxt(f'./data_files/power_spectra/galaxy_galaxy/{box}/{isim}/{iz}/lightcone{lc}/galaxy_galaxy_power_spectrum_{im_name}_{slopes_name[i]}.txt', skiprows=1, usecols=(1,3) if covariance else 1)
                    auto_power_spectra_unsmoothed_component = auto_power_spectra[np.where(ell_namaster <= 1000)]
                    auto_power_spectra_smooth_component = savgol_filter(auto_power_spectra[ell_1000_mask], window_length=w, polyorder=p) if w > 0 and p > 0 else auto_power_spectra[ell_1000_mask]
                    auto_power_spectra = np.concatenate((auto_power_spectra_unsmoothed_component, auto_power_spectra_smooth_component))
                    with open(f"./data_files/dndz_samples/{box}/{isim}/{iz}/lightcone{lc}/dndz_galaxies_sampled_{im_name}_{slopes_name[i]}.txt", "r") as f:
                        first_line = f.readline().strip()
                        nhalos = int(first_line.split(":")[-1])
                        nhalos_list.append(nhalos)
                        auto_spectra_shot_noise_list.append((4*np.pi)/nhalos)
                elif not shot_noise:
                    auto_power_spectra = np.loadtxt(f'./data_files/power_spectra/galaxy_galaxy/{box}/{isim}/{iz}/lightcone{lc}/galaxy_galaxy_power_spectrum_{im_name}_{slopes_name[i]}.txt', skiprows=1, usecols=(2,3) if covariance else 2)
                    auto_power_spectra_unsmoothed_component = auto_power_spectra[np.where(ell_namaster <= 1000)]
                    auto_power_spectra_smooth_component = savgol_filter(auto_power_spectra[ell_1000_mask], window_length=w, polyorder=p) if w > 0 and p > 0 else auto_power_spectra[ell_1000_mask]
                    auto_power_spectra = np.concatenate((auto_power_spectra_unsmoothed_component, auto_power_spectra_smooth_component))
                    with open(f"./data_files/dndz_samples/{box}/{isim}/{iz}/lightcone{lc}/dndz_galaxies_sampled_{im_name}_{slopes_name[i]}.txt", "r") as f:
                        first_line = f.readline().strip()
                        nhalos = int(first_line.split(":")[-1])
                        nhalos_list.append(nhalos)
                cross_power_spectra = np.loadtxt(f'./data_files/power_spectra/kappa_galaxy/{box}/{isim}/{iz}/lightcone{lc}/kappa_galaxy_power_spectrum_{im_name}_{slopes_name[i]}.txt', skiprows=1, usecols=(1,2) if covariance else 1)
                cross_power_spectra_unsmoothed_component = cross_power_spectra[np.where(ell_namaster <= 1000)]
                cross_power_spectra_smooth_component = savgol_filter(cross_power_spectra[ell_1000_mask], window_length=w, polyorder=p) if w > 0 and p > 0 else cross_power_spectra[ell_1000_mask]
                cross_power_spectra = np.concatenate((cross_power_spectra_unsmoothed_component, cross_power_spectra_smooth_component))

            elif multi_sim:
                if shot_noise:
                    auto_power_spectra = np.loadtxt(f"./data_files/power_spectra/galaxy_galaxy/{isim_boxes[i]}/{isim_dirs[i]}/{iz}/lightcone{sim_lc_idx[i]}/galaxy_galaxy_power_spectrum_{ims_name[i]}_{slopes_name[i]}.txt", skiprows=1, usecols=(1,3) if covariance else 1)
                    auto_power_spectra_unsmoothed_component = auto_power_spectra[np.where(ell_namaster <= 1000)]
                    auto_power_spectra_smooth_component = savgol_filter(auto_power_spectra[ell_1000_mask], window_length=w, polyorder=p) if w > 0 and p > 0 else auto_power_spectra[ell_1000_mask]
                    auto_power_spectra = np.concatenate((auto_power_spectra_unsmoothed_component, auto_power_spectra_smooth_component))
                    with open(f"./data_files/dndz_samples/{isim_boxes[i]}/{isim_dirs[i]}/{iz}/lightcone{sim_lc_idx[i]}/dndz_galaxies_sampled_{ims_name[i]}_{slopes_name[i]}.txt", "r") as f:
                        first_line = f.readline().strip()
                        nhalos = int(first_line.split(":")[-1])
                        nhalos_list.append(nhalos)
                        auto_spectra_shot_noise_list.append((4*np.pi)/nhalos)
                elif not shot_noise:
                    auto_power_spectra = np.loadtxt(f"./data_files/power_spectra/galaxy_galaxy/{isim_boxes[i]}/{isim_dirs[i]}/{iz}/lightcone{sim_lc_idx[i]}/galaxy_galaxy_power_spectrum_{ims_name[i]}_{slopes_name[i]}.txt", skiprows=1, usecols=(2,3) if covariance else 2)
                    auto_power_spectra_unsmoothed_component = auto_power_spectra[np.where(ell_namaster <= 1000)]
                    auto_power_spectra_smooth_component = savgol_filter(auto_power_spectra[ell_1000_mask], window_length=w, polyorder=p) if w > 0 and p > 0 else auto_power_spectra[ell_1000_mask]
                    auto_power_spectra = np.concatenate((auto_power_spectra_unsmoothed_component, auto_power_spectra_smooth_component))
                    with open(f"./data_files/dndz_samples/{isim_boxes[i]}/{isim_dirs[i]}/{iz}/lightcone{sim_lc_idx[i]}/dndz_galaxies_sampled_{ims_name[i]}_{slopes_name[i]}.txt", "r") as f:
                        first_line = f.readline().strip()
                        nhalos = int(first_line.split(":")[-1])
                        nhalos_list.append(nhalos)
                cross_power_spectra = np.loadtxt(f"./data_files/power_spectra/kappa_galaxy/{isim_boxes[i]}/{isim_dirs[i]}/{iz}/lightcone{sim_lc_idx[i]}/kappa_galaxy_power_spectrum_{ims_name[i]}_{slopes_name[i]}.txt", skiprows=1, usecols=(1,2) if covariance else 1)
                cross_power_spectra_unsmoothed_component = cross_power_spectra[np.where(ell_namaster <= 1000)]
                cross_power_spectra_smooth_component = savgol_filter(cross_power_spectra[ell_1000_mask], window_length=w, polyorder=p) if w > 0 and p > 0 else cross_power_spectra[ell_1000_mask]
                cross_power_spectra = np.concatenate((cross_power_spectra_unsmoothed_component, cross_power_spectra_smooth_component))
            
            if template:
                if covariance:
                    auto_spectra_list.append(auto_power_spectra / initial_auto_spectra)
                    cross_spectra_list.append(cross_power_spectra / initial_cross_spectra)
                else:
                    auto_spectra_list.append(auto_power_spectra[:,0] / initial_auto_spectra)
                    cross_spectra_list.append(cross_power_spectra[:,0] / initial_cross_spectra)
            elif not template:
                if covariance:
                    auto_spectra_list.append(auto_power_spectra[:,0])
                    cross_spectra_list.append(cross_power_spectra[:,0])
                else:
                    auto_spectra_list.append(auto_power_spectra)
                    cross_spectra_list.append(cross_power_spectra)
        
            if covariance:
                auto_spectra_covariance_list.append(auto_power_spectra[:,1])
                cross_spectra_covariance_list.append(cross_power_spectra[:,1])
    
    elif data != None:
        slopes = data['slopes']
        nhalos_list = data['nhalos']
        mean_mstar = data['mean_mstar']
        chi2 = data['chi2_list']
        auto_spectra_list = data['auto_spectra_list']
        cross_spectra_list = data['cross_spectra_list']
        if covariance:
            auto_spectra_covariance_list = data['auto_spectra_covariance_list']
            cross_spectra_covariance_list = data['cross_spectra_covariance_list']
        if shot_noise:
            auto_spectra_shot_noise_list = data['auto_spectra_shot_noise_list']

    mle_log_likelihood = []
    mle_chi2_auto = []
    mle_chi2_cross = []
    if multi_sim:
        for i in range(len(isim_dirs)):
            mle_log_likelihood.append(f"{np.loadtxt(f"./data_files/mle_parameters/{isim_boxes[i]}/{isim_dirs[i]}/{iz}/lightcone{sim_lc_idx[i]}/mle_values.txt", 
                                                    usecols=1, max_rows=1, delimiter='='):.3f}")
            mle_chi2_auto.append(f"{np.loadtxt(f"./data_files/mle_parameters/{isim_boxes[i]}/{isim_dirs[i]}/{iz}/lightcone{sim_lc_idx[i]}/mle_values.txt", 
                                                    usecols=1, skiprows=3, max_rows=1, delimiter='='):.3f}")
            mle_chi2_cross.append(f"{np.loadtxt(f"./data_files/mle_parameters/{isim_boxes[i]}/{isim_dirs[i]}/{iz}/lightcone{sim_lc_idx[i]}/mle_values.txt", 
                                                    usecols=1, skiprows=5, max_rows=1, delimiter='='):.3f}")
    elif single:
        mle_log_likelihood.append(f"{np.loadtxt(f"./data_files/mle_parameters/{box}/{isim}/{iz}/lightcone{lc}/mle_values.txt", 
                                                usecols=1, max_rows=1, delimiter='='):.3f}")
        mle_chi2_auto.append(f"{np.loadtxt(f"./data_files/mle_parameters/{box}/{isim}/{iz}/lightcone{lc}/mle_values.txt", 
                                                usecols=1, skiprows=3, max_rows=1, delimiter='='):.3f}")
        mle_chi2_cross.append(f"{np.loadtxt(f"./data_files/mle_parameters/{box}/{isim}/{iz}/lightcone{lc}/mle_values.txt", 
                                                usecols=1, skiprows=5, max_rows=1, delimiter='='):.3f}")
        
    else:
        mle_log_likelihood = ['' for i in range(len(variable_list))]

    if not single and data == None:
        if multi_slope:
            slopes = slopes[::-1]
        elif multi_im:
            ims = ims[::-1]
        elif multi_sim:
            if paper_ready:
                variable_list = FLAMINGO_names
                variable_name_list = FLAMINGO_names
            # keep isim_dirs in sync with label ordering
            isim_dirs = isim_dirs[::-1]
            isim_boxes = isim_boxes[::-1]
            sim_lc_idx = sim_lc_idx[::-1]
            FLAMINGO_colors_sorted = FLAMINGO_colors_sorted[::-1]
            ims = ims[::-1]
            slopes = slopes[::-1]
            mle_log_likelihood = mle_log_likelihood[::-1]
            mle_chi2_auto = mle_chi2_auto[::-1]
            mle_chi2_cross = mle_chi2_cross[::-1]
        variable_list = variable_list[::-1]
        variable_name_list = variable_name_list[::-1]
        auto_spectra_list = auto_spectra_list[::-1]
        cross_spectra_list = cross_spectra_list[::-1]
        auto_spectra_covariance_list = auto_spectra_covariance_list[::-1]
        cross_spectra_covariance_list = cross_spectra_covariance_list[::-1]
        auto_spectra_shot_noise_list = auto_spectra_shot_noise_list[::-1]
        nhalos_list = nhalos_list[::-1]

    print(auto_spectra_list)
    print(cross_spectra_list)
    if covariance:
        print(auto_spectra_covariance_list)
        print(cross_spectra_covariance_list)


    # PLOTTING SECTION

    # galaxy-galaxy auto-power spectra

    if multi_sim:
        fig, (ax, ax_ratio) = pb.subplots(
        2, 1, figsize=(8, 7.2),
        sharex=True,
        gridspec_kw={"height_ratios": [3.0, 1.0], "hspace": 0.05},
        )
    else:
        fig, ax = pb.subplots(1, 1, figsize=(8,6))

    if multi_sim and box == 'L2800N5040':
        FLAMINGO_names = FLAMINGO_names[::-1]
        l1000_idx = [i for i, b in enumerate(isim_boxes) if b == 'L1000N1800']
        l2800_idx = [i for i, b in enumerate(isim_boxes) if b == 'L2800N5040']

        # L1000 reference line
        i_ref = l1000_idx[0]
        fiducial_spec = np.asarray(auto_spectra_list[i_ref], dtype=float)

        ax.plot(
            ell_namaster,
            fiducial_spec,
            color=FLAMINGO_colors_sorted[i_ref],
            linestyle='solid',
            alpha=0.9,
            label=FLAMINGO_names[i_ref]
        )

        ax_ratio.axhline(1.0, color=FLAMINGO_colors_sorted[i_ref], linestyle='solid', alpha=0.8)

        # L2800 mean + min/max envelope
        l2800_auto = np.asarray([auto_spectra_list[i] for i in l2800_idx], dtype=float)

        l2800_mean = np.mean(l2800_auto, axis=0)
        l2800_min = np.min(l2800_auto, axis=0)
        l2800_max = np.max(l2800_auto, axis=0)

        ax.plot(
            ell_namaster,
            l2800_mean,
            color=FLAMINGO_colors_sorted[l2800_idx[0]],
            linestyle='solid',
            alpha=0.9,
            label=FLAMINGO_names[l2800_idx[0]]
        )

        ax.fill_between(
            ell_namaster,
            l2800_min,
            l2800_max,
            color=FLAMINGO_colors_sorted[l2800_idx[0]],
            alpha=0.25,
            linewidth=0,
            # label='L2p8_m9 lightcone min/max'
        )

        # Ratio panel: mean and min/max relative to L1000 fiducial
        ax_ratio.plot(
            ell_namaster,
            l2800_mean / fiducial_spec,
            color=FLAMINGO_colors_sorted[l2800_idx[0]],
            linestyle='solid',
            alpha=0.9
        )

        ax_ratio.fill_between(
            ell_namaster,
            l2800_min / fiducial_spec,
            l2800_max / fiducial_spec,
            color=FLAMINGO_colors_sorted[l2800_idx[0]],
            alpha=0.25,
            linewidth=0
        )
    else:
        for i, var in enumerate(variable_list):
            print(var)
            if var == min_val:
                var = abs(var)
                
                if multi_sim:
                    # determine fiducial (L1000 HYDRO_FIDUCIAL) index if present
                    try:
                        fiducial_idx = next(i for i, (b, n) in enumerate(zip(isim_boxes, isim_dirs)) if b == 'L1000N1800' and n == 'HYDRO_FIDUCIAL')
                    except StopIteration:
                        fiducial_idx = 0
                    fiducial_spec = np.asarray(auto_spectra_list[fiducial_idx], dtype=float)

                    if paper_ready and data == None:
                        line, = ax.plot(ell_namaster, auto_spectra_list[i], 
                                    linestyle='dashed', alpha=0.8, color=FLAMINGO_colors_sorted[i], 
                                    label=f'{var}')
                        
                    else:
                        line, = ax.plot(ell_namaster, auto_spectra_list[i], 
                                    linestyle='dashed', color=FLAMINGO_colors_sorted[i], alpha=0.8, 
                                    label=f'{var}, {ims[i]}, {slopes[i]}, {mean_mstar[i]:.5f}, {nhalos_list[i]}, {chi2[i]:.3f}' if data != None else f'{var}, {ims[i]}, {slopes[i]}, {nhalos_list[i]}, {mle_log_likelihood[i]}')

                    ratio = auto_spectra_list[i] / fiducial_spec
                    ax_ratio.plot(
                        ell_namaster, ratio,
                        color=line.get_color(),
                        linestyle=line.get_linestyle(),
                        alpha=0.9
                    )

                else:
                    line, = ax.plot(ell_namaster, auto_spectra_list[i], 
                                    linestyle='dashed', alpha=0.8, 
                                    label=f'{var}, {mean_mstar[i]:.5f}, {nhalos_list[i]}, {chi2[i]:.3f}' if data != None else f'{var}')
                if covariance:
                    ax.fill_between(x=ell_namaster, 
                                    y1=(auto_spectra_list[i] + auto_spectra_covariance_list[i]), y2=(auto_spectra_list[i] - auto_spectra_covariance_list[i]), 
                                    linewidth=0, alpha=.3)
                if shot_noise:
                    line_color = line.get_color()
                    ax.hlines(y=auto_spectra_shot_noise_list[i]*1e5, xmin=-1, xmax=ell_namaster[-1]+10, linestyle='dashed', color=line_color, alpha=0.3)
            else:
                if multi_sim:
                    try:
                        fiducial_idx = next(i for i, (b, n) in enumerate(zip(isim_boxes, isim_dirs)) if b == 'L1000N1800' and n == 'HYDRO_FIDUCIAL')
                    except StopIteration:
                        fiducial_idx = 0
                    fiducial_spec = np.asarray(auto_spectra_list[fiducial_idx], dtype=float)

                    if paper_ready and data == None:
                        line, = ax.plot(ell_namaster, auto_spectra_list[i], 
                                        linestyle='solid', alpha=0.8, color=FLAMINGO_colors_sorted[i], 
                                        label=f'{var}')
                        
                        ax.text(
                            0.1,                      # x position in axes coords
                            0.3 - 0.045*i,            # y position, spaced by loop index
                            f'({float(mle_chi2_auto[i]) - float(mle_chi2_auto[0]):.1f})' if i != 0 else f'{float(mle_chi2_auto[i]):.1f}',
                            transform=ax.transAxes,
                            color=FLAMINGO_colors_sorted[i],
                            fontsize=10,
                            fontfamily='serif',
                            ha='left',
                            va='top'
                        )
                    else:
                        line, = ax.plot(ell_namaster, auto_spectra_list[i], 
                                    linestyle='solid', alpha=0.8, color=FLAMINGO_colors_sorted[i], 
                                    label=f'{var}, {ims[i]}, {slopes[i]}, {mean_mstar[i]:.5f}, {nhalos_list[i]}, {chi2[i]:.3f}' if data != None else f'{var}, {ims[i]}, {slopes[i]}, {nhalos_list[i]}, {mle_log_likelihood[i]}')
                        
                    ratio = auto_spectra_list[i] / fiducial_spec
                    ax_ratio.plot(
                        ell_namaster, ratio,
                        color=line.get_color(),
                        linestyle=line.get_linestyle(),
                        alpha=0.9
                    )
                
                else:
                    line, = ax.plot(ell_namaster, auto_spectra_list[i], 
                                    linestyle= 'dashed' if var > max_val else 'solid', dash_capstyle='round' if var > max_val else 'projecting',  alpha=0.8, 
                                    label=f'{var}, {mean_mstar[i]:.5f}, {nhalos_list[i]}, {chi2[i]:.3f}' if data != None else f'{var}')
                if covariance:
                    ax.fill_between(x=ell_namaster, 
                                    y1=(auto_spectra_list[i] + auto_spectra_covariance_list[i]), y2=(auto_spectra_list[i] - auto_spectra_covariance_list[i]), 
                                    linewidth=0, alpha=.3)
                if shot_noise:
                    line_color = line.get_color()
                    ax.hlines(y=auto_spectra_shot_noise_list[i]*1e5, xmin=-1, xmax=ell_namaster[-1]+10, linestyle= 'dashed' if var > max_val else 'solid', dash_capstyle='round' if var > max_val else 'projecting',  color=line_color, alpha=0.3)

    ax.plot(obs_data[:,0], obs_data[:,1]*1e5, 
            color='k', marker='.', markersize=5, linewidth=0, label='ACT x unWISE')
    ax.fill_between(x=obs_data[:,0], 
                    y1=(obs_data[:,1]+obs_data_std)*1e5, y2=(obs_data[:,1]-obs_data_std)*1e5, 
                    color='k', linewidth=0, alpha=.3)
    
    if multi_sim:
        ratio_act_obs = obs_data[:,1]*1e5 / fiducial_spec
        ax_ratio.plot(
            ell_namaster, ratio_act_obs,
            marker='.', markersize=5,
            color='k', linewidth=0,
            alpha=0.9
        )
        # simple propagation assuming +/- is 1σ on the model only (reference treated fixed)
        ratio_hi = (obs_data[:,1] + obs_data_std) * 1e5 / fiducial_spec
        ratio_lo = (obs_data[:,1] - obs_data_std) * 1e5 / fiducial_spec
        ax_ratio.fill_between(
            x=ell_namaster, y1=ratio_hi, y2=ratio_lo,
            linewidth=0, alpha=0.3, color='k'
        )
    
    ax.plot(Planck_obs_data[:,0], Planck_obs_data[:,1]*1e5, 
            color='r', marker='.', markersize=5, linewidth=0, label='Planck x unWISE')
    ax.fill_between(x=Planck_obs_data[:,0], 
                    y1=(Planck_obs_data[:,1]+Planck_obs_data_std)*1e5, y2=(Planck_obs_data[:,1]-Planck_obs_data_std)*1e5, 
                    color='r', linewidth=0, alpha=.3)
    
    if multi_sim:
        ratio_planck_obs = Planck_obs_data[:,1]*1e5 / fiducial_spec[:len(Planck_obs_data[:,0])]
        ax_ratio.plot(
            Planck_obs_data[:,0], ratio_planck_obs,
            marker='.', markersize=5,
            color='r', linewidth=0,
            alpha=0.9
        )
        ratio_planck_hi = (Planck_obs_data[:,1] + Planck_obs_data_std) * 1e5 / fiducial_spec[:len(Planck_obs_data[:,0])]
        ratio_planck_lo = (Planck_obs_data[:,1] - Planck_obs_data_std) * 1e5 / fiducial_spec[:len(Planck_obs_data[:,0])]
        ax_ratio.fill_between(
            x=Planck_obs_data[:,0], y1=ratio_planck_hi, y2=ratio_planck_lo,
            linewidth=0, alpha=0.3, color='r'
        )

    ax.set_ylabel('$C^{gg}_{\mathrm{\ell}}x10^5$')
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(200, 4000)
    ax.set_ylim(bottom=1e-2)

    if multi_sim:
        ax_ratio.axhline(y=1.0, color='k', linestyle='dashed', alpha=0.7)
        ax_ratio.set_xlabel('Multipole moment $\mathrm{\ell}$')
        ax_ratio.set_ylabel('Model / Fiducial')
        ax_ratio.set_xscale("log")
        ax_ratio.set_xlim(200, 4000)
        ax_ratio.set_ylim(0.9, 1.1)
        ax_ratio.grid(alpha=0.25)
    else:
        ax.set_xlabel('Multipole moment $\mathrm{\ell}$')
        ax.set_xscale("log")
        ax.set_xlim(200, 4000)

    if paper_ready:
        pass
    else:
        if single or multi_slope:
            if shot_noise:
                ax.set_title("\n".join(textwrap.wrap(
                    rf"Power Spectrum of the Galaxy Overdensity map (with shot-noise) "
                    rf"(box={box}, sim={isim}, {iz} sample, "
                    rf"log$M_*$={im}, primary CMB={fits})",
                    width=80)))
            else:
                ax.set_title("\n".join(textwrap.wrap(
                    rf"Power Spectrum of the Galaxy Overdensity map (shot-noise subtracted) "
                    rf"(box={box}, sim={isim}, {iz} sample, "
                    rf"log$M_*$={im}, primary CMB={fits})",
                    width=80)))
        elif multi_im:
            if shot_noise:
                ax.set_title("\n".join(textwrap.wrap(
                    rf"Power Spectrum of the Galaxy Overdensity map (with shot-noise) "
                    rf"(box={box}, sim={isim}, {iz} sample, "
                    rf"slope={slope}, primary CMB={fits})",
                    width=80)))
            else:
                ax.set_title("\n".join(textwrap.wrap(
                    rf"Power Spectrum of the Galaxy Overdensity map (shot-noise subtracted) "
                    rf"(box={box}, sim={isim}, {iz} sample, "
                    rf"slope={slope}, primary CMB={fits})",
                    width=80)))
        elif multi_sim:
            if shot_noise:
                ax.set_title("\n".join(textwrap.wrap(
                    rf"Power Spectrum of the Galaxy Overdensity map (with shot-noise) "
                    rf"(box={box}, {iz} sample, log$M_*$ and slope MLE optimised, "
                    rf"primary CMB={fits})",
                    width=80)))
            else:
                ax.set_title("\n".join(textwrap.wrap(
                    rf"Power Spectrum of the Galaxy Overdensity map (shot-noise subtracted) "
                    rf"(box={box}, {iz} sample, log$M_*$ and slope MLE optimised, "
                    rf"primary CMB={fits})",
                    width=80)))

    if single:
        ax.legend(title="Slope, $N_{halo}$, $-\log \mathcal{L}(\\theta \mid x)$" if data == None else "Slope, $log_{10}M_{*,mean}$, $N_{halo}$, $\chi^2$", 
                  fontsize=9, ncols=1, loc='upper right')
        if shot_noise:
            pb.savefig(f'./Plots/halo_map_gg_power_spectrum_{box}_{isim}_{iz}_{im_name}_{slope_name}_{fits}_ntotal{template_prefix}{file_extension}', dpi=400)
        else:
            pb.savefig(f'./Plots/halo_map_gg_power_spectrum_{box}_{isim}_{iz}_{im_name}_{slope_name}_{fits}_ntotal_shot_noise_subtracted{template_prefix}{file_extension}', dpi=400)
    elif multi_slope:
        if paper_ready:
            ax.legend(title="Slope" if data == None else "Slope", 
                      fontsize=6 if max(variable_list) > max_val else 9, ncols=3 if max(variable_list) > max_val else 2, loc='upper right')
        else:
            ax.legend(title="Slope, $N_{halo}$, $-\log \mathcal{L}(\\theta \mid x)$" if data == None else "Slope, $log_{10}M_{*,mean}$, $N_{halo}$, $\chi^2$", 
                      fontsize=6 if max(variable_list) > max_val else 9, ncols=3 if max(variable_list) > max_val else 2, loc='upper right')
        if shot_noise:
            pb.savefig(f'./Plots/halo_map_gg_power_spectrum_{box}_{isim}_{iz}_{im_name}_all_slopes_{fits}_ntotal{template_prefix}{file_extension}', dpi=400)
        else:   
            pb.savefig(f'./Plots/halo_map_gg_power_spectrum_{box}_{isim}_{iz}_{im_name}_all_slopes_{fits}_ntotal_shot_noise_subtracted{template_prefix}{file_extension}', dpi=400)
    elif multi_im:
        if paper_ready:
            ax.legend(title="Mass cut" if data == None else "Mass cut", 
                      fontsize=9, ncols=2, loc='upper right')
        else:
            ax.legend(title="Mass cut, $N_{halo}$, $-\log \mathcal{L}(\\theta \mid x)$" if data == None else "Mass cut, $log_{10}M_{*,mean}$, $N_{halo}$, $\chi^2$", 
                      fontsize=9, ncols=2, loc='upper right')
        if shot_noise:
            pb.savefig(f'./Plots/halo_map_gg_power_spectrum_{box}_{isim}_{iz}_{slope_name}_all_mass_cuts_{fits}_ntotal{template_prefix}{file_extension}', dpi=400)
        else:
            pb.savefig(f'./Plots/halo_map_gg_power_spectrum_{box}_{isim}_{iz}_{slope_name}_all_mass_cuts_{fits}_ntotal_shot_noise_subtracted{template_prefix}{file_extension}', dpi=400)
    elif multi_sim:
        if paper_ready and data == None:
            ax.legend(title="Simulation", 
                      fontsize=9, ncols=1, loc='upper right')
        else:
            ax.legend(title="Simulation, Amp, Slope, $N_{halo}$, $-\log \mathcal{L}(\\theta \mid x)$" if data == None else "Simulation, Amp, Slope, $log_{10}M_{*,mean}$, $N_{halo}$, $\chi^2$", 
                      fontsize=6, ncols=1, loc='upper right')
        if shot_noise:
            pb.savefig(f'./Plots/halo_map_gg_power_spectrum_{box}_all_sims_{plot_type_extension}{iz}_mle_amp_slope_{fits}_ntotal{template_prefix}{file_extension}', dpi=400)
        else:
            pb.savefig(f'./Plots/halo_map_gg_power_spectrum_{box}_all_sims_{plot_type_extension}{iz}_mle_amp_slope_{fits}_ntotal_shot_noise_subtracted{template_prefix}{file_extension}', dpi=400)
    pb.close(fig)

    # galaxy-galaxy auto-power spectra multiplied by ell

    pb.figure(figsize=(8,6))
    for i, var in enumerate(variable_list):
        print(var)
        if var == min_val:
            var = abs(var)
            
            if multi_sim:
                pb.plot(ell_namaster, auto_spectra_list[i] *ell_namaster, 
                            linestyle='dashed', alpha=0.8, 
                            label=f'{var}, {ims[i]}, {slopes[i]}, {mean_mstar[i]:.5f}, {nhalos_list[i]}, {chi2[i]:.3f}' if data != None else f'{var}, {ims[i]}, {slopes[i]}, {nhalos_list[i]}, {mle_log_likelihood[i]}')
            else:
                pb.plot(ell_namaster, auto_spectra_list[i] *ell_namaster, 
                                linestyle='dashed', alpha=0.8, 
                                label=f'{var}, {mean_mstar[i]:.5f}, {nhalos_list[i]}, {chi2[i]:.3f}' if data != None else f'{var}, {nhalos_list[i]}, {mle_log_likelihood[i]}')
            if covariance:
                pb.fill_between(x=ell_namaster, 
                                y1=(auto_spectra_list[i] + auto_spectra_covariance_list[i]) *ell_namaster, y2=(auto_spectra_list[i] - auto_spectra_covariance_list[i]) *ell_namaster, 
                                linewidth=0, alpha=.3)
        else:
            if multi_sim:
                pb.plot(ell_namaster, auto_spectra_list[i] *ell_namaster, 
                            linestyle='solid', alpha=0.8, 
                            label=f'{var}, {ims[i]}, {slopes[i]}, {mean_mstar[i]:.5f}, {nhalos_list[i]}, {chi2[i]:.3f}' if data != None else f'{var}, {ims[i]}, {slopes[i]}, {nhalos_list[i]}, {mle_log_likelihood[i]}')
            else:
                pb.plot(ell_namaster, auto_spectra_list[i] *ell_namaster, 
                                linestyle= 'dashed' if var > max_val else 'solid', dash_capstyle='round' if var > max_val else 'projecting',  alpha=0.8, 
                                label=f'{var}, {mean_mstar[i]:.5f}, {nhalos_list[i]}, {chi2[i]:.3f}' if data != None else f'{var}, {nhalos_list[i]}, {mle_log_likelihood[i]}')
            if covariance:
                pb.fill_between(x=ell_namaster, 
                                y1=(auto_spectra_list[i] + auto_spectra_covariance_list[i]) *ell_namaster, y2=(auto_spectra_list[i] - auto_spectra_covariance_list[i]) *ell_namaster, 
                                linewidth=0, alpha=.3)
            
    pb.plot(obs_data[:,0], obs_data[:,1]*1e5 *obs_data[:,0], 
            color='k', marker='.', markersize=5, label='ACT x unWISE (Farren et al. 2023)')
    pb.fill_between(x=obs_data[:,0], 
                    y1=(obs_data[:,1]+obs_data_std)*1e5 *obs_data[:,0], y2=(obs_data[:,1]-obs_data_std)*1e5 *obs_data[:,0], 
                    color='k', linewidth=0, alpha=.5)
    pb.plot(Planck_obs_data[:,0], Planck_obs_data[:,1]*1e5 *Planck_obs_data[:,0], 
            color='r', marker='.', markersize=5, label='Planck x unWISE (Farren et al. 2023)')
    pb.fill_between(x=Planck_obs_data[:,0], 
                    y1=(Planck_obs_data[:,1]+Planck_obs_data_std)*1e5 *Planck_obs_data[:,0], y2=(Planck_obs_data[:,1]-Planck_obs_data_std)*1e5 *Planck_obs_data[:,0], 
                    color='r', linewidth=0, alpha=.5)
    pb.xlabel('Multipole moment $\mathrm{\ell}$')
    pb.ylabel('$\mathrm{\ell} \\times C^{gg}_{\mathrm{\ell}}x10^5$')
    if single or multi_slope:
        pb.title("\n".join(textwrap.wrap(
            rf"Power Spectrum of the Galaxy Overdensity map (shot-noise subtracted) "
            rf"(box={box}, sim={isim}, {iz} sample, "
            rf"log$M_*$={im}, primary CMB={fits})",
            width=80)))
    elif multi_im:
        pb.title("\n".join(textwrap.wrap(
            rf"Power Spectrum of the Galaxy Overdensity map (shot-noise subtracted) "
            rf"(box={box}, sim={isim}, {iz} sample, "
            rf"slope={slope}, primary CMB={fits})",
            width=80)))
    elif multi_sim:
        if paper_ready:
            pass
        else:
            pb.title("\n".join(textwrap.wrap(
                rf"Power Spectrum of the Galaxy Overdensity map (shot-noise subtracted) "
                rf"(box={box}, {iz} sample, log$M_*$ and slope MLE optimised, "
                rf"primary CMB={fits})",
                width=80)))
    pb.xscale("log")
    pb.yscale("log")
    pb.xlim(200, 4000)#ell_namaster[-1])
    if single or multi_slope:
        pb.legend(title="Slope, $N_{halo}$, $-\log \mathcal{L}(\\theta \mid x)$" if data == None else "Slope, $log_{10}M_{*,mean}$, $N_{halo}$, $\chi^2$", 
                  fontsize=6 if max(variable_list) > max_val else 9, ncols=3 if max(variable_list) > max_val else 2, loc='upper left')
        pb.savefig(f'./Plots/halo_map_gg_power_spectrum_{box}_{isim}_{iz}_{im_name}_all_slopes_{fits}_ntotal_shot_noise_subtracted{template_prefix}_ell.png', dpi=400)
    elif multi_im:
        pb.legend(title="Mass cut, $N_{halo}$, $-\log \mathcal{L}(\\theta \mid x)$" if data == None else "Mass cut, $log_{10}M_{*,mean}$, $N_{halo}$, $\chi^2$", 
                  fontsize=9, ncols=2, loc='upper left')
        pb.savefig(f'./Plots/halo_map_gg_power_spectrum_{box}_{isim}_{iz}_{slope_name}_all_mass_cuts_{fits}_ntotal_shot_noise_subtracted{template_prefix}_ell.png', dpi=400)
    elif multi_sim:
        pb.legend(title="Simulation, Amp, Slope, $N_{halo}$, $-\log \mathcal{L}(\\theta \mid x)$" if data == None else "Simulation, Amp, Slope, $log_{10}M_{*,mean}$, $N_{halo}$, $\chi^2$", 
                  fontsize=6, ncols=1, loc='upper left')
        pb.savefig(f'./Plots/halo_map_gg_power_spectrum_{box}_all_sims_{plot_type_extension}{iz}_mle_amp_slope_{fits}_ntotal_shot_noise_subtracted{template_prefix}_ell.png', dpi=400)
    pb.clf()

    # galaxy-CMB lensing cross-power spectra

    if multi_sim:
        fig, (ax, ax_ratio) = pb.subplots(
            2, 1, figsize=(8, 7.2),
            sharex=True,
            gridspec_kw={"height_ratios": [3.0, 1.0], "hspace": 0.05},
            )
    else:
        fig, ax = pb.subplots(1, 1, figsize=(8,6))

    if multi_sim and box == 'L2800N5040':
        l1000_idx = [i for i, b in enumerate(isim_boxes) if b == 'L1000N1800']
        l2800_idx = [i for i, b in enumerate(isim_boxes) if b == 'L2800N5040']

        # L1000 reference line
        i_ref = l1000_idx[0]
        fiducial_spec = np.asarray(cross_spectra_list[i_ref], dtype=float)

        ax.plot(
            ell_namaster,
            fiducial_spec,
            color=FLAMINGO_colors_sorted[i_ref],
            linestyle='solid',
            alpha=0.9,
            label=FLAMINGO_names[i_ref]
        )

        ax_ratio.axhline(1.0, color=FLAMINGO_colors_sorted[i_ref], linestyle='solid', alpha=0.8)

        # L2800 mean + min/max envelope
        l2800_cross = np.asarray([cross_spectra_list[i] for i in l2800_idx], dtype=float)

        l2800_mean = np.mean(l2800_cross, axis=0)
        l2800_min = np.min(l2800_cross, axis=0)
        l2800_max = np.max(l2800_cross, axis=0)

        ax.plot(
            ell_namaster,
            l2800_mean,
            color=FLAMINGO_colors_sorted[l2800_idx[0]],
            linestyle='solid',
            alpha=0.9,
            label=FLAMINGO_names[l2800_idx[0]]
        )

        ax.fill_between(
            ell_namaster,
            l2800_min,
            l2800_max,
            color=FLAMINGO_colors_sorted[l2800_idx[0]],
            alpha=0.25,
            linewidth=0,
            # label='L2p8_m9 lightcone min/max'
        )

        # Ratio panel: mean and min/max relative to L1000 fiducial
        ax_ratio.plot(
            ell_namaster,
            l2800_mean / fiducial_spec,
            color=FLAMINGO_colors_sorted[l2800_idx[0]],
            linestyle='solid',
            alpha=0.9
        )

        ax_ratio.fill_between(
            ell_namaster,
            l2800_min / fiducial_spec,
            l2800_max / fiducial_spec,
            color=FLAMINGO_colors_sorted[l2800_idx[0]],
            alpha=0.25,
            linewidth=0
        )
    else:
        for i, var in enumerate(variable_list):
            print(var)
            if var == min_val:
                var = abs(var)

                if multi_sim:
                    try:
                        fiducial_idx = next(i for i, (b, n) in enumerate(zip(isim_boxes, isim_dirs)) if b == 'L1000N1800' and n == 'HYDRO_FIDUCIAL')
                    except StopIteration:
                        fiducial_idx = 0
                    fiducial_spec = np.asarray(cross_spectra_list[fiducial_idx], dtype=float)

                    if paper_ready and data == None:
                        line, = ax.plot(ell_namaster, cross_spectra_list[i], 
                                        linestyle='dashed', color=FLAMINGO_colors_sorted[i], 
                                        label=f'{var}')

                    else:
                        line, = ax.plot(ell_namaster, cross_spectra_list[i], 
                                        linestyle='dashed', color=FLAMINGO_colors_sorted[i], 
                                        label=f'{var}, {ims[i]}, {slopes[i]}, {mean_mstar[i]:.5f}, {nhalos_list[i]}, {chi2[i]:.3f}' if data != None else f'{var}, {ims[i]}, {slopes[i]}, {nhalos_list[i]}, {mle_log_likelihood[i]}')

                    ratio = cross_spectra_list[i] / fiducial_spec
                    ax_ratio.plot(
                        ell_namaster, ratio,
                        color=line.get_color(),
                        linestyle=line.get_linestyle(),
                        alpha=0.9
                    )
                
                else:
                    ax.plot(ell_namaster, cross_spectra_list[i], 
                            linestyle='dashed', 
                            label=f'{var}, {mean_mstar[i]:.5f}, {nhalos_list[i]}, {chi2[i]:.3f}' if data != None else f'{var}')
                if covariance:
                    ax.fill_between(x=ell_namaster, 
                                    y1=(cross_spectra_list[i] + cross_spectra_covariance_list[i]), y2=(cross_spectra_list[i] - cross_spectra_covariance_list[i]), 
                                    linewidth=0, alpha=.3)
            else:
                if multi_sim:
                    try:
                        fiducial_idx = next(i for i, (b, n) in enumerate(zip(isim_boxes, isim_dirs)) if b == 'L1000N1800' and n == 'HYDRO_FIDUCIAL')
                    except StopIteration:
                        fiducial_idx = 0
                    fiducial_spec = np.asarray(cross_spectra_list[fiducial_idx], dtype=float)
                    
                    if paper_ready and data == None:
                        line, = ax.plot(ell_namaster, cross_spectra_list[i], 
                                        linestyle='solid', color=FLAMINGO_colors_sorted[i], 
                                        label=f'{var}')
                        
                        ax.text(
                            0.8,                      # x position in axes coords
                            0.95 - 0.045*i,            # y position, spaced by loop index
                            f'({float(mle_chi2_cross[i]) - float(mle_chi2_cross[0]):.1f})' if i != 0 else f'{float(mle_chi2_cross[i]):.1f}',
                            transform=ax.transAxes,
                            color=FLAMINGO_colors_sorted[i],
                            fontsize=10,
                            fontfamily='serif',
                            ha='left',
                            va='top'
                        )
                    else:
                        line, = ax.plot(ell_namaster, cross_spectra_list[i], 
                            linestyle='solid', color=FLAMINGO_colors_sorted[i], 
                            label=f'{var}, {ims[i]}, {slopes[i]}, {mean_mstar[i]:.5f}, {nhalos_list[i]}, {chi2[i]:.3f}' if data != None else f'{var}, {ims[i]}, {slopes[i]}, {nhalos_list[i]}, {mle_log_likelihood[i]}')
                        
                    ratio = cross_spectra_list[i] / fiducial_spec
                    ax_ratio.plot(
                        ell_namaster, ratio,
                        color=line.get_color(),
                        linestyle=line.get_linestyle(),
                        alpha=0.9
                    )
                
                else:
                    ax.plot(ell_namaster, cross_spectra_list[i], 
                            linestyle= 'dashed' if var > max_val else 'solid', dash_capstyle='round' if var > max_val else 'projecting',  
                            label=f'{var}, {mean_mstar[i]:.5f}, {nhalos_list[i]}, {chi2[i]:.3f}' if data != None else f'{var}')
                if covariance:
                    ax.fill_between(x=ell_namaster, 
                                    y1=(cross_spectra_list[i] + cross_spectra_covariance_list[i]), y2=(cross_spectra_list[i] - cross_spectra_covariance_list[i]), 
                                    linewidth=0, alpha=.3)
    
    ax.plot(obs_data[:,0], obs_data[:,3]*1e5, 
            color='k', marker='.', markersize=5, linewidth=0, label='ACT x unWISE')
    ax.fill_between(x=obs_data[:,0], 
                    y1=(obs_data[:,3]+obs_data_std_cross)*1e5, y2=(obs_data[:,3]-obs_data_std_cross)*1e5, 
                    color='k', linewidth=0, alpha=0.3)
    
    if multi_sim:
        ratio_act_obs = obs_data[:,3]*1e5 / fiducial_spec
        ax_ratio.plot(
            ell_namaster, ratio_act_obs,
            marker='.', markersize=5,
            color='k', linewidth=0,
            alpha=0.9
        )
        # simple propagation assuming +/- is 1σ on the model only (reference treated fixed)
        ratio_hi = (obs_data[:,3] + obs_data_std_cross) * 1e5 / fiducial_spec
        ratio_lo = (obs_data[:,3] - obs_data_std_cross) * 1e5 / fiducial_spec
        ax_ratio.fill_between(
            x=ell_namaster, y1=ratio_hi, y2=ratio_lo,
            linewidth=0, alpha=0.3, color='k'
        )
    
    ax.plot(Planck_obs_data[:,0], Planck_obs_data[:,3]*1e5, 
            color='r', marker='.', markersize=5, linewidth=0, label='Planck x unWISE')
    ax.fill_between(x=Planck_obs_data[:,0], 
                    y1=(Planck_obs_data[:,3]+Planck_obs_data_std_cross)*1e5, y2=(Planck_obs_data[:,3]-Planck_obs_data_std_cross)*1e5, 
                    color='r', linewidth=0, alpha=.3)
    
    if multi_sim:
        ratio_planck_obs = Planck_obs_data[:,3]*1e5 / fiducial_spec[:len(Planck_obs_data[:,0])]
        ax_ratio.plot(
            Planck_obs_data[:,0], ratio_planck_obs,
            marker='.', markersize=5,
            color='r', linewidth=0,
            alpha=0.9
        )
        # simple propagation assuming +/- is 1σ on the model only (reference treated fixed)
        ratio_hi = (Planck_obs_data[:,3] + Planck_obs_data_std_cross) * 1e5 / fiducial_spec[:len(Planck_obs_data[:,0])]
        ratio_lo = (Planck_obs_data[:,3] - Planck_obs_data_std_cross) * 1e5 / fiducial_spec[:len(Planck_obs_data[:,0])]
        ax_ratio.fill_between(
            x=Planck_obs_data[:,0], y1=ratio_hi, y2=ratio_lo,
            linewidth=0, alpha=0.3, color='r'
        )
    
    ax.set_ylabel('$C^{\kappa g}_{\mathrm{\ell}}x10^5$')
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(200, 4000)
    ax.set_ylim(bottom=1e-4)

    if multi_sim:
        ax_ratio.axhline(y=1.0, color='k', linestyle='dashed', alpha=0.7)
        ax_ratio.set_xlabel('Multipole moment $\mathrm{\ell}$')
        ax_ratio.set_ylabel('Model / Fiducial')
        ax_ratio.set_xscale("log")
        ax_ratio.set_xlim(200, 4000)
        ax_ratio.set_ylim(0.5, 1.5)
        ax_ratio.grid(alpha=0.25)
    else:
        ax.set_xlabel('Multipole moment $\mathrm{\ell}$')
        ax.set_xscale("log")
        ax.set_xlim(200, 4000)

    if paper_ready:
        pass
    else:
        if single or multi_slope:
            ax.set_title("\n".join(textwrap.wrap(
                rf"Cross-Spectra of the Galaxy Overdensity map and CMB lensing map "
                rf"(box={box}, sim={isim}, {iz} sample, "
                rf"log$M_*$={im}, primary CMB={fits})",
                width=80)))
        elif multi_im:
            ax.set_title("\n".join(textwrap.wrap(
                rf"Cross-Spectra of the Galaxy Overdensity map and CMB lensing map "
                rf"(box={box}, sim={isim}, {iz} sample, "
                rf"slope={slope}, primary CMB={fits})",
                width=80)))
        elif multi_sim:
            if paper_ready:
                pass
            else:
                ax.set_title("\n".join(textwrap.wrap(
                    rf"Cross-Spectra of the Galaxy Overdensity map and CMB lensing map "
                    rf"(box={box}, {iz} sample, log$M_*$ and slope MLE optimised, "
                    rf"primary CMB={fits})",
                    width=80)))

    if single or multi_slope:
        if paper_ready:
            ax.legend(title="Slope" if data == None else "Slope, $log_{10}M_{*,mean}$, $N_{halo}$, $\chi^2$", 
                      fontsize=6 if max(variable_list) > max_val else 9, ncols=3 if max(variable_list) > max_val else 2, loc='upper right')
        else:
            ax.legend(title="Slope, $N_{halo}$, $-\log \mathcal{L}(\\theta \mid x)$" if data == None else "Slope, $log_{10}M_{*,mean}$, $N_{halo}$, $\chi^2$", 
                      fontsize=6 if max(variable_list) > max_val else 9, ncols=3 if max(variable_list) > max_val else 2, loc='upper right')
        pb.savefig(f'./Plots/halo_map_kg_power_spectrum_{box}_{isim}_{iz}_{im_name}_all_slopes_{fits}_ntotal{template_prefix}{file_extension}', dpi=400)
    elif multi_im:
        if paper_ready:
            ax.legend(title="Mass cut" if data == None else "Mass cut, $log_{10}M_{*,mean}$, $N_{halo}$, $\chi^2$", 
                      fontsize=9, ncols=2, loc='upper right')
        else:
            ax.legend(title="Mass cut, $N_{halo}$, $-\log \mathcal{L}(\\theta \mid x)$" if data == None else "Mass cut, $log_{10}M_{*,mean}$, $N_{halo}$, $\chi^2$", 
                      fontsize=9, ncols=2, loc='upper right')
        pb.savefig(f'./Plots/halo_map_kg_power_spectrum_{box}_{isim}_{iz}_{slope_name}_all_mass_cuts_{fits}_ntotal{template_prefix}{file_extension}', dpi=400)
    elif multi_sim:
        if paper_ready and data == None:
            ax.legend(title="Simulation", fontsize=9, ncols=1, loc='lower left')
        else:
            ax.legend(title="Simulation, Amp, Slope, $N_{halo}$, $-\log \mathcal{L}(\\theta \mid x)$" if data == None else "Simulation, Amp, Slope, $log_{10}M_{*,mean}$, $N_{halo}$, $\chi^2$", 
                      fontsize=6, ncols=1, loc='upper right')
        pb.savefig(f'./Plots/halo_map_kg_power_spectrum_{box}_all_sims_{plot_type_extension}{iz}_mle_amp_slope_{fits}_ntotal{template_prefix}{file_extension}', dpi=400)
    pb.close(fig)

    # galaxy-CMB lensing cross-power spectra multiplied by ell

    pb.figure(figsize=(8,6))
    for i, var in enumerate(variable_list):
        print(var)
        if var == min_val:
            var = abs(var)

            if multi_sim:
                pb.plot(ell_namaster, cross_spectra_list[i] *ell_namaster, 
                    linestyle='dashed', 
                    label=f'{var}, {ims[i]}, {slopes[i]}, {mean_mstar[i]:.5f}, {nhalos_list[i]}, {chi2[i]:.3f}' if data != None else f'{var}, {ims[i]}, {slopes[i]}, {nhalos_list[i]}, {mle_log_likelihood[i]}')
            else:
                pb.plot(ell_namaster, cross_spectra_list[i] *ell_namaster, 
                        linestyle='dashed', 
                        label=f'{var}, {mean_mstar[i]:.5f}, {nhalos_list[i]}, {chi2[i]:.3f}' if data != None else f'{var}, {nhalos_list[i]}, {mle_log_likelihood[i]}')
            if covariance:
                pb.fill_between(x=ell_namaster, 
                                y1=(cross_spectra_list[i] + cross_spectra_covariance_list[i]) *ell_namaster, y2=(cross_spectra_list[i] - cross_spectra_covariance_list[i]) *ell_namaster, 
                                linewidth=0, alpha=.3)
        else:
            if multi_sim:
                pb.plot(ell_namaster, cross_spectra_list[i] *ell_namaster, 
                    linestyle='solid', 
                    label=f'{var}, {ims[i]}, {slopes[i]}, {mean_mstar[i]:.5f}, {nhalos_list[i]}, {chi2[i]:.3f}' if data != None else f'{var}, {ims[i]}, {slopes[i]}, {nhalos_list[i]}, {mle_log_likelihood[i]}')
            else:
                pb.plot(ell_namaster, cross_spectra_list[i] *ell_namaster, 
                        linestyle='dashed' if var > max_val else 'solid', dash_capstyle='round' if var > max_val else 'projecting', 
                        label=f'{var}, {mean_mstar[i]:.5f}, {nhalos_list[i]}, {chi2[i]:.3f}' if data != None else f'{var}, {nhalos_list[i]}, {mle_log_likelihood[i]}')
            if covariance:
                pb.fill_between(x=ell_namaster, 
                                y1=(cross_spectra_list[i] + cross_spectra_covariance_list[i]) *ell_namaster, y2=(cross_spectra_list[i] - cross_spectra_covariance_list[i]) *ell_namaster, 
                                linewidth=0, alpha=.3)
            
    pb.plot(obs_data[:,0], obs_data[:,3]*1e5 *obs_data[:,0], 
            color='k', marker='.', markersize=5, label='ACT x unWISE (Farren et al. 2023)')
    pb.fill_between(x=obs_data[:,0], 
                    y1=(obs_data[:,3]+obs_data_std_cross)*1e5 *obs_data[:,0], y2=(obs_data[:,3]-obs_data_std_cross)*1e5 *obs_data[:,0], 
                    color='k', linewidth=0, alpha=.5)
    pb.plot(Planck_obs_data[:,0], Planck_obs_data[:,3]*1e5 *Planck_obs_data[:,0], 
            color='r', marker='.', markersize=5, label='Planck x unWISE (Farren et al. 2023)')
    pb.fill_between(x=Planck_obs_data[:,0], 
                    y1=(Planck_obs_data[:,3]+Planck_obs_data_std_cross)*1e5 *Planck_obs_data[:,0], y2=(Planck_obs_data[:,3]-Planck_obs_data_std_cross)*1e5 *Planck_obs_data[:,0], 
                    color='r', linewidth=0, alpha=.5)
    pb.xlabel('Multipole moment $\mathrm{\ell}$')
    pb.ylabel('$\mathrm{\ell} \\times C^{\kappa g}_{\mathrm{\ell}}x10^5$')
    if single or multi_slope:
        pb.title("\n".join(textwrap.wrap(
            rf"Cross-Spectra of the Galaxy Overdensity map and CMB lensing map "
            rf"(box={box}, sim={isim}, {iz} sample, "
            rf"log$M_*$={im}, primary CMB={fits})",
            width=80)))
    elif multi_im:
        pb.title("\n".join(textwrap.wrap(
            rf"Cross-Spectra of the Galaxy Overdensity map and CMB lensing map "
            rf"(box={box}, sim={isim}, {iz} sample, "
            rf"slope={slope}, primary CMB={fits})",
            width=80)))
    elif multi_sim:
        if paper_ready:
            pass
        else:
            pb.title("\n".join(textwrap.wrap(
                rf"Cross-Spectra of the Galaxy Overdensity map and CMB lensing map "
                rf"(box={box}, sim={isim}, {iz} sample, log$M_*$ and slope MLE optimised, "
                rf"primary CMB={fits})",
                width=80)))
    pb.xscale("log")
    pb.yscale("log")
    pb.xlim(200, 4000)#ell_namaster[-1])
    pb.ylim(bottom=1e-4)
    if single or multi_slope:
        pb.legend(title="Slope, $N_{halo}$, $-\log \mathcal{L}(\\theta \mid x)$" if data == None else "Slope, $log_{10}M_{*,mean}$, $N_{halo}$, $\chi^2$", 
                  fontsize=6 if max(variable_list) > max_val else 9, ncols=3 if max(variable_list) > max_val else 2, loc='lower left')
        pb.savefig(f'./Plots/halo_map_kg_power_spectrum_{box}_{isim}_{iz}_{im_name}_all_slopes_{fits}_ntotal{template_prefix}_ell.png', dpi=400)
    elif multi_im:
        pb.legend(title="Mass cut, $N_{halo}$, $-\log \mathcal{L}(\\theta \mid x)$" if data == None else "Mass cut, $log_{10}M_{*,mean}$, $N_{halo}$, $\chi^2$", 
                  fontsize=9, ncols=2, loc='lower left')
        pb.savefig(f'./Plots/halo_map_kg_power_spectrum_{box}_{isim}_{iz}_{slope_name}_all_mass_cuts_{fits}_ntotal{template_prefix}_ell.png', dpi=400)
    elif multi_sim:
        pb.legend(title="Simulation, Amp, Slope, $N_{halo}$, $-\log \mathcal{L}(\\theta \mid x)$" if data == None else "Simulation, Amp, Slope, $log_{10}M_{*,mean}$, $N_{halo}$, $\chi^2$", 
                  fontsize=6, ncols=1, loc='lower left')
        pb.savefig(f'./Plots/halo_map_kg_power_spectrum_{box}_all_sims_{plot_type_extension}{iz}_mle_amp_slope_{fits}_ntotal{template_prefix}_ell.png', dpi=400)
    pb.clf()

    return


if __name__ == "__main__":

    box = str(sys.argv[1])
    isim = str(sys.argv[2])
    iz = str(sys.argv[3])
    im = float(sys.argv[4])
    slope = float(sys.argv[5])
    fits = str(sys.argv[6])

    single = sys.argv[7].lower() in ("true", "1", "yes", "y")

    lc = int(sys.argv[8])

    power_spectra_plot(box, isim, iz, im, slope, fits, single, paper_ready=True, template=False, multi_sim=True, plot_type=None, mle=True, shot_noise=False, lc=lc, file='png')