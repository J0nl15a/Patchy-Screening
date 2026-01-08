import numpy as np, pylab as pb
import sys, yaml, textwrap
import pymaster as nmt
import scipy.interpolate as interpolate

def power_spectra_plot(box, isim, iz, im, slope, fits, single, paper_ready=False, 
                       template=False, multi_im = False, multi_slope = False, multi_sim=False, covariance=False, shot_noise=False, 
                       data=None):

    if round(im, 1) == im:
        im_name = f"{im:.1f}".replace('.', 'p')
    else:
        im_name = f"{im:.3f}".replace('.', 'p')

    if round(slope, 1) == slope:
        slope_name = f"{slope:.1f}".replace('.', 'p')
    else:
        slope_name = f"{slope:.3f}".replace('.', 'p')

    if single:
        ims = [im]
        slopes = [slope]
        ims_name = [im_name]
        slopes_name = [slope_name]
        isim_names = [isim]
        variable_list = [slope]
        variable_name_list = [slope_name]
            
    elif not single or not multi_sim:

        if multi_slope:
            slopes = [round(n, 2) for n in np.arange(0.0, float(slope)+0.1, 0.1)] #CHANGE BACK   
            if round(slope, 1) == slope:
                slopes_name = [f"{round(n, 2):.1f}".replace('.', 'p') for n in np.arange(0.0, float(slope)+0.1, 0.1)]
            else:
                slopes_name = [f"{round(n, 2)}".replace('.', 'p') for n in np.arange(0.0, float(slope)+0.1, 0.1)]
            variable_list = slopes
            variable_name_list = slopes_name

        elif multi_im:
            ims = [round(n, 2) for n in np.arange(10.3, float(im)+0.1, 0.1)]
            if round(im, 1) == im:
                ims_name = [f"{round(n, 2):.1f}".replace('.', 'p') for n in np.arange(10.3, float(im)+0.1, 0.1)]
            else:
                ims_name = [f"{round(n, 2)}".replace('.', 'p') for n in np.arange(10.3, float(im)+0.1, 0.1)]
            variable_list = ims
            variable_name_list = ims_name

        if multi_sim:
            if box == 'L1000N1800':
                isim_names = ['HYDRO_FIDUCIAL', 'HYDRO_LOW_SIGMA8', 'HYDRO_PLANCK', 'HYDRO_JETS_published', 'HYDRO_STRONG_JETS_published', 'HYDRO_STRONG_SUPERNOVA', 'HYDRO_WEAK_AGN', 'HYDRO_STRONG_AGN', 'HYDRO_STRONGER_AGN', 'HYDRO_STRONGEST_AGN']
                FLAMINGO_names = ['L1_m9', 'LS8', 'Planck', 'Jet', 'Jet_fgas-4$\sigma$', '$M^*$-$\sigma$', 'fgas+2$\sigma$', 'fgas-2$\sigma$', 'fgas-4$\sigma$', 'fgas-8$\sigma$']
                FLAMINGO_colors_sorted = ['#117733', '#882255', '#44AA99', '#7EFF4B', '#55E18E', '#FF8C40', '#abd0e6', '#6aaed6', '#3787c0', '#105ba4']
            elif box == 'L2800N5040':
                isim_names = ['HYDRO_FIDUCIAL']
                FLAMINGO_names = ['L2p8_m9']
                FLAMINGO_colors_sorted = ['#332288']

            ims = []
            slopes = []
            ims_name = []
            slopes_name = []
            # amp_vals = [10.798, 10.756, 10.811]
            # slope_vals = [0.659, 0.744, 0.635]
            for name in isim_names:
                # for i in range(len(isim_names)):
                amp = np.loadtxt(f"./data_files/mle_values_{box}_{name}_{iz}.txt", usecols=1, skiprows=1, max_rows=1, delimiter='=')
                slope = np.loadtxt(f"./data_files/mle_values_{box}_{name}_{iz}.txt", usecols=1, skiprows=2, max_rows=1, delimiter='=')
                # amp = amp_vals[i]
                # slope = slope_vals[i]
                print(amp, slope)
                ims.append(amp)
                slopes.append(slope)
                ims_name.append(f"{amp:.3f}".replace('.', 'p'))
                slopes_name.append(f"{slope:.3f}".replace('.', 'p'))
                print(ims_name, slopes_name)
            variable_list = isim_names
            variable_name_list = isim_names

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
        initial_auto_spectra = np.loadtxt(f'/cosma8/data/dp004/dc-conl1/FLAMINGO/patchy_screening/data_files/power_spectra/galaxy_galaxy/{box}/{isim}/{iz}/galaxy_galaxy_power_spectrum_10p8_0p5.txt', 
                                          delimiter=' ', skiprows=1, usecols=2)
        initial_cross_spectra = np.loadtxt(f'/cosma8/data/dp004/dc-conl1/FLAMINGO/patchy_screening/data_files/power_spectra/kappa_galaxy/{box}/{isim}/{iz}/kappa_galaxy_power_spectrum_10p8_0p5.txt', 
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
                    auto_power_spectra = np.loadtxt(f'./data_files/power_spectra/galaxy_galaxy/{box}/{isim}/{iz}/galaxy_galaxy_power_spectrum_{im_name}_{slope_name}.txt', skiprows=1, usecols=(1,3) if covariance else 1)
                    with open(f"./data_files/dndz_samples/{box}/{isim}/{iz}/dndz_galaxies_sampled_{im_name}_{slope_name}.txt", "r") as f:
                        first_line = f.readline().strip()
                        nhalos = int(first_line.split(":")[-1])
                        nhalos_list.append(nhalos)
                        auto_spectra_shot_noise_list.append((4*np.pi)/nhalos)
                elif not shot_noise:
                    auto_power_spectra = np.loadtxt(f'./data_files/power_spectra/galaxy_galaxy/{box}/{isim}/{iz}/galaxy_galaxy_power_spectrum_{im_name}_{slope_name}.txt', skiprows=1, usecols=(2,3) if covariance else 2)
                    with open(f"./data_files/dndz_samples/{box}/{isim}/{iz}/dndz_galaxies_sampled_{im_name}_{slope_name}.txt", "r") as f:
                        first_line = f.readline().strip()
                        nhalos = int(first_line.split(":")[-1])
                        nhalos_list.append(nhalos)
                cross_power_spectra = np.loadtxt(f'./data_files/power_spectra/kappa_galaxy/{box}/{isim}/{iz}/kappa_galaxy_power_spectrum_{im_name}_{slope_name}.txt', skiprows=1, usecols=(1,2) if covariance else 1)

            elif multi_im:
                if shot_noise:
                    auto_power_spectra = np.loadtxt(f'./data_files/power_spectra/galaxy_galaxy/{box}/{isim}/{iz}/galaxy_galaxy_power_spectrum_{ims_name[i]}_{slope_name}.txt', skiprows=1, usecols=(1,3) if covariance else 1)
                    with open(f"./data_files/dndz_samples/{box}/{isim}/{iz}/dndz_galaxies_sampled_{ims_name[i]}_{slope_name}.txt", "r") as f:
                        first_line = f.readline().strip()
                        nhalos = int(first_line.split(":")[-1])
                        nhalos_list.append(nhalos)
                        auto_spectra_shot_noise_list.append((4*np.pi)/nhalos)
                elif not shot_noise:
                    auto_power_spectra = np.loadtxt(f'./data_files/power_spectra/galaxy_galaxy/{box}/{isim}/{iz}/galaxy_galaxy_power_spectrum_{ims_name[i]}_{slope_name}.txt', skiprows=1, usecols=(2,3) if covariance else 2)
                    with open(f"./data_files/dndz_samples/{box}/{isim}/{iz}/dndz_galaxies_sampled_{ims_name[i]}_{slope_name}.txt", "r") as f:
                        first_line = f.readline().strip()
                        nhalos = int(first_line.split(":")[-1])
                        nhalos_list.append(nhalos)
                cross_power_spectra = np.loadtxt(f'./data_files/power_spectra/kappa_galaxy/{box}/{isim}/{iz}/kappa_galaxy_power_spectrum_{ims_name[i]}_{slope_name}.txt', skiprows=1, usecols=(1,2) if covariance else 1)
                

            elif multi_slope:
                if shot_noise:
                    auto_power_spectra = np.loadtxt(f'./data_files/power_spectra/galaxy_galaxy/{box}/{isim}/{iz}/galaxy_galaxy_power_spectrum_{im_name}_{slopes_name[i]}.txt', skiprows=1, usecols=(1,3) if covariance else 1)
                    with open(f"./data_files/dndz_samples/{box}/{isim}/{iz}/dndz_galaxies_sampled_{im_name}_{slopes_name[i]}.txt", "r") as f:
                        first_line = f.readline().strip()
                        nhalos = int(first_line.split(":")[-1])
                        nhalos_list.append(nhalos)
                        auto_spectra_shot_noise_list.append((4*np.pi)/nhalos)
                elif not shot_noise:
                    auto_power_spectra = np.loadtxt(f'./data_files/power_spectra/galaxy_galaxy/{box}/{isim}/{iz}/galaxy_galaxy_power_spectrum_{im_name}_{slopes_name[i]}.txt', skiprows=1, usecols=(2,3) if covariance else 2)
                    with open(f"./data_files/dndz_samples/{box}/{isim}/{iz}/dndz_galaxies_sampled_{im_name}_{slopes_name[i]}.txt", "r") as f:
                        first_line = f.readline().strip()
                        nhalos = int(first_line.split(":")[-1])
                        nhalos_list.append(nhalos)
                cross_power_spectra = np.loadtxt(f'./data_files/power_spectra/kappa_galaxy/{box}/{isim}/{iz}/kappa_galaxy_power_spectrum_{im_name}_{slopes_name[i]}.txt', skiprows=1, usecols=(1,2) if covariance else 1)
                
            elif multi_sim:
                if shot_noise:
                    auto_power_spectra = np.loadtxt(f'./data_files/power_spectra/galaxy_galaxy/{box}/{isim_names[i]}/{iz}/galaxy_galaxy_power_spectrum_{ims_name[i]}_{slopes_name[i]}.txt', skiprows=1, usecols=(1,3) if covariance else 1)
                    with open(f"./data_files/dndz_samples/{box}/{isim_names[i]}/{iz}/dndz_galaxies_sampled_{ims_name[i]}_{slopes_name[i]}.txt", "r") as f:
                        first_line = f.readline().strip()
                        nhalos = int(first_line.split(":")[-1])
                        nhalos_list.append(nhalos)
                        auto_spectra_shot_noise_list.append((4*np.pi)/nhalos)
                elif not shot_noise:
                    auto_power_spectra = np.loadtxt(f'./data_files/power_spectra/galaxy_galaxy/{box}/{isim_names[i]}/{iz}/galaxy_galaxy_power_spectrum_{ims_name[i]}_{slopes_name[i]}.txt', skiprows=1, usecols=(2,3) if covariance else 2)
                    with open(f"./data_files/dndz_samples/{box}/{isim_names[i]}/{iz}/dndz_galaxies_sampled_{ims_name[i]}_{slopes_name[i]}.txt", "r") as f:
                        first_line = f.readline().strip()
                        nhalos = int(first_line.split(":")[-1])
                        nhalos_list.append(nhalos)
                cross_power_spectra = np.loadtxt(f'./data_files/power_spectra/kappa_galaxy/{box}/{isim_names[i]}/{iz}/kappa_galaxy_power_spectrum_{ims_name[i]}_{slopes_name[i]}.txt', skiprows=1, usecols=(1,2) if covariance else 1)
            
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

    if not single and data == None:
        if multi_slope:
            slopes = slopes[::-1]
        elif multi_im:
            ims = ims[::-1]
        elif multi_sim:
            isim_names = FLAMINGO_names[::-1]
            FLAMINGO_colors_sorted = FLAMINGO_colors_sorted[::-1]
            ims = ims[::-1]
            slopes = slopes[::-1]
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

    mle_log_likelihood = []
    if multi_sim:
        for i in range(len(isim_names)):
            mle_value = np.loadtxt(f"./data_files/mle_values_{box}_{isim_names[i]}_{iz}.txt", usecols=1, max_rows=1, delimiter='=')
            mle_log_likelihood.append(f"{mle_value:.3f}")
    elif single:
        mle_value = np.loadtxt(f"./data_files/mle_values_{box}_{isim}_{iz}.txt", usecols=1, max_rows=1, delimiter='=')
        mle_log_likelihood.append(f"{mle_value:.3f}")
    else:
        mle_log_likelihood = ['' for i in range(len(variable_list))]

    # PLOTTING SECTION

    # galaxy-galaxy auto-power spectra

    if paper_ready and multi_sim:
        fig, (ax, ax_ratio) = pb.subplots(
        2, 1, figsize=(8, 7.2),
        sharex=True,
        gridspec_kw={"height_ratios": [3.0, 1.0], "hspace": 0.05},
        )
    else:
        fig, ax = pb.subplots(1, 1, figsize=(8,6))


    for i, var in enumerate(variable_list):
        print(var)
        if var == 0.0:
            var = abs(var)
            
            if multi_sim:
                fiducial_idx = np.where(np.array(isim_names) == 'HYDRO_FIDUCIAL')[0][0]
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
                                label=f'{var}, {mean_mstar[i]:.5f}, {nhalos_list[i]}, {chi2[i]:.3f}' if data != None else f'{var}, {nhalos_list[i]}, {mle_log_likelihood[i]}')
            if covariance:
                ax.fill_between(x=ell_namaster, 
                                y1=(auto_spectra_list[i] + auto_spectra_covariance_list[i]), y2=(auto_spectra_list[i] - auto_spectra_covariance_list[i]), 
                                linewidth=0, alpha=.3)
            if shot_noise:
                line_color = line.get_color()
                ax.hlines(y=auto_spectra_shot_noise_list[i]*1e5, xmin=-1, xmax=ell_namaster[-1]+10, linestyle='dashed', color=line_color, alpha=0.3)
        else:
            if multi_sim:
                fiducial_idx = np.where(np.array(isim_names) == 'HYDRO_FIDUCIAL')[0][0]
                fiducial_spec = np.asarray(auto_spectra_list[fiducial_idx], dtype=float)

                if paper_ready and data == None:
                    line, = ax.plot(ell_namaster, auto_spectra_list[i], 
                                    linestyle='solid', alpha=0.8, color=FLAMINGO_colors_sorted[i], 
                                    label=f'{var}')
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
                                linestyle='dotted' if var < 0 else 'solid', alpha=0.8, 
                                label=f'{var}, {mean_mstar[i]:.5f}, {nhalos_list[i]}, {chi2[i]:.3f}' if data != None else f'{var}, {nhalos_list[i]}, {mle_log_likelihood[i]}')
            if covariance:
                ax.fill_between(x=ell_namaster, 
                                y1=(auto_spectra_list[i] + auto_spectra_covariance_list[i]), y2=(auto_spectra_list[i] - auto_spectra_covariance_list[i]), 
                                linewidth=0, alpha=.3)
            if shot_noise:
                line_color = line.get_color()
                ax.hlines(y=auto_spectra_shot_noise_list[i]*1e5, xmin=-1, xmax=ell_namaster[-1]+10, linestyle='dotted' if var < 0 else 'solid', color=line_color, alpha=0.3)

    ax.plot(obs_data[:,0], obs_data[:,1]*1e5, 
            color='k', marker='.', markersize=5, linewidth=0, label='ACT x unWISE (Farren et al. 2023)')
    ax.fill_between(x=obs_data[:,0], 
                    y1=(obs_data[:,1]+obs_data_std)*1e5, y2=(obs_data[:,1]-obs_data_std)*1e5, 
                    color='k', linewidth=0, alpha=.3)
    
    if paper_ready and multi_sim:
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
            color='r', marker='.', markersize=5, linewidth=0, label='Planck x unWISE (Farren et al. 2023)')
    ax.fill_between(x=Planck_obs_data[:,0], 
                    y1=(Planck_obs_data[:,1]+Planck_obs_data_std)*1e5, y2=(Planck_obs_data[:,1]-Planck_obs_data_std)*1e5, 
                    color='r', linewidth=0, alpha=.3)
    
    if paper_ready and multi_sim:
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

    ax.set_ylabel('$C^{gg}_{\ell}x10^5$')
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(200, 4000)

    if paper_ready and multi_sim:
        ax_ratio.axhline(y=1.0, color='k', linestyle='dashed', alpha=0.7)
        ax_ratio.set_xlabel('Multipole moment $\ell$')
        ax_ratio.set_ylabel('Model / Fiducial')
        ax_ratio.set_xscale("log")
        ax_ratio.set_xlim(200, 4000)
        ax_ratio.set_ylim(0.9, 1.1)
        ax_ratio.grid(alpha=0.25)

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
        ax.legend(title="Slope, $N_{halo}$, $-\log \mathcal{L}(\\theta \mid x)$" if data == None else "Slope, $log_{10}M_{*,mean}$, $N_{halo}$, $\chi^2$", fontsize=8, ncols=1, loc='upper right')
        if shot_noise:
            pb.savefig(f'./Plots/halo_map_gg_power_spectrum_{isim}_{iz}_{im_name}_{slope_name}_{fits}_ntotal{template_prefix}.png', dpi=400)
        else:
            pb.savefig(f'./Plots/halo_map_gg_power_spectrum_{isim}_{iz}_{im_name}_{slope_name}_{fits}_ntotal_shot_noise_subtracted{template_prefix}.png', dpi=400)
    elif multi_slope:
        ax.legend(title="Slope, $N_{halo}$, $-\log \mathcal{L}(\\theta \mid x)$" if data == None else "Slope, $log_{10}M_{*,mean}$, $N_{halo}$, $\chi^2$", fontsize=6, ncols=2, loc='upper right')
        if shot_noise:
            pb.savefig(f'./Plots/halo_map_gg_power_spectrum_{isim}_{iz}_{im_name}_all_slopes_{fits}_ntotal{template_prefix}.png', dpi=400)
        else:   
            pb.savefig(f'./Plots/halo_map_gg_power_spectrum_{isim}_{iz}_{im_name}_all_slopes_{fits}_ntotal_shot_noise_subtracted{template_prefix}.png', dpi=400)
    elif multi_im:
        ax.legend(title="Mass cut, $N_{halo}$, $-\log \mathcal{L}(\\theta \mid x)$" if data == None else "Mass cut, $log_{10}M_{*,mean}$, $N_{halo}$, $\chi^2$", fontsize=6, ncols=2, loc='upper right')
        if shot_noise:
            pb.savefig(f'./Plots/halo_map_gg_power_spectrum_{isim}_{iz}_{slope_name}_all_mass_cuts_{fits}_ntotal{template_prefix}.png', dpi=400)
        else:
            pb.savefig(f'./Plots/halo_map_gg_power_spectrum_{isim}_{iz}_{slope_name}_all_mass_cuts_{fits}_ntotal_shot_noise_subtracted{template_prefix}.png', dpi=400)
    elif multi_sim:
        if paper_ready and data == None:
            ax.legend(title="Simulation", fontsize=8, ncols=1, loc='upper right')
        else:
            ax.legend(title="Simulation, Amp, Slope, $N_{halo}$, $-\log \mathcal{L}(\\theta \mid x)$" if data == None else "Simulation, Amp, Slope, $log_{10}M_{*,mean}$, $N_{halo}$, $\chi^2$", fontsize=6, ncols=1, loc='upper right')
        if shot_noise:
            pb.savefig(f'./Plots/halo_map_gg_power_spectrum_all_sims_{iz}_mle_amp_slope_{fits}_ntotal{template_prefix}.png', dpi=400)
        else:
            pb.savefig(f'./Plots/halo_map_gg_power_spectrum_all_sims_{iz}_mle_amp_slope_{fits}_ntotal_shot_noise_subtracted{template_prefix}.png', dpi=400)
    # pb.clf()
    pb.close(fig)

    # galaxy-galaxy auto-power spectra multiplied by ell

    pb.figure(figsize=(8,6))
    for i, var in enumerate(variable_list):
        print(var)
        if var == 0.0:
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
                                linestyle='dotted' if var < 0 else 'solid', alpha=0.8, 
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
    pb.xlabel('Multipole moment $\ell$')
    pb.ylabel('$\ell \\times C^{gg}_{\ell}x10^5$')
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
        pb.title("\n".join(textwrap.wrap(
            rf"Power Spectrum of the Galaxy Overdensity map (shot-noise subtracted) "
            rf"(box={box}, {iz} sample, log$M_*$ and slope MLE optimised, "
            rf"primary CMB={fits})",
            width=80)))
    pb.xscale("log")
    pb.yscale("log")
    pb.xlim(200, 4000)#ell_namaster[-1])
    if single or multi_slope:
        pb.legend(title="Slope, $N_{halo}$, $-\log \mathcal{L}(\\theta \mid x)$" if data == None else "Slope, $log_{10}M_{*,mean}$, $N_{halo}$, $\chi^2$", fontsize=8, ncols=2, loc='upper left')
        pb.savefig(f'./Plots/halo_map_gg_power_spectrum_{isim}_{iz}_{im_name}_all_slopes_{fits}_ntotal_shot_noise_subtracted{template_prefix}_ell.png', dpi=400)
    elif multi_im:
        pb.legend(title="Mass cut, $N_{halo}$, $-\log \mathcal{L}(\\theta \mid x)$" if data == None else "Mass cut, $log_{10}M_{*,mean}$, $N_{halo}$, $\chi^2$", fontsize=6, ncols=2, loc='upper left')
        pb.savefig(f'./Plots/halo_map_gg_power_spectrum_{isim}_{iz}_{slope_name}_all_mass_cuts_{fits}_ntotal_shot_noise_subtracted{template_prefix}_ell.png', dpi=400)
    elif multi_sim:
        pb.legend(title="Simulation, Amp, Slope, $N_{halo}$, $-\log \mathcal{L}(\\theta \mid x)$" if data == None else "Simulation, Amp, Slope, $log_{10}M_{*,mean}$, $N_{halo}$, $\chi^2$", fontsize=6, ncols=1, loc='upper left')
        pb.savefig(f'./Plots/halo_map_gg_power_spectrum_all_sims_{iz}_mle_amp_slope_{fits}_ntotal_shot_noise_subtracted{template_prefix}_ell.png', dpi=400)
    pb.clf()

    # galaxy-CMB lensing cross-power spectra

    if paper_ready and multi_sim:
        fig, (ax, ax_ratio) = pb.subplots(
            2, 1, figsize=(8, 7.2),
            sharex=True,
            gridspec_kw={"height_ratios": [3.0, 1.0], "hspace": 0.05},
            )
    else:
        fig, ax = pb.subplots(1, 1, figsize=(8,6))

    for i, var in enumerate(variable_list):
        print(var)
        if var == 0.0:
            var = abs(var)

            if multi_sim:
                fiducial_idx = np.where(np.array(isim_names) == 'HYDRO_FIDUCIAL')[0][0]
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
                        label=f'{var}, {mean_mstar[i]:.5f}, {nhalos_list[i]}, {chi2[i]:.3f}' if data != None else f'{var}, {nhalos_list[i]}, {mle_log_likelihood[i]}')
            if covariance:
                ax.fill_between(x=ell_namaster, 
                                y1=(cross_spectra_list[i] + cross_spectra_covariance_list[i]), y2=(cross_spectra_list[i] - cross_spectra_covariance_list[i]), 
                                linewidth=0, alpha=.3)
        else:
            if multi_sim:
                fiducial_idx = np.where(np.array(isim_names) == 'HYDRO_FIDUCIAL')[0][0]
                fiducial_spec = np.asarray(cross_spectra_list[fiducial_idx], dtype=float)
                
                if paper_ready and data == None:
                    line, = ax.plot(ell_namaster, cross_spectra_list[i], 
                                    linestyle='solid', color=FLAMINGO_colors_sorted[i], 
                                    label=f'{var}')
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
                        linestyle='dotted' if var < 0 else 'solid', 
                        label=f'{var}, {mean_mstar[i]:.5f}, {nhalos_list[i]}, {chi2[i]:.3f}' if data != None else f'{var}, {nhalos_list[i]}, {mle_log_likelihood[i]}')
            if covariance:
                ax.fill_between(x=ell_namaster, 
                                y1=(cross_spectra_list[i] + cross_spectra_covariance_list[i]), y2=(cross_spectra_list[i] - cross_spectra_covariance_list[i]), 
                                linewidth=0, alpha=.3)
    
    ax.plot(obs_data[:,0], obs_data[:,3]*1e5, 
            color='k', marker='.', markersize=5, linewidth=0, label='ACT x unWISE (Farren et al. 2023)')
    ax.fill_between(x=obs_data[:,0], 
                    y1=(obs_data[:,3]+obs_data_std_cross)*1e5, y2=(obs_data[:,3]-obs_data_std_cross)*1e5, 
                    color='k', linewidth=0, alpha=0.3)
    
    if paper_ready and multi_sim:
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
            color='r', marker='.', markersize=5, linewidth=0, label='Planck x unWISE (Farren et al. 2023)')
    ax.fill_between(x=Planck_obs_data[:,0], 
                    y1=(Planck_obs_data[:,3]+Planck_obs_data_std_cross)*1e5, y2=(Planck_obs_data[:,3]-Planck_obs_data_std_cross)*1e5, 
                    color='r', linewidth=0, alpha=.3)
    
    if paper_ready and multi_sim:
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
    
    ax.set_ylabel('$C^{\kappa g}_{\ell}x10^5$')
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(200, 4000)

    if paper_ready and multi_sim:
        ax_ratio.axhline(y=1.0, color='k', linestyle='dashed', alpha=0.7)
        ax_ratio.set_xlabel('Multipole moment $\ell$')
        ax_ratio.set_ylabel('Model / Fiducial')
        ax_ratio.set_xscale("log")
        ax_ratio.set_xlim(200, 4000)
        ax_ratio.set_ylim(0.5, 1.5)
        ax_ratio.grid(alpha=0.25)

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
        ax.set_title("\n".join(textwrap.wrap(
            rf"Cross-Spectra of the Galaxy Overdensity map and CMB lensing map "
            rf"(box={box}, {iz} sample, log$M_*$ and slope MLE optimised, "
            rf"primary CMB={fits})",
            width=80)))

    if single or multi_slope:
        ax.legend(title="Slope, $N_{halo}$, $-\log \mathcal{L}(\\theta \mid x)$" if data == None else "Slope, $log_{10}M_{*,mean}$, $N_{halo}$, $\chi^2$", fontsize=8, ncols=2, loc='upper right')
        pb.savefig(f'./Plots/halo_map_kg_power_spectrum_{isim}_{iz}_{im_name}_all_slopes_{fits}_ntotal{template_prefix}.png', dpi=400)
    elif multi_im:
        ax.legend(title="Mass cut, $N_{halo}$, $-\log \mathcal{L}(\\theta \mid x)$" if data == None else "Mass cut, $log_{10}M_{*,mean}$, $N_{halo}$, $\chi^2$", fontsize=6, ncols=2, loc='upper right')
        pb.savefig(f'./Plots/halo_map_kg_power_spectrum_{isim}_{iz}_{slope_name}_all_mass_cuts_{fits}_ntotal{template_prefix}.png', dpi=400)
    elif multi_sim:
        if paper_ready and data == None:
            ax.legend(title="Simulation", fontsize=8, ncols=1, loc='lower left')
        else:
            ax.legend(title="Simulation, Amp, Slope, $N_{halo}$, $-\log \mathcal{L}(\\theta \mid x)$" if data == None else "Simulation, Amp, Slope, $log_{10}M_{*,mean}$, $N_{halo}$, $\chi^2$", fontsize=6, ncols=1, loc='upper right')
        pb.savefig(f'./Plots/halo_map_kg_power_spectrum_all_sims_{iz}_mle_amp_slope_{fits}_ntotal{template_prefix}.png', dpi=400)
    # pb.clf()
    pb.close(fig)

    # galaxy-CMB lensing cross-power spectra multiplied by ell

    pb.figure(figsize=(8,6))
    for i, var in enumerate(variable_list):
        print(var)
        if var == 0.0:
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
                        linestyle='dotted' if var < 0 else 'solid', 
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
    pb.xlabel('Multipole moment $\ell$')
    pb.ylabel('$\ell \\times C^{\kappa g}_{\ell}x10^5$')
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
        pb.title("\n".join(textwrap.wrap(
            rf"Cross-Spectra of the Galaxy Overdensity map and CMB lensing map "
            rf"(box={box}, sim={isim}, {iz} sample, log$M_*$ and slope MLE optimised, "
            rf"primary CMB={fits})",
            width=80)))
    pb.xscale("log")
    pb.yscale("log")
    pb.xlim(200, 4000)#ell_namaster[-1])
    if single or multi_slope:
        pb.legend(title="Slope, $N_{halo}$, $-\log \mathcal{L}(\\theta \mid x)$" if data == None else "Slope, $log_{10}M_{*,mean}$, $N_{halo}$, $\chi^2$", fontsize=8, ncols=2, loc='lower left')
        pb.savefig(f'./Plots/halo_map_kg_power_spectrum_{isim}_{iz}_{im_name}_all_slopes_{fits}_ntotal{template_prefix}_ell.png', dpi=400)
    elif multi_im:
        pb.legend(title="Mass cut, $N_{halo}$, $-\log \mathcal{L}(\\theta \mid x)$" if data == None else "Mass cut, $log_{10}M_{*,mean}$, $N_{halo}$, $\chi^2$", fontsize=6, ncols=2, loc='lower left')
        pb.savefig(f'./Plots/halo_map_kg_power_spectrum_{isim}_{iz}_{slope_name}_all_mass_cuts_{fits}_ntotal{template_prefix}_ell.png', dpi=400)
    elif multi_sim:
        pb.legend(title="Simulation, Amp, Slope, $N_{halo}$, $-\log \mathcal{L}(\\theta \mid x)$" if data == None else "Simulation, Amp, Slope, $log_{10}M_{*,mean}$, $N_{halo}$, $\chi^2$", fontsize=6, ncols=1, loc='lower left')
        pb.savefig(f'./Plots/halo_map_kg_power_spectrum_all_sims_{iz}_mle_amp_slope_{fits}_ntotal{template_prefix}_ell.png', dpi=400)
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

    power_spectra_plot(box, isim, iz, im, slope, fits, single, paper_ready=True, template=False, multi_slope=True, shot_noise=False)