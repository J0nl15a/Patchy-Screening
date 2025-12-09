import numpy as np, pylab as pb
import sys, yaml, textwrap
import pymaster as nmt
import scipy.interpolate as interpolate

def power_spectra_plot(isim, iz, im, slope, fits, single, template=False, multi_im = False, multi_slope = False, multi_sim=False, covariance=False, shot_noise=False, data=None):

    if round(im, 1) == im:
        im_name = f"{im:.1f}".replace('.', 'p')
    else:
        im_name = f"{im}".replace('.', 'p')

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
            
    elif not single or not multi_sim:

        if multi_slope:
            slopes = [round(n, 2) for n in np.arange(0.0, float(slope)+0.1, 0.1)] #CHANGE BACK   
            if round(slope, 1) == slope:
                slopes_name = [f"{round(n, 2):.1f}".replace('.', 'p') for n in np.arange(0.0, float(slope)+0.1, 0.1)]
            else:
                slopes_name = [f"{round(n, 2):.3f}".replace('.', 'p') for n in np.arange(0.0, float(slope)+0.1, 0.1)]
            variable_list = slopes
            variable_name_list = slopes_name

        elif multi_im:
            ims = [round(n, 2) for n in np.arange(10.3, float(im)+0.1, 0.1)]
            if round(im, 1) == im:
                ims_name = [f"{round(n, 2):.1f}".replace('.', 'p') for n in np.arange(10.3, float(im)+0.1, 0.1)]
            else:
                ims_name = [f"{round(n, 2):.3f}".replace('.', 'p') for n in np.arange(10.3, float(im)+0.1, 0.1)]
            variable_list = ims
            variable_name_list = ims_name

        if multi_sim:
            isim_names = ['HYDRO_FIDUCIAL', 'HYDRO_LOW_SIGMA8', 'HYDRO_PLANCK']
            ims = []
            slopes = []
            ims_name = []
            slopes_name = []
            # amp_vals = [10.798, 10.756, 10.811]
            # slope_vals = [0.659, 0.744, 0.635]
            for name in isim_names:
                # for i in range(len(isim_names)):
                amp = np.loadtxt(f"./data_files/mle_values_{name}_{iz}.txt", usecols=1, skiprows=1, max_rows=1, delimiter='=')
                slope = np.loadtxt(f"./data_files/mle_values_{name}_{iz}.txt", usecols=1, skiprows=2, max_rows=1, delimiter='=')
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

    l0 = np.ceil(bin_edges[:-1]).astype(int)
    lf = np.floor(bin_edges[1:]).astype(int)
    b = nmt.NmtBin.from_edges(l0, lf)

    ells = b.get_effective_ells()
    ell_200_mask = np.where(ells > 200)
    ell_namaster = ells[ell_200_mask]

    obs_data = np.loadtxt(f'./unWISExLens_lklh/data/v1.0/bandpowers/unWISExACT-DR6_{str(iz).lower()}_baseline_Clgg+Clkk+Clkg.dat', usecols=(0,1,2,3))[ell_200_mask, :].reshape(-1,4)
    Planck_obs_data = np.loadtxt(f'./unWISExLens_lklh/data/v1.0/bandpowers/unWISExPlanck-PR4_{str(iz).lower()}_baseline_Clgg+Clkk+Clkg.dat', usecols=(0,1,2,3))
    # Planck_ell_mask = np.where(Planck_obs_data[:,0] > 200)[0]
    # Planck_obs_data = Planck_obs_data[Planck_ell_mask, :].reshape(-1,4)

    obs_data_covariance = np.loadtxt(f'./unWISExLens_lklh/data/v1.0/covariances/covmat_Clgg+Clkg_unWISExACT-DR6_{str(iz).lower()}_baseline.dat')
    obs_data_variance = np.diag(obs_data_covariance)
    obs_data_std = np.sqrt(obs_data_variance[:int(len(obs_data_variance)/2)])[ell_200_mask]
    obs_data_std_cross = np.sqrt(obs_data_variance[int(len(obs_data_variance)/2):])[ell_200_mask]

    Planck_obs_data_covariance = np.loadtxt(f'./unWISExLens_lklh/data/v1.0/covariances/covmat_Clgg+Clkg_unWISExPlanck-PR4_{str(iz).lower()}_baseline.dat')
    Planck_obs_data_variance = np.diag(Planck_obs_data_covariance)
    Planck_obs_data_std = np.sqrt(Planck_obs_data_variance[:int(len(Planck_obs_data_variance)/2)])
    Planck_obs_data_std_cross = np.sqrt(Planck_obs_data_variance[int(len(Planck_obs_data_variance)/2):])

    if template:
        initial_auto_spectra = np.loadtxt(f'/cosma8/data/dp004/dc-conl1/FLAMINGO/patchy_screening/data_files/power_spectra/galaxy_galaxy/{isim}_{iz}_10p8_0p5.txt', 
                                          delimiter=' ', skiprows=1, usecols=2)
        initial_cross_spectra = np.loadtxt(f'/cosma8/data/dp004/dc-conl1/FLAMINGO/patchy_screening/data_files/power_spectra/kappa_galaxy/{isim}_{iz}_10p8_0p5.txt', 
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

        # for i in range(len(slopes)):
        for i in range(len(variable_list)):

            if single:
                if shot_noise:
                    auto_power_spectra = np.loadtxt(f'./data_files/power_spectra/galaxy_galaxy/{isim}_{iz}_{im_name}_{slope_name}.txt', skiprows=1, usecols=(1,3) if covariance else 1)
                    with open(f"./data_files/dndz_samples/dndz_galaxies_sampled_{isim}_{iz}_{im_name}_{slope_name}.txt", "r") as f:
                        first_line = f.readline().strip()
                        nhalos = int(first_line.split(":")[-1])
                        nhalos_list.append(nhalos)
                        auto_spectra_shot_noise_list.append((4*np.pi)/nhalos)
                elif not shot_noise:
                    auto_power_spectra = np.loadtxt(f'./data_files/power_spectra/galaxy_galaxy/{isim}_{iz}_{im_name}_{slope_name}.txt', skiprows=1, usecols=(2,3) if covariance else 2)
                    with open(f"./data_files/dndz_samples/dndz_galaxies_sampled_{isim}_{iz}_{im_name}_{slope_name}.txt", "r") as f:
                        first_line = f.readline().strip()
                        nhalos = int(first_line.split(":")[-1])
                        nhalos_list.append(nhalos)
                cross_power_spectra = np.loadtxt(f'./data_files/power_spectra/kappa_galaxy/{isim}_{iz}_{im_name}_{slope_name}.txt', skiprows=1, usecols=(1,2) if covariance else 1)

            elif multi_im:
                if shot_noise:
                    auto_power_spectra = np.loadtxt(f'./data_files/power_spectra/galaxy_galaxy/{isim}_{iz}_{ims_name[i]}_{slope_name}.txt', skiprows=1, usecols=(1,3) if covariance else 1)
                    with open(f"./data_files/dndz_samples/dndz_galaxies_sampled_{isim}_{iz}_{ims_name[i]}_{slope_name}.txt", "r") as f:
                        first_line = f.readline().strip()
                        nhalos = int(first_line.split(":")[-1])
                        nhalos_list.append(nhalos)
                        auto_spectra_shot_noise_list.append((4*np.pi)/nhalos)
                elif not shot_noise:
                    auto_power_spectra = np.loadtxt(f'./data_files/power_spectra/galaxy_galaxy/{isim}_{iz}_{ims_name[i]}_{slope_name}.txt', skiprows=1, usecols=(2,3) if covariance else 2)
                    with open(f"./data_files/dndz_samples/dndz_galaxies_sampled_{isim}_{iz}_{ims_name[i]}_{slope_name}.txt", "r") as f:
                        first_line = f.readline().strip()
                        nhalos = int(first_line.split(":")[-1])
                        nhalos_list.append(nhalos)
                cross_power_spectra = np.loadtxt(f'./data_files/power_spectra/kappa_galaxy/{isim}_{iz}_{ims_name[i]}_{slope_name}.txt', skiprows=1, usecols=(1,2) if covariance else 1)
                

            elif multi_slope:
                if shot_noise:
                    auto_power_spectra = np.loadtxt(f'./data_files/power_spectra/galaxy_galaxy/{isim}_{iz}_{im_name}_{slopes_name[i]}.txt', skiprows=1, usecols=(1,3) if covariance else 1)
                    with open(f"./data_files/dndz_samples/dndz_galaxies_sampled_{isim}_{iz}_{im_name}_{slopes_name[i]}.txt", "r") as f:
                        first_line = f.readline().strip()
                        nhalos = int(first_line.split(":")[-1])
                        nhalos_list.append(nhalos)
                        auto_spectra_shot_noise_list.append((4*np.pi)/nhalos)
                elif not shot_noise:
                    auto_power_spectra = np.loadtxt(f'./data_files/power_spectra/galaxy_galaxy/{isim}_{iz}_{im_name}_{slopes_name[i]}.txt', skiprows=1, usecols=(2,3) if covariance else 2)
                    with open(f"./data_files/dndz_samples/dndz_galaxies_sampled_{isim}_{iz}_{im_name}_{slopes_name[i]}.txt", "r") as f:
                        first_line = f.readline().strip()
                        nhalos = int(first_line.split(":")[-1])
                        nhalos_list.append(nhalos)
                cross_power_spectra = np.loadtxt(f'./data_files/power_spectra/kappa_galaxy/{isim}_{iz}_{im_name}_{slopes_name[i]}.txt', skiprows=1, usecols=(1,2) if covariance else 1)
                
            elif multi_sim:
                if shot_noise:
                    auto_power_spectra = np.loadtxt(f'./data_files/power_spectra/galaxy_galaxy/{isim_names[i]}_{iz}_{ims_name[i]}_{slopes_name[i]}.txt', skiprows=1, usecols=(1,3) if covariance else 1)
                    with open(f"./data_files/dndz_samples/dndz_galaxies_sampled_{isim_names[i]}_{iz}_{ims_name[i]}_{slopes_name[i]}.txt", "r") as f:
                        first_line = f.readline().strip()
                        nhalos = int(first_line.split(":")[-1])
                        nhalos_list.append(nhalos)
                        auto_spectra_shot_noise_list.append((4*np.pi)/nhalos)
                elif not shot_noise:
                    auto_power_spectra = np.loadtxt(f'./data_files/power_spectra/galaxy_galaxy/{isim_names[i]}_{iz}_{ims_name[i]}_{slopes_name[i]}.txt', skiprows=1, usecols=(2,3) if covariance else 2)
                    with open(f"./data_files/dndz_samples/dndz_galaxies_sampled_{isim_names[i]}_{iz}_{ims_name[i]}_{slopes_name[i]}.txt", "r") as f:
                        first_line = f.readline().strip()
                        nhalos = int(first_line.split(":")[-1])
                        nhalos_list.append(nhalos)
                cross_power_spectra = np.loadtxt(f'./data_files/power_spectra/kappa_galaxy/{isim_names[i]}_{iz}_{ims_name[i]}_{slopes_name[i]}.txt', skiprows=1, usecols=(1,2) if covariance else 1)
            
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
            isim_names = isim_names[::-1]
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
            mle_value = np.loadtxt(f"./data_files/mle_values_{isim_names[i]}_{iz}.txt", usecols=1, max_rows=1, delimiter='=')
            mle_log_likelihood.append(mle_value)
    else:
        mle_value = np.loadtxt(f"./data_files/mle_values_{isim}_{iz}.txt", usecols=1, max_rows=1, delimiter='=')
        mle_log_likelihood.append(mle_value)

    pb.figure(figsize=(8,6))
    for i, var in enumerate(variable_list):
        print(var)
        if var == 0.0:
            var = abs(var)
            
            if multi_sim:
                line, = pb.plot(ell_namaster, auto_spectra_list[i], 
                            linestyle='dashed', alpha=0.8, 
                            label=f'{var}, {ims[i]}, {slopes[i]}, {mean_mstar[i]:.5f}, {nhalos_list[i]}, {chi2[i]:.3f}' if data != None else f'{var}, {ims[i]}, {slopes[i]}, {nhalos_list[i]}, {mle_log_likelihood[i]:.3f}')
            else:
                line, = pb.plot(ell_namaster, auto_spectra_list[i], 
                                linestyle='dashed', alpha=0.8, 
                                label=f'{var}, {mean_mstar[i]:.5f}, {nhalos_list[i]}, {chi2[i]:.3f}' if data != None else f'{var}, {nhalos_list[i]}, {mle_log_likelihood[i]:.3f}')
            if covariance:
                pb.fill_between(x=ell_namaster, 
                                y1=(auto_spectra_list[i] + auto_spectra_covariance_list[i]), y2=(auto_spectra_list[i] - auto_spectra_covariance_list[i]), 
                                linewidth=0, alpha=.3)
            if shot_noise:
                line_color = line.get_color()
                pb.hlines(y=auto_spectra_shot_noise_list[i]*1e5, xmin=-1, xmax=ell_namaster[-1]+10, linestyle='dashed', color=line_color, alpha=0.3)
        else:
            if multi_sim:
                line, = pb.plot(ell_namaster, auto_spectra_list[i], 
                            linestyle='solid', alpha=0.8, 
                            label=f'{var}, {ims[i]}, {slopes[i]}, {mean_mstar[i]:.5f}, {nhalos_list[i]}, {chi2[i]:.3f}' if data != None else f'{var}, {ims[i]}, {slopes[i]}, {nhalos_list[i]}, {mle_log_likelihood[i]:.3f}')
            else:
                line, = pb.plot(ell_namaster, auto_spectra_list[i], 
                                linestyle='dotted' if var < 0 else 'solid', alpha=0.8, 
                                label=f'{var}, {mean_mstar[i]:.5f}, {nhalos_list[i]}, {chi2[i]:.3f}' if data != None else f'{var}, {nhalos_list[i]}, {mle_log_likelihood[i]:.3f}')
            if covariance:
                pb.fill_between(x=ell_namaster, 
                                y1=(auto_spectra_list[i] + auto_spectra_covariance_list[i]), y2=(auto_spectra_list[i] - auto_spectra_covariance_list[i]), 
                                linewidth=0, alpha=.3)
            if shot_noise:
                line_color = line.get_color()
                pb.hlines(y=auto_spectra_shot_noise_list[i]*1e5, xmin=-1, xmax=ell_namaster[-1]+10, linestyle='dotted' if var < 0 else 'solid', color=line_color, alpha=0.3)

    pb.plot(obs_data[:,0], obs_data[:,1]*1e5, 
            color='k', marker='.', markersize=5, label='ACT x unWISE (Farren et al. 2023)')
    if covariance:
        pb.fill_between(x=obs_data[:,0], 
                        y1=(obs_data[:,1]+obs_data_std)*1e5, y2=(obs_data[:,1]-obs_data_std)*1e5, 
                        color='k', linewidth=0, alpha=.5)
    pb.plot(Planck_obs_data[:,0], Planck_obs_data[:,1]*1e5, 
            color='r', marker='.', markersize=5, label='Planck x unWISE (Farren et al. 2023)')
    if covariance:
        pb.fill_between(x=Planck_obs_data[:,0], 
                        y1=(Planck_obs_data[:,1]+Planck_obs_data_std)*1e5, y2=(Planck_obs_data[:,1]-Planck_obs_data_std)*1e5, 
                        color='r', linewidth=0, alpha=.5)
    pb.xlabel('Multipole moment $\ell$')
    pb.ylabel('$C^{gg}_{\ell}x10^5$')
    if single or multi_slope:
        if shot_noise:
            pb.title("\n".join(textwrap.wrap(
                rf"Power Spectrum of the Galaxy Overdensity map (with shot-noise) "
                rf"(sim={isim}, {iz} sample, log$M_*$={im}, "
                rf"primary CMB={fits})",
                width=80)))
        else:
            pb.title("\n".join(textwrap.wrap(
                rf"Power Spectrum of the Galaxy Overdensity map (shot-noise subtracted) "
                rf"(sim={isim}, {iz} sample, log$M_*$={im}, "
                rf"primary CMB={fits})",
                width=80)))
    elif multi_im:
        if shot_noise:
            pb.title("\n".join(textwrap.wrap(
                rf"Power Spectrum of the Galaxy Overdensity map (with shot-noise) "
                rf"(sim={isim}, {iz} sample, slope={slope}, "
                rf"primary CMB={fits})",
                width=80)))
        else:
            pb.title("\n".join(textwrap.wrap(
                rf"Power Spectrum of the Galaxy Overdensity map (shot-noise subtracted) "
                rf"(sim={isim}, {iz} sample, slope={slope}, "
                rf"primary CMB={fits})",
                width=80)))
    elif multi_sim:
        if shot_noise:
            pb.title("\n".join(textwrap.wrap(
                rf"Power Spectrum of the Galaxy Overdensity map (with shot-noise) "
                rf"({iz} sample, log$M_*$ and slope MLE optimised, "
                rf"primary CMB={fits})",
                width=80)))
        else:
            pb.title("\n".join(textwrap.wrap(
                rf"Power Spectrum of the Galaxy Overdensity map (shot-noise subtracted) "
                rf"({iz} sample, log$M_*$ and slope MLE optimised, "
                rf"primary CMB={fits})",
                width=80)))
    pb.xscale("log")
    pb.yscale("log")
    pb.xlim(200, 4000)#ell_namaster[-1])
    if single:
        pb.legend(title="Slope, $N_{halo}$, $\log \mathcal{L}(\\theta \mid x)$" if data == None else "Slope, $log_{10}M_{*,mean}$, $N_{halo}$, $\chi^2$", fontsize=8, ncols=1, loc='upper right')
        if shot_noise:
            pb.savefig(f'./Plots/halo_map_gg_power_spectrum_{isim}_{iz}_{im_name}_{slope_name}_{fits}_ntotal{template_prefix}.png', dpi=400)
        else:
            pb.savefig(f'./Plots/halo_map_gg_power_spectrum_{isim}_{iz}_{im_name}_{slope_name}_{fits}_ntotal_shot_noise_subtracted{template_prefix}.png', dpi=400)
    elif multi_slope:
        pb.legend(title="Slope, $N_{halo}$, $\log \mathcal{L}(\\theta \mid x)$" if data == None else "Slope, $log_{10}M_{*,mean}$, $N_{halo}$, $\chi^2$", fontsize=6, ncols=2, loc='upper right')
        if shot_noise:
            pb.savefig(f'./Plots/halo_map_gg_power_spectrum_{isim}_{iz}_{im_name}_all_slopes_{fits}_ntotal{template_prefix}.png', dpi=400)
        else:   
            pb.savefig(f'./Plots/halo_map_gg_power_spectrum_{isim}_{iz}_{im_name}_all_slopes_{fits}_ntotal_shot_noise_subtracted{template_prefix}.png', dpi=400)
    elif multi_im:
        pb.legend(title="Mass cut, $N_{halo}$, $\log \mathcal{L}(\\theta \mid x)$" if data == None else "Mass cut, $log_{10}M_{*,mean}$, $N_{halo}$, $\chi^2$", fontsize=6, ncols=2, loc='upper right')
        if shot_noise:
            pb.savefig(f'./Plots/halo_map_gg_power_spectrum_{isim}_{iz}_{slope_name}_all_mass_cuts_{fits}_ntotal{template_prefix}.png', dpi=400)
        else:
            pb.savefig(f'./Plots/halo_map_gg_power_spectrum_{isim}_{iz}_{slope_name}_all_mass_cuts_{fits}_ntotal_shot_noise_subtracted{template_prefix}.png', dpi=400)
    elif multi_sim:
        pb.legend(title="Simulation, Amp, Slope, $N_{halo}$, $\log \mathcal{L}(\\theta \mid x)$" if data == None else "Simulation, Amp, Slope, $log_{10}M_{*,mean}$, $N_{halo}$, $\chi^2$", fontsize=6, ncols=1, loc='upper right')
        if shot_noise:
            pb.savefig(f'./Plots/halo_map_gg_power_spectrum_all_sims_{iz}_mle_amp_slope_{fits}_ntotal{template_prefix}.png', dpi=400)
        else:
            pb.savefig(f'./Plots/halo_map_gg_power_spectrum_all_sims_{iz}_mle_amp_slope_{fits}_ntotal_shot_noise_subtracted{template_prefix}.png', dpi=400)
    pb.clf()

    pb.figure(figsize=(8,6))
    for i, var in enumerate(variable_list):
        print(var)
        if var == 0.0:
            var = abs(var)
            
            if multi_sim:
                pb.plot(ell_namaster, auto_spectra_list[i] *ell_namaster, 
                            linestyle='dashed', alpha=0.8, 
                            label=f'{var}, {ims[i]}, {slopes[i]}, {mean_mstar[i]:.5f}, {nhalos_list[i]}, {chi2[i]:.3f}' if data != None else f'{var}, {ims[i]}, {slopes[i]}, {nhalos_list[i]}, {mle_log_likelihood[i]:.3f}')
            else:
                pb.plot(ell_namaster, auto_spectra_list[i] *ell_namaster, 
                                linestyle='dashed', alpha=0.8, 
                                label=f'{var}, {mean_mstar[i]:.5f}, {nhalos_list[i]}, {chi2[i]:.3f}' if data != None else f'{var}, {nhalos_list[i]}, {mle_log_likelihood[i]:.3f}')
            if covariance:
                pb.fill_between(x=ell_namaster, 
                                y1=(auto_spectra_list[i] + auto_spectra_covariance_list[i]) *ell_namaster, y2=(auto_spectra_list[i] - auto_spectra_covariance_list[i]) *ell_namaster, 
                                linewidth=0, alpha=.3)
        else:
            if multi_sim:
                pb.plot(ell_namaster, auto_spectra_list[i] *ell_namaster, 
                            linestyle='solid', alpha=0.8, 
                            label=f'{var}, {ims[i]}, {slopes[i]}, {mean_mstar[i]:.5f}, {nhalos_list[i]}, {chi2[i]:.3f}' if data != None else f'{var}, {ims[i]}, {slopes[i]}, {nhalos_list[i]}, {mle_log_likelihood[i]:.3f}')
            else:
                pb.plot(ell_namaster, auto_spectra_list[i] *ell_namaster, 
                                linestyle='dotted' if var < 0 else 'solid', alpha=0.8, 
                                label=f'{var}, {mean_mstar[i]:.5f}, {nhalos_list[i]}, {chi2[i]:.3f}' if data != None else f'{var}, {nhalos_list[i]}, {mle_log_likelihood[i]:.3f}')
            if covariance:
                pb.fill_between(x=ell_namaster, 
                                y1=(auto_spectra_list[i] + auto_spectra_covariance_list[i]) *ell_namaster, y2=(auto_spectra_list[i] - auto_spectra_covariance_list[i]) *ell_namaster, 
                                linewidth=0, alpha=.3)
            
    pb.plot(obs_data[:,0], obs_data[:,1]*1e5 *obs_data[:,0], 
            color='k', marker='.', markersize=5, label='ACT x unWISE (Farren et al. 2023)')
    if covariance:
        pb.fill_between(x=obs_data[:,0], 
                        y1=(obs_data[:,1]+obs_data_std)*1e5 *obs_data[:,0], y2=(obs_data[:,1]-obs_data_std)*1e5 *obs_data[:,0], 
                        color='k', linewidth=0, alpha=.5)
    pb.plot(Planck_obs_data[:,0], Planck_obs_data[:,1]*1e5 *Planck_obs_data[:,0], 
            color='r', marker='.', markersize=5, label='Planck x unWISE (Farren et al. 2023)')
    if covariance:
        pb.fill_between(x=Planck_obs_data[:,0], 
                        y1=(Planck_obs_data[:,1]+Planck_obs_data_std)*1e5 *Planck_obs_data[:,0], y2=(Planck_obs_data[:,1]-Planck_obs_data_std)*1e5 *Planck_obs_data[:,0], 
                        color='r', linewidth=0, alpha=.5)
    pb.xlabel('Multipole moment $\ell$')
    pb.ylabel('$\ell \\times C^{gg}_{\ell}x10^5$')
    if single or multi_slope:
        pb.title("\n".join(textwrap.wrap(
            rf"Power Spectrum of the Galaxy Overdensity map (shot-noise subtracted) "
            rf"(sim={isim}, {iz} sample, log$M_*$={im}, "
            rf"primary CMB={fits})",
            width=80)))
    elif multi_im:
        pb.title("\n".join(textwrap.wrap(
            rf"Power Spectrum of the Galaxy Overdensity map (shot-noise subtracted) "
            rf"(sim={isim}, {iz} sample, slope={slope}, "
            rf"primary CMB={fits})",
            width=80)))
    elif multi_sim:
        pb.title("\n".join(textwrap.wrap(
            rf"Power Spectrum of the Galaxy Overdensity map (shot-noise subtracted) "
            rf"({iz} sample, log$M_*$ and slope MLE optimised, "
            rf"primary CMB={fits})",
            width=80)))
    pb.xscale("log")
    pb.yscale("log")
    pb.xlim(200, 4000)#ell_namaster[-1])
    if single or multi_slope:
        pb.legend(title="Slope, $N_{halo}$, $\log \mathcal{L}(\\theta \mid x)$" if data == None else "Slope, $log_{10}M_{*,mean}$, $N_{halo}$, $\chi^2$", fontsize=8, ncols=2, loc='upper right')
        pb.savefig(f'./Plots/halo_map_gg_power_spectrum_{isim}_{iz}_{im_name}_all_slopes_{fits}_ntotal_shot_noise_subtracted{template_prefix}_ell.png', dpi=400)
    elif multi_im:
        pb.legend(title="Mass cut, $N_{halo}$, $\log \mathcal{L}(\\theta \mid x)$" if data == None else "Mass cut, $log_{10}M_{*,mean}$, $N_{halo}$, $\chi^2$", fontsize=6, ncols=2, loc='upper right')
        pb.savefig(f'./Plots/halo_map_gg_power_spectrum_{isim}_{iz}_{slope_name}_all_mass_cuts_{fits}_ntotal_shot_noise_subtracted{template_prefix}_ell.png', dpi=400)
    elif multi_sim:
        pb.legend(title="Simulation, Amp, Slope, $N_{halo}$, $\log \mathcal{L}(\\theta \mid x)$" if data == None else "Simulation, Amp, Slope, $log_{10}M_{*,mean}$, $N_{halo}$, $\chi^2$", fontsize=6, ncols=1, loc='upper right')
        pb.savefig(f'./Plots/halo_map_gg_power_spectrum_all_sims_{iz}_mle_amp_slope_{fits}_ntotal_shot_noise_subtracted{template_prefix}_ell.png', dpi=400)
    pb.clf()

    pb.figure(figsize=(8,6))
    for i, var in enumerate(variable_list):
        print(var)
        if var == 0.0:
            var = abs(var)

            if multi_sim:
                pb.plot(ell_namaster, cross_spectra_list[i], 
                linestyle='dashed', 
                label=f'{var}, {ims[i]}, {slopes[i]}, {mean_mstar[i]:.5f}, {nhalos_list[i]}, {chi2[i]:.3f}' if data != None else f'{var}, {ims[i]}, {slopes[i]}, {nhalos_list[i]}, {mle_log_likelihood[i]:.3f}')
            else:
                pb.plot(ell_namaster, cross_spectra_list[i], 
                        linestyle='dashed', 
                        label=f'{var}, {mean_mstar[i]:.5f}, {nhalos_list[i]}, {chi2[i]:.3f}' if data != None else f'{var}, {nhalos_list[i]}, {mle_log_likelihood[i]:.3f}')
            if covariance:
                pb.fill_between(x=ell_namaster, 
                                y1=(cross_spectra_list[i] + cross_spectra_covariance_list[i]), y2=(cross_spectra_list[i] - cross_spectra_covariance_list[i]), 
                                linewidth=0, alpha=.3)
        else:
            if multi_sim:
                pb.plot(ell_namaster, cross_spectra_list[i], 
                    linestyle='solid', 
                    label=f'{var}, {ims[i]}, {slopes[i]}, {mean_mstar[i]:.5f}, {nhalos_list[i]}, {chi2[i]:.3f}' if data != None else f'{var}, {ims[i]}, {slopes[i]}, {nhalos_list[i]}, {mle_log_likelihood[i]:.3f}')
            else:
                pb.plot(ell_namaster, cross_spectra_list[i], 
                        linestyle='dotted' if var < 0 else 'solid', 
                        label=f'{var}, {mean_mstar[i]:.5f}, {nhalos_list[i]}, {chi2[i]:.3f}' if data != None else f'{var}, {nhalos_list[i]}, {mle_log_likelihood[i]:.3f}')
            if covariance:
                pb.fill_between(x=ell_namaster, 
                                y1=(cross_spectra_list[i] + cross_spectra_covariance_list[i]), y2=(cross_spectra_list[i] - cross_spectra_covariance_list[i]), 
                                linewidth=0, alpha=.3)
    
    pb.plot(obs_data[:,0], obs_data[:,3]*1e5, 
            color='k', marker='.', markersize=5, label='ACT x unWISE (Farren et al. 2023)')
    if covariance:
        pb.fill_between(x=obs_data[:,0], 
                        y1=(obs_data[:,3]+obs_data_std_cross)*1e5, y2=(obs_data[:,3]-obs_data_std_cross)*1e5, 
                        color='k', linewidth=0, alpha=.5)
    pb.plot(Planck_obs_data[:,0], Planck_obs_data[:,3]*1e5, 
            color='r', marker='.', markersize=5, label='Planck x unWISE (Farren et al. 2023)')
    if covariance:
        pb.fill_between(x=Planck_obs_data[:,0], 
                        y1=(Planck_obs_data[:,3]+Planck_obs_data_std_cross)*1e5, y2=(Planck_obs_data[:,3]-Planck_obs_data_std_cross)*1e5, 
                        color='r', linewidth=0, alpha=.5)
    pb.xlabel('Multipole moment $\ell$')
    pb.ylabel('$C^{\kappa g}_{\ell}x10^5$')
    if single or multi_slope:
        pb.title("\n".join(textwrap.wrap(
            rf"Cross-Spectra of the Galaxy Overdensity map and CMB lensing map "
            rf"(sim={isim}, {iz} sample, log$M_*$={im}, "
            rf"primary CMB={fits})",
            width=80)))
    elif multi_im:
        pb.title("\n".join(textwrap.wrap(
            rf"Cross-Spectra of the Galaxy Overdensity map and CMB lensing map "
            rf"(sim={isim}, {iz} sample, slope={slope}, "
            rf"primary CMB={fits})",
            width=80)))
    elif multi_sim:
        pb.title("\n".join(textwrap.wrap(
            rf"Cross-Spectra of the Galaxy Overdensity map and CMB lensing map "
            rf"({iz} sample, log$M_*$ and slope MLE optimised, "
            rf"primary CMB={fits})",
            width=80)))
    pb.xscale("log")
    pb.yscale("log")
    pb.xlim(200, 4000)#ell_namaster[-1])
    if single or multi_slope:
        pb.legend(title="Slope, $N_{halo}$, $\log \mathcal{L}(\\theta \mid x)$" if data == None else "Slope, $log_{10}M_{*,mean}$, $N_{halo}$, $\chi^2$", fontsize=8, ncols=2, loc='upper right')
        pb.savefig(f'./Plots/halo_map_kg_power_spectrum_{isim}_{iz}_{im_name}_all_slopes_{fits}_ntotal{template_prefix}.png', dpi=400)
    elif multi_im:
        pb.legend(title="Mass cut, $N_{halo}$, $\log \mathcal{L}(\\theta \mid x)$" if data == None else "Mass cut, $log_{10}M_{*,mean}$, $N_{halo}$, $\chi^2$", fontsize=6, ncols=2, loc='upper right')
        pb.savefig(f'./Plots/halo_map_kg_power_spectrum_{isim}_{iz}_{slope_name}_all_mass_cuts_{fits}_ntotal{template_prefix}.png', dpi=400)
    elif multi_sim:
        pb.legend(title="Simulation, Amp, Slope, $N_{halo}$, $\log \mathcal{L}(\\theta \mid x)$" if data == None else "Simulation, Amp, Slope, $log_{10}M_{*,mean}$, $N_{halo}$, $\chi^2$", fontsize=6, ncols=1, loc='upper right')
        pb.savefig(f'./Plots/halo_map_kg_power_spectrum_all_sims_{iz}_mle_amp_slope_{fits}_ntotal{template_prefix}.png', dpi=400)
    pb.clf()

    pb.figure(figsize=(8,6))
    for i, var in enumerate(variable_list):
        print(var)
        if var == 0.0:
            var = abs(var)

            if multi_sim:
                pb.plot(ell_namaster, cross_spectra_list[i] *ell_namaster, 
                    linestyle='dashed', 
                    label=f'{var}, {ims[i]}, {slopes[i]}, {mean_mstar[i]:.5f}, {nhalos_list[i]}, {chi2[i]:.3f}' if data != None else f'{var}, {ims[i]}, {slopes[i]}, {nhalos_list[i]}, {mle_log_likelihood[i]:.3f}')
            else:
                pb.plot(ell_namaster, cross_spectra_list[i] *ell_namaster, 
                        linestyle='dashed', 
                        label=f'{var}, {mean_mstar[i]:.5f}, {nhalos_list[i]}, {chi2[i]:.3f}' if data != None else f'{var}, {nhalos_list[i]}, {mle_log_likelihood[i]:.3f}')
            if covariance:
                pb.fill_between(x=ell_namaster, 
                                y1=(cross_spectra_list[i] + cross_spectra_covariance_list[i]) *ell_namaster, y2=(cross_spectra_list[i] - cross_spectra_covariance_list[i]) *ell_namaster, 
                                linewidth=0, alpha=.3)
        else:
            if multi_sim:
                pb.plot(ell_namaster, cross_spectra_list[i] *ell_namaster, 
                    linestyle='solid', 
                    label=f'{var}, {ims[i]}, {slopes[i]}, {mean_mstar[i]:.5f}, {nhalos_list[i]}, {chi2[i]:.3f}' if data != None else f'{var}, {ims[i]}, {slopes[i]}, {nhalos_list[i]}, {mle_log_likelihood[i]:.3f}')
            else:
                pb.plot(ell_namaster, cross_spectra_list[i] *ell_namaster, 
                        linestyle='dotted' if var < 0 else 'solid', 
                        label=f'{var}, {mean_mstar[i]:.5f}, {nhalos_list[i]}, {chi2[i]:.3f}' if data != None else f'{var}, {nhalos_list[i]}, {mle_log_likelihood[i]:.3f}')
            if covariance:
                pb.fill_between(x=ell_namaster, 
                                y1=(cross_spectra_list[i] + cross_spectra_covariance_list[i]) *ell_namaster, y2=(cross_spectra_list[i] - cross_spectra_covariance_list[i]) *ell_namaster, 
                                linewidth=0, alpha=.3)
            
    pb.plot(obs_data[:,0], obs_data[:,3]*1e5 *obs_data[:,0], 
            color='k', marker='.', markersize=5, label='ACT x unWISE (Farren et al. 2023)')
    if covariance:
        pb.fill_between(x=obs_data[:,0], 
                        y1=(obs_data[:,3]+obs_data_std_cross)*1e5 *obs_data[:,0], y2=(obs_data[:,3]-obs_data_std_cross)*1e5 *obs_data[:,0], 
                        color='k', linewidth=0, alpha=.5)
    pb.plot(Planck_obs_data[:,0], Planck_obs_data[:,3]*1e5 *Planck_obs_data[:,0], 
            color='r', marker='.', markersize=5, label='Planck x unWISE (Farren et al. 2023)')
    if covariance:
        pb.fill_between(x=Planck_obs_data[:,0], 
                        y1=(Planck_obs_data[:,3]+Planck_obs_data_std_cross)*1e5 *Planck_obs_data[:,0], y2=(Planck_obs_data[:,3]-Planck_obs_data_std_cross)*1e5 *Planck_obs_data[:,0], 
                        color='r', linewidth=0, alpha=.5)
    pb.xlabel('Multipole moment $\ell$')
    pb.ylabel('$\ell \\times C^{\kappa g}_{\ell}x10^5$')
    if single or multi_slope:
        pb.title("\n".join(textwrap.wrap(
            rf"Cross-Spectra of the Galaxy Overdensity map and CMB lensing map "
            rf"(sim={isim}, {iz} sample, log$M_*$={im}, "
            rf"primary CMB={fits})",
            width=80)))
    elif multi_im:
        pb.title("\n".join(textwrap.wrap(
            rf"Cross-Spectra of the Galaxy Overdensity map and CMB lensing map "
            rf"(sim={isim}, {iz} sample, slope={slope}, "
            rf"primary CMB={fits})",
            width=80)))
    elif multi_sim:
        pb.title("\n".join(textwrap.wrap(
            rf"Cross-Spectra of the Galaxy Overdensity map and CMB lensing map "
            rf"({iz} sample, log$M_*$ and slope MLE optimised, "
            rf"primary CMB={fits})",
            width=80)))
    pb.xscale("log")
    pb.yscale("log")
    pb.xlim(200, 4000)#ell_namaster[-1])
    if single or multi_slope:
        pb.legend(title="Slope, $N_{halo}$, $\log \mathcal{L}(\\theta \mid x)$" if data == None else "Slope, $log_{10}M_{*,mean}$, $N_{halo}$, $\chi^2$", fontsize=8, ncols=2, loc='upper right')
        pb.savefig(f'./Plots/halo_map_kg_power_spectrum_{isim}_{iz}_{im_name}_all_slopes_{fits}_ntotal{template_prefix}_ell.png', dpi=400)
    elif multi_im:
        pb.legend(title="Mass cut, $N_{halo}$, $\log \mathcal{L}(\\theta \mid x)$" if data == None else "Mass cut, $log_{10}M_{*,mean}$, $N_{halo}$, $\chi^2$", fontsize=6, ncols=2, loc='upper right')
        pb.savefig(f'./Plots/halo_map_kg_power_spectrum_{isim}_{iz}_{slope_name}_all_mass_cuts_{fits}_ntotal{template_prefix}_ell.png', dpi=400)
    elif multi_sim:
        pb.legend(title="Simulation, Amp, Slope, $N_{halo}$, $\log \mathcal{L}(\\theta \mid x)$" if data == None else "Simulation, Amp, Slope, $log_{10}M_{*,mean}$, $N_{halo}$, $\chi^2$", fontsize=6, ncols=1, loc='upper right')
        pb.savefig(f'./Plots/halo_map_kg_power_spectrum_all_sims_{iz}_mle_amp_slope_{fits}_ntotal{template_prefix}_ell.png', dpi=400)
    pb.clf()

    return


if __name__ == "__main__":

    isim = str(sys.argv[1])
    iz = str(sys.argv[2])
    im = float(sys.argv[3])
    slope = float(sys.argv[4])
    fits = str(sys.argv[5])

    single = sys.argv[6].lower() in ("true", "1", "yes", "y")

    power_spectra_plot(isim, iz, im, slope, fits, single, template=False, multi_sim=True, shot_noise=False)