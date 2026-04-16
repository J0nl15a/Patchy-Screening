import numpy as np
from pathlib import Path
from scipy.signal import savgol_filter

def data_loader(spectra, box, isim, iz, low_halo_threshold=True, negative_power_threshold=False, shot_noise_included=False, lightcone=0, abundance_cut=0.05,
                 amp_min=10.3, amp_max=11.3, amp_step=0.1, slope_min=0.0, slope_max=1.0, slope_step=0.1):

    cut_amplitude = np.repeat(np.arange(amp_min, amp_max + amp_step, amp_step).reshape(-1,1), int((slope_max - slope_min) / slope_step) + 1, axis=0)
    cut_slope = np.tile(np.arange(slope_min, slope_max + slope_step, slope_step), int((amp_max - amp_min) / amp_step) + 1).reshape(-1,1)
    x_train = np.column_stack((np.round(cut_amplitude, 1), np.round(cut_slope, 1))) #Amplitude and slope parameters

    w = 5
    p = 2
    amp_min_name = f"{float(amp_min):.1f}".replace('.', 'p')
    slope_min_name = f"{float(slope_min):.1f}".replace('.', 'p')
    ell_bins = np.loadtxt(f'/cosma8/data/dp004/dc-conl1/FLAMINGO/patchy_screening/data_files/power_spectra/kappa_galaxy/{box}/{isim}/{iz}/lightcone{lightcone}/kappa_galaxy_power_spectrum_{amp_min_name}_{slope_min_name}.txt', 
                            delimiter=' ', skiprows=1, usecols=(0))
    ell_1000_mask = np.where(ell_bins > 1000)

    # initial_cross_spectrum = np.loadtxt(f'/cosma8/data/dp004/dc-conl1/FLAMINGO/patchy_screening/data_files/power_spectra/kappa_galaxy/{box}/{isim}/{iz}/lightcone{lightcone}/kappa_galaxy_power_spectrum_10p8_0p5.txt', 
    #                                     delimiter=' ', skiprows=1, usecols=(0,1))
    # initial_auto_spectrum = np.loadtxt(f'/cosma8/data/dp004/dc-conl1/FLAMINGO/patchy_screening/data_files/power_spectra/galaxy_galaxy/{box}/{isim}/{iz}/lightcone{lightcone}/galaxy_galaxy_power_spectrum_10p8_0p5.txt', 
    #                                     delimiter=' ', skiprows=1, usecols=(0,2) if not shot_noise_included else (0,1))

    negative_power = []
    for i, (a, s) in enumerate(x_train):
        amp_name = f"{float(a):.1f}".replace('.', 'p')
        slope_name = f"{float(s):.1f}".replace('.', 'p')
        print(a,s)
        cross_spectrum = np.loadtxt(f'/cosma8/data/dp004/dc-conl1/FLAMINGO/patchy_screening/data_files/power_spectra/kappa_galaxy/{box}/{isim}/{iz}/lightcone{lightcone}/kappa_galaxy_power_spectrum_{amp_name}_{slope_name}.txt', 
                                    delimiter=' ', skiprows=1, usecols=(1))
        cross_spectrum_unsmoothed_component = cross_spectrum[np.where(ell_bins <= 1000)]
        cross_spectrum_smooth_component = savgol_filter(cross_spectrum[ell_1000_mask], window_length=w, polyorder=p)
        cross_spectrum = np.concatenate((cross_spectrum_unsmoothed_component, cross_spectrum_smooth_component))

        auto_spectrum = np.loadtxt(f'/cosma8/data/dp004/dc-conl1/FLAMINGO/patchy_screening/data_files/power_spectra/galaxy_galaxy/{box}/{isim}/{iz}/lightcone{lightcone}/galaxy_galaxy_power_spectrum_{amp_name}_{slope_name}.txt', 
                                    delimiter=' ', skiprows=1, usecols=(2) if not shot_noise_included else (1)) 
        auto_spectrum_unsmoothed_component = auto_spectrum[np.where(ell_bins <= 1000)]
        auto_spectrum_smooth_component = savgol_filter(auto_spectrum[ell_1000_mask], window_length=w, polyorder=p)
        auto_spectrum = np.concatenate((auto_spectrum_unsmoothed_component, auto_spectrum_smooth_component))
        
        with open(f"./data_files/dndz_samples/{box}/{isim}/{iz}/lightcone{lightcone}/dndz_galaxies_sampled_{amp_name}_{slope_name}.txt", "r") as f:
            first_line = f.readline().strip()
            number = float(first_line.split(":")[-1])

        if not np.all(np.isfinite(cross_spectrum)):
            print(f"Non-finite cross spectrum at amp={a}, slope={s}")
        if not np.all(np.isfinite(auto_spectrum)):
            print(f"Non-finite auto spectrum at amp={a}, slope={s}")

        if i==0:
            y_train_cross = cross_spectrum.reshape(1, -1) #/ initial_cross_spectrum[:,1]
            y_train_auto = auto_spectrum.reshape(1, -1) #/ initial_auto_spectrum[:,1]
            y_train_abundance = number
            #std = initial_spectrum[:,2].reshape(1, -1)
        else:
            y_train_cross = np.vstack((y_train_cross, cross_spectrum)) #/ initial_cross_spectrum[:,1]) 
            y_train_auto = np.vstack((y_train_auto, auto_spectrum)) #/ initial_auto_spectrum[:,1])
            y_train_abundance = np.vstack((y_train_abundance, number))
            #std = np.vstack((std, np.loadtxt(f'/cosma8/data/dp004/dc-conl1/FLAMINGO/patchy_screening/data_files/power_spectra/{directory_name}/{isim}_{iz}_{amp_name}_{slope_name}.txt', 
            #                                delimiter=' ', skiprows=1, usecols=3 if spectra=='auto' else 2)))


        if negative_power_threshold:
            negative_mask_auto = np.where(auto_spectrum < 0)[0]
            negative_mask_cross = np.where(cross_spectrum < 0)[0]
            if len(negative_mask_auto) > 0 or len(negative_mask_cross) > 0:
                negative_power.append(i)

        print(y_train_cross.shape)

    if negative_power_threshold:
        x_train = np.delete(x_train, negative_power, 0)
        y_train_cross = np.delete(y_train_cross, negative_power, 0)
        y_train_auto = np.delete(y_train_auto, negative_power, 0)
    print(y_train_cross.shape)

    if iz == 'Blue':
        kusiak_observed_nbar_per_sq_deg = 3409
    elif iz == 'Green':
        kusiak_observed_nbar_per_sq_deg = 1846
    kusiak_observed_nbar_full_sky = kusiak_observed_nbar_per_sq_deg * 41253  # total sq deg in sky
    nhalo_lower_bound = kusiak_observed_nbar_full_sky * abundance_cut  # 95% abundance cut

    low_nhalos = []
    if low_halo_threshold:
        for i, (im, slope) in enumerate(x_train):
            im_name = f"{float(im):.1f}".replace('.', 'p')
            slope_name = f"{float(slope):.1f}".replace('.', 'p')
            with open(f"./data_files/dndz_samples/{box}/{isim}/{iz}/lightcone{lightcone}/dndz_galaxies_sampled_{im_name}_{slope_name}.txt", "r") as f:
                first_line = f.readline().strip()
                number = float(first_line.split(":")[-1])
                if number < nhalo_lower_bound:
                    low_nhalos.append(i)
        x_train = np.delete(x_train, low_nhalos, 0)
        y_train_cross = np.delete(y_train_cross, low_nhalos, 0)
        y_train_auto = np.delete(y_train_auto, low_nhalos, 0)

    print(y_train_cross.shape)
    

    cut_values = np.column_stack((np.round(cut_amplitude, 1), np.round(cut_slope, 1)))

    if low_halo_threshold:
        cut_values_low_nhalo = np.delete(cut_values.copy(), low_nhalos, 0)
        unique_first_col_low_nhalos = np.unique(cut_values_low_nhalo[:,0])

        low_nhalos_x = []
        low_nhalos_y = []

        for val in unique_first_col_low_nhalos:
            subset = cut_values_low_nhalo[cut_values_low_nhalo[:, 0] == val]
            low_nhalos_x.append(val)
            low_nhalos_y.append(np.max(subset[:, 1]))
            print(val)
            print(subset)
            print(low_nhalos_x)
            print(low_nhalos_y)

        low_nhalos_values = np.column_stack((low_nhalos_x, low_nhalos_y))
        amplitude_limit = len(low_nhalos_y) - 1
        vertical_limit = low_nhalos_x[amplitude_limit] #+ 0.1

        def diagonal_limit():
            c = 10.3
            mask = np.array([True]*len(low_nhalos_x))
            while True in mask:
                y = -1 * np.array((low_nhalos_x)) + c
                c += 0.1
                limit_points = np.round(np.column_stack((low_nhalos_x, y)), 1)
                low_set = set(map(tuple, cut_values_low_nhalo))
                mask = np.array([tuple(np.round(pt, 1)) in low_set for pt in limit_points])
                print(mask)
                continue
            c -= 0.2
            y -= 0.1
            return np.round(y, 3), np.round(c, 1)

        y, c = diagonal_limit()

        try:
            slope_limit = np.where(y == slope_max)[0][-1]
        except IndexError:
            slope_limit = 0
        plateau_point = low_nhalos_x[slope_limit]

        # plateau_point = min(np.where(low_nhalos_values[:,1] != max(low_nhalos_y))[0]-1)

        # m = (low_nhalos_y[-1] - low_nhalos_y[plateau_point])/(low_nhalos_x[-1] - low_nhalos_x[plateau_point])
        # c = -1 * m * low_nhalos_x[plateau_point] + 1

    elif not low_halo_threshold:
        plateau_point = max(x_train[:,0])
        c = 1.0
        vertical_limit = max(x_train[:,0])
        
    # cut_values_negative_power = np.delete(cut_values.copy(), negative_power, 0)
    # unique_first_col_negative_power = np.unique(cut_values_negative_power[:,0])

    # negative_power_x = []
    # negative_power_y = []

    # for val in unique_first_col_negative_power:
    #     subset = cut_values_negative_power[cut_values_negative_power[:, 0] == val]
    #     negative_power_x.append(val)
    #     negative_power_y.append(np.max(subset[:, 1]))


    path = f'./gpy_model/{box}/{isim}/{iz}/lightcone{lightcone}/training/'
    prior_path = Path(path+'prior_limits.npy')
    if spectra == 'auto' or spectra == 'cross':
        x_train_path_auto = Path(path+'X_training_data_auto.npy')
        x_train_path_cross = Path(path+'X_training_data_cross.npy')
        y_train_cross_path = Path(path+'Y_training_data_cross.npy')
        y_train_auto_path = Path(path+'Y_training_data_auto.npy')
    elif spectra == 'abundance':
        x_train_path_abundance = Path(path+'X_training_data_abundance.npy')
        y_train_abundance_path = Path(path+'Y_training_data_abundance.npy')

    prior_path.parent.mkdir(parents=True, exist_ok=True)
    if spectra == 'auto' or spectra == 'cross':
        x_train_path_auto.parent.mkdir(parents=True, exist_ok=True)
        x_train_path_cross.parent.mkdir(parents=True, exist_ok=True)
        y_train_cross_path.parent.mkdir(parents=True, exist_ok=True)
        y_train_auto_path.parent.mkdir(parents=True, exist_ok=True)
    elif spectra == 'abundance':
        x_train_path_abundance.parent.mkdir(parents=True, exist_ok=True)
        y_train_abundance_path.parent.mkdir(parents=True, exist_ok=True)

    # np.save(f'./gpy_model/training/prior_limits_{isim}_{iz}.npy', np.array([plateau_point, m, c]))
    print(plateau_point, c, vertical_limit)
    # quit()
    np.save(prior_path, np.array([plateau_point, c, vertical_limit]))
    if spectra == 'auto' or spectra == 'cross':
        np.save(x_train_path_auto, x_train)
        np.save(x_train_path_cross, x_train)
        np.save(y_train_cross_path, y_train_cross)
        np.save(y_train_auto_path, y_train_auto)
    elif spectra == 'abundance':
        np.save(x_train_path_abundance, x_train)
        np.save(y_train_abundance_path, y_train_abundance)

    return

if __name__ == "__main__":
    data_loader('L1000N1800', 'HYDRO_FIDUCIAL', 'Blue', low_halo_threshold=True, negative_power_threshold=False, shot_noise_included=True)

    x_train = np.load(f"./gpy_model/L1000N1800/HYDRO_FIDUCIAL/Blue/lightcone0/training/X_training_data.npy")
    y_train_auto = np.load(f"./gpy_model/L1000N1800/HYDRO_FIDUCIAL/Blue/lightcone0/training/Y_training_data_auto.npy")
    y_train_cross = np.load(f"./gpy_model/L1000N1800/HYDRO_FIDUCIAL/Blue/lightcone0/training/Y_training_data_cross.npy")
    x_train_min = (np.min(x_train[:,0]), np.min(x_train[:,1]))
    x_train_max = (np.max(x_train[:,0]), np.max(x_train[:,1]))
    x_train_normalised = np.column_stack(((x_train[:,0]-x_train_min[0])/(x_train_max[0]-x_train_min[0]), (x_train[:,1]-x_train_min[1])/(x_train_max[1]-x_train_min[1])))

    initial_spectrum = np.loadtxt(f'/cosma8/data/dp004/dc-conl1/FLAMINGO/patchy_screening/data_files/power_spectra/galaxy_galaxy/L1000N1800/HYDRO_FIDUCIAL/Blue/lightcone0/galaxy_galaxy_power_spectrum_10p8_0p5.txt',
                                        delimiter=' ', skiprows=1, usecols=0)

    y_train_auto_normalised = y_train_auto.copy()
    y_train_cross_normalised = y_train_cross.copy()
    mean_y_train_auto = np.mean(y_train_auto_normalised, axis=0)
    y_train_auto_normalised -= mean_y_train_auto
    mean_y_train_cross = np.mean(y_train_cross_normalised, axis=0)
    y_train_cross_normalised -= mean_y_train_cross
    std_y_train_auto = np.std(y_train_auto_normalised, axis=0)
    y_train_auto_normalised /= std_y_train_auto
    std_y_train_cross = np.std(y_train_cross_normalised, axis=0)
    y_train_cross_normalised /= std_y_train_cross

    import pylab as pb

    for i in range(y_train_auto_normalised.shape[0]):
        pb.plot(initial_spectrum, y_train_auto_normalised[i,:], label=f'x_train: {x_train[i,0]}, {x_train[i,1]}')
    pb.xlabel('Multipole l')
    pb.ylabel('Normalised Power Spectrum')
    pb.title('Auto Spectrum Training Data')
    pb.legend()
    pb.savefig('./Plots/emulator_auto_training_data.png', dpi=300)
    pb.clf()

    for i in range(y_train_cross_normalised.shape[0]):
        pb.plot(initial_spectrum, y_train_cross_normalised[i,:], label=f'x_train: {x_train[i,0]}, {x_train[i,1]}')
    pb.xlabel('Multipole l')
    pb.ylabel('Normalised Power Spectrum')
    pb.title('Cross Spectrum Training Data')
    pb.legend()
    pb.savefig('./Plots/emulator_cross_training_data.png', dpi=300)
    pb.clf()

    print(y_train_auto)