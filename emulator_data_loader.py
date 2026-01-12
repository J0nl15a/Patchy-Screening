import numpy as np
from pathlib import Path

def data_loader(box, isim, iz, low_halo_threshold=True, negative_power_threshold=False, shot_noise_included=False, lightcone=0):
    
    cut_amplitude = np.repeat(np.arange(10.3, 11.4, 0.1).reshape(-1,1), 11, axis=0)
    cut_slope = np.tile(np.arange(0.0, 1.1, 0.1), 11).reshape(-1,1)
    x_train = np.column_stack((np.round(cut_amplitude, 1), np.round(cut_slope, 1))) #Amplitude and slope parameters

    initial_cross_spectrum = np.loadtxt(f'/cosma8/data/dp004/dc-conl1/FLAMINGO/patchy_screening/data_files/power_spectra/kappa_galaxy/{box}/{isim}/{iz}/lightcone{lightcone}/kappa_galaxy_power_spectrum_10p8_0p5.txt', 
                                        delimiter=' ', skiprows=1, usecols=(0,1))
    initial_auto_spectrum = np.loadtxt(f'/cosma8/data/dp004/dc-conl1/FLAMINGO/patchy_screening/data_files/power_spectra/galaxy_galaxy/{box}/{isim}/{iz}/lightcone{lightcone}/galaxy_galaxy_power_spectrum_10p8_0p5.txt', 
                                        delimiter=' ', skiprows=1, usecols=(0,2) if not shot_noise_included else (0,1))

    negative_power = []
    for i, (a, s) in enumerate(x_train):
        amp_name = f"{float(a):.1f}".replace('.', 'p')
        slope_name = f"{float(s):.1f}".replace('.', 'p')
        if i == 0:
            cross_spectrum = np.loadtxt(f'/cosma8/data/dp004/dc-conl1/FLAMINGO/patchy_screening/data_files/power_spectra/kappa_galaxy/{box}/{isim}/{iz}/lightcone{lightcone}/kappa_galaxy_power_spectrum_{amp_name}_{slope_name}.txt', 
                                        delimiter=' ', skiprows=1, usecols=(0,1))
            auto_spectrum = np.loadtxt(f'/cosma8/data/dp004/dc-conl1/FLAMINGO/patchy_screening/data_files/power_spectra/galaxy_galaxy/{box}/{isim}/{iz}/lightcone{lightcone}/galaxy_galaxy_power_spectrum_{amp_name}_{slope_name}.txt', 
                                        delimiter=' ', skiprows=1, usecols=(0,2) if not shot_noise_included else (0,1)) 
            y_train_cross = cross_spectrum[:,1].reshape(1, -1) #/ initial_cross_spectrum[:,1]
            y_train_auto = auto_spectrum[:,1].reshape(1, -1) #/ initial_auto_spectrum[:,1]
            if negative_power_threshold:
                negative_mask = np.where(cross_spectrum < 0)[0]
                if len(negative_mask) > 0:
                    negative_power.append(i)
            #std = initial_spectrum[:,2].reshape(1, -1)
        else:
            print(a,s)
            cross_spectrum = np.loadtxt(f'/cosma8/data/dp004/dc-conl1/FLAMINGO/patchy_screening/data_files/power_spectra/kappa_galaxy/{box}/{isim}/{iz}/lightcone{lightcone}/kappa_galaxy_power_spectrum_{amp_name}_{slope_name}.txt', 
                                        delimiter=' ', skiprows=1, usecols=1) 
            auto_spectrum = np.loadtxt(f'/cosma8/data/dp004/dc-conl1/FLAMINGO/patchy_screening/data_files/power_spectra/galaxy_galaxy/{box}/{isim}/{iz}/lightcone{lightcone}/galaxy_galaxy_power_spectrum_{amp_name}_{slope_name}.txt', 
                                        delimiter=' ', skiprows=1, usecols=2 if not shot_noise_included else 1) 
            y_train_cross = np.vstack((y_train_cross, cross_spectrum)) #/initial_cross_spectrum[:,1])) 
            y_train_auto = np.vstack((y_train_auto, auto_spectrum)) #/initial_auto_spectrum[:,1]))
            if negative_power_threshold:
                negative_mask = np.where(cross_spectrum < 0)[0]
                if len(negative_mask) > 0:
                    negative_power.append(i)
            #std = np.vstack((std, np.loadtxt(f'/cosma8/data/dp004/dc-conl1/FLAMINGO/patchy_screening/data_files/power_spectra/{directory_name}/{isim}_{iz}_{amp_name}_{slope_name}.txt', 
            #                                delimiter=' ', skiprows=1, usecols=3 if spectra=='auto' else 2)))
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
    nhalo_lower_bound = kusiak_observed_nbar_full_sky * 0.05  # 50% abundance cut

    low_nhalos = []
    if low_halo_threshold:
        for i, (im, slope) in enumerate(x_train):
            im_name = f"{float(im):.1f}".replace('.', 'p')
            slope_name = f"{float(slope):.1f}".replace('.', 'p')
            with open(f"./data_files/dndz_samples/{box}/{isim}/{iz}/lightcone{lightcone}/dndz_galaxies_sampled_{im_name}_{slope_name}.txt", "r") as f:
                first_line = f.readline().strip()
                number = int(first_line.split(":")[-1])
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

        slope_limit = np.where(y == 1.0)[0][-1]
        plateau_point = low_nhalos_x[slope_limit]

        # plateau_point = min(np.where(low_nhalos_values[:,1] != max(low_nhalos_y))[0]-1)

        # m = (low_nhalos_y[-1] - low_nhalos_y[plateau_point])/(low_nhalos_x[-1] - low_nhalos_x[plateau_point])
        # c = -1 * m * low_nhalos_x[plateau_point] + 1

    elif not low_halo_threshold:
        plateau_point = max(x_train[:,0])
        m = 0.0
        c = 1.0
        
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
    x_train_path = Path(path+'X_training_data.npy')
    y_train_cross_path = Path(path+'Y_training_data_cross.npy')
    y_train_auto_path = Path(path+'Y_training_data_auto.npy')
    prior_path.parent.mkdir(parents=True, exist_ok=True)
    x_train_path.parent.mkdir(parents=True, exist_ok=True)
    y_train_cross_path.parent.mkdir(parents=True, exist_ok=True)
    y_train_auto_path.parent.mkdir(parents=True, exist_ok=True)

    # np.save(f'./gpy_model/training/prior_limits_{isim}_{iz}.npy', np.array([plateau_point, m, c]))
    np.save(prior_path, np.array([plateau_point, c, vertical_limit]))
    np.save(x_train_path, x_train)
    np.save(y_train_cross_path, y_train_cross)
    np.save(y_train_auto_path, y_train_auto)

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