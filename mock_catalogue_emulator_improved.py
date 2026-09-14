import numpy as np, pylab as pb
import GPy
from emulator_data_loader_improved import data_loader
from pathlib import Path

# import importlib.util
# import sys
# # Absolute or relative path to your script
# script_path = '/cosma8/data/dp004/dc-conl1/FLAMINGO/patchy_screening/emulator_data_loader.py'
# # Module name to give it (can be anything)
# module_name = 'data_loader'
# # Load the module from the file
# spec = importlib.util.spec_from_file_location(module_name, script_path)
# module = importlib.util.module_from_spec(spec)
# sys.modules[module_name] = module
# spec.loader.exec_module(module)  

def emulator(x, spectra, box, isim, iz,
             save=False, load=False, log=True, oos_test=False, retrain=False, lightcone=0, abundance_cut=0.5,
             amp_min=10.3, amp_max=11.3, amp_step=0.1, slope_min=0.0, slope_max=1.0, slope_step=0.1):
    
    try:
        x_train = np.load(f"./gpy_model/{box}/{isim}/{iz}/lightcone{lightcone}/training/X_training_data_{spectra}.npy")
        y_train = np.load(f"./gpy_model/{box}/{isim}/{iz}/lightcone{lightcone}/training/Y_training_data_{spectra}.npy")
    except FileNotFoundError:
        data_loader(spectra, box, isim, iz, lightcone=lightcone, abundance_cut=abundance_cut,
                    amp_min=amp_min, amp_max=amp_max, amp_step=amp_step, slope_min=slope_min, slope_max=slope_max, slope_step=slope_step)
        x_train = np.load(f"./gpy_model/{box}/{isim}/{iz}/lightcone{lightcone}/training/X_training_data_{spectra}.npy")
        y_train = np.load(f"./gpy_model/{box}/{isim}/{iz}/lightcone{lightcone}/training/Y_training_data_{spectra}.npy")
    if retrain:
        data_loader(spectra, box, isim, iz, low_halo_threshold=True, negative_power_threshold=False, shot_noise_included=False, lightcone=lightcone, abundance_cut=abundance_cut,
                    amp_min=amp_min, amp_max=amp_max, amp_step=amp_step, slope_min=slope_min, slope_max=slope_max, slope_step=slope_step)
        x_train = np.load(f"./gpy_model/{box}/{isim}/{iz}/lightcone{lightcone}/training/X_training_data_{spectra}.npy")
        y_train = np.load(f"./gpy_model/{box}/{isim}/{iz}/lightcone{lightcone}/training/Y_training_data_{spectra}.npy")
        print(y_train)

    # if spectra != 'abundance' and spectra != 'cross':
        # y_train = y_train[:, :-3]


    x = np.asarray(x, dtype=float).reshape(-1)

    if x.size != x_train.shape[1]:
        raise ValueError(f"Expected {x_train.shape[1]} input parameters, got {x.size}")

    x_train_full = x_train.copy()
    x_train_min = np.min(x_train_full, axis=0)
    x_train_max = np.max(x_train_full, axis=0)
    x_train_range = x_train_max - x_train_min

    if oos_test:
        held_out = np.where(np.all(np.isclose(x_train, x[None, :], rtol=0.0, atol=1.0e-8), axis=1))[0]

        if held_out.size != 1:
            raise ValueError(f"Expected exactly one training point matching {x}, found {held_out.size}")

        held_out_index = held_out[0]

        print(f"Holding out training point {x_train[held_out_index]}")

        x_train = np.delete(x_train, held_out_index, axis=0)
        y_train = np.delete(y_train, held_out_index, axis=0)

        # A genuine holdout model must be trained without the held-out point.
        load = False
        save = False

    if np.any(x_train_range <= 0):
        raise ValueError("At least one input parameter has zero training range")

    x_train_normalised = ((x_train - x_train_min[None, :]) / x_train_range[None, :])

    if not np.all(np.isfinite(y_train)):
        raise ValueError("y_train contains non-finite values before log10")

    if log and np.any(y_train <= 0):
        bad = np.argwhere(y_train <= 0)
        raise ValueError(f"y_train has non-positive entries; first few: {bad}")

    if log:
        y_model = np.log10(y_train)
    else:
        y_model = y_train.copy()

    if spectra == "abundance":
        kernel = GPy.kern.Matern32(x_train_normalised.shape[1], ARD=True)
    else:
        kernel = GPy.kern.RBF(x_train_normalised.shape[1], ARD=True)

    path = f'./gpy_model/{box}/{isim}/{iz}/lightcone{lightcone}/'
    model_path = Path(path+f'gpy_model_{spectra}.npz')

    if not load:
        model = GPy.models.GPRegression(X=x_train_normalised, Y=y_model, kernel=kernel, normalizer=True, noise_var=1.0e-6)

        if spectra == "abundance":
            num_restarts = 30
            model.Gaussian_noise.variance.constrain_bounded(1.0e-8, 1.0e-3, warning=False)
        elif oos_test:
            num_restarts = 3
            model.Gaussian_noise.variance.fix(1.0e-6)
        else:
            num_restarts = 10
            model.Gaussian_noise.variance.fix(1.0e-6)

        model.optimize_restarts(num_restarts=num_restarts, optimizer="lbfgsb", max_iters=3000, verbose=False, parallel=False)

        if save:
            model_path.parent.mkdir(parents=True, exist_ok=True)

            np.savez_compressed(model_path, parameters=model.param_array, x_min=x_train_min, x_max=x_train_max, log_output=np.asarray(log, dtype=bool))

    elif load:
        saved = np.load(model_path)

        saved_x_min = np.asarray(saved["x_min"], dtype=float)
        saved_x_max = np.asarray(saved["x_max"], dtype=float)

        saved_log = bool(saved["log_output"])

        if saved_log != log:
            raise ValueError(f"Saved emulator uses log={saved_log}, but prediction requested log={log}")

        if not np.allclose(saved_x_min, x_train_min):
            raise ValueError("Saved and current training input minima differ")

        if not np.allclose(saved_x_max, x_train_max):
            raise ValueError("Saved and current training input maxima differ")

        model = GPy.models.GPRegression(X=x_train_normalised, Y=y_model, kernel=kernel, normalizer=True, noise_var=1.0e-6, initialize=False)

        model.update_model(False)
        model.initialize_parameter()
        model[:] = saved["parameters"]
        model.update_model(True)
    
    x_test = ((x[None, :] - x_train_min[None, :]) / (x_train_max - x_train_min)[None, :])

    # Prediction in the units supplied to the model:
    # log10(C_ell) when log=True, otherwise C_ell.
    mean_model, variance_model = model.predict_noiseless(x_test, full_cov=False)

    mean_model = np.asarray(mean_model, dtype=float)[0]
    variance_model = np.asarray(variance_model, dtype=float)

    # GPy may return one latent normalized variance and broadcast it through
    # the output normalizer. Make the final shape explicitly match the spectrum.
    if variance_model.ndim == 2:
        variance_model = variance_model[0]

    variance_model = np.broadcast_to(variance_model, mean_model.shape).copy()
    variance_model = np.clip(variance_model, 0.0, None)

    if log:
        # mean_model and variance_model describe log10(C_ell).
        mu_log10 = mean_model
        var_log10 = variance_model

        ln10 = np.log(10.0)

        # Median in linear C_ell space.
        median_spectrum = 10.0**mu_log10

        # Mean of the implied lognormal distribution in linear C_ell space.
        mean_spectrum = np.exp(ln10 * mu_log10 + 0.5 * ln10**2 * var_log10)

        # Per-ell-bin emulator variance in linear C_ell space.
        variance_spectrum = (np.exp(ln10**2 * var_log10) - 1.0) * np.exp(2.0 * ln10 * mu_log10 + ln10**2 * var_log10)
        variance_spectrum = np.clip(variance_spectrum, 0.0, None)

        if spectra == "abundance":
            prediction = median_spectrum
        else:
            prediction = mean_spectrum

    else:
        prediction = mean_model
        mean_spectrum = mean_model
        median_spectrum = None
        variance_spectrum = variance_model

    std_spectrum = np.sqrt(variance_spectrum)

    return prediction, median_spectrum, variance_spectrum, std_spectrum


if __name__ == '__main__':
    import sys

    box = str(sys.argv[1])
    isim = str(sys.argv[2])
    iz = str(sys.argv[3])
    lightcone = int(sys.argv[5])

    spectra = str(sys.argv[4])
    amp = 10.813
    slope = 0.183
    residual = 0.0

    if iz == 'Blue':
        kusiak_nbar = 3409
    elif iz == 'Green':
        kusiak_nbar = 1846

    kusiak_observed_abundance = kusiak_nbar * 41253

    test = np.array((amp+residual, slope))
    test_name = [f"{float(test[0]):.3f}".replace('.', 'p'), f"{float(test[1]):.3f}".replace('.', 'p')]
    test_name_base = [f"{float(amp):.1f}".replace('.', 'p'), f"{float(slope):.1f}".replace('.', 'p')]
    if amp+residual >= 11.3:
        pass
    else:
        test_name_plus_1 = [f"{float(amp+0.1):.1f}".replace('.', 'p'), f"{float(slope):.1f}".replace('.', 'p')]

    if spectra == 'auto' or spectra == 'abundance':
        dir_type = 'galaxy_galaxy'
    elif spectra == 'cross':
        dir_type = 'kappa_galaxy'

    if amp+residual > 11.3:
        pass
    else:
        true = np.loadtxt(f'/cosma8/data/dp004/dc-conl1/FLAMINGO/patchy_screening/data_files/power_spectra/{dir_type}/{box}/{isim}/{iz}/lightcone{lightcone}/{dir_type}_power_spectrum_{test_name[0]}_{test_name[1]}.txt', 
                          # delimiter=' ', skiprows=1, usecols=(0,1) if spectra=='auto' else (0,1))
                          delimiter=' ', skiprows=1, usecols=(0,2) if spectra=='auto' else (0,1))
    
        with open(f"/cosma8/data/dp004/dc-conl1/FLAMINGO/patchy_screening/data_files/dndz_samples/{box}/{isim}/{iz}/lightcone{lightcone}/dndz_galaxies_sampled_{test_name[0]}_{test_name[1]}.txt", "r") as f:
            first_line = f.readline().strip()
            abundance = int(first_line.split(":")[-1])
            print(f"Abundance for test point {test_name}: {abundance}")
    
        # true_plus_1 = np.loadtxt(f'/cosma8/data/dp004/dc-conl1/FLAMINGO/patchy_screening/data_files/power_spectra/{dir_type}/{box}/{isim}/{iz}/lightcone{lightcone}/{dir_type}_power_spectrum_{test_name_plus_1[0]}_{test_name_plus_1[1]}.txt', 
        #                         # delimiter=' ', skiprows=1, usecols=(0,1) if spectra=='auto' else (0,1))
        #                         delimiter=' ', skiprows=1, usecols=(0,2) if spectra=='auto' else (0,1))

    pred_train, pred_train_median, pred_train_var, pred_train_std = emulator(test, spectra, box, isim, iz, save=True, retrain=True, lightcone=lightcone, 
                    abundance_cut=0.5 if spectra != 'abundance' else 0.0, log=True, slope_max=2.0 if iz == 'Blue' else 1.0)
    pred, pred_median, pred_var, pred_std = emulator(test, spectra, box, isim, iz, load=True, lightcone=lightcone, 
                    abundance_cut=0.5 if spectra != 'abundance' else 0.0, log=True, slope_max=2.0 if iz == 'Blue' else 1.0)
    print(pred_train, abundance)
    print(pred, pred_var, pred_std)
    print(pred - pred_std, pred + pred_std)
    print(pred_std/pred)
    # quit()

    if spectra == 'abundance':

        # Confusion matrix for abundance > 50% of observed
        threshold = 0.5 * kusiak_observed_abundance
        tp = 0  # True Positive: both > threshold
        tn = 0  # True Negative: both <= threshold
        fp = 0  # False Positive: pred > threshold, true <= threshold
        fn = 0  # False Negative: pred <= threshold, true > threshold

        fp_list = []
        fn_list = []
        
        x_train = np.load(f"./gpy_model/{box}/{isim}/{iz}/lightcone{lightcone}/training/X_training_data_{spectra}.npy")
        pred_errors = []

        for i in range(x_train.shape[0]): 

            name_i = [f"{float(x_train[i,0]):.1f}".replace('.', 'p'), f"{float(x_train[i,1]):.1f}".replace('.', 'p')]
            print(x_train[i,:])
            pred_i, pred_i_median, pred_i_var, pred_i_std = emulator(x_train[i,:], spectra, box, isim, iz, oos_test=True, load=True, lightcone=lightcone, 
                              abundance_cut=0.0, log=True, slope_max=2.0 if iz == 'Blue' else 1.0)
            print(pred_i)

            fractional_emulator_error = pred_i_std / pred_i
            print(fractional_emulator_error)

            with open(f"./data_files/dndz_samples/{box}/{isim}/{iz}/lightcone{lightcone}/dndz_galaxies_sampled_{name_i[0]}_{name_i[1]}.txt", "r") as f:
                first_line = f.readline().strip()
                true_i = float(first_line.split(":")[-1])

            true_positive = true_i > threshold
            pred_positive = pred_i > threshold
            
            if true_positive and pred_positive:
                tp += 1
            elif not true_positive and not pred_positive:
                tn += 1
            elif not true_positive and pred_positive:
                fp += 1
                fp_list.append((name_i, pred_i[0], true_i))
            elif true_positive and not pred_positive:
                fn += 1
                fn_list.append((name_i, pred_i[0], true_i))

            error_i = (pred_i)/true_i
            error_i = np.array(error_i)
            print(error_i)
            pred_errors.append(error_i)

        print(pred_errors)
        pred_errors = np.array(pred_errors)
        mean_pred_errors = np.mean(pred_errors, axis=0)
        print(f"Mean Prediction Errors: {mean_pred_errors}")
        std_pred_errors = np.std(pred_errors, axis=0)
        print(f"Standard Deviation of Prediction Errors: {std_pred_errors}")

        print(f"Confusion Matrix for abundance > {threshold}:")
        print(f"True Positives (TP): {tp}")
        print(f"True Negatives (TN): {tn}")
        print(f"False Positives (FP): {fp}")
        print(f"False Negatives (FN): {fn}")
        
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"Accuracy: {accuracy:.3f}")
        print(f"Precision: {precision:.3f}")
        print(f"Recall: {recall:.3f}")
        print(f"F1 Score: {f1:.3f}")

        print("\nFalse Positives:")
        for fp_entry in fp_list:
            print(f"Parameters: {fp_entry[0]}, Predicted Abundance: {fp_entry[1]:.1f}, True Abundance: {fp_entry[2]:.1f}")
        print("\nFalse Negatives:")
        for fn_entry in fn_list:
            print(f"Parameters: {fn_entry[0]}, Predicted Abundance: {fn_entry[1]:.1f}, True Abundance: {fn_entry[2]:.1f}")  
        
        # Plot confusion matrix
        cm = np.array([[tp, fp], [fn, tn]])
        pb.imshow(cm, cmap='Blues', interpolation='nearest')
        pb.colorbar()
        pb.xticks([0, 1], ['Predicted > threshold', 'Predicted <= threshold'])
        pb.yticks([0, 1], ['Actual > threshold', 'Actual <= threshold'])
        for i in range(2):
            for j in range(2):
                pb.text(j, i, cm[i, j], ha='center', va='center', color='black', fontsize=12)
        pb.title(f'Confusion Matrix (threshold = {threshold:.0f})')
        pb.savefig('./Plots/confusion_matrix_abundance.png', dpi=400)
        pb.clf()

    else:

        farren_data = np.loadtxt(f'./unWISExLens_lklh/data/v1.0/bandpowers/unWISExACT-DR6_{str(iz).lower()}_baseline_Clgg+Clkk+Clkg.dat', usecols=(0,1,3)).reshape(-1,3)
        ell_200_mask = np.where(farren_data[:,0] > 200)

        pb.loglog(farren_data[:,0][ell_200_mask], farren_data[:,1][ell_200_mask]*1e5 if spectra=='auto' else farren_data[:,2][ell_200_mask]*1e5, color='k', marker='.', markersize=5, label='ACT x unWISE (Farren et al. 2023)')
        try:
            pb.loglog(true[:,0], true[:,1], label=f'Mock catalog {amp+residual, slope}', color='b')
        except:
            pass
        
        if amp+residual >= 11.3:
            pass
        else:
            pass
            # pb.loglog(true_plus_1[:,0], true_plus_1[:,1], label=f'Mock catalog {amp+0.1, slope}', color='g')
        pb.loglog(farren_data[:,0][ell_200_mask], pred, label='Emulator', color='r')
        # pb.loglog(farren_data[:,0][ell_200_mask], pred_median, label='Emulator (Median)', color='r', linestyle='dashed')
        pb.fill_between(farren_data[:, 0][ell_200_mask], pred - pred_std, pred + pred_std, alpha=0.25, label=r"Emulator $1\sigma$")
        pb.title(f'Mock catalog emulator test (x_test: Amplitude={test[0]:.3f}, Slope={test[1]})')
        pb.xlabel('$\ell$')
        pb.ylabel('$C_{\ell}^{gg}$x10^5' if spectra=='auto' else '$C_{\ell}^{\kappa g}x10^5$')
        pb.legend()
        pb.tight_layout()
        pb.savefig('./Plots/mock_catalog_emulator_test.png', dpi=400)
        pb.clf()

        pb.hlines(y=1.000, xmin=-1, xmax=np.max(true[:,0])*1.1, color='k', linestyles='solid', alpha=0.5, label=None)
        pb.hlines(y=0.990, xmin=-1, xmax=np.max(true[:,0])*1.1, color='k', linestyles='dashed', alpha=0.5, label='1% error')
        pb.hlines(y=1.010, xmin=-1, xmax=np.max(true[:,0])*1.1, color='k', linestyles='dashed', alpha=0.5, label=None)
        pb.hlines(y=0.950, xmin=-1, xmax=np.max(true[:,0])*1.1, color='k', linestyles='dotted', alpha=0.5, label='5% error')
        pb.hlines(y=1.050, xmin=-1, xmax=np.max(true[:,0])*1.1, color='k', linestyles='dotted', alpha=0.5, label=None)
        try:
            pb.plot(true[:,0], pred/true[:,1], label='Error w.r.t. simulated clustering')
            # pb.plot(true[:,0], pred_median/true[:,1], label='Error w.r.t. simulated clustering (Median)', linestyle='dashed')
        except:
            pass
        pb.plot(true[:,0], pred/(farren_data[:,1][ell_200_mask]*1e5) if spectra=='auto' else pred/(farren_data[:,2][ell_200_mask]*1e5), label='Error w.r.t. observed clustering')
        # pb.plot(true[:,0], pred_median/(farren_data[:,1][ell_200_mask]*1e5) if spectra=='auto' else pred_median/(farren_data[:,2][ell_200_mask]*1e5), label='Error w.r.t. observed clustering (Median)', linestyle='dashed')
        try:
            pb.fill_between(true[:, 0], (pred - pred_std)/true[:,1], (pred + pred_std)/true[:,1], alpha=0.25, label=r"Emulator $1\sigma$ w.r.t. simulated clustering")
            # pb.fill_between(true[:, 0], (pred_median - pred_std)/true[:,1], (pred_median + pred_std)/true[:,1], alpha=0.25, label=r"Emulator $1\sigma$ w.r.t. simulated clustering (Median)")
        except:
            pass
        pb.fill_between(farren_data[:, 0][ell_200_mask], (pred - pred_std)/(farren_data[:,1][ell_200_mask]*1e5) if spectra=='auto' else (pred - pred_std)/(farren_data[:,2][ell_200_mask]*1e5), 
                        (pred + pred_std)/(farren_data[:,1][ell_200_mask]*1e5) if spectra=='auto' else (pred + pred_std)/(farren_data[:,2][ell_200_mask]*1e5), alpha=0.25, label=r"Emulator $1\sigma$")
        pb.title(f'Emulator error test (x_test: Amplitude={test[0]}, Slope={test[1]})')
        pb.xlabel('$\ell$')
        pb.ylabel('Residual')
        if spectra == 'cross':
            pb.ylim(top=1.055, bottom=0.945)
        pb.xlim(100, 3000)
        pb.legend()
        pb.tight_layout()
        pb.savefig('./Plots/mock_catalog_emulator_error_test.png', dpi=400)
        pb.clf()
        quit()

        x_train = np.load(f"./gpy_model/{box}/{isim}/{iz}/lightcone{lightcone}/training/X_training_data_{spectra}.npy")
        # x_train = np.loadtxt('./data_files/mock_catalog_test_points.txt')
        pred_errors = []
        pred_errors_median = []
        for i in range(x_train.shape[0]): 
            name_i = [f"{float(x_train[i,0]):.1f}".replace('.', 'p'), f"{float(x_train[i,1]):.1f}".replace('.', 'p')]
            pred_i, pred_i_median, pred_i_var, pred_i_std = emulator(x_train[i,:], spectra, box, isim, iz, oos_test=True, load=True, lightcone=lightcone, log=True,
                                                      abundance_cut=0.5, slope_max=2.0 if iz == 'Blue' else 1.0)

            fractional_emulator_error = pred_i_std / pred_i
            print(fractional_emulator_error)
            fractional_emulator_error_median = pred_i_std / pred_i_median
            print(fractional_emulator_error_median)

            true_i = np.loadtxt(f'/cosma8/data/dp004/dc-conl1/FLAMINGO/patchy_screening/data_files/power_spectra/{dir_type}/{box}/{isim}/{iz}/lightcone{lightcone}/{dir_type}_power_spectrum_{name_i[0]}_{name_i[1]}.txt',
                            # delimiter=' ', skiprows=1, usecols=(0,1) if spectra=='auto' else (0,1))
                            delimiter=' ', skiprows=1, usecols=(0,2) if spectra=='auto' else (0,1))
            error_i = (pred_i)/true_i[:,1]
            error_i = np.array(error_i)
            error_i_median = (pred_i_median)/true_i[:,1]
            error_i_median = np.array(error_i_median)
            print(error_i)
            pred_errors.append(error_i)
            pred_errors_median.append(error_i_median)

        print(pred_errors)
        pred_errors = np.array(pred_errors)
        mean_pred_errors = np.mean(pred_errors, axis=0)
        print(mean_pred_errors.shape)
        std_pred_errors = np.std(pred_errors, axis=0)

        print(pred_errors_median)
        pred_errors_median = np.array(pred_errors_median)
        mean_pred_errors_median = np.mean(pred_errors_median, axis=0)
        print(mean_pred_errors_median.shape)
        std_pred_errors_median = np.std(pred_errors_median, axis=0)

        pb.hlines(y=1.000, xmin=-1, xmax=np.max(true[:,0])*1.1, color='k', linestyles='solid', alpha=0.5, label=None)
        pb.hlines(y=0.990, xmin=-1, xmax=np.max(true[:,0])*1.1, color='k', linestyles='dashed', alpha=0.5, label='1% error')
        pb.hlines(y=1.010, xmin=-1, xmax=np.max(true[:,0])*1.1, color='k', linestyles='dashed', alpha=0.5, label=None)
        pb.hlines(y=0.950, xmin=-1, xmax=np.max(true[:,0])*1.1, color='k', linestyles='dotted', alpha=0.5, label='5% error')
        pb.hlines(y=1.050, xmin=-1, xmax=np.max(true[:,0])*1.1, color='k', linestyles='dotted', alpha=0.5, label=None)
        pb.plot(true[:,0], mean_pred_errors, label='Mean error w.r.t. simulated clustering')
        # pb.plot(true[:,0], mean_pred_errors_median, label='Mean error w.r.t. simulated clustering (Median)', linestyle='dashed')
        pb.fill_between(true[:,0], mean_pred_errors - std_pred_errors, mean_pred_errors + std_pred_errors, color='gray', alpha=0.5, label='1$\sigma$ scatter')
        # pb.fill_between(true[:,0], mean_pred_errors_median - std_pred_errors_median, mean_pred_errors_median + std_pred_errors_median, color='red', alpha=0.25, label='1$\sigma$ scatter (Median)')
        # pb.title(f'Mean emulator error test over training set ({spectra}, {box}, {isim}, {iz})', wrap=True)
        pb.xlabel('Multipole moment $\ell$')
        pb.ylabel('Residual error')
        pb.xlim(100, 3000)
        pb.legend()
        pb.tight_layout()
        pb.savefig(f'./Plots/mock_catalog_emulator_mean_error_test_{spectra}.pdf', dpi=400)
        pb.clf()