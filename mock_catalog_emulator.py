import numpy as np, pylab as pb
import GPy
from emulator_data_loader import data_loader
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
             save=False, load=False, log=True, retrain=False, lightcone=0):
    
    try:
        x_train = np.load(f"./gpy_model/{box}/{isim}/{iz}/lightcone{lightcone}/training/X_training_data.npy")
        y_train = np.load(f"./gpy_model/{box}/{isim}/{iz}/lightcone{lightcone}/training/Y_training_data_{spectra}.npy")
    except FileNotFoundError:
        data_loader(box, isim, iz, lightcone=lightcone)
        x_train = np.load(f"./gpy_model/{box}/{isim}/{iz}/lightcone{lightcone}/training/X_training_data.npy")
        y_train = np.load(f"./gpy_model/{box}/{isim}/{iz}/lightcone{lightcone}/training/Y_training_data_{spectra}.npy")
    if retrain:
        data_loader(box, isim, iz, low_halo_threshold=True, negative_power_threshold=False, shot_noise_included=False, lightcone=lightcone)
        x_train = np.load(f"./gpy_model/{box}/{isim}/{iz}/lightcone{lightcone}/training/X_training_data.npy")
        y_train = np.load(f"./gpy_model/{box}/{isim}/{iz}/lightcone{lightcone}/training/Y_training_data_{spectra}.npy")
        print(y_train)
        if spectra == 'auto':
            dir_type = 'galaxy_galaxy'
        elif spectra == 'cross':
            dir_type = 'kappa_galaxy'
        initial_spectra = np.loadtxt(f'/cosma8/data/dp004/dc-conl1/FLAMINGO/patchy_screening/data_files/power_spectra/{dir_type}/{box}/{isim}/{iz}/lightcone{lightcone}/{dir_type}_power_spectrum_10p8_0p5.txt', 
                                    delimiter=' ', skiprows=1, usecols=0)
                                    # delimiter=' ', skiprows=1, usecols=2 if spectra=='auto' else 1)

    x_train_min = (np.min(x_train[:,0]), np.min(x_train[:,1]))
    x_train_max = (np.max(x_train[:,0]), np.max(x_train[:,1]))
    
    x_train_normalised = np.column_stack(((x_train[:,0]-x_train_min[0])/(x_train_max[0]-x_train_min[0]), (x_train[:,1]-x_train_min[1])/(x_train_max[1]-x_train_min[1])))
    

    for i, (a, s) in enumerate(x_train):
        if a == x[0] and s == x[1]:
            print('DELETING!')
            x_train_normalised = np.delete(x_train_normalised, (i), axis=0)
            y_train =  np.delete(y_train, (i), axis=0)

    if log:
        y_train_normalised = np.log10(y_train.copy())
    else:
        y_train_normalised = y_train.copy()
    mean_y_train = np.mean(y_train_normalised, axis=0)
    y_train_normalised -= mean_y_train
    std_y_train = np.std(y_train_normalised, axis=0)
    y_train_normalised /= std_y_train

    # for i in range(y_train_normalised.shape[0]):
    #     pb.plot(initial_spectra, y_train_normalised[i,:], label=f'x_train: {x_train[i,0]}, {x_train[i,1]}')
    # pb.xlabel('Multipole l')
    # pb.ylabel('Normalised Power Spectrum')
    # pb.title('Auto Spectrum Training Data')
    # pb.legend()
    # pb.savefig('./Plots/emulator_auto_training_data_inside.png', dpi=300)
    # pb.clf()

    kernel = GPy.kern.RBF(x_train_normalised.shape[1], ARD=True)

    if not load:

        model = GPy.models.GPRegression(X=x_train_normalised, Y=y_train_normalised, kernel=kernel) 
        model.optimize()
    
        if save:
            # normalisation_params = np.array((mean_y_train, std_y_train)).reshape(-1, 2)
            path = Path(f'./gpy_model/{box}/{isim}/{iz}/lightcone{lightcone}/')
            path.mkdir(parents=True, exist_ok=True)
            np.savez(path+f'normalisation_parameters_{spectra}.npz', mean=mean_y_train, std=std_y_train)
            np.save(path+f'gpy_model_{spectra}.npy', model.param_array)

    elif load: 
        model = GPy.models.GPRegression(X=x_train_normalised, Y=y_train_normalised, kernel=kernel, initialize=False)
        model.update_model(False) # do not call the underlying expensive algebra on load
        model.initialize_parameter() # Initialize the parameters (connect the parameters up)
        
        model[:] = np.load(path+f'gpy_model_{spectra}.npy') # Load the parameters
        model.update_model(True) # Call the algebra only once
    
    x_test = np.array(((x[0] - x_train_min[0])/(x_train_max[0] - x_train_min[0]), (x[1] - x_train_min[1])/(x_train_max[1] - x_train_min[1]))).reshape(1,-1)
    model_output = model._raw_predict(x_test)

    if load:
        normalisation_params = np.load(path+f'normalisation_parameters_{spectra}.npz')

        mean = np.asarray(normalisation_params['mean']).reshape(-1)
        std  = np.asarray(normalisation_params['std']).reshape(-1)

        if log:
            y_test = 10**((model_output[0][0] * std) + mean)
        else:
            y_test = (model_output[0][0] * std) + mean
    elif not load:
        if log:
            y_test = 10**((model_output[0][0] * std_y_train) + mean_y_train)
        else:
            y_test = ((model_output[0][0] * std_y_train) + mean_y_train) #* initial_spectra
    posterior_variance = model_output[1]
    #print(normalisation_params)

    return y_test

if __name__ == '__main__':
    import sys

    box = str(sys.argv[1])
    isim = str(sys.argv[2])
    iz = str(sys.argv[3])

    spectra = str(sys.argv[4])
    amp = 10.8
    slope = 0.483
    residual = 0.079

    test = np.array((amp+residual, slope))
    test_name = [f"{float(test[0]):.1f}".replace('.', 'p'), f"{float(test[1]):.1f}".replace('.', 'p')]
    test_name_base = [f"{float(amp):.1f}".replace('.', 'p'), f"{float(slope):.1f}".replace('.', 'p')]
    if amp+residual >= 11.3:
        pass
    else:
        test_name_plus_1 = [f"{float(amp+0.1):.1f}".replace('.', 'p'), f"{float(slope):.1f}".replace('.', 'p')]

    if spectra == 'auto':
        dir_type = 'galaxy_galaxy'
    elif spectra == 'cross':
        dir_type = 'kappa_galaxy'

    true = np.loadtxt(f'/cosma8/data/dp004/dc-conl1/FLAMINGO/patchy_screening/data_files/power_spectra/{dir_type}/{box}/{isim}/{iz}/lightcone0/{dir_type}_power_spectrum_{test_name[0]}_{test_name[1]}.txt', 
                    # delimiter=' ', skiprows=1, usecols=(0,1) if spectra=='auto' else (0,1))
                    delimiter=' ', skiprows=1, usecols=(0,2) if spectra=='auto' else (0,1))

    true_old = np.loadtxt(f'/cosma8/data/dp004/dc-conl1/FLAMINGO/patchy_screening/data_files/power_spectra/{dir_type}/{box}/{isim}/{iz}/lightcone0/{dir_type}_power_spectrum_{test_name_base[0]}_{test_name_base[1]}.txt', 
                    # delimiter=' ', skiprows=1, usecols=(0,1) if spectra=='auto' else (0,1))
                    delimiter=' ', skiprows=1, usecols=(0,2) if spectra=='auto' else (0,1))
    
    if amp+residual >= 11.3:
        pass
    else:
        true_plus_1 = np.loadtxt(f'/cosma8/data/dp004/dc-conl1/FLAMINGO/patchy_screening/data_files/power_spectra/{dir_type}/{box}/{isim}/{iz}/lightcone0/{dir_type}_power_spectrum_{test_name_plus_1[0]}_{test_name_plus_1[1]}.txt', 
                                # delimiter=' ', skiprows=1, usecols=(0,1) if spectra=='auto' else (0,1))
                                delimiter=' ', skiprows=1, usecols=(0,2) if spectra=='auto' else (0,1))

    pred = emulator(test, spectra, box, isim, iz, save=True, retrain=True)#, log=False)
    print(pred)
    #print(true[:, 1])
    #print(pred - true[:, 1])

    farren_data = np.loadtxt(f'./unWISExLens_lklh/data/v1.0/bandpowers/unWISExACT-DR6_{str(iz).lower()}_baseline_Clgg+Clkk+Clkg.dat', usecols=(0,1,3)).reshape(-1,3)
    ell_200_mask = np.where(farren_data[:,0] > 200)

    pb.loglog(farren_data[:,0][ell_200_mask], farren_data[:,1][ell_200_mask]*1e5 if spectra=='auto' else farren_data[:,2][ell_200_mask]*1e5, color='k', marker='.', markersize=5, label='ACT x unWISE (Farren et al. 2023)')
    pb.loglog(true[:,0], true[:,1], label=f'Mock catalog {amp+residual, slope}', color='b')
    if amp+residual >= 11.3:
        pass
    else:
        pass
        # pb.loglog(true_plus_1[:,0], true_plus_1[:,1], label=f'Mock catalog {amp+0.1, slope}', color='g')
    # pb.loglog(true[:,0], pred, label='Emulator', color='r')
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
    pb.plot(true[:,0], pred/true[:,1], label='Error w.r.t. simulated clustering')
    #pb.plot(true[:,0], pred/(farren_data[:,1][ell_200_mask]*1e5) if spectra=='auto' else pred/(farren_data[:,2][ell_200_mask]*1e5), label='Error w.r.t. observed clustering')
    pb.title(f'Emulator error test (x_test: Amplitude={test[0]}, Slope={test[1]})')
    pb.xlabel('$\ell$')
    pb.ylabel('Residual')
    pb.xlim(100, 3000)
    pb.legend()
    pb.tight_layout()
    pb.savefig('./Plots/mock_catalog_emulator_error_test.png', dpi=400)
    pb.clf()

    x_train = np.load(f"./gpy_model/{box}/{isim}/{iz}/lightcone0/training/X_training_data.npy")
    # x_train = np.loadtxt('./data_files/mock_catalog_test_points.txt')
    pred_errors = []
    for i in range(x_train.shape[0]): 
        name_i = [f"{float(x_train[i,0]):.1f}".replace('.', 'p'), f"{float(x_train[i,1]):.1f}".replace('.', 'p')]
        pred_i = emulator(x_train[i,:], spectra, box, isim, iz)
        true_i = np.loadtxt(f'/cosma8/data/dp004/dc-conl1/FLAMINGO/patchy_screening/data_files/power_spectra/{dir_type}/{box}/{isim}/{iz}/lightcone0/{dir_type}_power_spectrum_{name_i[0]}_{name_i[1]}.txt',
                        # delimiter=' ', skiprows=1, usecols=(0,1) if spectra=='auto' else (0,1))
                        delimiter=' ', skiprows=1, usecols=(0,2) if spectra=='auto' else (0,1))
        error_i = (pred_i)/true_i[:,1]
        error_i = np.array(error_i)
        print(error_i)
        pred_errors.append(error_i)
    print(pred_errors)
    pred_errors = np.array(pred_errors)
    mean_pred_errors = np.mean(pred_errors, axis=0)
    print(mean_pred_errors.shape)
    std_pred_errors = np.std(pred_errors, axis=0)

    pb.hlines(y=1.000, xmin=-1, xmax=np.max(true[:,0])*1.1, color='k', linestyles='solid', alpha=0.5, label=None)
    pb.hlines(y=0.990, xmin=-1, xmax=np.max(true[:,0])*1.1, color='k', linestyles='dashed', alpha=0.5, label='1% error')
    pb.hlines(y=1.010, xmin=-1, xmax=np.max(true[:,0])*1.1, color='k', linestyles='dashed', alpha=0.5, label=None)
    pb.hlines(y=0.950, xmin=-1, xmax=np.max(true[:,0])*1.1, color='k', linestyles='dotted', alpha=0.5, label='5% error')
    pb.hlines(y=1.050, xmin=-1, xmax=np.max(true[:,0])*1.1, color='k', linestyles='dotted', alpha=0.5, label=None)
    pb.plot(true[:,0], mean_pred_errors, label='Mean error w.r.t. simulated clustering')
    pb.fill_between(true[:,0], mean_pred_errors - std_pred_errors, mean_pred_errors + std_pred_errors, color='gray', alpha=0.5, label='1$\sigma$ scatter')
    pb.title(f'Mean emulator error test over training set ({spectra}, {box}, {isim}, {iz})', wrap=True)
    pb.xlabel('$\ell$')
    pb.ylabel('Residual')
    pb.xlim(100, 3000)
    pb.legend()
    pb.tight_layout()
    pb.savefig('./Plots/mock_catalog_emulator_mean_error_test.png', dpi=400)
    pb.clf()