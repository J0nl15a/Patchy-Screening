from pathlib import Path
import numpy as np, pylab as pb
from mock_catalog_emulator import emulator
import emcee
from scipy.optimize import minimize
import os, sys
import multiprocessing
import time
os.environ["OMP_NUM_THREADS"] = "1"
import importlib.util

# # Absolute or relative path to your script
# script_path = '/cosma8/data/dp004/dc-conl1/FLAMINGO/patchy_screening/mock_catalog_emulator.py'
# # Module name to give it (can be anything)
# module_name = 'mock_catalog_emulator'
# # Load the module from the file
# spec = importlib.util.spec_from_file_location(module_name, script_path)
# module = importlib.util.module_from_spec(spec)
# sys.modules[module_name] = module
# spec.loader.exec_module(module)


def log_likelihood(theta, f_obs, f_obs_err, box, isim, iz, lightcone=0, abundance_cut=0.05, 
                   amp_min=10.3, amp_max=11.3, amp_step=0.1, slope_min=0.0, slope_max=1.0, slope_step=0.1):
    amp, slope = theta
    # print(theta)
    # print(amp, slope)
    x = np.array((amp, slope))
    # print(x)
    f_sim_auto = emulator(x, 'auto', box, isim, iz, load=True, lightcone=lightcone, abundance_cut=abundance_cut,
                           amp_min=amp_min, amp_max=amp_max, amp_step=amp_step, slope_min=slope_min, slope_max=slope_max, slope_step=slope_step) #module.emulator(x, 'auto', box, isim, iz, load=True)
    f_sim_cross = emulator(x, 'cross', box, isim, iz, load=True, lightcone=lightcone, abundance_cut=abundance_cut,
                            amp_min=amp_min, amp_max=amp_max, amp_step=amp_step, slope_min=slope_min, slope_max=slope_max, slope_step=slope_step) #module.emulator(x, 'cross', box, isim, iz, load=True)
    # print(f"f_obs['auto'] = {f_obs['auto']}, f_obs['cross'] = {f_obs['cross']}")
    # print(f"f_sim_auto/f_obs['auto'] = {f_sim_auto/f_obs['auto']}, f_sim_cross/f_obs['cross'] = {f_sim_cross/f_obs['cross']}")
    chi_sq_auto = np.sum(((f_obs['auto'] - f_sim_auto)/f_obs_err['auto'])**2)
    chi_sq_cross = np.sum(((f_obs['cross'] - f_sim_cross)/f_obs_err['cross'])**2)
    # chi_sq_auto = np.sum(((f_obs['auto'] - f_sim_auto)**2)/(f_obs['auto'] * 0.01))   #using 1% error bars for chi^2 calculation
    # chi_sq_cross = np.sum(((f_obs['cross'] - f_sim_cross)**2)/(f_obs['cross'] * 0.01))   #using 1% error bars for chi^2 calculation
    # print(chi_sq_auto, chi_sq_cross)
    return -0.5 * (chi_sq_auto + chi_sq_cross)

def log_likelihood_mock_catalogue(theta, f_obs, f_obs_err, box, isim, iz, lightcone=0):
    amp, slope = theta
    amp_name = f"{float(amp):.3f}".replace('.', 'p')
    slope_name = f"{float(slope):.3f}".replace('.', 'p')
    f_mock_auto = np.loadtxt(f'./data_files/power_spectra/galaxy_galaxy/{box}/{str(isim)}/{str(iz)}/lightcone{lightcone}/galaxy_galaxy_power_spectrum_{amp_name}_{slope_name}.txt', skiprows=1, usecols=(2))    
    f_mock_cross = np.loadtxt(f'./data_files/power_spectra/kappa_galaxy/{box}/{str(isim)}/{str(iz)}/lightcone{lightcone}/kappa_galaxy_power_spectrum_{amp_name}_{slope_name}.txt', skiprows=1, usecols=(1))
    chi_sq_auto = np.sum(((f_obs['auto'] - f_mock_auto)/f_obs_err['auto'])**2)
    chi_sq_cross = np.sum(((f_obs['cross'] - f_mock_cross)/f_obs_err['cross'])**2)
    return -0.5 * (chi_sq_auto + chi_sq_cross)

# def log_prior(theta, plateau_point, m, c):
def log_prior(theta, plateau_point, c, vertical_limit, box, isim, iz, lightcone=0, abundance_cut=0.05,
              amp_min=10.3, amp_max=11.3, amp_step=0.1, slope_min=0.0, slope_max=1.0, slope_step=0.1):
    amp, slope = theta
    if iz == 'Blue':
        obs_nbar = 3409
    elif iz == 'Green':
        obs_nbar = 1846
    obs_nbar_full_sky = obs_nbar * 41253
    # if amp_min <= amp <= plateau_point and slope_min <= slope <= slope_max:
    #     return 0.0
    # elif plateau_point < amp <= vertical_limit and slope_min <= slope <= (-1 * amp + c):
    #     return 0.0
    if amp_min <= amp <= amp_max and slope_min <= slope <= slope_max:
        predicted_nbar = emulator(theta, 'abundance', box, isim, iz, load=True, lightcone=lightcone, abundance_cut=0.0,
                                 amp_min=amp_min, amp_max=amp_max, amp_step=amp_step, slope_min=slope_min, slope_max=slope_max, slope_step=slope_step)
        if predicted_nbar >= (obs_nbar_full_sky * abundance_cut):
            return 0.0
        elif predicted_nbar < (obs_nbar_full_sky * abundance_cut):
            return -np.inf
    else:
        return -np.inf

def log_probability(theta, f_obs, f_obs_err, box, isim, iz, plateau_point, c, vertical_limit, lightcone=0, abundance_cut=0.05,
                    amp_min=10.3, amp_max=11.3, amp_step=0.1, slope_min=0.0, slope_max=1.0, slope_step=0.1):
    lp = log_prior(theta, plateau_point, c, vertical_limit, box, isim, iz, lightcone=lightcone, abundance_cut=abundance_cut,
                   amp_min=amp_min, amp_max=amp_max, amp_step=amp_step, slope_min=slope_min, slope_max=slope_max, slope_step=slope_step)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, f_obs, f_obs_err, box, isim, iz, lightcone=lightcone, abundance_cut=abundance_cut,
                                amp_min=amp_min, amp_max=amp_max, amp_step=amp_step, slope_min=slope_min, slope_max=slope_max, slope_step=slope_step)

def multiprocess(f_obs, f_obs_err, box, isim, iz, plateau_point, c, vertical_limit, lightcone=0, abundance_cut=0.05,
                 amp_min=10.3, amp_max=11.3, amp_step=0.1, slope_min=0.0, slope_max=1.0, slope_step=0.1): #, backend):

    with multiprocessing.get_context("spawn").Pool() as pool:
        start = time.time()
        sampler = emcee.EnsembleSampler(
            nwalkers, ndim, log_probability, args=(f_obs, f_obs_err, box, isim, iz, plateau_point, c, vertical_limit, lightcone, abundance_cut,
                                                   amp_min, amp_max, amp_step, slope_min, slope_max, slope_step), 
            pool=pool #, backend=backend
        )
        sampler.run_mcmc(pos, steps, progress=True)
        end = time.time()
        multi_time = end - start
        print("Multiprocessing took {0:.1f} seconds".format(multi_time))

        return sampler

if __name__ == '__main__':
    from matplotlib.ticker import FormatStrFormatter

    box = sys.argv[2]
    isim = sys.argv[3]
    iz = sys.argv[4]
    lightcone = int(sys.argv[5])
    abundance_cut = float(sys.argv[6])

    amp_min = float(sys.argv[7])
    amp_max = float(sys.argv[8])
    amp_step = float(sys.argv[9])
    slope_min = float(sys.argv[10])
    slope_max = float(sys.argv[11])
    slope_step = float(sys.argv[12])

    farren_data = np.loadtxt(f'./unWISExLens_lklh/data/v1.0/bandpowers/unWISExACT-DR6_{str(iz).lower()}_baseline_Clgg+Clkk+Clkg.dat', usecols=(0,1,3)).reshape(-1,3)
    obs_data_covariance = np.loadtxt(f'./unWISExLens_lklh/data/v1.0/covariances/covmat_Clgg+Clkg_unWISExACT-DR6_{str(iz).lower()}_baseline.dat')
    ell_200_mask = np.where(farren_data[:,0] > 200)

    f_sim_auto = emulator(np.array((10.65, 0.45)), 'auto', box, isim, iz, save=True, retrain=True, lightcone=lightcone, abundance_cut=abundance_cut,
                           amp_min=amp_min, amp_max=amp_max, amp_step=amp_step, slope_min=slope_min, slope_max=slope_max, slope_step=slope_step) #module
    f_sim_cross = emulator(np.array((10.65, 0.45)), 'cross', box, isim, iz, save=True, retrain=True, lightcone=lightcone, abundance_cut=abundance_cut,
                            amp_min=amp_min, amp_max=amp_max, amp_step=amp_step, slope_min=slope_min, slope_max=slope_max, slope_step=slope_step) #module
    f_sim_abundance = emulator(np.array((10.65, 0.45)), 'abundance', box, isim, iz, save=True, retrain=True, lightcone=lightcone, abundance_cut=0.0,
                                amp_min=amp_min, amp_max=amp_max, amp_step=amp_step, slope_min=slope_min, slope_max=slope_max, slope_step=slope_step) #module
    prior_limits = np.load(f'./gpy_model/{box}/{isim}/{iz}/lightcone{lightcone}/training/prior_limits.npy')
    plateau_point = prior_limits[0]
    # m = prior_limits[1]
    c = prior_limits[1] #+ 0.1
    vertical_limit = prior_limits[2]
    print(plateau_point, c, vertical_limit)

    farren_data_variance = np.diag(obs_data_covariance)
    farren_data_var_auto = farren_data_variance[:int(len(farren_data_variance)/2)][ell_200_mask]
    farren_data_var_cross = farren_data_variance[int(len(farren_data_variance)/2):][ell_200_mask]
    print(farren_data_variance.shape, ell_200_mask, farren_data_var_auto.shape, farren_data_var_cross.shape)

    # if isim == 'HYDRO_FIDUCIAL':
    #     test_points = np.loadtxt('./data_files/mock_catalog_test_points.txt', delimiter=' ')[1,:]
    #     test_name = [f"{float(test_points[0]):.1f}".replace('.', 'p'), f"{float(test_points[1]):.3f}".replace('.', 'p')]
    #     test_auto = np.loadtxt(f'/cosma8/data/dp004/dc-conl1/FLAMINGO/patchy_screening/data_files/power_spectra/galaxy_galaxy/{box}/{str(isim)}/{str(iz)}/galaxy_galaxy_power_spectrum_{test_name[0]}_{test_name[1]}.txt', 
    #                             delimiter=' ', skiprows=1, usecols=(2))
    #     test_cross = np.loadtxt(f'/cosma8/data/dp004/dc-conl1/FLAMINGO/patchy_screening/data_files/power_spectra/kappa_galaxy/{box}/{str(isim)}/{str(iz)}/kappa_galaxy_power_spectrum_{test_name[0]}_{test_name[1]}.txt', 
    #                             delimiter=' ', skiprows=1, usecols=(1))
    
    f_obs = {'auto': farren_data[:,1][ell_200_mask] * 1e5, 'cross': farren_data[:,2][ell_200_mask] * 1e5}
    f_obs_err = {'auto': np.sqrt(farren_data_var_auto) * 1e5, 'cross': np.sqrt(farren_data_var_cross) * 1e5}
    print(np.sqrt(f_obs_err['auto'])*1e5, np.sqrt(f_obs_err['cross'])*1e5)
    print(f_obs['auto'], f_obs['cross'])
    print(f_obs_err['auto'], f_obs_err['cross'])

    # test_auto = 10.8
    # test_cross = 0.5
    # f_obs = {'auto': test_auto, 'cross': test_cross}
    # f_obs_err = {'auto': (test_auto * .01)**2, 'cross': (test_cross * .01)**2}

    # if isim == 'HYDRO_FIDUCIAL':
        # print(log_probability((test_points[0], test_points[1]), f_obs, f_obs_err, box, isim, iz, plateau_point, c, vertical_limit))
    
    num_points = 1
    amp_positions = np.random.uniform(amp_min, amp_max+0.001, num_points)
    slope_positions = np.random.uniform(slope_min, slope_max+0.001, num_points)
    
    mle_amps = []
    mle_slopes = []
    mle_likelihoods = []

    for i in range(num_points):

        allowed = False
        while not allowed:
            if log_prior((amp_positions[i], slope_positions[i]), plateau_point, c, vertical_limit, box, isim, iz, lightcone, abundance_cut, 
                         amp_min, amp_max, amp_step, slope_min, slope_max, slope_step) == -np.inf:
                amp_positions[i] = np.random.uniform(amp_min, amp_max+0.001, 1)[0]
                slope_positions[i] = np.random.uniform(slope_min, slope_max+0.001, 1)[0]
            else:
                allowed = True

        np.random.seed(1000)
        # nll = lambda *args: -log_likelihood(*args)
        initial = amp_positions[i], slope_positions[i] #10.6, 0.4 #test_points[0], test_points[1]
        print(f"[INFO] Starting MCMC with initial guess: AMP = {initial[0]}, SLOPE = {initial[1]}")
        initial_name = f"{float(initial[0]):.3f}".replace('.', 'p'), f"{float(initial[1]):.3f}".replace('.', 'p')
        gaussian_offset = 0.05
        gaussian_offset_name = f"{float(gaussian_offset):.3f}".replace('.', 'p')
        walkers = int(2 ** 5)
        # soln = minimize(nll, initial, args=(f_obs, f_obs_err, box, isim, iz))
        # print(soln)
        pos = initial + gaussian_offset * np.random.randn(walkers, 2) 
        # pos = soln.x + 1e-4 * np.random.randn(walkers, 2)
        
        nwalkers, ndim = pos.shape
        steps = 5000

        prec = 4  # <-- change this to the number of decimals you want
        fmt  = f"%.{prec}f"

        log_prob = log_probability((initial[0], initial[1]), f_obs, f_obs_err, box, isim, iz, plateau_point, c, vertical_limit, lightcone=lightcone, abundance_cut=abundance_cut, 
                                   amp_min=amp_min, amp_max=amp_max, amp_step=amp_step, slope_min=slope_min, slope_max=slope_max, slope_step=slope_step)
        print(f"[INFO] Log-probability at initial guess: {log_prob}")

        # filename = f"./data_files/mcmc_chains/chain_walkers{nwalkers}_steps{steps}.h5"
        # backend = emcee.backends.HDFBackend(filename)
        # backend.reset(nwalkers, ndim)

        # vals = np.array([log_probability(p, f_obs, f_obs_err, box, isim, iz, plateau_point, c, vertical_limit, lightcone=lightcone, abundance_cut=abundance_cut,
        #                             amp_min=amp_min, amp_max=amp_max, amp_step=amp_step, slope_min=slope_min, slope_max=slope_max, slope_step=slope_step)
        #                 for p in pos], dtype=float)

        # print("finite:", np.isfinite(vals).sum(), "/", len(vals))
        # bad = np.where(~np.isfinite(vals))[0]
        # print("bad idx (first 10):", bad[:10])
        # for i in bad[:5]:
        #     print("theta:", pos[i], "logp:", vals[i])

        success = False
        while not success:
            try:
                sampler = multiprocess(f_obs, f_obs_err, box, isim, iz, plateau_point, c, vertical_limit, lightcone=lightcone, abundance_cut=abundance_cut,
                                    amp_min=amp_min, amp_max=amp_max, amp_step=amp_step, slope_min=slope_min, slope_max=slope_max, slope_step=slope_step) #, backend)
                tau = sampler.get_autocorr_time()
                # print(tau)
                success = True

                burnin = int(2 * np.max(tau))
                thin = int(0.5 * np.min(tau))
                flat_samples = sampler.get_chain(discard=burnin, thin=thin, flat=True)

                mle_amp = np.percentile(flat_samples[:, 0], [50])[0]
                mle_slope = np.percentile(flat_samples[:, 1], [50])[0]

                # if plateau_point < mle_amp <= vertical_limit:
                #     if (-1 * mle_amp + c)*0.99 <= mle_slope <= (-1 * mle_amp + c)*1.01:
                #         # print(f"[INFO] MLE AMP: {mle_amp} and MLE SLOPE: {mle_slope} is within 1% of the prior boundary.")
                #         plateau_point += 0.1
                #         c += 0.1
                #         # print(f"[INFO] Updated plateau_point to {plateau_point} and c to {c} to ensure MLE is within the prior. Rerunning MCMC...")
                #         success = False

            except emcee.autocorr.AutocorrError:
                print("[WARNING] The chain is too short to estimate the autocorrelation time reliably.")
                success = False
                steps += 1000
                continue

        print(f"[INFO] MLE AMP: {mle_amp}, MLE SLOPE: {mle_slope}")

        log_likelihood_mle = log_likelihood((mle_amp, mle_slope), f_obs, f_obs_err, box, isim, iz, lightcone=lightcone)
        print(f"[INFO] Log-Likelihood at MLE: {log_likelihood_mle}")
        print(f"[INFO] Expected number of galaxies at MLE: {emulator((mle_amp, mle_slope), 'abundance', box, isim, iz, load=True, lightcone=lightcone, abundance_cut=0.0, amp_min=amp_min, amp_max=amp_max, amp_step=amp_step, slope_min=slope_min, slope_max=slope_max, slope_step=slope_step)}")

        mle_amps.append(mle_amp)
        mle_slopes.append(mle_slope)
        mle_likelihoods.append(log_likelihood_mle)

        x = np.array((mle_amp, mle_slope))
        f_sim_auto = emulator(x, 'auto', box, isim, iz, load=True, lightcone=lightcone) 
        f_sim_cross = emulator(x, 'cross', box, isim, iz, load=True, lightcone=lightcone) 
        chi_sq_auto = np.sum(((f_obs['auto'] - f_sim_auto)**2)/(f_obs_err['auto']**2))
        chi_sq_cross = np.sum(((f_obs['cross'] - f_sim_cross)**2)/(f_obs_err['cross']**2))
        print(f"Auto chi^2 = {chi_sq_auto}")
        print(f"Auto likelihood = {-0.5 * chi_sq_auto}")
        print(f"Auto reduced chi^2 (Using N-P as d.o.f.) = {chi_sq_auto / (55-2)}")
        print(f"Auto reduced chi^2 (Using N-1 as d.o.f.) = {chi_sq_auto / (55-1)}") 
        print(f"Cross chi^2 = {chi_sq_cross}")
        print(f"Cross likelihood = {-0.5 * chi_sq_cross}")
        print(f"Cross reduced chi^2 (Using N-P as d.o.f.) = {chi_sq_cross / (55-2)}")
        print(f"Cross reduced chi^2 (Using N-1 as d.o.f.) = {chi_sq_cross / (55-1)}")


    for i in range(num_points):
        print(mle_amps[i], mle_slopes[i], mle_likelihoods[i])
    # quit()
    
    # np.save(f'./gpy_model/{box}/{isim}/{iz}/lightcone{lightcone}/training/prior_limits.npy', np.array([plateau_point, c, vertical_limit]))

    fig, axes = pb.subplots(2, figsize=(10, 7), sharex=True)
    samples = sampler.get_chain()
    labels = ["amp", "slope"]
    for i in range(ndim):
        ax = axes[i]
        ax.plot(samples[:, :, i], "k", alpha=0.3)
        ax.set_xlim(0, len(samples))
        ax.set_ylabel(labels[i])
        ax.yaxis.set_label_coords(-0.1, 0.5)
        ax.yaxis.set_major_formatter(FormatStrFormatter(fmt))

    axes[-1].set_xlabel("step number")
    # pb.savefig(f'./Plots/mcmc_chains_{test_name[0]}_{test_name[1]}_steps{steps}_walkers{nwalkers}_initialpos{initial_name[0]}_{initial_name[1]}_offset{gaussian_offset_name}.png', dpi=400)
    pb.savefig(f'./Plots/mcmc_chains_optimal_value_{box}_{isim}_{iz}_steps{steps}_walkers{nwalkers}_initialpos{initial_name[0]}_{initial_name[1]}_offset{gaussian_offset_name}.png', dpi=400)
    pb.clf()
    
    import corner

    mle = []
    mle_err_lower = []
    mle_err_upper = []

    for i in range(ndim):
        mcmc = np.percentile(flat_samples[:, i], [16, 50, 84])
        q = np.diff(mcmc)
        mle.append(mcmc[1])
        mle_err_lower.append(q[0])   # 50th - 16th percentile
        mle_err_upper.append(q[1])   # 84th - 50th percentile

    err_amp_lower = mle_err_lower[0]
    err_amp_upper = mle_err_upper[0]
    err_slope_lower = mle_err_lower[1]
    err_slope_upper = mle_err_upper[1]

    fig = corner.corner(
        flat_samples, 
        labels=labels, 
        quantiles=[.16, .5, .84],
        show_titles=True,
        title_fmt=f".{prec}f", 
        title_kwargs={"fontsize": 12},
        range=[(10.3, vertical_limit), (0.0, 1.0)])
    for ax in fig.axes:
        ax.xaxis.set_major_formatter(FormatStrFormatter(fmt))
        ax.yaxis.set_major_formatter(FormatStrFormatter(fmt))
    # pb.savefig(f'./Plots/mcmc_corner_{test_name[0]}_{test_name[1]}_steps{steps}_walkers{nwalkers}_initialpos{initial_name[0]}_{initial_name[1]}_offset{gaussian_offset_name}.png', dpi=400)
    pb.savefig(f'./Plots/mcmc_corner_optimal_value_{box}_{isim}_{iz}_steps{steps}_walkers{nwalkers}_initialpos{initial_name[0]}_{initial_name[1]}_offset{gaussian_offset_name}.png', dpi=400)
    pb.clf()

    fig = corner.corner(
        flat_samples, 
        labels=labels, 
        quantiles=[.16, .5, .84],
        show_titles=True,
        title_fmt=f".{prec}f", 
        title_kwargs={"fontsize": 12})
    for ax in fig.axes:
        ax.xaxis.set_major_formatter(FormatStrFormatter(fmt))
        ax.yaxis.set_major_formatter(FormatStrFormatter(fmt))
    # pb.savefig(f'./Plots/mcmc_corner_zoom_{test_name[0]}_{test_name[1]}_steps{steps}_walkers{nwalkers}_initialpos{initial_name[0]}_{initial_name[1]}_offset{gaussian_offset_name}.png', dpi=400)
    pb.savefig(f'./Plots/mcmc_corner_zoom_optimal_value_{box}_{isim}_{iz}_steps{steps}_walkers{nwalkers}_initialpos{initial_name[0]}_{initial_name[1]}_offset{gaussian_offset_name}.png', dpi=400)
    pb.clf()


    mle_amp = 10.808
    mle_slope = 0.257
    mle_amp_name = f"{float(mle_amp):.3f}".replace('.', 'p')
    mle_slope_name = f"{float(mle_slope):.3f}".replace('.', 'p')
    print(f"[INFO] MLE AMP: {mle_amp}, MLE SLOPE: {mle_slope}")

    log_likelihood_mle = log_likelihood((mle_amp, mle_slope), f_obs, f_obs_err, box, isim, iz, lightcone=lightcone)
    print(f"[INFO] Log-Likelihood at MLE (emulator comparison): {log_likelihood_mle}")

    # write to text file in a known place
    path = f"./data_files/mle_parameters/{box}/{isim}/{iz}/lightcone{lightcone}/mle_values.txt"
    outfile = Path(path)
    outfile.parent.mkdir(parents=True, exist_ok=True)

    # with open(outfile, "w") as f:
    #     f.write(f"LOG_LIKELIHOOD={log_likelihood_mle:.13f}\n")
    #     f.write(f"AMP={mle_amp:.3f}\n")
    #     f.write(f"SLOPE={mle_slope:.3f}\n")
    #     f.write(f"AMP_ERR_LOWER={err_amp_lower:.4f}\n")
    #     f.write(f"AMP_ERR_UPPER={err_amp_upper:.4f}\n")
    #     f.write(f"SLOPE_ERR_LOWER={err_slope_lower:.4f}\n")
    #     f.write(f"SLOPE_ERR_UPPER={err_slope_upper:.4f}\n")

    # print(f"[INFO] Wrote MLEs to {outfile}")

    x = np.array((mle_amp, mle_slope))
    f_sim_auto = emulator(x, 'auto', box, isim, iz, load=True, lightcone=lightcone) 
    f_sim_cross = emulator(x, 'cross', box, isim, iz, load=True, lightcone=lightcone) 
    chi_sq_auto = np.sum(((f_obs['auto'] - f_sim_auto)**2)/(f_obs_err['auto']**2))
    chi_sq_cross = np.sum(((f_obs['cross'] - f_sim_cross)**2)/(f_obs_err['cross']**2))
    # chi_sq_auto = np.sum(((f_obs['auto'] - f_sim_auto)**2)/(f_obs['auto'] * 0.01))   #using 1% error bars for chi^2 calculation
    # chi_sq_cross = np.sum(((f_obs['cross'] - f_sim_cross)**2)/(f_obs['cross'] * 0.01))   #using 1% error bars for chi^2 calculation
    print(f"Auto chi^2 (Emulator) = {chi_sq_auto}")
    print(f"Auto likelihood (Emulator) = {-0.5 * chi_sq_auto}")
    print(f"Auto reduced chi^2 (Emulator) (Using N-P as d.o.f.) = {chi_sq_auto / (55-2)}")
    print(f"Auto reduced chi^2 (Emulator) (Using N-1 as d.o.f.) = {chi_sq_auto / (55-1)}") 
    print(f"Cross chi^2 (Emulator) = {chi_sq_cross}")
    print(f"Cross likelihood (Emulator) = {-0.5 * chi_sq_cross}")
    print(f"Cross reduced chi^2 (Emulator) (Using N-P as d.o.f.) = {chi_sq_cross / (55-2)}")
    print(f"Cross reduced chi^2 (Emulator) (Using N-1 as d.o.f.) = {chi_sq_cross / (55-1)}")

    log_likelihood_mock_catalogue = log_likelihood_mock_catalogue((mle_amp, mle_slope), f_obs, f_obs_err, box, isim, iz, lightcone=lightcone)    
    print(f"[INFO] Log-Likelihood at MLE (mock catalogue comparison): {log_likelihood_mock_catalogue}")

    f_mock_auto = np.loadtxt(f'./data_files/power_spectra/galaxy_galaxy/{box}/{str(isim)}/{str(iz)}/lightcone{lightcone}/galaxy_galaxy_power_spectrum_{mle_amp_name}_{mle_slope_name}.txt', skiprows=1, usecols=(2))    
    f_mock_cross = np.loadtxt(f'./data_files/power_spectra/kappa_galaxy/{box}/{str(isim)}/{str(iz)}/lightcone{lightcone}/kappa_galaxy_power_spectrum_{mle_amp_name}_{mle_slope_name}.txt', skiprows=1, usecols=(1))
    chi_sq_mock_auto = np.sum(((f_obs['auto'] - f_mock_auto)**2)/(f_obs_err['auto']**2))
    chi_sq_mock_cross = np.sum(((f_obs['cross'] - f_mock_cross)**2)/(f_obs_err['cross']**2))
    print(f"Auto chi^2 (Mock Catalogue) = {chi_sq_mock_auto}")
    print(f"Auto likelihood (Mock Catalogue) = {-0.5 * chi_sq_mock_auto}")
    print(f"Auto reduced chi^2 (Mock Catalogue) (Using N-P as d.o.f.) = {chi_sq_mock_auto / (55-2)}")
    print(f"Auto reduced chi^2 (Mock Catalogue) (Using N-1 as d.o.f.) = {chi_sq_mock_auto / (55-1)}") 
    print(f"Cross chi^2 (Mock Catalogue) = {chi_sq_mock_cross}")
    print(f"Cross likelihood (Mock Catalogue) = {-0.5 * chi_sq_mock_cross}")
    print(f"Cross reduced chi^2 (Mock Catalogue) (Using N-P as d.o.f.) = {chi_sq_mock_cross / (55-2)}")
    print(f"Cross reduced chi^2 (Mock Catalogue) (Using N-1 as d.o.f.) = {chi_sq_mock_cross / (55-1)}")


