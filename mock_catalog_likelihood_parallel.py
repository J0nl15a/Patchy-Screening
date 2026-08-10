import numpy as np, pylab as pb
# from mock_catalog_emulator import emulator
import emcee
from scipy.optimize import minimize
import os, sys
import multiprocessing
import time
os.environ["OMP_NUM_THREADS"] = "1"
import importlib.util
from pathlib import Path

# Absolute or relative path to your script
# script_path = '/cosma8/data/dp004/dc-conl1/FLAMINGO/patchy_screening/mock_catalog_emulator.py'
script_path = '/cosma8/data/dp004/dc-conl1/FLAMINGO/patchy_screening/mock_catalogue_emulator_improved.py'
# Module name to give it (can be anything)
module_name = 'mock_catalog_emulator'
# Load the module from the file
spec = importlib.util.spec_from_file_location(module_name, script_path)
module = importlib.util.module_from_spec(spec)
sys.modules[module_name] = module
spec.loader.exec_module(module)


def log_likelihood(theta, f_obs, f_obs_err, box, isim, iz, lightcone=0, abundance_cut=0.5, 
                   amp_min=10.3, amp_max=11.3, amp_step=0.1, slope_min=0.0, slope_max=1.0, slope_step=0.1):
    amp, slope = theta
    # print(theta)
    # print(amp, slope)
    x = np.array((amp, slope))
    # print(x)
    f_sim_auto, _, f_sim_auto_var, f_sim_auto_std = module.emulator(x, 'auto', box, isim, iz, load=True, lightcone=lightcone, abundance_cut=abundance_cut,
                                                                 amp_min=amp_min, amp_max=amp_max, amp_step=amp_step, slope_min=slope_min, slope_max=slope_max, slope_step=slope_step)
    f_sim_cross, _, f_sim_cross_var, f_sim_cross_std = module.emulator(x, 'cross', box, isim, iz, load=True, lightcone=lightcone, abundance_cut=abundance_cut,
                                                                    amp_min=amp_min, amp_max=amp_max, amp_step=amp_step, slope_min=slope_min, slope_max=slope_max, slope_step=slope_step)

    total_variance_auto =  f_obs_err['auto']**2 #+ f_sim_auto_var
    total_variance_cross = f_obs_err['cross']**2 #+ f_sim_cross_var
    
    chi_sq_auto = np.sum(((f_obs['auto'] - f_sim_auto)/np.sqrt(total_variance_auto))**2)
    chi_sq_cross = np.sum(((f_obs['cross'] - f_sim_cross)/np.sqrt(total_variance_cross))**2)

    # log_likelihood_auto = -0.5 * np.sum(((f_obs['auto'] - f_sim_auto)**2)/total_variance_auto 
                                        # + np.log(2 * np.pi * total_variance_auto))
    # log_likelihood_cross = -0.5 * np.sum(((f_obs['cross'] - f_sim_cross)**2)/total_variance_cross 
                                        #  + np.log(2 * np.pi * total_variance_cross))

    # print(chi_sq_auto, chi_sq_cross)
    return -0.5 * (chi_sq_auto + chi_sq_cross)
    # return log_likelihood_auto + log_likelihood_cross

def log_prior(theta, box, isim, iz, lightcone=0, abundance_cut=0.5,
              amp_min=10.3, amp_max=11.3, amp_step=0.1, slope_min=0.0, slope_max=1.0, slope_step=0.1):
    amp, slope = theta
    if iz == 'Blue':
        obs_nbar = 3409
    elif iz == 'Green':
        obs_nbar = 1846
    obs_nbar_full_sky = obs_nbar * 41253

    if amp_min <= amp <= amp_max and slope_min <= slope <= slope_max:
        predicted_nbar, _, _, _ = module.emulator(theta, 'abundance', box, isim, iz, load=True, lightcone=lightcone, abundance_cut=0.0,
                                 amp_min=amp_min, amp_max=amp_max, amp_step=amp_step, slope_min=slope_min, slope_max=slope_max, slope_step=slope_step)
        predicted_nbar = float(np.asarray(predicted_nbar).squeeze())
        
        if predicted_nbar >= (obs_nbar_full_sky * abundance_cut):
            return 0.0
        elif predicted_nbar < (obs_nbar_full_sky * abundance_cut):
            return -np.inf
    else:
        return -np.inf

def log_probability(theta, f_obs, f_obs_err, box, isim, iz, lightcone=0, abundance_cut=0.5,
                    amp_min=10.3, amp_max=11.3, amp_step=0.1, slope_min=0.0, slope_max=1.0, slope_step=0.1):
    lp = log_prior(theta, box, isim, iz, lightcone=lightcone, abundance_cut=abundance_cut,
                   amp_min=amp_min, amp_max=amp_max, amp_step=amp_step, slope_min=slope_min, slope_max=slope_max, slope_step=slope_step)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, f_obs, f_obs_err, box, isim, iz, lightcone=lightcone, abundance_cut=abundance_cut,
                               amp_min=amp_min, amp_max=amp_max, amp_step=amp_step, slope_min=slope_min, slope_max=slope_max, slope_step=slope_step)

def multiprocess(f_obs, f_obs_err, box, isim, iz, steps, lightcone=0, abundance_cut=0.5,
                 amp_min=10.3, amp_max=11.3, amp_step=0.1, slope_min=0.0, slope_max=1.0, slope_step=0.1): #, backend):

    with multiprocessing.get_context("spawn").Pool() as pool:
        start = time.time()

        moves = [(emcee.moves.StretchMove(a=2.5), 0.6), (emcee.moves.DEMove(), 0.3), (emcee.moves.DESnookerMove(), 0.1)]

        sampler = emcee.EnsembleSampler(
            nwalkers, ndim, log_probability, args=(f_obs, f_obs_err, box, isim, iz, lightcone, abundance_cut,
                                                   amp_min, amp_max, amp_step, slope_min, slope_max, slope_step), 
            pool=pool, moves=moves #, backend=backend
        )

        check_steps = 1000
        old_tau = np.full(ndim, np.inf)

        state = sampler.run_mcmc(pos, steps, progress=True)

        while True:
            try:
                tau = sampler.get_autocorr_time(tol=0)

                long_enough = np.all(sampler.iteration > 50.0 * tau)

                stable = (np.all(np.isfinite(old_tau)) and np.all(np.abs(old_tau - tau) / tau < 0.05))

                print(f"steps={sampler.iteration}, tau={tau}, long_enough={long_enough}, stable={stable}, acceptance={np.mean(sampler.acceptance_fraction):.3f}")

                if long_enough and stable:
                    break

                old_tau = tau.copy()

            except emcee.autocorr.AutocorrError as error:
                print(f"steps={sampler.iteration}: tau not reliable yet: {error}")

            # Always extend the chain
            state = sampler.run_mcmc(state, check_steps, progress=True)

        end = time.time()
        multi_time = end - start
        print("Multiprocessing took {0:.1f} seconds".format(multi_time))

        return sampler

if __name__ == '__main__':
    from matplotlib.ticker import FormatStrFormatter
    np.random.seed(1000)

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

    farren_data_ = np.loadtxt(f'./unWISExLens_lklh/data/v1.0/bandpowers/unWISExACT-DR6_{str(iz).lower()}_baseline_Clgg+Clkk+Clkg.dat', usecols=(0,1,3)).reshape(-1,3)
    obs_data_covariance = np.loadtxt(f'./unWISExLens_lklh/data/v1.0/covariances/covmat_Clgg+Clkg_unWISExACT-DR6_{str(iz).lower()}_baseline.dat')
    ell_200_mask = np.where(farren_data_[:,0] > 200)

    f_sim_auto, _, _, _ = module.emulator(np.array((10.65, 0.45)), 'auto', box, isim, iz, save=True, retrain=True, lightcone=lightcone, abundance_cut=abundance_cut,
                                 amp_min=amp_min, amp_max=amp_max, amp_step=amp_step, slope_min=slope_min, slope_max=slope_max, slope_step=slope_step) #module.emulator(x, 'auto', box, isim, iz, load=True)
    f_sim_cross, _, _, _ = module.emulator(np.array((10.65, 0.45)), 'cross', box, isim, iz, save=True, retrain=True, lightcone=lightcone, abundance_cut=abundance_cut,
                                   amp_min=amp_min, amp_max=amp_max, amp_step=amp_step, slope_min=slope_min, slope_max=slope_max, slope_step=slope_step) #module.emulator(x, 'cross', box, isim, iz, load=True)
    f_sim_abundance, _, _, _ = module.emulator(np.array((10.65, 0.45)), 'abundance', box, isim, iz, save=True, retrain=True, lightcone=lightcone, abundance_cut=0.0,
                                amp_min=amp_min, amp_max=amp_max, amp_step=amp_step, slope_min=slope_min, slope_max=slope_max, slope_step=slope_step) #module

    farren_data_variance = np.diag(obs_data_covariance)
    farren_data_var_auto = farren_data_variance[:int(len(farren_data_variance)/2)][ell_200_mask]
    farren_data_var_cross = farren_data_variance[int(len(farren_data_variance)/2):][ell_200_mask]
    
    f_obs = {'auto': farren_data_[:,1][ell_200_mask] * 1e5, 'cross': farren_data_[:,2][ell_200_mask] * 1e5}
    f_obs_err = {'auto': np.sqrt(farren_data_var_auto) * 1e5, 'cross': np.sqrt(farren_data_var_cross) * 1e5}

    num_points = 1
    amp_positions = np.random.uniform(amp_min, amp_max+0.001, num_points)[0]
    slope_positions = np.random.uniform(slope_min, slope_max+0.001, num_points)[0]

    allowed = False
    while not allowed:
        if log_prior((amp_positions, slope_positions), box, isim, iz, lightcone, abundance_cut, 
                        amp_min, amp_max, amp_step, slope_min, slope_max, slope_step) == -np.inf:
            amp_positions = np.random.uniform(amp_min, amp_max+0.001, 1)[0]
            slope_positions = np.random.uniform(slope_min, slope_max+0.001, 1)[0]
        else:
            allowed = True

    nll = lambda *args: -log_probability(*args)
    initial = amp_positions, slope_positions #10.7, 0.4 #test_points[0], test_points[1]
    initial_name = f"{float(initial[0]):.3f}".replace('.', 'p'), f"{float(initial[1]):.3f}".replace('.', 'p')
    gaussian_offset = 0.05
    gaussian_offset_name = f"{float(gaussian_offset):.3f}".replace('.', 'p')
    walkers = int(2 ** 5)
    soln = minimize(nll, np.asarray(initial, dtype=float), args=(f_obs, f_obs_err, box, isim, iz, lightcone, 
                                                                 abundance_cut, amp_min, amp_max, amp_step, 
                                                                 slope_min, slope_max, slope_step),
                                                                 method="L-BFGS-B", bounds=[(amp_min, amp_max), (slope_min, slope_max)])
    #print(soln)
    initial_scale = np.array([0.005 * (amp_max - amp_min), 0.005 * (slope_max - slope_min)])
    pos = soln.x + np.random.randn(walkers, 2) * initial_scale
    # pos = initial + gaussian_offset * np.random.randn(walkers, 2)
    # pos = soln.x + 1e-4 * np.random.randn(walkers, 2)
    nwalkers, ndim = pos.shape
    steps = 10000

    prec = 4  # <-- change this to the number of decimals you want
    fmt  = f"%.{prec}f"

    # filename = f"./data_files/mcmc_chains/chain_walkers{nwalkers}_steps{steps}.h5"
    # backend = emcee.backends.HDFBackend(filename)
    # backend.reset(nwalkers, ndim)

    # success = False
    # while not success:
        # try:
    sampler = multiprocess(f_obs, f_obs_err, box, isim, iz, steps, lightcone=lightcone, abundance_cut=abundance_cut, 
                            amp_min=amp_min, amp_max=amp_max, amp_step=amp_step, slope_min=slope_min, slope_max=slope_max, slope_step=slope_step) #, backend)
    tau = sampler.get_autocorr_time()
    success = True

    burnin = int(2 * np.max(tau))
    thin = int(0.5 * np.min(tau))
    flat_samples = sampler.get_chain(discard=burnin, thin=thin, flat=True)

    # mle_amp = np.percentile(flat_samples[:, 0], [50])[0]
    # mle_slope = np.percentile(flat_samples[:, 1], [50])[0]

    # flat_log_prob = sampler.get_log_prob(discard=burnin, thin=thin, flat=True)
    # map_index = np.argmax(flat_log_prob)
    # map_theta = flat_samples[map_index]
    # mle_amp, mle_slope = map_theta

    flat_loglike = np.array([log_likelihood(theta, f_obs, f_obs_err, box, isim, iz, lightcone=lightcone,
                                            abundance_cut=abundance_cut, amp_min=amp_min, amp_max=amp_max, amp_step=amp_step,
                                            slope_min=slope_min, slope_max=slope_max, slope_step=slope_step) 
                                            for theta in flat_samples])

    mle_index = np.argmax(flat_loglike)
    mle_amp, mle_slope = flat_samples[mle_index]

            # if plateau_point < mle_amp <= vertical_limit:
            #     if (-1 * mle_amp + c)*0.99 <= mle_slope <= (-1 * mle_amp + c)*1.01:
            #         print(f"[INFO] MLE AMP: {mle_amp} and MLE SLOPE: {mle_slope} is within 1% of the prior boundary.")
            #         plateau_point += 0.1
            #         c += 0.1
            #         print(f"[INFO] Updated plateau_point to {plateau_point} and c to {c} to ensure MLE is within the prior. Rerunning MCMC...")
            #         success = False

        # except emcee.autocorr.AutocorrError:
        #     print("[WARNING] The chain is too short to estimate the autocorrelation time reliably.")
        #     success = False
        #     steps += 1000
        #     continue

    print(np.mean(sampler.acceptance_fraction))

    print(tau)

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
    pb.savefig(f'./Plots/mcmc_chains_optimal_value_{box}_{isim}_{iz}_mle.png', dpi=400)
    pb.clf()

    print(flat_samples.shape)
    
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

    median_amp = mle[0]
    median_slope = mle[1]
    err_amp_lower = mle_err_lower[0]
    err_amp_upper = mle_err_upper[0]
    err_slope_lower = mle_err_lower[1]
    err_slope_upper = mle_err_upper[1]
    print(f"[INFO] MLE AMP: {mle_amp}, MLE SLOPE: {mle_slope}")
    print(f"[INFO] Median AMP: {median_amp}, Median SLOPE: {median_slope}")

    log_likelihood_mle = log_likelihood((mle_amp, mle_slope), f_obs, f_obs_err, box, isim, iz, lightcone=lightcone,
                                        abundance_cut=abundance_cut, amp_min=amp_min, amp_max=amp_max, amp_step=amp_step,
                                        slope_min=slope_min, slope_max=slope_max, slope_step=slope_step)
    print(f"[INFO] Log-Likelihood at MLE: {log_likelihood_mle}")

    # Save MCMC chain
    np.save(f"./data_files/mcmc_chains/flat_samples_{box}_{isim}_{iz}_lightcone{lightcone}_{mle_amp}_{mle_slope}.npy", flat_samples)

    # write to text file in a known place
    path = f"./data_files/mle_parameters/{box}/{isim}/{iz}/lightcone{lightcone}/mle_values.txt"
    outfile = Path(path)
    outfile.parent.mkdir(parents=True, exist_ok=True)

    x = np.array((mle_amp, mle_slope))
    f_sim_auto, _, f_sim_auto_var, f_sim_auto_std = module.emulator(x, 'auto', box, isim, iz, load=True, lightcone=lightcone, 
                                                                    abundance_cut=abundance_cut, amp_min=amp_min, amp_max=amp_max, amp_step=amp_step,
                                                                    slope_min=slope_min, slope_max=slope_max, slope_step=slope_step) 
    f_sim_cross, _, f_sim_cross_var, f_sim_cross_std = module.emulator(x, 'cross', box, isim, iz, load=True, lightcone=lightcone, 
                                                                       abundance_cut=abundance_cut, amp_min=amp_min, amp_max=amp_max, amp_step=amp_step,
                                                                       slope_min=slope_min, slope_max=slope_max, slope_step=slope_step)
    chi_sq_auto = np.sum(((f_obs['auto'] - f_sim_auto)**2)/((f_obs_err['auto']**2)))
    chi_sq_cross = np.sum(((f_obs['cross'] - f_sim_cross)**2)/((f_obs_err['cross']**2)))
    # log_norm_auto = np.sum(np.log(2.0 * np.pi * ((f_obs_err['auto']**2) + f_sim_auto_var)))
    # log_norm_cross = np.sum(np.log(2.0 * np.pi * ((f_obs_err['cross']**2) + f_sim_cross_var)))

    with open(outfile, "w") as f:
        f.write(f"LOG_LIKELIHOOD={log_likelihood_mle:.13f}\n")
        f.write(f"CHI2={(log_likelihood_mle * -2):.13f}\n")
        # f.write(f"CHI2={(chi_sq_auto + chi_sq_cross):.13f}\n")
        f.write(f"LOG_LIKELIHOOD_AUTO={(chi_sq_auto * -0.5):.13f}\n")
        # f.write(f"LOG_LIKELIHOOD_AUTO={(-0.5 * (chi_sq_auto + log_norm_auto)):.13f}\n")
        f.write(f"CHI2_AUTO={chi_sq_auto:.13f}\n")
        f.write(f"LOG_LIKELIHOOD_CROSS={(chi_sq_cross * -0.5):.13f}\n")
        # f.write(f"LOG_LIKELIHOOD_CROSS={(-0.5 * (chi_sq_cross + log_norm_cross)):.13f}\n")
        f.write(f"CHI2_CROSS={chi_sq_cross:.13f}\n")
        f.write(f"AMP={mle_amp:.3f}\n")
        f.write(f"SLOPE={mle_slope:.3f}\n")
        f.write(f"AMP_ERR_LOWER={err_amp_lower:.4f}\n")
        f.write(f"AMP_ERR_UPPER={err_amp_upper:.4f}\n")
        f.write(f"SLOPE_ERR_LOWER={err_slope_lower:.4f}\n")
        f.write(f"SLOPE_ERR_UPPER={err_slope_upper:.4f}\n")

    print(f"[INFO] Wrote MLEs to {outfile}")


    fig = corner.corner(
        flat_samples, 
        labels=labels, 
        quantiles=[.16, .5, .84],
        show_titles=True,
        title_fmt=f".{prec}f", 
        title_kwargs={"fontsize": 12},
        truths=[mle_amp, mle_slope])
    for ax in fig.axes:
        ax.xaxis.set_major_formatter(FormatStrFormatter(fmt))
        ax.yaxis.set_major_formatter(FormatStrFormatter(fmt))
    # pb.savefig(f'./Plots/mcmc_corner_{test_name[0]}_{test_name[1]}_steps{steps}_walkers{nwalkers}_initialpos{initial_name[0]}_{initial_name[1]}_offset{gaussian_offset_name}.png', dpi=400)
    pb.savefig(f'./Plots/mcmc_corner_optimal_value_{box}_{isim}_{iz}_mle.png', dpi=400)
    pb.clf()
