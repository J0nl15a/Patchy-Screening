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
script_path = '/cosma8/data/dp004/dc-conl1/FLAMINGO/patchy_screening/mock_catalog_emulator.py'
# Module name to give it (can be anything)
module_name = 'mock_catalog_emulator'
# Load the module from the file
spec = importlib.util.spec_from_file_location(module_name, script_path)
module = importlib.util.module_from_spec(spec)
sys.modules[module_name] = module
spec.loader.exec_module(module)


def log_likelihood(theta, f_obs, f_obs_err, box, isim, iz):
    amp, slope = theta
    # print(theta)
    # print(amp, slope)
    x = np.array((amp, slope))
    # print(x)
    f_sim_auto = module.emulator(x, 'auto', box, isim, iz, load=True)
    f_sim_cross = module.emulator(x, 'cross', box, isim, iz, load=True)
    # print(f'f_sim_auto = {f_sim_auto}, f_sim_cross = {f_sim_cross}')
    # print(f"f_obs['auto'] = {f_obs['auto']}, f_obs['cross'] = {f_obs['cross']}")
    # print(f"f_sim_auto/f_obs['auto'] = {f_sim_auto/f_obs['auto']}, f_sim_cross/f_obs['cross'] = {f_sim_cross/f_obs['cross']}")
    chi_sq_auto = np.sum(((f_obs['auto'] - f_sim_auto)**2)/f_obs_err['auto'])
    chi_sq_cross = np.sum(((f_obs['cross'] - f_sim_cross)**2)/f_obs_err['cross'])
    # print(chi_sq_auto, chi_sq_cross)
    return -0.5 * (chi_sq_auto + chi_sq_cross)

# def log_prior(theta, plateau_point, m, c):
def log_prior(theta, plateau_point, c, vertical_limit):
    amp, slope = theta
    if 10.3 <= amp <= plateau_point and 0.0 <= slope <= 1.0:
        return 0.0
    elif plateau_point < amp <= vertical_limit and 0.0 <= slope <= (-1 * amp + c):
        return 0.0
    # if 10.3 <= amp <= 11.0 and 0.0 <= slope <= 1.0:
    #     return 0.0
    else:
        return -np.inf

def log_probability(theta, f_obs, f_obs_err, box, isim, iz, plateau_point, m, c):
    lp = log_prior(theta, plateau_point, m, c)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, f_obs, f_obs_err, box, isim, iz)

def multiprocess(f_obs, f_obs_err, box, isim, iz, plateau_point, m, c): #, backend):

    with multiprocessing.get_context("spawn").Pool() as pool:
        start = time.time()
        sampler = emcee.EnsembleSampler(
            nwalkers, ndim, log_probability, args=(f_obs, f_obs_err, box, isim, iz, plateau_point, m, c), pool=pool #, backend=backend
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

    farren_data_ = np.loadtxt(f'./unWISExLens_lklh/data/v1.0/bandpowers/unWISExACT-DR6_{str(iz).lower()}_baseline_Clgg+Clkk+Clkg.dat', usecols=(0,1,3)).reshape(-1,3)
    obs_data_covariance = np.loadtxt(f'./unWISExLens_lklh/data/v1.0/covariances/covmat_Clgg+Clkg_unWISExACT-DR6_{str(iz).lower()}_baseline.dat')
    ell_200_mask = np.where(farren_data_[:,0] > 200)

    f_sim_auto = module.emulator(np.array((10.65, 0.45)), 'auto', box, isim, iz, save=True, retrain=True) #module.emulator(x, 'auto', box, isim, iz, load=True)
    f_sim_cross = module.emulator(np.array((10.65, 0.45)), 'cross', box, isim, iz, save=True, retrain=True) #module.emulator(x, 'cross', box, isim, iz, load=True)                
    prior_limits = np.load(f'./gpy_model/training/prior_limits_{box}_{isim}_{iz}.npy')
    plateau_point = prior_limits[0]
    # m = prior_limits[1]
    c = prior_limits[1]
    vertical_limit = prior_limits[2]
    print(plateau_point, c, vertical_limit)

    farren_data_variance = np.diag(obs_data_covariance)
    farren_data_var_auto = farren_data_variance[:int(len(farren_data_variance)/2)][ell_200_mask]
    farren_data_var_cross = farren_data_variance[int(len(farren_data_variance)/2):][ell_200_mask]
    
    f_obs = {'auto': farren_data_[:,1][ell_200_mask] * 1e5, 'cross': farren_data_[:,2][ell_200_mask] * 1e5}
    f_obs_err = {'auto': farren_data_var_auto * 1e5, 'cross': farren_data_var_cross * 1e5}
    
    np.random.seed(1000)
    #nll = lambda *args: -log_likelihood(*args)
    initial = 10.7, 0.4 #test_points[0], test_points[1]
    initial_name = f"{float(initial[0]):.3f}".replace('.', 'p'), f"{float(initial[1]):.3f}".replace('.', 'p')
    gaussian_offset = 0.05
    gaussian_offset_name = f"{float(gaussian_offset):.3f}".replace('.', 'p')
    walkers = int(2 ** 5)
    #soln = minimize(nll, initial, args=(f_obs, f_obs_err, box, isim, iz, ncpu))
    #print(soln)
    pos = initial + gaussian_offset * np.random.randn(walkers, 2) #soln.x + 1e-4 * np.random.randn(32, 3)
    nwalkers, ndim = pos.shape
    steps = 5000

    prec = 4  # <-- change this to the number of decimals you want
    fmt  = f"%.{prec}f"

    f_sim_auto = module.emulator(np.array((10.65, 0.45)), 'auto', box, isim, iz, save=True, retrain=True) 
    f_sim_cross = module.emulator(np.array((10.65, 0.45)), 'cross', box, isim, iz, save=True, retrain=True) 

    # filename = f"./data_files/mcmc_chains/chain_walkers{nwalkers}_steps{steps}.h5"
    # backend = emcee.backends.HDFBackend(filename)
    # backend.reset(nwalkers, ndim)

    sampler = multiprocess(f_obs, f_obs_err, box, isim, iz, plateau_point, c, vertical_limit)#, backend)

    print(np.mean(sampler.acceptance_fraction))

    tau = sampler.get_autocorr_time()
    print(tau)
    burnin = int(2 * np.max(tau))
    thin = int(0.5 * np.min(tau))

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


    flat_samples = sampler.get_chain(discard=burnin, thin=thin, flat=True)
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

    mle_amp = mle[0]
    mle_slope = mle[1]
    err_amp_lower = mle_err_lower[0]
    err_amp_upper = mle_err_upper[0]
    err_slope_lower = mle_err_lower[1]
    err_slope_upper = mle_err_upper[1]
    print(f"[INFO] MLE AMP: {mle_amp}, MLE SLOPE: {mle_slope}")

    log_likelihood_mle = log_likelihood((mle_amp, mle_slope), f_obs, f_obs_err, box, isim, iz)
    print(f"[INFO] Log-Likelihood at MLE: {log_likelihood_mle}")

    # write to text file in a known place
    path = f"./data_files/mle_parameters/{box}/{isim}/{iz}/mle_values.txt"
    outfile = Path(path)
    outfile.parent.mkdir(parents=True, exist_ok=True)

    with open(outfile, "w") as f:
        f.write(f"LOG_LIKELIHOOD={log_likelihood_mle:.13f}\n")
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
        title_kwargs={"fontsize": 12})
    for ax in fig.axes:
        ax.xaxis.set_major_formatter(FormatStrFormatter(fmt))
        ax.yaxis.set_major_formatter(FormatStrFormatter(fmt))
    # pb.savefig(f'./Plots/mcmc_corner_{test_name[0]}_{test_name[1]}_steps{steps}_walkers{nwalkers}_initialpos{initial_name[0]}_{initial_name[1]}_offset{gaussian_offset_name}.png', dpi=400)
    pb.savefig(f'./Plots/mcmc_corner_optimal_value_{box}_{isim}_{iz}_steps{steps}_walkers{nwalkers}_initialpos{initial_name[0]}_{initial_name[1]}_offset{gaussian_offset_name}.png', dpi=400)
    pb.clf()
