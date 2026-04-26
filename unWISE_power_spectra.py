import numpy as np, healpy as hp, pymaster as nmt
from scipy.interpolate import CubicSpline
from imp_patchy_screening import patchyScreening
import yaml, sys, time
from scipy.signal import savgol_filter
from joblib import Parallel, delayed
from unWISE_power_spectra_plot import power_spectra_plot
from kappa_map_gen_forJonah import kappa_map_gen_forJonah
from pathlib import Path

theta_d = np.arange(0.5, 11, 0.5)
ncpu = int(sys.argv[1])
box = sys.argv[2]
isim = sys.argv[3]
iz = sys.argv[4]
im = float(sys.argv[5])
slope = float(sys.argv[6])
fits = str(sys.argv[7])
sig = sys.argv[8]

covariance = sys.argv[9].lower() in ("true", "1", "yes", "y")
smooth = sys.argv[10].lower() in ("true", "1", "yes", "y")
single = sys.argv[11].lower() in ("true", "1", "yes", "y")
save = sys.argv[12].lower() in ("true", "1", "yes", "y")
plot = sys.argv[13].lower() in ("true", "1", "yes", "y")

lightcone = int(sys.argv[14])

if round(float(im), 1) == float(im):
    im_name = f"{float(im):.1f}".replace('.', 'p')
else:
    im_name = f"{float(im):.3f}".replace('.', 'p')

if single == False:
    slopes = [round(n, 2) for n in np.arange(0.0, float(slope)+0.1, 0.1)] #CHANGE BACK
elif single == True:
    slopes = [slope]
slopes_name = []
print(im, slopes)

job_start_time = time.time()

source_vectors = []
nhalos = []
mean_mstar = []
# ps = patchyScreening(isim, iz, im, slope, ncpu, theta_d, fits_file=fits, signal=sig)
# ps.get_halo_coordinates()

def compute_catalog(slope, box=box, isim=isim, iz=iz, im=im, ncpu=ncpu, theta_d=theta_d, fits=fits, sig=sig, lightcone=lightcone):
    if slope == 0.0:
        slope = abs(slope)
        
    ps = patchyScreening(box, isim, iz, im, slope, ncpu, theta_d, fits_file=fits, signal=sig, lightcone=lightcone)
    #ps_camb = patchyScreening(isim, iz, im, im_name, ncpu, theta_d, cmb_method='CAMB', signal=sig, rect_size=20)
    ps.get_halo_coordinates()
    return ps.source_vector, ps.nhalo, np.log10(np.mean(ps.merge['mstar'].to_numpy()))

if single == False:
    results = Parallel(n_jobs=ncpu, backend="loky")(delayed(compute_catalog)(slope) for slope in slopes)
    for i in range(len(results)):
        if round(slopes[i], 1) == slopes[i]:
            slopes_name.append(f"{float(slopes[i]):.1f}".replace('.', 'p'))
        else:
            slopes_name.append(f"{float(slopes[i]):.3f}".replace('.', 'p'))
        print(slope)
        source_vectors.append(results[i][0])
        nhalos.append(results[i][1])
        mean_mstar.append(results[i][2])
elif single == True:
    results = compute_catalog(slope)
    if round(slope, 1) == slope:
        slopes_name.append(f"{float(slope):.1f}".replace('.', 'p'))
    else:
        slopes_name.append(f"{float(slope):.3f}".replace('.', 'p'))
    source_vectors.append(results[0])
    nhalos.append(results[1])
    mean_mstar.append(results[2])
print(round(mean_mstar[0], 5))

print('Finished computing halo catalogs: {:.2f} seconds'.format(time.time() - job_start_time))

print(mean_mstar)

bin_setup = yaml.safe_load(open("./unWISExLens_lklh/unWISExLens_lklh/config_files/binning_setup.yaml"))
if iz == 'Blue':
    bin_edges = np.array(bin_setup["Blue_ACT"]["ell_bin_edges"])
elif iz == 'Green':
    bin_edges = np.array(bin_setup["Green_ACT"]["ell_bin_edges"])
print(bin_edges)
#ell_edges = bin_edges[np.where(bin_edges > 200)]
#print(ell_edges)

edges_int = np.rint(bin_edges).astype(int)  # 19.5->20, 51.5->52, ...
l0 = edges_int[:-1]
lf = edges_int[1:]
b = nmt.NmtBin.from_edges(l0, lf)
lmax_bins = b.lmax

ells = b.get_effective_ells()
ell_200_mask = np.where(ells > 200)
ell_namaster = ells[ell_200_mask]
print(ell_namaster)

# wide_bins = bin_edges #np.linspace(4000, np.min(bin_edges), num=30, endpoint=True)[::-1]
# print(wide_bins)

# l0_wide = np.ceil(wide_bins[:-1]).astype(int)
# lf_wide = np.floor(wide_bins[1:]).astype(int)
# b_wide = nmt.NmtBin.from_edges(l0_wide, lf_wide)
# lmax_bins_wide = b_wide.lmax

# ell_namaster_wide = b_wide.get_effective_ells()
# #ell_200_mask_wide = np.where(ell_namaster_wide > 200)
# #ell_namaster_wide = ell_namaster_wide[ell_200_mask_wide]
# print(ell_namaster_wide)

#print(ell_edges)

auto_path = f'./data_files/power_spectra/galaxy_galaxy/{box}/{isim}/{iz}/lightcone{lightcone}/galaxy_galaxy_power_spectrum_{im_name}'
cross_path = f'./data_files/power_spectra/kappa_galaxy/{box}/{isim}/{iz}/lightcone{lightcone}/kappa_galaxy_power_spectrum_{im_name}'


obs_data = np.loadtxt(f'./unWISExLens_lklh/data/v1.0/bandpowers/unWISExACT-DR6_{str(iz).lower()}_baseline_Clgg+Clkk+Clkg.dat', usecols=(0,1,2,3))[ell_200_mask, :].reshape(-1,4)
# obs_clkk_data = np.loadtxt(f'./unWISExLens_lklh/data/v1.0/bandpowers/clkk_bandpowers_act.txt', skiprows=2)
print(obs_data.shape)

Planck_obs_data = np.loadtxt(f'./unWISExLens_lklh/data/v1.0/bandpowers/unWISExPlanck-PR4_{str(iz).lower()}_baseline_Clgg+Clkk+Clkg.dat', usecols=(0,1,2,3))#[ell_200_mask, :]

if iz == 'Blue':
    obs_nbar_sq_deg = 3409
elif iz == 'Green':
    obs_nbar_sq_deg = 1846

obs_nbar_sr = obs_nbar_sq_deg * ((180/np.pi)**2)

print(obs_nbar_sr)

obs_shot_noise = 1/obs_nbar_sr #Shot-noise is 1/(source number density per steradian)
print(obs_shot_noise)

##on unit sphere---might not be necessary, but a standard way
# Create an empty HEALPix map
# This creates a map with all pixels initialized to zero
nside_cl = 2048
npix = hp.nside2npix(nside_cl)

try:
    kappa_map = hp.read_map(f'./data_files/kappa_maps/{box}/{isim}/lightcone{lightcone}/kappa_nonrot.fits', dtype=np.float64, verbose=False)
except FileNotFoundError:
    # kappa_map = load_kappa_map(isim)
    kappa_map = kappa_map_gen_forJonah(box, isim, lightcone)
kappa_map = hp.pixelfunc.ud_grade(kappa_map, nside_cl)

print('Finished loading kappa map: {:.2f} seconds'.format(time.time() - job_start_time))

## choose ell binning - here we include 10 modes per ell bin
#b = nmt.NmtBin.from_nside_linear(nside_cl, 10)
#ell_namaster = b.get_effective_ells()

obs_data_covariance = np.loadtxt(f'./unWISExLens_lklh/data/v1.0/covariances/covmat_Clgg+Clkg_unWISExACT-DR6_{str(iz).lower()}_baseline.dat')
#obs_data_covariance = np.loadtxt(f'./unWISExLens_lklh/data/v1.0/covariances/covmat_Clgg+Clkg_unWISExACT-DR6_{str(iz).lower()}_cmbmarg.dat')
print(obs_data_covariance.shape)

obs_data_variance = np.diag(obs_data_covariance)
obs_data_std = np.sqrt(obs_data_variance[:int(len(obs_data_variance)/2)])[ell_200_mask]
obs_data_std_cross = np.sqrt(obs_data_variance[int(len(obs_data_variance)/2):])[ell_200_mask]

Planck_obs_data_covariance = np.loadtxt(f'./unWISExLens_lklh/data/v1.0/covariances/covmat_Clgg+Clkg_unWISExPlanck-PR4_{str(iz).lower()}_baseline.dat')
print(Planck_obs_data_covariance.shape)
Planck_obs_data_variance = np.diag(Planck_obs_data_covariance)
Planck_obs_data_std = np.sqrt(Planck_obs_data_variance[:int(len(Planck_obs_data_variance)/2)])
Planck_obs_data_std_cross = np.sqrt(Planck_obs_data_variance[int(len(Planck_obs_data_variance)/2):])


auto_spectra_list = []
cross_spectra_list = []
chi2_list = []
auto_spectra_shot_noise_list = []
auto_spectra_covariance_list = []
cross_spectra_covariance_list = []

anafast_list = []

kappa_mask = kappa_map*0.0+1.0
f_kappa = nmt.NmtField(kappa_mask, [kappa_map], lmax=lmax_bins, n_iter=0)
# f_kappa_wide = nmt.NmtField(kappa_mask, [kappa_map], lmax=lmax_bins_wide, n_iter=0)

print('Finished preparing kappa map and NmtField: {:.2f} seconds'.format(time.time() - job_start_time))

for i in range(len(nhalos)):
    pixels = hp.pixelfunc.vec2pix(nside_cl, source_vectors[i][:,0], source_vectors[i][:,1], source_vectors[i][:,2])
    density_map = np.bincount(pixels, minlength=npix).astype(np.float64)
    mean_density = np.mean(density_map)
    galaxy_overdensity = ((density_map - mean_density) / mean_density)

    galaxy_mask = galaxy_overdensity*0.0+1.0
    f_galaxy = nmt.NmtField(galaxy_mask, [galaxy_overdensity], lmax=lmax_bins, n_iter=0)
    # f_galaxy_wide = nmt.NmtField(galaxy_mask, [galaxy_overdensity], lmax=lmax_bins_wide, n_iter=0)


    pcl_auto = nmt.compute_coupled_cell(f_galaxy, f_galaxy)

    pcl_shape = (f_galaxy.nmaps * f_galaxy.nmaps, f_galaxy.ainfo.lmax+1)
    clg = np.zeros(pcl_shape)
    deproj_auto = nmt.deprojection_bias(f_galaxy, f_galaxy, clg)
    
    w_auto = nmt.NmtWorkspace.from_fields(f_galaxy, f_galaxy, b)
    cl_auto_namaster = w_auto.decouple_cell(pcl_auto - deproj_auto).squeeze()[ell_200_mask]
    #cl_auto = nmt.compute_full_master(f_galaxy, f_galaxy, b)
    #cl_auto_namaster = np.squeeze(cl_auto)
    #auto_spectra_list.append(cl_auto_namaster)
    if smooth:
        auto_spectra_list.append(savgol_filter(cl_auto_namaster, window_length=10, polyorder=5))
    elif not smooth:
        auto_spectra_list.append(cl_auto_namaster)

    if covariance:
        cl_th = np.zeros(lmax_bins+1)
        for k in range(len(cl_auto_namaster)):
            cl_th[l0[k]:lf[k]+1] = cl_auto_namaster[k]

        cw_auto = nmt.NmtCovarianceWorkspace.from_fields(f_galaxy, f_galaxy)
        cw_auto.compute_coupling_coefficients(f_galaxy, f_galaxy)
        print(b)
        print(lmax_bins)
        print(cl_auto_namaster.shape)
        #cov_auto = nmt.gaussian_covariance(cw_auto, 0, 0, 0, 0, [cl_auto_namaster], [cl_auto_namaster], [cl_auto_namaster], [cl_auto_namaster], w_auto)
        cov_auto = nmt.gaussian_covariance(cw_auto, 0, 0, 0, 0, [cl_th], [cl_th], [cl_th], [cl_th], w_auto)[(len(bin_edges)-len(ell_namaster)-1):, (len(bin_edges)-len(ell_namaster)-1):]
        print(cov_auto.shape)
        #quit()
        auto_spectra_covariance_list.append(cov_auto)


    # pcl_cross = nmt.compute_coupled_cell(f_kappa_wide, f_galaxy_wide)

    # pcl_shape = (f_kappa_wide.nmaps * f_galaxy_wide.nmaps, f_galaxy_wide.ainfo.lmax+1)
    # clg = np.zeros(pcl_shape)
    # deproj_cross = nmt.deprojection_bias(f_kappa_wide, f_galaxy_wide, clg)

    # w_cross = nmt.NmtWorkspace.from_fields(f_kappa_wide, f_galaxy_wide, b_wide)
    # cl_cross_namaster_wide = w_cross.decouple_cell(pcl_cross - deproj_cross).squeeze()#[ell_200_mask]
    # h = CubicSpline(ell_namaster_wide, cl_cross_namaster_wide)
    # cl_cross_namaster = h(ells)[ell_200_mask]

    print("Mean real-space correlation:", np.mean(kappa_map[kappa_mask>0] * [galaxy_overdensity[galaxy_mask>0]]))

    print("After flipping kappa:", np.mean((-kappa_map)[kappa_mask>0] * [galaxy_overdensity[galaxy_mask>0]]))


    pcl_cross = nmt.compute_coupled_cell(f_kappa, f_galaxy)

    pcl_shape = (f_kappa.nmaps * f_galaxy.nmaps, f_galaxy.ainfo.lmax+1)
    clg = np.zeros(pcl_shape)
    deproj_cross = nmt.deprojection_bias(f_kappa, f_galaxy, clg)

    w_cross = nmt.NmtWorkspace.from_fields(f_kappa, f_galaxy, b)
    cl_cross_namaster = w_cross.decouple_cell(pcl_cross - deproj_cross).squeeze()[ell_200_mask]

    if smooth:
        cross_spectra_list.append(savgol_filter(cl_cross_namaster, window_length=10, polyorder=5))
    elif not smooth:
        cross_spectra_list.append(cl_cross_namaster)

    if covariance:
        w_cross = nmt.NmtWorkspace.from_fields(f_kappa, f_galaxy, b)
        cl_th = np.zeros(lmax_bins+1)
        for k in range(len(cl_cross_namaster)):
            cl_th[l0[k]:lf[k]+1] = cl_cross_namaster[k]

        cw_cross = nmt.NmtCovarianceWorkspace.from_fields(f_kappa, f_galaxy)
        cw_cross.compute_coupling_coefficients(f_kappa, f_galaxy)
        #cov_cross = nmt.gaussian_covariance(cw_cross, 0, 0, 0, 0, [cl_cross_namaster], [cl_cross_namaster], [cl_cross_namaster], [cl_cross_namaster], w_cross)
        cov_cross = nmt.gaussian_covariance(cw_cross, 0, 0, 0, 0, [cl_th], [cl_th], [cl_th], [cl_th], w_cross)[(len(bin_edges)-len(ell_namaster)-1):, (len(bin_edges)-len(ell_namaster)-1):]
        cross_spectra_covariance_list.append(cov_cross)
        print(cov_cross.shape)
        
    #f_2 = nmt.NmtField(kappa_mask, [galaxy_overdensity], lmax=lmax_bins)
    #cl_cross = nmt.compute_full_master(f_kappa, f_galaxy, b)
    #cl_cross_namaster = np.squeeze(cl_cross)
    #cross.append(cl_cross_namaster)
    #cross.append(savgol_filter(cl_cross_namaster, window_length=7, polyorder=5))

    auto_spectra_shot_noise_list.append((4*np.pi)/nhalos[i])
    print((4*np.pi)/nhalos[i])

    chi2_list.append(np.sum(((((cl_auto_namaster-((4*np.pi)/nhalos[i]))+obs_shot_noise)-obs_data[:,1])/obs_data_std)**2))
    print(chi2_list[i])

    #anafast_cross = hp.anafast(kappa_map, galaxy_overdensity)
    #anafast_list.append(anafast_cross)

    if save == False:
        pass
    elif save == True:
        auto_output_path = Path(auto_path+f'_{slopes_name[i]}.txt')
        cross_output_path = Path(cross_path+f'_{slopes_name[i]}.txt')
        auto_output_path.parent.mkdir(parents=True, exist_ok=True)
        cross_output_path.parent.mkdir(parents=True, exist_ok=True)
        if covariance:
            auto_out = np.column_stack((ell_namaster, auto_spectra_list[i]*1e5, ((auto_spectra_list[i]-auto_spectra_shot_noise_list[i])+obs_shot_noise)*1e5, np.sqrt(auto_spectra_covariance_list[i].diagonal())*1e5))
            cross_out = np.column_stack((ell_namaster, cross_spectra_list[i]*1e5, np.sqrt(cross_spectra_covariance_list[i].diagonal())*1e5))
            np.savetxt(auto_output_path, auto_out, fmt='%f %.13f %.13f %.13f', header=f"Galaxy-galaxy power spectra for mock catalog with nhalos = {nhalos[i]}", comments='')
            np.savetxt(cross_output_path, cross_out, fmt='%f %.13f %.13f', header=f"Kappa-galaxy cross spectra for mock catalog with nhalos = {nhalos[i]}", comments='')
        elif not covariance:
            auto_out = np.column_stack((ell_namaster, auto_spectra_list[i]*1e5, ((auto_spectra_list[i]-auto_spectra_shot_noise_list[i])+obs_shot_noise)*1e5))
            cross_out = np.column_stack((ell_namaster, cross_spectra_list[i]*1e5))
            np.savetxt(auto_output_path, auto_out, fmt='%f %.13f %.13f', header=f"Galaxy-galaxy power spectra for mock catalog with nhalos = {nhalos[i]}", comments='')
            np.savetxt(cross_output_path, cross_out, fmt='%f %.13f', header=f"Kappa-galaxy cross spectra for mock catalog with nhalos = {nhalos[i]}", comments='')

print('Finished computing power spectra and saving results: {:.2f} seconds'.format(time.time() - job_start_time))


if plot:

    slopes = slopes[::-1]
    nhalos = nhalos[::-1]
    mean_mstar = mean_mstar[::-1]
    auto_spectra_list = auto_spectra_list[::-1]
    cross_spectra_list = cross_spectra_list[::-1]
    auto_spectra_shot_noise_list = auto_spectra_shot_noise_list[::-1]
    auto_spectra_covariance_list = auto_spectra_covariance_list[::-1]
    cross_spectra_covariance_list = cross_spectra_covariance_list[::-1]
    chi2_list = chi2_list[::-1]
    anafast_list = anafast_list[::-1]

    plotting_data = {'slopes': slopes,
                    'nhalos': nhalos,
                    'mean_mstar': mean_mstar,
                    'auto_spectra_list': auto_spectra_list,
                    'cross_spectra_list': cross_spectra_list,
                    'auto_spectra_shot_noise_list': auto_spectra_shot_noise_list,
                    'auto_spectra_covariance_list': auto_spectra_covariance_list,
                    'cross_spectra_covariance_list': cross_spectra_covariance_list,
                    'chi2_list': chi2_list,
                    'anafast_list': anafast_list}

    if covariance:
        print(auto_spectra_covariance_list)
        print(auto_spectra_covariance_list[0])
        print(auto_spectra_covariance_list[0].diagonal())
        print(np.sqrt(auto_spectra_covariance_list[0].diagonal()))
        print(np.sqrt(auto_spectra_covariance_list[0].diagonal())*1e5)
        print(auto_spectra_list[0])
        print((auto_spectra_list[0]+np.sqrt(auto_spectra_covariance_list[0].diagonal()))*1e5)

    power_spectra_plot(box, isim, iz, im, slope, fits, single, covariance=covariance, data=plotting_data)

####################################################################################################################################
