import sys, yaml, textwrap
import numpy as np, healpy as hp, pymaster as nmt, pylab as pb
from kappa_map_gen_forJonah import kappa_map_gen_forJonah
from scipy.signal import savgol_filter

box = str(sys.argv[1])
isim = sys.argv[2]
iz = sys.argv[3]
im = float(sys.argv[4])
slope = float(sys.argv[5])
lightcone = int(sys.argv[8])

smooth = sys.argv[6].lower() in ("true", "1", "yes", "y")
save = sys.argv[7].lower() in ("true", "1", "yes", "y")

if round(float(im), 1) == float(im):
    im_name = f"{float(im):.1f}".replace('.', 'p')
else:
    im_name = f"{float(im):.3f}".replace('.', 'p')

if round(slope, 1) == slope:
    slope_name = f"{float(slope):.1f}".replace('.', 'p')
else:
    slope_name = f"{float(slope):.3f}".replace('.', 'p')

nside_cl = 2048

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

try:
    kappa_map = hp.read_map(f'./data_files/kappa_maps/{box}/{isim}/lightcone{lightcone}/kappa_rot.fits', dtype=np.float64, verbose=False)
except FileNotFoundError:
    kappa_map = kappa_map_gen_forJonah(box, isim, lightcone)
kappa_map = hp.pixelfunc.ud_grade(kappa_map, nside_cl)

if isim in ['HYDRO_FIDUCIAL', 'HYDRO_LOW_SIGMA8', 'HYDRO_STRONGEST_AGN', 'HYDRO_STRONG_SUPERNOVA'] and box == 'L1000N1800':
    Tianyi_map = True
    Tianyi_box = 'L1_m9'
elif box == 'L2800N5040':
    Tianyi_map = True
    Tianyi_box = 'L2p8_m9_fid'
else:
    Tianyi_map = False
    

if Tianyi_map:
    if Tianyi_box == 'L1_m9':
        kappa_map_Tianyi = hp.read_map(f'/cosma8/data/dp004/dc-yang3/maps/Jeger_rot/{Tianyi_box}/{isim}/lightcone{lightcone}_shells/CMB_lensing_rot_Jeger_rot.fits', 
                                       dtype=np.float64, verbose=False)
    elif Tianyi_box == 'L2p8_m9_fid':
        kappa_map_Tianyi = hp.read_map(f'/cosma8/data/dp004/dc-yang3/maps/Jeger_rot/{Tianyi_box}/lightcone{lightcone}_shells/CMB_lensing_rot_Jeger_rot.fits', 
                                       dtype=np.float64, verbose=False)
    kappa_map_Tianyi = hp.pixelfunc.ud_grade(kappa_map_Tianyi, nside_cl)

auto_spectra_list = []

kappa_mask = kappa_map*0.0+1.0
f_kappa = nmt.NmtField(kappa_mask, [kappa_map], lmax=lmax_bins, n_iter=0)

pcl_auto = nmt.compute_coupled_cell(f_kappa, f_kappa)

pcl_shape = (f_kappa.nmaps * f_kappa.nmaps, f_kappa.ainfo.lmax+1)
clg = np.zeros(pcl_shape)
deproj_auto = nmt.deprojection_bias(f_kappa, f_kappa, clg)

w_auto = nmt.NmtWorkspace.from_fields(f_kappa, f_kappa, b)
cl_auto_namaster = w_auto.decouple_cell(pcl_auto - deproj_auto).squeeze()[ell_200_mask]
print(cl_auto_namaster)

if smooth:
    auto_spectra_list.append(savgol_filter(cl_auto_namaster, window_length=10, polyorder=5))
elif not smooth:
    auto_spectra_list.append(cl_auto_namaster)


if Tianyi_map:
    kappa_mask_Tianyi = kappa_map_Tianyi*0.0+1.0
    f_kappa_Tianyi = nmt.NmtField(kappa_mask_Tianyi, [kappa_map_Tianyi], lmax=lmax_bins, n_iter=0)

    pcl_auto = nmt.compute_coupled_cell(f_kappa_Tianyi, f_kappa_Tianyi)

    pcl_shape = (f_kappa_Tianyi.nmaps * f_kappa_Tianyi.nmaps, f_kappa_Tianyi.ainfo.lmax+1)
    clg = np.zeros(pcl_shape)
    deproj_auto = nmt.deprojection_bias(f_kappa_Tianyi, f_kappa_Tianyi, clg)

    w_auto = nmt.NmtWorkspace.from_fields(f_kappa_Tianyi, f_kappa_Tianyi, b)
    cl_auto_namaster_Tianyi = w_auto.decouple_cell(pcl_auto - deproj_auto).squeeze()[ell_200_mask]

    if smooth:
        auto_spectra_list.append(savgol_filter(cl_auto_namaster_Tianyi, window_length=10, polyorder=5))
    elif not smooth:
        auto_spectra_list.append(cl_auto_namaster_Tianyi)


auto_output_path = f'./data_files/power_spectra/kappa_kappa_{isim}_{iz}_{im_name}'
if save == False:
    pass
elif save == True:
    auto_out = np.column_stack((ell_namaster, auto_spectra_list[0]*1e5, auto_spectra_list[0]*1e5))
    np.savetxt(auto_output_path+f'_{slope_name}.txt', auto_out, fmt='%f %.13f %.13f', header=f"kappa-kappa power spectra for mock catalog", comments='')

ims = [im]
slopes = [slope]
ims_name = [im_name]
slopes_name = [slope_name]
isim_names = [isim]
# variable_list = [slope]
variable_list = auto_spectra_list
# variable_name_list = [slope_name]
if Tianyi_map:
    variable_name_list = ["My rotated map", "Tianyi's rotated map"]
else: 
    variable_name_list = ["My rotated map"]

# obs_data = np.loadtxt(f'./unWISExLens_lklh/data/v1.0/bandpowers/unWISExACT-DR6_{str(iz).lower()}_baseline_Clgg+Clkk+Clkg.dat', usecols=(0,1,2,3))[ell_200_mask, :].reshape(-1,4)
# Planck_obs_data = np.loadtxt(f'./unWISExLens_lklh/data/v1.0/bandpowers/unWISExPlanck-PR4_{str(iz).lower()}_baseline_Clgg+Clkk+Clkg.dat', usecols=(0,1,2,3))
# Planck_ell_mask = np.where(Planck_obs_data[:,0] > 200)[0]
# Planck_obs_data = Planck_obs_data[Planck_ell_mask, :].reshape(-1,4)

# obs_data_covariance = np.loadtxt(f'./unWISExLens_lklh/data/v1.0/covariances/covmat_ACT-DR6_Clkk_baseline.txt')
# print(obs_data_covariance.shape)
# obs_data_variance = np.diag(obs_data_covariance)
# obs_data_std = np.sqrt(obs_data_variance[:int(len(obs_data_variance)/2)])[ell_200_mask]
# obs_data_std_cross = np.sqrt(obs_data_variance[int(len(obs_data_variance)/2):])[ell_200_mask]

# Planck_obs_data_covariance = np.loadtxt(f'./unWISExLens_lklh/data/v1.0/covariances/covmat_Planck-PR4_Clkk_baseline.txt')
# Planck_obs_data_variance = np.diag(Planck_obs_data_covariance)
# Planck_obs_data_std = np.sqrt(Planck_obs_data_variance[:int(len(Planck_obs_data_variance)/2)])[Planck_ell_mask]
# Planck_obs_data_std_cross = np.sqrt(Planck_obs_data_variance[int(len(Planck_obs_data_variance)/2):])[Planck_ell_mask]

fig, ax = pb.subplots(
1, 1, figsize=(8, 7.2),
sharex=True,
# gridspec_kw={"height_ratios": [3.0, 1.0], "hspace": 0.05},
)

# pb.figure(figsize=(8,6))
for i, var in enumerate(variable_list):

    line, = ax.plot(ell_namaster, auto_spectra_list[i]*1e5, 
                    linestyle='solid', alpha=0.8, 
                    label=f'{variable_name_list[i]}')

# ax.plot(obs_data[:,0], obs_data[:,2]*1e5, 
#         color='k', marker='.', markersize=5, linewidth=0, label='ACT x unWISE (Farren et al. 2023)')
# ax.fill_between(x=obs_data[:,0], 
#                 y1=(obs_data[:,2]+obs_data_std)*1e5, y2=(obs_data[:,2]-obs_data_std)*1e5, 
#                 color='k', linewidth=0, alpha=.3)


# ax.plot(Planck_obs_data[:,0], Planck_obs_data[:,2]*1e5, 
#         color='r', marker='.', markersize=5, linewidth=0, label='Planck x unWISE (Farren et al. 2023)')
# ax.fill_between(x=Planck_obs_data[:,0], 
#                 y1=(Planck_obs_data[:,2]+Planck_obs_data_std)*1e5, y2=(Planck_obs_data[:,2]-Planck_obs_data_std)*1e5, 
#                 color='r', linewidth=0, alpha=.3)


ax.set_ylabel(r'$C^{\kappa\kappa}_{\ell}x10^5$')
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(200, 4000)

ax.set_title("\n".join(textwrap.wrap(
    rf"Power Spectrum of the kappa map (shot-noise subtracted) "
    rf"(sim={isim}, {iz} sample, log$M_*$={im}, "
    rf"primary CMB=unlensed)",
    width=80)))

ax.legend(title="Slope", fontsize=8, ncols=1, loc='upper right')
pb.savefig(f'./Plots/kk_power_spectrum_{box}_{isim}_{iz}_{im_name}_{slope_name}_lc{lightcone}_rot.png', dpi=400)

pb.close(fig)


