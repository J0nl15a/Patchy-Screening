import numpy as np, healpy as hp, pymaster as nmt, pandas as pd, pylab as pb
import yaml, sys, time, textwrap
from scipy.signal import savgol_filter
from kappa_map_gen_forJonah import kappa_map_gen_forJonah
import astropy.units as u
from pathlib import Path

job_start_time = time.time()

theta_d = np.arange(0.5, 11, 0.5)
box = sys.argv[1]
isim = sys.argv[2]
iz = sys.argv[3]
im = float(sys.argv[4])
slope = float(sys.argv[5])

lightcone = int(sys.argv[6])

if round(float(im), 1) == float(im):
    im_name = f"{float(im):.1f}".replace('.', 'p')
else:
    im_name = f"{float(im):.3f}".replace('.', 'p')

if round(float(slope), 1) == float(slope):
    slope_name = f"{float(slope):.1f}".replace('.', 'p')
else:    
    slope_name = f"{float(slope):.3f}".replace('.', 'p')
print(im, slope)

bin_setup = yaml.safe_load(open("./unWISExLens_lklh/unWISExLens_lklh/config_files/binning_setup.yaml"))
if iz == 'Blue':
    bin_edges = np.array(bin_setup["Blue_ACT"]["ell_bin_edges"])
elif iz == 'Green':
    bin_edges = np.array(bin_setup["Green_ACT"]["ell_bin_edges"])
print(bin_edges)

edges_int = np.rint(bin_edges).astype(int)  # 19.5->20, 51.5->52, ...
l0 = edges_int[:-1]
lf = edges_int[1:]
b = nmt.NmtBin.from_edges(l0, lf)
lmax_bins = b.lmax

ells = b.get_effective_ells()
ell_200_mask = np.where(ells > 200)
ell_namaster = ells[ell_200_mask]
ell_1000_mask = np.where(ell_namaster > 1000)
print(ell_namaster)

# halo_redshift_bins = np.loadtxt(f"/cosma8/data/dp004/dc-conl1/FLAMINGO/patchy_screening/data_files/halo_redshifts/{box}/{isim}/lightcone{lightcone}/FLAMINGO_halo_redshift_values.txt", usecols=(0,1,2,3))
map_redshift_bins = np.loadtxt(f"/cosma8/data/dp004/flamingo/Runs/{box}/{isim}/shell_redshifts_z3.txt", skiprows=1, usecols=(0,1), delimiter=',')

w = 5
p = 2

obs_data = np.loadtxt(f'./unWISExLens_lklh/data/v1.0/bandpowers/unWISExACT-DR6_{str(iz).lower()}_baseline_Clgg+Clkk+Clkg.dat', usecols=(0,1,2,3))[ell_200_mask, :].reshape(-1,4)
Planck_obs_data = np.loadtxt(f'./unWISExLens_lklh/data/v1.0/bandpowers/unWISExPlanck-PR4_{str(iz).lower()}_baseline_Clgg+Clkk+Clkg.dat', usecols=(0,1,2,3))

if iz == 'Blue':
    obs_nbar_sq_deg = 3409
elif iz == 'Green':
    obs_nbar_sq_deg = 1846

obs_nbar_sr = obs_nbar_sq_deg * ((180/np.pi)**2)
obs_shot_noise = 1/obs_nbar_sr #Shot-noise is 1/(source number density per steradian)

auto_path = f'./data_files/power_spectra/galaxy_galaxy/{box}/{isim}/{iz}/lightcone{lightcone}/galaxy_galaxy_power_spectrum_{im_name}'
cross_path = f'./data_files/power_spectra/kappa_galaxy/{box}/{isim}/{iz}/lightcone{lightcone}/kappa_galaxy_power_spectrum_{im_name}'

nside_cl = 2048
npix = hp.nside2npix(nside_cl)

# try:
kappa_map_no_rotation = hp.read_map(f'./data_files/kappa_maps/{box}/{isim}/lightcone{lightcone}/kappa_nonrot.fits', dtype=np.float64, verbose=False)
kappa_map_no_rotation = hp.pixelfunc.ud_grade(kappa_map_no_rotation, nside_cl)
# except FileNotFoundError:
#     # kappa_map = load_kappa_map(isim)
#     kappa_map = kappa_map_gen_forJonah(box, isim, lightcone)
if box == 'L1000N1800':
    box_alt = 'L1_m9'
elif box == 'L2800N5040':
    box_alt = 'L2p8_m9_fid'
kappa_map = hp.read_map(f'/cosma8/data/dp004/dc-yang3/maps/Jeger_rot/{box_alt}/{isim}/lightcone{lightcone}_shells/CMB_lensing_rot_Jeger_rot.fits', dtype=np.float64, verbose=False)
kappa_map = hp.pixelfunc.ud_grade(kappa_map, nside_cl)

print('Finished loading kappa map: {:.2f} seconds'.format(time.time() - job_start_time))

kappa_mask = kappa_map*0.0+1.0
f_kappa = nmt.NmtField(kappa_mask, [kappa_map], lmax=lmax_bins, n_iter=0)

kappa_mask_no_rotation = kappa_map_no_rotation*0.0+1.0
f_kappa_no_rotation = nmt.NmtField(kappa_mask_no_rotation, [kappa_map_no_rotation], lmax=lmax_bins, n_iter=0)

print('Finished preparing kappa map and NmtField: {:.2f} seconds'.format(time.time() - job_start_time))

## for 1Gpc

angles = np.array([[0. , 0. , 3.26757547 , 3.26757547 , 3.26757547 , 1.51289711, 1.51289711, 3.13885639, 3.13885639 ,3.13885639, 2.17061318, 2.17061318, 2.17061318, 2.17061318, 4.59420579, 4.59420579,4.59420579, 1.14273623, 1.14273623, 1.14273623, 1.14273623, 2.02717201, 2.02717201, 2.02717201, 2.02717201, 2.77675054, 2.77675054, 2.77675054, 2.77675054, 2.77675054, 0.83245259, 0.83245259, 0.83245259, 0.83245259, 0.83245259, 0.83245259, 4.95779263, 4.95779263, 4.95779263, 4.95779263, 4.95779263, 4.95779263, 4.95779263,2.52359739, 2.52359739, 2.52359739, 2.52359739, 2.52359739, 2.52359739, 2.52359739,2.52359739, 2.69301628, 2.69301628, 2.69301628, 2.69301628, 2.69301628, 2.69301628, 2.69301628, 2.69301628, 2.69301628], [0. , 0. , 1.41518902, 1.41518902 , 1.41518902 , 0.80580058, 0.80580058, 0.71830831, 0.71830831, 0.71830831, 1.77536892, 1.77536892,1.77536892, 1.77536892, 0.62434822, 0.62434822,0.62434822, 2.14076603, 2.14076603, 2.14076603, 2.14076603, 0.49840908, 0.49840908, 0.49840908,0.49840908, 2.0136344 , 2.0136344, 2.0136344 , 2.0136344, 2.0136344, 2.25356928 , 2.25356928, 2.25356928 , 2.25356928 , 2.25356928, 2.25356928, 1.85187078, 1.85187078, 1.85187078, 1.85187078, 1.85187078, 1.85187078 , 1.85187078, 1.36014098, 1.36014098, 1.36014098, 1.36014098, 1.36014098, 1.36014098, 1.36014098, 1.36014098, 2.35895331, 2.35895331, 2.35895331, 2.35895331, 2.35895331, 2.35895331, 2.35895331, 2.35895331, 2.35895331]])

###for 2.8 Gpc
#angles = np.array(([[0. , 0. , 0. , 0. , 0. , 0., 0. , 2.11833333, 2.11833333 , 2.11833333, 2.11833333 , 2.11833333, 2.11833333, 2.11833333 , 2.11833333 , 2.11833333 , 1.29070838, 1.29070838, 1.29070838, 1.29070838, 1.29070838, 1.29070838, 1.29070838, 1.29070838, 1.29070838, 1.29070838, 1.29070838, 5.69217656, 5.69217656, 5.69217656, 5.69217656, 5.69217656, 5.69217656, 5.69217656, 5.69217656, 5.69217656, 5.69217656,5.69217656, 5.69217656, 5.69217656, 5.69217656, 5.69217656, 5.69217656, 5.69217656, 3.79736641, 3.79736641, 3.79736641, 3.79736641, 3.79736641, 3.79736641, 3.79736641, 3.79736641, 3.79736641, 3.79736641, 3.79736641, 3.79736641, 3.79736641, 3.79736641, 3.79736641, 3.79736641, 3.79736641, 3.79736641, 1.32878635 , 1.32878635 , 1.32878635 , 1.32878635 , 1.32878635 , 1.32878635 ], [0. , 0. , 0. , 0. , 0. , 0., 0. , 0.96440001 , 0.96440001, 0.96440001 , 0.96440001 , 0.96440001,0.96440001 , 0.96440001 , 0.96440001 , 0.96440001 , 1.74841793, 1.74841793, 1.74841793, 1.74841793, 1.74841793, 1.74841793, 1.74841793, 1.74841793, 1.74841793, 1.74841793, 1.74841793, 0.56258515, 0.56258515, 0.56258515, 0.56258515, 0.56258515, 0.56258515, 0.56258515, 0.56258515, 0.56258515, 0.56258515, 0.56258515, 0.56258515, 0.56258515, 0.56258515, 0.56258515, 0.56258515, 0.56258515, 1.45462313, 1.45462313, 1.45462313, 1.45462313, 1.45462313, 1.45462313, 1.45462313, 1.45462313, 1.45462313, 1.45462313, 1.45462313, 1.45462313, 1.45462313, 1.45462313, 1.45462313, 1.45462313, 1.45462313, 1.45462313, 2.48706614, 2.48706614, 2.48706614, 2.48706614 , 2.48706614, 2.48706614]]))

print(angles.shape)

mock_catalogue = pd.read_parquet(
f"/cosma8/data/dp004/dc-conl1/FLAMINGO/patchy_screening/data_files/mock_halo_catalogs/{box}/{isim}/{iz}/lightcone{lightcone}/sampled_halo_data_{im_name}_{slope_name}.parquet"
)

dndz_file = np.loadtxt(f"/cosma8/data/dp004/dc-conl1/FLAMINGO/patchy_screening/data_files/dndz_samples/{box}/{isim}/{iz}/lightcone{lightcone}/dndz_galaxies_sampled_{im_name}_{slope_name}.txt", skiprows=1, usecols=1)

dndz_total = 0
mock_catalogue_total = 0
mock_catalogue_z_total = 0

snap_max = 77 if box == 'L1000N1800' else 78

for i in range(len(angles[0])):
    # print((angles[1, i]*(180.0/np.pi)*u.deg).to_value(u.deg), (angles[0, i]*(180.0/np.pi)*u.deg).to_value(u.deg))
    rot_custom = hp.Rotator(rot=[(angles[1, i]*(180.0/np.pi)*u.deg).to_value(u.deg), (angles[0, i]*(180.0/np.pi)*u.deg).to_value(u.deg)], inv=True)
    # print(rot_custom)

    snap_num = snap_max - i
    # zmin, zmax = halo_redshift_bins[i, 1], halo_redshift_bins[i, 3]
    zmin, zmax = map_redshift_bins[i, 0], map_redshift_bins[i, 1]

    galaxies_in_snap_z = mock_catalogue[
        mock_catalogue['z'].between(zmin, zmax, inclusive='left')
    ]
    galaxies_in_snap = mock_catalogue[mock_catalogue['SnapNum'] == snap_num]
    # galaxies_in_snap_mask = np.where(mock_catalogue['SnapNum'].to_numpy() == (77-i))[0]
    print(len(galaxies_in_snap), len(galaxies_in_snap_z))
    # galaxies_in_snap = mock_catalogue.iloc[galaxies_in_snap_mask]

    # print(dndz_file[i] - len(galaxies_in_snap))
    dndz_total += dndz_file[i]
    mock_catalogue_total += len(galaxies_in_snap)
    mock_catalogue_z_total += len(galaxies_in_snap_z)

    vec = np.zeros((len(galaxies_in_snap_z), 3))
    vec[:,0]=galaxies_in_snap_z['xminpot'].to_numpy()
    vec[:,1]=galaxies_in_snap_z['yminpot'].to_numpy()
    vec[:,2]=galaxies_in_snap_z['zminpot'].to_numpy()

    theta, phi = hp.pixelfunc.vec2ang(vec, lonlat=True)
    # print(theta, phi)

    theta_rot, phi_rot = rot_custom(theta, phi, lonlat=True)

    # print(theta_rot, phi_rot)
    if i == 0:
        source_vector = hp.ang2vec(theta_rot, phi_rot, lonlat=True)
        source_vector_no_rotation = hp.ang2vec(theta, phi, lonlat=True)
    else:
        source_vector = np.concatenate((source_vector, hp.ang2vec(theta_rot, phi_rot, lonlat=True)), axis=0)
        source_vector_no_rotation = np.concatenate((source_vector_no_rotation, hp.ang2vec(theta, phi, lonlat=True)), axis=0)
print(len(mock_catalogue['mvir']), source_vector.shape)
print(dndz_total, mock_catalogue_total, mock_catalogue_z_total)

print('Finished computing halo catalogs: {:.2f} seconds'.format(time.time() - job_start_time))

pixels = hp.pixelfunc.vec2pix(nside_cl, source_vector[:,0], source_vector[:,1], source_vector[:,2])
density_map = np.bincount(pixels, minlength=npix).astype(np.float64)
mean_density = np.mean(density_map)
galaxy_overdensity = ((density_map - mean_density) / mean_density)

pixels_no_rotation = hp.pixelfunc.vec2pix(nside_cl, source_vector_no_rotation[:,0], source_vector_no_rotation[:,1], source_vector_no_rotation[:,2])
density_map_no_rotation = np.bincount(pixels_no_rotation, minlength=npix).astype(np.float64)
mean_density_no_rotation = np.mean(density_map_no_rotation)
galaxy_overdensity_no_rotation = ((density_map_no_rotation - mean_density_no_rotation) / mean_density_no_rotation)

galaxy_mask = galaxy_overdensity*0.0+1.0
f_galaxy = nmt.NmtField(galaxy_mask, [galaxy_overdensity], lmax=lmax_bins, n_iter=0)

galaxy_mask_no_rotation = galaxy_overdensity_no_rotation*0.0+1.0
f_galaxy_no_rotation = nmt.NmtField(galaxy_mask_no_rotation, [galaxy_overdensity_no_rotation], lmax=lmax_bins, n_iter=0)


pcl_auto = nmt.compute_coupled_cell(f_galaxy, f_galaxy)

pcl_shape = (f_galaxy.nmaps * f_galaxy.nmaps, f_galaxy.ainfo.lmax+1)
clg = np.zeros(pcl_shape)
deproj_auto = nmt.deprojection_bias(f_galaxy, f_galaxy, clg)

w_auto = nmt.NmtWorkspace.from_fields(f_galaxy, f_galaxy, b)
cl_auto_namaster = w_auto.decouple_cell(pcl_auto - deproj_auto).squeeze()[ell_200_mask]

auto_spectra_shot_noise = (4*np.pi)/len(mock_catalogue['mvir']) #Shot-noise is 4pi/(total number of sources in the map)

auto_spectra = ((cl_auto_namaster - auto_spectra_shot_noise) + obs_shot_noise)*1e5
auto_spectra_unsmoothed_component = auto_spectra[np.where(ell_namaster <= 1000)]
auto_spectra_smoothed_component = savgol_filter(auto_spectra[ell_1000_mask], window_length=w, polyorder=p)
auto_spectra = np.concatenate((auto_spectra_unsmoothed_component, auto_spectra_smoothed_component))
print(auto_spectra)

pcl_auto_no_rotation = nmt.compute_coupled_cell(f_galaxy_no_rotation, f_galaxy_no_rotation)

pcl_shape_no_rotation = (f_galaxy_no_rotation.nmaps * f_galaxy_no_rotation.nmaps, f_galaxy_no_rotation.ainfo.lmax+1)
clg_no_rotation = np.zeros(pcl_shape_no_rotation)
deproj_auto_no_rotation = nmt.deprojection_bias(f_galaxy_no_rotation, f_galaxy_no_rotation, clg_no_rotation)

w_auto_no_rotation = nmt.NmtWorkspace.from_fields(f_galaxy_no_rotation, f_galaxy_no_rotation, b)
cl_auto_namaster_no_rotation = w_auto_no_rotation.decouple_cell(pcl_auto_no_rotation - deproj_auto_no_rotation).squeeze()[ell_200_mask]

auto_spectra_no_rotation = ((cl_auto_namaster_no_rotation - auto_spectra_shot_noise) + obs_shot_noise)*1e5
auto_spectra_unsmoothed_component_no_rotation = auto_spectra_no_rotation[np.where(ell_namaster <= 1000)]
auto_spectra_smoothed_component_no_rotation = savgol_filter(auto_spectra_no_rotation[ell_1000_mask], window_length=w, polyorder=p)
auto_spectra_no_rotation = np.concatenate((auto_spectra_unsmoothed_component_no_rotation, auto_spectra_smoothed_component_no_rotation))
print(auto_spectra_no_rotation)


pcl_auto_kk = nmt.compute_coupled_cell(f_kappa, f_kappa)

pcl_shape_kk = (f_kappa.nmaps * f_kappa.nmaps, f_kappa.ainfo.lmax+1)
clg_kk = np.zeros(pcl_shape_kk)
deproj_auto_kk = nmt.deprojection_bias(f_kappa, f_kappa, clg_kk)

w_auto_kk = nmt.NmtWorkspace.from_fields(f_kappa, f_kappa, b)
cl_auto_namaster_kk = w_auto_kk.decouple_cell(pcl_auto_kk - deproj_auto_kk).squeeze()[ell_200_mask]

auto_spectra_shot_noise = (4*np.pi)/len(mock_catalogue['mvir']) #Shot-noise is 4pi/(total number of sources in the map)

auto_spectra_kk = cl_auto_namaster_kk*1e5
auto_spectra_unsmoothed_component_kk = auto_spectra_kk[np.where(ell_namaster <= 1000)]
auto_spectra_smoothed_component_kk = savgol_filter(auto_spectra_kk[ell_1000_mask], window_length=w, polyorder=p)
auto_spectra_kk = np.concatenate((auto_spectra_unsmoothed_component_kk, auto_spectra_smoothed_component_kk))
print(auto_spectra_kk)

pcl_auto_no_rotation_kk = nmt.compute_coupled_cell(f_kappa_no_rotation, f_kappa_no_rotation)

pcl_shape_no_rotation_kk = (f_kappa_no_rotation.nmaps * f_kappa_no_rotation.nmaps, f_kappa_no_rotation.ainfo.lmax+1)
clg_no_rotation_kk = np.zeros(pcl_shape_no_rotation_kk)
deproj_auto_no_rotation_kk = nmt.deprojection_bias(f_kappa_no_rotation, f_kappa_no_rotation, clg_no_rotation_kk)

w_auto_no_rotation_kk = nmt.NmtWorkspace.from_fields(f_kappa_no_rotation, f_kappa_no_rotation, b)
cl_auto_namaster_no_rotation_kk = w_auto_no_rotation_kk.decouple_cell(pcl_auto_no_rotation_kk - deproj_auto_no_rotation_kk).squeeze()[ell_200_mask]

auto_spectra_no_rotation_kk = cl_auto_namaster_no_rotation_kk*1e5
auto_spectra_unsmoothed_component_no_rotation_kk = auto_spectra_no_rotation_kk[np.where(ell_namaster <= 1000)]
auto_spectra_smoothed_component_no_rotation_kk = savgol_filter(auto_spectra_no_rotation_kk[ell_1000_mask], window_length=w, polyorder=p)
auto_spectra_no_rotation_kk = np.concatenate((auto_spectra_unsmoothed_component_no_rotation_kk, auto_spectra_smoothed_component_no_rotation_kk))
print(auto_spectra_no_rotation_kk)


pcl_cross = nmt.compute_coupled_cell(f_kappa, f_galaxy)

pcl_shape = (f_kappa.nmaps * f_galaxy.nmaps, f_galaxy.ainfo.lmax+1)
clg = np.zeros(pcl_shape)
deproj_cross = nmt.deprojection_bias(f_kappa, f_galaxy, clg)

w_cross = nmt.NmtWorkspace.from_fields(f_kappa, f_galaxy, b)
cl_cross_namaster = w_cross.decouple_cell(pcl_cross - deproj_cross).squeeze()[ell_200_mask]

cross_spectra = cl_cross_namaster*1e5
cross_spectra_unsmoothed_component = cross_spectra[np.where(ell_namaster <= 1000)]
cross_spectra_smoothed_component = savgol_filter(cross_spectra[ell_1000_mask], window_length=w, polyorder=p)
cross_spectra = np.concatenate((cross_spectra_unsmoothed_component, cross_spectra_smoothed_component))
print(cross_spectra)

pcl_cross_no_rotation = nmt.compute_coupled_cell(f_kappa_no_rotation, f_galaxy_no_rotation)

pcl_shape_no_rotation = (f_kappa_no_rotation.nmaps * f_galaxy_no_rotation.nmaps, f_galaxy_no_rotation.ainfo.lmax+1)
clg_no_rotation = np.zeros(pcl_shape_no_rotation)
deproj_cross_no_rotation = nmt.deprojection_bias(f_kappa_no_rotation, f_galaxy_no_rotation, clg_no_rotation)

w_cross_no_rotation = nmt.NmtWorkspace.from_fields(f_kappa_no_rotation, f_galaxy_no_rotation, b)
cl_cross_namaster_no_rotation = w_cross_no_rotation.decouple_cell(pcl_cross_no_rotation - deproj_cross_no_rotation).squeeze()[ell_200_mask]

cross_spectra_no_rotation = cl_cross_namaster_no_rotation*1e5
cross_spectra_unsmoothed_component_no_rotation = cross_spectra_no_rotation[np.where(ell_namaster <= 1000)]
cross_spectra_smoothed_component_no_rotation = savgol_filter(cross_spectra_no_rotation[ell_1000_mask], window_length=w, polyorder=p)
cross_spectra_no_rotation = np.concatenate((cross_spectra_unsmoothed_component_no_rotation, cross_spectra_smoothed_component_no_rotation))
print(cross_spectra_no_rotation)

non_rotated_auto_spectra = np.loadtxt(f'./data_files/power_spectra/galaxy_galaxy/{box}/{isim}/{iz}/lightcone{lightcone}/galaxy_galaxy_power_spectrum_{im_name}_{slope_name}.txt', skiprows=1, usecols=(2))
non_rotated_cross_spectra = np.loadtxt(f'./data_files/power_spectra/kappa_galaxy/{box}/{isim}/{iz}/lightcone{lightcone}/kappa_galaxy_power_spectrum_{im_name}_{slope_name}.txt', skiprows=1, usecols=(1))

pb.plot(ell_namaster, auto_spectra, label='Rotated map')
pb.plot(ell_namaster, non_rotated_auto_spectra, label='Non-rotated map (from file)')
pb.plot(ell_namaster, auto_spectra_no_rotation, label='Non-rotated map (from calculation)')
pb.plot(obs_data[:,0], obs_data[:,1]*1e5, marker='.', markersize=5, linewidth=0, label='ACT x unWISE', color='k')
pb.plot(Planck_obs_data[:,0], Planck_obs_data[:,1]*1e5, marker='.', markersize=5, linewidth=0, label='Planck x unWISE', color='red')
pb.xlabel('Multipole moment $\mathrm{\ell}$')
pb.ylabel('$C^{gg}_{\mathrm{\ell}}x10^5$')
pb.xscale("log")
pb.yscale("log")
pb.xlim(200, 4000)
pb.ylim(bottom=1e-2)
pb.title(f'Galaxy-galaxy power spectrum for mock catalogue with nhalos = {len(mock_catalogue["mvir"])}')
pb.legend(fontsize=9, ncols=1, loc='upper right')
pb.savefig(f'./Plots/gg_power_spectrum_{box}_{isim}_{iz}_lightcone{lightcone}_{im_name}_{slope_name}_rotated.png', dpi=300)
pb.clf()

pb.plot(ell_namaster, cross_spectra, label='Rotated map')
pb.plot(ell_namaster, non_rotated_cross_spectra, label='Non-rotated map (from file)')
pb.plot(ell_namaster, cross_spectra_no_rotation, label='Non-rotated map (from calculation)')
pb.plot(obs_data[:,0], obs_data[:,3]*1e5, marker='.', markersize=5, linewidth=0, label='ACT x unWISE', color='k')
pb.plot(Planck_obs_data[:,0], Planck_obs_data[:,3]*1e5, marker='.', markersize=5, linewidth=0, label='Planck x unWISE', color='red')
pb.xlabel('Multipole moment $\mathrm{\ell}$')
pb.ylabel('$C^{\kappa g}_{\mathrm{\ell}}x10^5$')
pb.xscale("log")
pb.yscale("log")
pb.xlim(200, 4000)
# pb.ylim(bottom=1e-4)
pb.title(f'Kappa-galaxy cross spectrum for mock catalogue with nhalos = {len(mock_catalogue["mvir"])}')
pb.legend(fontsize=9, ncols=1, loc='upper right')
pb.savefig(f'./Plots/kg_cross_spectrum_{box}_{isim}_{iz}_lightcone{lightcone}_{im_name}_{slope_name}_rotated.png', dpi=300)
pb.clf()

pb.plot(ell_namaster, auto_spectra_kk, label='Rotated map')
pb.plot(ell_namaster, auto_spectra_no_rotation_kk, label='Non-rotated map (from file)')
# pb.plot(obs_data[:,0], obs_data[:,2]*1e5, marker='.', markersize=5, linewidth=0, label='ACT x unWISE', color='k')
# pb.plot(Planck_obs_data[:,0], Planck_obs_data[:,2]*1e5, marker='.', markersize=5, linewidth=0, label='Planck x unWISE', color='red')
pb.xlabel('Multipole moment $\mathrm{\ell}$')
pb.ylabel('$C^{\kappa\kappa}_{\mathrm{\ell}}x10^5$')
pb.xscale("log")
pb.yscale("log")
pb.xlim(200, 4000)
# pb.ylim(bottom=1e-2)
pb.title(f'Kappa-kappa power spectrum for mock catalogue with nhalos = {len(mock_catalogue["mvir"])}')
pb.legend(fontsize=9, ncols=1, loc='upper right')
pb.savefig(f'./Plots/kk_power_spectrum_{box}_{isim}_{iz}_lightcone{lightcone}_{im_name}_{slope_name}_rotated.png', dpi=300)
pb.clf()

quit()

auto_output_path = Path(auto_path+f'_{slope_name[i]}_rotated.txt')
cross_output_path = Path(cross_path+f'_{slope_name[i]}_rotated.txt')
auto_output_path.parent.mkdir(parents=True, exist_ok=True)
cross_output_path.parent.mkdir(parents=True, exist_ok=True)

auto_out = np.column_stack((ell_namaster, auto_spectra*1e5, ((auto_spectra-auto_spectra_shot_noise)+obs_shot_noise)*1e5))
cross_out = np.column_stack((ell_namaster, cross_spectra*1e5))
np.savetxt(auto_output_path, auto_out, fmt='%f %.13f %.13f', header=f"Galaxy-galaxy power spectra for mock catalog with nhalos = {len(mock_catalogue['mvir'])}", comments='')
np.savetxt(cross_output_path, cross_out, fmt='%f %.13f', header=f"Kappa-galaxy cross spectra for mock catalog with nhalos = {len(mock_catalogue['mvir'])}", comments='')