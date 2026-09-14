import sys, os, pickle
import h5py, pandas as pd, numpy as np
import healpy as hp, matplotlib.pyplot as plt
import astropy.units as u
from astropy.cosmology import FlatLambdaCDM
from joblib import Parallel, delayed
import camb
import time
from datetime import datetime
from numbers import Real
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, wait
from mpi4py import MPI


def name_float(x, mle=False):
    if mle:
        return f"{float(x):.3f}".replace(".", "p")
    elif not mle:
        return f"{float(x):.1f}".replace(".", "p")

class patchyScreening:
    def __init__(self, box, isim, iz, im, n_cut, ncpu, theta_d=np.arange(0.5, 11, 0.5), nside=8192, cmb_method='FITS', fits_file='unlensed', lightcone_method=('FULL','dndz'), 
                 signal=True, rotate=False, mle=True, rect_size=20, lightcone=0, cleanup_tmp=False, run_id=None, test_name=None):

        self.job_start_time = time.time()
        box_list = ['L1000N1800', 'L1000N3600', 'L2800N5040']
        sim_list = ['HYDRO_FIDUCIAL','HYDRO_PLANCK','HYDRO_PLANCK_LARGE_NU_FIXED','HYDRO_PLANCK_LARGE_NU_VARY','HYDRO_STRONG_AGN','HYDRO_WEAK_AGN','HYDRO_LOW_SIGMA8','HYDRO_STRONGER_AGN','HYDRO_JETS_published','HYDRO_STRONGEST_AGN','HYDRO_STRONG_SUPERNOVA','HYDRO_STRONG_JETS_published','HYDRO_LOW_SIGMA8_STRONGEST_AGN']

        self.test_name = test_name
        self.cleanup_tmp = cleanup_tmp

        if run_id is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            jobid = os.environ.get("SLURM_JOB_ID", "nojob")
            self.run_id = f"{timestamp}_job{jobid}"
        else:
            self.run_id = str(run_id)

        try:
            box = int(box)
            self.boxname = box_list[box]
        except (ValueError, IndexError):
            self.boxname = str(box)

        try:
            isim = int(isim)
            self.simname = sim_list[isim]                
        except (ValueError, IndexError):
            self.simname = str(isim)
        
        survey = {'Blue':11, 'Green':22, 'Red':30}
        try:
            isinstance(iz,Real)
            self.z_sample = int(iz)
            self.z_sample_name = f'Custom_shell_{iz}'
        except (ValueError, IndexError):
            self.z_sample = int(survey[iz])
            self.z_sample_name = iz

        self.ncpu = int(ncpu)
        self.theta_d = theta_d
        self.nside = nside

        if self.boxname == 'L2800N5040' and self.nside > 4096:
            self.nside = 4096
            print("Setting nside to 4096 for L2800N5040.")

        self.pix_res = hp.pixelfunc.nside2resol(nside, arcmin=True)
        self.npix_cutout = int(20.0/self.pix_res)
        self.rtheta = self.make_cutout_radius_grid()
        # self.rtheta = np.zeros((self.npix_cutout, self.npix_cutout))

        # for ix in range(self.npix_cutout):
        #     for iy in range(self.npix_cutout):
        #         self.rtheta[ix, iy] = self.pix_res*np.sqrt((ix - int(self.npix_cutout/2.0))**2 + (iy - int(self.npix_cutout/2.0))**2)
        
        self.cmb_method = cmb_method
        self.fits_file = str(fits_file)
        self.lightcone_method = lightcone_method

        if isinstance(signal, bool):
            self.signal = signal
        else:
            self.signal = signal.lower() in ("true", "1", "yes", "y")

        self.rotate = rotate
        self.rect_size = rect_size
        self.lightcone = lightcone

        self.cosmology = FlatLambdaCDM(H0=68.1, Om0=0.3, Tcmb0=2.725)
        self.mock_CMB_primary = None

        self.mle = mle
        if self.mle == True:
            mle_cut_file = f"/cosma8/data/dp004/dc-conl1/FLAMINGO/patchy_screening/data_files/mle_parameters/{self.boxname}/{self.simname}/{self.z_sample_name}/lightcone{self.lightcone}/mle_values.txt"
            self.im = np.loadtxt(mle_cut_file, usecols=1, skiprows=6, max_rows=1, delimiter='=')
            self.slope = np.loadtxt(mle_cut_file, usecols=1, skiprows=7, max_rows=1, delimiter='=')
            self.im_name = name_float(float(self.im), mle=self.mle)
            self.slope_name = name_float(float(self.slope), mle=self.mle)
        else:
            if float(im) == 0.0:
                self.im = 0.0
            else:
                self.im = 10**np.array(float(im))
            
            self.slope = np.array(float(n_cut))
            self.im_name = name_float(float(im), mle=self.mle)
            self.slope_name = name_float(float(n_cut), mle=self.mle)


    def make_cutout_radius_grid(self):
        pix = np.arange(self.npix_cutout)
        x = (pix - (self.npix_cutout - 1) / 2.0) * self.pix_res
        y = (pix - (self.npix_cutout - 1) / 2.0) * self.pix_res
        xx, yy = np.meshgrid(x, y, indexing="ij")
        return np.sqrt(xx**2 + yy**2)

    
    def generate_cmb_map(self, plot=False):
                
        # Generating primary CMB map with CAMB or loading pre-generated FITS
        np.random.seed(1000)

        lensed_dir = f'/cosma8/data/dp004/dc-yang3/maps/{self.boxname}/{self.simname}/lightcone0_shells/patchy_screening_folder'

        A_s_values = {'HYDRO_FIDUCIAL':                 2.099e-9,
                      'HYDRO_JETS_published':           2.099e-9,
                      'HYDRO_STRONG_JETS_published':    2.099e-9,
                      'HYDRO_STRONG_SUPERNOVA':         2.099e-9,
                      'HYDRO_STRONG_AGN':               2.099e-9,
                      'HYDRO_STRONGER_AGN':             2.099e-9,
                      'HYDRO_STRONGEST_AGN':            2.099e-9,
                      'HYDRO_WEAK_AGN':                 2.099e-9,
                      'HYDRO_PLANCK':                   2.101e-9,
                      'HYDRO_PLANCK_LARGE_NU_FIXED':    2.101e-9,
                      'HYDRO_PLANCK_LARGE_NU_VARY':     2.109e-9,
                      'HYDRO_LOW_SIGMA8':               1.836e-9,
                      'HYDRO_LOW_SIGMA8_STRONGEST_AGN': 1.836e-9}
        
        n_s_values = {'HYDRO_FIDUCIAL':                 0.967,
                      'HYDRO_JETS_published':           0.967,
                      'HYDRO_STRONG_JETS_published':    0.967,
                      'HYDRO_STRONG_SUPERNOVA':         0.967,
                      'HYDRO_STRONG_AGN':               0.967,
                      'HYDRO_STRONGER_AGN':             0.967,
                      'HYDRO_STRONGEST_AGN':            0.967,
                      'HYDRO_WEAK_AGN':                 0.967,
                      'HYDRO_PLANCK':                   0.966,
                      'HYDRO_PLANCK_LARGE_NU_FIXED':    0.966,
                      'HYDRO_PLANCK_LARGE_NU_VARY':     0.968,
                      'HYDRO_LOW_SIGMA8':               0.965,
                      'HYDRO_LOW_SIGMA8_STRONGEST_AGN': 0.965}

        # if self.cmb_method == 'CAMB':
        #     pars = camb.set_params(H0=68.1, ombh2=0.048600*(0.681**2), omch2=0.256011*(0.681**2), mnu=0.06, As=2.099e-9, ns=0.967, lmax=3*self.nside+10)
        #     results = camb.get_results(pars)
        #     powers = results.get_cmb_power_spectra(pars, raw_cl=True, CMB_unit='muK', lmax=3*self.nside)
        #     unlensed_total_CL = powers['unlensed_total']
        #     print(len(unlensed_total_CL))
        #     self.mock_CMB_primary = hp.synfast(unlensed_total_CL[:,0], nside=self.nside, lmax=5024, mmax=5024) # order of CMB modes: TT, EE, BB, TE
        #     #self.mock_CMB_primary = hp.synfast(unlensed_scalar_CL[:,1], nside=self.nside)
            
        # elif self.cmb_method == 'FITS':
        #     lensed_dir = f'/cosma8/data/dp004/dc-yang3/maps/{self.boxname}/{self.simname}/lightcone0_shells/patchy_screening_folder'
        #     if self.fits_file == 'unlensed':
        #         self.mock_CMB_primary = hp.read_map(f'{lensed_dir}/CMB_T_map_unl.fits', dtype=np.float64, verbose=False)
        #     elif self.fits_file == 'lensed_z2':
        #         self.mock_CMB_primary = hp.read_map(f'{lensed_dir}/CMB_T_map_l_kappa_z2.fits', dtype=np.float64, verbose=False)
        #     elif self.fits_file == 'lensed_z3':
        #         self.mock_CMB_primary = hp.read_map(f'{lensed_dir}/CMB_T_map_l_kappa_z3.fits', dtype=np.float64, verbose=False)

        if self.fits_file == 'unlensed':
            if self.boxname == 'L2800N5040':
                snap_dir = 'snapshots_downsampled'
            else:
                snap_dir = 'snapshots'

            try:
                cosmo_info_file = f'/cosma8/data/dp004/flamingo/Runs/{self.boxname}/{self.simname}/{snap_dir}/flamingo_0009/flamingo_0009.0.hdf5'
                cosmo_info=h5py.File(cosmo_info_file,'r')
            except FileNotFoundError:
                cosmo_info_file = f'/cosma8/data/dp004/flamingo/Runs/{self.boxname}/{self.simname}/{snap_dir}/flamingo_0010/flamingo_0010.0.hdf5'
                cosmo_info=h5py.File(cosmo_info_file,'r')

            h = cosmo_info['Cosmology'].attrs['h']
            H0 = h*100.0
            Oc0h2 = cosmo_info['Cosmology'].attrs['Omega_cdm']*(h**2)
            Ob0h2=cosmo_info['Cosmology'].attrs['Omega_b']*(h**2)
            A_s = A_s_values[self.simname]
            n_s = n_s_values[self.simname]

            lmax_unlensCl = 3*self.nside #(nside = 4096)
            print(lmax_unlensCl)

            pars = camb.set_params(H0=H0, ombh2=Ob0h2, omch2=Oc0h2, As=A_s, ns=n_s, lmax = lmax_unlensCl+10)

            results = camb.get_results(pars)
            powers =results.get_cmb_power_spectra(pars, CMB_unit='muK', lmax = lmax_unlensCl, raw_cl = True)
            for name in powers: 
                print(name)

            cls_unlensed = powers['unlensed_scalar']
            print(cls_unlensed.shape)

            self.mock_CMB_primary = hp.synfast(cls_unlensed[:,0], nside=self.nside, lmax=lmax_unlensCl) ##in miuK
        
        elif self.fits_file == 'lensed_z2':
            self.mock_CMB_primary = hp.read_map(f'{lensed_dir}/CMB_T_map_l_kappa_z2_new_nside8192.fits', dtype=np.float64, verbose=False)
            self.mock_CMB_primary = hp.pixelfunc.ud_grade(self.mock_CMB_primary, self.nside)

        elif self.fits_file == 'lensed_z3':
            self.mock_CMB_primary = hp.read_map(f'{lensed_dir}/CMB_T_map_l_kappa_z3_new_nside8192.fits', dtype=np.float64, verbose=False)
            self.mock_CMB_primary = hp.pixelfunc.ud_grade(self.mock_CMB_primary, self.nside)

        else:
            raise ValueError("Unknown CMB map generation method")
        

        if plot == True:
            hp.mollview(self.mock_CMB_primary, title=f"Mock Primary CMB temperature map (box={self.boxname}, sim={self.simname})", cmap="jet")#, min=-1.5e-4, max=1.5e-4)
            hp.graticule()
            plt.savefig(f'./Plots/primary_CMB_map_{self.boxname}_{self.simname}.png', dpi=400)
            plt.clf()

        print(f'Generating mock primary CMB: {time.time() - self.job_start_time}s')

        return

    
    def load_lightcones(self, plot=False):
                
        # Loading tau map from FLAMINGO lightcone shells
        if self.lightcone_method[0] == 'SHELL':
            if self.boxname == 'L1000N1800' and self.lightcone == 0:
                map_dir = 'neutrino_corrected_maps'
            elif self.boxname == 'L2800N5040' and self.simname == 'HYDRO_FIDUCIAL':
                map_dir = 'neutrino_corrected_maps_downsampled_4096'
            else:
                print("Lightcone map not available for this box/simulation combination.")
                sys.exit()
            
            # map_lightcone = f'/cosma8/data/dp004/flamingo/Runs/{self.boxname}/{self.simname}/{map_dir}/lightcone{self.lightcone}_shells/shell_{self.z_sample}/lightcone{self.lightcone}.shell_{self.z_sample}.0.hdf5'
            # g = h5py.File(map_lightcone,'r')
            # conversion_factor = g['DM'].attrs['Conversion factor to CGS (not including cosmological corrections)']
            # tau = g['DM'][...] * conversion_factor * 6.6524587321e-25 #6.65246e-25 = Thomson cross-section (in cgs)
            # redshift = g['DM'].attrs['Central redshift assumed for correction']
            # tau *= (1+redshift)
            # g.close()

            tau = hp.read_map(f'/cosma8/data/dp004/dc-conl1/FLAMINGO/patchy_screening/data_files/DM_maps/{self.boxname}/{self.simname}/lightcone{self.lightcone}/shells/tau_map_shell_{self.z_sample}_scale_factor.fits', dtype=np.float64, verbose=False)
            map_central_redshifts = np.loadtxt(f'/cosma8/data/dp004/dc-conl1/FLAMINGO/patchy_screening/data_files/DM_maps/{self.boxname}/{self.simname}/lightcone{self.lightcone}/shell_diagnostics_scale_factor.txt', skiprows=1, delimiter=' ', usecols=(0, 1))
            redshift = map_central_redshifts[self.z_sample, 1]

            print(f'Map z of {self.z_sample_name} sample = {redshift} (shell: {self.z_sample})')
            print(tau[0:10])
            print(f'Loading first lightcone shell: {time.time() - self.job_start_time}s')

            if plot == True:
                hp.mollview(tau, title=f"tau map (box={self.boxname}, sim={self.simname}, lightcone shell={self.z_sample})", cmap="jet", min=2e-5, max=2e-3)
                hp.graticule()
                plt.savefig(f'./Plots/tau_map_{self.boxname}_{self.simname}_{self.z_sample_name}_shell_{self.z_sample}.png', dpi=400)
                plt.clf()

            # map_lightcone_lower = f'/cosma8/data/dp004/flamingo/Runs/{self.boxname}/{self.simname}/{map_dir}/lightcone{self.lightcone}_shells/shell_{self.z_sample-1}/lightcone{self.lightcone}.shell_{self.z_sample-1}.0.hdf5'
            # g_low = h5py.File(map_lightcone_lower,'r')
            # conversion_factor = g_low['DM'].attrs['Conversion factor to CGS (not including cosmological corrections)']
            # tau_low = g_low['DM'][...] * conversion_factor * 6.6524587321e-25
            # redshift_low = g_low['DM'].attrs['Central redshift assumed for correction']
            # tau_low *= (1+redshift_low)

            # map_lightcone_higher = f'/cosma8/data/dp004/flamingo/Runs/{self.boxname}/{self.simname}/{map_dir}/lightcone{self.lightcone}_shells/shell_{self.z_sample+1}/lightcone{self.lightcone}.shell_{self.z_sample+1}.0.hdf5'
            # g_high = h5py.File(map_lightcone_higher,'r')
            # conversion_factor = g_high['DM'].attrs['Conversion factor to CGS (not including cosmological corrections)']
            # tau_high = g_high['DM'][...] * conversion_factor * 6.6524587321e-25
            # redshift_high = g_high['DM'].attrs['Central redshift assumed for correction']
            # tau_high *= (1+redshift_high)

            tau_low = hp.read_map(f'/cosma8/data/dp004/dc-conl1/FLAMINGO/patchy_screening/data_files/DM_maps/{self.boxname}/{self.simname}/lightcone{self.lightcone}/shells/tau_map_shell_{self.z_sample-1}_scale_factor.fits', dtype=np.float64, verbose=False)
            redshift_low = map_central_redshifts[self.z_sample-1, 1]

            tau_high = hp.read_map(f'/cosma8/data/dp004/dc-conl1/FLAMINGO/patchy_screening/data_files/DM_maps/{self.boxname}/{self.simname}/lightcone{self.lightcone}/shells/tau_map_shell_{self.z_sample+1}_scale_factor.fits', dtype=np.float64, verbose=False)
            redshift_high = map_central_redshifts[self.z_sample+1, 1]

            tau = tau_low + tau + tau_high

            print(f"Map lightcone z = [{redshift_low},{redshift_high}]")
            # g_low.close()
            # g_high.close()

        elif self.lightcone_method[0] == 'FULL':
            try:
                tau = hp.read_map(f'/cosma8/data/dp004/dc-conl1/FLAMINGO/patchy_screening/data_files/DM_maps/{self.boxname}/{self.simname}/lightcone{self.lightcone}/stacked_tau_map_z3p0_scale_factor.fits', dtype=np.float64, verbose=False)
            except FileNotFoundError:
                from stacked_DM_maps import stack_DM_maps_z3
                tau = stack_DM_maps_z3(self.ncpu, self.boxname, self.simname, lightcone=self.lightcone, scale_factor=True, save_shells=False)
            # tau_2 = hp.pixelfunc.ud_grade(tau,self.nside)
            # alm = hp.map2alm(tau_2)
            # #tau_2 = hp.alm2map(alm, nside=self.nside, lmax=5024)
            # tau_2 = hp.sphtfunc.resize_alm(alm, lmax=3*self.nside, mmax=3*self.nside, lmax_out=5024, mmax_out=5024)
            print(f"Map lightcone integrated up to z=3")

        self.tau_map = hp.pixelfunc.ud_grade(tau,self.nside)
        mean_tau = np.mean(self.tau_map)
        self.tau_map -= mean_tau

        # #alm = hp.map2alm(self.tau_map)
        # #self.tau_map_2 = hp.pixelfunc.ud_grade(tau_2,self.nside)
        # self.tau_map_2 = hp.alm2map(tau_2, nside=self.nside, lmax=5024)
        print(self.tau_map[0:10])
        # print(self.tau_map_2)

        if plot == True:
            if self.lightcone_method[0] == 'SHELL':
                hp.mollview(self.tau_map, title=f"Tau map (box={self.boxname}, sim={self.simname}, lightcone shell={self.z_sample-1}+{self.z_sample}+{self.z_sample+1})", cmap="jet")#, min=2e-5, max=2e-3)
                hp.graticule()
                plt.savefig(f'./Plots/tau_map_{self.boxname}_{self.simname}_{self.z_sample_name}_shell_{self.z_sample-1}-{self.z_sample+1}.png', dpi=400)
            elif self.lightcone_method[0] == 'FULL':
                hp.mollview(self.tau_map, title=f"Stacked Tau map, integrated up to z=3 (box={self.boxname}, sim={self.simname})", cmap="jet")#, min=2e-5, max=2e-3)
                hp.graticule()
                plt.savefig(f'./Plots/tau_map_{self.boxname}_{self.simname}_stacked_z3p0.png', dpi=400)
            plt.clf()
        print(f'Loading relevant lightcone shells: {time.time() - self.job_start_time}s')

        return

    
    def load_halo_data(self, lightcone_type='HBT'):
        
        # Load halo lightcone and SOAP data into DataFrames
        if lightcone_type == 'HBT':
            if self.boxname == 'L1000N1800':
                snap_max = 77
                halo_lc_dir = 'hbt_lightcone_halos'
            elif self.boxname == 'L1000N3600':
                snap_max = 78
                halo_lc_dir = 'hbt_lightcone_halos'
            elif self.boxname == 'L2800N5040':
                snap_max = 78
                halo_lc_dir = 'sorted_hbt_lightcone_halos'
            else:
                print("Halo lightcone not available for this box/simulation combination.")
                sys.exit()
            halo_lightcone = f'/cosma8/data/dp004/flamingo/Runs/{self.boxname}/{self.simname}/{halo_lc_dir}/lightcone{self.lightcone}/lightcone_halos_{snap_max-self.z_sample:04d}.hdf5'
            f = h5py.File(halo_lightcone, 'r')
            halo_lc_data = pd.DataFrame({
                'ID':          f['InputHalos/HaloCatalogueIndex'][...],
                'SnapNum':     f['Lightcone/SnapshotNumber'][...],
                'z':           f['Lightcone/Redshift'][...],
                'xminpot':     f['Lightcone/HaloCentre'][...][:,0],
                'yminpot':     f['Lightcone/HaloCentre'][...][:,1],
                'zminpot':     f['Lightcone/HaloCentre'][...][:,2],
            }).sort_values('ID')
            f.close()
        elif lightcone_type == 'VR':
            halo_lightcone = f'/cosma8/data/dp004/jch/FLAMINGO/lightcone_halos/{self.simname}/lightcone_halos/lightcone0/lightcone_halos_{self.z_sample:04d}.hdf5'
            f = h5py.File(halo_lightcone, 'r')
            halo_lc_data['ID'] = f['Subhalo/ID'][...]
            halo_lc_data['SnapNum'] = f['Subhalo/SnapNum'][...]
            halo_lc_data['z'] = f['Subhalo/LightconeRedshift'][...]
            halo_lc_data['xminpot'] = f['Subhalo/LightconeXcminpot'][...]
            halo_lc_data['yminpot'] = f['Subhalo/LightconeYcminpot'][...]
            halo_lc_data['zminpot'] = f['Subhalo/LightconeZcminpot'][...]
            f.close()

        print(f"Halo lightcone z = [{halo_lc_data['z'].min()},{halo_lc_data['z'].max()}]")
        Dcom = self.cosmology.comoving_distance(halo_lc_data['z'].mean())*0.681  # comoving distance to galaxy in Mpc/h
        self.Dcom = Dcom.value
        snap = int(halo_lc_data['SnapNum'].iloc[0])
        print(f'D_com = {self.Dcom}, Snap number = {snap}')

        if lightcone_type == 'HBT':
            # HBT_file = f'/cosma8/data/dp004/flamingo/Runs/{self.boxname}/{self.simname}/SOAP-HBT/halo_properties_{snap:04d}.hdf5'
            # f = h5py.File(HBT_file, 'r')
            # df_HBT = pl.DataFrame({
            #     'ID':          f['InputHalos/HaloCatalogueIndex'][...],
            #     'Structuretype': f['InputHalos/IsCentral'][...],
            #     'mvir':       f['SO/500_crit/TotalMass'][...] * 1e10,
            #     'mstar':       f['ExclusiveSphere/50kpc/StellarMass'][...] * 1e10,
            #     'HaloID':      f['SOAP/HostHaloIndex'][...],
            # })
            # f.close()

            # --- inside load_halo_data(), in the `if lightcone_type == 'HBT':` block ---

            HBT_file = f'/cosma8/data/dp004/flamingo/Runs/{self.boxname}/{self.simname}/SOAP-HBT/halo_properties_{snap:04d}.hdf5'
            with h5py.File(HBT_file, 'r') as f:
                ids    = f['InputHalos/HaloCatalogueIndex'][...]
                struct = f['InputHalos/IsCentral'][...]                  # 1=central, 0=satellite (your convention)
                mvir   = f['SO/500_crit/TotalMass'][...] * 1e10
                mstar  = f['ExclusiveSphere/50kpc/StellarMass'][...] * 1e10
                hid    = f['SOAP/HostHaloIndex'][...]                    # index into the FULL arrays above
            f.close()

            # ---- apply host-mvir fix for satellites (from dndz_dndm_curve.py logic) ----
            SATELLITE_FLAG = 0
            CENTRAL_FLAG   = 1

            sat_mask = (struct == SATELLITE_FLAG) & (hid >= 0)

            mvir_fixed = mvir
            if np.any(sat_mask):
                host_ids    = hid[sat_mask].astype(np.int64)
                host_struct = struct[host_ids]
                host_mvir   = mvir[host_ids]

                # only accept hosts that are centrals; otherwise leave original mvir
                good_host = (host_struct == CENTRAL_FLAG)

                mvir_fixed = mvir.copy()
                mvir_fixed[sat_mask] = np.where(good_host, host_mvir, mvir[sat_mask])

            # build dataframe (satellites now carry host mvir in the mvir column)
            df_HBT = pd.DataFrame({
                'ID':            ids,
                'Structuretype': struct,
                'mvir':          mvir_fixed,
                'mstar':         mstar,
                'HostHaloID':        hid,
            })


            print(f'Loading halo lightcone data: {time.time() - self.job_start_time}s')

            return halo_lc_data, df_HBT
        elif lightcone_type == 'VR':
            VR_file = f'/cosma8/data/dp004/flamingo/Runs/{self.boxname}/{self.simname}/SOAP/halo_properties_{snap:04d}.hdf5'
            f = h5py.File(HBT_file, 'r')
            df_VR = pd.DataFrame()
            df_VR['ID'] = f['VR/ID'][...]
            #df_VR['hostHaloID'] = f['VR/HostHaloID'][...]
            df_VR['Structuretype'] = f['VR/StructureType'][...]
            df_VR['m_vir'] = f['SO/500_crit/TotalMass'][...]
            df_VR['mstar'] = f['ExclusiveSphere/50kpc/StellarMass'][...]
            f.close()
            print(f'Loading halo lightcone data: {time.time() - self.job_start_time}s')

            return halo_lc_data, df_VR

        
    def filter_stellar_mass(self, halo_lc_data=None, df_halo=None):
        
        # Merge and filter the DataFrames based on stellar mass bin
        if self.lightcone_method[1] == 'shell':
            if halo_lc_data is None or df_halo is None:
                halo_lc_data, df_halo = self.load_halo_data()
                
            df_mass = df_halo[df_halo['mstar'] > self.im]
            self.merge = pd.merge(
                df_mass,
                halo_lc_data,
                on='ID',
                how='inner'
            ).sort_values('ID').reset_index(drop=True)
            if self.merge.empty:
                print(f"No halos for stellar cut = {np.log10(self.im)}")
                self.nhalo = 0
                return
        elif self.lightcone_method[1] == 'dndz':
            print(f"/cosma8/data/dp004/dc-conl1/FLAMINGO/patchy_screening/data_files/mock_halo_catalogs/{self.boxname}/{self.simname}/{self.z_sample_name}/lightcone{self.lightcone}/sampled_halo_data_{self.im_name}_{self.slope_name}.parquet")
            self.merge = pd.read_parquet(
                f"/cosma8/data/dp004/dc-conl1/FLAMINGO/patchy_screening/data_files/mock_halo_catalogs/{self.boxname}/{self.simname}/{self.z_sample_name}/lightcone{self.lightcone}/sampled_halo_data_{self.im_name}_{self.slope_name}.parquet"
            )
            mean_z = {'Blue':0.6, 'Green':1.1, 'Red':1.5} 
            Dcom = self.cosmology.comoving_distance(mean_z[self.z_sample_name])*0.681  # comoving distance to galaxy in Mpc/h
            self.Dcom = Dcom.value

        self.x = self.merge['xminpot'].to_numpy()
        self.y = self.merge['yminpot'].to_numpy()
        self.z = self.merge['zminpot'].to_numpy()
        mvir = self.merge['mvir'].to_numpy()
        mstar = self.merge['mstar'].to_numpy()
        self.nhalo = mvir.size
        print(self.nhalo)
        print(np.log10(self.im), np.log10(np.min(mstar)), np.log10(np.mean(mstar)), np.log10(np.mean(mvir)), self.nhalo)
        print(f'Identifying stackable objects: {time.time() - self.job_start_time}s')

        return

    def compute_alm_maps(self, plot=False):
                    
        # Computing and filtering spherical harmonic coefficients to create large and small scale maps
        try:
            alm = hp.map2alm(self.T_cmb_ps, lmax=3*self.nside)
        except AttributeError:
            self.get_patchy_screening_map(plot)
            alm = hp.map2alm(self.T_cmb_ps, lmax=3*self.nside)

        ell, m = hp.Alm.getlm(lmax=3*self.nside)

        if self.rotate == True:
            np.random.seed(int(sys.argv[-1]))
            rotated_alm = hp.Rotator(deg=True, rot=(np.random.uniform(0, 180), np.random.uniform(0, 360))).rotate_alm(alm, lmax=3*self.nside)
            alm = rotated_alm

        pixel_window = hp.pixwin(self.nside, lmax=3*self.nside)
        alm = hp.almxfl(alm, 1/pixel_window)

        lowpass_values = np.array([self.f_lowpass(l) for l in ell])
        highpass_values = np.array([self.f_highpass(l) for l in ell])
        lowpass_alm = hp.almxfl(alm.copy(), lowpass_values)
        highpass_alm = hp.almxfl(alm.copy(), highpass_values)

        if plot == True:
            m_zero = np.where(m==0)
            plt.plot(ell[m_zero], lowpass_values[m_zero], label='lowpass filter', color='b')
            plt.plot(ell[m_zero], highpass_values[m_zero], label='highpass filter', color='r')
            plt.xlabel(r'$\ell$')
            plt.ylabel('Filter values')
            plt.title('My version of the Coulton frequency filters (m=0)')
            plt.legend()
            plt.savefig('./Plots/filter_plot.png', dpi=400)
            plt.clf()

        self.large_scale_map = hp.alm2map(lowpass_alm, nside=self.nside, lmax=3*self.nside)
        self.small_scale_map = hp.alm2map(highpass_alm, nside=self.nside, lmax=3*self.nside)

        if plot == True:
            hp.mollview(self.large_scale_map.copy(), title=f"Large scale CMB temperature map (box={self.boxname}, sim={self.simname})", cmap="jet")#, min=-1.5e-4, max=1.5e-4)
            hp.graticule()
            plt.savefig(f'./Plots/T_ps_map_large_scale_{self.boxname}_{self.simname}_{self.z_sample_name}_{self.im_name}_{self.slope_name}.png', dpi=400)
            plt.clf()

            hp.mollview(self.small_scale_map.copy(), title=f"Small scale CMB temperature map (box={self.boxname}, sim={self.simname})", cmap="jet")#, min=-1e-6, max=1e-6)
            hp.graticule()
            plt.savefig(f'./Plots/T_ps_map_small_scale_{self.boxname}_{self.simname}_{self.z_sample_name}_{self.im_name}_{self.slope_name}.png', dpi=400)
            plt.clf()

        self.mean_mod_T_large_scale = np.mean(np.abs(self.large_scale_map.copy()))
        print(self.mean_mod_T_large_scale)

        if self.rotate == True:
            print(f'Computing, rotating and filtering alms: {time.time() - self.job_start_time}s')
        elif self.rotate == False:
            print(f'Computing and filtering alms: {time.time() - self.job_start_time}s')

        return

    def reconstruct_tau_map(self, plot=False):
        """
        Build the signed tau estimator map and high-pass filter it
        by removing modes with ell < ell_min.
        """

        ell_min = 1600

        try:
            tau_est = -(np.sign(self.large_scale_map.copy()) * self.small_scale_map.copy()) / self.mean_mod_T_large_scale.copy()
        except AttributeError:
            self.compute_alm_maps(plot)
            tau_est = -(np.sign(self.large_scale_map.copy()) * self.small_scale_map.copy()) / self.mean_mod_T_large_scale.copy()

        lmax = 3 * self.nside - 1
        tau_alm = hp.map2alm(tau_est, lmax=lmax)
        ell, _ = hp.Alm.getlm(lmax=lmax)

        tau_values = np.array([self.lensing_filter(l) for l in ell])
        tau_alm = hp.almxfl(tau_alm.copy(), tau_values)

        self.reconstructed_tau_map = hp.alm2map(tau_alm, nside=self.nside, lmax=lmax)

        # self.reconstructed_tau_map = hp.ud_grade(self.reconstructed_tau_map, 8192)

        self.reconstructed_tau_map = hp.smoothing(self.reconstructed_tau_map, fwhm=1.3*np.pi/60.0/180.0)

        if plot:
            hp.mollview(self.reconstructed_tau_map, title=f"Reconstructed tau map, ell >= {ell_min}", cmap="jet")
            hp.graticule()
            plt.savefig(f"./Plots/reconstructed_tau_map_{self.boxname}_{self.simname}_{self.z_sample_name}.png", dpi=400)
            plt.clf()

        print(f"Reconstructing and filtering tau map: {time.time() - self.job_start_time}s")

        return

    def f_lowpass(self, l):
        # Low frequency bandpass filter
        if l < 600:
            return 1
        elif l >= 650:
            return 0
        else:
            return np.cos(((l - 600) * np.pi) / 100)

    def f_highpass(self, l):
        # High frequency bandpass filter
        if l < 850:
            return 0
        elif l >= 900:
            return 1
        else:
            return np.sin(((l - 850) * np.pi) / 100)
        
    def lensing_filter(self, l):
        # Lensing filter for the tau map
        if l < 1600:
            return 0
        elif l >= 1600:
            return 1

    def balance_large_scale_signs(self, plot=False, seed=1000):
        """
        Keep equal numbers of halos with positive and negative T_large values
        at the halo positions.
        """

        try:
            pixels = hp.vec2pix(self.nside, self.source_vector[:, 0], self.source_vector[:, 1], self.source_vector[:, 2])
        
        except AttributeError:
            try:
                halo = np.load(self.halo_tmp_file())
                self.theta = halo["theta"]
                self.phi = halo["phi"]
                self.source_vector = halo["source_vector"]
                self.nhalo = int(halo["nhalo"])

            except FileNotFoundError:
                self.get_halo_coordinates()

            pixels = hp.vec2pix(self.nside, self.source_vector[:, 0], self.source_vector[:, 1], self.source_vector[:, 2])

        print(len(pixels), len(self.source_vector), self.nhalo)

        try:
            t_large_at_halos = self.large_scale_map[pixels]
        except AttributeError:
            self.compute_alm_maps(plot=plot)
            t_large_at_halos = self.large_scale_map[pixels]

        pos_idx = np.where(t_large_at_halos > 0)[0]
        neg_idx = np.where(t_large_at_halos < 0)[0]

        n_keep = min(len(pos_idx), len(neg_idx))

        if n_keep == 0:
            raise ValueError("Cannot balance signs: one sign class has zero halos.")

        rng = np.random.default_rng(seed)

        pos_keep = rng.choice(pos_idx, size=n_keep, replace=False)
        neg_keep = rng.choice(neg_idx, size=n_keep, replace=False)

        keep_idx = np.concatenate([pos_keep, neg_keep])
        rng.shuffle(keep_idx)

        self.theta = self.theta[keep_idx]
        self.phi = self.phi[keep_idx]
        self.source_vector = self.source_vector[keep_idx]
        self.nhalo = len(keep_idx)

        print(f"Balanced large-scale signs: {n_keep} positive + {n_keep} negative = {self.nhalo} halos")
        
        return


    def tau_prof_image(self, i): #, plot=(False,False)):

        # Compute the tau profile image around the i-th halo using gnomview
        fluxes=hp.gnomview(self.reconstructed_tau_map, rot=[self.theta[i],self.phi[i]], xsize=self.npix_cutout, ysize=self.npix_cutout, reso=self.pix_res, 
                           return_projected_map=True, no_plot=True)

        # Old method for computing tau profiles using query_disc and large/small scale maps
        # Compute the tau profiles in annuli around halo centres
        # halo_pixels = hp.query_disc(self.nside, self.source_vector[i,:], radius=(0.25*u.deg).to_value(u.radian))
        # theta_pix, phi_pix = hp.pix2ang(self.nside, halo_pixels, lonlat=True)
        # rtheta = hp.rotator.angdist([self.theta[i], self.phi[i]], [theta_pix, phi_pix], lonlat=True)*60.0*180.0/np.pi
        
        # large_scale_fluxes=np.array(self.large_scale_map[halo_pixels].copy())
        # small_scale_fluxes=np.array(self.small_scale_map[halo_pixels].copy())
        
        # if plot[0] == True and i == plot[1]:
        #     tau_2D = (np.sign(large_scale_fluxes.copy())*small_scale_fluxes.copy())/self.mean_mod_T_large_scale.copy()
        #     grid_x = np.linspace(min(theta_pix.copy()*60.0), max(theta_pix.copy()*60.0), 25)  # Adjust grid size as needed
        #     grid_y = np.linspace(min(phi_pix.copy()*60.0), max(phi_pix.copy()*60.0), 25)
        #     temperature_map = self.T_cmb_ps[halo_pixels].copy()
        #     grid_values_temp, xedges_temp, yedges_temp = np.histogram2d(theta_pix.copy()*60.0, phi_pix.copy()*60.0, bins=[grid_x, grid_y], weights=temperature_map)
            
        #     plt.figure(figsize=(6, 6))
        #     plt.imshow(grid_values_temp.T, origin='lower', extent=[grid_x.min(), grid_x.max(), grid_y.min(), grid_y.max()], cmap='inferno')#, vmin=-1.5e-4, vmax=1.5e-4)
        #     plt.colorbar(label=r'Temperature ($\mu$K)')
        #     plt.xlabel('X (arcmin)')
        #     plt.ylabel('Y (arcmin)')
        #     plt.title(f'Rectangular Cutout Around Halo = {i}', wrap=True)
        #     plt.savefig(f'./Plots/random_cutout_T_ps_map_{self.boxname}_{self.simname}_{self.z_sample_name}.png', dpi=400)
        #     plt.clf()
        #     plt.close('all')
            
        #     grid_values_tau = np.histogram2d(theta_pix.copy()*60.0, phi_pix.copy()*60.0, bins=[grid_x, grid_y], weights=tau_2D)
            
        #     plt.figure(figsize=(6, 6))
        #     plt.imshow(grid_values_tau[0].T, origin='lower', extent=[grid_x.min(), grid_x.max(), grid_y.min(), grid_y.max()], cmap='inferno')#, vmin=-1e-6, vmax=1e-6)
        #     plt.colorbar(label='Optical Depth')
        #     plt.xlabel('X (arcmin)')
        #     plt.ylabel('Y (arcmin)')
        #     plt.title(f'Rectangular Cutout Around Halo = {i} w/ Patchy Screening', wrap=True)
        #     plt.savefig(f'./Plots/random_cutout_T_filtered_map_{self.boxname}_{self.simname}_{self.z_sample_name}.png', dpi=400)
        #     plt.clf()
        #     plt.close('all')

        # tau_1D = np.zeros(len(self.theta_d))

        # for j in range(len(self.theta_d)):
        #     idx_in = np.where((rtheta > self.theta_d[j]-0.25) & (rtheta <= self.theta_d[j]+0.25))
        #     tau_1D[j] = (np.mean(np.sign(large_scale_fluxes[idx_in])*small_scale_fluxes[idx_in])) #(-1*np.mean(np.sign(large_scale_fluxes[idx_in])*small_scale_fluxes[idx_in]))/self.mean_mod_T_large_scale
        
        return fluxes #tau_1D
    
    def tau_prof(self, i):
        halo_pixels = hp.query_disc(self.nside, self.source_vector[i, :], radius=(0.25 * u.deg).to_value(u.radian))
        theta_pix, phi_pix = hp.pix2ang(self.nside, halo_pixels, lonlat=True)
        rtheta = hp.rotator.angdist([self.theta[i], self.phi[i]], [theta_pix, phi_pix], lonlat=True)*60.0*180.0/np.pi
        tau_values = np.array(self.reconstructed_tau_map[halo_pixels].copy())

        tau_1D = np.zeros(len(self.theta_d))

        for j in range(len(self.theta_d)):
            idx_in = np.where((rtheta > self.theta_d[j] - 0.25) & (rtheta <= self.theta_d[j] + 0.25))
            tau_1D[j] = np.mean(tau_values[idx_in])

        return tau_1D

    def run_tau_profiles(self):
        # Parallelisation of computing tau profiles for each halo
        batch_size=max(1, self.nhalo // (self.ncpu*2))

        print(f'Starting profile loop: {time.time() - self.job_start_time}s')
        
        results = Parallel(n_jobs=self.ncpu, backend="loky", batch_size=batch_size)(delayed(self.tau_prof)(i) for i in range(self.nhalo))
        self.data_1D = np.asarray(results)
        print(f'Ending profile loop: {time.time() - self.job_start_time}s')
        return
    
    def run_tau_profiles_mpi(self):
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        tmpdir = os.path.join(self.get_output_dir(), "mpi_tau_chunks")
        if rank == 0:
            os.makedirs(tmpdir, exist_ok=True)

            t0 = time.time()
            print(f"Starting MPI tau profiles: {t0 - self.job_start_time:.2f}s")

        comm.Barrier()

        if rank == 0:
            print(f"MPI setup complete: {time.time() - self.job_start_time:.2f}s")

        all_indices = np.arange(self.nhalo)
        my_indices = all_indices[rank::size]

        print(
            f"Rank {rank}/{size} processing {len(my_indices)} halos "
            f"from total {self.nhalo}"
        )

        t_local_start = time.time()

        # STABLE VERSION (without joblib parallelization within each rank):
        local_results = [
            self.tau_prof(i)
            for i in my_indices
        ]

        # local_results = Parallel(
        #     n_jobs=self.ncpu,
        #     backend="threading",
        #     batch_size=max(1, len(my_indices) // (self.ncpu * 2)),
        # )(
        #     delayed(self.tau_prof)(i)
        #     for i in my_indices
        # )

        if len(local_results) > 0:
            local_results = np.asarray(local_results)
        else:
            local_results = np.empty((0, len(self.theta_d)))

        local_compute_time = time.time() - t_local_start
        print(f"Rank {rank} finished local profiles in {local_compute_time:.2f}s")

        chunk_file = os.path.join(tmpdir, f"rank_{rank:04d}.npz")

        np.savez_compressed(
            chunk_file,
            indices=my_indices,
            values=local_results,
        )

        print(f"Rank {rank} wrote {chunk_file}: {time.time() - self.job_start_time:.2f}s")

        comm.Barrier()

        if rank == 0:
            self.data_1D = np.empty((self.nhalo, len(self.theta_d)))

            for r in range(size):
                chunk_file = os.path.join(tmpdir, f"rank_{r:04d}.npz")
                chunk = np.load(chunk_file)

                indices = chunk["indices"]
                values = chunk["values"]

                self.data_1D[indices, :] = values

            print(f"Finished MPI profile loop: {time.time() - self.job_start_time}s")

        comm.Barrier()


    def run_tau_image_stack_mpi(self):
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        if rank == 0:
            t0 = time.time()
            print(f"Starting MPI tau image stack with {size} ranks")

        all_indices = np.arange(self.nhalo)
        my_indices = all_indices[rank::size]

        print(
            f"Rank {rank}/{size} processing {len(my_indices)} halos "
            f"from total {self.nhalo}",
            flush=True,
        )

        local_sum = np.zeros(
            (self.npix_cutout, self.npix_cutout),
            dtype=np.float64,
        )

        # local_count = np.zeros(
        #     (self.npix_cutout, self.npix_cutout),
        #     dtype=np.int64,
        # )

        local_count = np.array(len(my_indices), dtype=np.int64)

        progress_every = max(1, len(my_indices) // 20)  # roughly every 5% on rank 0

        for n_done, i in enumerate(my_indices, start=1):
            img = self.tau_prof_image(i)

            # if hasattr(img, "filled"):
            #     img = img.filled(np.nan)

            img = np.asarray(img, dtype=np.float64)

            # good = np.isfinite(img)
            # local_sum[good] += img[good]
            # local_count[good] += 1
            
            local_sum += img

            if rank == 0 and (n_done % progress_every == 0 or n_done == len(my_indices)):
                approx_global_done = min(n_done * size, self.nhalo)
                print(
                    f"Approx progress: {approx_global_done}/{self.nhalo} "
                    f"({100 * approx_global_done / self.nhalo:.1f}%)",
                    flush=True,
                )

        global_sum = np.zeros_like(local_sum) if rank == 0 else None
        # global_count = np.zeros_like(local_count) if rank == 0 else None
        global_count = np.array(0, dtype=np.int64) if rank == 0 else None

        comm.Reduce(local_sum, global_sum, op=MPI.SUM, root=0)
        comm.Reduce(local_count, global_count, op=MPI.SUM, root=0)

        if rank == 0:
            # self.tau_2D_stack = np.divide(
            #     global_sum,
            #     global_count,
            #     out=np.full_like(global_sum, np.nan),
            #     where=global_count > 0,
            # )
            self.tau_2D_stack = global_sum / int(global_count)

            print(
                f"Finished MPI image stack: "
                f"{time.time() - t0:.2f}s"
            )
        

    def run_tau_mpi_from_saved(self):
        self.load_branch_outputs()
        self.run_tau_profiles_mpi()

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()

        # if rank == 0:
        #     self.balance_large_scale_signs_and_save(seed=1000)

        # comm.Barrier()

        # self.load_balanced_halo_outputs()

        # comm.Barrier()

        # if rank == 0:
        #     print(
        #         f"Balanced catalogue loaded on all ranks: "
        #         f"nhalo = {self.nhalo}"
        #     )

        # self.run_tau_profiles_mpi()

        if rank == 0:
            self.stack_and_save()

            if self.cleanup_tmp:
                self.cleanup_tmp_dir()

        comm.Barrier()

    def run_tau_mpi_from_saved_image(self, plot=False):
        self.load_branch_outputs()
        self.run_tau_image_stack_mpi()

        comm = MPI.COMM_WORLD
        if comm.Get_rank() == 0:
            self.stack_and_save_image(plot=plot)

            if self.cleanup_tmp:
                self.cleanup_tmp_dir()

        comm.Barrier()
            

    def stack_and_save(self):
        comm = MPI.COMM_WORLD
        if comm.Get_rank() != 0:
            return
        
        # Stacking of tau profiles and save as pickle files
        tau_1D_stack = np.zeros(len(self.theta_d))
        for i in range(self.nhalo):
            tau_1D = self.data_1D[i,:]
            tau_1D_stack += tau_1D
        # tau_1D_stack *= -1.0/(self.mean_mod_T_large_scale * self.nhalo)
        tau_1D_stack /= self.nhalo
        
        rows, cols = (len(self.theta_d), 4)
        data = [0]*cols
        data[0] = self.theta_d
        data[1] = tau_1D_stack
        data[2] = (self.theta_d*np.pi/(180.0*60.0))*self.Dcom
        data[3] = self.nhalo

        fits_suffix = "" if self.cmb_method=='CAMB' else f"_{self.fits_file}"
        signal_suffix = "" if self.signal==True else "_no_ps"
        noise_suffix = "" if self.rotate==False else "_noise"
        test_suffix = "" if self.test_name == None else f"_{self.test_name}"

        outfile = os.path.join(f'./data_files/tau_profiles/{self.boxname}', self.simname, self.z_sample_name, f'lightcone{self.lightcone}', 
                               f'tau_mle_catalogue_nside{self.nside}_{self.cmb_method}{fits_suffix}{signal_suffix}{noise_suffix}{test_suffix}.pickle'
                               if self.mle else f'tau_Mstar_bin{self.im_name}_{self.slope_name}_nside{self.nside}_{self.cmb_method}{fits_suffix}{signal_suffix}{noise_suffix}{test_suffix}.pickle')
        os.makedirs(os.path.dirname(outfile), exist_ok=True)
        with open(outfile, 'wb') as f:
            pickle.dump(data, f)
        f.close()
        print(f'Writing out data: {time.time() - self.job_start_time}s')
        return


    def stack_and_save_image(self, plot=False):
        comm = MPI.COMM_WORLD
        if comm.Get_rank() != 0:
            return

        tau_1D_stack = np.zeros(len(self.theta_d))

        for j in range(len(self.theta_d)):
            idx_in = (
                (self.rtheta > self.theta_d[j] - 0.25) &
                (self.rtheta <= self.theta_d[j] + 0.25)
            )
            tau_1D_stack[j] = np.mean(self.tau_2D_stack[idx_in])

        if plot:
            os.makedirs("./Plots", exist_ok=True)

            vmax = np.max(np.abs(self.tau_2D_stack))

            extent = [
                -0.5 * self.npix_cutout * self.pix_res,
                0.5 * self.npix_cutout * self.pix_res,
                -0.5 * self.npix_cutout * self.pix_res,
                0.5 * self.npix_cutout * self.pix_res,
            ]

            plt.imshow(
                self.tau_2D_stack.T,
                origin="lower",
                extent=extent,
                cmap="RdBu",
                vmax=vmax,
                vmin=-vmax
            )
            plt.xlabel("x [arcmin]")
            plt.ylabel("y [arcmin]")
            plt.colorbar(label=r"$\tau$")
            plt.savefig(
                f"./Plots/tau_2D_stack_{self.boxname}_{self.simname}_{self.z_sample_name}.png",
                dpi=400,
                bbox_inches="tight",
            )
            plt.clf()

        data = [0] * 5
        data[0] = self.theta_d
        data[1] = tau_1D_stack
        data[2] = (self.theta_d * np.pi / (180.0 * 60.0)) * self.Dcom
        data[3] = self.nhalo
        data[4] = self.tau_2D_stack

        fits_suffix = "" if self.cmb_method == "CAMB" else f"_{self.fits_file}"
        signal_suffix = "" if self.signal else "_no_ps"
        noise_suffix = "" if not self.rotate else "_noise"

        outfile = os.path.join(
            f"./data_files/tau_profiles/{self.boxname}",
            self.simname,
            self.z_sample_name,
            f"lightcone{self.lightcone}",
            (
                f"tau_from_image_mle_catalogue_nside{self.nside}_{self.cmb_method}"
                f"{fits_suffix}{signal_suffix}{noise_suffix}.pickle"
                if self.mle else
                f"tau_from_image_Mstar_bin{self.im_name}_{self.slope_name}_nside{self.nside}"
                f"_{self.cmb_method}{fits_suffix}{signal_suffix}{noise_suffix}.pickle"
            ),
        )

        os.makedirs(os.path.dirname(outfile), exist_ok=True)

        with open(outfile, "wb") as f:
            pickle.dump(data, f)

        print(f"Writing out data: {time.time() - self.job_start_time}s")

    def _run_cmb_branch(self, proc_workers, plot=False):
        self.generate_cmb_map(plot)          # reads self.CAMB_params, writes self.cmb_map
        self.load_lightcones(plot)           # reads self.lightcone_params, writes self.lightcones
        self.get_patchy_screening_map(plot)  # reads cmb_map & lightcones, writes self.patchy_map

        # compute_alm_maps is CPU‐heavy, so run it in its own process:
        with ProcessPoolExecutor(max_workers=proc_workers) as proc:
            alm_process = proc.submit(self.compute_alm_maps(plot))

    def run_cmb_branch_and_save(self, plot=False):
        self.generate_cmb_map(plot)
        self.load_lightcones(plot)
        self.get_patchy_screening_map(plot)
        self.compute_alm_maps(plot)
        self.reconstruct_tau_map(plot)

        np.savez_compressed(
            self.cmb_tmp_file(),
            large_scale_map=self.large_scale_map,
            small_scale_map=self.small_scale_map,
            mean_mod_T_large_scale=self.mean_mod_T_large_scale,
            reconstructed_tau_map=self.reconstructed_tau_map,
        )

        print(f"Saved CMB branch data to {self.cmb_tmp_file()}")

    def _run_halo_branch(self):
        if self.lightcone_method[1] == 'shell':
            halo_lc_data, df_halo = self.load_halo_data()
            self.filter_stellar_mass(halo_lc_data, df_halo)
        elif self.lightcone_method[1] == 'dndz':
            self.filter_stellar_mass()     # reads self.lightcone_method, writes self.filtered_halos
        self.get_halo_coordinates()    # reads filtered_halos, writes self.halo_coords
        self.balance_large_scale_signs(seed=1000)

    def run_halo_branch_and_save(self):
        if self.lightcone_method[1] == "shell":
            halo_lc_data, df_halo = self.load_halo_data()
            self.filter_stellar_mass(halo_lc_data, df_halo)
        elif self.lightcone_method[1] == "dndz":
            self.filter_stellar_mass()

        if self.nhalo == 0:
            raise ValueError("No halos found; cannot run halo branch.")

        self.get_halo_coordinates()

        np.savez_compressed(
            self.halo_tmp_file(),
            theta=self.theta,
            phi=self.phi,
            source_vector=self.source_vector,
            nhalo=self.nhalo,
            Dcom=self.Dcom,
        )

        print(f"Saved halo branch data to {self.halo_tmp_file()}")

    def balance_large_scale_signs_and_save(self, seed=1000):
        """
        Balance the full halo catalogue by T_large sign on rank 0,
        then save the balanced catalogue for all MPI ranks to load.
        """

        pixels = hp.vec2pix(
            self.nside,
            self.source_vector[:, 0],
            self.source_vector[:, 1],
            self.source_vector[:, 2],
        )

        t_large_at_halos = self.large_scale_map[pixels]

        pos_idx = np.where(t_large_at_halos > 0)[0]
        neg_idx = np.where(t_large_at_halos < 0)[0]

        n_keep = min(len(pos_idx), len(neg_idx))

        if n_keep == 0:
            raise ValueError("Cannot balance signs: one sign class has zero halos.")

        rng = np.random.default_rng(seed)

        pos_keep = rng.choice(pos_idx, size=n_keep, replace=False)
        neg_keep = rng.choice(neg_idx, size=n_keep, replace=False)

        keep_idx = np.concatenate([pos_keep, neg_keep])
        rng.shuffle(keep_idx)

        theta_balanced = self.theta[keep_idx]
        phi_balanced = self.phi[keep_idx]
        source_vector_balanced = self.source_vector[keep_idx]
        nhalo_balanced = len(keep_idx)

        np.savez_compressed(
            self.balanced_halo_tmp_file(),
            theta=theta_balanced,
            phi=phi_balanced,
            source_vector=source_vector_balanced,
            nhalo=nhalo_balanced,
            Dcom=self.Dcom,
            keep_idx=keep_idx,
        )

        print(
            f"Saved balanced halo catalogue: "
            f"{n_keep} positive + {n_keep} negative = {nhalo_balanced} halos"
        )
        print(f"Saved to {self.balanced_halo_tmp_file()}")

    def load_branch_outputs(self):
        cmb = np.load(self.cmb_tmp_file())
        halo = np.load(self.halo_tmp_file())

        self.large_scale_map = cmb["large_scale_map"]
        self.small_scale_map = cmb["small_scale_map"]
        self.mean_mod_T_large_scale = float(cmb["mean_mod_T_large_scale"])
        self.reconstructed_tau_map = cmb["reconstructed_tau_map"]

        self.theta = halo["theta"]
        self.phi = halo["phi"]
        self.source_vector = halo["source_vector"]
        self.nhalo = int(halo["nhalo"])
        self.Dcom = float(halo["Dcom"])

        print(f"Loaded CMB branch data from {self.cmb_tmp_file()}")
        print(f"Loaded halo branch data from {self.halo_tmp_file()}")

    def load_balanced_halo_outputs(self):
        halo = np.load(self.balanced_halo_tmp_file())

        self.theta = halo["theta"]
        self.phi = halo["phi"]
        self.source_vector = halo["source_vector"]
        self.nhalo = int(halo["nhalo"])
        self.Dcom = float(halo["Dcom"])

        print(f"Loaded balanced halo catalogue from {self.balanced_halo_tmp_file()}")

    def run_analysis_old(self, plot=False):
        
        # Full analysis
        n_cpus = os.cpu_count() or 1           # should be 128 on your node
        n_thread_workers = min(2, n_cpus)      # two “branches” → 2 threads
        n_proc_workers   = max(1, n_cpus // 2)  # devote half your cores to the CPU‐heavy step
        
        # 1) fire off both branches concurrently in threads
        with ThreadPoolExecutor(max_workers=n_thread_workers) as exe:
            f_cmb  = exe.submit(self._run_cmb_branch(n_proc_workers, plot))
            f_halo = exe.submit(self._run_halo_branch)

            # 2) wait for both to finish
            wait([f_cmb, f_halo])

        # 3) now both self.alm and self.halo_coords exist
        '''self.generate_cmb_map(plot)
        self.load_lightcones(plot)
        self.get_patchy_screening_map(plot)
        if self.lightcone_method[1] == 'shell':
            halo_lc_data, df_halo = self.load_halo_data()
            self.filter_stellar_mass(halo_lc_data, df_halo)
        elif self.lightcone_method[1] == 'dndz':
            self.filter_stellar_mass()
        self.compute_alm_maps(plot)
        self.get_halo_coordinates()'''
        
        self.run_tau_profiles()
        self.stack_and_save()

        return
    
    def run_analysis(self, mode="full", plot=False):
        if mode == "cmb":
            self.run_cmb_branch_and_save(plot)

        elif mode == "halo":
            self.run_halo_branch_and_save()

        elif mode == "tau_mpi":
            self.run_tau_mpi_from_saved()
            # self.run_tau_mpi_from_saved_image(plot)

        elif mode == "full":
            self.generate_cmb_map(plot)
            self.load_lightcones(plot)
            self.get_patchy_screening_map(plot)

            if self.lightcone_method[1] == "shell":
                halo_lc_data, df_halo = self.load_halo_data()
                self.filter_stellar_mass(halo_lc_data, df_halo)
            elif self.lightcone_method[1] == "dndz":
                self.filter_stellar_mass()

            self.compute_alm_maps(plot)
            self.get_halo_coordinates()
            self.run_tau_profiles()
            self.stack_and_save()

        else:
            raise ValueError(f"Unknown mode: {mode}")

    def get_halo_coordinates(self):

        # Compute source vectors of each halo
        try:
            rows, cols = (self.nhalo, 3)
        except AttributeError:
            self.filter_stellar_mass()
            if self.nhalo == 0:
                print("No halos to compute coordinates")
                sys.exit()
            rows, cols = (self.nhalo, 3)
        vec = [[0]*cols]*rows
        vec=1.0*np.asarray(vec)
        vec[:,0]=self.x
        vec[:,1]=self.y
        vec[:,2]=self.z
        self.theta, self.phi = hp.pixelfunc.vec2ang(vec, lonlat=True)
        self.source_vector = hp.ang2vec(self.theta, self.phi, lonlat=True)
        print(f'Computing halo source vectors: {time.time() - self.job_start_time}s')

        return

    def get_patchy_screening_map(self, plot=False):

        # Incorporate patchy screening signal into primary CMB
        if self.signal == True:
            try:
                T_patchy_screening = -1 * self.tau_map.copy() * self.mock_CMB_primary.copy()
                # T_patchy_screening_2 = -1 * self.tau_map_2.copy() * self.mock_CMB_primary.copy()

            except AttributeError:
                self.generate_cmb_map(plot)
                self.load_lightcones(plot)
                T_patchy_screening = -1 * self.tau_map.copy() * self.mock_CMB_primary.copy()
                # T_patchy_screening_2 = -1 * self.tau_map_2.copy() * self.mock_CMB_primary.copy()
            self.T_cmb_ps = T_patchy_screening + self.mock_CMB_primary.copy()
            # self.T_cmb_ps_2 = T_patchy_screening_2 + self.mock_CMB_primary.copy()
            
            #alm = hp.map2alm(self.T_cmb_ps, lmax=3*self.nside)
            #self.T_cmb_ps = hp.alm2map(alm, nside=self.nside, lmax=5024)
            
        elif self.signal == False:
            try:
                self.T_cmb_ps = self.mock_CMB_primary.copy()

            except AttributeError:
                self.generate_cmb_map(plot)
                self.T_cmb_ps = self.mock_CMB_primary.copy()

        if plot == True:
            hp.mollview(self.T_cmb_ps, title="CMB temperature map w/ Patchy Screening", cmap="jet")#, min=-1.5e-4, max=1.5e-4)
            hp.graticule()
            plt.savefig(f'./Plots/T_ps_map_{self.boxname}_{self.simname}_{self.z_sample_name}.png', dpi=400)
            plt.clf()
        print(f'Generating patchy screening map: {time.time() - self.job_start_time}s')

        return


    def get_output_dir(self):
        fits_suffix = "" if self.cmb_method == "CAMB" else f"_{self.fits_file}"
        signal_suffix = "" if self.signal else "_no_ps"
        noise_suffix = "" if not self.rotate else "_noise"
        test_suffix = "" if self.test_name == None else f"_{self.test_name}"

        config_name = (
            f"mle_catalogue_nside{self.nside}_{self.cmb_method}"
            f"{fits_suffix}{signal_suffix}{noise_suffix}{test_suffix}"
            if self.mle else
            f"Mstar_bin{self.im_name}_{self.slope_name}_nside{self.nside}"
            f"_{self.cmb_method}{fits_suffix}{signal_suffix}{noise_suffix}{test_suffix}"
        )

        outdir = os.path.join(
            f"./data_files/tau_profiles/{self.boxname}",
            self.simname,
            self.z_sample_name,
            f"lightcone{self.lightcone}",
            "tmp",
            config_name,
            self.run_id,
        )

        os.makedirs(outdir, exist_ok=True)
        return outdir


    def cmb_tmp_file(self):
        return os.path.join(self.get_output_dir(), "cmb_branch.npz")


    def halo_tmp_file(self):
        return os.path.join(self.get_output_dir(), "halo_branch.npz")


    def balanced_halo_tmp_file(self):
        return os.path.join(self.get_output_dir(), "halo_branch_balanced.npz")


    def cleanup_tmp_dir(self):
        import shutil

        tmpdir = self.get_output_dir()

        # Safety checks: never delete broad directories by mistake
        if not os.path.isdir(tmpdir):
            print(f"No tmp directory to clean: {tmpdir}")
            return

        if "/tmp/" not in tmpdir and "/tmp" not in tmpdir:
            raise RuntimeError(f"Refusing to delete non-tmp directory: {tmpdir}")

        if not self.run_id or self.run_id in ("", ".", ".."):
            raise RuntimeError(f"Refusing to delete unsafe run_id: {self.run_id}")

        print(f"Deleting temporary directory: {tmpdir}")
        shutil.rmtree(tmpdir)


if __name__ == '__main__':

    ncpu = sys.argv[1]
    box = sys.argv[2]
    isim = sys.argv[3]
    iz = sys.argv[4]
    im = sys.argv[5]
    slope = sys.argv[6]
    fits = sys.argv[7]
    sig = sys.argv[8]
    mode = sys.argv[9] if len(sys.argv) > 9 else "full"
    run_id = sys.argv[10] if len(sys.argv) > 10 else None
    cleanup_tmp = sys.argv[11].lower() in ("true", "1", "yes", "y") if len(sys.argv) > 11 else False


    ps = patchyScreening(box, isim, iz, im, slope, ncpu, fits_file=fits, signal=sig, run_id=run_id, cleanup_tmp=cleanup_tmp)#, cmb_method='CAMB')#, lightcone_method=('SHELL','shell'))#, cmb_method='CAMB')
    ps.run_analysis(mode=mode, plot=False)
    # ps.balance_large_scale_signs(plot=False, seed=1000)
    # ps.reconstruct_tau_map(plot=True)
    quit()
    #ps.get_halo_coordinates()
    #ps.generate_cmb_map(plot=True)
    ps.get_patchy_screening_map(plot=True)

    unlensed_total_CL = hp.anafast(ps.T_cmb_ps)
    unlensed_total_CL_2 = hp.anafast(ps.T_cmb_ps_2)
    ell = np.arange(len(unlensed_total_CL))
    ell_2 = np.arange(len(unlensed_total_CL_2))

    ps_fits = patchyScreening(box, isim, iz, im, slope, ncpu, fits_file=fits, signal=sig)
    ps_fits.get_patchy_screening_map(plot=True)

    unlensed_total_CL_fits = hp.anafast(ps_fits.T_cmb_ps)
    ell_fits = np.arange(len(unlensed_total_CL_fits))

    print(ell, ell_2)
    print(unlensed_total_CL.shape)

    no_cut_tau = (ell*(ell+1)*unlensed_total_CL)/(2*np.pi)
    cut_tau = (ell_2*(ell_2+1)*unlensed_total_CL_2)/(2*np.pi)
    fits_result = (ell_fits*(ell_fits+1)*unlensed_total_CL_fits)/(2*np.pi)

    print(unlensed_total_CL_fits.shape)
    
    '''from mpl_toolkits.axes_grid1 import make_axes_locatable
    fig, ax = plt.subplots(figsize=(8,6), sharey=True, sharex=True)

    divider = make_axes_locatable(ax)
    ax2 = divider.append_axes("bottom", size="35%", pad=0)
    ax.figure.add_axes(ax2)

    ax.plot(ell, no_cut_tau, label='CMB cut at $\ell$=5024, no cut in tau')
    ax.plot(ell_2, cut_tau, label='CMB cut at $\ell$=5024, cut in tau map at $\ell$=5024')
    ax2.plot(ell, unlensed_total_CL_2/unlensed_total_CL, color="tab:red", label='$\frac{cut}{no cut}$')

    ax.set_xticks([])
    ax.set_xlabel(r'Multipole moment $\ell$')
    ax.set_ylabel(r'$\frac{\ell(\ell+1)C_{\ell}}{2\pi}$')
    ax.set_ylabel('Residual')
    ax.set_xscale('log')
    ax2.set_xscale('log')
    ax.set_title('Primary CMB comparison')
    ax.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(f'./Plots/cmb_comp_tau.png', dpi=400)
    plt.clf()'''

    plt.plot(ell, (ell*(ell+1)*unlensed_total_CL)/(2*np.pi), label='CMB cut at $\ell$=5024, no cut in tau')
    plt.plot(ell_2, (ell_2*(ell_2+1)*unlensed_total_CL_2)/(2*np.pi), label='CMB cut at $\ell$=5024, cut in tau map at $\ell$=5024')
    plt.plot(ell_fits, (ell_fits*(ell_fits+1)*unlensed_total_CL_fits)/(2*np.pi), label='FITS')
    
    plt.xlabel(r'Multipole moment $\ell$')
    plt.ylabel(r'$\frac{\ell(\ell+1)C_{\ell}}{2\pi}$')
    plt.xscale('log')
    plt.title('Primary CMB comparison')
    plt.legend(loc='upper right')
    plt.savefig(f'./Plots/cmb_comp_tau.png', dpi=400)
    plt.clf()
    
####################################################################################################################################
