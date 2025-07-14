import sys, os, pickle
os.environ["POLARS_MAX_THREADS"] = str(sys.argv[1])
import h5py, pandas as pd, numpy as np, polars as pl
import healpy as hp, matplotlib.pyplot as plt
import astropy.units as u
from astropy.cosmology import FlatLambdaCDM
from joblib import Parallel, delayed
import camb
from camb import model, initialpower
import time
from numbers import Real
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, wait


class patchyScreening:
    def __init__(self, isim, iz, im, n_cut, ncpu, theta_d=np.arange(0.5, 11, 0.5), nside=8192, cmb_method='FITS', fits_file='unlensed', lightcone_method=('FULL','dndz'), signal=True, rotate=False, rect_size=20):

        self.job_start_time = time.time()
        sim_list = ['HYDRO_FIDUCIAL','HYDRO_PLANCK','HYDRO_PLANCK_LARGE_NU_FIXED','HYDRO_PLANCK_LARGE_NU_VARY','HYDRO_STRONG_AGN','HYDRO_WEAK_AGN','HYDRO_LOW_SIGMA8','HYDRO_STRONGER_AGN','HYDRO_JETS_published','HYDRO_STRONGEST_AGN','HYDRO_STRONG_SUPERNOVA','HYDRO_STRONGER_AGN_STRONG_SUPERNOVA','HYDRO_STRONG_JETS']

        try:
            isim = int(isim)
            self.simname = sim_list[isim]                
        except (ValueError, IndexError):
            self.simname = str(isim)
            
        if self.simname not in sim_list:
            raise ValueError(f"‘{self.simname}’ is not a valid simulation name; choose one of:\n  {sim_list!r}")
        
        survey = {'Blue':11, 'Green':22, 'Red':30}
        try:
            isinstance(iz,Real)
            self.z_sample = int(iz)
            self.z_sample_name = f'Custom_shell_{iz}'
        except (ValueError, IndexError):
            self.z_sample = int(survey[iz])
            self.z_sample_name = iz
            
        self.im = 10**np.array(float(im))
        self.im_name = f"{float(im):.1f}".replace('.', 'p')
        self.slope = float(n_cut)
        self.slope_name = f"{float(n_cut):.1f}".replace('.', 'p')
        if self.slope < 0.0:
            self.slope_name = f"{self.slope_name}".replace('-', 'minus')
        self.ncpu = int(ncpu)
        self.theta_d = theta_d
        self.nside = nside
        self.cmb_method = cmb_method
        self.fits_file = str(fits_file)
        self.lightcone_method = lightcone_method
        self.signal = bool(signal)
        self.rotate = rotate
        self.rect_size = rect_size

        self.cosmology = FlatLambdaCDM(H0=68.1, Om0=0.3, Tcmb0=2.725)
        self.mock_CMB_primary = None

        
    def _path(self, name):
        return os.path.join(self.ckpt, f"{name}.pkl")

    
    def generate_cmb_map(self, plot=False):
        try:
            checkpoint_path = self._path("primary_cmb_map")
            if os.path.exists(checkpoint_path):
                with open(checkpoint_path, "rb") as f:
                    self.mock_CMB_primary = pickle.load(f)
        except AttributeError:
            pass
                
        # Generating primary CMB map with CAMB or loading pre-generated FITS
        np.random.seed(1000)
        if self.cmb_method == 'CAMB':
            pars = camb.set_params(H0=68.1, ombh2=0.048600*(0.681**2), omch2=0.256011*(0.681**2), mnu=0.06, As=2.099e-9, ns=0.967, lmax=3*self.nside-1)
            results = camb.get_results(pars)
            powers = results.get_cmb_power_spectra(pars, raw_cl=True, CMB_unit='muK')
            unlensed_total_CL = powers['unlensed_total']
            self.mock_CMB_primary = hp.synfast(unlensed_total_CL[:,0], nside=self.nside)
        elif self.cmb_method == 'FITS':
            lensed_dir = '/cosma8/data/dp004/dc-yang3/maps/L1000N1800/HYDRO_FIDUCIAL/lightcone0_shells/patchy_screening_folder'
            if self.fits_file == 'unlensed':
                self.mock_CMB_primary = hp.read_map(f'{lensed_dir}/CMB_T_map_unl.fits', dtype=np.float64, verbose=False)
            elif self.fits_file == 'lensed_z2':
                self.mock_CMB_primary = hp.read_map(f'{lensed_dir}/CMB_T_map_l_kappa_z2.fits', dtype=np.float64, verbose=False)
            elif self.fits_file == 'lensed_z3':
                self.mock_CMB_primary = hp.read_map(f'{lensed_dir}/CMB_T_map_l_kappa_z3.fits', dtype=np.float64, verbose=False)
        else:
            raise ValueError("Unknown CMB map generation method")
        print(self.mock_CMB_primary)
        if plot == True:
            hp.mollview(self.mock_CMB_primary, title=f"Mock Primary CMB temperature map (sim={self.simname})", cmap="jet")#, min=-1.5e-4, max=1.5e-4)
            hp.graticule()
            plt.savefig(f'./Plots/primary_CMB_map_{self.simname}.png', dpi=400)
            plt.clf()
        print(f'Generating mock primary CMB: {time.time() - self.job_start_time}s')

        try:
            with open(checkpoint_path, "wb") as f:
                pickle.dump(self.mock_CMB_primary, f)
        except NameError:
            pass
        return

    
    def load_lightcones(self, plot=False):
        try:
            checkpoint_path = self._path("lightcone_maps")
            if os.path.exists(checkpoint_path):
                with open(checkpoint_path, "rb") as f:
                    self.DM_map = pickle.load(f)
        except AttributeError:
            pass
                
        # Loading tau map from FLAMINGO lightcone shells
        if self.lightcone_method[0] == 'SHELL':
            map_lightcone = f'/cosma8/data/dp004/flamingo/Runs/L1000N1800/{self.simname}/neutrino_corrected_maps/lightcone0_shells/shell_{self.z_sample}/lightcone0.shell_{self.z_sample}.0.hdf5'
            g = h5py.File(map_lightcone,'r')
            conversion_factor = g['DM'].attrs['Conversion factor to CGS (not including cosmological corrections)']
            DM = g['DM'][...]*conversion_factor*6.6524587321e-25 #6.65246e-25 = Thomson cross-section (in cgs)
            redshift = g['DM'].attrs['Central redshift assumed for correction']
            g.close()
            print(f'Map z of {self.z_sample_name} sample = {redshift} (shell: {self.z_sample})')
            print(DM)
            print(f'Loading first lightcone shell: {time.time() - self.job_start_time}s')

            if plot == True:
                hp.mollview(DM, title=f"DM map (sim={self.simname}, lightcone shell={self.z_sample})", cmap="jet", min=2e-5, max=2e-3)
                hp.graticule()
                plt.savefig(f'./Plots/DM_map_{self.simname}_{self.z_sample_name}_shell_{self.z_sample}.png', dpi=400)
                plt.clf()

            map_lightcone_lower = f'/cosma8/data/dp004/flamingo/Runs/L1000N1800/{self.simname}/neutrino_corrected_maps/lightcone0_shells/shell_{self.z_sample-1}/lightcone0.shell_{self.z_sample-1}.0.hdf5'
            g_low = h5py.File(map_lightcone_lower,'r')
            conversion_factor = g_low['DM'].attrs['Conversion factor to CGS (not including cosmological corrections)']
            DM += g_low['DM'][...]*conversion_factor*6.6524587321e-25
            g_low.close()

            map_lightcone_higher = f'/cosma8/data/dp004/flamingo/Runs/L1000N1800/{self.simname}/neutrino_corrected_maps/lightcone0_shells/shell_{self.z_sample+1}/lightcone0.shell_{self.z_sample+1}.0.hdf5'
            g_high = h5py.File(map_lightcone_higher,'r')
            conversion_factor = g_high['DM'].attrs['Conversion factor to CGS (not including cosmological corrections)']
            DM += g_high['DM'][...]*conversion_factor*6.6524587321e-25
            g_high.close()

        elif self.lightcone_method[0] == 'FULL':
            DM = hp.read_map(f'/cosma8/data/dp004/dc-conl1/FLAMINGO/patchy_screening/data_files/DM_maps/stacked_DM_map_z3p0.fits', dtype=np.float64, verbose=False)

        self.DM_map = hp.pixelfunc.ud_grade(DM,self.nside)
        print(self.DM_map)
        if self.lightcone_method[0] == 'SHELL':
            print(f"Map lightcone z = [{g_low['DM'].attrs['Central redshift assumed for correction']},{g_high['DM'].attrs['Central redshift assumed for correction']}]")
        elif self.lightcone_method[0] == 'FULL':
            print(f"Map lightcone integrated up to z=3")
        if plot == True:
            if self.lightcone_method[0] == 'SHELL':
                hp.mollview(self.DM_map, title=f"DM map (sim={self.simname}, lightcone shell={self.z_sample-1}+{self.z_sample}+{self.z_sample+1})", cmap="jet")#, min=2e-5, max=2e-3)
                hp.graticule()
                plt.savefig(f'./Plots/DM_map_{self.simname}_{self.z_sample_name}_shell_{self.z_sample-1}-{self.z_sample+1}.png', dpi=400)
            elif self.lightcone_method[0] == 'FULL':
                hp.mollview(self.DM_map, title=f"Stacked DM map, integrated up to z=3 (sim={self.simname})", cmap="jet")#, min=2e-5, max=2e-3)
                hp.graticule()
                plt.savefig(f'./Plots/DM_map_{self.simname}_stacked_z3p0.png', dpi=400)
            plt.clf()
        print(f'Loading relevant lightcone shells: {time.time() - self.job_start_time}s')

        try:
            with open(checkpoint_path, "wb") as f:
                pickle.dump(self.DM_map, f)
        except NameError:
            pass
        return

    
    def load_halo_data(self, lightcone_type='HBT'):
        try:
            checkpoint_path = self._path("halo_lightcones")
            if os.path.exists(checkpoint_path):
                with open(checkpoint_path, "rb") as f:
                    data = pickle.load(f)
                self.Dcom = data["Dcom"]
                halo_lc_data = data["halo_lc_data"]
                if lightcone_type == 'HBT':
                    df_HBT = data["df_HBT"]
                elif lightcone_type == 'VR':
                    df_VR = data["df_VR"]
        except AttributeError:
            pass
        
        # Load halo lightcone and SOAP data into DataFrames
        if lightcone_type == 'HBT':
            halo_lightcone = f'/cosma8/data/dp004/flamingo/Runs/L1000N1800/{self.simname}/hbt_lightcone_halos/lightcone0/lightcone_halos_{77-self.z_sample:04d}.hdf5'
            f = h5py.File(halo_lightcone, 'r')
            halo_lc_data = pl.DataFrame({
                'ID':          f['InputHalos/HaloCatalogueIndex'][...],
                'SnapNum':     f['Lightcone/SnapshotNumber'][...],
                'z':           f['Lightcone/Redshift'][...],
                'xminpot':     f['Lightcone/HaloCentre'][...][:,0],
                'yminpot':     f['Lightcone/HaloCentre'][...][:,1],
                'zminpot':     f['Lightcone/HaloCentre'][...][:,2],
            }).sort('ID')
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
        snap = int(halo_lc_data['SnapNum'].first())
        print(f'D_com = {self.Dcom}, Snap number = {snap}')

        if lightcone_type == 'HBT':
            HBT_file = f'/cosma8/data/dp004/flamingo/Runs/L1000N1800/{self.simname}/SOAP-HBT/halo_properties_{snap:04d}.hdf5'
            f = h5py.File(HBT_file, 'r')
            df_HBT = pl.DataFrame({
                'ID':          f['InputHalos/HaloCatalogueIndex'][...],
                'Structuretype': f['InputHalos/IsCentral'][...],
                'mvir':       f['SO/500_crit/TotalMass'][...] * 1e10,
                'mstar':       f['ExclusiveSphere/50kpc/StellarMass'][...] * 1e10,
            })
            f.close()
            print(f'Loading halo lightcone data: {time.time() - self.job_start_time}s')

            try:
                data = {
                    "Dcom": self.Dcom,
                    "halo_lc_data": halo_lc_data,
                    "df_HBT": df_HBT
                }
                with open(p,"wb") as f:
                    pickle.dump(data, f)
            except NameError:
                pass
            return halo_lc_data, df_HBT
        elif lightcone_type == 'VR':
            VR_file = f'/cosma8/data/dp004/flamingo/Runs/L1000N1800/{self.simname}/SOAP/halo_properties_{snap:04d}.hdf5'
            f = h5py.File(HBT_file, 'r')
            df_VR = pd.DataFrame()
            df_VR['ID'] = f['VR/ID'][...]
            #df_VR['hostHaloID'] = f['VR/HostHaloID'][...]
            df_VR['Structuretype'] = f['VR/StructureType'][...]
            df_VR['m_vir'] = f['SO/500_crit/TotalMass'][...]
            df_VR['mstar'] = f['ExclusiveSphere/50kpc/StellarMass'][...]
            f.close()
            print(f'Loading halo lightcone data: {time.time() - self.job_start_time}s')

            try:
                data = {
                    "Dcom": self.Dcom,
                    "halo_lc_data": halo_lc_data,
                    "df_VR": df_VR
                }
                with open(p,"wb") as f:
                    pickle.dump(data, f)
            except NameError:
                pass
            return halo_lc_data, df_VR

        
    def filter_stellar_mass(self, halo_lc_data=None, df_halo=None):
        try:
            checkpoint_path = self._path("mock_halo_catalog")
            if os.path.exists(checkpoint_path):
                with open(checkpoint_path, "rb") as f:
                    data = pickle.load(f)
                self.merge = data["merge"]
                self.Dcom = data["Dcom"] 
                self.x = data["x"]
                self.y = data["y"]
                self.z = data["z"]
                self.nhalo = data["nhalo"]
        except AttributeError:
            pass
        
        # Merge and filter the DataFrames based on stellar mass bin
        if self.lightcone_method[1] == 'shell':
            if halo_lc_data is None or df_halo is None:
                halo_lc_data, df_halo = self.load_halo_data()
                
            df_mass = df_halo.filter(pl.col('mstar') > self.im)
            self.merge = df_mass.join(halo_lc_data, on='ID', how='inner').sort('ID')
            if self.merge.is_empty():
                self.merge = np.nan
        elif self.lightcone_method[1] == 'dndz':
            self.merge = pl.read_parquet(
                f"/cosma8/data/dp004/dc-conl1/FLAMINGO/patchy_screening/data_files/mock_halo_catalogs/sampled_halo_data_{self.simname}_{self.z_sample_name}_{self.im_name}_{self.slope_name}.parquet"
            )
            mean_z = {'Blue':0.6, 'Green':1.1, 'Red':1.5} 
            Dcom = self.cosmology.comoving_distance(mean_z[self.z_sample_name])*0.681  # comoving distance to galaxy in Mpc/h
            self.Dcom = Dcom.value
            
        if isinstance(self.merge, float) and np.isnan(self.merge):
            print(f"No halos in shell = {int(halo_lc_data['SnapNum'].first())} for a stellar cut = {np.log10(self.im)}")
            self.nhalo = 0
            return
        else:
            self.x = self.merge['xminpot'].to_numpy()
            self.y = self.merge['yminpot'].to_numpy()
            self.z = self.merge['zminpot'].to_numpy()
            mvir = self.merge['mvir'].to_numpy()
            mstar = self.merge['mstar'].to_numpy()
            self.nhalo = mvir.size
            print(self.nhalo)
            print(np.log10(self.im), np.log10(np.min(mstar)), np.log10(np.mean(mstar)), np.log10(np.mean(mvir)), self.nhalo)
            print(f'Identifying stackable objects: {time.time() - self.job_start_time}s')

            try:
                data = {
                    "merge": self.merge,
                    "Dcom": self.Dcom,
                    "x": self.x,
                    "y": self.y,
                    "z": self.z,
                    "nhalo": self.nhalo
                }
                with open(p,"wb") as f:
                    pickle.dump(data, f)
            except NameError:
                pass
            return

    def compute_alm_maps(self, plot=False):
        try:
            checkpoint_path = self._path("alm_filtered_maps")
            if os.path.exists(checkpoint_path):
                with open(checkpoint_path, "rb") as f:
                    data = pickle.load(f)
                self.large_scale_map = data["large_scale_map"]
                self.small_scale_map = data["small_scale_map"]
                self.mean_mod_T_large_scale = data["mean_mod_T_large_scale"]
        except AttributeError:
            pass
                    
        # Computing and filtering spherical harmonic coefficients to create large and small scale maps
        try:
            alm = hp.map2alm(self.T_cmb_ps, lmax=3*self.nside-1)
        except AttributeError:
            self.get_patchy_screening_map(plot)
            alm = hp.map2alm(self.T_cmb_ps, lmax=3*self.nside-1)
        ell, m = hp.Alm.getlm(lmax=3*self.nside-1)
        if self.rotate == True:
            np.random.seed(int(sys.argv[-1]))
            rotated_alm = hp.Rotator(deg=True, rot=(np.random.uniform(0, 180), np.random.uniform(0, 360))).rotate_alm(alm, lmax=3*self.nside-1)
            alm = rotated_alm
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
        self.large_scale_map = hp.alm2map(lowpass_alm, nside=self.nside, lmax=3*self.nside-1)
        self.small_scale_map = hp.alm2map(highpass_alm, nside=self.nside, lmax=3*self.nside-1)
        if plot == True:
            hp.mollview(self.large_scale_map.copy(), title=f"Large scale CMB temperature map (sim={self.simname})", cmap="jet")#, min=-1.5e-4, max=1.5e-4)
            hp.graticule()
            plt.savefig(f'./Plots/T_ps_map_large_scale_{self.simname}_{self.z_sample_name}_{self.im_name}_{self.slope_name}.png', dpi=400)
            plt.clf()
            hp.mollview(self.small_scale_map.copy(), title=f"Small scale CMB temperature map (sim={self.simname})", cmap="jet")#, min=-1e-6, max=1e-6)
            hp.graticule()
            plt.savefig(f'./Plots/T_ps_map_small_scale_{self.simname}_{self.z_sample_name}_{self.im_name}_{self.slope_name}.png', dpi=400)
            plt.clf()
        self.mean_mod_T_large_scale = np.mean(np.abs(self.large_scale_map.copy()))
        print(self.mean_mod_T_large_scale)
        if self.rotate == True:
            print(f'Computing, rotating and filtering alms: {time.time() - self.job_start_time}s')
        elif self.rotate == False:
            print(f'Computing and filtering alms: {time.time() - self.job_start_time}s')

        try:
            data = {
                "large_scale_map": self.large_scale_map,
                "small_scale_map": self.small_scale_map,
                "mean_mod_T_large_scale": self.mean_mod_T_large_scale
            }
            with open(p,"wb") as f:
                pickle.dump(data, f)
        except NameError:
            pass
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

    def tau_prof(self, i, plot=(False,False)):
        # Compute the tau profiles in annuli around halo centres
        halo_pixels = hp.query_disc(self.nside, self.source_vector[i,:], radius=(0.25*u.deg).to_value(u.radian))
        theta_pix, phi_pix = hp.pix2ang(self.nside, halo_pixels, lonlat=True)
        rtheta = hp.rotator.angdist([self.theta[i], self.phi[i]], [theta_pix, phi_pix], lonlat=True)*60.0*180.0/np.pi
        large_scale_fluxes=np.array(self.large_scale_map[halo_pixels].copy())
        small_scale_fluxes=np.array(self.small_scale_map[halo_pixels].copy())
        if plot[0] == True and i == plot[1]:
            tau_2D = (np.sign(large_scale_fluxes.copy())*small_scale_fluxes.copy())/self.mean_mod_T_large_scale.copy()
            grid_x = np.linspace(min(theta_pix.copy()*60.0), max(theta_pix.copy()*60.0), 25)  # Adjust grid size as needed
            grid_y = np.linspace(min(phi_pix.copy()*60.0), max(phi_pix.copy()*60.0), 25)
            temperature_map = self.T_cmb_ps[halo_pixels].copy()
            grid_values_temp, xedges_temp, yedges_temp = np.histogram2d(theta_pix.copy()*60.0, phi_pix.copy()*60.0, bins=[grid_x, grid_y], weights=temperature_map)
            plt.figure(figsize=(6, 6))
            plt.imshow(grid_values_temp.T, origin='lower', extent=[grid_x.min(), grid_x.max(), grid_y.min(), grid_y.max()], cmap='inferno')#, vmin=-1.5e-4, vmax=1.5e-4)
            plt.colorbar(label=r'Temperature ($\mu$K)')
            plt.xlabel('X (arcmin)')
            plt.ylabel('Y (arcmin)')
            plt.title(f'Rectangular Cutout Around Halo = {i}', wrap=True)
            plt.savefig(f'./Plots/random_cutout_T_ps_map_{self.simname}_{self.z_sample_name}.png', dpi=400)
            plt.clf()
            plt.close('all')
            grid_values_tau = np.histogram2d(theta_pix.copy()*60.0, phi_pix.copy()*60.0, bins=[grid_x, grid_y], weights=tau_2D)
            plt.figure(figsize=(6, 6))
            plt.imshow(grid_values_tau[0].T, origin='lower', extent=[grid_x.min(), grid_x.max(), grid_y.min(), grid_y.max()], cmap='inferno')#, vmin=-1e-6, vmax=1e-6)
            plt.colorbar(label='Optical Depth')
            plt.xlabel('X (arcmin)')
            plt.ylabel('Y (arcmin)')
            plt.title(f'Rectangular Cutout Around Halo = {i} w/ Patchy Screening', wrap=True)
            plt.savefig(f'./Plots/random_cutout_T_filtered_map_{self.simname}_{self.z_sample_name}.png', dpi=400)
            plt.clf()
            plt.close('all')
        tau_1D = np.zeros(len(self.theta_d))
        for j in range(len(self.theta_d)):
            idx_in = np.where((rtheta > self.theta_d[j]-0.25) & (rtheta <= self.theta_d[j]+0.25))
            tau_1D[j] = (-1*np.mean(np.sign(large_scale_fluxes[idx_in])*small_scale_fluxes[idx_in]))/self.mean_mod_T_large_scale
        return tau_1D

    def run_tau_profiles(self, plot):
        # Parallelisation of computing tau profiles for each halo
        batch_size=max(1, self.nhalo // (self.ncpu*2))
        randint = np.random.randint(self.nhalo)        
        print(f'Starting profile loop: {time.time() - self.job_start_time}s')
        
        results = Parallel(n_jobs=self.ncpu, backend="loky", batch_size=batch_size)(delayed(self.tau_prof)(i, (plot,randint)) for i in range(self.nhalo))
        self.data_1D = np.asarray(results)
        print(f'Ending profile loop: {time.time() - self.job_start_time}s')
        return

    def stack_and_save(self):
        # Stacking of tau profiles and save as pickle files
        tau_1D_stack = np.zeros(len(self.theta_d))
        for i in range(self.nhalo):
            tau_1D = self.data_1D[i,:]
            tau_1D_stack += tau_1D
        tau_1D_stack /= self.nhalo
        print(self.simname,tau_1D_stack)
        
        rows, cols = (len(self.theta_d), 4)
        data = [0]*cols
        data[0] = self.theta_d
        data[1] = tau_1D_stack
        data[2] = (self.theta_d*np.pi/(180.0*60.0))*self.Dcom
        data[3] = self.nhalo
        fits_suffix = "" if self.cmb_method=='CAMB' else f"_{self.fits_file}"
        signal_suffix = "" if self.signal==True else "_no_ps"
        noise_suffix = "" if self.rotate==False else "_noise"
        outfile = os.path.join('./L1000N1800', self.z_sample_name, f'{self.simname}_tau_Mstar_bin{self.im_name}_{self.slope_name}_nside{self.nside}_{self.cmb_method}{fits_suffix}{signal_suffix}{noise_suffix}.pickle')
        os.makedirs(os.path.dirname(outfile), exist_ok=True)
        with open(outfile, 'wb') as f:
            pickle.dump(data, f)
        f.close()
        print(f'Writing out data: {time.time() - self.job_start_time}s')
        return

    def _run_cmb_branch(self, proc_workers, plot=False):
        self.generate_cmb_map(plot)          # reads self.CAMB_params, writes self.cmb_map
        self.load_lightcones(plot)           # reads self.lightcone_params, writes self.lightcones
        self.get_patchy_screening_map(plot)  # reads cmb_map & lightcones, writes self.patchy_map

        # compute_alm_maps is CPU‐heavy, so run it in its own process:
        with ProcessPoolExecutor(max_workers=proc_workers) as proc:
            alm_process = proc.submit(self.compute_alm_maps(plot))

    def _run_halo_branch(self):
        if self.lightcone_method[1] == 'shell':
            halo_lc_data, df_halo = self.load_halo_data()
            self.filter_stellar_mass(halo_lc_data, df_halo)
        elif self.lightcone_method[1] == 'dndz':
            self.filter_stellar_mass()     # reads self.lightcone_method, writes self.filtered_halos
        self.get_halo_coordinates()    # reads filtered_halos, writes self.halo_coords

    def run_analysis(self, plot=False):
        jobid = os.environ.get("SLURM_JOB_ID", "nojobid")
        self.ckpt = os.environ.get('CHECKPOINT_DIR',
                                  f"/cosma8/data/dp004/dc-conl1/FLAMINGO/patchy_screening/batch_files/checkpoints/{jobid}")
        os.makedirs(self.ckpt, exist_ok=True)
        
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
        self.run_tau_profiles(plot)
        self.stack_and_save()

        return

    def get_halo_coordinates(self):
        try:
            checkpoint_path = self._path("source_vector")
            if os.path.exists(checkpoint_path):
                with open(checkpoint_path, "rb") as f:
                    data = pickle.load(f)
                self.theta = data["theta"]
                self.phi = data["phi"]
                self.source_vector = data["source_vector"]
        except AttributeError:
            pass

        # Compute source vectors of each halo
        try:
            rows, cols = (self.nhalo, 3)
        except AttributeError:
            self.filter_stellar_mass()
            if self.nhalo == 0:
                print("No halos to compute coordinates")
                quit()
            rows, cols = (self.nhalo, 3)
        vec = [[0]*cols]*rows
        vec=1.0*np.asarray(vec)
        vec[:,0]=self.x
        vec[:,1]=self.y
        vec[:,2]=self.z
        self.theta, self.phi = hp.pixelfunc.vec2ang(vec, lonlat=True)
        self.source_vector = hp.ang2vec(self.theta, self.phi, lonlat=True)
        print(f'Computing halo source vectors: {time.time() - self.job_start_time}s')

        try:
            data = {
                "theta": self.theta,
                "phi": self.phi,
                "source_vector": self.source_vector
            }
            with open(checkpoint_path, "wb") as f:
                pickle.dump(data, f)
        except NameError:
            pass
        return

    def get_patchy_screening_map(self, plot=False):
        try:
            checkpoint_path = self._path("patchy_screening_T_map")
            if os.path.exists(checkpoint_path):
                with open(checkpoint_path, "rb") as f:
                    self.T_cmb_ps = pickle.load(f)
        except AttributeError:
            pass

        # Incorporate patchy screening signal into primary CMB
        if self.signal == True:
            try:
                T_patchy_screening = -1 * self.DM_map.copy() * self.mock_CMB_primary.copy()
            except AttributeError:
                self.generate_cmb_map(plot)
                self.load_lightcones(plot)
                T_patchy_screening = -1 * self.DM_map.copy() * self.mock_CMB_primary.copy()
            self.T_cmb_ps = T_patchy_screening + self.mock_CMB_primary.copy()
            self.T_cmb_ps = hp.smoothing(self.T_cmb_ps,fwhm=1.3*np.pi/60.0/180.0)
        elif self.signal == False:
            try:
                self.T_cmb_ps = hp.smoothing(self.mock_CMB_primary.copy(),fwhm=1.3*np.pi/60.0/180.0)
            except AttributeError:
                self.generate_cmb_map(plot)
                self.T_cmb_ps = hp.smoothing(self.mock_CMB_primary.copy(),fwhm=1.3*np.pi/60.0/180.0)
        if plot == True:
            hp.mollview(self.T_cmb_ps, title="CMB temperature map w/ Patchy Screening", cmap="jet")#, min=-1.5e-4, max=1.5e-4)
            hp.graticule()
            plt.savefig(f'./Plots/T_ps_map_{self.simname}_{self.z_sample_name}.png', dpi=400)
            plt.clf()
        print(f'Generating patchy screening map: {time.time() - self.job_start_time}s')

        try:
            with open(checkpoint_path, "wb") as f:
                pickle.dump(self.T_cmb_ps, f)
        except NameError:
            pass
        return

if __name__ == '__main__':

    ncpu = sys.argv[1]
    isim = sys.argv[2]
    iz = sys.argv[3]
    im = sys.argv[4]
    slope = sys.argv[5]
    fits = sys.argv[6]
    sig = sys.argv[7]

    ps = patchyScreening(isim, iz, im, slope, ncpu, fits_file=fits, signal=sig, cmb_method='CAMB')
    ps.run_analysis(plot=False)
    #ps.get_halo_coordinates()
    #ps.generate_cmb_map()

####################################################################################################################################
