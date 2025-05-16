import sys, os, pickle
import h5py, pandas as pd, numpy as np
import healpy as hp, matplotlib.pyplot as plt
import astropy.units as u
from astropy.cosmology import FlatLambdaCDM
from joblib import Parallel, delayed
import camb
from camb import model, initialpower
import time

class patchyScreening:
    def __init__(self, isim, iz, im, im_name, ncpu, theta_d, nside=8192, cmb_method='FITS', fits_file='unlensed', lightcone_method='FULL', signal=True, rotate=False, rect_size=20):

        self.job_start_time = time.time()
        sim_list = ['HYDRO_FIDUCIAL','HYDRO_PLANCK','HYDRO_PLANCK_LARGE_NU_FIXED','HYDRO_PLANCK_LARGE_NU_VARY','HYDRO_STRONG_AGN','HYDRO_WEAK_AGN','HYDRO_LOW_SIGMA8','HYDRO_STRONGER_AGN','HYDRO_JETS','HYDRO_STRONGEST_AGN','HYDRO_STRONG_SUPERNOVA','HYDRO_STRONGER_AGN_STRONG_SUPERNOVA','HYDRO_STRONG_JETS']
        sim_list2=['HYDRO_FIDUCIAL','HYDRO_PLANCK','HYDRO_PLANCK_LARGE_NU_FIXED','HYDRO_PLANCK_LARGE_NU_VARY','HYDRO_STRONG_AGN','HYDRO_WEAK_AGN','HYDRO_LOW_SIGMA8','HYDRO_STRONGER_AGN','HYDRO_JETS_old','HYDRO_STRONGEST_AGN','HYDRO_STRONG_SUPERNOVA','HYDRO_STRONGER_AGN_STRONG_SUPERNOVA','HYDRO_STRONG_JETS_old']

        self.simname = sim_list[isim]
        self.simname2 = sim_list2[isim]
        survey = {'Blue':11, 'Green':22, 'Red':30}
        obs_samples = {'Blue':8692981, 'Green':3645917}
        if isinstance(iz, str):
            self.z_sample = int(survey[iz])
            self.z_sample_name = iz
            self.nsamp = obs_samples[iz]
        elif isinstance(iz,int) or isinstance(iz,float):
            self.z_sample = int(iz)
            self.z_sample_name = f'Custom_shell_{iz}'
        self.im = im
        self.im_name = im_name
        self.ncpu = ncpu
        self.theta_d = theta_d
        self.nside = nside
        self.cmb_method = cmb_method
        self.fits_file = fits_file
        self.lightcone_method = lightcone_method
        self.signal = signal
        self.rotate = rotate
        self.rect_size = rect_size

        self.cosmology = FlatLambdaCDM(H0=68.1, Om0=0.3, Tcmb0=2.725)
        self.mock_CMB_primary = None

    def generate_cmb_map(self, plot=False):
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
            plt.savefig(f'./Plots/primary_CMB_map_{self.simname}.png', dpi=1200)
            plt.clf()
        print(f'Generating mock primary CMB: {time.time() - self.job_start_time}s')
        return

    def load_lightcones(self, plot=False):
        # Loading tau map from FLAMINGO lightcone shells
        if self.lightcone_method == 'SHELL':
            map_lightcone = f'/cosma8/data/dp004/flamingo/Runs/L1000N1800/{self.simname2}/neutrino_corrected_maps/lightcone0_shells/shell_{self.z_sample}/lightcone0.shell_{self.z_sample}.0.hdf5'
            g = h5py.File(map_lightcone,'r')
            conversion_factor = g['DM'].attrs['Conversion factor to CGS (not including cosmological corrections)']
            DM = g['DM'][...]*conversion_factor*6.6524587321e-25 #6.65246e-25 = Thomson cross-section (in cgs)
            redshift = g['DM'].attrs['Central redshift assumed for correction']
            print(f'Map z of {self.z_sample_name} sample = {redshift} (shell: {self.z_sample})')
            print(DM)
            print(f'Loading first lightcone shell: {time.time() - self.job_start_time}s')

            if plot == True:
                hp.mollview(DM, title=f"DM map (sim={self.simname2}, lightcone shell={self.z_sample})", cmap="jet", min=2e-5, max=2e-3)
                hp.graticule()
                plt.savefig(f'./Plots/DM_map_{self.simname2}_{self.z_sample_name}_shell_{self.z_sample}.png', dpi=1200)
                plt.clf()

            map_lightcone_lower = f'/cosma8/data/dp004/flamingo/Runs/L1000N1800/{self.simname2}/neutrino_corrected_maps/lightcone0_shells/shell_{self.z_sample-1}/lightcone0.shell_{self.z_sample-1}.0.hdf5'
            g_low = h5py.File(map_lightcone_lower,'r')
            conversion_factor = g_low['DM'].attrs['Conversion factor to CGS (not including cosmological corrections)']
            DM += g_low['DM'][...]*conversion_factor*6.6524587321e-25

            map_lightcone_higher = f'/cosma8/data/dp004/flamingo/Runs/L1000N1800/{self.simname2}/neutrino_corrected_maps/lightcone0_shells/shell_{self.z_sample+1}/lightcone0.shell_{self.z_sample+1}.0.hdf5'
            g_high = h5py.File(map_lightcone_higher,'r')
            conversion_factor = g_high['DM'].attrs['Conversion factor to CGS (not including cosmological corrections)']
            DM += g_high['DM'][...]*conversion_factor*6.6524587321e-25

        elif self.lightcone_method == 'FULL':
            DM = hp.read_map(f'/cosma8/data/dp004/dc-conl1/FLAMINGO/patchy_screening/stacked_DM_map_{self.z_sample_name}.fits', dtype=np.float64, verbose=False)

        self.DM_map = hp.pixelfunc.ud_grade(DM,self.nside)
        print(self.DM_map)
        if self.lightcone_method == 'SHELL':
            print(f"Map lightcone z = [{g_low['DM'].attrs['Central redshift assumed for correction']},{g_high['DM'].attrs['Central redshift assumed for correction']}]")
        elif self.lightcone_method == 'FULL':
            print(f"Map lightcone integrated up to shell: {self.z_sample}")
        if plot == True:
            if self.lightcone_method == 'SHELL':
                hp.mollview(self.DM_map, title=f"DM map (sim={self.simname2}, lightcone shell={self.z_sample-1}+{self.z_sample}+{self.z_sample+1})", cmap="jet")#, min=2e-5, max=2e-3)
                hp.graticule()
                plt.savefig(f'./Plots/DM_map_{self.simname2}_{self.z_sample_name}_shell_{self.z_sample-1}-{self.z_sample+1}.png', dpi=1200)
            elif self.lightcone_method == 'FULL':
                hp.mollview(self.DM_map, title=f"Stacked DM map (sim={self.simname2}, lightcone shells=0-{self.z_sample+1})", cmap="jet")#, min=2e-5, max=2e-3)
                hp.graticule()
                plt.savefig(f'./Plots/DM_map_{self.simname2}_{self.z_sample_name}_stacked.png', dpi=1200)
            plt.clf()
        print(f'Loading relevant lightcone shells: {time.time() - self.job_start_time}s')
        return

    def load_halo_data(self, lightcone_type='HBT'):
        # Load halo lightcone and SOAP data into DataFrames
        halo_lc_data = pd.DataFrame()
        if lightcone_type == 'HBT':
            halo_lightcone = f'/cosma8/data/dp004/jch/FLAMINGO/HBT/L1000N1800/{self.simname}/lightcone_halos/lightcone0/lightcone_halos_{77-self.z_sample:04d}.hdf5'
            f = h5py.File(halo_lightcone, 'r')
            halo_lc_data['ID'] = f['InputHalos/HaloCatalogueIndex'][...]
            halo_lc_data['SnapNum'] = f['Lightcone/SnapshotNumber'][...]
            halo_lc_data['z'] = f['Lightcone/Redshift'][...]
            halo_centre = f['Lightcone/HaloCentre'][...]
            halo_lc_data['xminpot'] = halo_centre[:,0]
            halo_lc_data['yminpot'] = halo_centre[:,1]
            halo_lc_data['zminpot'] = halo_centre[:,2]
        elif lightcone_type == 'VR':
            halo_lightcone = f'/cosma8/data/dp004/jch/FLAMINGO/lightcone_halos/{self.simname}/lightcone_halos/lightcone0/lightcone_halos_{self.z_sample:04d}.hdf5'
            f = h5py.File(halo_lightcone, 'r')
            halo_lc_data['ID'] = f['Subhalo/ID'][...]
            halo_lc_data['SnapNum'] = f['Subhalo/SnapNum'][...]
            halo_lc_data['z'] = f['Subhalo/LightconeRedshift'][...]
            halo_lc_data['xminpot'] = f['Subhalo/LightconeXcminpot'][...]
            halo_lc_data['yminpot'] = f['Subhalo/LightconeYcminpot'][...]
            halo_lc_data['zminpot'] = f['Subhalo/LightconeZcminpot'][...]

        print(f'Halo lightcone z = [{np.min(halo_lc_data.z)},{np.max(halo_lc_data.z)}]')
        Dcom = self.cosmology.comoving_distance(np.mean(halo_lc_data.z))*0.681  # comoving distance to galaxy in Mpc/h
        self.Dcom = Dcom.value
        halo_lc_data.sort_values(by='ID', inplace=True)
        halo_lc_data.reset_index(inplace=True, drop=True)
        snap = int(halo_lc_data.iloc[0]['SnapNum'])
        print(f'D_com = {self.Dcom}, Snap number = {snap}')

        if lightcone_type == 'HBT':
            HBT_file = f'/cosma8/data/dp004/flamingo/Runs/L1000N1800/{self.simname2}/SOAP-HBT/halo_properties_{snap:04d}.hdf5'
            f = h5py.File(HBT_file, 'r')
            df_HBT = pd.DataFrame()
            df_HBT['ID'] = f['InputHalos/HaloCatalogueIndex'][...]
            #df_HBT['hostHaloID'] = f['HBT/HostHaloID'][...]
            df_HBT['Structuretype'] = f['InputHalos/IsCentral'][...]
            df_HBT['m_vir'] = f['SO/500_crit/TotalMass'][...]
            df_HBT['mstar'] = f['ExclusiveSphere/50kpc/StellarMass'][...]
            df_HBT.mstar*=1e10
            df_HBT.m_vir*=1e10
            print(f'Loading halo lightcone data: {time.time() - self.job_start_time}s')
            return halo_lc_data, df_HBT
        elif lightcone_type == 'VR':
            VR_file = f'/cosma8/data/dp004/flamingo/Runs/L1000N1800/{self.simname2}/SOAP/halo_properties_{snap:04d}.hdf5'
            f = h5py.File(HBT_file, 'r')
            df_VR = pd.DataFrame()
            df_VR['ID'] = f['VR/ID'][...]
            #df_VR['hostHaloID'] = f['VR/HostHaloID'][...]
            df_VR['Structuretype'] = f['VR/StructureType'][...]
            df_VR['m_vir'] = f['SO/500_crit/TotalMass'][...]
            df_VR['mstar'] = f['ExclusiveSphere/50kpc/StellarMass'][...]
            print(f'Loading halo lightcone data: {time.time() - self.job_start_time}s')
            return halo_lc_data, df_VR

    def filter_stellar_mass(self, halo_lc_data=None, df_halo=None):        
        # Merge and filter the DataFrames based on stellar mass bin
        if halo_lc_data is None or df_halo is None:
            halo_lc_data, df_halo = self.load_halo_data()
        df_mass = df_halo
        df_mass = df_mass.loc[df_mass.mstar > self.im]
        df_mass.sort_values(by='ID', inplace=True)
        df_mass.reset_index(inplace=True, drop=True)
        self.merge = pd.merge_ordered(df_mass, halo_lc_data, on=['ID'], how='inner')
        self.x=np.asarray(self.merge.xminpot)
        self.y=np.asarray(self.merge.yminpot)
        self.z=np.asarray(self.merge.zminpot)
        mvir=np.asarray(self.merge.m_vir)
        mstar=np.asarray(self.merge.mstar)
        self.nhalo=len(mvir)
        print(np.log10(self.im), np.log10(np.min(mstar)), np.log10(np.mean(mstar)), np.log10(np.mean(mvir)), self.nhalo)
        print(f'Identifying stackable objects: {time.time() - self.job_start_time}s')
        return

    def compute_alm_maps(self, plot=False):
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
            plt.savefig('./Plots/filter_plot.png', dpi=1200)
            plt.clf()
        self.large_scale_map = hp.alm2map(lowpass_alm, nside=self.nside, lmax=3*self.nside-1)
        self.small_scale_map = hp.alm2map(highpass_alm, nside=self.nside, lmax=3*self.nside-1)
        if plot == True:
            hp.mollview(self.large_scale_map.copy(), title=f"Large scale CMB temperature map (sim={self.simname})", cmap="jet")#, min=-1.5e-4, max=1.5e-4)
            hp.graticule()
            plt.savefig(f'./Plots/T_ps_map_large_scale_{self.simname}_{self.z_sample_name}_{self.im_name}.png', dpi=1200)
            plt.clf()
            hp.mollview(self.small_scale_map.copy(), title=f"Small scale CMB temperature map (sim={self.simname})", cmap="jet")#, min=-1e-6, max=1e-6)
            hp.graticule()
            plt.savefig(f'./Plots/T_ps_map_small_scale_{self.simname}_{self.z_sample_name}_{self.im_name}.png', dpi=1200)
            plt.clf()
        self.mean_mod_T_large_scale = np.mean(np.abs(self.large_scale_map.copy()))
        print(self.mean_mod_T_large_scale)
        if self.rotate == True:
            print(f'Computing, rotating and filtering alms: {time.time() - self.job_start_time}s')
        elif self.rotate == False:
            print(f'Computing and filtering alms: {time.time() - self.job_start_time}s')
        return

    def f_lowpass(self, l):
        # Low frequency bandpass filter
        if l < 2000:
            return 1
        elif l > 2150:
            return 0
        else:
            return np.cos(((l - 2000) * np.pi) / 300)

    def f_highpass(self, l):
        # High frequency bandpass filter
        if l < 2350:
            return 0
        elif l > 2500:
            return 1
        else:
            return np.sin(((l - 2350) * np.pi) / 300)

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
            plt.savefig(f'./Plots/random_cutout_T_ps_map_{self.simname2}_{self.z_sample_name}.png', dpi=1200)
            plt.clf()
            plt.close('all')
            grid_values_tau = np.histogram2d(theta_pix.copy()*60.0, phi_pix.copy()*60.0, bins=[grid_x, grid_y], weights=tau_2D)
            plt.figure(figsize=(6, 6))
            plt.imshow(grid_values_tau[0].T, origin='lower', extent=[grid_x.min(), grid_x.max(), grid_y.min(), grid_y.max()], cmap='inferno')#, vmin=-1e-6, vmax=1e-6)
            plt.colorbar(label='Optical Depth')
            plt.xlabel('X (arcmin)')
            plt.ylabel('Y (arcmin)')
            plt.title(f'Rectangular Cutout Around Halo = {i} w/ Patchy Screening', wrap=True)
            plt.savefig(f'./Plots/random_cutout_T_filtered_map_{self.simname2}_{self.z_sample_name}.png', dpi=1200)
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
        outfile = os.path.join('./L1000N1800', self.z_sample_name, f'{self.simname}_tau_Mstar_bin{self.im_name}_nside{self.nside}_{self.cmb_method}{fits_suffix}{signal_suffix}{noise_suffix}.pickle')
        os.makedirs(os.path.dirname(outfile), exist_ok=True)
        with open(outfile, 'wb') as f:
            pickle.dump(data, f)
        f.close()
        print(f'Writing out data: {time.time() - self.job_start_time}s')
        return

    def run_analysis(self, plot=False):
        # Full analysis
        self.generate_cmb_map(plot)
        self.load_lightcones(plot)
        self.get_patchy_screening_map(plot)
        halo_lc_data, df_halo = self.load_halo_data()
        self.filter_stellar_mass(halo_lc_data, df_halo)
        self.compute_alm_maps(plot)
        self.get_halo_coordinates()
        self.run_tau_profiles(plot)
        self.stack_and_save()
        return

    def get_halo_coordinates(self):
        # Compute source vectors of each halo
        try:
            rows, cols = (self.nhalo, 3)
        except AttributeError:
            self.filter_stellar_mass()
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
            plt.savefig(f'./Plots/T_ps_map_{self.simname2}_{self.z_sample_name}.png', dpi=1200)
            plt.clf()
        print(f'Generating patchy screening map: {time.time() - self.job_start_time}s')
        return

if __name__ == '__main__':
    sim_list = ['HYDRO_FIDUCIAL','HYDRO_PLANCK','HYDRO_PLANCK_LARGE_NU_FIXED','HYDRO_PLANCK_LARGE_NU_VARY','HYDRO_STRONG_AGN','HYDRO_WEAK_AGN','HYDRO_LOW_SIGMA8','HYDRO_STRONGER_AGN','HYDRO_JETS','HYDRO_STRONGEST_AGN','HYDRO_STRONG_SUPERNOVA','HYDRO_STRONGER_AGN_STRONG_SUPERNOVA','HYDRO_STRONG_JETS']
    
    theta_d = np.arange(0.5, 11, 0.5)
    ncpu = int(sys.argv[1])
    isim = int(sys.argv[2])
    iz = str(sys.argv[3])
    im = 10**np.array(float(sys.argv[4]))
    im_name = f"{float(sys.argv[4]):.1f}".replace('.', 'p')
    fits = str(sys.argv[5])
    sig = bool(sys.argv[6])

    ps = patchyScreening(isim, iz, im, im_name, ncpu, theta_d, fits_file=fits, signal=sig, rect_size=20)
    #ps_camb = patchyScreening(isim, iz, im, im_name, ncpu, theta_d, cmb_method='CAMB', signal=sig, rect_size=20)
    #ps.run_analysis(plot=False)
    ps.get_halo_coordinates()
    quit()
    '''cl_fits = hp.anafast(ps.mock_CMB_primary)
    cl_camb = hp.anafast(ps_camb.mock_CMB_primary)

    plt.figure(figsize=(8,6))
    plt.plot(np.arange(len(cl_fits)), cl_fits*np.arange(len(cl_fits))*(np.arange(len(cl_fits))+1)/(2*np.pi), color='red', label='FITS')
    plt.plot(np.arange(len(cl_camb)), cl_camb*np.arange(len(cl_camb))*(np.arange(len(cl_camb))+1)/(2*np.pi), color='green', label='CAMB')
    plt.xlabel(r'Multipole moment $\ell$')
    plt.ylabel(r'$\frac{C_{\ell}\ell(\ell+1)}{2\pi}$')
    plt.title(f"Power Spectrum of the mock primary CMB\n(sim={ps.simname}, {ps.z_sample_name} sample, log$M_*$={np.log10(ps.im)})")
    plt.xscale("log")
    #plt.yscale("log")  # Using a log scale can help if the spectrum spans several orders of magnitude
    #plt.xlim(0, np.arange(len(cl_fits))[-1])
    #plt.ylim(bottom=0.009)
    plt.legend()
    plt.savefig(f'./Plots/primary_cmb_power_spectrum.png', dpi=1200)
    plt.clf()'''

    num_to_keep = int(ps.nhalo) // 2
    chosen_indices = np.random.choice(int(ps.nhalo), size=num_to_keep, replace=False)
    #reduced_arr = np.arange(int(ps.nhalo))[chosen_indices]
    #print(reduced_arr)
    vals, freqs = np.unique(chosen_indices, return_counts=True)
    for v, f in zip(vals, freqs):
        if f > 1:
            print('FAIL')

    obs_data_blue = np.loadtxt('./unWISExLens_lklh/data/v1.0/bandpowers/unWISExACT-DR6_blue_baseline_Clgg+Clkk+Clkg.dat', usecols=(0,1))
    ##on unit sphere---might not be necessary, but a standard way
    # Create an empty HEALPix map
    # This creates a map with all pixels initialized to zero
    npix = hp.nside2npix(2048)
    healpix_map = np.zeros(npix)
    
    # Convert vectors to HEALPix pixel indices
    pixels = hp.pixelfunc.vec2pix(2048, ps.source_vector[:,0], ps.source_vector[:,1], ps.source_vector[:,2])  ##in ring order by default
    index0 = np.where(pixels == 0)
    index1 = np.where(pixels == 1)
    index2 = np.where(pixels == 2)
    index3 = np.where(pixels == 3)
    index34= np.where(pixels == min(pixels))
     
    #print(pixels[index34])
        
    # Increment the map values at the halo positions
    # If you have weights for each halo (e.g., halo mass), you could use np.bincount with weights, or other functions to compute
    #other statistics such as mean blablabla
    galaxy_number = ps.nhalo
    number_density= galaxy_number/npix
    print(galaxy_number/41253)
    print(number_density)
    mstar = np.asarray(ps.merge.mstar)
    mvir = np.asarray(ps.merge.m_vir)
    #mass_weights = np.bincount(pixels, weights=mvir, minlength=npix)
    density_map = np.bincount(pixels, minlength=npix).astype(np.float64)
    density_map_reduced = np.bincount(pixels[chosen_indices], minlength=npix).astype(np.float64)
    index=np.where(density_map>1)
    print(density_map[index], len(density_map[index]))
    vals, freqs = np.unique(density_map, return_counts=True)
    # vals[i] = the count‐value (e.g. 0,1,2,…)
    # freqs[i] = the number of pixels that have that count

    for v, f in zip(vals, freqs):
        print(f"Value {v} appears {f} times")
    #quit()
    mean_density = np.mean(density_map)
    mean_density_reduced = np.mean(density_map_reduced)
    #healpix_map = mass_weights
    #print(healpix_map.shape)
    #print(pixels.shape)
    #print(pixels[chosen_indices].shape)
    print(mean_density)
    #print(mass_weights.shape)
    galaxy_overdensity = ((density_map - mean_density) / mean_density)
    galaxy_overdensity_reduced = (density_map_reduced - mean_density_reduced) / mean_density_reduced
    print(galaxy_overdensity.shape)
    #np.add.at(healpix_map, pixels, galaxy_overdensity) #—---this gives you the summed value per pixel.
    
    hp.mollview(galaxy_overdensity, title=f"Galaxy Overdensity Map\n(sim={ps.simname}, {ps.z_sample_name} sample, log$M_*$={np.log10(ps.im)}, primary CMB={fits})", unit=r"$\delta_g$", cmap="viridis")
    hp.graticule()
    plt.savefig(f'./Plots/halo_galaxy_overdensity_{ps.simname}_{ps.z_sample_name}_{ps.im_name}_{fits}.png', dpi=1200)
    plt.clf()

    # Compute the power spectrum from your halo map
    #cl = hp.anafast(healpix_map)
    cl = hp.anafast(galaxy_overdensity)/(4*np.pi)#/(2*np.pi**2)
    cl_reduced = hp.anafast(galaxy_overdensity_reduced)#/(2*np.pi**2)
    
    # Create an array of multipole moments l (the length of cl is usually lmax+1)
    l = np.arange(len(cl))
    l_reduced = np.arange(len(cl_reduced))

    from scipy.interpolate import interp1d
    from scipy.stats import chisquare
    p = interp1d(l, cl, kind='cubic')
    interp_sim = p(obs_data_blue[:,0])
    obs_data_blue_covariance = np.loadtxt('./unWISExLens_lklh/data/v1.0/covariances/covmat_Clgg+Clkg_unWISExACT-DR6_blue_baseline.dat')
    #obs_data_blue_covariance = np.loadtxt('./unWISExLens_lklh/data/v1.0/covariances/covmat_Clgg+Clkg_unWISExACT-DR6_blue_cmbmarg.dat')
    print(obs_data_blue_covariance.shape)
    obs_data_blue_variance = np.diag(obs_data_blue_covariance)
    obs_data_blue_std = np.sqrt(obs_data_blue_variance[:59])

    print(np.sum(obs_data_blue[:,1]), np.sum(interp_sim), np.sum(obs_data_blue[:,1]) - np.sum(interp_sim))
    print(obs_data_blue_std.shape)
    #quit()
    chi2 = np.sum(((interp_sim - obs_data_blue[:,1])/obs_data_blue_std)**2)
    #chi2, pvalue = chisquare(f_obs=obs_data_blue[:,1], f_exp=interp_sim)
    print(chi2)
    #quit()
    '''textstr = (
        rf"$\chi^2 = {chi2:.2f}$" "\n"
        rf"$p = {p_value:.2g}$"
    )'''
    
    # Plot the power spectrum
    plt.figure(figsize=(8,6))
    plt.plot(l, cl*1e5, color='tab:blue', label=r'FLAMINGO (x$\frac{1}{4\pi}$)')# (x$\frac{1}{2\pi^2}$)')
    plt.plot(obs_data_blue[:,0], interp_sim*1e5, color='r', marker='.', markersize=5, label='Farren et al. (2023) (interpolated)')
    #plt.plot(l_reduced, cl_reduced*1e5, color='tab:red', label='FLAMINGO (reduced)')
    plt.plot(obs_data_blue[:,0], obs_data_blue[:,1]*1e5, color='b', marker='.', markersize=5, label='Farren et al. (2023)')
    '''plt.text(0.90, 0.70,
             textstr,
             transform=plt.transAxes,
             fontsize=12,
             verticalalignment='top',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7)
             )'''
    plt.xlabel(r'Multipole moment $\ell$')
    plt.ylabel(r'$C^{gg}_{\ell}x10^5$')
    plt.title(f"Power Spectrum of the Galaxy Overdensity map\n(sim={ps.simname}, {ps.z_sample_name} sample, log$M_*$={np.log10(ps.im)}, primary CMB={fits})")
    plt.xscale("log")
    plt.yscale("log")  # Using a log scale can help if the spectrum spans several orders of magnitude
    plt.xlim(0, l[-1])
    #plt.xlim(0, 1100)
    plt.ylim(bottom=0.001)
    plt.legend()
    plt.savefig(f'./Plots/halo_map_power_spectrum_{ps.simname}_{ps.z_sample_name}_{ps.im_name}_{fits}.png', dpi=1200)
    plt.clf()

    print(f'For stellar cut of {np.log10(ps.im)}: {interp_sim/obs_data_blue[:,1]}')

    plt.plot(obs_data_blue[:,0], interp_sim/obs_data_blue[:,1])
    plt.hlines(1, -1, l[-1], color='k', alpha=.5)
    plt.xlabel(r'Multipole moment $\ell$')
    plt.ylabel(r'Residual')
    plt.title(f"Residual of Power Spectrum from the Galaxy Overdensity map\n(sim={ps.simname}, {ps.z_sample_name} sample, log$M_*$={np.log10(ps.im)}, primary CMB={fits})")
    plt.xscale("log")
    #plt.xlim(0, obs_data_blue[-1,0])
    plt.savefig(f'./Plots/halo_map_power_spectrum_{ps.simname}_{ps.z_sample_name}_{ps.im_name}_{fits}_residual.png', dpi=1200)
    plt.clf()

    

####################################################################################################################################
