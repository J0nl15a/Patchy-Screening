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
    def __init__(self, theta_d, isam, survey, mstar_bins, mstar_bins_name, ncpu, isel, rect_size=20, nside=8192):

        self.job_start_time = time.time()
        self.sim_list = ['HYDRO_FIDUCIAL','HYDRO_PLANCK','HYDRO_PLANCK_LARGE_NU_FIXED','HYDRO_PLANCK_LARGE_NU_VARY','HYDRO_STRONG_AGN','HYDRO_WEAK_AGN','HYDRO_LOW_SIGMA8','HYDRO_STRONGER_AGN','HYDRO_JETS','HYDRO_STRONGEST_AGN','HYDRO_STRONG_SUPERNOVA','HYDRO_STRONGER_AGN_STRONG_SUPERNOVA','HYDRO_STRONG_JETS']
        self.sim_list2=['HYDRO_FIDUCIAL','HYDRO_PLANCK','HYDRO_PLANCK_LARGE_NU_FIXED','HYDRO_PLANCK_LARGE_NU_VARY','HYDRO_STRONG_AGN','HYDRO_WEAK_AGN','HYDRO_LOW_SIGMA8','HYDRO_STRONGER_AGN','HYDRO_JETS_old','HYDRO_STRONGEST_AGN','HYDRO_STRONG_SUPERNOVA','HYDRO_STRONGER_AGN_STRONG_SUPERNOVA','HYDRO_STRONG_JETS_old']
        
        self.theta_d = theta_d
        self.isam = isam
        self.survey = survey
        self.mstar_bins = mstar_bins
        self.mstar_bins_name = mstar_bins_name
        self.nside = nside
        self.ncpu = ncpu
        self.isel = isel
        self.rect_size = rect_size
        
        self.cosmology = FlatLambdaCDM(H0=68.1, Om0=0.3, Tcmb0=2.725)
        self.mock_CMB_primary = None

    def generate_cmb_map(self, method='FITS', fits_file='unlensed'):
        np.random.seed(1000)
        if method.upper() == 'CAMB':
            pars = camb.set_params(H0=68.1, ombh2=0.048600*(0.681**2), omch2=0.256011*(0.681**2), mnu=0.06, As=2.099e-9, ns=0.967, lmax=3*self.nside-1)
            results = camb.get_results(pars)
            powers = results.get_cmb_power_spectra(pars, raw_cl=True)
            unlensed_total_CL = powers['unlensed_total']
            self.mock_CMB_primary = hp.synfast(unlensed_total_CL[:,0], nside=self.nside)
        elif method.upper() == 'FITS':
            lensed_dir = '/cosma8/data/dp004/dc-yang3/maps/L1000N1800/HYDRO_FIDUCIAL/lightcone0_shells/patchy_screening_folder'
            if fits_file == 'unlensed':
                self.mock_CMB_primary = hp.read_map(f'{lensed_dir}/CMB_T_map_unl.fits', dtype=np.float64, verbose=False)
            elif fits_file == 'lensed_z2':
                self.mock_CMB_primary = hp.read_map(f'{lensed_dir}/CMB_T_map_l_kappa_z2.fits', dtype=np.float64, verbose=False)
            elif fits_file == 'lensed_z3':
                self.mock_CMB_primary = hp.read_map(f'{lensed_dir}/CMB_T_map_l_kappa_z3.fits', dtype=np.float64, verbose=False)
        else:
            raise ValueError("Unknown CMB map generation method")
        print(f'Generating mock primary CMB: {time.time() - self.job_start_time}s')
        print(self.mock_CMB_primary)
        return

    def load_lightcones(self, simname2, i, plot=(False)):
        map_lightcone = f'/cosma8/data/dp004/flamingo/Runs/L1000N1800/'+simname2+'/neutrino_corrected_maps/lightcone0_shells/shell_'+str(i)+'/lightcone0.shell_'+str(i)+'.0.hdf5'
        g = h5py.File(map_lightcone,'r')
        conversion_factor = g['DM'].attrs['Conversion factor to CGS (not including cosmological corrections)']
        DM = g['DM'][...]*conversion_factor*6.6524587321e-25 #6.65246e-25 = Thomson cross-section (in cgs)
        print(DM)
        print(f'Loading first lightcone shell: {time.time() - self.job_start_time}s')

        if plot[0] == True:
            hp.mollview(DM, title=f"DM map (sim={simname2}, lightcone shell={i})", cmap="jet", min=2e-5, max=2e-3)
            hp.graticule()
            plt.savefig(f'./Plots/DM_map_{simname2}_{self.survey[plot[1]]}_shell_{i}_new.png', dpi=1200)
            plt.clf()

        map_lightcone = f'/cosma8/data/dp004/flamingo/Runs/L1000N1800/'+simname2+'/neutrino_corrected_maps/lightcone0_shells/shell_'+str(i-1)+'/lightcone0.shell_'+str(i-1)+'.0.hdf5'
        g = h5py.File(map_lightcone,'r')
        conversion_factor = g['DM'].attrs['Conversion factor to CGS (not including cosmological corrections)']
        DM += g['DM'][...]*conversion_factor*6.6524587321e-25

        map_lightcone = f'/cosma8/data/dp004/flamingo/Runs/L1000N1800/'+simname2+'/neutrino_corrected_maps/lightcone0_shells/shell_'+str(i+1)+'/lightcone0.shell_'+str(i+1)+'.0.hdf5'
        g = h5py.File(map_lightcone,'r')
        conversion_factor = g['DM'].attrs['Conversion factor to CGS (not including cosmological corrections)']
        DM += g['DM'][...]*conversion_factor*6.6524587321e-25

        self.DM_map = hp.pixelfunc.ud_grade(DM,self.nside)
        print(self.DM_map)

        return

    def load_halo_data(self, simname, simname2, i, lightcone_type='HBT'):
        # Load halo lightcone and SOAP data into DataFrames
        halo_lc_data = pd.DataFrame()
        if lightcone_type == 'HBT':
            if i >= 10: halo_lightcone = f'/cosma8/data/dp004/jch/FLAMINGO/HBT/L1000N1800/'+simname+'/lightcone_halos/lightcone0/lightcone_halos_00'+str(77-i)+'.hdf5'
            if i < 10: halo_lightcone = f'/cosma8/data/dp004/jch/FLAMINGO/HBT/L1000N1800/'+simname+'/lightcone_halos/lightcone0/lightcone_halos_000'+str(77-i)+'.hdf5'
            f = h5py.File(halo_lightcone, 'r')
            halo_lc_data['ID'] = f['InputHalos/HaloCatalogueIndex'][...]
            halo_lc_data['SnapNum'] = f['Lightcone/SnapshotNumber'][...]
            halo_lc_data['z'] = f['Lightcone/Redshift'][...]
            halo_centre = f['Lightcone/HaloCentre'][...]
            halo_lc_data['xminpot'] = halo_centre[:,0]
            halo_lc_data['yminpot'] = halo_centre[:,1]
            halo_lc_data['zminpot'] = halo_centre[:,2]
        elif lightcone_type == 'VR':
            if i >= 10: halo_lightcone = f'/cosma8/data/dp004/jch/FLAMINGO/lightcone_halos/'+simname+'/lightcone_halos/lightcone0/lightcone_halos_00'+str(i)+'.hdf5'
            if i < 10: halo_lightcone = f'/cosma8/data/dp004/jch/FLAMINGO/lightcone_halos/'+simname+'/lightcone_halos/lightcone0/lightcone_halos_000'+str(i)+'.hdf5'
            f = h5py.File(halo_lightcone, 'r')
            halo_lc_data['ID'] = f['Subhalo/ID'][...]
            halo_lc_data['SnapNum'] = f['Subhalo/SnapNum'][...]
            halo_lc_data['z'] = f['Subhalo/LightconeRedshift'][...]
            halo_lc_data['xminpot'] = f['Subhalo/LightconeXcminpot'][...]
            halo_lc_data['yminpot'] = f['Subhalo/LightconeYcminpot'][...]
            halo_lc_data['zminpot'] = f['Subhalo/LightconeZcminpot'][...]

        print(np.min(halo_lc_data.z),np.max(halo_lc_data.z))
        Dcom = self.cosmology.comoving_distance(np.mean(halo_lc_data.z))*0.681  # comoving distance to galaxy in Mpc/h
        self.Dcom = Dcom.value
        halo_lc_data.sort_values(by='ID', inplace=True)
        halo_lc_data.reset_index(inplace=True, drop=True)
        snap = int(halo_lc_data.iloc[0]['SnapNum'])
        print(self.Dcom, snap)

        if lightcone_type == 'HBT':
            HBT_file = f'/cosma8/data/dp004/flamingo/Runs/L1000N1800/'+simname2+'/SOAP-HBT/halo_properties_00'+str(snap)+'.hdf5'
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
            VR_file = f'/cosma8/data/dp004/flamingo/Runs/L1000N1800/'+simname2+'/SOAP/halo_properties_00'+str(snap)+'.hdf5'
            f = h5py.File(HBT_file, 'r')
            df_VR = pd.DataFrame()
            df_VR['ID'] = f['VR/ID'][...]
            #df_VR['hostHaloID'] = f['VR/HostHaloID'][...]
            df_VR['Structuretype'] = f['VR/StructureType'][...]
            df_VR['m_vir'] = f['SO/500_crit/TotalMass'][...]
            df_VR['mstar'] = f['ExclusiveSphere/50kpc/StellarMass'][...]
            print(f'Loading halo lightcone data: {time.time() - self.job_start_time}s')
            return halo_lc_data, df_VR

    def filter_stellar_mass(self, df_halo, halo_lc_data, mstar_bin):
        # Merge and filter the DataFrames based on mstar_bin
        df_mass = df_halo
        df_mass = df_mass.loc[df_mass.mstar > self.mstar_bins[mstar_bin]]
        df_mass.sort_values(by='ID', inplace=True)
        df_mass.reset_index(inplace=True, drop=True)
        merge = pd.merge_ordered(df_mass, halo_lc_data, on=['ID'], how='inner')
        self.x=np.asarray(merge.xminpot)
        self.y=np.asarray(merge.yminpot)
        self.z=np.asarray(merge.zminpot)
        mvir=np.asarray(merge.m_vir)
        mstar=np.asarray(merge.mstar)
        self.nhalo=len(mvir)
        print(mstar_bin, np.log10(np.min(mstar)), np.log10(np.mean(mstar)), np.log10(np.mean(mvir)), self.nhalo)
        print(f'Identifying stackable objects: {time.time() - self.job_start_time}s')
        return merge

    def compute_alm_maps(self, plot=(False,False,False,False)):
        alm = hp.map2alm(self.T_cmb_ps, lmax=3*self.nside-1)
        ell, m = hp.Alm.getlm(lmax=3*self.nside-1)
        lowpass_values = np.array([self.f_lowpass(l) for l in ell])
        highpass_values = np.array([self.f_highpass(l) for l in ell])
        lowpass_alm = hp.almxfl(alm.copy(), lowpass_values)
        highpass_alm = hp.almxfl(alm.copy(), highpass_values)
        if plot[0] == True:
            m_zero = np.where(m==0)
            plt.plot(ell[m_zero], lowpass_values[m_zero], label='lowpass filter', color='b')
            plt.plot(ell[m_zero], highpass_values[m_zero], label='highpass filter', color='r')
            plt.xlabel(r'$\ell$')
            plt.ylabel('Filter values')
            plt.title('My version of the Coulton frequency filters (m=0)')
            plt.legend()
            plt.savefig('./Plots/filter_plot_new.png', dpi=1200)
            plt.clf()
        self.large_scale_map = hp.alm2map(lowpass_alm, nside=self.nside, lmax=3*self.nside-1)
        self.small_scale_map = hp.alm2map(highpass_alm, nside=self.nside, lmax=3*self.nside-1)
        if plot[0] == True:
            hp.mollview(self.large_scale_map.copy(), title=f"Large scale CMB temperature map (sim={plot[1]})", cmap="jet")#, min=-1.5e-4, max=1.5e-4)
            hp.graticule()
            plt.savefig(f'./Plots/T_ps_map_large_scale_{plot[1]}_{self.survey[plot[2]]}_{self.mstar_bins_name[plot[3]]}_new.png', dpi=1200)
            plt.clf()
            hp.mollview(self.small_scale_map.copy(), title=f"Small scale CMB temperature map (sim={plot[1]})", cmap="jet")#, min=-1e-6, max=1e-6)
            hp.graticule()
            plt.savefig(f'./Plots/T_ps_map_small_scale_{plot[1]}_{self.survey[plot[2]]}_{self.mstar_bins_name[plot[3]]}_new.png', dpi=1200)
            plt.clf()
        self.mean_mod_T_large_scale = np.mean(np.abs(self.large_scale_map.copy()))
        print(self.mean_mod_T_large_scale)
        print(f'Generating, rotating and filtering alms: {time.time() - self.job_start_time}s')
        return

    def f_lowpass(self, l):
        if l < 2000:
            return 1
        elif l > 2150:
            return 0
        else:
            return np.cos(((l - 2000) * np.pi) / 300)

    def f_highpass(self, l):
        if l < 2350:
            return 0
        elif l > 2500:
            return 1
        else:
            return np.sin(((l - 2350) * np.pi) / 300)

    def tau_prof(self, i, source_vector, theta, phi, plot=(False,False,False,False)):
        
        halo_pixels = hp.query_disc(self.nside, source_vector[i,:], radius=(0.25*u.deg).to_value(u.radian))
        theta_pix, phi_pix = hp.pix2ang(self.nside, halo_pixels, lonlat=True)
        rtheta = hp.rotator.angdist([theta[i], phi[i]], [theta_pix, phi_pix], lonlat=True)*60.0*180.0/np.pi
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
            plt.savefig(f'./Plots/random_cutout_T_ps_map_{plot[2]}_{self.survey[plot[3]]}_new.png', dpi=1200)
            plt.clf()
            plt.close('all')
            grid_values_tau = np.histogram2d(theta_pix.copy()*60.0, phi_pix.copy()*60.0, bins=[grid_x, grid_y], weights=tau_2D)
            plt.figure(figsize=(6, 6))
            plt.imshow(grid_values_tau[0].T, origin='lower', extent=[grid_x.min(), grid_x.max(), grid_y.min(), grid_y.max()], cmap='inferno')#, vmin=-1e-6, vmax=1e-6)
            plt.colorbar(label='Optical Depth')
            plt.xlabel('X (arcmin)')
            plt.ylabel('Y (arcmin)')
            plt.title(f'Rectangular Cutout Around Halo = {i} w/ Patchy Screening', wrap=True)
            plt.savefig(f'./Plots/random_cutout_T_filtered_map_{plot[2]}_{self.survey[plot[3]]}_new.png', dpi=1200)
            plt.clf()
            plt.close('all')
        tau_1D = np.zeros(len(self.theta_d))
        for j in range(len(self.theta_d)):
            idx_in = np.where((rtheta > self.theta_d[j]-0.25) & (rtheta <= self.theta_d[j]+0.25))
            tau_1D[j] = (-1*np.mean(np.sign(large_scale_fluxes[idx_in])*small_scale_fluxes[idx_in]))/self.mean_mod_T_large_scale
        return tau_1D

    def run_tau_profiles(self, source_vector, theta, phi, plot):
        batch_size=max(1, self.nhalo // (self.ncpu*2))
        randint = np.random.randint(self.nhalo)        
        print(f'Starting profile loop: {time.time() - self.job_start_time}s')
        results = Parallel(n_jobs=self.ncpu, backend="loky", batch_size=batch_size)(delayed(self.tau_prof)(i, source_vector, theta, phi, (plot[0],randint,plot[1],plot[2])) for i in range(self.nhalo))
        print(f'Ending profile loop: {time.time() - self.job_start_time}s')
        #data_1D = zip(*results)
        self.data_1D = np.asarray(results)
        return

    def stack_and_save(self, iz, simname, im, method, fits_file):
        tau_1D_stack = np.zeros(len(self.theta_d))
        for i in range(self.nhalo):
            tau_1D = self.data_1D[i,:]
            tau_1D_stack += tau_1D
        tau_1D_stack /= self.nhalo
        print(simname,tau_1D_stack)
        
        rows, cols = (len(self.theta_d), 4)
        data = [0]*cols
        data[0] = self.theta_d
        data[1] = tau_1D_stack
        data[2] = (self.theta_d*np.pi/(180.0*60.0))*self.Dcom
        data[3] = self.nhalo
        if method == 'CAMB':
            outfile = os.path.join('./L1000N1800', self.survey[iz], f'{simname}_tau_Mstar_bin{im}_nside{self.nside}_new.pickle')
        elif method == 'FITS' and fits_file == 'unlensed':
            outfile = os.path.join('./L1000N1800', self.survey[iz], f'unlensed_{simname}_tau_Mstar_bin{im}_nside{self.nside}_new.pickle')
        elif method == 'FITS' and fits_file == 'lensed_z2':
            outfile = os.path.join('./L1000N1800', self.survey[iz], f'lensed_z2_{simname}_tau_Mstar_bin{im}_nside{self.nside}_new.pickle')
        elif method == 'FITS' and fits_file == 'lensed_z3':
            outfile = os.path.join('./L1000N1800', self.survey[iz], f'lensed_z3_{simname}_tau_Mstar_bin{im}_nside{self.nside}_new.pickle')
        else:
            raise ValueError("Unknown parameter configuration")
        os.makedirs(os.path.dirname(outfile), exist_ok=True)
        with open(outfile, 'wb') as f:
            pickle.dump(data, f)
        f.close()
        print(f'Writing out data: {time.time() - self.job_start_time}s')
        return

    def run_analysis(self, isim, iz, im, method='FITS', fits_file='unlensed', plot=False):
        #for isim in range(self.isel, self.isel+1):
        simname = self.sim_list[isim]
        simname2 = self.sim_list2[isim]
        self.generate_cmb_map(method=method, fits_file=fits_file)
        if plot == True:
            hp.mollview(self.mock_CMB_primary, title=f"Mock Primary CMB temperature map (sim={simname})", cmap="jet")#, min=-1.5e-4, max=1.5e-4)
            hp.graticule()
            plt.savefig(f'./Plots/primary_CMB_map_{simname}_new.png', dpi=1200)
            plt.clf()
        #for iz in range(len(self.survey)):
        i = self.isam[iz]
        self.load_lightcones(simname2, i, (plot,iz))
        if plot == True:
            hp.mollview(self.DM_map, title=f"DM map (sim={simname2}, lightcone shell={i-1}+{i}+{i+1})", cmap="jet")#, min=2e-5, max=2e-3)
            hp.graticule()
            plt.savefig(f'./Plots/DM_map_{simname2}_{self.survey[iz]}_shell_{i-1}-{i+1}_new.png', dpi=1200)
            plt.clf()
        self.T_cmb_ps = -1 * self.DM_map.copy() * self.mock_CMB_primary.copy()
        self.T_cmb_ps = self.T_cmb_ps + self.mock_CMB_primary.copy()
        self.T_cmb_ps = hp.smoothing(self.T_cmb_ps,fwhm=1.3*np.pi/60.0/180.0)
        print(f'Generating patchy screening map: {time.time() - self.job_start_time}s')
        if plot == True:
            hp.mollview(self.T_cmb_ps, title="CMB temperature map w/ Patchy Screening", cmap="jet")#, min=-1.5e-4, max=1.5e-4)
            hp.graticule()
            plt.savefig(f'./Plots/T_ps_map_{simname2}_{self.survey[iz]}_new.png', dpi=1200)
            plt.clf()
        halo_lc_data, df_halo = self.load_halo_data(simname, simname2, i)
        #for im in range(len(self.mstar_bins)):
        merge = self.filter_stellar_mass(df_halo, halo_lc_data, im)
        self.compute_alm_maps((plot,simname,iz,im))
        if plot == True:
            hp.mollview(self.large_scale_map, title=f"Large scale CMB temperature map (sim={simname2})", cmap="jet")#, min=-1.5e-4, max=1.5e-4)
            hp.graticule()
            plt.savefig(f'./Plots/T_ps_map_large_scale_{simname2}_{self.survey[iz]}_{mstar_bins_name[im]}_new.png', dpi=1200)
            plt.clf()
            hp.mollview(self.small_scale_map, title=f"Small scale CMB temperature map (sim={simname2})", cmap="jet")#, min=-1e-6, max=1e-6)
            hp.graticule()
            plt.savefig(f'./Plots/T_ps_map_small_scale_{simname2}_{self.survey[iz]}_{mstar_bins_name[im]}_new.png', dpi=1200)
            plt.clf()
        source_vector, theta, phi = self.get_halo_coordinates()
        self.run_tau_profiles(source_vector, theta, phi, (plot, simname2, iz))
        self.stack_and_save(iz, simname, im, method, fits_file)
        return merge, source_vector, theta, phi

    def get_halo_coordinates(self):
        rows, cols = (self.nhalo, 3)
        vec = [[0]*cols]*rows
        vec=1.0*np.asarray(vec)
        vec[:,0]=self.x
        vec[:,1]=self.y
        vec[:,2]=self.z
        theta, phi = hp.pixelfunc.vec2ang(vec,lonlat=True)
        source_vector = hp.ang2vec(theta, phi,lonlat=True)
        return source_vector, theta, phi

if __name__ == '__main__':
    sim_list = ['HYDRO_FIDUCIAL','HYDRO_PLANCK','HYDRO_PLANCK_LARGE_NU_FIXED','HYDRO_PLANCK_LARGE_NU_VARY','HYDRO_STRONG_AGN','HYDRO_WEAK_AGN','HYDRO_LOW_SIGMA8','HYDRO_STRONGER_AGN','HYDRO_JETS','HYDRO_STRONGEST_AGN','HYDRO_STRONG_SUPERNOVA','HYDRO_STRONGER_AGN_STRONG_SUPERNOVA','HYDRO_STRONG_JETS']
    
    theta_d = np.arange(0.5, 11, 0.5)
    isam = np.array([11, 21, 29])
    survey = ['Blue', 'Green', 'Red']
    mstar_bins = 10**np.array([10.9, 11.0, 11.1, 11.2, 11.3, 11.4, 11.5, 11.6])
    mstar_bins_name = ['10p9', '11p0', '11p1', '11p2', '11p3', '11p4', '11p5', '11p6']
    ncpu = int(sys.argv[1])
    isel = int(sys.argv[2])
    iz = int(sys.argv[3])
    im = int(sys.argv[4])
    
    ps = patchyScreening(theta_d, isam, survey, mstar_bins, mstar_bins_name, ncpu, isel, rect_size=20)
    lc_data, source_vector, theta, phi = ps.run_analysis(isel, iz, im, plot=True)

    ##on unit sphere---might not be necessary, but a standard way
    # Create an empty HEALPix map
    # This creates a map with all pixels initialized to zero
    npix = hp.nside2npix(ps.nside)
    healpix_map = np.zeros(npix)

    # Convert vectors to HEALPix pixel indices
    pixels = hp.pixelfunc.vec2pix(ps.nside, source_vector[:,0], source_vector[:,1], source_vector[:,2])  ##in ring order by default

    # Increment the map values at the halo positions
    # If you have weights for each halo (e.g., halo mass), you could use np.bincount with weights, or other functions to compute 
    #other statistics such as mean blablabla
    mstar = np.asarray(lc_data.mstar)
    mass_weights = np.bincount(pixels, weights=mstar, minlength=npix)
    healpix_map = mass_weights
    print(healpix_map.shape)
    print(pixels.shape)
    print(mass_weights.shape)
    
    #np.add.at(healpix_map, pixels, mstar) #—---this gives you the summed value per pixel.
    
    hp.mollview(healpix_map, title=f"Halo Density Map", unit="Count", cmap="viridis")
    hp.graticule()
    plt.savefig(f'./Plots/halo_density_map.png', dpi=1200)
    plt.clf()
    
'''if plot[0] == True and i == plot[1]:
            #print('worked1')
            tau_2D = (np.sign(large_scale_fluxes.copy())*small_scale_fluxes.copy())/self.mean_mod_T_large_scale.copy()
            #print('worked2')
            grid_x = np.linspace(min(theta_pix.copy()*60.0), max(theta_pix.copy()*60.0), 25)  # Adjust grid size as needed
            grid_y = np.linspace(min(phi_pix.copy()*60.0), max(phi_pix.copy()*60.0), 25)
            #print('worked3')
            temperature_map = self.T_cmb_ps[halo_pixels].copy()
            #print('worked4')
            grid_values_temp, xedges_temp, yedges_temp = np.histogram2d(theta_pix.copy()*60.0, phi_pix.copy()*60.0, bins=[grid_x, grid_y], weights=temperature_map)
            #print('worked5')
            plt.figure(figsize=(6, 6))
            plt.imshow(grid_values_temp.T, origin='lower', extent=[grid_x.min(), grid_x.max(), grid_y.min(), grid_y.max()], cmap='inferno')#, vmin=-1.5e-4, vmax=1.5e-4)
            plt.colorbar(label=r'Temperature ($\mu$K)')
            plt.xlabel('X (arcmin)')
            plt.ylabel('Y (arcmin)')
            plt.title(f'Rectangular Cutout Around Halo = {i}', wrap=True)
            plt.savefig(f'./Plots/random_cutout_T_ps_map_{plot[2]}_{self.survey[plot[3]].copy()}_new.png', dpi=1200)
            plt.clf()
            plt.close('all')
            tau_map = tau_2D
            grid_values_tau = np.histogram2d(theta_pix.copy()*60.0, phi_pix.copy()*60.0, bins=[grid_x, grid_y], weights=tau_map)
            plt.figure(figsize=(6, 6))
            plt.imshow(grid_values_tau[0].T, origin='lower', extent=[grid_x.min(), grid_x.max(), grid_y.min(), grid_y.max()], cmap='inferno')#, vmin=-1e-6, vmax=1e-6)
            plt.colorbar(label='Optical Depth')
            plt.xlabel('X (arcmin)')
            plt.ylabel('Y (arcmin)')
            plt.title(f'Rectangular Cutout Around Halo = {i} w/ Patchy Screening', wrap=True)
            plt.savefig(f'./Plots/random_cutout_T_filtered_map_{plot[2]}_{self.survey[plot[3]].copy()}_new.png', dpi=1200)
            plt.clf()
            plt.close('all')'''
'''if i==plot[1]:
print('tau_1D')
print(tau_1D)'''
####################################################################################################################################
