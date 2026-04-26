import sys, os, pickle
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
from mpi4py import MPI

'''comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()'''

class patchyScreening:
    def __init__(self, box, isim, iz, im, n_cut, ncpu, theta_d=np.arange(0.5, 11, 0.5), nside=8192, cmb_method='FITS', fits_file='unlensed', lightcone_method=('FULL','dndz'), signal=True, rotate=False, rect_size=20, lightcone=0):

        os.environ["POLARS_MAX_THREADS"] = str(ncpu)
        self.job_start_time = time.time()
        box_list = ['L1000N1800', 'L1000N3600', 'L2800N5040']
        sim_list = ['HYDRO_FIDUCIAL','HYDRO_PLANCK','HYDRO_PLANCK_LARGE_NU_FIXED','HYDRO_PLANCK_LARGE_NU_VARY','HYDRO_STRONG_AGN','HYDRO_WEAK_AGN','HYDRO_LOW_SIGMA8','HYDRO_STRONGER_AGN','HYDRO_JETS_published','HYDRO_STRONGEST_AGN','HYDRO_STRONG_SUPERNOVA','HYDRO_STRONGER_AGN_STRONG_SUPERNOVA','HYDRO_STRONG_JETS_published','HYDRO_LOW_SIGMA8_STRONGEST_AGN']

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
            
        self.im = 10**np.array(float(im))
        if round(im, 1) == im:
            self.im_name = f"{float(im):.1f}".replace('.', 'p')
        else:
            self.im_name = f"{float(im):.3f}".replace('.', 'p')

        self.slope = float(n_cut)
        if round(self.slope, 1) == self.slope:
            self.slope_name = f"{float(n_cut):.1f}".replace('.', 'p')
        else:
            self.slope_name = f"{float(n_cut):.3f}".replace('.', 'p')

        if self.slope < 0.0:
            self.slope_name = f"{self.slope_name}".replace('-', 'minus')

        self.ncpu = int(ncpu)
        self.theta_d = theta_d
        self.nside = nside
        if self.boxname == 'L2800N5040' and self.nside > 4096:
            self.nside = 4096
            print("Setting nside to 4096 for L2800N5040.")
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

    
    def generate_cmb_map(self, plot=False):
                
        # Generating primary CMB map with CAMB or loading pre-generated FITS
        np.random.seed(1000)
        if self.cmb_method == 'CAMB':
            pars = camb.set_params(H0=68.1, ombh2=0.048600*(0.681**2), omch2=0.256011*(0.681**2), mnu=0.06, As=2.099e-9, ns=0.967, lmax=3*self.nside-1+10)
            results = camb.get_results(pars)
            powers = results.get_cmb_power_spectra(pars, raw_cl=True, CMB_unit='muK', lmax=3*self.nside-1)
            unlensed_total_CL = powers['unlensed_total']
            print(len(unlensed_total_CL))
            self.mock_CMB_primary = hp.synfast(unlensed_total_CL[:,0], nside=self.nside, lmax=5024, mmax=5024) # order of CMB modes: TT, EE, BB, TE
            #self.mock_CMB_primary = hp.synfast(unlensed_scalar_CL[:,1], nside=self.nside)
            
        elif self.cmb_method == 'FITS':
            lensed_dir = f'/cosma8/data/dp004/dc-yang3/maps/{self.boxname}/{self.simname}/lightcone0_shells/patchy_screening_folder'
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
            map_lightcone = f'/cosma8/data/dp004/flamingo/Runs/{self.boxname}/{self.simname}/{map_dir}/lightcone{self.lightcone}_shells/shell_{self.z_sample}/lightcone{self.lightcone}.shell_{self.z_sample}.0.hdf5'
            g = h5py.File(map_lightcone,'r')
            conversion_factor = g['DM'].attrs['Conversion factor to CGS (not including cosmological corrections)']
            DM = g['DM'][...]*conversion_factor*6.6524587321e-25 #6.65246e-25 = Thomson cross-section (in cgs)
            redshift = g['DM'].attrs['Central redshift assumed for correction']
            DM *= (1+redshift)
            g.close()
            print(f'Map z of {self.z_sample_name} sample = {redshift} (shell: {self.z_sample})')
            print(DM)
            print(f'Loading first lightcone shell: {time.time() - self.job_start_time}s')

            if plot == True:
                hp.mollview(DM, title=f"DM map (box={self.boxname}, sim={self.simname}, lightcone shell={self.z_sample})", cmap="jet", min=2e-5, max=2e-3)
                hp.graticule()
                plt.savefig(f'./Plots/DM_map_{self.boxname}_{self.simname}_{self.z_sample_name}_shell_{self.z_sample}.png', dpi=400)
                plt.clf()

            map_lightcone_lower = f'/cosma8/data/dp004/flamingo/Runs/{self.boxname}/{self.simname}/{map_dir}/lightcone{self.lightcone}_shells/shell_{self.z_sample-1}/lightcone{self.lightcone}.shell_{self.z_sample-1}.0.hdf5'
            g_low = h5py.File(map_lightcone_lower,'r')
            conversion_factor = g_low['DM'].attrs['Conversion factor to CGS (not including cosmological corrections)']
            DM += g_low['DM'][...]*conversion_factor*6.6524587321e-25
            redshift_low = g_low['DM'].attrs['Central redshift assumed for correction']
            DM *= (1+redshift_low)

            map_lightcone_higher = f'/cosma8/data/dp004/flamingo/Runs/{self.boxname}/{self.simname}/{map_dir}/lightcone{self.lightcone}_shells/shell_{self.z_sample+1}/lightcone{self.lightcone}.shell_{self.z_sample+1}.0.hdf5'
            g_high = h5py.File(map_lightcone_higher,'r')
            conversion_factor = g_high['DM'].attrs['Conversion factor to CGS (not including cosmological corrections)']
            DM += g_high['DM'][...]*conversion_factor*6.6524587321e-25
            redshift_high = g_high['DM'].attrs['Central redshift assumed for correction']
            DM *= (1+redshift_high)

            print(f"Map lightcone z = [{g_low['DM'].attrs['Central redshift assumed for correction']},{g_high['DM'].attrs['Central redshift assumed for correction']}]")
            g_low.close()
            g_high.close()

        elif self.lightcone_method[0] == 'FULL':
            try:
                DM = hp.read_map(f'/cosma8/data/dp004/dc-conl1/FLAMINGO/patchy_screening/data_files/DM_maps/{self.boxname}/{self.simname}/lightcone{self.lightcone}/stacked_DM_map_z3p0.fits', dtype=np.float64, verbose=False)
            except FileNotFoundError:
                from stacked_DM_maps import stack_DM_maps_z3
                DM = stack_DM_maps_z3(self.boxname, self.simname)
            DM_2 = hp.pixelfunc.ud_grade(DM,self.nside)
            alm = hp.map2alm(DM_2)
            #DM_2 = hp.alm2map(alm, nside=self.nside, lmax=5024)
            DM_2 = hp.sphtfunc.resize_alm(alm, lmax=3*self.nside-1, mmax=3*self.nside-1, lmax_out=5024, mmax_out=5024)
            print(f"Map lightcone integrated up to z=3")

        self.DM_map = hp.pixelfunc.ud_grade(DM,self.nside)
        #alm = hp.map2alm(self.DM_map)
        #self.DM_map_2 = hp.pixelfunc.ud_grade(DM_2,self.nside)
        self.DM_map_2 = hp.alm2map(DM_2, nside=self.nside, lmax=5024)
        print(self.DM_map)
        print(self.DM_map_2)

        if plot == True:
            if self.lightcone_method[0] == 'SHELL':
                hp.mollview(self.DM_map, title=f"DM map (box={self.boxname}, sim={self.simname}, lightcone shell={self.z_sample-1}+{self.z_sample}+{self.z_sample+1})", cmap="jet")#, min=2e-5, max=2e-3)
                hp.graticule()
                plt.savefig(f'./Plots/DM_map_{self.boxname}_{self.simname}_{self.z_sample_name}_shell_{self.z_sample-1}-{self.z_sample+1}.png', dpi=400)
            elif self.lightcone_method[0] == 'FULL':
                hp.mollview(self.DM_map, title=f"Stacked DM map, integrated up to z=3 (box={self.boxname}, sim={self.simname})", cmap="jet")#, min=2e-5, max=2e-3)
                hp.graticule()
                plt.savefig(f'./Plots/DM_map_{self.boxname}_{self.simname}_stacked_z3p0.png', dpi=400)
            plt.clf()
        print(f'Loading relevant lightcone shells: {time.time() - self.job_start_time}s')

        return

    
    def load_halo_data(self, lightcone_type='HBT'):
        
        # Load halo lightcone and SOAP data into DataFrames
        if lightcone_type == 'HBT':
            if self.boxname == 'L1000N1800' and self.lightcone == 0:
                snap_max = 77
                halo_lc_dir = 'hbt_lightcone_halos'
            elif self.boxname == 'L2800N5040' and self.simname == 'HYDRO_FIDUCIAL':
                snap_max = 78
                halo_lc_dir = 'sorted_hbt_lightcone_halos'
            else:
                print("Halo lightcone not available for this box/simulation combination.")
                sys.exit()
            halo_lightcone = f'/cosma8/data/dp004/flamingo/Runs/{self.boxname}/{self.simname}/{halo_lc_dir}/lightcone{self.lightcone}/lightcone_halos_{snap_max-self.z_sample:04d}.hdf5'
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
            df_HBT = pl.DataFrame({
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
                
            df_mass = df_halo.filter(pl.col('mstar') > self.im)
            self.merge = df_mass.join(halo_lc_data, on='ID', how='inner').sort('ID')
            if self.merge.is_empty():
                self.merge = np.nan
        elif self.lightcone_method[1] == 'dndz':
            print(f"/cosma8/data/dp004/dc-conl1/FLAMINGO/patchy_screening/data_files/mock_halo_catalogs/{self.boxname}/{self.simname}/{self.z_sample_name}/lightcone{self.lightcone}/sampled_halo_data_{self.im_name}_{self.slope_name}.parquet")
            self.merge = pl.read_parquet(
                f"/cosma8/data/dp004/dc-conl1/FLAMINGO/patchy_screening/data_files/mock_halo_catalogs/{self.boxname}/{self.simname}/{self.z_sample_name}/lightcone{self.lightcone}/sampled_halo_data_{self.im_name}_{self.slope_name}.parquet"
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
            plt.savefig('./Plots/filter_plot.png', dpi=400)
            plt.clf()
        self.large_scale_map = hp.alm2map(lowpass_alm, nside=self.nside, lmax=3*self.nside-1)
        self.small_scale_map = hp.alm2map(highpass_alm, nside=self.nside, lmax=3*self.nside-1)
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
            plt.savefig(f'./Plots/random_cutout_T_ps_map_{self.boxname}_{self.simname}_{self.z_sample_name}.png', dpi=400)
            plt.clf()
            plt.close('all')
            grid_values_tau = np.histogram2d(theta_pix.copy()*60.0, phi_pix.copy()*60.0, bins=[grid_x, grid_y], weights=tau_2D)
            plt.figure(figsize=(6, 6))
            plt.imshow(grid_values_tau[0].T, origin='lower', extent=[grid_x.min(), grid_x.max(), grid_y.min(), grid_y.max()], cmap='inferno')#, vmin=-1e-6, vmax=1e-6)
            plt.colorbar(label='Optical Depth')
            plt.xlabel('X (arcmin)')
            plt.ylabel('Y (arcmin)')
            plt.title(f'Rectangular Cutout Around Halo = {i} w/ Patchy Screening', wrap=True)
            plt.savefig(f'./Plots/random_cutout_T_filtered_map_{self.boxname}_{self.simname}_{self.z_sample_name}.png', dpi=400)
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

        '''all_indices = np.arange(self.nhalo)
        my_indices  = all_indices[rank::size]

        # each rank computes its subset
        results = [ self.tau_prof(i, (plot, randint)) for i in my_indices ]

        # gather lists of arrays at root
        gathered = comm.gather(results, root=0)

        if rank == 0:
            # flatten and store
            flat = np.concatenate([np.stack(x, axis=0) for x in gathered], axis=0)
            self.data_1D = flat

        comm.Barrier()'''

        print(f'Starting profile loop: {time.time() - self.job_start_time}s')
        
        results = Parallel(n_jobs=self.ncpu, backend="loky", batch_size=batch_size)(delayed(self.tau_prof)(i, (plot,randint)) for i in range(self.nhalo))
        self.data_1D = np.asarray(results)
        print(f'Ending profile loop: {time.time() - self.job_start_time}s')
        return

    def stack_and_save(self):

        '''if rank != 0:
            return'''
        
        # Stacking of tau profiles and save as pickle files
        tau_1D_stack = np.zeros(len(self.theta_d))
        for i in range(self.nhalo):
            tau_1D = self.data_1D[i,:]
            tau_1D_stack += tau_1D
        tau_1D_stack /= self.nhalo

        '''tau_1D_stack = np.mean(self.data_1D, axis=0)
        print(self.simname,tau_1D_stack)
        quit()'''
        
        rows, cols = (len(self.theta_d), 4)
        data = [0]*cols
        data[0] = self.theta_d
        data[1] = tau_1D_stack
        data[2] = (self.theta_d*np.pi/(180.0*60.0))*self.Dcom
        data[3] = self.nhalo
        fits_suffix = "" if self.cmb_method=='CAMB' else f"_{self.fits_file}"
        signal_suffix = "" if self.signal==True else "_no_ps"
        noise_suffix = "" if self.rotate==False else "_noise"
        outfile = os.path.join(f'./{self.boxname}', self.z_sample_name, f'{self.simname}_tau_Mstar_bin{self.im_name}_{self.slope_name}_nside{self.nside}_{self.cmb_method}{fits_suffix}{signal_suffix}{noise_suffix}_ell_limited.pickle')
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
        
        self.run_tau_profiles(plot)
        self.stack_and_save()

        return

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
                T_patchy_screening = -1 * self.DM_map.copy() * self.mock_CMB_primary.copy()
                T_patchy_screening_2 = -1 * self.DM_map_2.copy() * self.mock_CMB_primary.copy()
            except AttributeError:
                self.generate_cmb_map(plot)
                self.load_lightcones(plot)
                T_patchy_screening = -1 * self.DM_map.copy() * self.mock_CMB_primary.copy()
                T_patchy_screening_2 = -1 * self.DM_map_2.copy() * self.mock_CMB_primary.copy()
            self.T_cmb_ps = T_patchy_screening + self.mock_CMB_primary.copy()
            self.T_cmb_ps_2 = T_patchy_screening_2 + self.mock_CMB_primary.copy()
            
            #alm = hp.map2alm(self.T_cmb_ps, lmax=3*self.nside-1)
            #self.T_cmb_ps = hp.alm2map(alm, nside=self.nside, lmax=5024)
            
            self.T_cmb_ps = hp.smoothing(self.T_cmb_ps,fwhm=1.3*np.pi/60.0/180.0)
            self.T_cmb_ps_2 = hp.smoothing(self.T_cmb_ps_2,fwhm=1.3*np.pi/60.0/180.0)
        elif self.signal == False:
            try:
                self.T_cmb_ps = hp.smoothing(self.mock_CMB_primary.copy(),fwhm=1.3*np.pi/60.0/180.0)
            except AttributeError:
                self.generate_cmb_map(plot)
                self.T_cmb_ps = hp.smoothing(self.mock_CMB_primary.copy(),fwhm=1.3*np.pi/60.0/180.0)
        if plot == True:
            hp.mollview(self.T_cmb_ps, title="CMB temperature map w/ Patchy Screening", cmap="jet")#, min=-1.5e-4, max=1.5e-4)
            hp.graticule()
            plt.savefig(f'./Plots/T_ps_map_{self.boxname}_{self.simname}_{self.z_sample_name}.png', dpi=400)
            plt.clf()
        print(f'Generating patchy screening map: {time.time() - self.job_start_time}s')

        return

if __name__ == '__main__':

    ncpu = sys.argv[1]
    box = sys.argv[2]
    isim = sys.argv[3]
    iz = sys.argv[4]
    im = sys.argv[5]
    slope = sys.argv[6]
    fits = sys.argv[7]
    sig = sys.argv[8]

    ps = patchyScreening(box, isim, iz, im, slope, ncpu, fits_file=fits, signal=sig)#, cmb_method='CAMB')#, lightcone_method=('SHELL','shell'))#, cmb_method='CAMB')
    ps.run_analysis(plot=False)
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
