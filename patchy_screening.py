import sys
import h5py
import pandas as pd
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import numpy.ma as ma
import astropy.units as u
import os
import pickle
from astropy.cosmology import FlatLambdaCDM
from joblib import Parallel, delayed
import camb
from camb import model, initialpower
import time

job_start_time = time.time()

ncpu=int(sys.argv[1])
isel=int(sys.argv[2])

cosmo = FlatLambdaCDM(H0=68.1, Om0=0.256011, Tcmb0=2.726)

hp.disable_warnings()
plt.rcParams.update({'font.size': 14})
plt.rcParams.update({'font.family': 'STIXGeneral'})
plt.rcParams.update({'mathtext.fontset': 'stix'})

sim_list=['HYDRO_FIDUCIAL','HYDRO_PLANCK','HYDRO_PLANCK_LARGE_NU_FIXED','HYDRO_PLANCK_LARGE_NU_VARY','HYDRO_STRONG_AGN','HYDRO_WEAK_AGN','HYDRO_LOW_SIGMA8','HYDRO_STRONGER_AGN','HYDRO_JETS','HYDRO_STRONGEST_AGN','HYDRO_STRONG_SUPERNOVA','HYDRO_STRONGER_AGN_STRONG_SUPERNOVA','HYDRO_STRONG_JETS']

sim_list2=['HYDRO_FIDUCIAL','HYDRO_PLANCK','HYDRO_PLANCK_LARGE_NU_FIXED','HYDRO_PLANCK_LARGE_NU_VARY','HYDRO_STRONG_AGN','HYDRO_WEAK_AGN','HYDRO_LOW_SIGMA8','HYDRO_STRONGER_AGN','HYDRO_JETS_old','HYDRO_STRONGEST_AGN','HYDRO_STRONG_SUPERNOVA','HYDRO_STRONGER_AGN_STRONG_SUPERNOVA','HYDRO_STRONG_JETS_old']


nsim=len(sim_list)

Tcmb=2.726

#Rectangle size in arcminutes
rect_size = 20
rect_shape = rect_size * (0.5/60.0) #In degrees

# define some angular bins (arcmins) for the kSZ profiles to be measured
theta_d=np.arange(0.5,11.0,0.5)
ntheta_d=len(theta_d)

#For VR
# i=30 1.475 - 1.525
# i=29 1.425 - 1.475
# i=28 1.375 - 1.425
# i=27 1.325 - 1.375
# i=26 1.275 - 1.325
# i=25 1.225 - 1.275
# i=24 1.175 - 1.225
# i=23 1.125 - 1.175
# i=22 1.075 - 1.125
# i=21 1.025 - 1.075
# i=20 0.975 - 1.025
# i=19 0.925 - 0.975
# i=18 0.875 - 0.925
# i=17 0.825 - 0.875
# i=16 0.775 - 0.825
# i=15 0.725 - 0.775
# i=14 0.675 - 0.725
#
# i=13 0.625 - 0.675
# i=12 0.575 - 0.625
# i=11 0.525 - 0.575
# i=10 0.475 - 0.525
# i=9  0.425 - 0.475
# i=8  0.375 - 0.425
# i=7  0.325 - 0.375
# i=6  0.275 - 0.325
# i=5  0.225 - 0.275
# i=4  0.175 - 0.225
# i=3  0.125 - 0.175
# i=2  0.075 - 0.125 
# i=1  0.025 - 0.075
#

# Define the frequency band-pass filter function
def f_lowpass(l):
    if l < 2000:
        return 1
    elif l > 2150:
        return 0
    else:
        return np.cos(((l - 2000) * np.pi) / 300)

def f_highpass(l):
    if l < 2350:
        return 0
    elif l > 2500:
        return 1
    else:
        return np.sin(((l - 2350) * np.pi) / 300)


# select appropriate snapshot numbers for your galaxy samples
#isam=np.array([12,22,30])
isam=np.array([11,21,29])
survey=['Blue','Green','Red']
ns=len(survey)

# randomly select nsamp systems for analysis
nsamp=100000

# create profiles in bins of minimum stellar mass
mstar_bins=10**(np.array([10.9,11.0,11.1,11.2,11.3,11.4,11.5,11.6]))
mstar_bins_name = ['10p9','11p0','11p1','11p2','11p3','11p4','11p5','11p6']
#mstar_bins=10**(np.array([10.9,11.0]))
nm=len(mstar_bins)

randint = np.random.randint(nsamp)
def tau_prof(i):

    '''theta_center, phi_center = hp.vec2ang(source_vector[i, :], lonlat=True)
    print('theta_center, phi_center')
    print(theta_center, phi_center)

    vertices = [
        [theta_center - rect_shape , phi_center - rect_shape],
        [theta_center + rect_shape , phi_center - rect_shape],
        [theta_center + rect_shape , phi_center + rect_shape],
        [theta_center - rect_shape , phi_center + rect_shape]
    ]
    print('vertices')
    print(vertices)
    
    vec_vertices = np.array([hp.ang2vec(t, p, lonlat=True) for t, p in vertices]).reshape(-1,3)
    print('vec_vertices')
    print(vec_vertices)
    print(vec_vertices.shape)
    halo_pixels = hp.query_polygon(nside, vec_vertices)'''
    halo_pixels = hp.query_disc(nside, source_vector[i,:], radius=(0.25*u.deg).to_value(u.radian))
    theta_pix, phi_pix = hp.pix2ang(nside,halo_pixels,lonlat=True)
    rtheta=hp.rotator.angdist([theta[i],phi[i]],[theta_pix,phi_pix],lonlat=True)*60.0*180.0/np.pi

    #fluxes=dT[halo_pixels]
    large_scale_fluxes=large_scale_map[halo_pixels]
    small_scale_fluxes=small_scale_map[halo_pixels]

    tau_2D = (np.sign(large_scale_fluxes)*small_scale_fluxes)/mean_mod_T_large_scale

    grid_x = np.linspace(min(theta_pix.copy()*60.0), max(theta_pix.copy()*60.0), 25)  # Adjust grid size as needed
    grid_y = np.linspace(min(phi_pix.copy()*60.0), max(phi_pix.copy()*60.0), 25)
    
    if i==randint:
        #if isinstance(i,bool):
    
        print('halo_pixels')
        print(len(halo_pixels))
        print('theta_pix, phi_pix')
        print(theta_pix.copy()*60.0, phi_pix.copy()*60.0)
        print('theta[i],phi[i]')
        print(theta[i].copy()*60.0,phi[i].copy()*60.0)
        print('rtheta')
        print(rtheta)
        print('large_scale_fluxes')
        print(large_scale_fluxes)
        print('small_scale_fluxes')
        print(small_scale_fluxes)
        print('tau_2D.shape')
        print(tau_2D.shape)
        
        #grid_x = np.linspace(-rect_size, rect_size, 20)  # Adjust grid size as needed
        #grid_y = np.linspace(-rect_size, rect_size, 20)
        #grid_x = np.linspace(min(theta_pix), max(theta_pix), 30)  # Adjust grid size as needed
        #grid_y = np.linspace(min(phi_pix), max(phi_pix), 30)
        print('grid_x')
        print(grid_x)
        print('grid_y')
        print(grid_y)

        temperature_map = T_cmb_ps[halo_pixels]
        grid_values_temp, xedges_temp, yedges_temp = np.histogram2d(theta_pix.copy()*60.0, phi_pix.copy()*60.0, bins=[grid_x, grid_y], weights=temperature_map)
        print('temperature_map')
        print(temperature_map)
        #print('grid_values_temp')
        #print(grid_values_temp)
        #print(xedges_temp, yedges_temp)
        
        plt.figure(figsize=(6, 6))
        #plt.imshow(grid_values_temp.T, origin='lower', extent=[xedges_temp[0], xedges_temp[-1], yedges_temp[0], yedges_temp[-1]], cmap='inferno')#, vmin=-1.5e-4, vmax=1.5e-4)
        plt.imshow(grid_values_temp.T, origin='lower', extent=[grid_x.min(), grid_x.max(), grid_y.min(), grid_y.max()], cmap='inferno')#, vmin=-1.5e-4, vmax=1.5e-4)
        plt.colorbar(label=r'Temperature ($\mu$K)')
        plt.xlabel('X (arcmin)')
        plt.ylabel('Y (arcmin)')

        #plt.imshow(masked_cutout.T, origin='lower', cmap='jet', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
        #plt.colorbar(label='Temperature (muK)')
        plt.title(f'Rectangular Cutout Around Halo = {i}', wrap=True)
        #plt.xlabel(r'$\theta$ (deg)')
        #plt.ylabel(r'$\phi$ (deg)')
        plt.savefig(f'./Plots/random_cutout_T_ps_map_{simname2}_{survey[iz]}.png', dpi=1200)
        plt.clf()

        
        tau_map = tau_2D
        grid_values_tau = np.histogram2d(theta_pix.copy()*60.0, phi_pix.copy()*60.0, bins=[grid_x, grid_y], weights=tau_map)

        plt.figure(figsize=(6, 6))
        plt.imshow(grid_values_tau[0].T, origin='lower', extent=[grid_x.min(), grid_x.max(), grid_y.min(), grid_y.max()], cmap='inferno')#, vmin=-1e-6, vmax=1e-6)
        plt.colorbar(label='Optical Depth')
        plt.xlabel('X (arcmin)')
        plt.ylabel('Y (arcmin)')
        plt.title(f'Rectangular Cutout Around Halo = {i} w/ Patchy Screening', wrap=True)
        plt.savefig(f'./Plots/random_cutout_T_filtered_map_{simname2}_{survey[iz]}.png', dpi=1200)
        plt.clf()
    
    '''tau=np.zeros(ntheta_d)
    for j in range(ntheta_d):
        # set up compensated aperture photometry weights
        print('theta_d[j]')
        print(theta_d[j])
        idx_in=np.where(rtheta < theta_d[j])
        print('idx_in')
        print(idx_in)
        idx_out=np.where((rtheta >= theta_d[j]) & (rtheta < np.sqrt(2.0)*theta_d[j]))
        print('idx_out')
        print(idx_out)
        print('pix_area_arcmin2')
        print(pix_area_arcmin2)
        area_in=np.pi*theta_d[j]**2.0/(len(rtheta[idx_in])*pix_area_arcmin2)
        area_out=np.pi*theta_d[j]**2.0/(len(rtheta[idx_out])*pix_area_arcmin2)
        print('area_in')
        print(area_in)
        print('area_out')
        print(area_out)
        tau[j]=(np.sum(fluxes[idx_in])*area_in - np.sum(fluxes[idx_out])*area_out)*pix_area_arcmin2
    print(tau)'''

    tau_1D = np.zeros(ntheta_d)
    for j in range(ntheta_d):
        #print('theta_d[j]')
        #print(theta_d[j])
        #idx_in = np.where(rtheta < theta_d[j])
        #area_in = np.pi*theta_d[j]**2.0/(len(rtheta[idx_in])*pix_area_arcmin2)
        #print('area_in')
        #print(area_in)
        #tau_1D[j] = (np.sum(np.sign(large_scale_fluxes[idx_in])*small_scale_fluxes[idx_in])*area_in)*pix_area_arcmin2
        idx_in = np.where((rtheta > theta_d[j]-0.25) & (rtheta <= theta_d[j]+0.25))
        #area_in = np.pi*((theta_d[j]+0.1)**2.0 - (theta_d[j]-0.1)**2.0)/(len(rtheta[idx_in])*pix_area_arcmin2)
        #print('area_in')
        #print(area_in)
        #print('area_out')
        #print(area_out)
        tau_1D[j] = (1*np.mean(np.sign(large_scale_fluxes[idx_in])*small_scale_fluxes[idx_in]))/mean_mod_T_large_scale    #*area_in)*pix_area_arcmin2 #Mean
        #tau_1D[j] = (np.sum(np.sign(large_scale_fluxes[idx_out])*small_scale_fluxes[idx_out])*area_out/len(idx_out) - np.sum(np.sign(large_scale_fluxes[idx_in])*small_scale_fluxes[idx_in])*area_in/len(idx_in ))*pix_area_arcmin2 #Sum

    if i==randint:
        print('tau_1D')
        print(tau_1D)

    return tau_1D, tau_2D


for isim in range(isel,isel+1):

    simname=sim_list[isim]
    simname2=sim_list2[isim]

    nside=8192
    #nside=4096
    
    #Using FLAMINGO FIDUCIAL cosmology parameters
    pars = camb.set_params(H0=68.1, ombh2=0.048600*(0.681**2), omch2=0.256011*(0.681**2), mnu=0.06, As=2.099e-9, ns=0.967, lmax=3*nside)
    results = camb.get_results(pars)
    powers = results.get_cmb_power_spectra(pars, raw_cl=True)
    
    unlens_total_CL = powers['unlensed_total']
    #unlens_scalar_CL = powers['unlensed_scalar']
    print(unlens_total_CL[:,0])

    np.random.seed(1000)
    mock_CMB_primary = hp.synfast(unlens_total_CL[:,0], nside=nside)
    print(mock_CMB_primary)

    print(f'Generating mock primary CMB: {time.time() - job_start_time}s')

    hp.mollview(mock_CMB_primary, title=f"Mock Primary CMB temperature map (sim={simname})", cmap="jet")#, min=-1.5e-4, max=1.5e-4)
    hp.graticule()
    plt.savefig(f'./Plots/primary_CMB_map_{simname}.png', dpi=1200)
    plt.clf()

    #mock_CMB_primary_alt = hp.synfast(unlens_scalar_CL[:,0], nside=nside)

    #hp.mollview(mock_CMB_primary_alt, title=f"Mock Primary CMB temperature map 'unlensed scalar' (sim={simname})", cmap="jet")
    #hp.graticule()
    #plt.savefig(f'./Plots/primary_CMB_map_{simname}_alt.png', dpi=1200)
    #plt.clf()


    lensed_dir = '/cosma8/data/dp004/dc-yang3/maps/L1000N1800/HYDRO_FIDUCIAL/lightcone0_shells/patchy_screening_folder'
    unlensed_CMB = hp.read_map(f'{lensed_dir}/CMB_T_map_unl.fits', dtype=None, verbose=False)
    lensed_CMB_z2 = hp.read_map(f'{lensed_dir}/CMB_T_map_l_kappa_z2.fits', dtype=None, verbose=False)
    lensed_CMB_z3 = hp.read_map(f'{lensed_dir}/CMB_T_map_l_kappa_z3.fits', dtype=None, verbose=False)
    
    '''alm_primary_cmb = hp.map2alm(mock_CMB_primary)

    plt.scatter(np.arange(3*nside-1), np.abs(alm_primary_cmb[np.arange(3*nside-1)]))
    plt.xlabel(r'$\ell$')
    plt.ylabel('Alm values')
    plt.yscale('log')
    plt.title(f'Alm for Primary CMB (total)')
    #plt.xlim(left=1900, right=2600)
    plt.savefig(f'./Plots/alm_plot_CMB.png', dpi=1200)
    plt.clf()

    print(len(unlens_total_CL[:, 0]), 3 * nside - 1)

    plt.plot(unlens_total_CL[:, 0])
    plt.xlabel("l")
    plt.ylabel("C_l")
    plt.title("Input C_l from CAMB")
    plt.xlim(left=-1, right=100)
    plt.savefig(f'./Plots/CAMB_cls.png', dpi=1200)
    plt.clf()'''

    '''DM_total = 0
    for i in range(60):
        map_lightcone = f'/cosma8/data/dp004/flamingo/Runs/L1000N1800/'+simname2+'/neutrino_corrected_maps/lightcone0_shells/shell_'+str(i)+'/lightcone0.shell_'+str(i)+'.0.hdf5'
        g = h5py.File(map_lightcone,'r')
        DM = g['DM'][...]*g['DM'].attrs['Conversion factor to CGS (not including cosmological corrections)']*6.6524587321e-25
        DM_total += DM

    DM_mean = np.mean(DM_total)
    print(f'Mean tau = {DM_mean}')

    DM_total -= DM_mean

    hp.mollview(DM_total, title=f"Total DM map (sim={simname2})", cmap="jet")#, min=2e-5, max=2e-3)
    hp.graticule()
    plt.savefig(f'./Plots/DM_map_{simname2}_total.png', dpi=1200)
    plt.clf()

    DM_total = hp.pixelfunc.ud_grade(DM_total,nside)

    T_patchy_screening = DM_total.copy() * mock_CMB_primary.copy()

    T_cmb_ps = T_patchy_screening.copy() + mock_CMB_primary.copy()

    T_cmb_ps = hp.smoothing(T_cmb_ps,fwhm=1.3*np.pi/60.0/180.0)  # smooth map by ACT 150GHz (1.3 arcmin) beam'''
    
    for iz in range(0,ns):

        i = isam[iz]
        map_lightcone = f'/cosma8/data/dp004/flamingo/Runs/L1000N1800/'+simname2+'/neutrino_corrected_maps/lightcone0_shells/shell_'+str(i)+'/lightcone0.shell_'+str(i)+'.0.hdf5'
        #map_lightcone = f'/cosma8/data/dp004/flamingo/Runs/L1000N1800/'+simname2+'/neutrino_corrected_maps_downsampled_4096/lightcone0_shells/shell_'+str(i)+'/lightcone0.shell_'+str(i)+'.0.hdf5' #Low Res lighcone
        g = h5py.File(map_lightcone,'r')
        conversion_factor = g['DM'].attrs['Conversion factor to CGS (not including cosmological corrections)']
        #print(g['DM'][...])
        DM = g['DM'][...]*conversion_factor*6.6524587321e-25 #6.65246e-25 = Thomson cross-section (in cgs)
        #print(DM)

        redshift = g['DM'].attrs['Central redshift assumed for correction']
        #print(f'redshift of {survey[iz]} sample')
        #print(redshift)

        hp.mollview(DM, title=f"DM map (sim={simname2}, lightcone shell={i})", cmap="jet", min=2e-5, max=2e-3)
        hp.graticule()
        plt.savefig(f'./Plots/DM_map_{simname2}_{survey[iz]}_shell_{i}.png', dpi=1200)
        plt.clf()

        print(f'Loading first lightcone shell: {time.time() - job_start_time}s')

        map_lightcone = f'/cosma8/data/dp004/flamingo/Runs/L1000N1800/'+simname2+'/neutrino_corrected_maps/lightcone0_shells/shell_'+str(i-1)+'/lightcone0.shell_'+str(i-1)+'.0.hdf5'
        #map_lightcone = f'/cosma8/data/dp004/flamingo/Runs/L1000N1800/'+simname2+'/neutrino_corrected_maps_downsampled_4096/lightcone0_shells/shell_'+str(i)+'/lightcone0.shell_'+str(i)+'.0.hdf5' #Low Res lighcone
        g = h5py.File(map_lightcone,'r')
        conversion_factor = g['DM'].attrs['Conversion factor to CGS (not including cosmological corrections)']
        DM += g['DM'][...]*conversion_factor*6.6524587321e-25

        map_lightcone = f'/cosma8/data/dp004/flamingo/Runs/L1000N1800/'+simname2+'/neutrino_corrected_maps/lightcone0_shells/shell_'+str(i+1)+'/lightcone0.shell_'+str(i+1)+'.0.hdf5'
        #map_lightcone = f'/cosma8/data/dp004/flamingo/Runs/L1000N1800/'+simname2+'/neutrino_corrected_maps_downsampled_4096/lightcone0_shells/shell_'+str(i)+'/lightcone0.shell_'+str(i)+'.0.hdf5' #Low Res lighcone
        g = h5py.File(map_lightcone,'r')
        conversion_factor = g['DM'].attrs['Conversion factor to CGS (not including cosmological corrections)']
        DM += g['DM'][...]*conversion_factor*6.6524587321e-25

        hp.mollview(DM, title=f"DM map (sim={simname2}, lightcone shell={i-1}+{i}+{i+1})", cmap="jet")#, min=2e-5, max=2e-3)
        hp.graticule()
        plt.savefig(f'./Plots/DM_map_{simname2}_{survey[iz]}_shell_{i-1}-{i+1}.png', dpi=1200)
        plt.clf()

        DM = hp.pixelfunc.ud_grade(DM,nside)

        '''alm_DM = hp.map2alm(DM)

        plt.scatter(np.arange(3*nside-1), np.abs(alm_DM[np.arange(3*nside-1)]))
        plt.xlabel(r'$\ell$')
        plt.ylabel('Alm values')
        plt.yscale('log')
        plt.title(f'Alm for DM')
        plt.savefig(f'./Plots/alm_DM.png', dpi=1200)
        plt.clf()'''

        pix_area_arcmin2=hp.pixelfunc.nside2pixarea(nside,degrees=True)*60.0*60.0

        T_patchy_screening = DM.copy() * unlensed_CMB.copy() #lensed_CMB_z3.copy() #unlensed_CMB.copy() #mock_CMB_primary.copy()
        
        T_cmb_ps = T_patchy_screening.copy() + unlensed_CMB.copy() #lensed_CMB_z3.copy() #unlensed_CMB.copy() #mock_CMB_primary.copy() 

        '''alm_cmb_ps = hp.map2alm(T_cmb_ps)

        plt.scatter(np.arange(3*nside-1), np.abs(alm_cmb_ps[np.arange(3*nside-1)]))
        plt.xlabel(r'$\ell$')
        plt.ylabel('Alm values')
        plt.yscale('log')
        plt.title(f'Alm for CMB w/ PS')
        plt.savefig(f'./Plots/alm_plot_CMB_PS.png', dpi=1200)
        plt.clf()'''

        lmax = 3 * nside - 1
        #beam = '1p3'
        T_cmb_ps = hp.smoothing(T_cmb_ps,fwhm=1.3*np.pi/60.0/180.0)  # smooth map by ACT 150GHz (1.3 arcmin) beam
        #T_cmb_ps = hp.smoothing(T_cmb_ps,fwhm=1.6*np.pi/60.0/180.0)

        print(f'Generating patchy screening map: {time.time() - job_start_time}s')
        
        '''
        # transform Doppler B maps into temperature fluctuation maps (in microKelvin)
        dT=-Tcmb*DM/1e-6
        del DM
        '''
        '''hp.mollview(T_cmb_ps, title="CMB temperature map w/ Patchy Screening", cmap="jet")#, min=-1.5e-4, max=1.5e-4)
        hp.graticule()
        plt.savefig(f'./Plots/T_ps_map_{simname2}_{survey[iz]}.png', dpi=1200)
        plt.clf()

        #plt.hist(DM, bins='auto')
        #plt.xlabel('Tau')
        #plt.ylabel('Freqency')
        #plt.savefig(f'./Plots/tau_histogram_{simname2}_{survey[iz]}.png', dpi=1200)
        #plt.clf()'''

        # Read halo lightcone data
        #if i >= 10: halo_lightcone = f'/cosma8/data/dp004/jch/FLAMINGO/lightcone_halos/'+simname+'/lightcone_halos/lightcone0/lightcone_halos_00'+str(i)+'.hdf5'   #VR
        #if i < 10: halo_lightcone = f'/cosma8/data/dp004/jch/FLAMINGO/lightcone_halos/'+simname+'/lightcone_halos/lightcone0/lightcone_halos_000'+str(i)+'.hdf5'   #VR
        if i >= 10: halo_lightcone = f'/cosma8/data/dp004/jch/FLAMINGO/HBT/L1000N1800/'+simname+'/lightcone_halos/lightcone0/lightcone_halos_00'+str(77-i)+'.hdf5'     #HBT
        if i < 10: halo_lightcone = f'/cosma8/data/dp004/jch/FLAMINGO/HBT/L1000N1800/'+simname+'/lightcone_halos/lightcone0/lightcone_halos_000'+str(77-i)+'.hdf5'     #HBT
        f = h5py.File(halo_lightcone, 'r')
        halo_lc_data = pd.DataFrame()
        '''
        halo_lc_data['ID'] = f['Subhalo/ID'][...]
        halo_lc_data['SnapNum'] = f['Subhalo/SnapNum'][...]
        halo_lc_data['z'] = f['Subhalo/LightconeRedshift'][...]
        halo_lc_data['xminpot'] = f['Subhalo/LightconeXcminpot'][...]
        halo_lc_data['yminpot'] = f['Subhalo/LightconeYcminpot'][...]
        halo_lc_data['zminpot'] = f['Subhalo/LightconeZcminpot'][...]
        '''
        halo_lc_data['ID'] = f['InputHalos/HaloCatalogueIndex'][...]
        halo_lc_data['SnapNum'] = f['Lightcone/SnapshotNumber'][...]
        halo_lc_data['z'] = f['Lightcone/Redshift'][...]
        halo_centre = f['Lightcone/HaloCentre'][...]
        halo_lc_data['xminpot'] = halo_centre[:,0]
        halo_lc_data['yminpot'] = halo_centre[:,1]
        halo_lc_data['zminpot'] = halo_centre[:,2]
        
        print(np.min(halo_lc_data.z),np.max(halo_lc_data.z))

        Dcom = cosmo.comoving_distance(np.mean(halo_lc_data.z))*0.681  # comoving distance to galaxy in Mpc/h
        Dcom = Dcom.value

        # Sorting makes the matching faster:
        halo_lc_data.sort_values(by='ID', inplace=True)
        halo_lc_data.reset_index(inplace=True, drop=True)

        # Get the SNAP number
        snap = int(halo_lc_data.iloc[0]['SnapNum'])

        # Read SOAP or VR data:
        #VR_file = f'/cosma8/data/dp004/flamingo/Runs/L1000N1800/'+simname2+'/VR/catalogue_00'+str(snap)+'/vr_catalogue_00'+str(snap)+'.properties.0'
        #VR_file = f'/cosma8/data/dp004/flamingo/Runs/L1000N1800/'+simname2+'/SOAP/halo_properties_00'+str(snap)+'.hdf5'
        HBT_file = f'/cosma8/data/dp004/flamingo/Runs/L1000N1800/'+simname2+'/SOAP-HBT/halo_properties_00'+str(snap)+'.hdf5'
        f = h5py.File(HBT_file, 'r')
        '''
        df_VR = pd.DataFrame()
        df_VR['ID'] = f['VR/ID'][...]
        #df_VR['hostHaloID'] = f['VR/HostHaloID'][...]
        df_VR['Structuretype'] = f['VR/StructureType'][...]
        df_VR['m_vir'] = f['SO/500_crit/TotalMass'][...]
        df_VR['mstar'] = f['ExclusiveSphere/50kpc/StellarMass'][...]
        '''
        df_HBT = pd.DataFrame()
        df_HBT['ID'] = f['InputHalos/HaloCatalogueIndex'][...]
        #df_HBT['hostHaloID'] = f['HBT/HostHaloID'][...]
        df_HBT['Structuretype'] = f['InputHalos/IsCentral'][...]
        df_HBT['m_vir'] = f['SO/500_crit/TotalMass'][...]
        df_HBT['mstar'] = f['ExclusiveSphere/50kpc/StellarMass'][...]
        df_HBT.mstar*=1e10
        df_HBT.m_vir*=1e10

        print(f'Loading halo lightcone data: {time.time() - job_start_time}s')
        
        
        '''
        Vel=f['BoundSubhaloProperties/CentreOfMassVelocity'][...]
        vx=np.asarray(Vel[:,0])
        vy=np.asarray(Vel[:,1])
        vz=np.asarray(Vel[:,2])
        df_VR['vx']=vx
        df_VR['vy']=vy
        df_VR['vz']=vz
        '''

        # select all systems or just centrals (structuretype=10)
        '''
        df_VR = df_VR.loc[df_VR.Structuretype == 10]
        '''

        # loop over halo mass bins
        for im in range(0,nm):
            #df_mass = df_VR
            df_mass = df_HBT
            #df_mass = df_mass.loc[df_mass.m_vir > 0.95*mvir_bins[im]]
            #df_mass = df_mass.loc[df_mass.m_vir < 1.05*mvir_bins[im]]
            #df_mass = df_mass.loc[df_mass.mstar > 0.98*mstar_bins[im]]
            #df_mass = df_mass.loc[df_mass.mstar < 1.02*mstar_bins[im]]
            df_mass = df_mass.loc[df_mass.mstar > mstar_bins[im]]
            #df_mass = df_mass.loc[df_mass.mstar < 5.5e11]
            df_mass.sort_values(by='ID', inplace=True)
            df_mass.reset_index(inplace=True, drop=True)
            merge = pd.merge_ordered(df_mass, halo_lc_data, on=['ID'], how='inner')
            x=np.asarray(merge.xminpot)
            y=np.asarray(merge.yminpot)
            z=np.asarray(merge.zminpot)
            '''
            vx=np.asarray(merge.vx)
            vy=np.asarray(merge.vy)
            vz=np.asarray(merge.vz)
            '''
            mvir=np.asarray(merge.m_vir)
            mstar=np.asarray(merge.mstar)
            #vr=(vx*x+vy*y+vz*z)/np.sqrt(x*x+y*y+z*z)
            nhalo=len(mvir)

            '''if nhalo > nsamp:
                idx=np.random.randint(0,nhalo-1,size=nsamp)
                x=x[idx]
                y=y[idx]
                z=z[idx]
                #vr=vr[idx]
                mvir=mvir[idx]
                mstar=mstar[idx]
                nhalo=len(mvir)'''

            print(im,np.log10(np.min(mstar)),np.log10(np.mean(mstar)),np.log10(np.mean(mvir)),nhalo)
            print(f'Identifying stackable objects: {time.time() - job_start_time}s')

            alm = hp.map2alm(T_cmb_ps, lmax=lmax)
            #print('alm')
            #print(alm)
            #print(alm.shape)
            ell,m = hp.Alm.getlm(lmax=lmax)
            #print('ell, m')
            #print(ell, m)
            #print(ell.shape)

            '''m_zero_1_2 = np.where((m==0) | (m==1) | (m==2))
            #plt.scatter(ell[m_zero_1_2], np.abs(alm[m_zero_1_2].copy()), label='Original alm (m=0,1,2)')
            plt.scatter(np.arange(lmax), np.abs(alm[np.arange(lmax)].copy()), label='Original alm (m=0)')
            plt.xlabel(r'$\ell$')
            plt.ylabel('Alm values')
            plt.yscale('log')
            plt.title(f'Alm')
            plt.legend()
            plt.savefig(f'./Plots/alm_plot_orig.png', dpi=1200)
            plt.clf()'''

            #angles = np.random.uniform(0, 360)  # Random Euler angles (degrees)
            #angle_range = hp.pix2ang(nside=nside, ipix=T_cmb_ps, lonlat=True)
            #angle_theta = np.random.uniform(low=np.min(angle_range[0]), high=np.max(angle_range[0]))
            #angle_phi = np.random.uniform(low=np.min(angle_range[1]), high=np.max(angle_range[1]))
            
            lowpass_values = np.array([f_lowpass(l) for l in ell])
            #print('lowpass_values')
            #print(lowpass_values)
            #print(lowpass_values.shape)
            highpass_values = np.array([f_highpass(l) for l in ell])
            #print('highpass_values')
            #print(highpass_values)
            #print(highpass_values.shape)

            #rotations = 100
            #for r in range(rotations):
            '''np.random.seed(int(sys.argv[3]))
            rotated_alm = hp.Rotator(deg=True, rot=(np.random.uniform(0, 180), np.random.uniform(0, 360))).rotate_alm(alm, lmax=lmax)
            alm = rotated_alm
            
                #print(hp.get_nside(T_cmb_ps))
                #print("Expected lmax:", 3 * hp.get_nside(T_cmb_ps) - 1)
                #nside_get = hp.get_nside(T_cmb_ps)
                #pixwin = hp.pixwin(nside_get)
                #alm_corrected = hp.almxfl(alm.copy(), 1 / pixwin[:lmax+1])
                #print(f"Beam FWHM (arcmin): {hp.nside2resol(hp.get_nside(T_cmb_ps)) * 60}")
                #plt.plot(pixwin)
                #plt.xlabel("l")
                #plt.ylabel("Pixel Window Function")
                #plt.savefig(f'./Plots/pixwin.png', dpi=1200)
                #plt.clf()
            
                        
                #sample_ell = np.arange(max(ell))
                #m_zero = np.where(m==0)
                #m_pone = np.where(m==1)
                #m_ptwo = np.where(m==2)
                #plt.scatter(ell[sample_ell], lowpass_values[sample_ell], label='lowpass filter', color='b')
                #plt.scatter(ell[sample_ell], highpass_values[sample_ell], label='highpass filter', color='r')
                
                #plt.plot(ell[m_zero], lowpass_values[m_zero], label='lowpass filter', color='b')
                #plt.plot(ell[m_zero], highpass_values[m_zero], label='highpass filter', color='r')
                #plt.xlabel(r'$\ell$')
                #plt.ylabel('Filter values')
                #plt.title('My version of the Coulton frequency filters (m=0,1,2)')
                #plt.legend()
                #plt.savefig('./Plots/filter_plot.png', dpi=1200)
                #plt.clf()

                #plt.plot(ell[m_zero], lowpass_values[m_zero], label='lowpass filter', color='b')
                #plt.plot(ell[m_zero], highpass_values[m_zero], label='highpass filter', color='r')
                #plt.xlabel(r'$\ell$')
                #plt.ylabel('Filter values')
                #plt.xlim(left=1900, right=2600)
                #plt.title('My version of the Coulton frequency filters (m=0,1,2)')
                #plt.legend()
                #plt.savefig('./Plots/filter_plot_zoom.png', dpi=1200)
                #plt.clf()
                
                #lowpass_alm = hp.almxfl(rotated_alm.copy(), lowpass_values)
                lowpass_alm_rotated = hp.almxfl(rotated_alm.copy(), lowpass_values)
                
                #print('lowpass_alm')
                #print(lowpass_alm)
                #print(np.abs(lowpass_alm))
                #print(lowpass_alm.shape)

                #highpass_alm = hp.almxfl(rotated_alm.copy(), highpass_values)
                highpass_alm_rotated = hp.almxfl(rotated_alm.copy(), highpass_values)
                
                #print('highpass_alm')
                #print(highpass_alm)
                #print(np.abs(highpass_alm))
                #print(highpass_alm.shape)

                #UNCOMMENT
                
                #plt.scatter(np.arange(lmax), np.abs(alm[np.arange(lmax)]), label='Original alm (m=0)')#, color='blue')
                #plt.scatter(np.arange(lmax), np.abs(highpass_alm[np.arange(lmax)]), label='Highpass filtered alm', alpha=.5)
                #plt.scatter(np.arange(lmax), np.abs(lowpass_alm[np.arange(lmax)]), label='Lowpass filtered alm', alpha=.5)
                #plt.xlabel(r'$\ell$')
                #plt.ylabel('Alm values')
                #plt.yscale('log')
                #plt.title(f'Alm before and after filtering')
                #plt.legend()
                #plt.savefig(f'./Plots/alm_plot.png', dpi=1200)
                #plt.clf()

                #############

                #plt.scatter(np.arange(lmax), np.abs(alm[np.arange(lmax)]), label='Original alm (m=0)')
                #plt.scatter(np.arange(lmax), np.abs(highpass_alm[np.arange(lmax)]), label='Highpass filtered alm', alpha=.5)
                #plt.scatter(np.arange(lmax), np.abs(lowpass_alm[np.arange(lmax)]), label='Lowpass filtered alm', alpha=.5)
                #plt.xlabel(r'$\ell$')
                #plt.ylabel('Alm values')
                #plt.yscale('log')
                #plt.xlim(left=1900, right=2600)
                #plt.title(f'Alm before and after filtering')
                #plt.legend()
                #plt.savefig(f'./Plots/alm_plot_zoom.png', dpi=1200)
                #plt.clf()

                #plt.scatter(ell[m_pone], np.abs(alm[m_pone]), label='Original alm (m=+1)')
                #plt.scatter(ell[m_pone], np.abs(highpass_alm[m_pone]), label='Highpass filtered alm', alpha=.5)
                #plt.scatter(ell[m_pone], np.abs(lowpass_alm[m_pone]), label='Lowpass filtered alm', alpha=.5)
                #plt.xlabel(r'$\ell$')
                #plt.ylabel('Alm values')
                #plt.yscale('log')
                #plt.title('Alm before and after filtering')
                #plt.legend()
                #plt.savefig('./Plots/alm_plot_mone.png', dpi=1200)
                #plt.clf()

                #plt.scatter(ell[m_zero_1_2], np.abs(alm[m_zero_1_2]), label='Original alm (m=0,1,2)')
                #plt.scatter(ell[m_zero_1_2], np.abs(highpass_alm[m_zero_1_2]), label='Highpass filtered alm', alpha=.5)
                #plt.scatter(ell[m_zero_1_2], np.abs(lowpass_alm[m_zero_1_2]), label='Lowpass filtered alm', alpha=.5)
                #plt.plot(ell[m_zero], lowpass_values[m_zero], label='lowpass filter', color='b')
                #plt.plot(ell[m_zero], highpass_values[m_zero], label='highpass filter', color='r')
                #plt.xlabel(r'$\ell$')
                #plt.ylabel('Alm values')
                #plt.yscale('log')
                #plt.title(f'Alm before and after filtering (beam size={beam}`)')
                #plt.legend()
                #plt.savefig(f'./Plots/alm_plot_wfilter_{beam}beam.png', dpi=1200)
                #plt.clf()
                
                ###########
                
                #UNCOMMENT
                
                #plt.scatter(ell[np.arange(lmax)], np.abs(highpass_alm[np.arange(lmax)])/np.abs(alm[np.arange(lmax)]), label='Highpass/Original alm', color='tab:orange')
                #plt.scatter(ell[np.arange(lmax)], np.abs(lowpass_alm[np.arange(lmax)])/np.abs(alm[np.arange(lmax)]), label='Lowpass/Original alm', color='tab:green')
                #plt.xlabel(r'$\ell$')
                #plt.title('Alm after filtering/original Alm (m=0)')
                #plt.legend()
                #plt.savefig('./Plots/alm_ratio_plot.png', dpi=1200)
                #plt.clf()
                
                #large_scale_map = hp.alm2map(lowpass_alm, nside, lmax=lmax)
                large_scale_map_rotated = hp.alm2map(lowpass_alm_rotated, nside, lmax=lmax)
                #print('large_scale_map')
                #print(large_scale_map)
                #UNCOMMENT
                
                #hp.mollview(large_scale_map, title=f"Large scale CMB temperature map (sim={simname2})", cmap="jet")#, min=-1.5e-4, max=1.5e-4)
                #hp.graticule()
                #plt.savefig(f'./Plots/T_ps_map_large_scale_{simname2}_{survey[iz]}_{mstar_bins_name[im]}.png', dpi=1200)
                #plt.clf()
                
                #small_scale_map = hp.alm2map(highpass_alm, nside, lmax=lmax)
                small_scale_map_rotated = hp.alm2map(highpass_alm_rotated, nside, lmax=lmax)
                #print('small_scale_map')
                #print(small_scale_map)
                #UNCOMMENT
                
                #hp.mollview(small_scale_map, title=f"Small scale CMB temperature map (sim={simname2})", cmap="jet")#, min=-1e-6, max=1e-6)
                #hp.graticule()
                #plt.savefig(f'./Plots/T_ps_map_small_scale_{simname2}_{survey[iz]}_{mstar_bins_name[im]}.png', dpi=1200)
                #plt.clf()
                
                large_scale_map_noise += large_scale_map_rotated
                small_scale_map_noise += small_scale_map_rotated

            large_scale_map = large_scale_map_noise / rotations
            small_scale_map = small_scale_map_noise / rotations'''
            
            
            lowpass_alm = hp.almxfl(alm.copy(), lowpass_values)
            highpass_alm = hp.almxfl(alm.copy(), highpass_values)
            large_scale_map = hp.alm2map(lowpass_alm, nside, lmax=lmax)
            small_scale_map = hp.alm2map(highpass_alm, nside, lmax=lmax)

            hp.mollview(large_scale_map, title=f"Large scale CMB temperature map (sim={simname2})", cmap="jet")#, min=-1.5e-4, max=1.5e-4)
            hp.graticule()
            plt.savefig(f'./Plots/T_ps_map_large_scale_{simname2}_{survey[iz]}_{mstar_bins_name[im]}.png', dpi=1200)
            plt.clf()

            hp.mollview(small_scale_map, title=f"Small scale CMB temperature map (sim={simname2})", cmap="jet")#, min=-1e-6, max=1e-6)
            hp.graticule()
            plt.savefig(f'./Plots/T_ps_map_small_scale_{simname2}_{survey[iz]}_{mstar_bins_name[im]}.png', dpi=1200)
            plt.clf()
            
            mean_mod_T_large_scale = np.mean(np.abs(large_scale_map))
            print('mean_mod_T_large_scale')
            print(mean_mod_T_large_scale)
            print(f'Generating, rotating and filtering alms: {time.time() - job_start_time}s')
                
            # convert halo 3D light cone position to latitude, longitude on the Healpix map
            rows, cols = (nhalo, 3)
            vec = [[0]*cols]*rows
            vec=1.0*np.asarray(vec)
            vec[:,0]=x
            vec[:,1]=y
            vec[:,2]=z
            theta, phi = hp.pixelfunc.vec2ang(vec,lonlat=True)
            source_vector = hp.ang2vec(theta, phi,lonlat=True)
                
            batch_size = max(1, nhalo // (ncpu * 2))

            print(f'Starting profile loop: {time.time() - job_start_time}s')

            # loop over haloes, creating delta sigma profiles as we go
            data = Parallel(n_jobs=ncpu, backend="loky", batch_size=batch_size)(delayed(tau_prof)(i) for i in range(nhalo))
            print(f'Ending profile loop: {time.time() - job_start_time}s')
            #data=tau_prof(0)
            data_1D, data_2D = zip(*data)
            data_1D = np.asarray(data_1D)
            data_2D = np.asarray(data_1D)
            print(data_1D.shape)
            print(data_2D.shape)
            
            tau_1D_stack = np.zeros(ntheta_d)
            #tau_2D_stack = np.zeros(data_2D.shape)
            
            #tau_stack=0
            for i in range(nhalo):
                tau_1D = data_1D[i,:]
                #tau_stack+=tau_kSZ*vr[i]
                tau_1D_stack += tau_1D
                
            tau_1D_stack/=nhalo
         
            #tau_stack = data/nhalo

            #tau_stack*=-np.std(vr)/np.sum(vr**2.0)
            print(simname,tau_1D_stack)

            rows, cols = (ntheta_d, 4)
            data = [0]*cols
            data[0]=theta_d
            data[1]=tau_1D_stack
            data[2]=(theta_d*np.pi/(180.0*60.0))*Dcom
            data[3]=nhalo

            outfile=r'./L1000N1800/'+survey[iz]+'/lensed_z3_'+simname +'_tau_Mstar_bin'+str(im)+'_nside'+str(nside)+'.pickle'
            os.makedirs(os.path.dirname(outfile), exist_ok=True)
                
            file = open(outfile,'wb')
            pickle.dump(data,file)
            file.close()
            print(f'Writing out data: {time.time() - job_start_time}s')
            quit()
