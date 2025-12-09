import numpy as np
import h5py
import healpy
import glob
#import camb
#import os
import astropy.units as u
from astropy.cosmology import FlatLambdaCDM
from astropy.cosmology import z_at_value
import astropy.constants as const
import sys

def get_num(x):
    return int(x.split('/')[-2].lstrip().split('_')[-1])

def map_reading_kernel(file_dir, quantity, nside, ibox, nrot, theta, phi, dchi, z_mid, chi_mid, chi_CMB, cosmology, matter_mean, rotate=False):

    map_stacked = np.zeros((healpy.nside2npix(nside)))

    # prefactor = 3.0*cosmo.Om0*cosmo.H0*cosmo.H0/2.0/(const.c.to(u.km/u.s))**2.0
    prefactor = 3.0*cosmology[0]* cosmology[1]*cosmology[1]/2.0/(const.c.to(u.km/u.s))**2.0
    kernel_input = prefactor* (1+z_mid)*chi_mid*(1-chi_mid/chi_CMB)
    ##FLAMINGO S_8 tension paper Eq 5


    if rotate == True:

       chunck_num = np.hstack( (np.unique(ibox, return_index = True)[1], len(file_dir)) )
       count = 0


       for j in range(0, nrot):

           arg_min = int(chunck_num[j])
           arg_max = int(chunck_num[j+1])

           rot_custom=healpy.Rotator(rot=[theta[count], phi[count]], deg=True)
           map_chunck = np.zeros((healpy.nside2npix(nside)))
           print(j)
           print(theta[count])
           print(phi[count])

           for i in range(arg_min, arg_max):
               map_read = np.asarray(h5py.File(file_dir[i], 'r')[quantity])*1e10*u.solMass
               map_chunck += kernel_input[i]*dchi[i]*(map_read-matter_mean[i])/matter_mean[i]

           if j ==0:
              map_rot = map_chunck
           else:
              map_rot = rot_custom.rotate_map_alms(map_chunck)

           map_stacked += map_rot
           count+=1
           #map_write = healpy.write_map('....../' + box + sim_list +'lightcone'+str(lc_id)+'_shells/'+quantity+'_rot_'+str(j)+'.fits',map_stacked,overwrite=True) just in case you want to save kappa map per shell, feel free to add your own directory

    elif rotate == False:

           for i in range(0, len(file_dir)):
               map_read = np.asarray(h5py.File(file_dir[i], 'r')[quantity])*1e10*u.solMass
               map_save = kernel_input[i]*dchi[i]*(map_read-matter_mean[i])/matter_mean[i]
               #map_write = healpy.write_map('...../' + box + sim_list +'lightcone'+str(lc_id)+'_shells/kappa_per_shell/kappa_map_shell_'+str(i)+'_nonrot.fits',map_save,overwrite=True) just in case you want to save kappa map per shell, feel free to add your own directory
               map_stacked += map_save
               del map_save

    return map_stacked


def kappa_map_gen_forJonah(my_task_input):

    #box_id = my_task_input ## loop ober all resolution runs, default here: L1000N1800/
    box_id = int(1)

    #boxsize_id = my_task_input ##loop over all boxsize: 1000 or 2800, default here: 1000 Mpc
    boxsize_id = int(0)

    sim_id = my_task_input ##loop over all sim_list, feel free to simplify the simplist if you only care about FIDUCIAL and LS8
    #sim_id = int(0)

    #lc_id = my_task_input ##loop over all lightcones, 2 for 1Gpc, 8 for 2.8 Gpc , default here: lightcone0_shells 
    lc_id = int(0)

    base_dir = '/cosma8/data/dp004/flamingo/Runs/'
    box_list = ['L1000N0900', 'L1000N1800', 'L1000N3600', 'L2800N5040']
    box = box_list[box_id]
    sim_total = ['HYDRO_FIDUCIAL', 'HYDRO_ADIABATIC', 'HYDRO_JETS_published', 'HYDRO_LOW_SIGMA8', 'HYDRO_STRONG_AGN', 'HYDRO_STRONGEST_AGN', 'HYDRO_STRONG_SUPERNOVA', 'HYDRO_WEAK_AGN', 'HYDRO_PLANCK', 'HYDRO_LOW_SIGMA8_STRONGEST_AGN', 'HYDRO_STRONG_JETS_published']
    if isinstance(sim_id, int):
        sim_list = sim_total[sim_id]
    elif isinstance(sim_id, str):
        sim_list = sim_id
    parent_dir = f'{base_dir}{box}/{sim_list}/neutrino_corrected_maps_downsampled_4096/lightcone{str(lc_id)}_shells/'
    nside = 4096
    boxsize_list= [1000*u.Mpc, 2800*u.Mpc]
    boxsize = boxsize_list[boxsize_id]
    zcmb = 1100
    cosmo_info_file = f'{base_dir}{box}/{sim_list}/snapshots/flamingo_0009/flamingo_0009.0.hdf5'
    cosmo_info=h5py.File(cosmo_info_file,'r')
    H0=cosmo_info['Cosmology'].attrs['h']*100.0
    Om0=cosmo_info['Cosmology'].attrs['Omega_m']
    Ob0=cosmo_info['Cosmology'].attrs['Omega_b']
    Onu0=cosmo_info['Cosmology'].attrs['Omega_nu_0']
    m_nu=93.14*Onu0*(H0/100.0)**2.0 * u.eV
    T_nu=cosmo_info['Cosmology'].attrs['T_nu_0 [internal units]']
    T_cmb = T_nu*(4.0/11.0)**(-1.0/3.0)
    cosmo = FlatLambdaCDM(H0=H0[0], Om0=Om0[0], m_nu=m_nu[0], Ob0=Ob0[0], Tcmb0=T_cmb[0], Neff=1)
    chi_CMB =cosmo.comoving_distance(zcmb)
    print(cosmo)
    print(chi_CMB)

    fulldir_list = []
    for file in glob.glob(f'{parent_dir}shell_*/*.hdf5'):
        fulldir_list.append(file)

    lightcone_files = sorted(fulldir_list, key = get_num)
    print(len(lightcone_files))

    chi_min = np.zeros(len(lightcone_files))
    chi_max = np.zeros(len(lightcone_files))
    chi_mid = np.zeros(len(lightcone_files))
    for i in range(len(lightcone_files)):

        f = h5py.File(lightcone_files[i], 'r')
        chi_min[i]=f['Shell'].attrs['comoving_inner_radius']
        chi_max[i]=f['Shell'].attrs['comoving_outer_radius']
        chi_mid[i]=0.5*(chi_min[i]+chi_max[i])

    chi_min*=u.Mpc
    chi_max*=u.Mpc
    chi_mid*=u.Mpc
    dchi = chi_max - chi_min
    z_mid = z_at_value(cosmo.comoving_distance, chi_mid)
    box_index=np.floor(chi_max/boxsize)
    rot_times=int(np.max(box_index))+1
    print(np.max(np.asarray(z_mid)))

    ###for kappa
    rho_matter_mean = (cosmo.Odm0+cosmo.Ob0+cosmo.Onu0)*cosmo.critical_density(0.0).to(u.solMass/(u.Mpc)**3.0) #solar/Mpc^3
    area_per_pixel = healpy.nside2pixarea(nside,degrees=False)
    com_vol_per_pixel = (chi_max**3-chi_min**3)*area_per_pixel/3  #https://astronomy.stackexchange.com/questions/44380/comoving-volume-calculation, Mpc^3
    matter_mean = rho_matter_mean*com_vol_per_pixel ## solar mass


    np.random.seed(10)
    theta_rot =np.random.uniform(0.0,360.0,rot_times)
    phi_rot =np.random.uniform(-90.0,90.0,rot_times)
    print(theta_rot)
    print(phi_rot)

    kappa_stacked = map_reading_kernel(lightcone_files, 'TotalMass', nside, box_index, 
                                       rot_times, theta_rot, phi_rot, 
                                       dchi, z_mid, chi_mid, chi_CMB, 
                                       [cosmo.Om0, cosmo.H0], matter_mean, rotate=False) ##change rotate to True if you want box rotation
    try:
        kappa_map_write = healpy.write_map(f'/cosma8/data/dp004/dc-conl1/FLAMINGO/patchy_screening/data_files/kappa_maps/{box}_{sim_list}_kappa_nonrot.fits', kappa_stacked, overwrite=True) ##feel free to add your own directory
    except FileNotFoundError:
        kappa_map_write = healpy.write_map(f'/cosma8/data/dp004/dc-conl1/FLAMINGO/patchy_screening/data_files/kappa_maps/{box}_{sim_list}_kappa_nonrot.fits', kappa_stacked) ##feel free to add your own directory
    # del kappa_stacked
    return kappa_stacked

if __name__ == "__main__":

    my_task_input = int(sys.argv[1])
    num_tasks_input = int(sys.argv[2])
    
    kappa_map_gen_forJonah(my_task_input)