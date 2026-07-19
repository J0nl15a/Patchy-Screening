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
from pathlib import Path

def get_num(x):
    return int(x.split('/')[-2].lstrip().split('_')[-1])

def map_reading_kernel(file_dir, quantity, nside, ibox, nrot, theta, phi, dchi, z_mid, chi_mid, chi_CMB, cosmology, matter_mean, rotate=False, Jeger_rot=False):

    map_stacked = np.zeros((healpy.nside2npix(nside)))

    # prefactor = 3.0*cosmo.Om0*cosmo.H0*cosmo.H0/2.0/(const.c.to(u.km/u.s))**2.0
    prefactor = 3.0*cosmology[0]* cosmology[1]*cosmology[1]/2.0/(const.c.to(u.km/u.s))**2.0
    kernel_input = prefactor* (1+z_mid)*chi_mid*(1-chi_mid/chi_CMB)
    ##FLAMINGO S_8 tension paper Eq 5


    if rotate:
       if Jeger_rot:
            
            for i in range(len(file_dir)):
                print(i)
                map_chunck = np.zeros((healpy.nside2npix(nside)))
                rot_custom = healpy.Rotator(rot=[theta[i], phi[i]], inv=True)

                map_read = np.asarray(h5py.File(file_dir[i], 'r')[quantity])*1e10*u.solMass
                map_chunck += kernel_input[i]*dchi[i]*(map_read-matter_mean[i])/matter_mean[i]

                map_rot = rot_custom.rotate_map_alms(map_chunck, datapath='./data_files/healpy-data/')

                map_stacked += map_rot

    elif not Jeger_rot:

       chunck_num = np.hstack( (np.unique(ibox, return_index = True)[1], len(file_dir)) )
       count = 0


       for j in range(0, nrot):

           arg_min = int(chunck_num[j])
           arg_max = int(chunck_num[j+1])

           rot_custom=healpy.Rotator(rot=[theta[count], phi[count]], inv=True)
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
              map_rot = rot_custom.rotate_map_alms(map_chunck, datapath='./data_files/healpy-data/')

           map_stacked += map_rot
           count+=1
           #map_write = healpy.write_map('....../' + box + sim_list +'lightcone'+str(lc_id)+'_shells/'+quantity+'_rot_'+str(j)+'.fits',map_stacked,overwrite=True) just in case you want to save kappa map per shell, feel free to add your own directory

    elif not rotate:

           for i in range(0, len(file_dir)):
               map_read = np.asarray(h5py.File(file_dir[i], 'r')[quantity])*1e10*u.solMass
               map_save = kernel_input[i]*dchi[i]*(map_read-matter_mean[i])/matter_mean[i]
               #map_write = healpy.write_map('...../' + box + sim_list +'lightcone'+str(lc_id)+'_shells/kappa_per_shell/kappa_map_shell_'+str(i)+'_nonrot.fits',map_save,overwrite=True) just in case you want to save kappa map per shell, feel free to add your own directory
               map_stacked += map_save
               del map_save

    return map_stacked


def kappa_map_gen_forJonah(box_id, sim_id, lc_id=0, rotate=False):

    #box_id = my_task_input[0] ## loop ober all resolution runs, default here: L1000N1800/
    #box_id = int(1)

    #boxsize_id = my_task_input ##loop over all boxsize: 1000 or 2800, default here: 1000 Mpc
    #boxsize_id = int(0)

    #sim_id = my_task_input[1] ##loop over all sim_list, feel free to simplify the simplist if you only care about FIDUCIAL and LS8
    #sim_id = int(0)

    #lc_id = my_task_input ##loop over all lightcones, 2 for 1Gpc, 8 for 2.8 Gpc , default here: lightcone0_shells 
    #lc_id = int(0)

    base_dir = '/cosma8/data/dp004/flamingo/Runs/'
    box_list = ['L1000N0900', 'L1000N1800', 'L1000N3600', 'L2800N5040']
    if isinstance(box_id, int):
        box = box_list[box_id]
    elif isinstance(box_id, str):
        box = box_id
    if box == 'L2800N5040':
        snap_dir = 'snapshots_downsampled'
    else:
        snap_dir = 'snapshots'
    sim_total = ['HYDRO_FIDUCIAL', 'HYDRO_ADIABATIC', 'HYDRO_JETS_published', 'HYDRO_LOW_SIGMA8', 'HYDRO_STRONG_AGN', 'HYDRO_STRONGEST_AGN', 'HYDRO_STRONG_SUPERNOVA', 'HYDRO_WEAK_AGN', 'HYDRO_PLANCK', 'HYDRO_LOW_SIGMA8_STRONGEST_AGN', 'HYDRO_STRONG_JETS_published']
    if isinstance(sim_id, int):
        sim_list = sim_total[sim_id]
    elif isinstance(sim_id, str):
        sim_list = sim_id
    parent_dir = f'{base_dir}{box}/{sim_list}/neutrino_corrected_maps_downsampled_4096/lightcone{str(lc_id)}_shells/'
    nside = 4096
    boxsize_list= [1000*u.Mpc, 2800*u.Mpc]
    if box == 'L1000N0900' or box == 'L1000N1800' or box == 'L1000N3600':
        boxsize_id = int(0)
    elif box == 'L2800N5040':
        boxsize_id = int(1)
    boxsize = boxsize_list[boxsize_id]
    zcmb = 1100
    z_max = 3.0
    try:
        cosmo_info_file = f'{base_dir}{box}/{sim_list}/{snap_dir}/flamingo_0009/flamingo_0009.0.hdf5'
        cosmo_info=h5py.File(cosmo_info_file,'r')
    except FileNotFoundError:
        cosmo_info_file = f'{base_dir}{box}/{sim_list}/{snap_dir}/flamingo_0010/flamingo_0010.0.hdf5'
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
    chi_z_max = cosmo.comoving_distance(z_max)
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
        # chi_inner_shell=f['Shell'].attrs['comoving_inner_radius']
        # z_min = z_at_value(cosmo.comoving_distance, chi_inner_shell*u.Mpc)
        # print(i, z_min)

        chi_min[i]=f['Shell'].attrs['comoving_inner_radius']
        chi_max[i]=f['Shell'].attrs['comoving_outer_radius']
        chi_mid[i]=0.5*(chi_min[i]+chi_max[i])
        print(i, chi_z_max, chi_mid[i]*u.Mpc)

        if chi_mid[i]*u.Mpc > chi_z_max:
            print(f"Shell {i} has z_mid > 3.0, skipping.")
            lightcone_files = np.delete(lightcone_files, np.arange(i, len(lightcone_files)))
            chi_min = chi_min[:i] 
            chi_mid = chi_mid[:i] 
            chi_max = chi_max[:i]
            break

    print(lightcone_files)
    print(chi_mid[-1])

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
    # theta_rot =np.random.uniform(0.0,360.0,rot_times)
    # phi_rot =np.random.uniform(-90.0,90.0,rot_times)
    if box == 'L1000N1800' or box == 'L1000N3600':
        ## for 1Gpc
        angles = np.array([[0. , 0. , 3.26757547 , 3.26757547 , 3.26757547 , 1.51289711, 1.51289711, 3.13885639, 3.13885639 ,3.13885639, 2.17061318, 2.17061318, 2.17061318, 2.17061318, 4.59420579, 4.59420579,4.59420579, 1.14273623, 1.14273623, 1.14273623, 1.14273623, 2.02717201, 2.02717201, 2.02717201, 2.02717201, 2.77675054, 2.77675054, 2.77675054, 2.77675054, 2.77675054, 0.83245259, 0.83245259, 0.83245259, 0.83245259, 0.83245259, 0.83245259, 4.95779263, 4.95779263, 4.95779263, 4.95779263, 4.95779263, 4.95779263, 4.95779263,2.52359739, 2.52359739, 2.52359739, 2.52359739, 2.52359739, 2.52359739, 2.52359739,2.52359739, 2.69301628, 2.69301628, 2.69301628, 2.69301628, 2.69301628, 2.69301628, 2.69301628, 2.69301628, 2.69301628], [0. , 0. , 1.41518902, 1.41518902 , 1.41518902 , 0.80580058, 0.80580058, 0.71830831, 0.71830831, 0.71830831, 1.77536892, 1.77536892,1.77536892, 1.77536892, 0.62434822, 0.62434822,0.62434822, 2.14076603, 2.14076603, 2.14076603, 2.14076603, 0.49840908, 0.49840908, 0.49840908,0.49840908, 2.0136344 , 2.0136344, 2.0136344 , 2.0136344, 2.0136344, 2.25356928 , 2.25356928, 2.25356928 , 2.25356928 , 2.25356928, 2.25356928, 1.85187078, 1.85187078, 1.85187078, 1.85187078, 1.85187078, 1.85187078 , 1.85187078, 1.36014098, 1.36014098, 1.36014098, 1.36014098, 1.36014098, 1.36014098, 1.36014098, 1.36014098, 2.35895331, 2.35895331, 2.35895331, 2.35895331, 2.35895331, 2.35895331, 2.35895331, 2.35895331, 2.35895331]])

    elif box == 'L2800N5040':
        ##for 2.8 Gpc
        angles = np.array(([[0. , 0. , 0. , 0. , 0. , 0., 0. , 2.11833333, 2.11833333 , 2.11833333, 2.11833333 , 2.11833333, 2.11833333, 2.11833333 , 2.11833333 , 2.11833333 , 1.29070838, 1.29070838, 1.29070838, 1.29070838, 1.29070838, 1.29070838, 1.29070838, 1.29070838, 1.29070838, 1.29070838, 1.29070838, 5.69217656, 5.69217656, 5.69217656, 5.69217656, 5.69217656, 5.69217656, 5.69217656, 5.69217656, 5.69217656, 5.69217656,5.69217656, 5.69217656, 5.69217656, 5.69217656, 5.69217656, 5.69217656, 5.69217656, 3.79736641, 3.79736641, 3.79736641, 3.79736641, 3.79736641, 3.79736641, 3.79736641, 3.79736641, 3.79736641, 3.79736641, 3.79736641, 3.79736641, 3.79736641, 3.79736641, 3.79736641, 3.79736641, 3.79736641, 3.79736641, 1.32878635 , 1.32878635 , 1.32878635 , 1.32878635 , 1.32878635 , 1.32878635 ], [0. , 0. , 0. , 0. , 0. , 0., 0. , 0.96440001 , 0.96440001, 0.96440001 , 0.96440001 , 0.96440001,0.96440001 , 0.96440001 , 0.96440001 , 0.96440001 , 1.74841793, 1.74841793, 1.74841793, 1.74841793, 1.74841793, 1.74841793, 1.74841793, 1.74841793, 1.74841793, 1.74841793, 1.74841793, 0.56258515, 0.56258515, 0.56258515, 0.56258515, 0.56258515, 0.56258515, 0.56258515, 0.56258515, 0.56258515, 0.56258515, 0.56258515, 0.56258515, 0.56258515, 0.56258515, 0.56258515, 0.56258515, 0.56258515, 1.45462313, 1.45462313, 1.45462313, 1.45462313, 1.45462313, 1.45462313, 1.45462313, 1.45462313, 1.45462313, 1.45462313, 1.45462313, 1.45462313, 1.45462313, 1.45462313, 1.45462313, 1.45462313, 1.45462313, 1.45462313, 2.48706614, 2.48706614, 2.48706614, 2.48706614 , 2.48706614, 2.48706614]]))
    
    theta_rot = (angles[1, :]*(180.0/np.pi)*u.deg).to_value(u.deg)
    phi_rot = (angles[0, :]*(180.0/np.pi)*u.deg).to_value(u.deg)

    print(theta_rot)
    print(phi_rot)

    path = Path(f'/cosma8/data/dp004/dc-conl1/FLAMINGO/patchy_screening/data_files/kappa_maps/{box}/{sim_list}/lightcone{lc_id}/')
    path.mkdir(parents=True, exist_ok=True)

    kappa_stacked = map_reading_kernel(lightcone_files, 'TotalMass', nside, box_index, 
                                       rot_times, theta_rot, phi_rot, 
                                       dchi, z_mid, chi_mid, chi_CMB, 
                                       [cosmo.Om0, cosmo.H0], matter_mean, rotate=rotate, Jeger_rot=True) ##change rotate to True if you want box rotation
    if rotate:
        try:
            kappa_map_write = healpy.write_map(f'{path}/kappa_rot.fits', kappa_stacked, overwrite=True) ##feel free to add your own directory
        except FileNotFoundError:
            kappa_map_write = healpy.write_map(f'{path}/kappa_rot.fits', kappa_stacked) ##feel free to add your own directory
        # del kappa_stacked
    elif not rotate:
        try:
            kappa_map_write = healpy.write_map(f'{path}/kappa_nonrot.fits', kappa_stacked, overwrite=True) ##feel free to add your own directory
        except FileNotFoundError:
            kappa_map_write = healpy.write_map(f'{path}/kappa_nonrot.fits', kappa_stacked) ##feel free to add your own directory
        # del kappa_stacked
    return kappa_stacked

if __name__ == "__main__":

    #my_task_input = int(sys.argv[1])
    #num_tasks_input = int(sys.argv[2])
    box = str(sys.argv[1])
    isim = str(sys.argv[2])
    lc = int(sys.argv[3])

    kappa_map_gen_forJonah(box, isim, lc, rotate=True)
