import numpy as np
import pylab as pb
from scipy.interpolate import interp1d
from scipy.integrate import simpson
import textwrap

dndz_blue_match = np.loadtxt("/cosma8/data/dp004/dc-conl1/FLAMINGO/patchy_screening/unWISExLens_lklh/data/v1.0/aux_data/dndz/unWISE_blue_xmatch_dndz.txt", usecols=(0,1))
dndz_green_match = np.loadtxt("/cosma8/data/dp004/dc-conl1/FLAMINGO/patchy_screening/unWISExLens_lklh/data/v1.0/aux_data/dndz/unWISE_green_xmatch_dndz.txt", usecols=(0,1))
#dndz_blue_corr = np.loadtxt("/cosma8/data/dp004/dc-conl1/FLAMINGO/patchy_screening/unWISExLens_lklh/data/v1.0/aux_data/dndz/unWISE_blue_xcorr_bdndz.txt", usecols=(0,1))
#dndz_green_corr = np.loadtxt("/cosma8/data/dp004/dc-conl1/FLAMINGO/patchy_screening/unWISExLens_lklh/data/v1.0/aux_data/dndz/unWISE_green_xcorr_bdndz.txt", usecols=(0,1))

print(min(dndz_blue_match[:,0]), max(dndz_blue_match[:,0]))

FLAMINGO_halo_bins = np.loadtxt('/cosma8/data/dp004/dc-conl1/FLAMINGO/patchy_screening/data_files/FLAMINGO_halo_redshift_values.txt', usecols=2)
z_max_idx = np.where(FLAMINGO_halo_bins <= 3)
#FLAMINGO_z_bins = np.zeros((len(FLAMINGO_halo_bins)+2))
FLAMINGO_z_bins = np.zeros((len(z_max_idx[0])+2))
FLAMINGO_z_bins[1:-1] = FLAMINGO_halo_bins[z_max_idx[0]]
FLAMINGO_z_bins[-1] = 3.0 #4.0
print(FLAMINGO_z_bins)

mblue = interp1d(dndz_blue_match[:,0], dndz_blue_match[:,1], kind='cubic')
dndz_blue_match_interpolated = mblue(FLAMINGO_z_bins)
mgreen = interp1d(dndz_green_match[:,0], dndz_green_match[:,1], kind='cubic')
dndz_green_match_interpolated = mgreen(FLAMINGO_z_bins)

pb.plot(dndz_blue_match[:,0], dndz_blue_match[:,1], color='tab:blue', label='Blue sample')
pb.plot(dndz_green_match[:,0], dndz_green_match[:,1], color='tab:green', label='Green sample')
pb.plot(FLAMINGO_z_bins, dndz_blue_match_interpolated, color='tab:cyan', marker='.', label='FLAMINGO bins')
pb.plot(FLAMINGO_z_bins, dndz_green_match_interpolated, color='tab:olive', marker='.', label='FLAMINGO bins')
pb.xlim(left=0, right=4)
pb.xlabel('z')
pb.ylabel('dn/dz')
pb.title('unWISE galaxy redshift distribution cross-match')
pb.legend()
pb.savefig('./Plots/unWISE_dndz_match.png', dpi=1200)
pb.clf()

pb.plot(FLAMINGO_z_bins, dndz_blue_match_interpolated*100000, color='tab:cyan', marker='.', label='FLAMINGO bins')
pb.plot(FLAMINGO_z_bins, dndz_green_match_interpolated*100000, color='tab:olive', marker='.', label='FLAMINGO bins')
pb.xlim(left=0, right=3)
pb.xlabel('z')
pb.ylabel('dn/dz')
pb.title('unWISE galaxy redshift distribution cross-match (scaled to 100,000 galaxies)')
pb.legend()
pb.savefig('./Plots/unWISE_dndz_match_multiplied.png', dpi=1200)
pb.clf()

total_area_blue = simpson(y=dndz_blue_match_interpolated, x=FLAMINGO_z_bins)
total_area_green = simpson(y=dndz_green_match_interpolated, x=FLAMINGO_z_bins)
print(total_area_blue)
fractional_contribution_blue = np.zeros((len(FLAMINGO_z_bins)-1))
fractional_contribution_green = np.zeros((len(FLAMINGO_z_bins)-1))

for f in range(len(FLAMINGO_z_bins)-1):
    fraction_blue = simpson(y=[dndz_blue_match_interpolated[f], dndz_blue_match_interpolated[f+1]], x=[FLAMINGO_z_bins[f], FLAMINGO_z_bins[f+1]])
    fraction_green = simpson(y=[dndz_green_match_interpolated[f], dndz_green_match_interpolated[f+1]], x=[FLAMINGO_z_bins[f], FLAMINGO_z_bins[f+1]])
    print(fraction_blue)
    fractional_contribution_blue[f] = fraction_blue/total_area_blue
    fractional_contribution_green[f] = fraction_green/total_area_green
    print(fractional_contribution_blue[f])

pb.plot(FLAMINGO_z_bins[1:], fractional_contribution_blue, color='tab:brown', label='Blue')
pb.plot(FLAMINGO_z_bins[1:], fractional_contribution_green, linestyle='dotted', color='tab:brown', label='Green')
pb.xlim(left=0, right=3)
pb.xlabel('z')
pb.ylabel('Contribution')
pb.title("\n".join(textwrap.wrap('Fractional contribution of the area under the unWISE galaxy redshift distribution', width=75)))
pb.legend()
pb.tight_layout()
pb.savefig('./Plots/unWISE_dndz_match_fraction.png', dpi=1200)
pb.clf()

print(mblue(0.6)*100000*(0.05), mgreen(0.6)*100000*(0.05))
'''for i in range(1,len(FLAMINGO_z_bins)):
    print(FLAMINGO_z_bins[i], FLAMINGO_z_bins[i-1], mblue(((FLAMINGO_z_bins[i]-FLAMINGO_z_bins[i-1])/2)+FLAMINGO_z_bins[i-1])*100000*(FLAMINGO_z_bins[i]-FLAMINGO_z_bins[i-1]), mgreen(((FLAMINGO_z_bins[i]-FLAMINGO_z_bins[i-1])/2)+FLAMINGO_z_bins[i-1])*100000*(FLAMINGO_z_bins[i]-FLAMINGO_z_bins[i-1]))'''

print(round(mblue(((FLAMINGO_z_bins[1]-FLAMINGO_z_bins[0])/2)+FLAMINGO_z_bins[0])*100000*(FLAMINGO_z_bins[1]-FLAMINGO_z_bins[0])))

kusiak_nbar_blue = 3409 #From Kusiak paper
kusiak_nbar_green = 1846

kusiak_ntotal_blue = kusiak_nbar_blue * 41253 #4pi steradians
kusiak_ntotal_green = kusiak_nbar_green * 41253

nsamp = (kusiak_ntotal_blue, kusiak_ntotal_green) #8000000
running_total_blue = 0
running_total_green = 0
# Open both files simultaneously
with open('/cosma8/data/dp004/dc-conl1/FLAMINGO/patchy_screening/data_files/dndz_galaxies_sampled_Blue_ntotal.txt', 'w') as blue_out, open('/cosma8/data/dp004/dc-conl1/FLAMINGO/patchy_screening/data_files/dndz_galaxies_sampled_Green_ntotal.txt', 'w') as green_out:
    for i in range(1, len(FLAMINGO_z_bins)):
        # Calculate the midpoint between the current and previous bin
        mid_point = ((FLAMINGO_z_bins[i] - FLAMINGO_z_bins[i-1]) / 2) + FLAMINGO_z_bins[i-1]
        delta_z = FLAMINGO_z_bins[i] - FLAMINGO_z_bins[i-1]
        
        # Perform the computations for both samples
        blue_value = mblue(mid_point) * (nsamp[0]/total_area_blue) * delta_z
        green_value = mgreen(mid_point) * (nsamp[1]/total_area_green) * delta_z
        
        # Print the values to the screen for verification
        print(f"Blue: {FLAMINGO_z_bins[i-1]} {FLAMINGO_z_bins[i]} {blue_value} {round(blue_value)}")
        print(f"Green: {FLAMINGO_z_bins[i-1]} {FLAMINGO_z_bins[i]} {green_value} {round(green_value)}")

        if i==1:
            blue_out.write(f"#Total number of sampled galaxies: {nsamp[0]}\n")
            green_out.write(f"#Total number of sampled galaxies: {nsamp[1]}\n")
        
        # Write to both files in one iteration
        blue_out.write(f"{FLAMINGO_z_bins[i-1]} {FLAMINGO_z_bins[i]} {round(blue_value)}\n")
        green_out.write(f"{FLAMINGO_z_bins[i-1]} {FLAMINGO_z_bins[i]} {round(green_value)}\n")
        running_total_blue += round(blue_value)
        running_total_green += round(green_value)
        
print(running_total_blue, running_total_green)
pb.plot(FLAMINGO_z_bins, dndz_blue_match_interpolated*(nsamp[0]/total_area_blue), color='tab:blue', marker='.', label='Blue sample (FLAMINGO bins)')
pb.plot(FLAMINGO_z_bins, dndz_green_match_interpolated*(nsamp[1]/total_area_green), color='tab:green', marker='.', label='Green sample (FLAMINGO bins)')
pb.xlim(left=0, right=3)
pb.xlabel('z')
pb.ylabel('dn/dz')
pb.title('unWISE galaxy redshift distribution cross-match (rescaled)')
pb.legend()
pb.savefig('./Plots/unWISE_dndz_match_rescaled.png', dpi=1200)
pb.clf()
quit()
#pb.plot(dndz_blue_corr[:,0], dndz_blue_corr[:,1], linestyle='dashed', color='tab:blue', label='Blue sample')
#pb.plot(dndz_green_corr[:,0], dndz_green_corr[:,1], linestyle='dashed', color='tab:green', label='Green sample')
#pb.xlim(left=0, right=4)
#pb.xlabel('z')
#pb.ylabel('b(z) dn/dz')
#pb.title('unWISE galaxy redshift distribution cross-correlated')
#pb.legend()
#pb.savefig('./Plots/unWISE_dndz_corr.png', dpi=1200)
#pb.clf()

kusiak_nbar_blue = 3409 #From Kusiak paper
kusiak_nbar_green = 1846

kusiak_ntotal_blue = kusiak_nbar_blue * 41253 #4pi steradians
kusiak_ntotal_green = kusiak_nbar_green * 41253


print(kusiak_ntotal_blue, kusiak_ntotal_green)

dndz_blue_match_ntot = simpson(y=dndz_blue_match[:,1], x=dndz_blue_match[:,0]) #Area under dn/dz curve
dndz_green_match_ntot = simpson(y=dndz_green_match[:,1], x=dndz_green_match[:,0])

#dndz_blue_match_interp_ntot = simpson(y=dndz_blue_match_interpolated, x=FLAMINGO_z_bins)
#dndz_green_match_interp_ntot = simpson(y=dndz_green_match_interpolated, x=FLAMINGO_z_bins)

print(f"Blue/Green match Ntot: {dndz_blue_match_ntot}, {dndz_green_match_ntot}")
#print(f"Blue/Green match interp Ntot: {dndz_blue_match_interp_ntot}, {dndz_green_match_interp_ntot}")

print(f"Ratio (Blue, match): {kusiak_ntotal_blue/dndz_blue_match_ntot}")
print(f"Ratio (Green, match): {kusiak_ntotal_green/dndz_green_match_ntot}")

rescaled_dndz_blue_match = dndz_blue_match_interpolated*(kusiak_ntotal_blue)#/dndz_blue_match_ntot) #dn/dz multiplied by ratio of N_total: (Kusiak paper value/area under curve)
rescaled_dndz_green_match = dndz_green_match_interpolated*(kusiak_ntotal_green)#/dndz_green_match_ntot)

pb.plot(FLAMINGO_z_bins, rescaled_dndz_blue_match, color='tab:blue', marker='.', label='Blue sample (FLAMINGO bins)')
pb.plot(FLAMINGO_z_bins, rescaled_dndz_green_match, color='tab:green', marker='.', label='Green sample (FLAMINGO bins)')
pb.xlim(left=0, right=3)
pb.xlabel('z')
pb.ylabel('dn/dz')
pb.title('unWISE galaxy redshift distribution cross-match (rescaled)')
pb.legend()
pb.savefig('./Plots/unWISE_dndz_match_rescaled.png', dpi=1200)
pb.clf()

#print(kusiak_ntotal_blue - simpson(y=rescaled_dndz_blue_match, x=FLAMINGO_z_bins))
#print(kusiak_ntotal_green - simpson(y=rescaled_dndz_green_match, x=FLAMINGO_z_bins))

hblue = interp1d(FLAMINGO_z_bins, rescaled_dndz_blue_match, kind='cubic')
hgreen = interp1d(FLAMINGO_z_bins, rescaled_dndz_green_match, kind='cubic')

print(f"Number of galaxies needed for Blue sample = {hblue([0.6])*(FLAMINGO_z_bins[13]-FLAMINGO_z_bins[12])}") #Printing number of galaxies that need sampling
print(f"Number of galaxies needed for Green sample = {hgreen([1.1])*(FLAMINGO_z_bins[23]-FLAMINGO_z_bins[22])}")

