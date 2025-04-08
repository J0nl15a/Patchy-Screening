import numpy as np
import pylab as pb
import pandas as pd
import pickle
import os
import glob
import re
from matplotlib.ticker import ScalarFormatter
from matplotlib.colors import LinearSegmentedColormap, to_rgb
import matplotlib.colors as mcolors

infile = glob.glob('./L1000N1800/*/H*_tau_Mstar_bin7_nside8*_F*_u*.pickle')
#infile = glob.glob('./L1000N1800/Red/H*_tau_Mstar_bin*_nside8*_F*_u*.pickle')
pattern = re.compile(r'_bin(\d+)_')
infile.sort(key=lambda fn: int(pattern.search(fn).group(1)))
print(infile)

fig, ax1 = pb.subplots(constrained_layout=True, dpi=1200)
ax2 = ax1.twiny()
ax1.hlines(y=0, xmin=-1, xmax=12, linestyles='-', color='k', label=None)

for i,f in enumerate(infile):

    alpha = [.9,.8,.7,.6,.5,.4,.3,.2]
    
    direc, name = os.path.split(f)
    print(direc)
    print(name)
    direc_parts = direc.split(os.sep)
    print(direc_parts)

    components = name.split('_')
    print(components)
    #simname = components
    stellar_bins = {'bin0': ('10p9', 10.9), 'bin1': ('11p0', 11.0), 'bin2': ('11p1', 11.1), 'bin3': ('11p2', 11.2), 'bin4': ('11p3', 11.3), 'bin5': ('11p4', 11.4), 'bin6': ('11p5', 11.5), 'bin7': ('11p6', 11.6)}
    colours = {'Blue':'tab:blue', 'Green':'tab:green', 'Red':'tab:red'}

    base_color = mcolors.to_rgb(colours[direc_parts[2]])
    cmap = LinearSegmentedColormap.from_list("my_colour", [(1,1,1), base_color], N=256)
    colour_shade = cmap(alpha[i])
    
    data = pd.read_pickle(f)
    print(data)
    print(type(data))
    print(len(data))

    for j in range(len(data)):
        print(data[j])

    ax1.plot(data[0], data[1], label=f'{direc_parts[2]}', color=colours[direc_parts[2]])
    #ax1.plot(data[0], data[1], label=f'{stellar_bins[components[4]][1]}', color=colour_shade)

    '''blue_noise_data = pd.read_pickle('./L1000N1800/Blue/noise_files/blue_noise_data.pickle')
    green_noise_data = pd.read_pickle('./L1000N1800/Green/noise_files/green_noise_data.pickle')
    red_noise_data = pd.read_pickle('./L1000N1800/Red/noise_files/red_noise_data.pickle')
    if direc_parts[2]=='Blue':
        ax1.plot(data[0], blue_noise_data['mean'], color=colours[direc_parts[2]], linestyle='dashed')
        ax1.fill_between(data[0], blue_noise_data['mean']-blue_noise_data['std'], blue_noise_data['mean']+blue_noise_data['std'], color=colours[direc_parts[2]], alpha=.3, linewidth=0)
    elif direc_parts[2]=='Green':
        ax1.plot(data[0], green_noise_data['mean'], color=colours[direc_parts[2]], linestyle='dashed')
        ax1.fill_between(data[0], green_noise_data['mean']-green_noise_data['std'], green_noise_data['mean']+green_noise_data['std'], color=colours[direc_parts[2]], alpha=.3, linewidth=0)
    elif direc_parts[2]=='Red':
        ax1.plot(data[0], red_noise_data['mean'], label='Noise data', color=colours[direc_parts[2]], linestyle='dashed')
        ax1.fill_between(data[0], red_noise_data['mean']-red_noise_data['std'], red_noise_data['mean']+red_noise_data['std'], color=colours[direc_parts[2]], alpha=.3, linewidth=0)'''

formatter = ScalarFormatter(useMathText=True)
formatter.set_powerlimits((-4, -4))
ax1.yaxis.set_major_formatter(formatter)
ax1.ticklabel_format(axis='y', style='sci', scilimits=(-4, -4))

ax1.set_xlabel('Theta [degrees]')
ax1.set_ylabel('Tau')
ax1.set_title(f'Patchy Screening tau plot (Sim={direc_parts[1]} {components[0]}_{components[1]}, log$M_*$={stellar_bins[components[4]][1]})', wrap=True)
#ax1.set_title(f'Patchy Screening tau plot (Sim={direc_parts[1]} {components[4]}_{components[5]}, log$M_*$={stellar_bins[components[8]][1]}, {components[0]} to {components[1]} w/o Patchy Screening)', wrap=True)
#ax1.set_title(f'Patchy Screening tau plot (Sim={direc_parts[1]} {components[0]}_{components[1]}, sample={direc_parts[2]})', wrap=True)
ax1.set_xlim(left=0, right=11)
ax1.legend(fontsize=10)

ax2.plot(data[2], data[1], alpha=0)
ax2.set_xlabel('r [Mpc/h]')

#ax2.set_xticks(ax1.get_xticks())
#ax2.set_xticklabels(data[2])

#pb.savefig(f'./Plots/tau_plot_test_stellar_mass_{stellar_bins[components[4]][0]}_subtracted.png', dpi=1200)
pb.savefig(f'./Plots/tau_plot_test_stellar_mass_{stellar_bins[components[4]][0]}.png', dpi=1200)
#pb.savefig(f'./Plots/tau_plot_test_sample_{direc_parts[2]}.png', dpi=1200)
#pb.savefig(f'./Plots/tau_plot_test_sample_{direc_parts[2]}_subtracted.png', dpi=1200)
#pb.savefig(f'./Plots/tau_plot_test_unl_blue.png', dpi=1200)
#pb.savefig(f'./Plots/tau_plot_test_stellar_mass_{stellar_bins[components[8]][0]}_lensed_z3_no_ps.png', dpi=1200)
