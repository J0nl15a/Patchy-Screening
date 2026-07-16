import numpy as np, pylab as pb
import sys

box = str(sys.argv[1])
isim = str(sys.argv[2])
iz = str(sys.argv[3])
lc = str(sys.argv[4])

min_amplitude = 10.3
max_amplitude = 11.3
amplitude_step = 0.1
min_slope = 0.0
slope_step = 0.1

if iz == 'Blue':
    max_slope = 2.0
elif iz == 'Green':
    max_slope = 1.0

cut_amplitude = np.repeat(np.arange(min_amplitude, max_amplitude + amplitude_step, amplitude_step).reshape(-1,1), int((max_slope - min_slope) / slope_step) + 1, axis=0)
cut_slope = np.tile(np.arange(min_slope, max_slope + slope_step, slope_step), int((max_amplitude - min_amplitude) / amplitude_step) + 1).reshape(-1,1)
print(cut_amplitude.shape, cut_slope.shape)
cut_values = np.column_stack((np.round(cut_amplitude, 1), np.round(cut_slope, 1))) #Amplitude and slope parameters

abundance_cut = 0.5
if iz == 'Blue':
    kusiak_observed_nbar_per_sq_deg = 3409
elif iz == 'Green':
    kusiak_observed_nbar_per_sq_deg = 1846
kusiak_observed_nbar_full_sky = kusiak_observed_nbar_per_sq_deg * 41253  # total sq deg in sky
nhalo_lower_bound = kusiak_observed_nbar_full_sky * abundance_cut  # 50% abundance cut

nhalos = []
low_nhalos = []
negative_power = []
for i, (im, slope) in enumerate(cut_values):
        im_name = f"{float(im):.1f}".replace('.', 'p')
        slope_name = f"{float(slope):.1f}".replace('.', 'p')
        with open(f"./data_files/dndz_samples/{box}/{isim}/{iz}/lightcone{lc}/dndz_galaxies_sampled_{im_name}_{slope_name}.txt", "r") as f:
            first_line = f.readline().strip()
            number = int(first_line.split(":")[-1])
            nhalos.append(number)
            if number < nhalo_lower_bound:
                low_nhalos.append(i)
        auto = np.loadtxt(f"./data_files/power_spectra/galaxy_galaxy/{box}/{isim}/{iz}/lightcone{lc}/galaxy_galaxy_power_spectrum_{im_name}_{slope_name}.txt", skiprows=1, usecols=2)
        cross = np.loadtxt(f"./data_files/power_spectra/kappa_galaxy/{box}/{isim}/{iz}/lightcone{lc}/kappa_galaxy_power_spectrum_{im_name}_{slope_name}.txt", skiprows=1, usecols=1)
        negative_mask_cross = np.where(cross < 0)[0]
        negative_mask_auto = np.where(auto < 0)[0]
        if len(negative_mask_auto) > 0 or len(negative_mask_cross) > 0:
             negative_power.append(i)
             
print(low_nhalos, negative_power)
cut_values_low_nhalo = np.delete(cut_values.copy(), low_nhalos, 0)
cut_values_negative_power = np.delete(cut_values.copy(), negative_power, 0)
print(cut_values_low_nhalo)
print(cut_values_low_nhalo.shape)
print(cut_values_negative_power.shape)
# quit()

unique_first_col_low_nhalos = np.unique(cut_values_low_nhalo[:,0])
unique_first_col_negative_power = np.unique(cut_values_negative_power[:,0])

low_nhalos_x = []
low_nhalos_y = []
negative_power_x = []
negative_power_y = []

for val in unique_first_col_low_nhalos:
    subset = cut_values_low_nhalo[cut_values_low_nhalo[:, 0] == val]
    low_nhalos_x.append(val)
    low_nhalos_y.append(np.max(subset[:, 1]))
    print(val)
    print(subset)
    print(low_nhalos_x)
    print(low_nhalos_y)

for val in unique_first_col_negative_power:
    subset = cut_values_negative_power[cut_values_negative_power[:, 0] == val]
    negative_power_x.append(val)
    negative_power_y.append(np.max(subset[:, 1]))

low_nhalos_values = np.column_stack((low_nhalos_x, low_nhalos_y))
print(low_nhalos_values)
try:
    plateau_point = min(np.where(low_nhalos_values[:,1] != max(low_nhalos_y))[0]-1)
except ValueError:
    plateau_point = max(cut_amplitude)
print(plateau_point)

print(low_nhalos_y)
print(len(low_nhalos_y))

amplitude_limit = len(low_nhalos_y) - 1
vertical_limit = low_nhalos_x[amplitude_limit] #+ 0.1

def diagonal_limit():
    c = 10.3
    mask = np.array([True]*len(low_nhalos_x))
    while True in mask:
        y = -1 * np.array((low_nhalos_x)) + c
        c += 0.1
        limit_points = np.round(np.column_stack((low_nhalos_x, y)), 1)
        low_set = set(map(tuple, cut_values_low_nhalo))
        mask = np.array([tuple(np.round(pt, 1)) in low_set for pt in limit_points])
        print(mask)
        continue
    c -= 0.2
    y -= 0.1
    return np.round(y, 3), np.round(c, 1)

y, c = diagonal_limit()
y += 0.1
c += 0.1

try:
    slope_limit = np.where(y == max_slope)[0][-1]
except IndexError:
    slope_limit = 0
plateau_point = low_nhalos_x[slope_limit]

print(plateau_point, c, vertical_limit)
# quit()

# z = np.polyfit(low_nhalos_x[plateau_point:], low_nhalos_y[plateau_point:], deg=1)
# print(z)

# m = (low_nhalos_y[-1] - low_nhalos_y[plateau_point])/(low_nhalos_x[-1] - low_nhalos_x[plateau_point])
# c = -1 * m * low_nhalos_x[plateau_point] + 1
# y = -1 * np.array((low_nhalos_x[plateau_point:])) + ((1 * 10.3) + 2.0)
# print(m)
# print(c)

# pb.plot(low_nhalos_x, low_nhalos_y, label='Number of halos > 50% of observed')
# pb.scatter(cut_values_negative_power[:,0], cut_values_negative_power[:,1], c='tab:orange', label='No negative power in spectra')
# # pb.plot(low_nhalos_x[plateau_point:], z[0]*np.array((low_nhalos_x[plateau_point:]))+z[1], color='tab:red', label=f'np.polyfit (m={z[0]:.3f}, c={z[1]:.3f})')
# # pb.plot(low_nhalos_x[plateau_point:], z[0]*np.array((low_nhalos_x[plateau_point:]))+round(z[1], 2), color='tab:green', label=f'adjusted np.polyfit (m={z[0]:.3f}, c={z[1]:.2f})')
# pb.plot(low_nhalos_x[plateau_point:], m * np.array((low_nhalos_x[plateau_point:]))+c, color='tab:purple', label=f'Manual calcualtion (m={m:.3f}, c={c:.3f})')
# pb.fill_betweenx(y=[-0.1, 1.1], x1=low_nhalos_x[-1], x2=cut_values[-1,0]+0.01, color='grey', alpha=0.5, label='Excluded from training', linewidth=0)
# pb.xlabel('Amplitude')
# pb.ylabel('Slope')
# pb.title(f'Models used in the emulator training data (sim={isim}, sample={iz})', wrap=True)
# pb.legend(loc='lower left', fontsize=8)
# # pb.legend(loc='upper right', fontsize=8)
# pb.savefig("./Plots/amp_v_slope.png", dpi=400)
# pb.clf()

print(low_nhalos_x)
print(low_nhalos_y)

sims = ['HYDRO_LOW_SIGMA8_STRONGEST_AGN', 'HYDRO_LOW_SIGMA8', 'HYDRO_PLANCK_LARGE_NU_FIXED', 'HYDRO_PLANCK_LARGE_NU_VARY', 'HYDRO_PLANCK',
        'HYDRO_STRONG_JETS_published', 'HYDRO_JETS_published', 'HYDRO_STRONG_SUPERNOVA', 
        'HYDRO_STRONGEST_AGN', 'HYDRO_STRONGER_AGN', 'HYDRO_STRONG_AGN', 'HYDRO_WEAK_AGN',
        'HYDRO_FIDUCIAL']
sim_names = ['LS8_fgas-8$\sigma$', 'LS8', 'PlanckNu0p24Fix', 'PlanckNu0p24Var', 'Planck',
            'Jet_fgas-4$\sigma$', 'Jet', '$M^*$-$\sigma$',
            'fgas-8$\sigma$', 'fgas-4$\sigma$', 'fgas-2$\sigma$', 'fgas+2$\sigma$',
            'L1_m9']
FLAMINGO_colors_sorted = ['#7B68EE', '#882255', '#999933', '#AA4499', '#44AA99',
                          '#55E18E', '#7EFF4B', '#FF8C40',
                          '#105ba4', '#3787c0', '#6aaed6', '#abd0e6',
                          '#117733']
sims = sims[::-1]
sim_names = sim_names[::-1]
FLAMINGO_colors_sorted = FLAMINGO_colors_sorted[::-1]
sim_parameters = []
sim_errors = []
lower_errors = []
upper_errors = []
for i, s in enumerate(sims):
    amp = np.loadtxt(f"./data_files/mle_parameters/{box}/{s}/{iz}/lightcone{lc}/mle_values.txt", usecols=1, skiprows=6, max_rows=1, delimiter='=')
    slope = np.loadtxt(f"./data_files/mle_parameters/{box}/{s}/{iz}/lightcone{lc}/mle_values.txt", usecols=1, skiprows=7, max_rows=1, delimiter='=')
    amp_lower = np.loadtxt(f"./data_files/mle_parameters/{box}/{s}/{iz}/lightcone{lc}/mle_values.txt", usecols=1, skiprows=8, max_rows=1, delimiter='=')
    amp_upper = np.loadtxt(f"./data_files/mle_parameters/{box}/{s}/{iz}/lightcone{lc}/mle_values.txt", usecols=1, skiprows=9, max_rows=1, delimiter='=')
    slope_lower = np.loadtxt(f"./data_files/mle_parameters/{box}/{s}/{iz}/lightcone{lc}/mle_values.txt", usecols=1, skiprows=10, max_rows=1, delimiter='=')
    slope_upper = np.loadtxt(f"./data_files/mle_parameters/{box}/{s}/{iz}/lightcone{lc}/mle_values.txt", usecols=1, skiprows=11, max_rows=1, delimiter='=')
    sim_parameters.append((amp, slope))
    lower_errors.append((amp_lower, slope_lower))
    upper_errors.append((amp_upper, slope_upper))
    # sim_errors.append(((amp_lower, slope_lower), (amp_upper, slope_upper)))
    # sim_errors.append((np.array((manual_errors[i][0][0], manual_errors[i][0][1])).reshape(2,-1), np.array((manual_errors[i][1][0], manual_errors[i][1][1])).reshape(2,-1)))
# sim_errors = np.array(sim_errors).reshape(2, -1)

res = ['L2800N5040',
       'L1000N3600',
       'L1000N1800']
sim = 'HYDRO_FIDUCIAL'
res_names = ['L2p8_m9',
             'L1_m8',
             'L1_m9']
FLAMINGO_colors_sorted_res = ["#332288",
                              "#CC6677",
                              '#117733']
res = res[::-1]
res_names = res_names[::-1]
FLAMINGO_colors_sorted_res = FLAMINGO_colors_sorted_res[::-1]
res_parameters = []
res_errors = []
lower_errors_res = []
upper_errors_res = []
for i, r in enumerate(res):
    if r == 'L2800N5040':
        lc_2p8 = 0
        while lc_2p8 != 8:
            amp = np.loadtxt(f"./data_files/mle_parameters/{r}/{sim}/{iz}/lightcone{lc_2p8}/mle_values.txt", usecols=1, skiprows=6, max_rows=1, delimiter='=')
            slope = np.loadtxt(f"./data_files/mle_parameters/{r}/{sim}/{iz}/lightcone{lc_2p8}/mle_values.txt", usecols=1, skiprows=7, max_rows=1, delimiter='=')
            amp_lower = np.loadtxt(f"./data_files/mle_parameters/{r}/{sim}/{iz}/lightcone{lc_2p8}/mle_values.txt", usecols=1, skiprows=8, max_rows=1, delimiter='=')
            amp_upper = np.loadtxt(f"./data_files/mle_parameters/{r}/{sim}/{iz}/lightcone{lc_2p8}/mle_values.txt", usecols=1, skiprows=9, max_rows=1, delimiter='=')
            slope_lower = np.loadtxt(f"./data_files/mle_parameters/{r}/{sim}/{iz}/lightcone{lc_2p8}/mle_values.txt", usecols=1, skiprows=10, max_rows=1, delimiter='=')
            slope_upper = np.loadtxt(f"./data_files/mle_parameters/{r}/{sim}/{iz}/lightcone{lc_2p8}/mle_values.txt", usecols=1, skiprows=11, max_rows=1, delimiter='=')
            res_parameters.append((amp, slope))
            lower_errors_res.append((amp_lower, slope_lower))
            upper_errors_res.append((amp_upper, slope_upper))
            lc_2p8 += 1
    else:
        amp = np.loadtxt(f"./data_files/mle_parameters/{r}/{sim}/{iz}/lightcone{lc}/mle_values.txt", usecols=1, skiprows=6, max_rows=1, delimiter='=')
        slope = np.loadtxt(f"./data_files/mle_parameters/{r}/{sim}/{iz}/lightcone{lc}/mle_values.txt", usecols=1, skiprows=7, max_rows=1, delimiter='=')
        amp_lower = np.loadtxt(f"./data_files/mle_parameters/{r}/{sim}/{iz}/lightcone{lc}/mle_values.txt", usecols=1, skiprows=8, max_rows=1, delimiter='=')
        amp_upper = np.loadtxt(f"./data_files/mle_parameters/{r}/{sim}/{iz}/lightcone{lc}/mle_values.txt", usecols=1, skiprows=9, max_rows=1, delimiter='=')
        slope_lower = np.loadtxt(f"./data_files/mle_parameters/{r}/{sim}/{iz}/lightcone{lc}/mle_values.txt", usecols=1, skiprows=10, max_rows=1, delimiter='=')
        slope_upper = np.loadtxt(f"./data_files/mle_parameters/{r}/{sim}/{iz}/lightcone{lc}/mle_values.txt", usecols=1, skiprows=11, max_rows=1, delimiter='=')
        res_parameters.append((amp, slope))
        lower_errors_res.append((amp_lower, slope_lower))
        upper_errors_res.append((amp_upper, slope_upper))

# xerr = np.vstack((amp_err_lower, amp_err_upper))  # shape (2, N)
# yerr = np.vstack((slope_err_lower, slope_err_upper))

mle_amp = np.loadtxt(f"./data_files/mle_parameters/{box}/{isim}/{iz}/lightcone{lc}/mle_values.txt", usecols=1, skiprows=6, max_rows=1, delimiter='=')
mle_slope = np.loadtxt(f"./data_files/mle_parameters/{box}/{isim}/{iz}/lightcone{lc}/mle_values.txt", usecols=1, skiprows=7, max_rows=1, delimiter='=')
mle_amp_name = f"{float(mle_amp):.3f}".replace('.', 'p')
mle_slope_name = f"{float(mle_slope):.3f}".replace('.', 'p')
with open(f"./data_files/dndz_samples/{box}/{isim}/{iz}/lightcone{lc}/dndz_galaxies_sampled_{mle_amp_name}_{mle_slope_name}.txt", "r") as f:
    first_line = f.readline().strip()
    mle_nhalos = int(first_line.split(":")[-1])
print(mle_amp, mle_slope, mle_nhalos)

pb.scatter(cut_values_low_nhalo[:,0], cut_values_low_nhalo[:,1], c='tab:green', label='Trained nodes')
pb.scatter(cut_values[:,0][low_nhalos], cut_values[:,1][low_nhalos], c='tab:red', label='Not used in training')
pb.plot(low_nhalos_x[slope_limit:], y[slope_limit:], color='tab:purple', label=f'Manual calcualtion (m={-1:.3f}, c={c:.3f})')
pb.hlines(y=low_nhalos_y[0], xmin=low_nhalos_x[0], xmax=low_nhalos_x[slope_limit], colors='tab:purple')
pb.vlines(x=vertical_limit, ymin=0.0, ymax=y[-1], colors='tab:purple')
pb.scatter(mle_amp, mle_slope, c='black', marker='*', s=50, label=f'MLE from MCMC (amp={mle_amp:.3f}, slope={mle_slope:.3f})')
pb.xlabel('Amplitude')
pb.ylabel('Slope')
pb.title(f'Models used in the emulator training data (sim={isim}, sample={iz})', wrap=True)
pb.legend(loc='lower left', fontsize=8)
pb.savefig(f"./Plots/amp_v_slope_{isim}_{iz}.png", dpi=400)
pb.clf()


slope_ticks = np.arange(0.0, 1.1, 0.1)
amp_ticks = slope_ticks + 10.3
print(sim_errors)
fig = pb.figure()
for s in range(len(sims)):
    pb.errorbar(sim_parameters[s][0], sim_parameters[s][1], 
                xerr=np.array([[lower_errors[s][0]], [upper_errors[s][0]]]), 
                yerr=np.array([[lower_errors[s][1]], [upper_errors[s][1]]]), 
                fmt='*', capsize=3, color=FLAMINGO_colors_sorted[s])
    pb.scatter(sim_parameters[s][0], sim_parameters[s][1], marker='*', s=50, color=FLAMINGO_colors_sorted[s], 
            #    label=f'{sims[s]}: amp=${sim_parameters[s][0]:.3f}_{{-{lower_errors[s][0]}}}^{{+{upper_errors[s][0]}}}$, slope=${sim_parameters[s][1]:.3f}_{{-{lower_errors[s][1]}}}^{{+{upper_errors[s][1]}}}$')
                label=f'{sim_names[s]}')

pb.xlabel('Amplitude')
pb.ylabel('Slope')
# pb.xticks(amp_ticks)
# pb.yticks(slope_ticks)
pb.grid(which='major', linestyle='--', linewidth=0.5, alpha=0.7)
# pb.title(f'MLE parameters (sample={iz})', wrap=True)
pb.xlim(10.3, 11.3)
pb.ylim(0.0, 1.0)
# fig.legend(loc='outside right upper', fontsize=4)
pb.legend(loc='upper right', fontsize=6)
pb.savefig(f"./Plots/amp_v_slope_mle_{iz}.pdf", dpi=400)
pb.clf()
print(cut_values_low_nhalo.shape)


slope_ticks = np.arange(0.0, 2.1, 0.1)
amp_ticks = slope_ticks + 10.3
print(res_errors)
fig = pb.figure()
tab10 = pb.get_cmap("tab10")
l2p8_facecolors = [tab10(i) for i in range(8)]
lc_2p8 = 0
for r in range(len(res_parameters)):
    is_l2p8 = (res[r - lc_2p8] == "L2800N5040")

    if is_l2p8:
        edge_col = "#332288"
        face_col = l2p8_facecolors[lc_2p8]
        label = f"L2p8_m9 (lc={lc_2p8})"
    else:
        edge_col = FLAMINGO_colors_sorted_res[r - lc_2p8]
        face_col = edge_col
        label = res_names[r - lc_2p8]

    print(lc_2p8, res[r-lc_2p8], FLAMINGO_colors_sorted_res[r-lc_2p8], res_names[r-lc_2p8])
    pb.errorbar(res_parameters[r][0], res_parameters[r][1], 
                xerr=np.array([[lower_errors_res[r][0]], [upper_errors_res[r][0]]]), 
                yerr=np.array([[lower_errors_res[r][1]], [upper_errors_res[r][1]]]), 
                fmt='none', capsize=3, ecolor=face_col, zorder=1)
    pb.scatter(res_parameters[r][0], res_parameters[r][1], marker='*', s=50, facecolor=face_col, edgecolor=edge_col, zorder=2,
            #    label=f'{sims[s]}: amp=${sim_parameters[s][0]:.3f}_{{-{lower_errors[s][0]}}}^{{+{upper_errors[s][0]}}}$, slope=${sim_parameters[s][1]:.3f}_{{-{lower_errors[s][1]}}}^{{+{upper_errors[s][1]}}}$')
                label=label)
    if is_l2p8:
        lc_2p8 += 1

pb.xlabel('Amplitude')
pb.ylabel('Slope')
# pb.xticks(amp_ticks)
# pb.yticks(slope_ticks)
pb.grid(which='major', linestyle='--', linewidth=0.5, alpha=0.7)
# pb.title(f'MLE parameters (sample={iz})', wrap=True)
pb.xlim(10.3, 11.3)
pb.ylim(0.0, 1.2)
# fig.legend(loc='outside right upper', fontsize=4)
pb.legend(loc='upper right', fontsize=6)
pb.savefig(f"./Plots/amp_v_slope_mle_{iz}_resolution.pdf", dpi=400)
pb.clf()


# print(z[0]*np.array((low_nhalos_x[plateau_point:]))+z[1])
# print(np.round(z[0]*np.array((low_nhalos_x[plateau_point:]))+z[1], 1))

print(kusiak_observed_nbar_full_sky)
kusiak_observed_nbar_full_sky_bottom_decile = kusiak_observed_nbar_full_sky - (0.1 * kusiak_observed_nbar_full_sky)
kusiak_observed_nbar_full_sky_top_decile = kusiak_observed_nbar_full_sky + (0.1 * kusiak_observed_nbar_full_sky)
kusiak_observed_nbar_full_sky_bottom_half = kusiak_observed_nbar_full_sky - (0.5 * kusiak_observed_nbar_full_sky)
kusiak_observed_nbar_full_sky_top_half = kusiak_observed_nbar_full_sky + (0.5 * kusiak_observed_nbar_full_sky)
kusiak_observed_nbar_full_sky_bottom_cut = kusiak_observed_nbar_full_sky - ((1 - abundance_cut) * kusiak_observed_nbar_full_sky)
kusiak_observed_nbar_full_sky_top_cut = kusiak_observed_nbar_full_sky + ((1 - abundance_cut) * kusiak_observed_nbar_full_sky)
print(kusiak_observed_nbar_full_sky_bottom_decile, kusiak_observed_nbar_full_sky_top_decile)
print(kusiak_observed_nbar_full_sky_bottom_half, kusiak_observed_nbar_full_sky_top_half)


cmap = pb.get_cmap('tab10')
for i in range(len(nhalos)):
    # print(cut_amplitude[i][0], cut_slope[i][0], nhalos[i])
    if cut_slope[i] == 0.0:
         pb.plot(cut_amplitude[i][0], nhalos[i], 's', color=cmap(0), label=f'Slope={cut_slope[i][0]:.1f}' if cut_amplitude[i][0]==10.3 else "", markersize=3)
    else:
        pb.plot(cut_amplitude[i][0], nhalos[i], 'o', label=f'Slope={cut_slope[i][0]:.1f}' if cut_amplitude[i][0]==10.3 else "", markersize=3)
pb.scatter(mle_amp, mle_nhalos, c='black', marker='*', s=50, label=f'MLE from MCMC (amp={mle_amp:.3f}, slope={mle_slope:.3f})')
pb.axhline(y=kusiak_observed_nbar_full_sky, color='r', linestyle='--', label='Observed nbar all-sky')
pb.fill_betweenx(y=[kusiak_observed_nbar_full_sky_bottom_decile, kusiak_observed_nbar_full_sky_top_decile], x1=10.3, x2=11.3, color='grey', alpha=0.7, label='10% range around observed nbar', linewidth=0)
pb.fill_betweenx(y=[kusiak_observed_nbar_full_sky_bottom_half, kusiak_observed_nbar_full_sky_top_half], x1=10.3, x2=11.3, color='grey', alpha=0.5, label='50% range around observed nbar', linewidth=0)
pb.fill_betweenx(y=[kusiak_observed_nbar_full_sky_bottom_cut, kusiak_observed_nbar_full_sky_top_cut], x1=10.3, x2=11.3, color='grey', alpha=0.3, label=f'{(1-abundance_cut) * 100}% range around abundance cut', linewidth=0)
pb.xlabel('Amplitude')
pb.ylabel('Number of halos')
pb.yscale('log')
pb.title(f'Number of halos vs Amplitude (sim={isim}, sample={iz})')
pb.legend(loc='lower left', fontsize=6)
pb.savefig(f"./Plots/nhalos_vs_amplitude_{isim}_{iz}.png", dpi=400)
pb.clf()

zoom = 0.7
for i in range(len(nhalos)):
    # print(cut_amplitude[i][0], cut_slope[i][0], nhalos[i])
    if cut_slope[i] == 0.0:
         pb.plot(cut_amplitude[i][0], nhalos[i], 's', color=cmap(0), label=f'Slope={cut_slope[i][0]:.1f}' if cut_amplitude[i][0]==10.3 else "", markersize=3)
    else:
        pb.plot(cut_amplitude[i][0], nhalos[i], 'o', label=f'Slope={cut_slope[i][0]:.1f}' if cut_amplitude[i][0]==10.3 else "", markersize=3)
pb.scatter(mle_amp, mle_nhalos, c='black', marker='*', s=50, label=f'MLE from MCMC (amp={mle_amp:.3f}, slope={mle_slope:.3f})')
pb.axhline(y=kusiak_observed_nbar_full_sky, color='r', linestyle='--', label='Observed nbar all-sky')
pb.fill_betweenx(y=[kusiak_observed_nbar_full_sky_bottom_decile, kusiak_observed_nbar_full_sky_top_decile], x1=10.3, x2=11.3, color='grey', alpha=0.7, label='10% range around observed nbar', linewidth=0)
pb.fill_betweenx(y=[kusiak_observed_nbar_full_sky_bottom_half, kusiak_observed_nbar_full_sky_top_half], x1=10.3, x2=11.3, color='grey', alpha=0.5, label='50% range around observed nbar', linewidth=0)
pb.fill_betweenx(y=[kusiak_observed_nbar_full_sky_bottom_cut, kusiak_observed_nbar_full_sky_top_cut], x1=10.3, x2=11.3, color='grey', alpha=0.3, label=f'{(1-abundance_cut) * 100}% range around abundance cut', linewidth=0)
pb.xlabel('Amplitude')
pb.ylabel(f'Number of halos vs Amplitude (zoom = {zoom}) (sim={isim}, sample={iz})')
# pb.yscale('log')
pb.ylim(kusiak_observed_nbar_full_sky - (zoom * kusiak_observed_nbar_full_sky), kusiak_observed_nbar_full_sky + (zoom * kusiak_observed_nbar_full_sky))
pb.title(f'Number of halos vs Amplitude (sim={isim}, sample={iz})')
pb.legend(loc='upper right', ncols=2, fontsize=6)
pb.savefig(f"./Plots/nhalos_vs_amplitude_zoom_{isim}_{iz}.png", dpi=400)
pb.clf()


fig = pb.figure()
ax = fig.add_subplot(projection='3d')

ax.plot(cut_amplitude[:21], cut_slope[:21], zs=np.log10(kusiak_observed_nbar_full_sky), zdir='z', label='observed abundance', color='r', linestyle='--')
ax.plot(cut_amplitude[20::21], cut_slope[20::21], zs=np.log10(kusiak_observed_nbar_full_sky), zdir='z', color='r', linestyle='--')

ax.plot(cut_amplitude[:21], cut_slope[:21], zs=np.log10(nhalo_lower_bound), zdir='z', label=f'lower bound ({abundance_cut})', color='grey', linestyle='--')
ax.plot(cut_amplitude[20::21], cut_slope[20::21], zs=np.log10(nhalo_lower_bound), zdir='z', color='grey', linestyle='--')

for i, (im, slope) in enumerate(cut_values):
    if slope == 0.0:
        ax.scatter(im, slope, np.log10(nhalos[i]), marker='s', color = cmap(0), label=f'Slope={slope:.1f}' if im==10.3 else "", s=20)
    else:
        ax.scatter(im, slope, np.log10(nhalos[i]), marker='o' if slope <= 1.0 else '2', label=f'Slope={slope:.1f}' if im==10.3 else "", s=20)

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
ax.set_title(f'Number of halos vs Amplitude and Slope (sim={isim}, sample={iz})', wrap=True)
# ax.set_zlim(0.0, 150000000)
# ax.set_zticks([1, 10, 100, 1000, 10000, 100000, 1000000, 10000000, 100000000])

fig.legend(loc='outside left upper', fontsize=6)
# ax.set_zscale('log')
ax.view_init(elev=10, azim=-5, roll=0)
# ax.set_aspect('equal', adjustable='box')

pb.savefig(f"./Plots/amp_slope_abundance_3d_{isim}_{iz}_better_angle.png", dpi=400)