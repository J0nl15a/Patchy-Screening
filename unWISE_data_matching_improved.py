import numpy as np, pylab as pb
from scipy.interpolate import interp1d
import re, textwrap
from io import StringIO
from pathlib import Path

def name_float(x, mle=False):
    if mle:
        return f"{float(x):.3f}".replace(".", "p")
    else:
        return f"{float(x):.1f}".replace(".", "p")
    

def unWISE_data_matching(boxname, simname, z_sample, mass_cut, n_cut, nsamp='ntotal', mle=False, plot=False, lightcone=0):
    box_list = ['L1000N1800', 'L1000N3600', 'L2800N5040']
    sim_list = ['HYDRO_FIDUCIAL','HYDRO_PLANCK','HYDRO_PLANCK_LARGE_NU_FIXED','HYDRO_PLANCK_LARGE_NU_VARY','HYDRO_STRONG_AGN','HYDRO_WEAK_AGN','HYDRO_LOW_SIGMA8','HYDRO_STRONGER_AGN','HYDRO_JETS_published','HYDRO_STRONGEST_AGN','HYDRO_STRONG_SUPERNOVA','HYDRO_STRONGER_AGN_STRONG_SUPERNOVA','HYDRO_STRONG_JETS']

    try:
        box = int(boxname)
        boxname = box_list[box]
    except (ValueError, IndexError):
        boxname = str(boxname)

    try:
        isim = int(simname)
        simname = sim_list[isim]
    except (ValueError, IndexError):
        simname = str(simname)

    z_sample = str(z_sample)

    im_name = name_float(mass_cut, mle=mle)
    slope_name = name_float(n_cut, mle=mle)

    try:
        nsamp = int(nsamp)
    except ValueError:
        nsamp = str(nsamp)

    lightcone = int(lightcone)

    dndz_match = np.loadtxt(f"/cosma8/data/dp004/dc-conl1/FLAMINGO/patchy_screening/unWISExLens_lklh/data/v1.0/aux_data/dndz/unWISE_{z_sample.lower()}_xmatch_dndz.txt", usecols=(0,1))

    print(min(dndz_match[:,0]), max(dndz_match[:,0]))

    halo_z_bins = np.genfromtxt(
        f'/cosma8/data/dp004/dc-conl1/FLAMINGO/patchy_screening/data_files/halo_redshifts/{boxname}/{simname}/lightcone{lightcone}/FLAMINGO_halo_redshift_values.txt',
        dtype=[('i',   'i4'),
               ('z_min', 'f8'),
               ('mid_z', 'f8'),
               ('z_max', 'f8')],
        delimiter=None
    )

    z_mask = halo_z_bins["mid_z"] <= 3.0
    FLAMINGO_mid_point = halo_z_bins["mid_z"][z_mask]
    FLAMINGO_z_bins = np.concatenate([[halo_z_bins["z_min"][z_mask][0]], halo_z_bins["z_max"][z_mask]])
    print(FLAMINGO_mid_point)
    print(FLAMINGO_z_bins)

    if len(FLAMINGO_z_bins) != len(FLAMINGO_mid_point) + 1:
        raise ValueError("Redshift shell edges and midpoints are inconsistent")

    for i in range(len(FLAMINGO_mid_point)):
        if not (FLAMINGO_z_bins[i] <= FLAMINGO_mid_point[i] <= FLAMINGO_z_bins[i + 1]):
            raise ValueError(f"Invalid shell {i}: {FLAMINGO_z_bins[i]} < {FLAMINGO_mid_point[i]} < {FLAMINGO_z_bins[i + 1]}")

    with open(f"/cosma8/data/dp004/dc-conl1/FLAMINGO/patchy_screening/data_files/halo_totals/{boxname}/{simname}/{z_sample}/lightcone{lightcone}/FLAMINGO_halo_totals_{im_name}_{slope_name}.txt", "r") as f:
        first_line = f.readline().strip()
        remaining_lines = f.readlines()

    # Extract number from the first line
    match = re.search(r"(\d+)", first_line)
    if match:
        total_available_halos = int(match.group(1))
        print("Total number of suitable halos:", total_available_halos)
    else:
        raise ValueError("No number found in the first line")

    # Convert remaining lines to a NumPy array
    halo_lightcones_str = "".join(remaining_lines)
    halo_lightcones = np.loadtxt(StringIO(halo_lightcones_str), usecols=(0,1))
    print(halo_lightcones.shape)

    m = interp1d(dndz_match[:,0], dndz_match[:,1], kind='cubic', bounds_error=False, fill_value=0.0)

    def dndz_func(z):
        return np.maximum(m(z), 0.0)
    

    if plot==True:
        pb.plot(dndz_match[:,0], dndz_match[:,1], color='g', marker='.', label='Original curve')
        pb.plot(FLAMINGO_z_bins, dndz_func(FLAMINGO_z_bins), color='b', marker='.', label='Interpolated curve')
        pb.plot(FLAMINGO_mid_point, dndz_func(FLAMINGO_mid_point), color='r', marker='.', label='Midpoint')
        pb.xlim(left=0, right=3)
        pb.xlabel('z')
        pb.ylabel('dn/dz')
        pb.title("\n".join(textwrap.wrap(f'unWISE dndz ({z_sample} sample)', width=75)))
        pb.legend()
        pb.savefig(f'./Plots/unWISE_dndz_match_curve_{z_sample}_test.png', dpi=400)
        pb.clf()

    outfile_name = f'/cosma8/data/dp004/dc-conl1/FLAMINGO/patchy_screening/data_files/dndz_samples/{boxname}/{simname}/{z_sample}/lightcone{lightcone}/dndz_galaxies_sampled_{im_name}_{slope_name}.txt'
    outfile_name = Path(outfile_name)
    outfile_name.parent.mkdir(parents=True, exist_ok=True)

    if nsamp == "ntotal":

        if z_sample == "Blue":
            kusiak_nbar = 3409

        elif z_sample == "Green":
            kusiak_nbar = 1846

        else:
            raise ValueError(f"Unknown redshift sample: {z_sample}")

        nsamp = int(round(kusiak_nbar * 41253))

    elif isinstance(nsamp, (int, np.integer)):
        nsamp = int(nsamp)

    else:
        raise ValueError(f"nsamp must be an integer or 'ntotal', got {nsamp!r}")

    nsamp = min(nsamp, total_available_halos)
        
    print(nsamp)
    
    count = 0
    conflict = True

    while conflict:
        count += 1

        galaxies_required = compute_galaxies_required(FLAMINGO_mid_point, FLAMINGO_z_bins, dndz_func, nsamp, halo_lightcones[:, 1])

        (nsamp, galaxies_required, conflict, z_limit) = rescale_dndz(galaxies_required, halo_lightcones[:, 1], nsamp, FLAMINGO_mid_point, z_sample)

        if count > 100:
            raise RuntimeError("dN/dz rescaling failed to converge after 100 iterations")

        if plot==True:
            pb.plot(FLAMINGO_mid_point, galaxies_required, color='b', marker='.', label='Galaxies required')
            pb.plot(FLAMINGO_mid_point, halo_lightcones[:, 1], color='r', marker='.', label='Galaxies available')
            pb.xlim(left=0, right=3)
            pb.xlabel('z')
            pb.ylabel('dn/dz')
            pb.title("\n".join(textwrap.wrap(f'unWISE galaxy redshift distribution cross-match (rescaled, boxname = {boxname}, simname = {simname}, {z_sample} sample, stellar cut at mean z = {mass_cut}, stellar cut slope = {n_cut}, total number of galaxies = {nsamp})', width=75)))
            pb.legend()
            pb.savefig(f'./Plots/unWISE_dndz_match_rescaled_{boxname}_{simname}_{z_sample}_{im_name}_{slope_name}_{count}.png', dpi=400)
            pb.clf()
    
    if plot==True:
        pb.plot(FLAMINGO_mid_point, galaxies_required, color='b', marker='.', label='Galaxies required')
        pb.plot(FLAMINGO_mid_point, halo_lightcones[:, 1], color='r', marker='.', label='Galaxies available')
        pb.plot(FLAMINGO_mid_point, np.minimum(galaxies_required, halo_lightcones[:, 1]), color='g', marker='.', label='Galaxies sampled')
        pb.xlim(left=0, right=3)
        pb.xlabel('z')
        pb.ylabel('dn/dz')
        pb.title("\n".join(textwrap.wrap(f'unWISE galaxy redshift distribution cross-match (rescaled, boxname = {boxname}, simname = {simname}, {z_sample} sample, stellar cut at mean z = {mass_cut}, stellar cut slope = {n_cut}, total number of galaxies = {nsamp})', width=75)))
        pb.legend()
        pb.savefig(f'./Plots/unWISE_dndz_match_rescaled_{boxname}_{simname}_{z_sample}_{im_name}_{slope_name}_test.png', dpi=400)
        pb.clf()

    sampled_per_shell = np.rint(galaxies_required).astype(int)
    sampled_per_shell = np.minimum(sampled_per_shell, halo_lightcones[:, 1].astype(int))
    sampled_per_shell[halo_lightcones[:, 1] == 0] = 0

    if plot:
        delta_z = np.diff(FLAMINGO_z_bins)
        sampled_dndz = np.divide(sampled_per_shell, delta_z)
        sampled_shape = (sampled_dndz / np.sum(sampled_per_shell))
        target_midpoint_shape = dndz_func(FLAMINGO_mid_point)

        # normalize consistently as a density
        normalisation = np.sum(target_midpoint_shape * delta_z)
        target_midpoint_shape /= normalisation

        pb.plot(FLAMINGO_mid_point, target_midpoint_shape, marker="o", label="Target midpoint dN/dz")
        pb.plot(FLAMINGO_mid_point, sampled_shape, marker="o", label="Sampled dN/dz")
        pb.xlim(left=0, right=3)
        pb.xlabel('z')
        pb.ylabel('dN/dz (normalized)')
        pb.title("\n".join(textwrap.wrap(f'unWISE galaxy redshift distribution cross-match (rescaled, boxname = {boxname}, simname = {simname}, {z_sample} sample, stellar cut at mean z = {mass_cut}, stellar cut slope = {n_cut}, total number of galaxies = {nsamp})', width=75)))
        pb.legend()
        pb.savefig(f'./Plots/unWISE_dndz_match_rescaled_{boxname}_{simname}_{z_sample}_{im_name}_{slope_name}_shape_test.png', dpi=400)
        pb.clf()

    print("Floating shell total:", np.sum(galaxies_required))
    print("Integer sampled total:", np.sum(sampled_per_shell))

    actual_total_sampled = (write_sampled_galaxies_file(outfile_name, FLAMINGO_mid_point, sampled_per_shell))

    return


# This refactored version separates the loop from file-writing so the
# user can inspect galaxy counts and rescale before final output

def compute_galaxies_required(FLAMINGO_mid_point, FLAMINGO_z_bins, dndz_func, nsamp, available_counts):
    """
    Match the shape of dN/dz using midpoint evaluation.

    N_i ∝ dN/dz(z_mid_i) * delta_z_i

    Empty lightcone shells are assigned zero galaxies.
    """

    FLAMINGO_mid_point = np.asarray(FLAMINGO_mid_point, dtype=float)
    FLAMINGO_z_bins = np.asarray(FLAMINGO_z_bins, dtype=float)
    available_counts = np.asarray(available_counts, dtype=float)

    delta_z = np.diff(FLAMINGO_z_bins)

    if len(delta_z) != len(FLAMINGO_mid_point):
        raise ValueError("Number of shell widths does not match number of shell midpoints")

    if len(available_counts) != len(FLAMINGO_mid_point):
        raise ValueError("Available halo counts do not match number of redshift shells")

    midpoint_dndz = np.asarray(dndz_func(FLAMINGO_mid_point), dtype=float)

    if not np.all(np.isfinite(midpoint_dndz)):
        raise ValueError("Interpolated midpoint dN/dz contains non-finite values")

    midpoint_dndz = np.clip(midpoint_dndz, 0.0,  None)

    # Shape weights based exactly on your midpoint method.
    shell_weights = midpoint_dndz * delta_z

    weight_sum = np.sum(shell_weights)

    if not np.isfinite(weight_sum) or weight_sum <= 0:
        raise ValueError(f"Invalid midpoint dN/dz weight sum: {weight_sum}")

    if np.sum(shell_weights) <= 0:
        raise ValueError("dN/dz midpoint weights have zero total")

    shell_weights /= weight_sum
    galaxies_required = (float(nsamp) * shell_weights)

    # Your requested rule:
    # if the lightcone shell is empty, sample nothing.
    galaxies_required[available_counts == 0] = 0.0

    return galaxies_required


def write_sampled_galaxies_file(outfile_name, FLAMINGO_mid_point, sampled_per_shell):

    sampled_per_shell = np.asarray(sampled_per_shell, dtype=int)

    actual_total = int(np.sum(sampled_per_shell))

    with open(outfile_name, "w") as outfile:

        outfile.write(f"#Total number of sampled galaxies: {actual_total}\n")

        for zmid, n_gals in zip(FLAMINGO_mid_point, sampled_per_shell):
            outfile.write(f"{zmid:.8f} {n_gals}\n")

    print(f"Actual total sampled = {actual_total:,}")
    print(f"Wrote file: {outfile_name}")

    return actual_total


def rescale_dndz(galaxies_required, available_counts, nsamp, FLAMINGO_mid_point, galaxy_sample):
    """
    Apply availability constraints while preserving the midpoint dN/dz shape.

    Rules:
      - empty shells sample zero;
      - shortages at z <= z_limit cause a global normalization reduction;
      - shortages at z > z_limit are simply capped at availability.
    """

    galaxies_required = np.asarray(galaxies_required, dtype=float)
    available_counts = np.asarray(available_counts, dtype=float)
    FLAMINGO_mid_point = np.asarray(FLAMINGO_mid_point, dtype=float)

    if galaxy_sample == "Blue":
        z_mean = 0.6
        z_width = 0.3

    elif galaxy_sample == "Green":
        z_mean = 1.1
        z_width = 0.4

    else:
        raise ValueError(f"Unknown galaxy sample: {galaxy_sample}")

    z_limit = (z_mean + 7.0 * (z_width / 2.0))

    # Empty shells never constrain the normalization.
    non_empty = available_counts > 0

    before_limit = (FLAMINGO_mid_point <= z_limit)

    shortage_before = (non_empty & before_limit & (galaxies_required > available_counts))

    # --------------------------------------------------
    # Shortage before z_limit:
    # reduce the overall normalization.
    # --------------------------------------------------

    if np.any(shortage_before):

        ratios = (available_counts[shortage_before] / galaxies_required[shortage_before])
        scale = min(1.0, np.min(ratios))
        new_nsamp = float(nsamp) * scale

        return (new_nsamp, galaxies_required, True, z_limit)

    shortage_beyond = (non_empty & (FLAMINGO_mid_point > z_limit) & (galaxies_required > available_counts))

    galaxies_required[shortage_beyond] = (available_counts[shortage_beyond])
    galaxies_required[~non_empty] = 0.0

    return (nsamp, galaxies_required, False, z_limit)


if __name__ == '__main__':
    import sys

    pb.rc("text", usetex=True)
    pb.rc("font", family="serif", size=8)
    pb.rcParams["font.size"] = 11

    boxname = sys.argv[1]
    simname = sys.argv[2]
    z_sample = sys.argv[3]
    mass_cut = sys.argv[4]
    n_cut = sys.argv[5]
    nsamp = sys.argv[6]
    mle = sys.argv[7].lower() in ("true", "1", "yes", "y")
    lightcone = sys.argv[8] if len(sys.argv) > 8 else 0

    plot=False
    unWISE_data_matching(boxname, simname, z_sample, mass_cut, n_cut, nsamp=nsamp, mle=mle, plot=plot, lightcone=lightcone)

    if plot==True:
        dndz_blue_match = np.loadtxt(f"/cosma8/data/dp004/dc-conl1/FLAMINGO/patchy_screening/unWISExLens_lklh/data/v1.0/aux_data/dndz/unWISE_blue_xmatch_dndz.txt", usecols=(0,1))
        dndz_green_match = np.loadtxt(f"/cosma8/data/dp004/dc-conl1/FLAMINGO/patchy_screening/unWISExLens_lklh/data/v1.0/aux_data/dndz/unWISE_green_xmatch_dndz.txt", usecols=(0,1))

        pb.plot(dndz_blue_match[:,0], dndz_blue_match[:,1], color='tab:blue', label='Blue sample')
        pb.plot(dndz_green_match[:,0], dndz_green_match[:,1], color='tab:green', label='Green sample')
        pb.vlines(x=0.6, color='tab:grey', linestyle='--', ymin=0, ymax=2.0, label='Mean z')
        pb.vlines(x=0.6, color='tab:blue', linestyle='--', ymin=0, ymax=2.0)
        pb.vlines(x=1.1, color='tab:green', linestyle='--', ymin=0, ymax=2.0)
        # pb.plot(FLAMINGO_z_bins, dndz_blue_match_interpolated, color='tab:cyan', marker='.', label='FLAMINGO bins')
        # pb.plot(FLAMINGO_z_bins, dndz_green_match_interpolated, color='tab:olive', marker='.', label='FLAMINGO bins')
        pb.xlim(left=0, right=4.0)
        pb.ylim(bottom=0, top=1.2)
        pb.xlabel('$z$')
        pb.ylabel(r"$\frac{1}{N_{\rm g, total}}\frac{dN_{\rm g}}{dz}$")
        # pb.title('unWISE galaxy redshift distribution cross-match')
        pb.legend(frameon=False)
        pb.savefig('./Plots/unWISE_dndz_match.pdf', dpi=400)
        pb.clf()
