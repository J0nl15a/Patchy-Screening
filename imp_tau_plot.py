import os, pickle
import numpy as np, pylab as pb
from matplotlib.ticker import ScalarFormatter
from matplotlib.colors import LinearSegmentedColormap, to_rgb
import matplotlib.colors as mcolors
from scipy.interpolate import interp1d

pb.rc("text", usetex=True)
pb.rc("font", family="serif", size=11)
pb.rcParams["font.size"] = 11

class TauPlotter:
    """
    A class for loading and plotting tau profile data from pickle files.
    
    The file name format is assumed to be of the form:
      ./<sim_size>/<colour>/<sim>_tau_Mstar_<stellar_bin>_nside<nside>_<primary_method>_<file_method>_<signal_flag>.pickle
      
    This class provides methods to create plots for:
      - Each stellar bin (plotting all colours on one axis)
      - Each colour (plotting all stellar mass bins on one axis)
      - Each file method (plotting all results for a given stellar bin and colour)
      - A generic custom plot where you can provide an arbitrary list of files.
    
    Optional noise data can be overplotted.
    """
    def __init__(self, base_dir):
        # Base directory for the simulation results, e.g., "./L1000N1800"
        self.base_dir = base_dir

    def construct_filepath(self, sim, colour, mass_bin, slope_bin, nside=8192, primary_method='FITS', file_method='unlensed', has_signal=True):
        signal_suffix = "" if has_signal else "_no_ps"
        fits_suffix = "" if primary_method != 'FITS' else f"_{file_method}"
        self.stellar_bins = f"{mass_bin:.1f}".replace('.', 'p')
        self.slope = f"{slope_bin:.1f}".replace('.', 'p')
        filename = f"{sim}_tau_Mstar_bin{self.stellar_bins}_{self.slope}_nside{nside}_{primary_method}{fits_suffix}{signal_suffix}.pickle"
        filepath = os.path.join(self.base_dir, colour, filename)
        return filepath


    def load_data(self, sim, colour, mass_bin, slope_bin, nside=8192, primary_method='FITS', file_method='unlensed', signal_flag=True):
        # Load the pickle file corresponding to the given parameters.
        filepath = self.construct_filepath(sim, colour, mass_bin, slope_bin, nside, primary_method, file_method, signal_flag)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        # Assume data is a list [theta_d, tau_profile, converted_distance] (adjust if needed)
        return data

    def plot_by_stellar_bin(self, sim, mass_bin, slope_bin, colours, nside=8192, primary_method='FITS', file_method='unlensed', signal_flag=True, noise_data=None):
        # For a fixed stellar bin, plot tau profiles for multiple colours.
        fig, ax = pb.subplots(figsize=(8,6), constrained_layout=True, dpi=400)
        ax2 = ax.twiny()
        ax.hlines(y=0, xmin=-1, xmax=12, linestyles='-', color='k', label=None)
        line_colours = {'Blue':'tab:blue', 'Green':'tab:green', 'Red':'tab:red'}
        for colour in colours:
            data = self.load_data(sim, colour, mass_bin, slope_bin, nside, primary_method, file_method, signal_flag)
            theta_d = data[0]
            tau_profile = data[1]
            ax.plot(theta_d, tau_profile, color=line_colours[colour], label=f"{colour}")
            if noise_data is not None and colour in noise_data:
                ax.plot(theta_d, noise_data[colour], '--', color=line_colours[colour], label=f"{colour} noise")

        formatter = ScalarFormatter(useMathText=True)
        formatter.set_powerlimits((-4, -4))
        ax.yaxis.set_major_formatter(formatter)
        ax.ticklabel_format(axis='y', style='sci', scilimits=(-4, -4))

        ax.set_xlabel("Annulus centre (arcmin)")
        ax.set_ylabel(r"Filtered $\tau$")
        ax.set_title(f"$\\tau$ Profiles for log$M_*$ = {mass_bin}\n(sim={sim}, nside={nside}, primary CMB={file_method})" if primary_method=='FITS' else f"$\\tau$ Profiles for log$M_*$ = {mass_bin}\n(sim={sim}, nside={nside}, primary CMB={primary_method})")
        ax.set_xlim(left=0, right=11)
        ax.legend(fontsize=8, loc="best", frameon=False)

        ax2.plot(data[2], data[1], alpha=0)
        ax2.set_xlabel('r [Mpc/h]')

        primary_suffix = "_CAMB" if primary_method=='CAMB' else "_FITS"
        file_suffix = "" if file_method==False else f"_{file_method}"
        signal_suffix = "" if signal_flag==True else "_no_ps"
        noise_suffix = "" if noise_data==None else "_noise"
        pb.savefig(f"./Plots/{sim}_stellar_bin_{self.stellar_bins}_{self.slope}_nside{nside}{primary_suffix}{file_suffix}{signal_suffix}{noise_suffix}.png", dpi=400)
        pb.clf()

    def plot_by_colour(self, sim, colour, mass_bins, slope_bin, nside=8192, primary_method='FITS', file_method='unlensed', signal_flag=True, noise_data=None):
        # For a fixed colour, plot tau profiles for multiple stellar mass bins.
        fig, ax = pb.subplots(figsize=(8,6), constrained_layout=True, dpi=400)
        ax2 = ax.twiny()
        ax.hlines(y=0, xmin=-1, xmax=12, linestyles='-', color='k', label=None)
        alpha = [a for a in np.linspace(1.0, 0.2, len(mass_bins))] 
        line_colours = {'Blue':'tab:blue', 'Green':'tab:green', 'Red':'tab:red'}
        base_colour = mcolors.to_rgb(line_colours[colour])
        cmap = LinearSegmentedColormap.from_list("my_colour", [(1,1,1), base_colour], N=256)
        for i, mass_bin in enumerate(mass_bins):
            data = self.load_data(sim, colour, mass_bin, slope_bin, nside, primary_method, file_method, signal_flag)
            theta_d = data[0]
            tau_profile = data[1]
            colour_shade = cmap(alpha[i])
            ax.plot(theta_d, tau_profile, color=colour_shade, label=f"{mass_bin}")
            if noise_data is not None and mass_bin in noise_data and i==0:
                ax.plot(theta_d, noise_data[mass_bin], '--', color=line_colours[colour], label=f"{mass_bin} noise")

        formatter = ScalarFormatter(useMathText=True)
        formatter.set_powerlimits((-4, -4))
        ax.yaxis.set_major_formatter(formatter)
        ax.ticklabel_format(axis='y', style='sci', scilimits=(-4, -4))
                
        ax.set_xlabel("Annulus centre (arcmin)")
        ax.set_ylabel(r"Filtered $\tau$")
        ax.set_title(f"$\\tau$ Profiles for unWISE sample = {colour}\n(sim={sim}, nside={nside}, primary CMB={file_method})" if primary_method=='FITS' else f"$\\tau$ Profiles for unWISE sample = {colour}\n(sim={sim}, nside={nside}, primary CMB={primary_method})")
        ax.set_xlim(left=0, right=11)
        ax.legend(fontsize=8, loc="best", title=f"{colour} sample", frameon=False)

        ax2.plot(data[2], data[1], alpha=0)
        ax2.set_xlabel('r [Mpc/h]')

        primary_suffix = "_CAMB" if primary_method=='CAMB' else "_FITS"
        file_suffix = "" if file_method==False else f"_{file_method}"
        signal_suffix = "" if signal_flag==True else "_no_ps"
        noise_suffix = "" if noise_data==None else "_noise"
        pb.savefig(f"./Plots/{sim}_sample_{colour}_{self.slope}_nside{nside}{primary_suffix}{file_suffix}{signal_suffix}{noise_suffix}.png", dpi=400)
        pb.clf()

    def plot_by_file_method(self, sim, colour, mass_bin, slope_bin, nside=8192, primary_method='FITS', file_methods='unlensed', signal_flag=True, noise_data=None):
        # For a fixed stellar bin and colour, plot tau profiles for different file method suffixes.
        fig, ax = pb.subplots(figsize=(8,6), constrained_layout=True, dpi=400)
        ax2 = ax.twiny()
        ax.hlines(y=0, xmin=-1, xmax=12, linestyles='-', color='k', label=None)
        line_colours = {'Blue':'tab:blue', 'Green':'tab:green', 'Red':'tab:red'}
        line_styles = {'unlensed':'-', 'lensed_z2':'--', 'lensed_z3':':'}
        if signal_flag == False:
            for file_method in file_methods:
                data = self.load_data(sim, colour, mass_bin, slope_bin, nside, primary_method, file_method, signal_flag=True)
                theta_d = data[0]
                tau_profile = data[1]
                ax.plot(theta_d, tau_profile, line_styles[file_method], color=line_colours[colour], label=f"{file_method}")
                if noise_data is not None and file_method in noise_data:
                    ax.plot(theta_d, noise_data[file_method], '--', color=line_colours[colour], label=f"{file_method} noise")
            data = self.load_data(sim, colour, mass_bin, slope_bin, nside, primary_method, file_method, signal_flag)
            theta_d = data[0]
            tau_profile = data[1]
            ax.plot(theta_d, tau_profile, '-.', color=line_colours[colour], label=f"{file_method} (no PS)")
        else:
            for file_method in file_methods:
                data = self.load_data(sim, colour, mass_bin, slope_bin, nside, primary_method, file_method, signal_flag)
                theta_d = data[0]
                tau_profile = data[1]
                ax.plot(theta_d, tau_profile, line_styles[file_method], color=line_colours[colour], label=f"{file_method}")
                if noise_data is not None and file_method in noise_data:
                    ax.plot(theta_d, noise_data[file_method], '--', color=line_colours[colour], label=f"{file_method} noise")
        
        formatter = ScalarFormatter(useMathText=True)
        formatter.set_powerlimits((-4, -4))
        ax.yaxis.set_major_formatter(formatter)
        ax.ticklabel_format(axis='y', style='sci', scilimits=(-4, -4))
        
        ax.set_xlabel("Annulus centre (arcmin)")
        ax.set_ylabel(r"Filtered $\tau$")
        ax.set_title(f"$\\tau$ Profiles for {colour} sample, log$M_*$={mass_bin} for different lensing methods\n(sim={sim}, nside={nside})")
        ax.set_xlim(left=0, right=11)
        ax.legend(fontsize=8, loc="best", title=f"{colour} sample", frameon=False)

        ax2.plot(data[2], data[1], alpha=0)
        ax2.set_xlabel('r [Mpc/h]')

        primary_suffix = "_CAMB" if primary_method=='CAMB' else "_FITS"
        signal_suffix = "" if signal_flag==True else "_no_ps"
        noise_suffix = "" if noise_data==None else "_noise"
        pb.savefig(f"./Plots/{sim}_method_comp_{colour}_{self.stellar_bins}_{self.slope}_nside{nside}{primary_suffix}{signal_suffix}{noise_suffix}.png", dpi=400)
        pb.clf()

    # def generic_plot(self, file_list, labels, line_styles, colours, alpha, plot_title, label_title, outname, noise_data=None):
    #     # file_list: list of file paths to load
    #     # labels: list of labels corresponding to each file
    #     fig, ax = pb.subplots(figsize=(8,6), constrained_layout=True, dpi=400)
    #     ax2 = ax.twiny()
    #     ax.hlines(y=0, xmin=-1, xmax=12, linestyles='-', color='k', label=None)
    #     for fp, lab, style, colour, alpha in zip(file_list, labels, line_styles, colours, alpha):
    #         with open(fp, 'rb') as f:
    #             data = pickle.load(f)
    #         theta_d = data[0]
    #         tau_profile = data[1]
    #         ax.plot(theta_d, tau_profile, style, label=lab, color=colour, alpha=alpha)
    #         if noise_data is not None and lab in noise_data:
    #             ax.plot(theta_d, noise_data[lab], '--', label=f"{lab} noise")
    #     formatter = ScalarFormatter(useMathText=True)
    #     formatter.set_powerlimits((-4, -4))
    #     ax.yaxis.set_major_formatter(formatter)
    #     ax.ticklabel_format(axis='y', style='sci', scilimits=(-4, -4))
        
    #     ax.set_xlabel("Annulus centre (arcmin)")
    #     ax.set_ylabel(r"$\tau$")
    #     ax.set_title(plot_title)
    #     ax.set_xlim(left=0, right=11)
    #     ax.legend(fontsize=8, loc="best", title=label_title, frameon=False)

    #     ax2.plot(data[2], data[1], alpha=0)
    #     ax2.set_xlabel('r [Mpc/h]')

    #     pb.savefig(os.path.join("./Plots", outname), dpi=400)
    #     pb.clf()

    def generic_plot(self, file_list, labels, line_styles, colours, alpha, plot_title, label_title, outname, noise_data=None, observed_data=None, fiducial_index=0, 
                    ratio_ylim=None, main_ylim=None):
        """
        Generic tau-profile comparison plot.

        Parameters
        ----------
        observed_data : str or None
            Path to digitized observed tau data.
            Expected columns:
                theta [arcmin]
                tau
                upper y-bound
                lower y-bound

            The observed tau values are assumed to already be in units of
            tau x 10^4, matching the tau-overview plotting convention.

        fiducial_index : int
            Index in file_list used as the reference profile in the
            difference panel. For the current lensing comparison this
            should be the unlensed profile, normally index 0.
        """

        # ------------------------------------------------------------
        # Set up main panel + difference panel
        # ------------------------------------------------------------

        fig, (ax, ax_ratio) = pb.subplots(2, 1, figsize=(8, 7), sharex=True, gridspec_kw={"height_ratios": [3.0, 1.0],"hspace": 0.05}, constrained_layout=False, dpi=400)

        # Secondary distance axis on top of main panel.
        ax2 = ax.twiny()

        ax.hlines(y=0, xmin=-1, xmax=12, linestyles="-", color="k", label=None,)
        ax_ratio.axhline(0.0, color="k", linestyle="--", linewidth=0.8, alpha=0.7)

        if main_ylim is not None:
            ax.set_ylim(*main_ylim)

        if ratio_ylim is not None:
            ax_ratio.set_ylim(*ratio_ylim)

        # ------------------------------------------------------------
        # Load all profiles first
        # ------------------------------------------------------------

        profiles = []

        for fp in file_list:
            with open(fp, "rb") as f:
                data = pickle.load(f)

            profiles.append(
                {"theta": np.asarray(data[0]),
                "tau": np.asarray(data[1]),
                "distance": np.asarray(data[2])})

        # ------------------------------------------------------------
        # Fiducial/reference profile
        # ------------------------------------------------------------

        fid_theta = profiles[fiducial_index]["theta"]
        fid_tau = profiles[fiducial_index]["tau"]

        # ------------------------------------------------------------
        # Plot simulations
        # ------------------------------------------------------------

        for profile, lab, style, colour, a in zip(profiles, labels, line_styles, colours, alpha):

            theta_d = profile["theta"]
            tau_profile = profile["tau"]

            # Main panel: explicitly show tau x 10^4.
            ax.plot(theta_d, tau_profile * 1.0e4, style, label=lab, color=colour, alpha=a)

            # Difference from unlensed/fiducial profile.
            #
            # Interpolate reference in case theta grids are not identical.
            fid_interp = np.interp(theta_d, fid_theta, fid_tau)

            ax_ratio.plot(theta_d, (tau_profile - fid_interp) * 1.0e4, style, color=colour, alpha=a)

            if noise_data is not None and lab in noise_data:
                ax.plot(theta_d, noise_data[lab] * 1.0e4, "--", label=f"{lab} noise",)

        # ------------------------------------------------------------
        # Observed data
        # ------------------------------------------------------------

        if observed_data is not None:

            obs = np.loadtxt(observed_data)

            obs_theta = obs[:, 0]
            obs_tau = obs[:, 1]
            obs_upper = obs[:, 2]
            obs_lower = obs[:, 3]

            # The digitized observed values are already tau x 10^4.
            obs_yerr = np.vstack(
                (obs_tau - obs_lower, obs_upper - obs_tau))

            # Main-panel observations.
            ax.errorbar(obs_theta, obs_tau, yerr=obs_yerr, fmt="o", markersize=4, color="k", ecolor="k", capsize=2, linewidth=1.0, label="Coulton et al. 2025", zorder=10)

            # --------------------------------------------------------
            # Observed - fiducial in difference panel
            # --------------------------------------------------------

            fid_function = interp1d(fid_theta, fid_tau * 1.0e4, kind="linear", bounds_error=False, fill_value="extrapolate")

            fid_at_obs = fid_function(obs_theta)

            obs_difference = obs_tau - fid_at_obs

            ax_ratio.errorbar(obs_theta, obs_difference, yerr=obs_yerr, fmt="o", markersize=4, color="k", ecolor="k", capsize=2, linewidth=1.0, zorder=10)

        # ------------------------------------------------------------
        # Axis labels and formatting
        # ------------------------------------------------------------

        ax.set_ylabel(r"$\tau \times 10^4$")
        ax_ratio.set_ylabel(r"$(\tau-\tau_{\rm fid})\times10^4$")
        ax_ratio.set_xlabel("Annulus centre (arcmin)")
        ax.set_xlim(left=0, right=11)

        # No scientific-notation formatter is required any more,
        # because the profiles themselves have been multiplied by 1e4.

        if plot_title is not None:
            ax.set_title(plot_title)

        ax.legend(fontsize=10, loc="best", title=label_title, title_fontsize=11, frameon=False)

        # ------------------------------------------------------------
        # Top physical-distance axis
        # ------------------------------------------------------------

        fid_distance = profiles[fiducial_index]["distance"]

        ax2.plot(fid_distance, fid_tau * 1.0e4, alpha=0)

        ax2.set_xlabel(r"$r\ [{\rm Mpc}/h]$")

        # Keep main and lower panel visually aligned.
        ax.tick_params(axis="x", labelbottom=False)

        pb.savefig(
            os.path.join("./Plots", outname), dpi=400, bbox_inches="tight")
        pb.close(fig)


if __name__ == '__main__':

    tp = TauPlotter(base_dir="./L1000N1800")
    
    # tp.generic_plot(file_list=['./data_files/tau_profiles/L1000N1800//HYDRO_LOW_SIGMA8_tau_Mstar_bin10p725_0p894_nside8192_FITS_unlensed.pickle'],
    #                 labels=["Optimized catalog method"],
    #                 line_styles=["-"],
    #                 colours=["tab:blue"],
    #                 alpha=[1],
    #                 plot_title=f'$\\tau$ Profiles for Blue sample, $\log M_*$=10.725, $n_{{cut}}$=0.894, with optimized mock catalog, \n(sim=HYDRO_FIDUCIAL, nside=8192)',      
    #                 outname='HYDRO_LOW_SIGMA8_optimum_catalog_Blue_10p725_0p894_nside8192_unlensed.png')
    # quit()

    tp.generic_plot(file_list=['./data_files/tau_profiles/L1000N1800/HYDRO_FIDUCIAL/Green/lightcone0/tau_mle_catalogue_nside8192_FITS_unlensed.pickle',
                            #    './data_files/tau_profiles/L1000N1800/HYDRO_FIDUCIAL/Green/lightcone0/tau_mle_catalogue_nside8192_FITS_unlensed_no_ps.pickle',
                            #    './data_files/tau_profiles/L1000N1800/HYDRO_LOW_SIGMA8/Green/lightcone0/tau_mle_catalogue_nside8192_FITS_unlensed.pickle',
                            #    './data_files/tau_profiles/L1000N1800/HYDRO_LOW_SIGMA8/Green/lightcone0/tau_mle_catalogue_nside8192_FITS_unlensed_no_ps.pickle',
                            #    './data_files/tau_profiles/L1000N1800/HYDRO_FIDUCIAL/Green/lightcone0/tau_from_image_mle_catalogue_nside8192_FITS_unlensed.pickle',
                            #    './data_files/tau_profiles/L1000N1800/HYDRO_FIDUCIAL/Green/lightcone0/tau_mle_catalogue_nside8192_FITS_unlensed_no_ps.pickle',
                               './data_files/tau_profiles/L1000N1800/HYDRO_FIDUCIAL/Green/lightcone0/tau_mle_catalogue_nside8192_FITS_lensed_z2.pickle',
                            #    './data_files/tau_profiles/L1000N1800/HYDRO_FIDUCIAL/Green/lightcone0/tau_mle_catalogue_nside8192_FITS_lensed_z2_no_ps.pickle',
                               './data_files/tau_profiles/L1000N1800/HYDRO_FIDUCIAL/Green/lightcone0/tau_mle_catalogue_nside8192_FITS_lensed_z3.pickle',
                            #    './data_files/tau_profiles/L1000N1800/HYDRO_FIDUCIAL/Green/lightcone0/tau_mle_catalogue_nside8192_FITS_lensed_z3_no_ps.pickle'
                            ],
                    labels=["CMB unlensed", "CMB lensed to z=2", "CMB lensed to z=3"], #"CMB unlensed (no PS)", "CMB lensed to z=2", "CMB lensed to z=2 (no PS)", "CMB lensed to z=3", "CMB lensed to z=3 (no PS)"],
                    line_styles=["-", ":", "--"], #"*", "-.", "+"],
                    colours=["tab:green" for _ in range(3)], #+ ["darkblue" for _ in range(6)],
                    alpha=[1. for _ in range(3)],
                    plot_title=None,
                    label_title="Green sample",
                    outname='tau_1D_profile_method_comp_full_HYDRO_FIDUCIAL_Green_mle_catalogue_nside8192_FITS.png',
                    observed_data="./data_files/tau_profiles/digitized_obs_data_green.txt",
                    fiducial_index=0,
                    main_ylim=(-0.5, 1.5),
                    ratio_ylim=(-0.5, 0.5),
                    )
    quit()
    
    tp.generic_plot(file_list=['./data_files/tau_profiles/L1000N1800//HYDRO_FIDUCIAL_tau_Mstar_bin10p808_0p259_nside8192_FITS_unlensed_ell_limited.pickle', 
                               './data_files/tau_profiles/L1000N1800//HYDRO_FIDUCIAL_tau_Mstar_bin10p808_0p259_nside8192_FITS_unlensed_no_ps_ell_limited.pickle', 
                               './data_files/tau_profiles/L1000N1800//HYDRO_FIDUCIAL_tau_Mstar_bin10p808_0p259_nside8192_FITS_lensed_z2_ell_limited.pickle', 
                               './data_files/tau_profiles/L1000N1800//HYDRO_FIDUCIAL_tau_Mstar_bin10p808_0p259_nside8192_FITS_lensed_z2_no_ps_ell_limited.pickle', 
                               './data_files/tau_profiles/L1000N1800//HYDRO_FIDUCIAL_tau_Mstar_bin10p808_0p259_nside8192_FITS_lensed_z3_ell_limited.pickle', 
                               './data_files/tau_profiles/L1000N1800//HYDRO_FIDUCIAL_tau_Mstar_bin10p808_0p259_nside8192_FITS_lensed_z3_no_ps_ell_limited.pickle',
                               './data_files/tau_profiles/L1000N1800//HYDRO_FIDUCIAL_tau_Mstar_bin10p808_0p259_nside8192_CAMB_ell_limited.pickle',
                               './data_files/tau_profiles/L1000N1800//HYDRO_FIDUCIAL_tau_Mstar_bin10p808_0p259_nside8192_CAMB_no_ps_ell_limited.pickle',],
                    #labels=["CAMB", "FITS", f"CAMB ($\ell$ limited, no signal)", "CAMB (scalar)", "FITS no signal"],
                    #line_styles=["--", "-", "-.", ":", ":"],
                    #colours=["tab:blue", "tab:red", "tab:green", "tab:orange", "tab:brown"],#for _ in range(4)],
                    #alpha=[1 for _ in range(5)],
                    #plot_title=f'$\\tau$ Profiles for Blue sample, $\log M_*$=10.7 for different\slopes of the z-dependant stellar cut\n(sim=HYDRO_FIDUCIAL, nside=8192)',                        
                    #plot_title=f'$\\tau$ Profiles for Blue sample, $\log M_*$=10.7, $n_{{cut}}$=0.0, for a difference in primary CMB method, \n(sim=HYDRO_FIDUCIAL, nside=8192)',      
                    #outname='HYDRO_FIDUCIAL_cmb_comp_Blue_10p7_0p0_nside8192.png')
                    labels=["unlensed", "unlensed (no PS)", "lensed to z=2", "lensed to z=2 (no PS)", "lensed to z=3", "lensed to z=3 (no PS)", "CAMB", "CAMB (no PS)"],
                    line_styles=["-", ":", "--", "*", "-.", "+", "-", ":"],
                    colours=["tab:blue" for _ in range(6)] + ["darkblue" for _ in range(2)],
                    alpha=[1. for _ in range(8)],
                    plot_title=f'$\\tau$ Profiles for Blue sample, $\log M_*$=10.808, $n_{{cut}}$=0.259, for different lensing methods\n(sim=HYDRO_FIDUCIAL, nside=8192)',
                    outname='HYDRO_FIDUCIAL_method_comp_full_Blue_10p808_0p259_nside8192_FITS_ell_limited.png')
    quit()
    tp.generic_plot(file_list=['./data_files/tau_profiles/L1000N1800//HYDRO_FIDUCIAL_tau_Mstar_bin10p7_0p0_nside8192_CAMB.pickle', './data_files/tau_profiles/L1000N1800//HYDRO_FIDUCIAL_tau_Mstar_bin10p7_0p1_nside8192_CAMB.pickle', './data_files/tau_profiles/L1000N1800//HYDRO_FIDUCIAL_tau_Mstar_bin10p7_0p2_nside8192_CAMB.pickle', './data_files/tau_profiles/L1000N1800//HYDRO_FIDUCIAL_tau_Mstar_bin10p7_0p3_nside8192_CAMB.pickle', './data_files/tau_profiles/L1000N1800//HYDRO_FIDUCIAL_tau_Mstar_bin10p7_0p4_nside8192_CAMB.pickle', './data_files/tau_profiles/L1000N1800//HYDRO_FIDUCIAL_tau_Mstar_bin10p7_0p5_nside8192_CAMB.pickle', './data_files/tau_profiles/L1000N1800//HYDRO_FIDUCIAL_tau_Mstar_bin10p7_0p6_nside8192_CAMB.pickle', './data_files/tau_profiles/L1000N1800//HYDRO_FIDUCIAL_tau_Mstar_bin10p7_0p7_nside8192_CAMB.pickle', './data_files/tau_profiles/L1000N1800//HYDRO_FIDUCIAL_tau_Mstar_bin10p7_0p8_nside8192_CAMB.pickle', './data_files/tau_profiles/L1000N1800//HYDRO_FIDUCIAL_tau_Mstar_bin10p7_0p9_nside8192_CAMB.pickle', './data_files/tau_profiles/L1000N1800//HYDRO_FIDUCIAL_tau_Mstar_bin10p7_1p0_nside8192_CAMB.pickle'],
                    labels=["0.0", "0.1", "0.2" ,"0.3","0.4","0.5","0.6","0.7","0.8","0.9","1.0"],
                    line_styles=["-" for _ in range(11)], 
                    colours=["tab:blue" for _ in range(11)],
                    alpha=[a for a in np.linspace(1.0, 0.1, 11)],
                    #plot_title=f'$\\tau$ Profiles for Blue sample, $\log M_*$=10.7 for different slopes of the z-dependant stellar cut\n(sim=HYDRO_FIDUCIAL, nside=8192)',
                    plot_title=f'$\\tau$ Profiles for Blue sample, $\log M_*$=10.7, for different slopes of a\n z-dependant stellar cut\n(sim=HYDRO_FIDUCIAL, nside=8192, primary CMB=CAMB)',
                    outname='HYDRO_FIDUCIAL_slope_comp_Blue_10p7_nside8192_CAMB.png')              
    quit()
    tp.plot_by_stellar_bin(sim="HYDRO_FIDUCIAL",
                           mass_bin=10.9,
                           colours=["Blue"], #"Green"],# "Red"],
                           file_method="unlensed")
    
    tp.plot_by_colour(sim="HYDRO_JETS_published",
                      colour="Blue",
                      mass_bins=[10.3, 10.4, 10.5, 10.6, 10.7, 10.8, 10.9, 11.0, 11.1, 11.2, 11.3],
                      file_method="unlensed")
    quit()
    tp.plot_by_file_method(sim="HYDRO_FIDUCIAL",
                           colour="Blue",
                           mass_bin=10.7,
                           primary_method="FITS",
                           file_methods=["unlensed", "lensed_z2", "lensed_z3"],
                           )
    quit()
    tp.generic_plot(file_list=['./data_files/tau_profiles/L1000N1800//HYDRO_FIDUCIAL_tau_Mstar_bin11p6_nside8192_FITS_unlensed.pickle', './data_files/tau_profiles/L1000N1800//HYDRO_FIDUCIAL_tau_Mstar_bin11p6_nside8192_FITS_unlensed_no_ps.pickle', './data_files/tau_profiles/L1000N1800//HYDRO_FIDUCIAL_tau_Mstar_bin11p6_nside8192_FITS_lensed_z2.pickle', './data_files/tau_profiles/L1000N1800//HYDRO_FIDUCIAL_tau_Mstar_bin11p6_nside8192_FITS_lensed_z2_no_ps.pickle', './data_files/tau_profiles/L1000N1800//HYDRO_FIDUCIAL_tau_Mstar_bin11p6_nside8192_FITS_lensed_z3.pickle', './data_files/tau_profiles/L1000N1800//HYDRO_FIDUCIAL_tau_Mstar_bin11p6_nside8192_FITS_lensed_z3_no_ps.pickle'],
                    labels=["unlensed", "unlensed (no PS)", "lensed to z=2", "lensed to z=2 (no PS)", "lensed to z=3", "lensed to z=3 (no PS)"],
                    line_styles=["-", ":", "--", "*", "-.", "+"],
                    colours=["tab:blue" for _ in range(6)],
                    alpha=[1. for _ in range(6)],
                    plot_title=f'$\\tau$ Profiles for Blue sample, $\log M_*$=11.6 for different lensing methods\n(sim=HYDRO_FIDUCIAL, nside=8192)',
                    outname='HYDRO_FIDUCIAL_method_comp_full_Blue_11p6_nside8192_FITS.png')
    tp.generic_plot(file_list=['./L1000N1800/Green/HYDRO_FIDUCIAL_tau_Mstar_bin11p6_nside8192_FITS_unlensed.pickle', './L1000N1800/Green/HYDRO_FIDUCIAL_tau_Mstar_bin11p6_nside8192_FITS_unlensed_no_ps.pickle', './L1000N1800/Green/HYDRO_FIDUCIAL_tau_Mstar_bin11p6_nside8192_FITS_lensed_z2.pickle', './L1000N1800/Green/HYDRO_FIDUCIAL_tau_Mstar_bin11p6_nside8192_FITS_lensed_z2_no_ps.pickle', './L1000N1800/Green/HYDRO_FIDUCIAL_tau_Mstar_bin11p6_nside8192_FITS_lensed_z3.pickle', './L1000N1800/Green/HYDRO_FIDUCIAL_tau_Mstar_bin11p6_nside8192_FITS_lensed_z3_no_ps.pickle'],
                    labels=["unlensed", "unlensed (no PS)", "lensed to z=2", "lensed to z=2 (no PS)", "lensed to z=3", "lensed to z=3 (no PS)"],
                    line_styles=["-", ":", "--", "*", "-.", "+"],
                    colours=["tab:green" for _ in range(6)],
                    alpha=[1. for _ in range(6)],
                    plot_title=f'$\\tau$ Profiles for Green sample, $\log M_*$=11.6 for different lensing methods\n(sim=HYDRO_FIDUCIAL, nside=8192)',
                    outname='HYDRO_FIDUCIAL_method_comp_full_Green_11p6_nside8192_FITS.png')
    tp.generic_plot(file_list=['./L1000N1800/Red/HYDRO_FIDUCIAL_tau_Mstar_bin11p6_nside8192_FITS_unlensed.pickle', './L1000N1800/Red/HYDRO_FIDUCIAL_tau_Mstar_bin11p6_nside8192_FITS_unlensed_no_ps.pickle', './L1000N1800/Red/HYDRO_FIDUCIAL_tau_Mstar_bin11p6_nside8192_FITS_lensed_z2.pickle', './L1000N1800/Red/HYDRO_FIDUCIAL_tau_Mstar_bin11p6_nside8192_FITS_lensed_z2_no_ps.pickle', './L1000N1800/Red/HYDRO_FIDUCIAL_tau_Mstar_bin11p6_nside8192_FITS_lensed_z3.pickle', './L1000N1800/Red/HYDRO_FIDUCIAL_tau_Mstar_bin11p6_nside8192_FITS_lensed_z3_no_ps.pickle'],
                    labels=["unlensed", "unlensed (no PS)", "lensed to z=2", "lensed to z=2 (no PS)", "lensed to z=3", "lensed to z=3 (no PS)"],
                    line_styles=["-", ":", "--", "*", "-.", "+"],
                    colours=["tab:red" for _ in range(6)],
                    alpha=[1. for _ in range(6)],
                    plot_title=f'$\\tau$ Profiles for Red sample, $\log M_*$=11.6 for different lensing methods\n(sim=HYDRO_FIDUCIAL, nside=8192)',
                    outname='HYDRO_FIDUCIAL_method_comp_full_Red_11p6_nside8192_FITS.png')
