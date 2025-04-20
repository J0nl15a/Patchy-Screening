import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from matplotlib.colors import LinearSegmentedColormap, to_rgb
import matplotlib.colors as mcolors

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

    def construct_filepath(self, sim, colour, mass_bin, nside=8192, primary_method='FITS', file_method='unlensed', has_signal=True):
        signal_suffix = "" if has_signal else "_no_ps"
        fits_suffix = "" if primary_method != 'FITS' else f"_{file_method}"
        self.stellar_bins = f"{mass_bin:.1f}".replace('.', 'p')
        filename = f"{sim}_tau_Mstar_bin{self.stellar_bins}_nside{nside}_{primary_method}{fits_suffix}{signal_suffix}.pickle"
        filepath = os.path.join(self.base_dir, colour, filename)
        return filepath


    def load_data(self, sim, colour, mass_bin, nside=8192, primary_method='FITS', file_method='unlensed', signal_flag=True):
        # Load the pickle file corresponding to the given parameters.
        filepath = self.construct_filepath(sim, colour, mass_bin, nside, primary_method, file_method, signal_flag)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        # Assume data is a list [theta_d, tau_profile, converted_distance] (adjust if needed)
        return data

    def plot_by_stellar_bin(self, sim, mass_bin, colours, nside=8192, primary_method='FITS', file_method='unlensed', signal_flag=True, noise_data=None):
        # For a fixed stellar bin, plot tau profiles for multiple colours.
        fig, ax = plt.subplots(figsize=(8,6), constrained_layout=True, dpi=1200)
        ax2 = ax.twiny()
        ax.hlines(y=0, xmin=-1, xmax=12, linestyles='-', color='k', label=None)
        line_colours = {'Blue':'tab:blue', 'Green':'tab:green', 'Red':'tab:red'}
        for colour in colours:
            data = self.load_data(sim, colour, mass_bin, nside, primary_method, file_method, signal_flag)
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
        ax.legend(fontsize=10)

        ax2.plot(data[2], data[1], alpha=0)
        ax2.set_xlabel('r [Mpc/h]')

        primary_suffix = "_CAMB" if primary_method=='CAMB' else "_FITS"
        file_suffix = "" if file_method==False else f"_{file_method}"
        signal_suffix = "" if signal_flag==True else "_no_ps"
        noise_suffix = "" if noise_data==None else "_noise"
        plt.savefig(f"./Plots/{sim}_stellar_bin_{self.stellar_bins}_nside{nside}{primary_suffix}{file_suffix}{signal_suffix}{noise_suffix}.png", dpi=1200)
        plt.clf()

    def plot_by_colour(self, sim, colour, mass_bins, nside=8192, primary_method='FITS', file_method='unlensed', signal_flag=True, noise_data=None):
        # For a fixed colour, plot tau profiles for multiple stellar mass bins.
        fig, ax = plt.subplots(figsize=(8,6), constrained_layout=True, dpi=1200)
        ax2 = ax.twiny()
        ax.hlines(y=0, xmin=-1, xmax=12, linestyles='-', color='k', label=None)
        alpha = [.9,.8,.7,.6,.5,.4,.3,.2]
        line_colours = {'Blue':'tab:blue', 'Green':'tab:green', 'Red':'tab:red'}
        base_colour = mcolors.to_rgb(line_colours[colour])
        cmap = LinearSegmentedColormap.from_list("my_colour", [(1,1,1), base_colour], N=256)
        for i, mass_bin in enumerate(mass_bins):
            data = self.load_data(sim, colour, mass_bin, nside, primary_method, file_method, signal_flag)
            theta_d = data[0]
            tau_profile = data[1]
            colour_shade = cmap(alpha[i])
            ax.plot(theta_d, tau_profile, color=colour_shade, label=f"{mass_bin}")
            if noise_data is not None and stellar_bin in noise_data and i==0:
                ax.plot(theta_d, noise_data[stellar_bin], '--', color=line_colours[colour], label=f"{mass_bin} noise")

        formatter = ScalarFormatter(useMathText=True)
        formatter.set_powerlimits((-4, -4))
        ax.yaxis.set_major_formatter(formatter)
        ax.ticklabel_format(axis='y', style='sci', scilimits=(-4, -4))
                
        ax.set_xlabel("Annulus centre (arcmin)")
        ax.set_ylabel(r"Filtered $\tau$")
        ax.set_title(f"$\\tau$ Profiles for unWISE sample = {colour}\n(sim={sim}, nside={nside}, primary CMB={file_method})" if primary_method=='FITS' else f"$\\tau$ Profiles for unWISE sample = {colour}\n(sim={sim}, nside={nside}, primary CMB={primary_method})")
        ax.set_xlim(left=0, right=11)
        ax.legend(fontsize=10)

        ax2.plot(data[2], data[1], alpha=0)
        ax2.set_xlabel('r [Mpc/h]')

        primary_suffix = "_CAMB" if primary_method=='CAMB' else "_FITS"
        file_suffix = "" if file_method==False else f"_{file_method}"
        signal_suffix = "" if signal_flag==True else "_no_ps"
        noise_suffix = "" if noise_data==None else "_noise"
        plt.savefig(f"./Plots/{sim}_sample_{colour}_nside{nside}{primary_suffix}{file_suffix}{signal_suffix}{noise_suffix}.png", dpi=1200)
        plt.clf()

    def plot_by_file_method(self, sim, colour, mass_bin, nside=8192, primary_method='FITS', file_methods='unlensed', signal_flag=True, noise_data=None):
        # For a fixed stellar bin and colour, plot tau profiles for different file method suffixes.
        fig, ax = plt.subplots(figsize=(8,6), constrained_layout=True, dpi=1200)
        ax2 = ax.twiny()
        ax.hlines(y=0, xmin=-1, xmax=12, linestyles='-', color='k', label=None)
        line_colours = {'Blue':'tab:blue', 'Green':'tab:green', 'Red':'tab:red'}
        line_styles = {'unlensed':'-', 'lensed_z2':'--', 'lensed_z3':':'}
        if signal_flag == False:
            for file_method in file_methods:
                data = self.load_data(sim, colour, mass_bin, nside, primary_method, file_method, signal_flag=True)
                theta_d = data[0]
                tau_profile = data[1]
                ax.plot(theta_d, tau_profile, line_styles[file_method], color=line_colours[colour], label=f"{file_method}")
                if noise_data is not None and file_method in noise_data:
                    ax.plot(theta_d, noise_data[file_method], '--', color=line_colours[colour], label=f"{file_method} noise")
            data = self.load_data(sim, colour, mass_bin, nside, primary_method, file_method, signal_flag)
            theta_d = data[0]
            tau_profile = data[1]
            ax.plot(theta_d, tau_profile, '-.', color=line_colours[colour], label=f"{file_method} (no PS)")
        else:
            for file_method in file_methods:
                data = self.load_data(sim, colour, mass_bin, nside, primary_method, file_method, signal_flag)
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
        ax.legend(fontsize=10)

        ax2.plot(data[2], data[1], alpha=0)
        ax2.set_xlabel('r [Mpc/h]')

        primary_suffix = "_CAMB" if primary_method=='CAMB' else "_FITS"
        signal_suffix = "" if signal_flag==True else "_no_ps"
        noise_suffix = "" if noise_data==None else "_noise"
        plt.savefig(f"./Plots/{sim}_method_comp_{colour}_{self.stellar_bins}_nside{nside}{primary_suffix}{signal_suffix}{noise_suffix}.png", dpi=1200)
        plt.clf()

    def generic_plot(self, file_list, labels, line_styles, colours, alpha, plot_title, outname, noise_data=None):
        # file_list: list of file paths to load
        # labels: list of labels corresponding to each file
        fig, ax = plt.subplots(figsize=(8,6), constrained_layout=True, dpi=1200)
        ax2 = ax.twiny()
        ax.hlines(y=0, xmin=-1, xmax=12, linestyles='-', color='k', label=None)
        for fp, lab, style, colour, alpha in zip(file_list, labels, line_styles, colours, alpha):
            with open(fp, 'rb') as f:
                data = pickle.load(f)
            theta_d = data[0]
            tau_profile = data[1]
            ax.plot(theta_d, tau_profile, style, label=lab, color=colour, alpha=alpha)
            if noise_data is not None and lab in noise_data:
                ax.plot(theta_d, noise_data[lab], '--', label=f"{lab} noise")
        formatter = ScalarFormatter(useMathText=True)
        formatter.set_powerlimits((-4, -4))
        ax.yaxis.set_major_formatter(formatter)
        ax.ticklabel_format(axis='y', style='sci', scilimits=(-4, -4))
        
        ax.set_xlabel("Annulus centre (arcmin)")
        ax.set_ylabel(r"$\tau$")
        ax.set_title(plot_title)
        ax.set_xlim(left=0, right=11)
        ax.legend(fontsize=10)

        ax2.plot(data[2], data[1], alpha=0)
        ax2.set_xlabel('r [Mpc/h]')

        plt.savefig(os.path.join("./Plots", outname), dpi=1200)
        plt.clf()


if __name__ == '__main__':

    tp = TauPlotter(base_dir="./L1000N1800")
    tp.plot_by_stellar_bin(sim="HYDRO_FIDUCIAL",
                           mass_bin=10.9,
                           colours=["Blue", "Green", "Red"],
                           file_method="unlensed")
    
    tp.plot_by_colour(sim="HYDRO_FIDUCIAL",
                      colour="Blue",
                      mass_bins=[10.9, 11.0, 11.1, 11.2, 11.3, 11.4, 11.5, 11.6],
                      file_method="unlensed")
    
    tp.plot_by_file_method(sim="HYDRO_FIDUCIAL",
                           colour="Blue",
                           mass_bin=10.9,
                           primary_method="FITS",
                           file_methods=["unlensed"],
                           signal_flag=False)
    
    tp.generic_plot(file_list=['./L1000N1800/Blue/HYDRO_FIDUCIAL_tau_Mstar_bin11p6_nside8192_FITS_unlensed.pickle', './L1000N1800/Blue/HYDRO_FIDUCIAL_tau_Mstar_bin11p6_nside8192_FITS_unlensed_no_ps.pickle', './L1000N1800/Blue/HYDRO_FIDUCIAL_tau_Mstar_bin11p6_nside8192_FITS_lensed_z2.pickle', './L1000N1800/Blue/HYDRO_FIDUCIAL_tau_Mstar_bin11p6_nside8192_FITS_lensed_z2_no_ps.pickle', './L1000N1800/Blue/HYDRO_FIDUCIAL_tau_Mstar_bin11p6_nside8192_FITS_lensed_z3.pickle', './L1000N1800/Blue/HYDRO_FIDUCIAL_tau_Mstar_bin11p6_nside8192_FITS_lensed_z3_no_ps.pickle'],
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
