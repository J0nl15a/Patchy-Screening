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
        suffix = "" if has_signal else "_no_ps"
        self.stellar_bins = {10.9: ('10p9', 'bin0'), 11.0: ('11p0', 'bin1'), 11.1: ('11p1', 'bin2'), 11.2: ('11p2', 'bin3'), 11.3: ('11p3', 'bin4'), 11.4: ('11p4', 'bin5'), 11.5: ('11p5', 'bin6'), 11.6: ('11p6', 'bin7')}
        filename = f"{sim}_tau_Mstar_{self.stellar_bins[mass_bin][1]}_nside{nside}_{primary_method}_{file_method}{suffix}.pickle"
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
        ax.set_title(f"$\\tau$ Profiles for $\log M_*$ = {mass_bin}\n(sim={sim}, nside={nside}, primary CMB={file_method})" if primary_method=='FITS' else f"\\tau Profiles for log stellar mass = {mass_bin}\n(sim={sim}, nside={nside}, primary CMB={primary_method})")
        ax.set_xlim(left=0, right=11)
        ax.legend(fontsize=10)

        ax2.plot(data[2], data[1], alpha=0)
        ax2.set_xlabel('r [Mpc/h]')

        primary_suffix = "_CAMB" if primary_method=='CAMB' else "_FITS"
        file_suffix = "" if file_method==False else f"_{file_method}"
        signal_suffix = "" if signal_flag==True else "_no_ps"
        noise_suffix = "" if noise_data==None else "_noise"
        plt.savefig(f"./Plots/{sim}_stellar_bin_{self.stellar_bins[mass_bin][0]}_nside{nside}{primary_suffix}{file_suffix}{signal_suffix}{noise_suffix}.png", dpi=1200)
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
        ax.set_title(f"$\\tau$ Profiles for unWISE sample = {colour}\n(sim={sim}, nside={nside}, primary CMB={file_method})" if primary_method=='FITS' else f"\\tau Profiles for unWISE sample = {colour}\n(sim={sim}, nside={nside}, primary CMB={primary_method})")
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
        ax.set_ylabel(r"$\tau$")
        ax.set_title(f"$\\tau$ Profiles for {colour} sample, {mass_bin} for different lensing methods\n(sim={sim}, nside={nside})")
        ax.set_xlim(left=0, right=11)
        ax.legend(fontsize=10)

        ax2.plot(data[2], data[1], alpha=0)
        ax2.set_xlabel('r [Mpc/h]')

        primary_suffix = "_CAMB" if primary_method=='CAMB' else "_FITS"
        signal_suffix = "" if signal_flag==True else "_no_ps"
        noise_suffix = "" if noise_data==None else "_noise"
        plt.savefig(f"./Plots/{sim}_method_comp_{colour}_{self.stellar_bins[mass_bin][0]}_nside{nside}{primary_suffix}{signal_suffix}{noise_suffix}.png", dpi=1200)
        plt.clf()

    def generic_plot(self, file_list, labels, plot_title, xlabel="Angular Bin (arcmin)", ylabel="Tau", noise_data=None):
        # file_list: list of file paths to load
        # labels: list of labels corresponding to each file
        plt.figure(figsize=(8,6))
        for fp, lab in zip(file_list, labels):
            with open(fp, 'rb') as f:
                data = pickle.load(f)
            theta_d = data[0]
            tau_profile = data[1]
            plt.plot(theta_d, tau_profile, label=lab)
            if noise_data is not None and lab in noise_data:
                plt.plot(theta_d, noise_data[lab], '--', label=f"{lab} noise")
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(plot_title)
        plt.legend()
        outname = plot_title.replace(" ", "_") + ".png"
        plt.savefig(os.path.join("./Plots", outname), dpi=1200)
        plt.clf()


if __name__ == '__main__':

    tp = TauPlotter(base_dir="./L1000N1800")
    tp.plot_by_stellar_bin(sim="HYDRO_FIDUCIAL",
                           mass_bin=10.9,
                           colours=["Blue", "Green", "Red"])
    tp.plot_by_file_method(sim="HYDRO_FIDUCIAL",
                           colour="Green",
                           mass_bin=11.0,
                           primary_method="FITS",
                           file_methods=["unlensed", "lensed_z2"], signal_flag=False)


