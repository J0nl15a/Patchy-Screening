import yaml
import numpy as np, pylab as pb, pymaster as nmt
import scipy.interpolate as interpolate

def plot_two_power_spectra(ells_1, spectrum_1, spectrum_1_error, label_1, colour_1, linestyle_1,
                           ells_2, spectrum_2, spectrum_2_error, label_2, colour_2, linestyle_2,
                           ells_3=None, spectrum_3=None, spectrum_3_error=None, label_3=None, colour_3=None, linestyle_3=None,
                           ells_4=None, spectrum_4=None, spectrum_4_error=None, label_4=None, colour_4=None, linestyle_4=None,
                           ells_5=None, spectrum_5=None, spectrum_5_error=None, label_5=None, colour_5=None, linestyle_5=None,
                           title="Power Spectrum Comparison", 
                           ylabel="$C_{\ell}$", 
                           output_path="./power_spectrum_comparison.png"):
    """
    Plot two different power spectra for comparison.
    
    Parameters:
    -----------
    ells_1 : array
        Multipole moments for first spectrum
    spectrum_1 : array
        Power spectrum values for first dataset
    spectrum_1_error : array or None
        Error/covariance for first spectrum (optional)
    label_1 : str
        Label for first spectrum
    
    ells_2 : array
        Multipole moments for second spectrum
    spectrum_2 : array
        Power spectrum values for second dataset
    spectrum_2_error : array or None
        Error/covariance for second spectrum (optional)
    label_2 : str
        Label for second spectrum
    
    title : str
        Title for the plot
    ylabel : str
        Y-axis label
    output_path : str
        Path to save the figure
    """
    
    fig, (ax, ax_ratio) = pb.subplots(
        2, 1, figsize=(8, 7.2),
        sharex=True,
        gridspec_kw={"height_ratios": [3.0, 1.0], "hspace": 0.05},
    )
    
    # Plot first spectrum
    line1, = ax.plot(ells_1, spectrum_1, 
                     linestyle=linestyle_1, alpha=0.8, 
                     marker='o', markersize=4,
                     label=label_1, color=colour_1)
    
    if spectrum_1_error is not None:
        ax.fill_between(x=ells_1, 
                        y1=(spectrum_1 + spectrum_1_error), 
                        y2=(spectrum_1 - spectrum_1_error), 
                        color=line1.get_color(),
                        linewidth=0, alpha=0.3)
    
    # Plot second spectrum
    line2, = ax.plot(ells_2, spectrum_2, 
                     linestyle=linestyle_2, alpha=0.8, 
                     marker='s', markersize=4,
                     label=label_2, color=colour_2)
    
    if spectrum_2_error is not None:
        ax.fill_between(x=ells_2, 
                        y1=(spectrum_2 + spectrum_2_error), 
                        y2=(spectrum_2 - spectrum_2_error), 
                        color=line2.get_color(),
                        linewidth=0, alpha=0.3)

    if ells_3 is not None and spectrum_3 is not None and label_3 is not None:
        # Plot third spectrum
        line3, = ax.plot(ells_3, spectrum_3, 
                        linestyle=linestyle_3, alpha=0.8, 
                        marker='^', markersize=4,
                        label=label_3, color=colour_3)
        
        if spectrum_3_error is not None:
            ax.fill_between(x=ells_3, 
                            y1=(spectrum_3 + spectrum_3_error), 
                            y2=(spectrum_3 - spectrum_3_error), 
                            color=line3.get_color(),
                            linewidth=0, alpha=0.3)
    
    if ells_4 is not None and spectrum_4 is not None and label_4 is not None:
        # Plot fourth spectrum
        line4, = ax.plot(ells_4, spectrum_4, 
                        linestyle=linestyle_4, alpha=0.8, 
                        marker='d', markersize=4,
                        label=label_4, color=colour_4)
        
        if spectrum_4_error is not None:
            ax.fill_between(x=ells_4, 
                            y1=(spectrum_4 + spectrum_4_error), 
                            y2=(spectrum_4 - spectrum_4_error), 
                            color=line4.get_color(),
                            linewidth=0, alpha=0.3)
            
    if ells_5 is not None and spectrum_5 is not None and label_5 is not None:
        # Plot fifth spectrum
        line5, = ax.plot(ells_5, spectrum_5, 
                        linestyle=linestyle_5, alpha=0.8, 
                        marker='x', markersize=4,
                        label=label_5, color=colour_5)
        
        if spectrum_5_error is not None:
            ax.fill_between(x=ells_5, 
                            y1=(spectrum_5 + spectrum_5_error), 
                            y2=(spectrum_5 - spectrum_5_error), 
                            color=line5.get_color(),
                            linewidth=0, alpha=0.3)

    # Plot ratio
    # Interpolate second spectrum to match first spectrum's ells
    # func_interp = interpolate.interp1d(ells_2, spectrum_2, kind='cubic', fill_value="extrapolate")
    # spectrum_2_interp = func_interp(ells_1)
    # ratio = spectrum_1 / spectrum_2_interp
    
    ax_ratio.plot(ells_1, spectrum_1 / spectrum_2, 
                  color=line1.get_color(), marker='o', markersize=4,
                  linestyle=linestyle_1, alpha=0.8)
    
    ax_ratio.plot(ells_1, spectrum_2 / spectrum_2, 
                color=line2.get_color(), marker='o', markersize=4,
                linestyle=linestyle_2, alpha=0.8)

    if ells_3 is not None and spectrum_3 is not None:
        ax_ratio.plot(ells_1, spectrum_3 / spectrum_2, 
                    color=line3.get_color(), marker='o', markersize=4,
                    linestyle=linestyle_3, alpha=0.8)

    if ells_4 is not None and spectrum_4 is not None:
        ax_ratio.plot(ells_1, spectrum_4 / spectrum_2, 
                    color=line4.get_color(), marker='o', markersize=4,
                    linestyle=linestyle_4, alpha=0.8)
        
    if ells_5 is not None and spectrum_5 is not None:
        ax_ratio.plot(ells_1, spectrum_5 / spectrum_2, 
                    color=line5.get_color(), marker='o', markersize=4,
                    linestyle=linestyle_5, alpha=0.8)

    # Format axes
    ax.set_ylabel(ylabel)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(alpha=0.25, which='both')
    ax.legend(fontsize=10, loc='best')
    ax.set_xlabel('Multipole moment $\ell$')
    
    # ax_ratio.axhline(y=1.0, color='k', linestyle='dashed', alpha=0.7)
    # ax_ratio.set_xlabel('Multipole moment $\ell$')
    # ax_ratio.set_ylabel('Ratio (1st / 2nd)')
    # ax_ratio.set_xscale("log")
    # ax_ratio.grid(alpha=0.25, which='both')
    
    ax.set_title(title)
    
    pb.savefig(output_path, dpi=400, bbox_inches='tight')
    pb.close(fig)
    print(f"Figure saved to {output_path}")


# Example usage:
if __name__ == "__main__":
    bin_setup = yaml.safe_load(open("./unWISExLens_lklh/unWISExLens_lklh/config_files/binning_setup.yaml"))
    bin_edges = np.array(bin_setup["Blue_ACT"]["ell_bin_edges"])
    # bin_edges = np.array(bin_setup["Green_ACT"]["ell_bin_edges"])

    edges_int = np.rint(bin_edges).astype(int)  # 19.5->20, 51.5->52, ...
    l0 = edges_int[:-1]
    lf = edges_int[1:]

    b = nmt.NmtBin.from_edges(l0, lf)
    print(l0, lf)

    ells = b.get_effective_ells()
    ell_200_mask = np.where(ells > 200)
    ell_namaster = ells[ell_200_mask]
    ell_1000_mask = np.where(ell_namaster > 1000)

    # Load your data files here
    data_1 = np.loadtxt(f'./unWISExLens_lklh/data/v1.0/bandpowers/unWISExACT-DR6_blue_baseline_Clgg+Clkk+Clkg.dat', usecols=(0,1,2,3))
    data_1_cov = np.loadtxt(f'./unWISExLens_lklh/data/v1.0/covariances/covmat_Clgg+Clkg_unWISExACT-DR6_blue_baseline.dat')
    std = np.sqrt(np.diag(data_1_cov))
    auto_std = std[0:data_1.shape[0]]
    cross_std = std[data_1.shape[0]:]
    ells_1 = data_1[:,0][ell_200_mask]
    spectrum_1 = data_1[:,1][ell_200_mask] * 1e5  # Path to first spectrum values
    spectrum_1_error = auto_std[ell_200_mask] * 1e5  # Optional error for first spectrum

    # data_2 = np.loadtxt(f'./unWISExLens_lklh/data/v1.0/bandpowers/unWISExACT-DR6_blue_baseline_Clgg+Clkk+Clkg.dat', usecols=(0,1,2,3))
    # ells_2 = data_2[:,0]  # Path to second spectrum ells
    # spectrum_1 = data_1[:,3][ell_200_mask] * 1e5  # Path to second spectrum values
    # spectrum_1_error = cross_std[ell_200_mask] * 1e5  # Optional error for second spectrum

    # data_2 = np.loadtxt(f'./data_files/power_spectra/galaxy_galaxy/L1000N1800/HYDRO_FIDUCIAL/Blue/lightcone0/galaxy_galaxy_power_spectrum_10p815_0p138.txt', skiprows=1, usecols=(0,2))
    # ells_2 = data_2[:,0]
    # spectrum_2 = data_2[:,1]  # Path to second spectrum values
    # spectrum_2_error = None  # Optional error for second spectrum
    
    # data_3 = np.loadtxt(f'./data_files/power_spectra/galaxy_galaxy/L1000N3600/HYDRO_FIDUCIAL/Blue/lightcone0/galaxy_galaxy_power_spectrum_10p735_0p711.txt', skiprows=1, usecols=(0,2))
    # ells_3 = data_3[:,0]
    # spectrum_3 = data_3[:,1]  # Path to third spectrum values
    # spectrum_3_error = None  # Optional error for third spectrum

    data_2 = np.loadtxt(f'./data_files/power_spectra/galaxy_galaxy/L1000N1800/HYDRO_FIDUCIAL/Blue/lightcone0/galaxy_galaxy_power_spectrum_10p808_0p259.txt', skiprows=1, usecols=(0,2))
    ells_2 = data_2[:,0]
    spectrum_2 = data_2[:,1]  # Path to second spectrum values
    spectrum_2_error = None  # Optional error for second spectrum
    
    # data_3 = np.loadtxt(f'./data_files/power_spectra/galaxy_galaxy/L1000N3600/HYDRO_FIDUCIAL/Blue/lightcone0/galaxy_galaxy_power_spectrum_10p808_0p259.txt', skiprows=1, usecols=(0,2))
    # ells_3 = data_3[:,0]
    # spectrum_3 = data_3[:,1]  # Path to third spectrum values
    # spectrum_3_error = None  # Optional error for third spectrum

    data_3 = np.loadtxt(f'./data_files/power_spectra/galaxy_galaxy/L1000N1800/HYDRO_FIDUCIAL/Blue/lightcone0/galaxy_galaxy_power_spectrum_0p000_0p0.txt', skiprows=1, usecols=(0,2))
    ells_3 = data_3[:,0]
    spectrum_3 = data_3[:,1]  # Path to third spectrum values
    spectrum_3_error = None  # Optional error for third spectrum

    data_4 = np.loadtxt(f'./data_files/power_spectra/galaxy_galaxy/L1000N1800/HYDRO_LOW_SIGMA8/Blue/lightcone0/galaxy_galaxy_power_spectrum_0p000_0p0.txt', skiprows=1, usecols=(0,2))
    ells_4 = data_4[:,0]
    spectrum_4 = data_4[:,1]  # Path to fourth spectrum values
    spectrum_4_error = None  # Optional error for fourth spectrum

    data_5 = np.loadtxt(f'./data_files/power_spectra/galaxy_galaxy/L1000N1800/HYDRO_PLANCK/Blue/lightcone0/galaxy_galaxy_power_spectrum_0p000_0p0.txt', skiprows=1, usecols=(0,2))
    ells_5 = data_5[:,0]
    spectrum_5 = data_5[:,1]  # Path to fifth spectrum values
    spectrum_5_error = None  # Optional error for fifth spectrum

    plot_two_power_spectra(
        ells_1, spectrum_1, spectrum_1_error, "Obs auto data",
        ells_2, spectrum_2, spectrum_2_error, label_2="Simulated data (L1000N1800, FIDUCIAL, MLE cut)",
        ells_3=ells_3, spectrum_3=spectrum_3, spectrum_3_error=spectrum_3_error, label_3="Simulated data (L1000N1800, FIDUCIAL, abundance matched)",
        ells_4=ells_4, spectrum_4=spectrum_4, spectrum_4_error=spectrum_4_error, label_4="Simulated data (L1000N1800, LOW_SIGMA8, abundance matched)",
        ells_5=ells_5, spectrum_5=spectrum_5, spectrum_5_error=spectrum_5_error, label_5="Simulated data (L1000N1800, PLANCK, abundance matched)",
        title="Power Spectrum Comparison (MLE cut vs abundance matching)",
        ylabel="$C_{\ell}$",
        output_path="./Plots/power_spectrum_comparison.png"
    )
    
    pass
