import yaml
import numpy as np, pylab as pb, pymaster as nmt
import scipy.interpolate as interpolate
from scipy.signal import savgol_filter

def plot_two_power_spectra(spectrum_1, spectrum_1_error, label_1, colour_1, linestyle_1,
                           spectrum_2, spectrum_2_error, label_2, colour_2, linestyle_2,
                           spectrum_3=None, spectrum_3_error=None, label_3=None, colour_3=None, linestyle_3=None,
                           spectrum_4=None, spectrum_4_error=None, label_4=None, colour_4=None, linestyle_4=None,
                           spectrum_5=None, spectrum_5_error=None, label_5=None, colour_5=None, linestyle_5=None,
                           spectrum_6=None, spectrum_6_error=None, label_6=None, colour_6=None, linestyle_6=None,
                           spectrum_7=None, spectrum_7_error=None, label_7=None, colour_7=None, linestyle_7=None,
                           spectrum_8=None, spectrum_8_error=None, label_8=None, colour_8=None, linestyle_8=None,
                           output_path="./Plots/power_spectrum_comparison_paper_ready_auto.pdf"):
    """
    Plot two different power spectra for comparison.
    
    Parameters:
    -----------
    spectrum_1 : array
        Power spectrum values for first dataset
    spectrum_1_error : array or None
        Error/covariance for first spectrum (optional)
    label_1 : str
        Label for first spectrum
    
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

    obs_data_ACT = np.loadtxt(f'./unWISExLens_lklh/data/v1.0/bandpowers/unWISExACT-DR6_blue_baseline_Clgg+Clkk+Clkg.dat', usecols=(0,1,2,3))
    obs_data_Planck = np.loadtxt(f'./unWISExLens_lklh/data/v1.0/bandpowers/unWISExPlanck-PR4_blue_baseline_Clgg+Clkk+Clkg.dat', usecols=(0,1,2,3))
    Planck_ell_mask = np.where(obs_data_Planck[:,0] > 200)[0]

    obs_data_ACT_cov = np.loadtxt(f'./unWISExLens_lklh/data/v1.0/covariances/covmat_Clgg+Clkg_unWISExACT-DR6_blue_baseline.dat')
    obs_data_Planck_covariance = np.loadtxt(f'./unWISExLens_lklh/data/v1.0/covariances/covmat_Clgg+Clkg_unWISExPlanck-PR4_blue_baseline.dat')

    obs_data_ACT_auto = obs_data_ACT[:,1][ell_200_mask] * 1e5
    obs_data_ACT_cross = obs_data_ACT[:,3][ell_200_mask] * 1e5
    obs_data_Planck_auto = obs_data_Planck[:,1][Planck_ell_mask] * 1e5
    obs_data_Planck_cross = obs_data_Planck[:,3][Planck_ell_mask] * 1e5

    std_ACT = np.sqrt(np.diag(obs_data_ACT_cov))
    auto_std_ACT = std_ACT[0:obs_data_ACT.shape[0]][ell_200_mask] * 1e5
    cross_std_ACT = std_ACT[obs_data_ACT.shape[0]:][ell_200_mask] * 1e5
    std_Planck = np.sqrt(np.diag(obs_data_Planck_covariance))
    auto_std_Planck = std_Planck[0:obs_data_Planck.shape[0]][Planck_ell_mask] * 1e5
    cross_std_Planck = std_Planck[obs_data_Planck.shape[0]:][Planck_ell_mask] * 1e5

    spectrum_1 = smooth_spectrum(spectrum_1, ell_namaster, window_size=5, polyorder=2)
    spectrum_2 = smooth_spectrum(spectrum_2, ell_namaster, window_size=5, polyorder=2)
    if spectrum_3 is not None:
        spectrum_3 = smooth_spectrum(spectrum_3, ell_namaster, window_size=5, polyorder=2)
    if spectrum_4 is not None:
        spectrum_4 = smooth_spectrum(spectrum_4, ell_namaster, window_size=5, polyorder=2)
    if spectrum_5 is not None:
        spectrum_5 = smooth_spectrum(spectrum_5, ell_namaster, window_size=5, polyorder=2)
    if spectrum_6 is not None:
        spectrum_6 = smooth_spectrum(spectrum_6, ell_namaster, window_size=5, polyorder=2)
    if spectrum_7 is not None:
        spectrum_7 = smooth_spectrum(spectrum_7, ell_namaster, window_size=5, polyorder=2)
    if spectrum_8 is not None:
        spectrum_8 = smooth_spectrum(spectrum_8, ell_namaster, window_size=5, polyorder=2)
    
    fig, (ax, ax_ratio) = pb.subplots(
        2, 1, figsize=(8, 7.2),
        sharex=True,
        gridspec_kw={"height_ratios": [3.0, 1.0], "hspace": 0.05},
    )

    # Plot observed spectrum
    line_ACT, = ax.plot(ell_namaster, obs_data_ACT_auto, 
                     linestyle='solid', alpha=0.8, linewidth=0, 
                     marker='.', markersize=5,
                     label='ACT x unWISE', color='k')
    
    ax.fill_between(x=ell_namaster, 
                    y1=(obs_data_ACT_auto+auto_std_ACT), y2=(obs_data_ACT_auto-auto_std_ACT), 
                    color='k', linewidth=0, alpha=.3)
    
    line_Planck, = ax.plot(obs_data_Planck[:,0][Planck_ell_mask], obs_data_Planck_auto, 
                     linestyle='solid', alpha=0.8, linewidth=0, 
                     marker='.', markersize=5,
                     label='Planck x unWISE', color='r')
    
    ax.fill_between(x=obs_data_Planck[:,0][Planck_ell_mask], 
                    y1=(obs_data_Planck_auto+auto_std_Planck), y2=(obs_data_Planck_auto-auto_std_Planck), 
                    color='r', linewidth=0, alpha=.3)
    
    
    # Plot first spectrum
    line1, = ax.plot(ell_namaster, spectrum_1, 
                     linestyle=linestyle_1, alpha=0.8, 
                     label=label_1, color=colour_1)
    
    if spectrum_1_error is not None:
        ax.fill_between(x=ell_namaster, 
                        y1=(spectrum_1 + spectrum_1_error), 
                        y2=(spectrum_1 - spectrum_1_error), 
                        color=line1.get_color(),
                        linewidth=0, alpha=0.3)
    
    # Plot second spectrum
    line2, = ax.plot(ell_namaster, spectrum_2, 
                     linestyle=linestyle_2, alpha=0.8, 
                     label=label_2, color=colour_2)
    
    if spectrum_2_error is not None:
        ax.fill_between(x=ell_namaster, 
                        y1=(spectrum_2 + spectrum_2_error), 
                        y2=(spectrum_2 - spectrum_2_error), 
                        color=line2.get_color(),
                        linewidth=0, alpha=0.3)

    if spectrum_3 is not None and label_3 is not None:
        # Plot third spectrum
        line3, = ax.plot(ell_namaster, spectrum_3, 
                        linestyle=linestyle_3, alpha=0.8, 
                        label=label_3, color=colour_3)
        
        if spectrum_3_error is not None:
            ax.fill_between(x=ell_namaster, 
                            y1=(spectrum_3 + spectrum_3_error), 
                            y2=(spectrum_3 - spectrum_3_error), 
                            color=line3.get_color(),
                            linewidth=0, alpha=0.3)
    
    if spectrum_4 is not None and label_4 is not None:
        # Plot fourth spectrum
        line4, = ax.plot(ell_namaster, spectrum_4, 
                        linestyle=linestyle_4, alpha=0.8, 
                        label=label_4, color=colour_4)
        
        if spectrum_4_error is not None:
            ax.fill_between(x=ell_namaster, 
                            y1=(spectrum_4 + spectrum_4_error), 
                            y2=(spectrum_4 - spectrum_4_error), 
                            color=line4.get_color(),
                            linewidth=0, alpha=0.3)
            
    if spectrum_5 is not None and label_5 is not None:
        # Plot fifth spectrum
        line5, = ax.plot(ell_namaster, spectrum_5, 
                        linestyle=linestyle_5, alpha=0.8, 
                        label=label_5, color=colour_5)
        
        if spectrum_5_error is not None:
            ax.fill_between(x=ell_namaster, 
                            y1=(spectrum_5 + spectrum_5_error), 
                            y2=(spectrum_5 - spectrum_5_error), 
                            color=line5.get_color(),
                            linewidth=0, alpha=0.3)
            
    if spectrum_6 is not None and label_6 is not None:
        # Plot sixth spectrum
        line6, = ax.plot(ell_namaster, spectrum_6, 
                        linestyle=linestyle_6, alpha=0.8, 
                        label=label_6, color=colour_6)
        
        if spectrum_6_error is not None:
            ax.fill_between(x=ell_namaster, 
                            y1=(spectrum_6 + spectrum_6_error), 
                            y2=(spectrum_6 - spectrum_6_error), 
                            color=line6.get_color(),
                            linewidth=0, alpha=0.3)
            
    if spectrum_7 is not None and label_7 is not None:
        # Plot seventh spectrum
        line7, = ax.plot(ell_namaster, spectrum_7, 
                        linestyle=linestyle_7, alpha=0.8, 
                        label=label_7, color=colour_7)
        
        if spectrum_7_error is not None:
            ax.fill_between(x=ell_namaster, 
                            y1=(spectrum_7 + spectrum_7_error), 
                            y2=(spectrum_7 - spectrum_7_error), 
                            color=line7.get_color(),
                            linewidth=0, alpha=0.3)
            
    if spectrum_8 is not None and label_8 is not None:
        # Plot eighth spectrum
        line8, = ax.plot(ell_namaster, spectrum_8, 
                        linestyle=linestyle_8, alpha=0.8, 
                        label=label_8, color=colour_8)
        
        if spectrum_8_error is not None:
            ax.fill_between(x=ell_namaster, 
                            y1=(spectrum_8 + spectrum_8_error), 
                            y2=(spectrum_8 - spectrum_8_error), 
                            color=line8.get_color(),
                            linewidth=0, alpha=0.3)

    # Plot ratio of observed to first spectrum

    ratio_act_obs = obs_data_ACT_auto / spectrum_1
    ax_ratio.plot(
        ell_namaster, ratio_act_obs,
        marker='.', markersize=5,
        color='k', linewidth=0,
        alpha=0.9
    )
    # simple propagation assuming +/- is 1σ on the model only (reference treated fixed)
    ratio_hi = (obs_data_ACT_auto + auto_std_ACT) / spectrum_1
    ratio_lo = (obs_data_ACT_auto - auto_std_ACT) / spectrum_1
    ax_ratio.fill_between(
        x=ell_namaster, y1=ratio_hi, y2=ratio_lo,
        linewidth=0, alpha=0.3, color='k'
    )

    ratio_planck_obs = obs_data_Planck_auto / spectrum_1[:len(obs_data_Planck[:,0][Planck_ell_mask])]
    ax_ratio.plot(
        obs_data_Planck[:,0][Planck_ell_mask], ratio_planck_obs,
        marker='.', markersize=5,
        color='r', linewidth=0,
        alpha=0.9
    )
    ratio_planck_hi = (obs_data_Planck_auto + auto_std_Planck) / spectrum_1[:len(obs_data_Planck[:,0][Planck_ell_mask])]
    ratio_planck_lo = (obs_data_Planck_auto - auto_std_Planck) / spectrum_1[:len(obs_data_Planck[:,0][Planck_ell_mask])]
    ax_ratio.fill_between(
        x=obs_data_Planck[:,0][Planck_ell_mask], y1=ratio_planck_hi, y2=ratio_planck_lo,
        linewidth=0, alpha=0.3, color='r'
    )

    
    ax_ratio.plot(ell_namaster, spectrum_1 / spectrum_1, 
                  color=line1.get_color(),
                  linestyle=line1.get_linestyle(), alpha=0.9)
    
    ax_ratio.plot(ell_namaster, spectrum_2 / spectrum_1, 
                color=line2.get_color(),
                linestyle=line2.get_linestyle(), alpha=0.9)

    if spectrum_3 is not None:
        ax_ratio.plot(ell_namaster, spectrum_3 / spectrum_1, 
                    color=line3.get_color(),
                    linestyle=line3.get_linestyle(), alpha=0.9)

    if spectrum_4 is not None:
        ax_ratio.plot(ell_namaster, spectrum_4 / spectrum_1, 
                    color=line4.get_color(),
                    linestyle=line4.get_linestyle(), alpha=0.9)
        
    if spectrum_5 is not None:
        ax_ratio.plot(ell_namaster, spectrum_5 / spectrum_1, 
                    color=line5.get_color(), 
                    linestyle=line5.get_linestyle(), alpha=0.9)
        
    if spectrum_6 is not None:
        ax_ratio.plot(ell_namaster, spectrum_6 / spectrum_1, 
                    color=line6.get_color(), 
                    linestyle=line6.get_linestyle(), alpha=0.9)
        
    if spectrum_7 is not None:
        ax_ratio.plot(ell_namaster, spectrum_7 / spectrum_1, 
                    color=line7.get_color(), 
                    linestyle=line7.get_linestyle(), alpha=0.9)
        
    if spectrum_8 is not None:
        ax_ratio.plot(ell_namaster, spectrum_8 / spectrum_1, 
                    color=line8.get_color(), 
                    linestyle=line8.get_linestyle(), alpha=0.9)

    # Format axes
    ax.set_ylabel(r'$C^{\rm gg}_{\mathrm{\ell}}x10^5$')
    # ax.set_ylabel(r'$C^{\kappa \rm g}_{\mathrm{\ell}}x10^5$')
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(200, 4000)
    ax.set_ylim(bottom=1e-2)
    # ax.set_ylim(bottom=1e-4, top=4e-2)

    ax_ratio.axhline(y=1.0, color='k', linestyle='dashed', alpha=0.7)
    ax_ratio.set_xlabel('Multipole moment $\mathrm{\ell}$')
    ax_ratio.set_ylabel('Model / Fiducial')
    ax_ratio.set_xscale("log")
    ax_ratio.set_xlim(200, 4000)
    ax_ratio.set_ylim(0.8, 1.2)
    # ax_ratio.set_ylim(0.5, 1.5)
    ax_ratio.grid(alpha=0.25)

    ax.legend(title="Simulation", fontsize=9, ncols=1, loc='upper right')
    
    pb.savefig(output_path, dpi=400)
    pb.close(fig)
    print(f"Figure saved to {output_path}")


def smooth_spectrum(spectrum, ell, window_size=5, polyorder=2):
    """
    Smooth the input spectrum using a simple moving average.

    Parameters:
    -----------
    spectrum : array
        Input power spectrum to be smoothed.
    window_size : int
        Size of the moving average window. Must be an odd integer.

    Returns:
    --------
    smoothed_spectrum : array
        Smoothed power spectrum.
    """
    power_spectra_unsmoothed_component = spectrum[np.where(ell <= 1000)]
    
    power_spectra_smooth_component = savgol_filter(spectrum[np.where(ell > 1000)], 
                                                   window_length=window_size, polyorder=polyorder) if window_size > 0 and polyorder > 0 else spectrum[np.where(ell <= 1000)]
    
    power_spectra_smoothed = np.concatenate((power_spectra_unsmoothed_component, power_spectra_smooth_component))

    return power_spectra_smoothed


# Example usage:
if __name__ == "__main__":

    # Load your data files here

    data_1 = np.loadtxt(f'./data_files/power_spectra/galaxy_galaxy/L1000N1800/HYDRO_FIDUCIAL/Blue/lightcone0/galaxy_galaxy_power_spectrum_10p799_0p506.txt', skiprows=1, usecols=(0,2))
    ell_namaster = data_1[:,0]
    spectrum_1 = data_1[:,1]  # Path to first spectrum values
    spectrum_1_error = None  # Optional error for first spectrum

    data_2 = np.loadtxt(f'./data_files/power_spectra/galaxy_galaxy/L1000N1800/HYDRO_PLANCK/Blue/lightcone0/galaxy_galaxy_power_spectrum_10p801_0p585_non_rotated.txt', skiprows=1, usecols=(0,2))
    ell_namaster = data_2[:,0]
    spectrum_2 = data_2[:,1]  # Path to second spectrum values
    spectrum_2_error = None  # Optional error for second spectrum

    data_3 = np.loadtxt(f'./data_files/power_spectra/galaxy_galaxy/L1000N1800/HYDRO_LOW_SIGMA8/Blue/lightcone0/galaxy_galaxy_power_spectrum_10p730_0p887.txt', skiprows=1, usecols=(0,2))
    ell_namaster = data_3[:,0]
    spectrum_3 = data_3[:,1]  # Path to third spectrum values
    spectrum_3_error = None  # Optional error for third spectrum

    data_4 = np.loadtxt(f'./data_files/power_spectra/galaxy_galaxy/L1000N3600/HYDRO_FIDUCIAL/Blue/lightcone0/galaxy_galaxy_power_spectrum_10p679_0p972.txt', skiprows=1, usecols=(0,2))
    ell_namaster = data_4[:,0]
    spectrum_4 = data_4[:,1]  # Path to fourth spectrum values
    spectrum_4_error = None  # Optional error for fourth spectrum

    data_5 = np.loadtxt(f'./data_files/power_spectra/galaxy_galaxy/L1000N1800/HYDRO_FIDUCIAL/Blue/lightcone0/galaxy_galaxy_power_spectrum_0p0_0p0.txt', skiprows=1, usecols=(0,2))
    ell_namaster = data_5[:,0]
    spectrum_5 = data_5[:,1]  # Path to fifth spectrum values
    spectrum_5_error = None  # Optional error for fifth spectrum

    data_6 = np.loadtxt(f'./data_files/power_spectra/galaxy_galaxy/L1000N1800/HYDRO_PLANCK/Blue/lightcone0/galaxy_galaxy_power_spectrum_0p0_0p0_non_rotated.txt', skiprows=1, usecols=(0,2))
    ell_namaster = data_6[:,0]
    spectrum_6 = data_6[:,1]  # Path to sixth spectrum values
    spectrum_6_error = None  # Optional error for sixth spectrum

    data_7 = np.loadtxt(f'./data_files/power_spectra/galaxy_galaxy/L1000N1800/HYDRO_LOW_SIGMA8/Blue/lightcone0/galaxy_galaxy_power_spectrum_0p0_0p0.txt', skiprows=1, usecols=(0,2))
    ell_namaster = data_7[:,0]
    spectrum_7 = data_7[:,1]  # Path to seventh spectrum values
    spectrum_7_error = None  # Optional error for seventh spectrum

    data_8 = np.loadtxt(f'./data_files/power_spectra/galaxy_galaxy/L1000N3600/HYDRO_FIDUCIAL/Blue/lightcone0/galaxy_galaxy_power_spectrum_0p0_0p0.txt', skiprows=1, usecols=(0,2))
    ell_namaster = data_8[:,0]
    spectrum_8 = data_8[:,1]  # Path to eighth spectrum values
    spectrum_8_error = None  # Optional error for eighth spectrum



    plot_two_power_spectra(
        spectrum_1=spectrum_1, spectrum_1_error=spectrum_1_error, label_1="L1_m9 (MLE cut)",            colour_1='#117733', linestyle_1="solid",
        spectrum_2=spectrum_2, spectrum_2_error=spectrum_2_error, label_2="Planck (MLE cut)",           colour_2='#44AA99', linestyle_2="solid",
        spectrum_3=spectrum_3, spectrum_3_error=spectrum_3_error, label_3="LS8 (MLE cut)",              colour_3='#882255', linestyle_3="solid",
        spectrum_4=spectrum_4, spectrum_4_error=spectrum_4_error, label_4="L1_m8 (MLE cut)",            colour_4='#CC6677', linestyle_4="solid",
        spectrum_5=spectrum_5, spectrum_5_error=spectrum_5_error, label_5="L1_m9 (Abundance matched)",  colour_5='#117733', linestyle_5="dashed",
        spectrum_6=spectrum_6, spectrum_6_error=spectrum_6_error, label_6="Planck (Abundance matched)", colour_6='#44AA99', linestyle_6="dashed",
        spectrum_7=spectrum_7, spectrum_7_error=spectrum_7_error, label_7="LS8 (Abundance matched)",    colour_7='#882255', linestyle_7="dashed",
        spectrum_8=spectrum_8, spectrum_8_error=spectrum_8_error, label_8="L1_m8 (Abundance matched)",  colour_8='#CC6677', linestyle_8="dashed",
        output_path="./Plots/power_spectrum_comparison_paper_ready_auto.png"
    )
    
    pass
