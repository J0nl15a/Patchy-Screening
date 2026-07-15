import sys, yaml
import numpy as np, pylab as pb, pymaster as nmt, healpy as hp, pandas as pd
from imp_patchy_screening import patchyScreening
from kappa_map_gen_forJonah import kappa_map_gen_forJonah
from scipy.signal import savgol_filter
from pathlib import Path

w = 5
p = 2

def name_float(x):
    return f"{float(x):.3f}".replace(".", "p")


def setup_binning(iz):
    bin_setup = yaml.safe_load(open("./unWISExLens_lklh/unWISExLens_lklh/config_files/binning_setup.yaml"))

    if iz == "Blue":
        bin_edges = np.array(bin_setup["Blue_ACT"]["ell_bin_edges"])
    elif iz == "Green":
        bin_edges = np.array(bin_setup["Green_ACT"]["ell_bin_edges"])
    else:
        raise ValueError("iz must be Blue or Green")

    edges_int = np.rint(bin_edges).astype(int)
    b = nmt.NmtBin.from_edges(edges_int[:-1], edges_int[1:])
    ells = b.get_effective_ells()

    ell_200_mask = np.where(ells > 200)
    ell_namaster = ells[ell_200_mask]
    ell_1000_mask = np.where(ell_namaster > 1000)

    return b, ell_200_mask, ell_namaster, ell_1000_mask


def halos_to_source_vector(halos):
    n = len(halos)

    vec = np.zeros((n, 3), dtype=float)
    vec[:, 0] = halos["xminpot"].to_numpy()
    vec[:, 1] = halos["yminpot"].to_numpy()
    vec[:, 2] = halos["zminpot"].to_numpy()

    theta, phi = hp.pixelfunc.vec2ang(vec, lonlat=True)
    return hp.ang2vec(theta, phi, lonlat=True)


def compute_spectra_from_halos(
    halos,
    f_kappa,
    b,
    ell_200_mask,
    ell_namaster,
    ell_1000_mask,
    nside_cl,
    npix,
    lmax_bins,
    obs_shot_noise,
):
    nhalos = len(halos)

    if nhalos == 0:
        raise ValueError("Empty halo catalogue passed to spectra function.")

    source_vector = halos_to_source_vector(halos)

    pixels = hp.pixelfunc.vec2pix(
        nside_cl,
        source_vector[:, 0],
        source_vector[:, 1],
        source_vector[:, 2],
    )

    density_map = np.bincount(pixels, minlength=npix).astype(np.float64)
    mean_density = np.mean(density_map)

    galaxy_overdensity = (density_map - mean_density) / mean_density

    galaxy_mask = galaxy_overdensity * 0.0 + 1.0
    f_galaxy = nmt.NmtField(galaxy_mask, [galaxy_overdensity], lmax=lmax_bins, n_iter=0)

    # auto
    pcl_auto = nmt.compute_coupled_cell(f_galaxy, f_galaxy)

    pcl_shape = (f_galaxy.nmaps * f_galaxy.nmaps, f_galaxy.ainfo.lmax + 1)
    clg = np.zeros(pcl_shape)
    deproj_auto = nmt.deprojection_bias(f_galaxy, f_galaxy, clg)

    w_auto = nmt.NmtWorkspace.from_fields(f_galaxy, f_galaxy, b)
    cl_auto = w_auto.decouple_cell(pcl_auto - deproj_auto).squeeze()[ell_200_mask]

    auto_shot_noise = (4 * np.pi) / nhalos

    auto_low = cl_auto[np.where(ell_namaster <= 1000)]
    auto_high = (
        savgol_filter(cl_auto[ell_1000_mask], window_length=w, polyorder=p)
        if w > 0 and p > 0
        else cl_auto[ell_1000_mask]
    )

    auto_power = ((np.concatenate((auto_low, auto_high)) - auto_shot_noise) + obs_shot_noise) * 1e5

    # cross
    pcl_cross = nmt.compute_coupled_cell(f_kappa, f_galaxy)

    pcl_shape = (f_kappa.nmaps * f_galaxy.nmaps, f_galaxy.ainfo.lmax + 1)
    clg = np.zeros(pcl_shape)
    deproj_cross = nmt.deprojection_bias(f_kappa, f_galaxy, clg)

    w_cross = nmt.NmtWorkspace.from_fields(f_kappa, f_galaxy, b)
    cl_cross = w_cross.decouple_cell(pcl_cross - deproj_cross).squeeze()[ell_200_mask]

    cross_low = cl_cross[np.where(ell_namaster <= 1000)]
    cross_high = (
        savgol_filter(cl_cross[ell_1000_mask], window_length=w, polyorder=p)
        if w > 0 and p > 0
        else cl_cross[ell_1000_mask]
    )

    cross_power = np.concatenate((cross_low, cross_high)) * 1e5

    return auto_power, cross_power


def main():
    ncpu = int(sys.argv[1])
    box_fid = str(sys.argv[2])          # e.g. L1000N1800
    isim = str(sys.argv[3])
    iz = str(sys.argv[4])
    lightcone = int(sys.argv[5])
    file_ext = str(sys.argv[6])

    box_hires = "L1000N3600"
    s_cut = 0.0

    if iz == "Blue":
        shell = 12
        obs_nbar_sq_deg = 3409
    elif iz == "Green":
        shell = 22
        obs_nbar_sq_deg = 1846
    else:
        raise ValueError("iz must be Blue or Green")

    obs_nbar_sr = obs_nbar_sq_deg * ((180 / np.pi) ** 2)
    obs_shot_noise = 1 / obs_nbar_sr

    m_cuts = np.round(np.arange(10.3, 11.3 + 0.1, 0.1), 1)

    b, ell_200_mask, ell_namaster, ell_1000_mask = setup_binning(iz)
    lmax_bins = b.lmax

    nside_cl = 2048
    npix = hp.nside2npix(nside_cl)

    try:
        kappa_map = hp.read_map(
            f"./data_files/kappa_maps/{box_fid}/{isim}/lightcone{lightcone}/kappa_nonrot.fits",
            dtype=np.float64,
            verbose=False,
        )

        kappa_map_hires = hp.read_map(
            f"./data_files/kappa_maps/{box_hires}/{isim}/lightcone{lightcone}/kappa_nonrot.fits",
            dtype=np.float64,
            verbose=False,
        )
    except FileNotFoundError:
        kappa_map = kappa_map_gen_forJonah(box_fid, isim, lightcone)
        kappa_map_hires = kappa_map_gen_forJonah(box_hires, isim, lightcone)
        

    kappa_map = hp.pixelfunc.ud_grade(kappa_map, nside_cl)
    kappa_mask = kappa_map * 0.0 + 1.0
    f_kappa = nmt.NmtField(kappa_mask, [kappa_map], lmax=lmax_bins, n_iter=0)

    kappa_map_hires = hp.pixelfunc.ud_grade(kappa_map_hires, nside_cl)
    kappa_mask_hires = kappa_map_hires * 0.0 + 1.0
    f_kappa_hires = nmt.NmtField(kappa_mask_hires, [kappa_map_hires], lmax=lmax_bins, n_iter=0)

    matched_min_mstar = []
    fid_abundances = []

    auto_fid_list = []
    auto_hires_list = []
    cross_fid_list = []
    cross_hires_list = []

    ps_fid = patchyScreening(
        box_fid,
        isim,
        shell,
        0.0,
        float(s_cut),
        ncpu,
        lightcone_method=("FULL", "shell"),
        lightcone=lightcone,
        mle=False,
    )
    ps_fid.filter_stellar_mass()

    ps_hires = patchyScreening(
        box_hires,
        isim,
        shell,
        0.0,
        float(s_cut),
        ncpu,
        lightcone_method=("FULL", "shell"),
        lightcone=lightcone,
        mle=False,
    )
    ps_hires.filter_stellar_mass()

    fid_ranked = ps_fid.merge.sort_values("mstar", ascending=False)
    hires_ranked = ps_hires.merge.sort_values("mstar", ascending=False)

    for m_cut in m_cuts:
        print(f"Processing m_cut={m_cut:.1f}")

        fid_ranked_cut = fid_ranked[fid_ranked['mstar'] > (10**m_cut)]

        abundance = len(fid_ranked_cut)

        hires_matched = hires_ranked[:abundance]

        min_hires_mstar = np.log10(hires_matched["mstar"].min())

        fid_abundances.append(abundance)
        matched_min_mstar.append(min_hires_mstar)

        print(
            f"m_cut={m_cut:.1f}, abundance={abundance}, "
            f"log10(min mstar hires)={min_hires_mstar:.4f}"
        )

        auto_fid, cross_fid = compute_spectra_from_halos(
            fid_ranked_cut,
            f_kappa,
            b,
            ell_200_mask,
            ell_namaster,
            ell_1000_mask,
            nside_cl,
            npix,
            lmax_bins,
            obs_shot_noise,
        )

        auto_hires, cross_hires = compute_spectra_from_halos(
            hires_matched,
            f_kappa_hires,
            b,
            ell_200_mask,
            ell_namaster,
            ell_1000_mask,
            nside_cl,
            npix,
            lmax_bins,
            obs_shot_noise,
        )

        auto_fid_list.append(auto_fid)
        auto_hires_list.append(auto_hires)
        cross_fid_list.append(cross_fid)
        cross_hires_list.append(cross_hires)

    Path("./Plots").mkdir(exist_ok=True)

    # abundance-match relation
    fig, ax = pb.subplots(1, 1, figsize=(7, 5))

    ax.plot(m_cuts, matched_min_mstar, marker="o", color="k")
    ax.plot(m_cuts, m_cuts, linestyle="--", color="grey", alpha=0.7, label="1:1")

    ax.set_xlabel(r"Fiducial stellar cut amplitude, $\log_{10} M_*$")
    ax.set_ylabel(r"High-res abundance-matched minimum $\log_{10} M_*$")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    outname = f"./Plots/abundance_matched_hires_mstar_{box_fid}_{box_hires}_{isim}_{iz}.{file_ext}"
    pb.savefig(outname, dpi=400, bbox_inches="tight")
    pb.close(fig)

    # spectra plots
    cmap = pb.get_cmap("viridis")
    colors = [cmap(i / max(1, len(m_cuts) - 1)) for i in range(len(m_cuts))]

    fig, ax = pb.subplots(1, 1, figsize=(8, 6))

    for i, m_cut in enumerate(m_cuts):
        ax.plot(
            ell_namaster,
            auto_fid_list[i],
            color=colors[i],
            linestyle="-",
            label=f"{m_cut:.1f} fid" if i in (0, len(m_cuts)-1) else None,
        )
        ax.plot(
            ell_namaster,
            auto_hires_list[i],
            color=colors[i],
            linestyle="--",
            label=f"{m_cut:.1f} hires" if i in (0, len(m_cuts)-1) else None,
        )

    sm = pb.cm.ScalarMappable(cmap=cmap, norm=pb.Normalize(vmin=m_cuts.min(), vmax=m_cuts.max()))
    cbar = pb.colorbar(sm, ax=ax)
    cbar.set_label(r"Fiducial stellar cut amplitude")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(200, 4000)
    ax.set_ylim(bottom=1e-2)
    ax.set_xlabel(r"Multipole moment $\ell$")
    ax.set_ylabel(r"$C_\ell^{gg} \times 10^5$")
    ax.legend(fontsize=7, loc="best")

    outname = f"./Plots/abundance_matched_gg_spectra_{box_fid}_{box_hires}_{isim}_{iz}.{file_ext}"
    pb.savefig(outname, dpi=400, bbox_inches="tight")
    pb.close(fig)

    fig, ax = pb.subplots(1, 1, figsize=(8, 6))

    for i, m_cut in enumerate(m_cuts):
        ax.plot(
            ell_namaster,
            cross_fid_list[i],
            color=colors[i],
            linestyle="-",
            label=f"{m_cut:.1f} fid" if i in (0, len(m_cuts)-1) else None,
        )
        ax.plot(
            ell_namaster,
            cross_hires_list[i],
            color=colors[i],
            linestyle="--",
            label=f"{m_cut:.1f} hires" if i in (0, len(m_cuts)-1) else None,
        )

    sm = pb.cm.ScalarMappable(cmap=cmap, norm=pb.Normalize(vmin=m_cuts.min(), vmax=m_cuts.max()))
    cbar = pb.colorbar(sm, ax=ax)
    cbar.set_label(r"Fiducial stellar cut amplitude")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(200, 4000)
    ax.set_ylim(bottom=1e-4)
    ax.set_xlabel(r"Multipole moment $\ell$")
    ax.set_ylabel(r"$C_\ell^{\kappa g} \times 10^5$")
    ax.legend(fontsize=7, loc="best")

    outname = f"./Plots/abundance_matched_kg_spectra_{box_fid}_{box_hires}_{isim}_{iz}.{file_ext}"
    pb.savefig(outname, dpi=400, bbox_inches="tight")
    pb.close(fig)

def non_parametric_stellar_cut():
    catalogue = pd.read_parquet(f"./data_files/mock_halo_catalogs/L1000N1800/HYDRO_FIDUCIAL/Blue/lightcone0/sampled_halo_data_0p000_0p000.parquet")
    halo_redshifts = np.loadtxt(f"./data_files/halo_redshifts/L1000N1800/HYDRO_FIDUCIAL/lightcone0/FLAMINGO_halo_redshift_values.txt", usecols=(1, 2, 3))
    z_min = halo_redshifts[:, 0]
    z_max = halo_redshifts[:, 2]
    z_mid = halo_redshifts[:, 1]

    min_stellar_masses = []
    for i in range(len(halo_redshifts)):
        if z_mid[i] > 3.0:
            break

        in_shell = (catalogue['zminpot'] >= z_min[i]) & (catalogue['zminpot'] < z_max[i])
        shell_halos = catalogue[in_shell]

        min_stellar_mass = shell_halos['mstar'].min()
        min_stellar_masses.append(min_stellar_mass)
        print(f"Shell {i}: z_min={z_min[i]:.3f}, z_max={z_max[i]:.3f}, z_mid={z_mid[i]:.3f}, min mstar={min_stellar_mass:.3e}")

    # abundance-match relation
    fig, ax = pb.subplots(1, 1, figsize=(7, 5))

    ax.plot(z_mid[:len(min_stellar_masses)], np.log10(min_stellar_masses), marker="o", color="k")
    # ax.plot(m_cuts, m_cuts, linestyle="--", color="grey", alpha=0.7, label="1:1")

    ax.set_xlabel(r"Redshift, $z$")
    ax.set_ylabel(r"Minimum stellar mass $\log_{10} M_*$")
    ax.set_ylim(bottom=10.3, top=11.3)
    ax.grid(alpha=0.3)
    # ax.legend(fontsize=8)

    outname = f"./Plots/non_parametric_stellar_cut_L1000N1800_HYDRO_FIDUCIAL_Blue.png"
    pb.savefig(outname, dpi=400, bbox_inches="tight")
    pb.close(fig)


if __name__ == "__main__":
    non_parametric_stellar_cut()
    quit()
    main()

# ncpu = int(sys.argv[1])
# box = str(sys.argv[2])
# isim = str(sys.argv[3])
# iz = str(sys.argv[4])
# lightcone = int(sys.argv[5])

# file_ext = str(sys.argv[6])

# if iz == 'Blue':
#     mean_z = 0.6
#     shell = 12
# elif iz == 'Green':
#     mean_z = 1.1
#     shell = 22

# # m_cut = np.loadtxt(f'./data_files/mle_parameters/{box}/{isim}/{iz}/lightcone{lightcone}/mle_values.txt', skiprows=6, usecols=1, max_rows=1, delimiter='=')
# # s_cut = np.loadtxt(f'./data_files/mle_parameters/{box}/{isim}/{iz}/lightcone{lightcone}/mle_values.txt', skiprows=7, usecols=1, max_rows=1, delimiter='=')
# m_cut = 10.3
# s_cut = 0.0
# m_cut_name = name_float(m_cut)
# s_cut_name = name_float(s_cut)

# ps = patchyScreening(box, isim, shell, float(m_cut), float(s_cut), ncpu, lightcone_method=('FULL', 'shell'), lightcone=lightcone)
# # ps = patchyScreening(box, isim, iz, float(m_cut), float(s_cut), ncpu, lightcone=lightcone)
# ps.filter_stellar_mass()

# rank_ordered_halos = ps.merge.sort(by='mstar', descending=True)
# abundance = len(rank_ordered_halos)
# print(rank_ordered_halos['mstar'][:25], rank_ordered_halos[:25])

# ps_hires = patchyScreening('L1000N3600', isim, shell, 0.0, float(s_cut), ncpu, lightcone_method=('FULL', 'shell'), lightcone=lightcone)
# ps_hires.filter_stellar_mass()

# rank_ordered_halos_hires = ps_hires.merge.sort(by='mstar', descending=True)[:abundance]
# print(len(rank_ordered_halos_hires), rank_ordered_halos_hires[:25])
# print(rank_ordered_halos_hires['mstar'].min(), np.log10(rank_ordered_halos_hires['mstar'].min()))

# rows, cols = (len(mean_z_halos), 3)
# vec = [[0]*cols]*rows
# vec=1.0*np.asarray(vec)
# vec[:,0]=mean_z_halos['xminpot'].to_numpy()
# vec[:,1]=mean_z_halos['yminpot'].to_numpy()
# vec[:,2]=mean_z_halos['zminpot'].to_numpy()
# theta, phi = hp.pixelfunc.vec2ang(vec, lonlat=True)
# source_vector = hp.ang2vec(theta, phi, lonlat=True)

# bin_setup = yaml.safe_load(open("./unWISExLens_lklh/unWISExLens_lklh/config_files/binning_setup.yaml"))
# if iz == 'Blue':
#     bin_edges = np.array(bin_setup["Blue_ACT"]["ell_bin_edges"])
# elif iz == 'Green':
#     bin_edges = np.array(bin_setup["Green_ACT"]["ell_bin_edges"])
# print(bin_edges)

# edges_int = np.rint(bin_edges).astype(int)  # 19.5->20, 51.5->52, ...
# l0 = edges_int[:-1]
# lf = edges_int[1:]
# b = nmt.NmtBin.from_edges(l0, lf)
# lmax_bins = b.lmax

# ells = b.get_effective_ells()
# ell_200_mask = np.where(ells > 200)
# ell_namaster = ells[ell_200_mask]
# print(ell_namaster)
# ell_1000_mask = np.where(ell_namaster > 1000)

# if iz == 'Blue':
#     obs_nbar_sq_deg = 3409
# elif iz == 'Green':
#     obs_nbar_sq_deg = 1846

# obs_nbar_sr = obs_nbar_sq_deg * ((180/np.pi)**2)

# print(obs_nbar_sr)

# obs_shot_noise = 1/obs_nbar_sr #Shot-noise is 1/(source number density per steradian)
# print(obs_shot_noise)

# nside_cl = 2048
# npix = hp.nside2npix(nside_cl)

# try:
#     kappa_map = hp.read_map(f'./data_files/kappa_maps/{box}/{isim}/lightcone{lightcone}/kappa_nonrot.fits', dtype=np.float64, verbose=False)
# except FileNotFoundError:
#     kappa_map = kappa_map_gen_forJonah(box, isim, lightcone)
# kappa_map = hp.pixelfunc.ud_grade(kappa_map, nside_cl)


# kappa_mask = kappa_map*0.0+1.0
# f_kappa = nmt.NmtField(kappa_mask, [kappa_map], lmax=lmax_bins, n_iter=0)

# pixels = hp.pixelfunc.vec2pix(nside_cl, source_vector[:,0], source_vector[:,1], source_vector[:,2])
# density_map = np.bincount(pixels, minlength=npix).astype(np.float64)
# mean_density = np.mean(density_map)
# galaxy_overdensity = ((density_map - mean_density) / mean_density)

# galaxy_mask = galaxy_overdensity*0.0+1.0
# f_galaxy = nmt.NmtField(galaxy_mask, [galaxy_overdensity], lmax=lmax_bins, n_iter=0)

# pcl_auto = nmt.compute_coupled_cell(f_galaxy, f_galaxy)

# pcl_shape = (f_galaxy.nmaps * f_galaxy.nmaps, f_galaxy.ainfo.lmax+1)
# clg = np.zeros(pcl_shape)
# deproj_auto = nmt.deprojection_bias(f_galaxy, f_galaxy, clg)

# w_auto = nmt.NmtWorkspace.from_fields(f_galaxy, f_galaxy, b)
# cl_auto_namaster = w_auto.decouple_cell(pcl_auto - deproj_auto).squeeze()[ell_200_mask]

# auto_spectra_shot_noise = (4*np.pi)/len(mean_z_halos) #Shot-noise is 4pi/(number of sources in the map)
# print((4*np.pi)/len(mean_z_halos))

# auto_power_spectra_unsmoothed_component = cl_auto_namaster[np.where(ell_namaster <= 1000)]
# auto_power_spectra_smooth_component = savgol_filter(cl_auto_namaster[ell_1000_mask], window_length=w, polyorder=p) if w > 0 and p > 0 else cl_auto_namaster[ell_1000_mask]
# auto_power_spectra = ((np.concatenate((auto_power_spectra_unsmoothed_component, auto_power_spectra_smooth_component)) - auto_spectra_shot_noise) + obs_shot_noise) * 1e5

# pcl_cross = nmt.compute_coupled_cell(f_kappa, f_galaxy)

# pcl_shape = (f_kappa.nmaps * f_galaxy.nmaps, f_galaxy.ainfo.lmax+1)
# clg = np.zeros(pcl_shape)
# deproj_cross = nmt.deprojection_bias(f_kappa, f_galaxy, clg)

# w_cross = nmt.NmtWorkspace.from_fields(f_kappa, f_galaxy, b)
# cl_cross_namaster = w_cross.decouple_cell(pcl_cross - deproj_cross).squeeze()[ell_200_mask]

# cross_power_spectra_unsmoothed_component = cl_cross_namaster[np.where(ell_namaster <= 1000)]
# cross_power_spectra_smooth_component = savgol_filter(cl_cross_namaster[ell_1000_mask], window_length=w, polyorder=p) if w > 0 and p > 0 else cl_cross_namaster[ell_1000_mask]
# cross_power_spectra = np.concatenate((cross_power_spectra_unsmoothed_component, cross_power_spectra_smooth_component)) * 1e5

# full_auto_power_spectra = np.loadtxt(f'./data_files/power_spectra/galaxy_galaxy/{box}/{isim}/{iz}/lightcone{lightcone}/galaxy_galaxy_power_spectrum_{m_cut_name}_{s_cut_name}.txt', skiprows=1, usecols=2)
# full_cross_power_spectra = np.loadtxt(f'./data_files/power_spectra/kappa_galaxy/{box}/{isim}/{iz}/lightcone{lightcone}/kappa_galaxy_power_spectrum_{m_cut_name}_{s_cut_name}.txt', skiprows=1, usecols=1)


# fig, ax = pb.subplots(1, 1, figsize=(8,6))

# line, = ax.plot(ell_namaster, auto_power_spectra, linestyle='--', color="#117733", label=f'Mean z shell ({iz} sample)')
# line, = ax.plot(ell_namaster, full_auto_power_spectra, color="#117733", label=f'Full auto power spectrum ({iz} sample)')
# # ax_ratio.plot(ell, spec / fid, color=line.get_color(), alpha=0.9)

# # ax_ratio.axhline(1.0, color="k", linestyle="--", alpha=0.6)

# ax.set_xscale("log")
# ax.set_yscale("log")
# ax.set_xlim(200, 4000)

# # ax_ratio.set_xscale("log")
# # ax_ratio.set_xlim(200, 4000)
# # ax_ratio.set_ylim(*ratio_ylim)
# # ax_ratio.grid(alpha=0.25)


# ylabel = r"$C_\ell^{gg} \times 10^5$"
# bottom = 1e-2
# ratio_ylim = (0.9, 1.1)
# outname = f"./Plots/halo_map_gg_power_spectrum_{box}_{isim}_{iz}_mean_z_shell.{file_ext}"

# ax.set_ylabel(ylabel)
# ax.set_ylim(bottom=bottom)
# ax.legend(fontsize=6, loc="best")

# # ax_ratio.set_ylabel("Ratio")
# # ax_ratio.set_xlabel(r"Multipole moment $\ell$")

# Path("./Plots").mkdir(exist_ok=True)
# pb.savefig(outname, dpi=400, bbox_inches="tight")
# pb.close(fig)


# fig, ax = pb.subplots(1, 1, figsize=(8,6))

# line, = ax.plot(ell_namaster, cross_power_spectra, linestyle='--', color="#117733", label=f'Mean z shell ({iz} sample)')
# line, = ax.plot(ell_namaster, full_cross_power_spectra, color="#117733", label=f'Full cross power spectrum ({iz} sample)')
# # ax_ratio.plot(ell, spec / fid, color=line.get_color(), alpha=0.9)

# # ax_ratio.axhline(1.0, color="k", linestyle="--", alpha=0.6)

# ax.set_xscale("log")
# ax.set_yscale("log")
# ax.set_xlim(200, 4000)

# # ax_ratio.set_xscale("log")
# # ax_ratio.set_xlim(200, 4000)
# # ax_ratio.set_ylim(*ratio_ylim)
# # ax_ratio.grid(alpha=0.25)

# ylabel = r"$C_\ell^{\kappa g} \times 10^5$"
# bottom = 1e-4
# ratio_ylim = (0.5, 1.5)
# outname = f"./Plots/halo_map_kg_power_spectrum_{box}_{isim}_{iz}_mean_z_shell.{file_ext}"

# ax.set_ylabel(ylabel)
# ax.set_ylim(bottom=bottom)
# ax.legend(fontsize=6, loc="best")

# # ax_ratio.set_ylabel("Ratio")
# # ax_ratio.set_xlabel(r"Multipole moment $\ell$")

# Path("./Plots").mkdir(exist_ok=True)
# pb.savefig(outname, dpi=400, bbox_inches="tight")
# pb.close(fig)
