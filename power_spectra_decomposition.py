import sys, yaml
import numpy as np, pylab as pb, pymaster as nmt, healpy as hp
from imp_patchy_screening import patchyScreening
from kappa_map_gen_forJonah import kappa_map_gen_forJonah
from scipy.signal import savgol_filter
from pathlib import Path

w=5
p=2

ncpu = int(sys.argv[1])
box = str(sys.argv[2])
isim = str(sys.argv[3])
iz = str(sys.argv[4])
lightcone = int(sys.argv[5])

file_ext = str(sys.argv[6])
use_full_mean_density = sys.argv[7].lower() in ("true", "1", "yes", "y") if len(sys.argv) > 7 else False

density_suffix = "_fullmean" if use_full_mean_density else ""

def name_float(x):
    return f"{float(x):.3f}".replace(".", "p")


def select_halos(halos, column, lo, hi, log10=False):
    values = halos[column].to_numpy()

    if log10:
        values = np.log10(values)

    mask = (values >= lo) & (values < hi)
    return halos.filter(mask)


def compute_subset_spectra(halos, label, f_galaxy_full, mean_density_override=None):
    nhalos = len(halos)

    if nhalos == 0:
        print(f"{label}: no halos, skipping")
        return None, None, nhalos

    f_galaxy_subset, density_map_subset, mean_density_subset, nhalos_subset = make_galaxy_field(
        halos,
        mean_density_override=mean_density_override,
    )

    # gg decomposition: subset galaxy field x full catalogue galaxy field
    pcl_gg = nmt.compute_coupled_cell(f_galaxy_subset, f_galaxy_full)

    pcl_shape = (
        f_galaxy_subset.nmaps * f_galaxy_full.nmaps,
        f_galaxy_subset.ainfo.lmax + 1,
    )
    clg = np.zeros(pcl_shape)

    deproj_gg = nmt.deprojection_bias(f_galaxy_subset, f_galaxy_full, clg)

    w_gg = nmt.NmtWorkspace.from_fields(f_galaxy_subset, f_galaxy_full, b)
    cl_gg_namaster = w_gg.decouple_cell(pcl_gg - deproj_gg).squeeze()[ell_200_mask]

    gg_low = cl_gg_namaster[np.where(ell_namaster <= 1000)]
    gg_high = (
        savgol_filter(cl_gg_namaster[ell_1000_mask], window_length=w, polyorder=p)
        if w > 0 and p > 0
        else cl_gg_namaster[ell_1000_mask]
    )

    # Do not subtract ordinary auto shot noise here: this is subset x full, not subset x subset.
    auto_power = np.concatenate((gg_low, gg_high)) * 1e5

    # kg decomposition: kappa x subset galaxy field
    pcl_cross = nmt.compute_coupled_cell(f_kappa, f_galaxy_subset)

    pcl_shape = (f_kappa.nmaps * f_galaxy_subset.nmaps, f_galaxy_subset.ainfo.lmax + 1)
    clg = np.zeros(pcl_shape)

    deproj_cross = nmt.deprojection_bias(f_kappa, f_galaxy_subset, clg)

    w_cross = nmt.NmtWorkspace.from_fields(f_kappa, f_galaxy_subset, b)
    cl_cross_namaster = w_cross.decouple_cell(pcl_cross - deproj_cross).squeeze()[ell_200_mask]

    cross_low = cl_cross_namaster[np.where(ell_namaster <= 1000)]
    cross_high = (
        savgol_filter(cl_cross_namaster[ell_1000_mask], window_length=w, polyorder=p)
        if w > 0 and p > 0
        else cl_cross_namaster[ell_1000_mask]
    )

    cross_power = np.concatenate((cross_low, cross_high)) * 1e5

    print(f"{label}: nhalos = {nhalos}")

    return auto_power, cross_power, nhalos


def make_galaxy_field(halos, mean_density_override=None):
    nhalos = len(halos)

    vec = np.zeros((nhalos, 3), dtype=float)
    vec[:, 0] = halos["xminpot"].to_numpy()
    vec[:, 1] = halos["yminpot"].to_numpy()
    vec[:, 2] = halos["zminpot"].to_numpy()

    theta, phi = hp.pixelfunc.vec2ang(vec, lonlat=True)
    source_vector = hp.ang2vec(theta, phi, lonlat=True)

    pixels = hp.pixelfunc.vec2pix(
        nside_cl,
        source_vector[:, 0],
        source_vector[:, 1],
        source_vector[:, 2],
    )

    density_map = np.bincount(pixels, minlength=npix).astype(np.float64)

    if mean_density_override is None:
        mean_density = np.mean(density_map)
    else:
        mean_density = mean_density_override

    overdensity = (density_map - mean_density) / mean_density
    mask = overdensity * 0.0 + 1.0

    field = nmt.NmtField(mask, [overdensity], lmax=lmax_bins, n_iter=0)

    return field, density_map, mean_density, nhalos


def plot_decomposition(results_family, family, spectra_type, full_spectrum, all_bins):
    fig, ax = pb.subplots(1, 1, figsize=(8, 6))

    cmap = pb.get_cmap("tab10")

    bin_colours = {
        (lo, hi): cmap(i)
        for i, (lo, hi) in enumerate(all_bins)
    }

    for entry in results_family:
        lo = entry["lo"]
        hi = entry["hi"]
    
        label = f"{entry['lo']:.1f}–{entry['hi']:.1f}, N={entry['nhalos']}"
        ax.plot(
            ell_namaster,
            entry[spectra_type],
            linestyle="--",
            color=bin_colours[(lo, hi)],
            label=label,
        )

    ax.plot(
        ell_namaster,
        full_spectrum,
        color="k",
        linewidth=1.5,
        label="Full spectrum",
    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(200, 4000)

    if spectra_type == "auto":
        ax.set_ylabel(r"$C_\ell^{gg} \times 10^5$")
        ax.set_ylim(bottom=1e-5)
        prefix = "gg"
    else:
        ax.set_ylabel(r"$C_\ell^{\kappa g} \times 10^5$")
        ax.set_ylim(bottom=1e-7)
        prefix = "kg"

    ax.set_xlabel(r"Multipole moment $\ell$")
    ax.legend(fontsize=6, loc="best", title=f"{family.capitalize()} bins (stellar cut = {m_cut}, {s_cut})", title_fontsize=6)

    outname = (
        f"./Plots/halo_map_{prefix}_power_spectrum_"
        f"{box}_{isim}_{iz}_{family}_decomposition{density_suffix}.{file_ext}"
    )

    Path("./Plots").mkdir(exist_ok=True)
    pb.savefig(outname, dpi=400, bbox_inches="tight")
    pb.close(fig)

if iz == 'Blue':
    mean_z = 0.6
    shell = 12
elif iz == 'Green':
    mean_z = 1.1
    shell = 22

m_cut = np.loadtxt(f'./data_files/mle_parameters/{box}/{isim}/{iz}/lightcone{lightcone}/mle_values.txt', skiprows=6, usecols=1, max_rows=1, delimiter='=')
s_cut = np.loadtxt(f'./data_files/mle_parameters/{box}/{isim}/{iz}/lightcone{lightcone}/mle_values.txt', skiprows=7, usecols=1, max_rows=1, delimiter='=')
m_cut_name = name_float(m_cut)
s_cut_name = name_float(s_cut)

# ps = patchyScreening(box, isim, shell, float(m_cut), float(s_cut), ncpu, lightcone_method=('FULL', 'shell'), lightcone=lightcone, mle=False)
ps = patchyScreening(box, isim, iz, float(m_cut), float(s_cut), ncpu, lightcone=lightcone, mle=False)
ps.filter_stellar_mass()

decomposition_bins = {
    "redshift": {
        "column": "z",
        "bins": [(0.0, 0.5), (0.5, 1.0), (1.0, 1.5)],
        "log10": False,
    },
    "stellar_mass": {
        "column": "mstar",
        "bins": [(10.0, 10.5), (10.5, 11.0), (11.0, 11.5), (11.5, 12.0), (12.0, 12.5)],
        "log10": True,
    },
    "halo_mass": {
        "column": "mvir",
        "bins": [(10.5, 11.5), (11.5, 12.5), (12.5, 13.5), (13.5, 14.5), (14.5, 15.5)],
        "log10": True,
    },
}

# mean_z_halos = ps.merge.filter((ps.merge['z'] >= (mean_z-0.05)) & (ps.merge['z'] < (mean_z+0.05)))

# print(len(mean_z_halos))

bin_setup = yaml.safe_load(open("./unWISExLens_lklh/unWISExLens_lklh/config_files/binning_setup.yaml"))
if iz == 'Blue':
    bin_edges = np.array(bin_setup["Blue_ACT"]["ell_bin_edges"])
elif iz == 'Green':
    bin_edges = np.array(bin_setup["Green_ACT"]["ell_bin_edges"])
print(bin_edges)

edges_int = np.rint(bin_edges).astype(int)  # 19.5->20, 51.5->52, ...
l0 = edges_int[:-1]
lf = edges_int[1:]
b = nmt.NmtBin.from_edges(l0, lf)
lmax_bins = b.lmax

ells = b.get_effective_ells()
ell_200_mask = np.where(ells > 200)
ell_namaster = ells[ell_200_mask]
print(ell_namaster)
ell_1000_mask = np.where(ell_namaster > 1000)

if iz == 'Blue':
    obs_nbar_sq_deg = 3409
elif iz == 'Green':
    obs_nbar_sq_deg = 1846

obs_nbar_sr = obs_nbar_sq_deg * ((180/np.pi)**2)

print(obs_nbar_sr)

obs_shot_noise = 1/obs_nbar_sr #Shot-noise is 1/(source number density per steradian)
print(obs_shot_noise)

nside_cl = 2048
npix = hp.nside2npix(nside_cl)

try:
    kappa_map = hp.read_map(f'./data_files/kappa_maps/{box}/{isim}/lightcone{lightcone}/kappa_nonrot.fits', dtype=np.float64, verbose=False)
except FileNotFoundError:
    kappa_map = kappa_map_gen_forJonah(box, isim, lightcone)
kappa_map = hp.pixelfunc.ud_grade(kappa_map, nside_cl)


kappa_mask = kappa_map*0.0+1.0
f_kappa = nmt.NmtField(kappa_mask, [kappa_map], lmax=lmax_bins, n_iter=0)

f_galaxy_full, density_map_full, mean_density_full, nhalos_full = make_galaxy_field(ps.merge)
print("Full-catalogue mean density:", mean_density_full)

results = {}

for family, cfg in decomposition_bins.items():
    results[family] = []

    for lo, hi in cfg["bins"]:
        label = f"{family}_{lo:.1f}_{hi:.1f}".replace(".", "p")

        halos_bin = select_halos(
            ps.merge,
            column=cfg["column"],
            lo=lo,
            hi=hi,
            log10=cfg["log10"],
        )

        mean_density_override = mean_density_full if use_full_mean_density else None

        auto_spec, cross_spec, nhalos = compute_subset_spectra(
            halos_bin,
            label,
            f_galaxy_full,
            mean_density_override=mean_density_override,
        )

        if auto_spec is None:
            continue

        results[family].append({
            "lo": lo,
            "hi": hi,
            "label": label,
            "nhalos": nhalos,
            "auto": auto_spec,
            "cross": cross_spec,
        })

full_auto_power_spectra = np.loadtxt(f'./data_files/power_spectra/galaxy_galaxy/{box}/{isim}/{iz}/lightcone{lightcone}/galaxy_galaxy_power_spectrum_{m_cut_name}_{s_cut_name}.txt', skiprows=1, usecols=2)
full_cross_power_spectra = np.loadtxt(f'./data_files/power_spectra/kappa_galaxy/{box}/{isim}/{iz}/lightcone{lightcone}/kappa_galaxy_power_spectrum_{m_cut_name}_{s_cut_name}.txt', skiprows=1, usecols=1)

for family, entries in results.items():
    all_bins = decomposition_bins[family]["bins"]

    plot_decomposition(entries, family, "auto", full_auto_power_spectra, all_bins)
    plot_decomposition(entries, family, "cross", full_cross_power_spectra, all_bins)

