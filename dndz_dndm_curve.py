import sys
import pylab as pb, numpy as np, pandas as pd, h5py
from imp_patchy_screening import patchyScreening
from pathlib import Path
from scipy.integrate import simpson
# from plothist import make_hist, plot_error_hist, plot_hist
pb.rcParams['font.family'] = 'serif'

box = str(sys.argv[2])
isim = str(sys.argv[3])
iz = str(sys.argv[4])
lc = int(sys.argv[5])
m_cut = float(np.loadtxt(f"./data_files/mle_parameters/{box}/{isim}/{iz}/lightcone{lc}/mle_values.txt", usecols=1, skiprows=6, max_rows=1, delimiter='='))
s_cut = float(np.loadtxt(f"./data_files/mle_parameters/{box}/{isim}/{iz}/lightcone{lc}/mle_values.txt", usecols=1, skiprows=7, max_rows=1, delimiter='='))
# m_cut = 11.0
# s_cut = 0.4
m_cut_name = f"{m_cut:.3f}".replace(".", "p")
s_cut_name = f"{s_cut:.3f}".replace(".", "p")
print(m_cut, s_cut)
if iz == "Blue":
    z_mean = 0.6
elif iz == "Green":
    z_mean = 1.1

plot_all = str(sys.argv[6])
plot_suffix = str(sys.argv[7])
both_samples = sys.argv[8].lower() in ("true", "1", "yes", "y")

redshift_bins = np.loadtxt(f"/cosma8/data/dp004/dc-conl1/FLAMINGO/patchy_screening/data_files/halo_redshifts/{box}/{isim}/lightcone{lc}/FLAMINGO_halo_redshift_values.txt", usecols=(0,1,2,3))
redshift_midpoints = redshift_bins[:,2]
obs_dndz = np.loadtxt(f"/cosma8/data/dp004/dc-conl1/FLAMINGO/patchy_screening/unWISExLens_lklh/data/v1.0/aux_data/dndz/unWISE_{iz.lower()}_xmatch_dndz.txt", usecols=(0,1))

ps = patchyScreening(box, isim, iz, m_cut, s_cut, ncpu=int(sys.argv[1]), lightcone=lc, mle=False)
ps.filter_stellar_mass()

m_cut_hires = float(np.loadtxt(f"./data_files/mle_parameters/L1000N3600/{isim}/{iz}/lightcone{lc}/mle_values.txt", usecols=1, skiprows=6, max_rows=1, delimiter='='))
s_cut_hires = float(np.loadtxt(f"./data_files/mle_parameters/L1000N3600/{isim}/{iz}/lightcone{lc}/mle_values.txt", usecols=1, skiprows=7, max_rows=1, delimiter='='))
ps_hires = patchyScreening('L1000N3600', isim, iz, m_cut_hires, s_cut_hires, ncpu=int(sys.argv[1]), lightcone=lc, mle=False)
ps_hires.filter_stellar_mass()
m_cut_hires = float(np.loadtxt(f"./data_files/mle_parameters/L2800N5040/{isim}/{iz}/lightcone{lc}/mle_values.txt", usecols=1, skiprows=6, max_rows=1, delimiter='='))
s_cut_hires = float(np.loadtxt(f"./data_files/mle_parameters/L2800N5040/{isim}/{iz}/lightcone{lc}/mle_values.txt", usecols=1, skiprows=7, max_rows=1, delimiter='='))
ps_largebox = patchyScreening('L2800N5040', isim, iz, m_cut_hires, s_cut_hires, ncpu=int(sys.argv[1]), lightcone=lc, mle=False)
ps_largebox.filter_stellar_mass()

if both_samples:
    if iz == "Blue":
        z_mean_other = 0.6
    elif iz == "Green":
        z_mean_other = 1.1
    iz_other = 'Blue' if iz=='Green' else 'Green'
    m_cut_other = float(np.loadtxt(f"./data_files/mle_parameters/{box}/{isim}/{iz_other}/lightcone{lc}/mle_values.txt", usecols=1, skiprows=6, max_rows=1, delimiter='='))
    s_cut_other = float(np.loadtxt(f"./data_files/mle_parameters/{box}/{isim}/{iz_other}/lightcone{lc}/mle_values.txt", usecols=1, skiprows=7, max_rows=1, delimiter='='))
    m_cut_other_name = f"{m_cut_other:.3f}".replace(".", "p")
    s_cut_other_name = f"{s_cut_other:.3f}".replace(".", "p")


    ps_other = patchyScreening(box, isim, "Green" if iz == "Blue" else "Blue", m_cut_other, s_cut_other, ncpu=int(sys.argv[1]), lightcone=lc, mle=False)
    ps_other.filter_stellar_mass()

n_bins = 100

# ---- user config ----
# PARQUET_OUT = Path("")

HDF5_TEMPLATE = '/cosma8/data/dp004/flamingo/Runs/{box}/{isim}/SOAP-HBT/halo_properties_{snap:04d}.hdf5'  

# IMPORTANT: adjust if your encoding differs
SATELLITE_FLAG = 0
CENTRAL_FLAG = 1

# dataset paths inside your HDF5 (edit to match your file)
DS_ID = 'InputHalos/HaloCatalogueIndex'
DS_STRUCTURETYPE = 'InputHalos/IsCentral'
DS_MVIR = 'SO/500_crit/TotalMass'
DS_HOSTID = 'SOAP/HostHaloIndex'
# ---------------------

if ps.merge[ps.merge['Structuretype'] == SATELLITE_FLAG].iloc[0]['mvir'] == 0:
    print(ps.merge[ps.merge['Structuretype'] == SATELLITE_FLAG]['mvir'])

    # Load your (lightcone / selected) catalogue
    cat = ps.merge

    # Select satellites with valid HaloID
    if "HostHaloID" in cat.columns:
        sats = cat[
            (cat["Structuretype"] == SATELLITE_FLAG) &
            (cat["HostHaloID"] >= 0)
        ][["ID", "SnapNum", "HostHaloID"]]
    elif "HaloID" in cat.columns:
        sats = cat[
            (cat["Structuretype"] == SATELLITE_FLAG) &
            (cat["HaloID"] >= 0)
        ][["ID", "SnapNum", "HaloID"]]
    else:
        sats = cat[
            cat["Structuretype"] == SATELLITE_FLAG
        ][["ID", "SnapNum"]]

    snapnums = sats["SnapNum"].unique().tolist()
    snapnums.sort()

    updates = []  # will collect (ID, new_mvir) for all satellites across snaps

    for i, snap in enumerate(snapnums):
        sats_snap = sats[sats["SnapNum"] == snap]
        print(cat[(cat["SnapNum"] == snap) & (cat["Structuretype"] == 0)].head())

        h5_path = Path(HDF5_TEMPLATE.format(snap=int(snap), box=box, isim=isim))
        with h5py.File(h5_path, "r") as f:
            # Read full arrays (HaloID is an index into the FULL catalogue arrays)
            struct = f[DS_STRUCTURETYPE][...]  # shape (N,)
            mvir = f[DS_MVIR][...] *1e10       # shape (N,)

        if "HostHaloID" in sats_snap.columns:
            halo_ids = sats_snap["HostHaloID"].to_numpy()
        elif "HaloID" in sats_snap.columns:
            halo_ids = sats_snap["HaloID"].to_numpy()
        else:
            ids = sats_snap["ID"].to_numpy()
            with h5py.File(h5_path, "r") as f:
                host_halos = np.column_stack((f[DS_ID][...], f[DS_HOSTID][...]))
            halo_ids = np.array([
                host_halos[host_halos[:, 0] == halo_id, 1][0]
                if np.any(host_halos[:, 0] == halo_id)
                else -1
                for halo_id in ids
            ])
        # print(host_halos[host_halos[:,0] == ids[0]])
        # print(halo_ids)
                
        host_struct = struct[halo_ids]
        host_mvir = mvir[halo_ids]

        # Only accept hosts that are centrals; otherwise set null
        host_mvir = np.where(host_struct == CENTRAL_FLAG, host_mvir, np.nan)

        updates.append(
            pd.DataFrame({
                "ID": sats_snap["ID"].to_numpy(),
                "_host_mvir": host_mvir,
            })
        )

        print(updates[i][0:5])

    updates_df = pd.concat(updates) if updates else pd.DataFrame({"ID": [], "_host_mvir": []})

    # Join updates back onto the original catalogue and overwrite mvir for satellites only
    cat_fixed = pd.merge(cat, updates_df, on="ID", how="left")
    cat_fixed["mvir"] = np.where(
        (cat_fixed["Structuretype"] == SATELLITE_FLAG) & cat_fixed["_host_mvir"].notna(),
        cat_fixed["_host_mvir"],
        cat_fixed["mvir"],
    )
    cat_fixed = cat_fixed.drop(columns=["_host_mvir"])

    print(cat_fixed[(cat_fixed["SnapNum"] == snap) & (cat_fixed["Structuretype"] == 0)].head())

    # --- Save (commented for testing) ---
    # PARQUET_OUT.parent.mkdir(parents=True, exist_ok=True)
    # cat_fixed.write_parquet(PARQUET_OUT, compression="snappy")

    total_sample = cat_fixed
    centrals_sample = cat_fixed[cat_fixed["Structuretype"] == CENTRAL_FLAG]
    satellites_sample = cat_fixed[cat_fixed["Structuretype"] == SATELLITE_FLAG]

elif ps.merge[ps.merge["Structuretype"] == SATELLITE_FLAG]["mvir"].iloc[0] > 0:
    print(ps.merge[ps.merge["Structuretype"] == SATELLITE_FLAG]["mvir"])

    total_sample = ps.merge
    centrals_sample = ps.merge[ps.merge["Structuretype"] == CENTRAL_FLAG]
    satellites_sample = ps.merge[ps.merge["Structuretype"] == SATELLITE_FLAG]

    total_sample_hires = ps_hires.merge
    total_sample_largebox = ps_largebox.merge

    if both_samples:
        total_sample_other = ps_other.merge
        centrals_sample_other = ps_other.merge[ps_other.merge["Structuretype"] == CENTRAL_FLAG]
        satellites_sample_other = ps_other.merge[ps_other.merge["Structuretype"] == SATELLITE_FLAG]

# log_bins_stellar_mass = np.logspace(min(total_sample['mstar'].to_numpy()), max(total_sample['mstar'].to_numpy()), n_bins)
# log_bins_halo_mass = np.logspace(min(total_sample['mvir'].to_numpy()), max(total_sample['mvir'].to_numpy()), n_bins)

stellar_mass_hist, stellar_bins = np.histogram(np.log10(total_sample['mstar'].to_numpy()), bins=n_bins)
stellar_mass_hist_centrals, _ = np.histogram(np.log10(centrals_sample['mstar'].to_numpy()), bins=stellar_bins)
stellar_mass_hist_satellites, _ = np.histogram(np.log10(satellites_sample['mstar'].to_numpy()), bins=stellar_bins)

stellar_mass_hist_hires, stellar_bins_hires = np.histogram(np.log10(total_sample_hires['mstar'].to_numpy()), bins=n_bins)
stellar_mass_hist_largebox, stellar_bins_largebox = np.histogram(np.log10(total_sample_largebox['mstar'].to_numpy()), bins=n_bins)

halo_mass_hist, halo_bins = np.histogram(np.log10(total_sample['mvir'].to_numpy()), bins=n_bins)
halo_mass_hist_centrals, _ = np.histogram(np.log10(centrals_sample['mvir'].to_numpy()), bins=halo_bins)
halo_mass_hist_satellites, _ = np.histogram(np.log10(satellites_sample['mvir'].to_numpy()), bins=halo_bins)

halo_mass_hist_hires, halo_bins_hires = np.histogram(np.log10(total_sample_hires['mvir'].to_numpy()), bins=n_bins)
halo_mass_hist_largebox, halo_bins_largebox = np.histogram(np.log10(total_sample_largebox['mvir'].to_numpy()), bins=n_bins)

redshift_hist = []
redshift_hist_centrals = []
redshift_hist_satellites = []
for z in redshift_bins[:, 0]:
    snapnum = len(redshift_bins[:, 0]) - z

    total_z = total_sample[total_sample["SnapNum"] == snapnum]
    centrals_z = centrals_sample[centrals_sample["SnapNum"] == snapnum]
    satellites_z = satellites_sample[satellites_sample["SnapNum"] == snapnum]

    print(z, len(redshift_bins[:, 0]), len(total_z))

    if len(total_z) == 0:
        break

    redshift_hist.append(len(total_z))
    print(len(total_z))
    redshift_hist_centrals.append(len(centrals_z))
    redshift_hist_satellites.append(len(satellites_z))
redshift_hist = np.array(redshift_hist)
redshift_hist_centrals = np.array(redshift_hist_centrals)
redshift_hist_satellites = np.array(redshift_hist_satellites)
print(redshift_hist)

# redshift_hist, redshift_bins = np.histogram(total_sample['z'].to_numpy(), bins=n_bins, range=(0, 3))
# redshift_hist_centrals, _ = np.histogram(centrals_sample['z'].to_numpy(), bins=redshift_bins)
# redshift_hist_satellites, _ = np.histogram(satellites_sample['z'].to_numpy(), bins=redshift_bins)

# stellar_mass_hist_reduced, stellar_bins_reduced = np.histogram(total_sample['mstar'].to_numpy(), bins=n_bins, range=(stellar_bins.min(), stellar_bins.max()))
# halo_mass_hist_reduced, halo_bins_reduced = np.histogram(total_sample['mvir'].to_numpy(), bins=n_bins, range=(halo_bins.min(), halo_bins.max()))


satellite_fraction_stellar_mass = (stellar_mass_hist_satellites/stellar_mass_hist) #stellar_mass_hist_satellites/stellar_mass_hist_reduced #stellar_mass_hist_satellites/stellar_mass_hist_reduced
satellite_fraction_halo_mass = (halo_mass_hist_satellites/halo_mass_hist) #halo_mass_hist_satellites/halo_mass_hist_reduced #halo_mass_hist_satellites/halo_mass_hist_reduced
satellite_fraction_redshift = (redshift_hist_satellites/redshift_hist) #redshift_hist_satellites/redshift_hist #redshift_hist_satellites/redshift_hist
print(satellite_fraction_stellar_mass, satellite_fraction_halo_mass, satellite_fraction_redshift)
print(halo_mass_hist_satellites[:10], halo_mass_hist[:10], satellite_fraction_halo_mass[:10])

if plot_all == "True":

    # fig, ax = pb.subplots()
    # pb.hist(np.log10(total_sample['mstar'].to_numpy()), bins=n_bins, log=True, histtype='step', label='All galaxies', color='black')
    # pb.errorbar((stellar_bins[1:] - stellar_bins[:-1])/2 , stellar_mass_hist, yerr=np.sqrt(stellar_mass_hist)/(stellar_bins[1:] - stellar_bins[:-1]), ecolor='black', linewidth=0, alpha=1, elinewidth=2)
    # pb.hist(np.log10(centrals_sample['mstar'].to_numpy()), bins=n_bins, log=True, histtype='step', label='Centrals', color='red')
    # pb.errorbar((stellar_bins[1:] - stellar_bins[:-1])/2, stellar_mass_hist_centrals, yerr=np.sqrt(stellar_mass_hist_centrals)/(stellar_bins[1:] - stellar_bins[:-1]), ecolor='red', linewidth=0, alpha=1, elinewidth=2)
    # pb.hist(np.log10(satellites_sample['mstar'].to_numpy()), bins=n_bins, log=True, histtype='step', label='Satellites', color='blue')
    # pb.errorbar((stellar_bins[1:] - stellar_bins[:-1])/2, stellar_mass_hist_satellites, yerr=np.sqrt(stellar_mass_hist_satellites)/(stellar_bins[1:] - stellar_bins[:-1]), ecolor='blue', linewidth=0, alpha=1, elinewidth=2)
    stellar_bin_centres = 0.5 * (stellar_bins[1:] + stellar_bins[:-1])
    pb.stairs(stellar_mass_hist, stellar_bins, color='black', label='All galaxies')
    pb.errorbar(stellar_bin_centres, stellar_mass_hist, yerr=np.sqrt(stellar_mass_hist),
                fmt='none', ecolor='black', elinewidth=1)
    pb.stairs(stellar_mass_hist_centrals, stellar_bins, color='red', label='Centrals')
    pb.errorbar(stellar_bin_centres, stellar_mass_hist_centrals, yerr=np.sqrt(stellar_mass_hist_centrals),
                fmt='none', ecolor='red', elinewidth=1)
    pb.stairs(stellar_mass_hist_satellites, stellar_bins, color='blue', label='Satellites')
    pb.errorbar(stellar_bin_centres, stellar_mass_hist_satellites, yerr=np.sqrt(stellar_mass_hist_satellites),
                fmt='none', ecolor='blue', elinewidth=1)


    # h1 = make_hist(np.log10(total_sample['mstar'].to_numpy()), bins=stellar_bins)
    # h2 = make_hist(np.log10(centrals_sample['mstar'].to_numpy()), bins=stellar_bins)
    # h3 = make_hist(np.log10(satellites_sample['mstar'].to_numpy()), bins=stellar_bins)
    # plot_hist(h1, ax=ax, color="black", label="All galaxies", histtype='step', linewidth=1.5)
    # plot_hist(h2, ax=ax, color="red", label="Centrals", histtype='step', linewidth=1.5)
    # plot_hist(h3, ax=ax, color="blue", label="Satellites", histtype='step', linewidth=1.5)
    # plot_error_hist(h1, ax=ax, color="black", label="All galaxies", markersize=0)
    # plot_error_hist(h2, ax=ax, color="red", label="Centrals", markersize=0)
    # plot_error_hist(h3, ax=ax, color="blue", label="Satellites", markersize=0)
    # ax.set_title(f"Stellar mass distribution \n(Box = {box}, Sim = {isim}, Sample = {iz}, Mass cut = [{m_cut}, {s_cut}])")
    # ax.set_xlabel("Stellar Mass [$M_*$]")
    # ax.set_ylabel("Counts")
    # ax.set_xlim(10, 13)
    # ax.set_yscale('log')
    # ax.legend()
    # fig.savefig(f"./Plots/stellar_mass_distribution_mock_catalog_{box}_{isim}_{iz}_{m_cut_name}_{s_cut_name}.{plot_suffix}", bbox_inches="tight")
    if plot_suffix != 'pdf':
        pb.title(f"Stellar mass distribution \n(Box = {box}, Sim = {isim}, Sample = {iz}, Mass cut = [{m_cut}, {s_cut}])")
    pb.xlabel("Stellar Mass [$\log_{10}M_{\odot}$]")
    pb.ylabel("Counts")
    pb.legend(fontsize=6)
    # pb.xscale('log')
    pb.yscale('log')
    pb.xlim(10.5, 12.25)
    pb.ylim(bottom=1000, top=1.5e7)
    pb.savefig(f"./Plots/stellar_mass_distribution_mock_catalog_{box}_{isim}_{iz}_{m_cut_name}_{s_cut_name}.{plot_suffix}", dpi=400)
    pb.clf()

    stellar_bin_centres_hires = 0.5 * (stellar_bins_hires[1:] + stellar_bins_hires[:-1])
    stellar_bin_centres_largebox = 0.5 * (stellar_bins_largebox[1:] + stellar_bins_largebox[:-1])
    pb.stairs(stellar_mass_hist, stellar_bins, color="#117733", label='L1_m9')
    pb.errorbar(stellar_bin_centres, stellar_mass_hist, yerr=np.sqrt(stellar_mass_hist),
                fmt='none', ecolor="#117733", elinewidth=1)

    pb.stairs(stellar_mass_hist_hires, stellar_bins_hires, color="#CC6677", label='L1_m8')
    pb.errorbar(stellar_bin_centres_hires, stellar_mass_hist_hires, yerr=np.sqrt(stellar_mass_hist_hires),
                fmt='none', ecolor="#CC6677", elinewidth=1)
    pb.stairs(stellar_mass_hist_largebox, stellar_bins_largebox, color="#332288", label='L2p8_m9 (lc=0)')
    pb.errorbar(stellar_bin_centres_largebox, stellar_mass_hist_largebox, yerr=np.sqrt(stellar_mass_hist_largebox),
                fmt='none', ecolor="#332288", elinewidth=1)
    
    if plot_suffix != 'pdf':
        pb.title(f"Stellar mass distribution \n(Box = {box}, Sim = {isim}, Sample = {iz}, Mass cut = [{m_cut}, {s_cut}])")
    pb.xlabel("Stellar Mass [$\log_{10}M_{\odot}$]")
    pb.ylabel("Counts")
    pb.yscale('log')
    # pb.xlim(10, 16)
    pb.legend(fontsize=6)
    pb.savefig(f"./Plots/stellar_mass_distribution_mock_catalog_multi_res_{iz}.{plot_suffix}", dpi=400)
    pb.clf()

    # pb.hist(np.log10(total_sample['mvir'].to_numpy()), bins=n_bins, log=True, histtype='step', label='All galaxies', color='black')
    # pb.errorbar((halo_bins[1:]-halo_bins[:-1])/2, halo_mass_hist, yerr=np.sqrt(halo_mass_hist), ecolor='black', linewidth=0, alpha=1, elinewidth=2)
    # pb.hist(np.log10(centrals_sample['mvir'].to_numpy()), bins=n_bins, log=True, histtype='step', label='Centrals', color='red')
    # pb.errorbar((halo_bins[1:]-halo_bins[:-1])/2, halo_mass_hist_centrals, yerr=np.sqrt(halo_mass_hist_centrals), ecolor='red', linewidth=0, alpha=1, elinewidth=2)
    # pb.hist(np.log10(satellites_sample['mvir'].to_numpy()), bins=n_bins, log=True, histtype='step', label='Satellites', color='blue')
    # pb.errorbar((halo_bins[1:]-halo_bins[:-1])/2, halo_mass_hist_satellites, yerr=np.sqrt(halo_mass_hist_satellites), ecolor='blue', linewidth=0, alpha=1, elinewidth=2)
    halo_bin_centres = 0.5 * (halo_bins[1:] + halo_bins[:-1])
    pb.stairs(halo_mass_hist, halo_bins, color='black', label='All galaxies')
    pb.errorbar(halo_bin_centres, halo_mass_hist, yerr=np.sqrt(halo_mass_hist),
                fmt='none', ecolor='black', elinewidth=1)
    pb.stairs(halo_mass_hist_centrals, halo_bins, color='red', label='Centrals')
    pb.errorbar(halo_bin_centres, halo_mass_hist_centrals, yerr=np.sqrt(halo_mass_hist_centrals),
                fmt='none', ecolor='red', elinewidth=1)
    pb.stairs(halo_mass_hist_satellites, halo_bins, color='blue', label='Satellites')
    pb.errorbar(halo_bin_centres, halo_mass_hist_satellites, yerr=np.sqrt(halo_mass_hist_satellites),
                fmt='none', ecolor='blue', elinewidth=1)
    if plot_suffix != 'pdf':
        pb.title(f"Halo mass distribution \n(Box = {box}, Sim = {isim}, Sample = {iz}, Mass cut = [{m_cut}, {s_cut}])")
    pb.xlabel("Halo Mass [$\log_{10}M_{\odot}$]")
    pb.ylabel("Counts")
    pb.yscale('log')
    pb.xlim(11, 15.5)
    pb.ylim(bottom=1000, top=1.5e7)
    pb.legend(fontsize=6)
    pb.savefig(f"./Plots/halo_mass_distribution_mock_catalog_{box}_{isim}_{iz}_{m_cut_name}_{s_cut_name}.{plot_suffix}", dpi=400)
    pb.clf()

    halo_bin_centres_hires = 0.5 * (halo_bins_hires[1:] + halo_bins_hires[:-1])
    halo_bin_centres_largebox = 0.5 * (halo_bins_largebox[1:] + halo_bins_largebox[:-1])
    pb.stairs(halo_mass_hist, halo_bins, color="#117733", label='L1_m9')
    pb.errorbar(halo_bin_centres, halo_mass_hist, yerr=np.sqrt(halo_mass_hist),
                fmt='none', ecolor="#117733", elinewidth=1)

    pb.stairs(halo_mass_hist_hires, halo_bins_hires, color="#CC6677", label='L1_m8')
    pb.errorbar(halo_bin_centres_hires, halo_mass_hist_hires, yerr=np.sqrt(halo_mass_hist_hires),
                fmt='none', ecolor="#CC6677", elinewidth=1)
    pb.stairs(halo_mass_hist_largebox, halo_bins_largebox, color="#332288", label='L2p8_m9 (lc=0)')
    pb.errorbar(halo_bin_centres_largebox, halo_mass_hist_largebox, yerr=np.sqrt(halo_mass_hist_largebox),
                fmt='none', ecolor="#332288", elinewidth=1)
    
    if plot_suffix != 'pdf':
        pb.title(f"Halo mass distribution \n(Box = {box}, Sim = {isim}, Sample = {iz}, Mass cut = [{m_cut}, {s_cut}])")
    pb.xlabel("Halo Mass [$\log_{10}M_{\odot}$]")
    pb.ylabel("Counts")
    pb.yscale('log')
    # pb.xlim(10, 16)
    pb.legend(fontsize=6)
    pb.savefig(f"./Plots/halo_mass_distribution_mock_catalog_multi_res_{iz}.{plot_suffix}", dpi=400)
    pb.clf()

    # print(satellite_fraction_stellar_mass, stellar_bins)
    # print(satellite_fraction_halo_mass, halo_bins)
    # print(satellite_fraction_redshift, redshift_bins)


    # pb.plot(halo_bins[:-1], satellite_fraction_halo_mass, label='Halo mass')
    pb.errorbar(halo_bins[:-1], satellite_fraction_halo_mass, yerr=np.sqrt(satellite_fraction_halo_mass*(1-satellite_fraction_halo_mass)/halo_mass_hist))
    # pb.plot(stellar_bins[:-1], satellite_fraction_stellar_mass, label='Stellar mass')
    if plot_suffix != 'pdf':
        pb.title(f"Satellite fraction \n(Box = {box}, Sim = {isim}, Sample = {iz}, Mass cut = [{m_cut}, {s_cut}])")
    pb.xlabel("Halo Mass [$\log_{10}M_{\odot}$]")
    pb.ylabel("Satellite Fraction")
    # pb.xscale('log')
    # pb.xlim(1e8, 1e16)
    # pb.xlim(1e10, 5e13)
    pb.xlim(left=12)
    pb.savefig(f"./Plots/satellite_fraction_mock_catalog_{box}_{isim}_{iz}_{m_cut_name}_{s_cut_name}_mass.{plot_suffix}", dpi=400)
    pb.clf()

    # pb.plot(redshift_midpoints[:len(satellite_fraction_redshift)], satellite_fraction_redshift, label='Redshift')
    pb.errorbar(redshift_midpoints[:len(satellite_fraction_redshift)], satellite_fraction_redshift, yerr=np.sqrt(satellite_fraction_redshift*(1-satellite_fraction_redshift)/redshift_hist))
    if plot_suffix != 'pdf':
        pb.title(f"Satellite fraction \n(Box = {box}, Sim = {isim}, Sample = {iz}, Mass cut = [{m_cut}, {s_cut}])")
    pb.xlabel("Redshift")
    pb.ylabel("Satellite Fraction")
    pb.savefig(f"./Plots/satellite_fraction_mock_catalog_{box}_{isim}_{iz}_{m_cut_name}_{s_cut_name}_redshift.{plot_suffix}", dpi=400)
    pb.clf()

    # halo_mass_hist_redshift_bins, _ = np.histogram(total_sample['mvir'].to_numpy() * total_sample['z'].to_numpy(), bins=redshift_bins)
    # halo_mass_hist_centrals_redshift_bins, _ = np.histogram(centrals_sample['mvir'].to_numpy() * centrals_sample['z'].to_numpy(), bins=redshift_bins)
    # halo_mass_hist_satellites_redshift_bins, _ = np.histogram(satellites_sample['mvir'].to_numpy() * satellites_sample['z'].to_numpy(), bins=redshift_bins)

    # pb.plot(redshift_bins[:-1], halo_mass_hist_redshift_bins, label='All galaxies')
    # pb.plot(redshift_bins[:-1], halo_mass_hist_centrals_redshift_bins, label='Centrals')
    # pb.plot(redshift_bins[:-1], halo_mass_hist_satellites_redshift_bins, label='Satellites')
    # pb.title(f"Average host halo mass \n(Box = {box}, Sim = {isim}, Sample = {iz}, Mass cut = [{m_cut}, {s_cut}])")
    # pb.xlabel("Redshift")
    # pb.ylabel("Average host halo mass [$M_{\odot}$]")
    # # pb.yscale('log')
    # pb.legend()
    # pb.savefig(f"./Plots/average_host_halo_mass_redshift_mock_catalog_{box}_{isim}_{iz}_{m_cut}_{s_cut}.{plot_suffix}", dpi=400)
    # pb.clf()

    # Plot of mean number of galaxies as a function of halo mass, split by centrals and satellites
    # pb.plot(halo_bins[:-1], halo_mass_hist/total_sample.height, label='Centrals')
    # # pb.plot(halo_bins[:-1], halo_mass_hist/, label='Satellites')
    # pb.title(f"Mean stellar mass per halo mass \n(Box = {box}, Sim = {isim}, Sample = {iz}, Mass cut = [{m_cut}, {s_cut}])")
    # pb.xlabel("Halo Mass [$M_{\odot}$]")
    # pb.ylabel("Mean stellar mass per halo mass")
    # pb.xscale('log')
    # pb.yscale('log')
    # pb.xlim(1e8, 1e16)
    # pb.legend()
    # pb.savefig(f"./Plots/mean_number_galaxies_per_halo_mass_mock_catalog_{box}_{isim}_{iz}_{m_cut_name}_{s_cut_name}.{plot_suffix}", dpi=400)
    # pb.clf()

    stellar_cut = 10**(np.loadtxt(f"./data_files/z_dependant_stellar_cuts/{box}/{iz}/z_stellar_cut_data_{m_cut_name}_{s_cut_name}.txt")[:,1])
    if both_samples:
        stellar_cut_other = 10**(np.loadtxt(f"./data_files/z_dependant_stellar_cuts/{box}/{iz_other}/z_stellar_cut_data_{m_cut_other_name}_{s_cut_other_name}.txt")[:,1])

    median_stellar_mass = []
    tenth_percentile_stellar_mass = []
    ninetyth_percentile_stellar_mass = []
    min_stellar_mass = []
    for i, z in enumerate(redshift_midpoints):
        if z >= 3.0:
            break
        total_sample_z_bin = total_sample[total_sample["SnapNum"] == len(redshift_bins[:, 0]) - i]
        median_stellar_mass.append(total_sample_z_bin['mstar'].median())
        tenth_percentile_stellar_mass.append(total_sample_z_bin['mstar'].quantile(0.1))
        ninetyth_percentile_stellar_mass.append(total_sample_z_bin['mstar'].quantile(0.9))
        min_stellar_mass.append(total_sample_z_bin['mstar'].min())

    print(len(median_stellar_mass), len(tenth_percentile_stellar_mass), len(ninetyth_percentile_stellar_mass), len(redshift_midpoints))
    pb.plot(redshift_midpoints[:len(median_stellar_mass)], median_stellar_mass, color=iz.lower(), label=f'Median Stellar Mass ({iz} sample)')
    # pb.plot(redshift_midpoints[:len(tenth_percentile_stellar_mass)], tenth_percentile_stellar_mass, color=iz.lower(), linewidth=0.5, linestyle=':', label='10th Percentile Stellar Mass')
    # pb.plot(redshift_midpoints[:len(ninetyth_percentile_stellar_mass)], ninetyth_percentile_stellar_mass, color=iz.lower(), linewidth=0.5, linestyle=':', label='90th Percentile Stellar Mass')
    pb.fill_between(redshift_midpoints[:len(median_stellar_mass)], tenth_percentile_stellar_mass, ninetyth_percentile_stellar_mass, color=iz.lower(), alpha=0.25, label=f'10-90th Percentile Range ({iz} sample)')
    if not both_samples and plot_suffix != 'pdf':
        pb.plot(redshift_midpoints[:len(median_stellar_mass)], min_stellar_mass, color='grey', label='Minimum Stellar Mass', linewidth=0, marker='o', markersize=1)
    pb.plot(redshift_midpoints[:len(median_stellar_mass)], stellar_cut[:len(median_stellar_mass)], color='red' if not both_samples else iz.lower(), label=f'Optimised Stellar Mass Cut ({iz} sample)', linestyle='--')
    if both_samples:
        median_stellar_mass_other = []
        tenth_percentile_stellar_mass_other = []
        ninetyth_percentile_stellar_mass_other = []
        for i, z in enumerate(redshift_midpoints):
            if z >= 3.0:
                break
            total_sample_z_bin_other = total_sample_other[total_sample_other["SnapNum"] == len(redshift_bins[:, 0]) - i]
            median_stellar_mass_other.append(total_sample_z_bin_other['mstar'].median())
            tenth_percentile_stellar_mass_other.append(total_sample_z_bin_other['mstar'].quantile(0.1))
            ninetyth_percentile_stellar_mass_other.append(total_sample_z_bin_other['mstar'].quantile(0.9))
        pb.plot(redshift_midpoints[:len(median_stellar_mass_other)], median_stellar_mass_other, color='blue' if iz == 'Green' else 'green', label=f'Median Stellar Mass ({iz_other} sample)')
        # pb.plot(redshift_midpoints[:len(tenth_percentile_stellar_mass_other)], tenth_percentile_stellar_mass_other, color='blue' if iz == 'Green' else 'green', linewidth=0.5, linestyle=':', label='10th Percentile Stellar Mass Other Sample')
        # pb.plot(redshift_midpoints[:len(ninetyth_percentile_stellar_mass_other)], ninetyth_percentile_stellar_mass_other, color='blue' if iz == 'Green' else 'green', linewidth=0.5, linestyle=':', label='90th Percentile Stellar Mass Other Sample')
        pb.fill_between(redshift_midpoints[:len(median_stellar_mass_other)], tenth_percentile_stellar_mass_other, ninetyth_percentile_stellar_mass_other, color='blue' if iz == 'Green' else 'green', alpha=0.25, label=f'10-90th Percentile Range ({iz_other} sample)')
        pb.plot(redshift_midpoints[:len(median_stellar_mass_other)], stellar_cut_other[:len(median_stellar_mass_other)], color='blue' if iz == 'Green' else 'green', label=f'Optimised Stellar Mass Cut ({iz_other} sample)', linestyle='--')
    if plot_suffix != 'pdf':
        pb.title(f"Stellar Mass vs Redshift \n(Box = {box}, Sim = {isim}, Sample = {iz}, Mass cut = [{m_cut}, {s_cut}])" if not both_samples else f"Stellar Mass vs Redshift \n(Box = {box}, Sim = {isim})")
    pb.xlabel("Redshift")
    pb.ylabel("Stellar Mass [$\log_{10}M_{\odot}$]")
    pb.yscale('log')
    pb.legend(fontsize=6)
    pb.savefig(f"./Plots/median_stellar_mass_redshift_mock_catalog_{box}_{isim}_{iz}_{m_cut_name}_{s_cut_name}.{plot_suffix}" if not both_samples else f"./Plots/median_stellar_mass_redshift_mock_catalog_{box}_{isim}_both_samples.{plot_suffix}", dpi=400)
    pb.clf()


print(len(total_sample), len(centrals_sample), len(satellites_sample))
# vals, bins, _ = pb.hist(total_sample['z'], bins=n_bins, range=(0, 3), histtype='step')
# pb.hist(centrals_sample['z'], bins=n_bins, range=(0, 3), histtype='step', label='Centrals')
# pb.hist(satellites_sample['z'], bins=n_bins, range=(0, 3), histtype='step', label='Satellites')
pb.plot(redshift_midpoints[:len(redshift_hist)], redshift_hist, color='black', label='Total')
pb.plot(redshift_midpoints[:len(redshift_hist_centrals)], redshift_hist_centrals, color='red', label='Centrals')
pb.plot(redshift_midpoints[:len(redshift_hist_satellites)], redshift_hist_satellites, color='blue', label='Satellites')
pb.plot(obs_dndz[:,0], obs_dndz[:,1]*(len(total_sample)/simpson(obs_dndz[:,1], obs_dndz[:,0])*(0.05)), label='Observed dN/dz', linestyle='--')
if plot_suffix != 'pdf':
    pb.title(f"Redshift distribution \n(Box = {box}, Sim = {isim}, Sample = {iz}, Mass cut = [{m_cut}, {s_cut}])")
pb.xlabel("Redshift")
pb.ylabel("Counts")
pb.savefig(f"./Plots/redshift_distribution_mock_catalog_{box}_{isim}_{iz}_{m_cut_name}_{s_cut_name}.{plot_suffix}", dpi=400)
pb.legend(fontsize=6)
pb.clf()