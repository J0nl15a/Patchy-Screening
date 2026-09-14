# import hdfstream

# flamingo = hdfstream.open("cosma", "/FLAMINGO")
# snap = flamingo["L1_m9/L1_m9/SOAP-HBT/halo_properties_0077.hdf5"]
# m200crit = snap['SO/200_crit/TotalMass'][...] * 1e10

# print(m200crit)

#!/usr/bin/env python3

import argparse
from pathlib import Path

import hdfstream
import h5py
import numpy as np
import pandas as pd


def name_float(x, mle=False):
    if mle:
        return f"{x:.3f}".replace(".", "p")
    else:
        return f"{x:.1f}".replace(".", "p")


def flamingo_stream_names(box, sim):
    """
    Translate your local FLAMINGO box/simulation names to the names
    used in the hdfstream FLAMINGO archive.

    Extend this dictionary as required.
    """

    mapping = {
        ("L1000N1800", "HYDRO_FIDUCIAL"):                 ("L1_m9", "L1_m9"),
        ("L1000N1800", "HYDRO_LOW_SIGMA8"):               ("L1_m9", "LS8"),
        ("L1000N1800", "HYDRO_LOW_SIGMA8_STRONGEST_AGN"): ("L1_m9", "LS8_fgas-8sigma"),
        ("L1000N1800", "HYDRO_PLANCK"):                   ("L1_m9", "Planck"),
        ("L1000N1800", "HYDRO_PLANCK_LARGE_NU_FIXED"):    ("L1_m9", "PlanckNu0p24Fix"),
        ("L1000N1800", "HYDRO_PLANCK_LARGE_NU_VARY"):     ("L1_m9", "PlanckNu0p24Var"),
        ("L1000N1800", "HYDRO_STRONG_AGN"):               ("L1_m9", "fgas-2sigma"),
        ("L1000N1800", "HYDRO_STRONGER_AGN"):             ("L1_m9", "fgas-4sigma"),
        ("L1000N1800", "HYDRO_STRONGEST_AGN"):            ("L1_m9", "fgas-8sigma"),
        ("L1000N1800", "HYDRO_WEAK_AGN"):                 ("L1_m9", "fgas+2sigma"),
        ("L1000N1800", "HYDRO_STRONG_SUPERNOVA"):         ("L1_m9", "Mstar-1sigma"),
        ("L1000N1800", "HYDRO_JETS_published"):           ("L1_m9", "Jet"),
        ("L1000N1800", "HYDRO_STRONG_JETS_published"):    ("L1_m9", "Jet_fgas-4sigma"),
        ("L1000N3600", "HYDRO_FIDUCIAL"):                 ("L1_m8", "L1_m8"),
        ("L2800N5040", "HYDRO_FIDUCIAL"):                 ("L2p8_m9", "L2p8_m9"),


        # Add other simulations here, for example:
        #
        # ("L1000N1800", "HYDRO_STRONGEST_AGN"):
        #     ("L1_m9", "<hdfstream simulation name>"),
        #
        # ("L2800N5040", "HYDRO_FIDUCIAL"):
        #     ("L2p8_m9", "L2p8_m9"),
    }

    key = (box, sim)

    if key not in mapping:
        raise ValueError(
            f"No hdfstream mapping defined for box={box}, sim={sim}.\n"
            "Add this combination to flamingo_stream_names().")

    return mapping[key]


def load_soap_for_snapshot_hdfstream(flamingo, stream_box, stream_sim, snapnum, full=False):
    """
    Load SOAP-HBT information for one snapshot.

    For satellites, m200crit is replaced with the m200crit
    of the host central halo.
    """

    filename = (f"{stream_box}/{stream_sim}/SOAP-HBT/halo_properties_{snapnum:04d}.hdf5")

    print(f"Loading: {filename}")

    snap = flamingo[filename]

    ids = np.asarray(snap["InputHalos/HaloCatalogueIndex"][...])

    struct = np.asarray(snap["InputHalos/IsCentral"][...])

    host_index = np.asarray(snap["SOAP/HostHaloIndex"][...])

    m200crit = (np.asarray(snap["SO/200_crit/TotalMass"][...]) * 1.0e10)

    # ------------------------------------------------------------
    # Replace satellite m200crit with host m200crit
    # ------------------------------------------------------------

    SATELLITE_FLAG = 0
    CENTRAL_FLAG = 1

    m200crit_fixed = m200crit.copy()

    sat_mask = ((struct == SATELLITE_FLAG) & (host_index >= 0) & (host_index < len(m200crit)))

    if np.any(sat_mask):

        host_indices = host_index[sat_mask].astype(np.int64)

        host_struct = struct[host_indices]
        host_m200crit = m200crit[host_indices]

        # Only use the host mass when the referenced object is actually
        # a central halo. Otherwise retain the satellite's own m200crit.
        good_host = host_struct == CENTRAL_FLAG

        m200crit_fixed[sat_mask] = np.where(good_host, host_m200crit, m200crit[sat_mask])

    if not full:
        return ids, m200crit_fixed

    m500crit = (np.asarray(snap["SO/500_crit/TotalMass"][...]) * 1.0e10)
    mstar = (np.asarray(snap["ExclusiveSphere/50kpc/StellarMass"][...]) * 1.0e10)

    soap_data = pd.DataFrame({
        "ID": ids,
        "Structuretype": struct,
        # NOTE:
        # This is the object's OWN M500crit from SOAP.
        # It does not modify the catalogue's existing m500crit.
        "m500crit_soap": m500crit,
        # This one has the host correction applied to satellites.
        "m200crit": m200crit_fixed,
        "mstar": mstar,
        "HostHaloIndex": host_index,
    })

    return soap_data


def load_soap_for_snapshot_cosma6(data_root, box, sim, snapnum, full=False):
    filename = (f"{data_root}/{box}/{sim}/SOAP-HBT/halo_properties_{snapnum:04d}.hdf5")

    print(f"Loading: {filename}")

    with h5py.File(filename, "r") as snap:

        ids = np.asarray(snap["InputHalos/HaloCatalogueIndex"][...])

        struct = np.asarray(snap["InputHalos/IsCentral"][...])

        host_index = np.asarray(snap["SOAP/HostHaloIndex"][...])

        m200crit = (np.asarray(snap["SO/200_crit/TotalMass"][...]) * 1.0e10)

        SATELLITE_FLAG = 0
        CENTRAL_FLAG = 1

        m200crit_fixed = m200crit.copy()

        sat_mask = ((struct == SATELLITE_FLAG) & (host_index >= 0) & (host_index < len(m200crit)))

        if np.any(sat_mask):
            host_indices = host_index[sat_mask].astype(np.int64)

            host_struct = struct[host_indices]
            host_m200crit = m200crit[host_indices]

            good_host = host_struct == CENTRAL_FLAG

            m200crit_fixed[sat_mask] = np.where(good_host, host_m200crit, m200crit[sat_mask])

        if not full:
            return ids, m200crit_fixed

        m500crit = (np.asarray(snap["SO/500_crit/TotalMass"][...]) * 1.0e10)

        mstar = (np.asarray(snap["ExclusiveSphere/50kpc/StellarMass"][...]) * 1.0e10)

    soap_data = pd.DataFrame({
        "ID": ids,
        "Structuretype": struct,
        "m500crit_soap": m500crit,
        "m200crit": m200crit_fixed,
        "mstar": mstar,
        "HostHaloIndex": host_index,
    })

    return soap_data


def clean_catalogue_columns(catalogue):
    """
    Rename legacy columns and remove columns that are no longer required.
    """

    catalogue = catalogue.copy()

    rename_columns = {
        "mvir": "m500crit",
        "HostHaloID": "HostHaloIndex",
    }

    # Only rename columns that actually exist.
    rename_columns = {
        old: new
        for old, new in rename_columns.items()
        if old in catalogue.columns
    }

    catalogue = catalogue.rename(columns=rename_columns)

    # Remove old random-number column if present.
    catalogue = catalogue.drop(columns=["rand"], errors="ignore")

    return catalogue


def add_m200crit(mock_catalogue, box, sim):
    """
    Rename the requested columns and add M_200crit by matching each
    catalogue halo to its SOAP-HBT object within the appropriate snapshot.
    """

    catalogue = mock_catalogue.copy()

    # ------------------------------------------------------------
    # Check required catalogue columns
    # ------------------------------------------------------------

    required = ["ID", "SnapNum"]
    missing = [column for column in required if column not in catalogue.columns]

    if missing:
        raise ValueError(f"Mock catalogue is missing required columns: {missing}")

    # ------------------------------------------------------------
    # Open hdfstream
    # ------------------------------------------------------------

    # stream_box, stream_sim = flamingo_stream_names(box, sim)

    # flamingo = hdfstream.open("cosma", "/FLAMINGO")

    # Initialise new column
    catalogue["m200crit"] = np.nan

    # ------------------------------------------------------------
    # Use cosma6
    # ------------------------------------------------------------

    data_root = "/cosma6/data/dp004/flamingo/Runs"

    # ------------------------------------------------------------
    # Work snapshot-by-snapshot
    # ------------------------------------------------------------
    #
    # This is much faster than opening a SOAP file separately for
    # every halo.
    # ------------------------------------------------------------

    unique_snapshots = np.sort(catalogue["SnapNum"].unique().astype(int))

    print(f"Catalogue contains {len(catalogue):,} objects across {len(unique_snapshots)} snapshots.")

    for snapnum in unique_snapshots:

        cat_mask = catalogue["SnapNum"].to_numpy() == snapnum
        cat_indices = np.flatnonzero(cat_mask)

        catalogue_ids = (catalogue.iloc[cat_indices]["ID"].to_numpy())

        # soap_ids, soap_m200 = load_soap_for_snapshot_hdfstream(flamingo, stream_box, stream_sim, snapnum, full=False)
        soap_ids, soap_m200 = load_soap_for_snapshot_cosma6(data_root, box, sim, snapnum, full=False)

        # --------------------------------------------------------
        # Match halo ID -> SOAP row
        # --------------------------------------------------------
        #
        # Do not assume ID == array index.
        #
        # Using sort/searchsorted avoids constructing a huge Python
        # dictionary for each snapshot.
        # --------------------------------------------------------

        order = np.argsort(soap_ids)
        soap_ids_sorted = soap_ids[order]

        positions = np.searchsorted(soap_ids_sorted, catalogue_ids)

        valid = positions < len(soap_ids_sorted)

        matched = np.zeros(len(catalogue_ids), dtype=bool)

        matched[valid] = (soap_ids_sorted[positions[valid]] == catalogue_ids[valid])

        if np.any(matched):
            matched_soap_indices = order[positions[matched]]

            matched_catalogue_indices = cat_indices[matched]

            catalogue.loc[catalogue.index[matched_catalogue_indices], "m200crit"] = soap_m200[matched_soap_indices]

        n_match = np.count_nonzero(matched)
        n_total = len(catalogue_ids)

        print(f"Snapshot {snapnum:04d}: matched {n_match:,}/{n_total:,} ({100*n_match/n_total:.2f}%)")

        # Useful diagnostic information from your lightcone catalogue.
        if n_total > 0:
            subset = catalogue.iloc[cat_indices]

            print(f"z = [{subset['z'].min():.5f}, {subset['z'].max():.5f}]")
            print(f"xminpot = [{subset['xminpot'].min():.3f}, {subset['xminpot'].max():.3f}]")

    # ------------------------------------------------------------
    # Final checks
    # ------------------------------------------------------------

    missing_m200 = catalogue["m200crit"].isna()

    n_missing = missing_m200.sum()

    if n_missing:
        print(f"\nWARNING: {n_missing:,}/{len(catalogue):,} haloes could not be matched to a SOAP-HBT object.")
        print(catalogue.loc[missing_m200, ["ID", "SnapNum", "z", "xminpot"]].head(20))

    else:
        print("\nAll catalogue objects were successfully matched.")

    return catalogue


def print_random_match_checks(catalogue, box, sim, n=3, seed=None):
    """
    Select random successfully matched catalogue objects and print:

      1. the corresponding SOAP-HBT row
      2. the complete mock-catalogue row

    This is intended as a manual consistency check.
    """

    matched_catalogue = catalogue[catalogue["m200crit"].notna()]

    if matched_catalogue.empty:
        print("No successfully matched objects available for checking.")
        return

    n = min(n, len(matched_catalogue))

    rng = np.random.default_rng(seed)

    selected_positions = rng.choice(len(matched_catalogue), size=n, replace=False,)
    selected = matched_catalogue.iloc[selected_positions]

    stream_box, stream_sim = flamingo_stream_names(box, sim)
    flamingo = hdfstream.open("cosma", "/FLAMINGO")
    data_root = "/cosma6/data/dp004/flamingo/Runs"

    # Make sure pandas prints every column and does not truncate values.
    with pd.option_context(
        "display.max_columns", None,
        "display.width", None,
        "display.max_colwidth", None,
        "display.float_format", lambda x: f"{x:.12g}",
    ):

        for check_number, (_, cat_row) in enumerate(selected.iterrows(), start=1):
            snapnum = int(cat_row["SnapNum"])
            halo_id = cat_row["ID"]

            # soap = load_soap_for_snapshot_hdfstream(flamingo, stream_box, stream_sim, snapnum, full=True)
            soap = load_soap_for_snapshot_cosma6(data_root, box, sim, snapnum, full=True)
            soap_match = soap[soap["ID"] == halo_id]

            print("\n" + "=" * 100)
            print(f"CHECK {check_number}: snapshot={snapnum:04d}, ID={halo_id}")
            print("=" * 100)

            if len(soap_match) == 0:
                print("ERROR: halo is absent from SOAP-HBT snapshot.")
                continue

            if len(soap_match) > 1:
                print(f"WARNING: found {len(soap_match)} SOAP rows with ID={halo_id}")

            print("\nSOAP-HBT ROW:")
            print(soap_match.to_string(index=False))

            print("\nMOCK CATALOGUE ROW:")
            print(cat_row.to_frame().T.to_string(index=False))

            # ----------------------------------------------------
            # Direct comparison of columns common to both
            # ----------------------------------------------------

            common_columns = [
                column
                for column in [
                    "ID",
                    "Structuretype",
                    "m200crit",
                    "mstar",
                    "HostHaloIndex",
                ]

                if (column in catalogue.columns and column in soap_match.columns)
            ]

            print("\nCOMPARISON:")

            if "m500crit" in catalogue.columns and "m500crit_soap" in soap_match.columns:
                print("\nM500crit comparison "
                      "(catalogue may contain host mass for satellites):")
                print(f"catalogue m500crit = {cat_row['m500crit']}")
                print(f"SOAP own m500crit = {soap_match.iloc[0]['m500crit_soap']}")

            soap_row = soap_match.iloc[0]

            for column in common_columns:
                cat_value = cat_row[column]
                soap_value = soap_row[column]

                if np.issubdtype(np.asarray(cat_value).dtype, np.number):
                    same = np.isclose(cat_value, soap_value, rtol=1e-10, atol=0.0, equal_nan=True)
                else:
                    same = cat_value == soap_value

                print(
                    f"{column:20s} "
                    f"catalogue={cat_value!s:25s} "
                    f"SOAP={soap_value!s:25s} "
                    f"match={same}")


def main():

    parser = argparse.ArgumentParser(description=("Add M200crit to an existing FLAMINGO mock galaxy catalogue using hdfstream."))
    parser.add_argument("box", type=str, help="e.g. L1000N1800")
    parser.add_argument("sim", type=str, help="e.g. HYDRO_FIDUCIAL")
    parser.add_argument("sample", type=str, choices=["Blue", "Green"], help="Galaxy sample")
    parser.add_argument("--mle", action="store_true", help="Use MLE parameters")
    parser.add_argument("--amp", type=float, default=10.3, help="Mock catalogue amplitude / stellar-mass parameter")
    parser.add_argument("--slope", type=float, default=0.0, help="Mock catalogue slope parameter")
    parser.add_argument("--lightcone", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true", help="Overwrite the original parquet file.")
    args = parser.parse_args()

    if args.mle:
        mle_cut_file = f"/cosma8/data/dp004/dc-conl1/FLAMINGO/patchy_screening/data_files/mle_parameters/{args.box}/{args.sim}/{args.sample}/lightcone{args.lightcone}/mle_values.txt"
        args.amp = np.loadtxt(mle_cut_file, usecols=1, skiprows=6, max_rows=1, delimiter='=')
        args.slope = np.loadtxt(mle_cut_file, usecols=1, skiprows=7, max_rows=1, delimiter='=')

    amp_name = name_float(args.amp, mle=args.mle)
    slope_name = name_float(args.slope, mle=args.mle)

    input_path = Path(f"./data_files/mock_halo_catalogs/{args.box}/{args.sim}/{args.sample}/lightcone{args.lightcone}/sampled_halo_data_{amp_name}_{slope_name}.parquet")

    if not input_path.exists():
        raise FileNotFoundError(f"Mock catalogue does not exist:\n{input_path}")

    print(f"Reading catalogue: {input_path}")

    catalogue = pd.read_parquet(input_path)

    print("\nOriginal columns:")
    print(catalogue.columns.tolist())

    # Rename legacy columns and remove rand.
    catalogue = clean_catalogue_columns(catalogue)

    print("\nColumns after cleaning:")
    print(catalogue.columns.tolist())

    if 'm200crit' not in catalogue.columns:

        # Add M200crit.
        catalogue = add_m200crit(catalogue, args.box, args.sim)

        print("\nUpdated columns:")
        print(catalogue.columns.tolist())

        # Print three random SOAP/catalogue matches.
        print_random_match_checks(catalogue, args.box, args.sim, n=3)

        print("\nUpdated columns:")
        print(catalogue.columns.tolist())

    if args.overwrite:
        output_path = input_path
    else:
        output_path = input_path.with_name(input_path.stem + "_with_m200crit.parquet")

    print(f"\nSaving: {output_path}")

    catalogue.to_parquet(output_path, index=False)


if __name__ == "__main__":
    main()