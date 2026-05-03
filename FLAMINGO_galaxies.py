import os
import numpy as np, polars as pl
from pathlib import Path
from joblib import Parallel, delayed
from imp_patchy_screening import patchyScreening
from FLAMINGO_halo_redshifts import multiprocess_z_bins
import time

def build_shell_cache_one(
    boxname: str,
    simname: str,
    iz: int,
    lightcone: int,
    outdir: str,
    stellar_cut: float = 8.0,
    overwrite: bool = False,
    seed_base: int = 10_000,
):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    outpath = outdir / f"shell_{iz:03d}.parquet"
    if outpath.exists() and not overwrite:
        return str(outpath)

    # create ps inside worker (process-safe)
    ps = patchyScreening(
        boxname,
        simname,
        iz,
        im=stellar_cut,                    # kept constant; not used for caching
        n_cut=0,
        ncpu=1,                # kept constant; not used for caching
        lightcone_method=("FULL", "shell"),
        lightcone=lightcone,
    )

    # minimal columns for downstream filtering/sampling
    ps.filter_stellar_mass()
    df = ps.merge

    # deterministic rand for fast/reproducible sampling later
    rng = np.random.default_rng(seed_base + int(iz))
    df = df.with_columns(pl.Series("rand", rng.random(df.height)))

    df.write_parquet(outpath, compression="zstd")
    return str(outpath)


def build_shell_cache(
    boxname: str,
    simname: str,
    lightcone: int,
    outdir: str,
    max_z: float = 3.0,
    stellar_cut: float = 8.0,
    ncpu: int = 8,
    prefer: str = "processes",
    verbose: int = 10,
    overwrite: bool = False,
):
    
    if not os.path.isfile(f'/cosma8/data/dp004/dc-conl1/FLAMINGO/patchy_screening/data_files/halo_redshifts/{boxname}/{simname}/lightcone{lightcone}/FLAMINGO_halo_redshift_values.txt'):
        multiprocess_z_bins(ncpu, boxname, simname, lightcone=lightcone)

    halo_z_bins = np.genfromtxt(
        f'/cosma8/data/dp004/dc-conl1/FLAMINGO/patchy_screening/data_files/halo_redshifts/{boxname}/{simname}/lightcone{lightcone}/FLAMINGO_halo_redshift_values.txt',
        dtype=[('i',   'i4'),
            ('z_min', 'f8'),
            ('mid_z', 'f8'),
            ('z_max', 'f8')],
        delimiter=None
    )

    # only shells where midpoint z < 3.0 (halo_lightcones-style)
    iz_list = sorted([iz for iz, z in zip(halo_z_bins['i'], halo_z_bins['mid_z']) if z < max_z])

    job_start_time = time.time()

    return Parallel(n_jobs=ncpu, prefer=prefer, verbose=verbose)(
        delayed(build_shell_cache_one)(
            boxname, simname, iz, lightcone, outdir,
            stellar_cut=stellar_cut, overwrite=overwrite
        )
        for iz in iz_list
    )

    # 3+ hours
    # for iz in iz_list:
    #     build_shell_cache_one(boxname, simname, iz, lightcone, outdir, stellar_cut=stellar_cut, overwrite=overwrite)

    print(f"Shell cache built in {time.time() - job_start_time:.1f} seconds")

    return


if __name__ == "__main__":
    import sys

    boxname  = sys.argv[2]  # <- fill in
    simname  = sys.argv[3]          # <- fill in
    iz = int(sys.argv[4])
    ncpu     = int(sys.argv[1])     # <- fill in
    lightcone = int(sys.argv[5])
    outdir   = f"/cosma8/data/dp004/dc-conl1/FLAMINGO/patchy_screening/data_files/shell_caches/{boxname}/{simname}/lightcone{lightcone}"

    build_shell_cache(
        boxname, simname, lightcone,
        outdir,
        max_z=3.0,
        stellar_cut=8.0,
        ncpu=ncpu,
        prefer="processes",
        overwrite=True
    )

    # build_shell_cache_one(boxname, simname, iz, lightcone, outdir, stellar_cut=8.0, overwrite=True)



