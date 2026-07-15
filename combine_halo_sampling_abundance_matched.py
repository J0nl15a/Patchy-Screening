import sys
from pathlib import Path
import numpy as np
import pandas as pd


def combine_shells_abundance_matched(
    box,
    sim,
    z_sample,
    lightcone,
):
    shell_base = Path(
        f"./data_files/mock_halo_catalogs/{box}/{sim}/{z_sample}/lightcone{lightcone}"
    )

    if not shell_base.exists():
        raise FileNotFoundError(f"Missing directory: {shell_base}")

    catalogue_name = "0p0_0p0_abundance_matched"
    cut_dirs = [shell_base / catalogue_name]

    outdir = shell_base
    outdir.mkdir(parents=True, exist_ok=True)

    for cut_dir in cut_dirs:
        if not cut_dir.exists():
            print(f"Skipping missing directory: {cut_dir}")
            continue

        shell_files = sorted(cut_dir.glob("shell_*.parquet"))

        if not shell_files:
            print(f"Skipping {cut_dir}: no shell_*.parquet files found")
            continue

        dfs = [pd.read_parquet(f) for f in shell_files]

        combined = pd.concat(dfs, ignore_index=True)

        outfile = outdir / "sampled_halo_data_0p0_0p0.parquet"

        combined.to_parquet(
            outfile,
            index=False,
            engine="pyarrow",
            compression="snappy",
        )

        print(f"Combined {len(shell_files)} shell files")
        print(f"Total rows: {len(combined)}")
        print(f"Saved {outfile}")


if __name__ == "__main__":
    box = sys.argv[1]
    sim = sys.argv[2]
    z_sample = sys.argv[3]
    lightcone = int(sys.argv[4])

    combine_shells_abundance_matched(
        box,
        sim,
        z_sample,
        lightcone,
    )