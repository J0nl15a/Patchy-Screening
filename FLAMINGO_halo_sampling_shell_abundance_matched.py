import sys
from pathlib import Path
import numpy as np
import pandas as pd


def process_shell_no_stellar_cut(
    box,
    sim,
    z_sample,
    iz,
    lightcone,
    amp_name="0p0",
    slope_name="0p0",
):
    shell_file = Path(
        f"./data_files/shell_caches/{box}/{sim}/lightcone{lightcone}/shell_{iz:03d}.parquet"
    )

    df = pd.read_parquet(shell_file)

    dndz_file = Path(
        f"./data_files/dndz_samples/{box}/{sim}/{z_sample}/lightcone{lightcone}/"
        f"dndz_galaxies_sampled_{amp_name}_{slope_name}.txt"
    )

    dndz_sample = np.loadtxt(dndz_file)
    nsamp = int(dndz_sample[iz][1])

    outbase = Path(
        f"./data_files/mock_halo_catalogs/{box}/{sim}/{z_sample}/lightcone{lightcone}"
    )

    outdir = outbase / f"{amp_name}_{slope_name}_abundance_matched"
    outdir.mkdir(parents=True, exist_ok=True)

    if nsamp == 0:
        sampled = df.head(0)
    else:
        if nsamp > len(df):
            raise ValueError(
                f"Shell {iz}: requested {nsamp} halos from no-cut dN/dz, "
                f"but only {len(df)} halos are available."
            )

        sampled = (
            df.sort_values("mstar", ascending=False)
              .head(nsamp)
              .reset_index(drop=True)
        )

    outfile = outdir / f"shell_{iz:03d}.parquet"
    sampled.to_parquet(outfile, index=False, engine="pyarrow", compression="snappy")

    if len(sampled) > 0:
        print(
            f"iz={iz}, nsamp={nsamp}, "
            f"selected_min_mstar={sampled['mstar'].min():.6e}, "
            f"selected_max_mstar={sampled['mstar'].max():.6e}"
        )
    else:
        print(f"iz={iz}, nsamp=0, wrote empty shell")

    print(f"Saved {outfile}")


if __name__ == "__main__":
    box = sys.argv[1]
    sim = sys.argv[2]
    z_sample = sys.argv[3]
    iz = int(sys.argv[4])
    lightcone = int(sys.argv[5])

    process_shell_no_stellar_cut(
        box,
        sim,
        z_sample,
        iz,
        lightcone,
    )