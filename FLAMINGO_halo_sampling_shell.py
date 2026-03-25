import sys
from pathlib import Path
import numpy as np
import pandas as pd

def fmt_name(x, mle=False):
    if mle:
        return f"{x:.3f}".replace(".", "p")
    else:
        return f"{x:.1f}".replace(".", "p")

def process_shell(ncpu, box, sim, z_sample, iz, amp_min, amp_max, amp_step,
                  slope_min, slope_max, slope_step, lightcone, mle=False):
    shell_file = Path(
        f"./data_files/shell_caches/{box}/{sim}/lightcone{lightcone}/shell_{iz:03d}.parquet"
    )
    df = pd.read_parquet(shell_file)
    mstar = df["mstar"].to_numpy()

    if mle:
        amp_values = [amp_min]
        slope_values = [slope_min]
    else:
        amp_values = np.round(np.arange(amp_min, amp_max + 0.5 * amp_step, amp_step), 1)
        slope_values = np.round(np.arange(slope_min, slope_max + 0.5 * slope_step, slope_step), 1)

    outbase = Path(
        f"./data_files/mock_halo_catalogs/{box}/{sim}/{z_sample}/lightcone{lightcone}"
    )
    outbase.mkdir(parents=True, exist_ok=True)

    for mass_cut in amp_values:
        for n_cut in slope_values:
            amp_name = fmt_name(mass_cut, mle)
            slope_name = fmt_name(n_cut, mle)

            cut_file = Path(
                f"./data_files/z_dependant_stellar_cuts/{box}/{z_sample}/"
                f"z_stellar_cut_data_{amp_name}_{slope_name}.txt"
            )
            z_stellar_cuts = np.loadtxt(cut_file)
            im_shell = z_stellar_cuts[iz][1]

            dndz_file = Path(
                f"./data_files/dndz_samples/{box}/{sim}/{z_sample}/lightcone{lightcone}/"
                f"dndz_galaxies_sampled_{amp_name}_{slope_name}.txt"
            )
            dndz_sample = np.loadtxt(dndz_file)
            nsamp = int(dndz_sample[iz][1])

            #filtered = df.filter(pl.col("mstar") >= 10**float(im_shell))
            # filtered = df[df["mstar"] >= 10**float(im_shell)]
            mask = (mstar >= 10**float(im_shell))
            filtered = df.loc[mask]
            threshold = 10**float(im_shell)

            if not filtered.empty:
                print(
                    f"iz={iz}, mass_cut={mass_cut}, n_cut={n_cut}, "
                    f"im_shell={im_shell:.6f}, threshold={threshold:.6e}, "
                    f"filtered_min={filtered['mstar'].min():.6e}"
                )    

            if nsamp == 0:
                # sampled = filtered.head(0)
                sampled = filtered.head(0)  # Creates an empty DataFrame with the same columns as filtered
            else:
                if nsamp > filtered.shape[0]:#.height:
                    raise ValueError(
                        f"Shell {iz}, cut ({mass_cut}, {n_cut}): "
                        f"requested {nsamp} halos but only {filtered.shape[0]} available."
                    )
                sampled = filtered.sample(n=nsamp, replace=False, random_state=1000) #filtered.sample(n=nsamp, with_replacement=False, shuffle=True, seed=1000)

            if mle:
                outdir = outbase / f"mle"
            else:
                outdir = outbase / f"{amp_name}_{slope_name}"
            outdir.mkdir(parents=True, exist_ok=True)

            outfile = outdir / f"shell_{iz:03d}.parquet"
            # sampled.write_parquet(outfile)
            sampled.to_parquet(outfile, index=False, engine="pyarrow", compression="snappy")
            print(f"Saved {outfile}")

if __name__ == "__main__":
    ncpu = sys.argv[1]
    box = sys.argv[2]
    sim = sys.argv[3]
    z_sample = sys.argv[4]
    iz = int(sys.argv[5])
    amp_min = float(sys.argv[6])
    amp_max = float(sys.argv[7])
    amp_step = float(sys.argv[8])
    slope_min = float(sys.argv[9])
    slope_max = float(sys.argv[10])
    slope_step = float(sys.argv[11])
    lightcone = int(sys.argv[12])
    mle = sys.argv[13].lower() in ("true", "1", "yes", "y")

    process_shell(
        ncpu, box, sim, z_sample, iz,
        amp_min, amp_max, amp_step,
        slope_min, slope_max, slope_step,
        lightcone, mle
    )
