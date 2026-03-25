import sys
from pathlib import Path
import numpy as np, pandas as pd

def fmt_name(x, mle=False):
    if mle:
        return f"{x:.3f}".replace(".", "p")
    else:
        return f"{x:.1f}".replace(".", "p")

def process_shell(ncpu, box, sim, z_sample, iz, amp_min, amp_max, amp_step, slope_min, slope_max, slope_step, lightcone, mle=False):
    shell_file = Path(
        f"./data_files/shell_caches/{box}/{sim}/lightcone{lightcone}/shell_{iz:03d}.parquet"
    )
    df = pd.read_parquet(shell_file)
    mstar = df["mstar"].to_numpy()

    if mle:
        amp_values = [amp_min]
        slope_values = [slope_min]
        mle_suffix = "_mle"
    else:
        amp_values = np.round(np.arange(amp_min, amp_max + 0.001, amp_step), 1)
        slope_values = np.round(np.arange(slope_min, slope_max + 0.001, slope_step), 1)
        mle_suffix = ""

    results = []

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

            # nhalo = df[df["mstar"] >= 10**float(im_shell)].shape[0]
            nhalo = np.count_nonzero(mstar >= 10**float(im_shell))
            results.append((mass_cut, n_cut, nhalo))

    outdir = Path(
        f"./data_files/halo_totals/{box}/{sim}/{z_sample}/lightcone{lightcone}"
    )
    outdir.mkdir(parents=True, exist_ok=True)

    outfile = outdir / f"shell_{iz:03d}{mle_suffix}.txt"
    np.savetxt(outfile, np.array(results), fmt=["%.1f", "%.1f", "%d"] if not mle else ["%.3f", "%.3f", "%d"],
               header="mass_cut n_cut nhalo", comments="")
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

    process_shell(ncpu, box, sim, z_sample, iz, amp_min, amp_max, amp_step, slope_min, slope_max, slope_step, lightcone, mle)