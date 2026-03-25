import sys, time
from pathlib import Path
import numpy as np

def fmt_name(x, mle=False):
    if mle:
        return f"{x:.3f}".replace(".", "p")
    else:
        return f"{x:.1f}".replace(".", "p")

def combine_shells(box, sim, z_sample, lightcone, mle=False):
    shell_dir = Path(
        f"./data_files/halo_totals/{box}/{sim}/{z_sample}/lightcone{lightcone}"
    )

    if mle:
        shell_files = sorted(shell_dir.glob("shell_*_mle.txt"))
    else:
        shell_files = sorted(shell_dir.glob("shell_*.txt"))

    if not shell_files:
        raise FileNotFoundError(f"No shell files found in {shell_dir}")

    data = []
    for f in shell_files:
        arr = np.loadtxt(f, skiprows=1)
        arr = np.atleast_2d(arr)
        iz = int(f.stem.split("_")[1])
        for row in arr:
            mass_cut, n_cut, nhalo = row
            data.append((iz, mass_cut, n_cut, int(nhalo)))

    data = np.array(data, dtype=[("iz", int), ("mass_cut", float), ("n_cut", float), ("nhalo", int)])

    outdir = Path(
        f"./data_files/halo_totals/{box}/{sim}/{z_sample}/lightcone{lightcone}"
    )
    outdir.mkdir(parents=True, exist_ok=True)

    cuts = sorted(set((r["mass_cut"], r["n_cut"]) for r in data))
    for mass_cut, n_cut in cuts:
        mask = (data["mass_cut"] == mass_cut) & (data["n_cut"] == n_cut)
        sub = np.sort(data[mask], order="iz")

        amp_name = fmt_name(mass_cut, mle)
        slope_name = fmt_name(n_cut, mle)
        outfile = outdir / f"FLAMINGO_halo_totals_{amp_name}_{slope_name}.txt"

        out = np.column_stack([sub["iz"], sub["nhalo"]])
        total_nhalo = np.sum(sub["nhalo"])

        np.savetxt(outfile, out, fmt="%d %d",
                   header=f"Total number of suitable halos: {total_nhalo}",
                   comments="")
        print(f"Saved {outfile}")

if __name__ == "__main__":
    box = sys.argv[1]
    sim = sys.argv[2]
    z_sample = sys.argv[3]
    lightcone = int(sys.argv[4])
    mle = sys.argv[5].lower() in ("true", "1", "yes", "y")
    
    combine_shells(box, sim, z_sample, lightcone, mle)