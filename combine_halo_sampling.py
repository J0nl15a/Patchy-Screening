import sys
from pathlib import Path
import numpy as np, pandas as pd

def fmt_name(x, mle=False):
    if mle:
        return f"{x:.3f}".replace(".", "p")
    else:
        return f"{x:.1f}".replace(".", "p")

def combine_shells(box, sim, z_sample, lightcone, mle=False, custom_amp=False, custom_slope=False):
    shell_base = Path(
        f"./data_files/mock_halo_catalogs/{box}/{sim}/{z_sample}/lightcone{lightcone}"
    )

    if not shell_base.exists():
        raise FileNotFoundError(f"Missing directory: {shell_base}")

    if mle:
        cut_dirs = [shell_base / "mle"]  # Only one set of cuts for MLE, so we can just use the base directory
        amp = np.loadtxt(f"./data_files/mle_parameters/{box}/{sim}/{z_sample}/lightcone{lightcone}/mle_values.txt", usecols=1, skiprows=1, max_rows=1, delimiter='=') 
        slope = np.loadtxt(f"./data_files/mle_parameters/{box}/{sim}/{z_sample}/lightcone{lightcone}/mle_values.txt", usecols=1, skiprows=2, max_rows=1, delimiter='=')
        amp_name = fmt_name(amp, mle)
        slope_name = fmt_name(slope, mle)
    elif not mle and custom_amp != False and custom_slope != False:
        cut_dirs = [shell_base / "mle"]
        amp = custom_amp
        slope = custom_slope
        amp_name = fmt_name(amp, True)
        slope_name = fmt_name(slope, True)
    else:
        cut_dirs = sorted([p for p in shell_base.iterdir() if p.is_dir()])

    outdir = Path(
        f"./data_files/mock_halo_catalogs/{box}/{sim}/{z_sample}/lightcone{lightcone}"
    )
    outdir.mkdir(parents=True, exist_ok=True)

    for cut_dir in cut_dirs:
        shell_files = sorted(cut_dir.glob("shell_*.parquet"))
        if not shell_files:
            continue

        # dfs = [pd.read_parquet(f) for f in shell_files]
        dfs = []
        for f in shell_files:
            dfs.append(pd.read_parquet(f))
        combined = pd.concat(dfs, ignore_index=True)
        # combined = pd.concat(dfs, axis=0, ignore_index=True) #how="vertical")

        if mle or (custom_amp != False and custom_slope != False):
            pass
        else:
            amp_name, slope_name = cut_dir.name.split("_", 1)
        outfile = outdir / f"sampled_halo_data_{amp_name}_{slope_name}.parquet"

        # combined.write_parquet(outfile)
        combined.to_parquet(outfile, index=False, engine="pyarrow", compression="snappy")
        print(f"Saved {outfile}")

if __name__ == "__main__":
    box = sys.argv[1]
    sim = sys.argv[2]
    z_sample = sys.argv[3]
    lightcone = int(sys.argv[4])
    mle = sys.argv[5].lower() in ("true", "1", "yes", "y")
    custom_amp = float(sys.argv[6]) if len(sys.argv) > 6 else False
    custom_slope = float(sys.argv[7]) if len(sys.argv) > 7 else False

    combine_shells(box, sim, z_sample, lightcone, mle, custom_amp, custom_slope)
