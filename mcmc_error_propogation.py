#!/usr/bin/env python3

import argparse
from pathlib import Path

import numpy as np
from mock_catalog_emulator import emulator


def run_emulator_posterior_predictions(
    chain_file,
    output_file,
    box,
    isim,
    iz,
    lightcone=0,
    abundance_cut=0.05,
    amp_min=10.3,
    amp_max=11.3,
    amp_step=0.1,
    slope_min=0.0,
    slope_max=1.0,
    slope_step=0.1,
    nsamples=5000,
    seed=1000,
):
    """
    Propagate MCMC parameter uncertainty into emulator power-spectrum uncertainty.

    Input chain_file should contain either:
      - flat_samples array with shape (Nsamples, 2), columns [amp, slope]
      - or an emcee chain with shape (Nsteps, Nwalkers, 2)

    Output is a compressed .npz containing:
      - theta_samples
      - auto_predictions
      - cross_predictions
      - auto_mean, auto_std, auto_p16, auto_p50, auto_p84
      - cross_mean, cross_std, cross_p16, cross_p50, cross_p84
      - prediction_at_mean_theta_auto
      - prediction_at_mean_theta_cross
    """

    rng = np.random.default_rng(seed)

    chain = np.load(chain_file)

    if chain.ndim == 3:
        # emcee chain: (steps, walkers, ndim)
        flat_samples = chain.reshape(-1, chain.shape[-1])
    elif chain.ndim == 2:
        # already flattened: (samples, ndim)
        flat_samples = chain
    else:
        raise ValueError(f"Expected chain with ndim 2 or 3, got shape {chain.shape}")

    if flat_samples.shape[1] != 2:
        raise ValueError(f"Expected chain columns [amp, slope], got shape {flat_samples.shape}")

    if nsamples is not None and nsamples < len(flat_samples):
        idx = rng.choice(len(flat_samples), size=nsamples, replace=False)
        theta_samples = flat_samples[idx]
    else:
        theta_samples = flat_samples

    auto_predictions = []
    cross_predictions = []

    for n, theta in enumerate(theta_samples):
        theta = np.asarray(theta, dtype=float)

        f_auto = emulator(
            theta,
            "auto",
            box,
            isim,
            iz,
            load=True,
            lightcone=lightcone,
            abundance_cut=abundance_cut,
            amp_min=amp_min,
            amp_max=amp_max,
            amp_step=amp_step,
            slope_min=slope_min,
            slope_max=slope_max,
            slope_step=slope_step,
        )

        f_cross = emulator(
            theta,
            "cross",
            box,
            isim,
            iz,
            load=True,
            lightcone=lightcone,
            abundance_cut=abundance_cut,
            amp_min=amp_min,
            amp_max=amp_max,
            amp_step=amp_step,
            slope_min=slope_min,
            slope_max=slope_max,
            slope_step=slope_step,
        )

        f_auto = np.asarray(f_auto, dtype=float)
        f_cross = np.asarray(f_cross, dtype=float)

        auto_predictions.append(f_auto)
        cross_predictions.append(f_cross)

        if (n + 1) % 100 == 0:
            print(f"[INFO] Finished {n + 1}/{len(theta_samples)} emulator evaluations")

    auto_predictions = np.asarray(auto_predictions)
    cross_predictions = np.asarray(cross_predictions)

    theta_mean = (np.percentile(theta_samples[:, 0], [50])[0], np.percentile(theta_samples[:, 1], [50])[0])

    auto_at_mean_theta = emulator(
        theta_mean,
        "auto",
        box,
        isim,
        iz,
        load=True,
        lightcone=lightcone,
        abundance_cut=abundance_cut,
        amp_min=amp_min,
        amp_max=amp_max,
        amp_step=amp_step,
        slope_min=slope_min,
        slope_max=slope_max,
        slope_step=slope_step,
    )

    cross_at_mean_theta = emulator(
        theta_mean,
        "cross",
        box,
        isim,
        iz,
        load=True,
        lightcone=lightcone,
        abundance_cut=abundance_cut,
        amp_min=amp_min,
        amp_max=amp_max,
        slope_min=slope_min,
        slope_max=slope_max,
        slope_step=slope_step,
    )

    auto_at_mean_theta = np.asarray(auto_at_mean_theta, dtype=float)
    cross_at_mean_theta = np.asarray(cross_at_mean_theta, dtype=float)


    auto_mean  = np.mean(auto_predictions, axis=0)
    auto_std   = np.std(auto_predictions, axis=0, ddof=1)
    auto_p16   = np.percentile(auto_predictions, 16, axis=0)
    auto_p50   = np.percentile(auto_predictions, 50, axis=0)
    auto_p84   = np.percentile(auto_predictions, 84, axis=0)

    cross_mean = np.mean(cross_predictions, axis=0)
    cross_std  = np.std(cross_predictions, axis=0, ddof=1)
    cross_p16  = np.percentile(cross_predictions, 16, axis=0)
    cross_p50  = np.percentile(cross_predictions, 50, axis=0)
    cross_p84  = np.percentile(cross_predictions, 84, axis=0)

    theta_quartiles  = (np.percentile(flat_samples[:, 0], [16, 50, 84]), np.percentile(flat_samples[:, 1], [16, 50, 84]))
    theta_1sigma = (np.diff(theta_quartiles[0]), np.diff(theta_quartiles[1]))

    print("\n[PARAMETER POSTERIOR]")
    print(f"amp   = {theta_mean[0]:.6f} - {theta_1sigma[0][0]:.6f} + - {theta_1sigma[0][1]:.6f}")
    print(f"slope = {theta_mean[1]:.6f} - {theta_1sigma[1][0]:.6f} + - {theta_1sigma[1][1]:.6f}")

    print("\n[AUTO POSTERIOR PREDICTIVE]")
    print(f"auto_predictions shape = {auto_predictions.shape}")
    print(f"fractional 1sigma error = {np.asarray(auto_std / auto_mean)}")
    print(f"mean fractional 1sigma error = {np.mean(auto_std / auto_mean):.6e}")
    print(f"max  fractional 1sigma error = {np.max(auto_std / auto_mean):.6e}")

    print("\n[CROSS POSTERIOR PREDICTIVE]")
    print(f"cross_predictions shape = {cross_predictions.shape}")
    print(f"fractional 1sigma error = {np.asarray(cross_std / cross_mean)}")
    print(f"mean fractional 1sigma error = {np.mean(cross_std / cross_mean):.6e}")
    print(f"max  fractional 1sigma error = {np.max(cross_std / cross_mean):.6e}")

    print("\n[FIRST 10 AUTO BINS]")
    for i in range(min(10, len(auto_mean))):
        print(
            f"bin {i:03d}: "
            f"mean={auto_mean[i]: .6e}, "
            f"std={auto_std[i]: .6e}, "
            f"p16={auto_p16[i]: .6e}, "
            f"p50={auto_p50[i]: .6e}, "
            f"p84={auto_p84[i]: .6e}"
        )

    print("\n[FIRST 10 CROSS BINS]")
    for i in range(min(10, len(cross_mean))):
        print(
            f"bin {i:03d}: "
            f"mean={cross_mean[i]: .6e}, "
            f"std={cross_std[i]: .6e}, "
            f"p16={cross_p16[i]: .6e}, "
            f"p50={cross_p50[i]: .6e}, "
            f"p84={cross_p84[i]: .6e}"
        )

    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    quit()

    np.savez_compressed(
        output_file,

        auto_mean=np.mean(auto_predictions, axis=0),
        auto_std=np.std(auto_predictions, axis=0, ddof=1),
        auto_p16=np.percentile(auto_predictions, 16, axis=0),
        auto_p50=np.percentile(auto_predictions, 50, axis=0),
        auto_p84=np.percentile(auto_predictions, 84, axis=0),

        cross_mean=np.mean(cross_predictions, axis=0),
        cross_std=np.std(cross_predictions, axis=0, ddof=1),
        cross_p16=np.percentile(cross_predictions, 16, axis=0),
        cross_p50=np.percentile(cross_predictions, 50, axis=0),
        cross_p84=np.percentile(cross_predictions, 84, axis=0),
    )

    print(f"[INFO] Saved posterior predictive spectra to {output_file}")
    print(f"[INFO] Mean theta: amp={theta_mean[0]:.6f}, slope={theta_mean[1]:.6f}")
    print("[INFO] Compare auto_mean/cross_mean with prediction_at_mean_theta_*.")
    print("[INFO] They need not match exactly if emulator(theta) is nonlinear.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("chain_file")
    parser.add_argument("output_file")
    parser.add_argument("box")
    parser.add_argument("isim")
    parser.add_argument("iz")
    parser.add_argument("--lightcone", type=int, default=0)
    parser.add_argument("--abundance-cut", type=float, default=0.5)

    parser.add_argument("--amp-min", type=float, default=10.3)
    parser.add_argument("--amp-max", type=float, default=11.3)
    parser.add_argument("--amp-step", type=float, default=0.1)
    parser.add_argument("--slope-min", type=float, default=0.0)
    parser.add_argument("--slope-max", type=float, default=1.0)
    parser.add_argument("--slope-step", type=float, default=0.1)

    parser.add_argument("--nsamples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=1000)

    args = parser.parse_args()

    run_emulator_posterior_predictions(
        chain_file=args.chain_file,
        output_file=args.output_file,
        box=args.box,
        isim=args.isim,
        iz=args.iz,
        lightcone=args.lightcone,
        abundance_cut=args.abundance_cut,
        amp_min=args.amp_min,
        amp_max=args.amp_max,
        amp_step=args.amp_step,
        slope_min=args.slope_min,
        slope_max=args.slope_max,
        slope_step=args.slope_step,
        nsamples=args.nsamples,
        seed=args.seed,
    )