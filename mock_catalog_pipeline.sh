#!/usr/bin/env bash
set -euo pipefail

ISIM="${1:?usage: $0 ISIM IZ}"
IZ="${2:?usage: $0 ISIM IZ}"

ARR_ID=$(sbatch --parsable \
                --array=0-120%30 \
                --job-name=mle_pipeline_array \
                -c 128 \
                -p cosma8 \
                -A dp004 \
                -t 12:00:00 \
                -o ./batch_files/pipeline_logs/job.${ISIM}_${IZ}_%j.dump \
                -e ./batch_files/pipeline_logs/job.${ISIM}_${IZ}_%j.err \
                <<EOF
#!/usr/bin/env bash
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ARIJCONL@ljmu.ac.uk

set -euo pipefail

# grid sizes
A_SIZE=11   # number of amp values: 10.3..11.3 step 0.1
S_SIZE=11   # number of slope values: 0.0..1.0 step 0.1

# decode array index -> (amp, slope)
A_IDX=\$(( SLURM_ARRAY_TASK_ID / S_SIZE ))
S_IDX=\$(( SLURM_ARRAY_TASK_ID % S_SIZE ))

amp=\$(awk -v i="\$A_IDX" 'BEGIN{printf "%.1f", 10.3 + 0.1*i}')
slope=\$(awk -v i="\$S_IDX" 'BEGIN{printf "%.1f", 0.0 + 0.1*i}')

amp_name=\${amp//./p}
slope_name=\${slope//./p}

echo "Task \$SLURM_ARRAY_TASK_ID -> amp=\$amp slope=\$slope"

echo ">>> Launching pipeline with Sim='${ISIM}' Sample='${IZ}' M_cut(z_mean)='\$amp' n_cut='\$slope' nsamp='ntotal'"

module purge
set +u
if [ -f "$HOME/.bashrc" ]; then
    . "$HOME/.bashrc"
fi
mamba activate patchy_screening
set -u

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

echo "=== Step 1 (Job ID \$SLURM_JOB_ID) starting"
echo "    Received arguments: Sample='${IZ}', M_cut(z_mean)='\$amp', n_cut='\$slope'"
echo "=============================="

if [ -f "./data_files/z_dependant_stellar_cuts/z_stellar_cut_data_${IZ}_\${amp_name}_\${slope_name}.txt" ]; then
    echo "Found ./data_files/z_dependant_stellar_cuts/z_stellar_cut_data_${IZ}_\${amp_name}_\${slope_name}.txt — skipping stellar_cut_z.py"
else
    python3 stellar_cut_z.py "${IZ}" "\$amp" "\$slope"
fi

echo "Job 1: Compute z-dependant stellar cut values for ${IZ} sample with: M_cut(z_mean) = \$amp, n_cut = \$slope"

echo "=== Step 2 (Job ID \$SLURM_JOB_ID) starting"
echo "    Received arguments: ncpu='\$SLURM_CPUS_PER_TASK', Sim='${ISIM}', Sample='${IZ}', M_cut(z_mean)='\$amp', n_cut='\$slope'"
echo "=============================="

if [ -f "./data_files/halo_totals/FLAMINGO_halo_totals_${ISIM}_${IZ}_\${amp_name}_\${slope_name}.txt" ]; then
    echo "Found ./data_files/halo_totals/FLAMINGO_halo_totals_${ISIM}_${IZ}_\${amp_name}_\${slope_name}.txt — skipping FLAMINGO_halo_lightcones.py"
else
    python3 FLAMINGO_halo_lightcones.py "\$SLURM_CPUS_PER_TASK" "${ISIM}" "${IZ}" "\$amp" "\$slope"
fi

echo "Job 2: Collate halos in lightcone shells for sim ${ISIM}, ${IZ} sample with: M_cut(z_mean) = \$amp, n_cut = \$slope."

echo "=== Step 3 (Job ID \$SLURM_JOB_ID) starting"
echo "    Received arguments: Sim='${ISIM}', Sample='${IZ}', M_cut(z_mean)='\$amp', n_cut='\$slope', nsamp='ntotal'"
echo "=============================="

if [ -f "./data_files/dndz_samples/dndz_galaxies_sampled_${ISIM}_${IZ}_\${amp_name}_\${slope_name}.txt" ]; then
    echo "Found ./data_files/dndz_samples/dndz_galaxies_sampled_${ISIM}_${IZ}_\${amp_name}_\${slope_name}.txt — skipping unWISE_data_matching.py"
else
    python3 unWISE_data_matching.py "${ISIM}" "${IZ}" "\$amp" "\$slope" "ntotal"
fi

echo "Job 3: Rescale unWISE dndz curve for sim ${ISIM}, ${IZ} sample and stellar cut with: M_cut(z_mean) = \$amp, n_cut = \$slope."

echo "=== Step 4 (Job ID \$SLURM_JOB_ID) starting"
echo "    Received arguments: ncpu='\$SLURM_CPUS_PER_TASK', Sim='${ISIM}', Sample='${IZ}', M_cut(z_mean)='\$amp', n_cut='\$slope'"
echo "=============================="

if [ -f "./data_files/mock_halo_catalogs/sampled_halo_data_${ISIM}_${IZ}_\${amp_name}_\${slope_name}.parquet" ]; then
    echo "Found ./data_files/mock_halo_catalogs/sampled_halo_data_${ISIM}_${IZ}_\${amp_name}_\${slope_name}.parquet — skipping FLAMINGO_halo_sampling.py"
else
    python3 FLAMINGO_halo_sampling.py "\$SLURM_CPUS_PER_TASK" "${ISIM}" "${IZ}" "\$amp" "\$slope"
fi

echo "Job 4: Sampling FLAMINGO halo lightcones using rescaled unWISE dndz curve for sim ${ISIM}, ${IZ} sample and stellar cut with: M_cut(z_mean) = \$amp, n_cut = \$slope."

echo "=== Step 5 (Job ID \$SLURM_JOB_ID) starting"
echo "    Received arguments: ncpu='\$SLURM_CPUS_PER_TASK', Sim='${ISIM}', Sample='${IZ}', M_cut(z_mean)='\$amp', n_cut='\$slope'"
echo "=============================="

if [ -f "./data_files/power_spectra/galaxy_galaxy/${ISIM}_${IZ}_\${amp_name}_\${slope_name}.txt" ] || \
   [ -f "./data_files/power_spectra/kappa_galaxy/${ISIM}_${IZ}_\${amp_name}_\${slope_name}.txt" ]; then
    echo "Found existing galaxy-galaxy or kappa-galaxy spectra file — skipping unWISE_power_spectra.py"
else
    python3 unWISE_power_spectra.py "\$SLURM_CPUS_PER_TASK" "${ISIM}" "${IZ}" "\$amp" "\$slope" unlensed True False True True True False
fi

echo "Job 5: Computing the clustering and lensing cross-spectra of FLAMINGO unWISE mock catalogs for sim ${ISIM}, ${IZ} sample and stellar cut with: M_cut(z_mean) = \$amp, n_cut = \$slope."

echo "Job done, info follows."
sacct -j \$SLURM_JOB_ID --format=JobID,JobName,Partition,AveRSS,MaxRSS,AveVMSize,MaxVMSize,Elapsed,ExitCode

EOF
)

echo "Submitted array: $ARR_ID"

# 6) submit the MLE job to run AFTER all array tasks finish successfully
jid6=$(sbatch --parsable \
              --dependency=afterok:$ARR_ID \
              --kill-on-invalid-dep=yes \
              --job-name=mock_catalog_maximum_likelihood_estimation \
              -c 8 \
              -p cosma8 \
              -A dp004 \
              -t 00:30:00 \
              -o ./batch_files/maximum_likelihood_logs/job.${ISIM}_${IZ}_%j.dump \
              -e ./batch_files/maximum_likelihood_logs/job.${ISIM}_${IZ}_%j.err \
              <<EOF
#!/usr/bin/env bash
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ARIJCONL@ljmu.ac.uk

set -euo pipefail

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

module purge
set +u
if [ -f "$HOME/.bashrc" ]; then
    . "$HOME/.bashrc"
fi
mamba activate patchy_screening
set -u

echo "=== Step 6 (Job ID \$SLURM_JOB_ID) starting"
echo "    Received arguments: ncpu="\$SLURM_CPUS_PER_TASK", Sim='${ISIM}', Sample='${IZ}'"
echo "=============================="

python mock_catalog_likelihood_parallel.py "\$SLURM_CPUS_PER_TASK" "${ISIM}" "${IZ}"

echo "Job done, info follows."
sacct -j \$SLURM_JOB_ID --format=JobID,JobName,Partition,AveRSS,MaxRSS,AveVMSize,MaxVMSize,Elapsed,ExitCode

EOF
)
       
echo "Job 6: Computing the maximum likelihood estimates for the M_cut and n_cut values of the mock catalogs for sim ${ISIM}, ${IZ} sample."

# variables you already know in this script
MLE_FILE="./data_files/mle_values_${ISIM}_${IZ}.txt"

# 7) Launch step 7 after job‐ID=$jid6 succeeds
jid7=$(sbatch --parsable \
              --dependency=afterok:${jid6} \
              --kill-on-invalid-dep=yes \
              --job-name=mle_mock_catalog \
              -c 128 \
              -p cosma8 \
              -A dp004 \
              -t 01:30:00 \
              -o ./batch_files/pipeline_logs/job.mle_${ISIM}_${IZ}_%j.dump \
              -e ./batch_files/pipeline_logs/job.mle_${ISIM}_${IZ}_%j.err \
              <<EOF
#!/usr/bin/env bash
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ARIJCONL@ljmu.ac.uk

set -euo pipefail

# Read MLEs from file written by the MLE job (use only the first matching line)
amp=\$(grep -m1 '^AMP'   "${MLE_FILE}" | cut -d'=' -f2)
slope=\$(grep -m1 '^SLOPE' "${MLE_FILE}" | cut -d'=' -f2)
echo "Read MLEs: amp=\$amp, slope=\$slope"

echo ">>> Launching pipeline with Sim='${ISIM}' Sample='${IZ}' M_cut(z_mean)='\$amp' n_cut='\$slope' nsamp='ntotal'"

module purge
set +u
if [ -f "$HOME/.bashrc" ]; then
    . "$HOME/.bashrc"
fi
mamba activate patchy_screening
set -u

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

echo "=== Step 1 (Job ID \$SLURM_JOB_ID) starting"
echo "    Received arguments: Sample='${IZ}', M_cut(z_mean)='\$amp', n_cut='\$slope'"
echo "=============================="

python3 stellar_cut_z.py "${IZ}" "\$amp" "\$slope"

echo "Job 1: Compute z-dependant stellar cut values for ${IZ} sample with: M_cut(z_mean) = \$amp, n_cut = \$slope"

echo "=== Step 2 (Job ID \$SLURM_JOB_ID) starting"
echo "    Received arguments: ncpu='\$SLURM_CPUS_PER_TASK', Sim='${ISIM}', Sample='${IZ}', M_cut(z_mean)='\$amp', n_cut='\$slope'"
echo "=============================="

python3 FLAMINGO_halo_lightcones.py "\$SLURM_CPUS_PER_TASK" "${ISIM}" "${IZ}" "\$amp" "\$slope"

echo "Job 2: Collate halos in lightcone shells for sim ${ISIM}, ${IZ} sample with: M_cut(z_mean) = \$amp, n_cut = \$slope."

echo "=== Step 3 (Job ID \$SLURM_JOB_ID) starting"
echo "    Received arguments: Sim='${ISIM}', Sample='${IZ}', M_cut(z_mean)='\$amp', n_cut='\$slope', nsamp='ntotal'"
echo "=============================="

python3 unWISE_data_matching.py "${ISIM}" "${IZ}" "\$amp" "\$slope" "ntotal"

echo "Job 3: Rescale unWISE dndz curve for sim ${ISIM}, ${IZ} sample and stellar cut with: M_cut(z_mean) = \$amp, n_cut = \$slope."

echo "=== Step 4 (Job ID \$SLURM_JOB_ID) starting"
echo "    Received arguments: ncpu='\$SLURM_CPUS_PER_TASK', Sim='${ISIM}', Sample='${IZ}', M_cut(z_mean)='\$amp', n_cut='\$slope'"
echo "=============================="

python3 FLAMINGO_halo_sampling.py "\$SLURM_CPUS_PER_TASK" "${ISIM}" "${IZ}" "\$amp" "\$slope"

echo "Job 4: Sampling FLAMINGO halo lightcones using rescaled unWISE dndz curve for sim ${ISIM}, ${IZ} sample and stellar cut with: M_cut(z_mean) = \$amp, n_cut = \$slope."

echo "=== Step 5 (Job ID \$SLURM_JOB_ID) starting"
echo "    Received arguments: ncpu='\$SLURM_CPUS_PER_TASK', Sim='${ISIM}', Sample='${IZ}', M_cut(z_mean)='\$amp', n_cut='\$slope'"
echo "=============================="

python3 unWISE_power_spectra.py "\$SLURM_CPUS_PER_TASK" "${ISIM}" "${IZ}" "\$amp" "\$slope" unlensed True False True True True False

echo "Job 5: Computing the clustering and lensing cross-spectra of FLAMINGO unWISE mock catalogs for sim ${ISIM}, ${IZ} sample and stellar cut with: M_cut(z_mean) = \$amp, n_cut = \$slope."

echo "Job done, info follows."
sacct -j \$SLURM_JOB_ID --format=JobID,JobName,Partition,AveRSS,MaxRSS,AveVMSize,MaxVMSize,Elapsed,ExitCode

EOF
)

echo "Job 7: Computing mock catalog and power spectra with MLE stellar cut for sim ${ISIM}, ${IZ} sample."

echo "Chain: $ARR_ID (array) -> $jid6 (MLE) -> $jid7 (final catalog generation)"