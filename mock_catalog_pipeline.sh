#!/usr/bin/env bash
set -euo pipefail

BOX="${1:?usage: $0 BOX ISIM IZ}"
ISIM="${2:?usage: $0 BOX ISIM IZ}"
IZ="${3:?usage: $0 BOX ISIM IZ}"
LIGHTCONE="${4:-0}"

if [ -d "./batch_files/pipeline_logs/${BOX}/${ISIM}/${IZ}/lightcone${LIGHTCONE}" ] && \
   [ -d "./batch_files/maximum_likelihood_logs/${BOX}/${ISIM}/${IZ}/lightcone${LIGHTCONE}" ]; then
    echo "Directory exists."
else
    echo "Directory does not exist."
    mkdir -p "./batch_files/pipeline_logs/${BOX}/${ISIM}/${IZ}/lightcone${LIGHTCONE}"
    mkdir -p "./batch_files/maximum_likelihood_logs/${BOX}/${ISIM}/${IZ}/lightcone${LIGHTCONE}"
fi

rm ./batch_files/pipeline_logs/${BOX}/${ISIM}/${IZ}/lightcone${LIGHTCONE}/job.*.dump || true
rm ./batch_files/pipeline_logs/${BOX}/${ISIM}/${IZ}/lightcone${LIGHTCONE}/job.*.err || true
rm ./batch_files/maximum_likelihood_logs/${BOX}/${ISIM}/${IZ}/lightcone${LIGHTCONE}/job.*.dump || true
rm ./batch_files/maximum_likelihood_logs/${BOX}/${ISIM}/${IZ}/lightcone${LIGHTCONE}/job.*.err || true

ARR_ID=$(sbatch --parsable \
                --array=0-120%30 \
                --job-name=mle_pipeline_array \
                -c 128 \
                -p cosma8 \
                -A dp004 \
                -t 12:00:00 \
                -o ./batch_files/pipeline_logs/${BOX}/${ISIM}/${IZ}/lightcone${LIGHTCONE}/job.%j.dump \
                -e ./batch_files/pipeline_logs/${BOX}/${ISIM}/${IZ}/lightcone${LIGHTCONE}/job.%j.err \
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

echo ">>> Launching pipeline with Box='${BOX}' Sim='${ISIM}' Sample='${IZ}' Lightcone='${LIGHTCONE}' M_cut(z_mean)='\$amp' n_cut='\$slope' nsamp='ntotal'"

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
echo "    Received arguments: Box='${BOX}', Sample='${IZ}', M_cut(z_mean)='\$amp', n_cut='\$slope'"
echo "=============================="

if [ -f "./data_files/z_dependant_stellar_cuts/${BOX}/${IZ}/z_stellar_cut_data_\${amp_name}_\${slope_name}.txt" ]; then
    echo "Found ./data_files/z_dependant_stellar_cuts/${BOX}/${IZ}/z_stellar_cut_data_\${amp_name}_\${slope_name}.txt — skipping stellar_cut_z.py"
else
    python3 stellar_cut_z.py "\$SLURM_CPUS_PER_TASK" "${BOX}" "${ISIM}" "${IZ}" "\$amp" "\$slope" "${LIGHTCONE}"
fi

echo "Job 1: Compute z-dependant stellar cut values for box ${BOX}, ${IZ} sample with: M_cut(z_mean) = \$amp, n_cut = \$slope"

echo "=== Step 2 (Job ID \$SLURM_JOB_ID) starting"
echo "    Received arguments: ncpu='\$SLURM_CPUS_PER_TASK', Box='${BOX}', Sim='${ISIM}', Lightcone='${LIGHTCONE}', Sample='${IZ}', M_cut(z_mean)='\$amp', n_cut='\$slope'"
echo "=============================="

if [ -f "./data_files/halo_totals/${BOX}/${ISIM}/${IZ}/lightcone${LIGHTCONE}/FLAMINGO_halo_totals_\${amp_name}_\${slope_name}.txt" ]; then
    echo "Found ./data_files/halo_totals/${BOX}/${ISIM}/${IZ}/lightcone${LIGHTCONE}/FLAMINGO_halo_totals_\${amp_name}_\${slope_name}.txt — skipping FLAMINGO_halo_lightcones.py"
else
    python3 FLAMINGO_halo_lightcones.py "\$SLURM_CPUS_PER_TASK" "${BOX}" "${ISIM}" "${IZ}" "\$amp" "\$slope" "${LIGHTCONE}"
fi

echo "Job 2: Collate halos in lightcone shells for box ${BOX}, sim ${ISIM}, lightcone ${LIGHTCONE}, ${IZ} sample with: M_cut(z_mean) = \$amp, n_cut = \$slope."

echo "=== Step 3 (Job ID \$SLURM_JOB_ID) starting"
echo "    Received arguments: Box='${BOX}', Sim='${ISIM}', Lightcone='${LIGHTCONE}', Sample='${IZ}', M_cut(z_mean)='\$amp', n_cut='\$slope', nsamp='ntotal'"
echo "=============================="

if [ -f "./data_files/dndz_samples/${BOX}/${ISIM}/${IZ}/lightcone${LIGHTCONE}/dndz_galaxies_sampled_\${amp_name}_\${slope_name}.txt" ]; then
    echo "Found ./data_files/dndz_samples/${BOX}/${ISIM}/${IZ}/lightcone${LIGHTCONE}/dndz_galaxies_sampled_\${amp_name}_\${slope_name}.txt — skipping unWISE_data_matching.py"
else
    python3 unWISE_data_matching.py "${BOX}" "${ISIM}" "${IZ}" "\$amp" "\$slope" "ntotal" "${LIGHTCONE}"
fi

echo "Job 3: Rescale unWISE dndz curve for box ${BOX}, sim ${ISIM}, lightcone ${LIGHTCONE}, ${IZ} sample and stellar cut with: M_cut(z_mean) = \$amp, n_cut = \$slope."

echo "=== Step 4 (Job ID \$SLURM_JOB_ID) starting"
echo "    Received arguments: ncpu='\$SLURM_CPUS_PER_TASK', Box='${BOX}', Sim='${ISIM}', Lightcone='${LIGHTCONE}', Sample='${IZ}', M_cut(z_mean)='\$amp', n_cut='\$slope'"
echo "=============================="

if [ -f "./data_files/mock_halo_catalogs/${BOX}/${ISIM}/${IZ}/lightcone${LIGHTCONE}/sampled_halo_data_\${amp_name}_\${slope_name}.parquet" ]; then
    echo "Found ./data_files/mock_halo_catalogs/${BOX}/${ISIM}/${IZ}/lightcone${LIGHTCONE}/sampled_halo_data_\${amp_name}_\${slope_name}.parquet — skipping FLAMINGO_halo_sampling.py"
else
    python3 FLAMINGO_halo_sampling.py "\$SLURM_CPUS_PER_TASK" "${BOX}" "${ISIM}" "${IZ}" "\$amp" "\$slope" "${LIGHTCONE}"
fi

echo "Job 4: Sampling FLAMINGO halo lightcones using rescaled unWISE dndz curve for box ${BOX}, sim ${ISIM}, lightcone ${LIGHTCONE}, ${IZ} sample and stellar cut with: M_cut(z_mean) = \$amp, n_cut = \$slope."

echo "=== Step 5 (Job ID \$SLURM_JOB_ID) starting"
echo "    Received arguments: ncpu='\$SLURM_CPUS_PER_TASK', Box='${BOX}', Sim='${ISIM}', Lightcone='${LIGHTCONE}', Sample='${IZ}', M_cut(z_mean)='\$amp', n_cut='\$slope'"
echo "=============================="

if [ -f "./data_files/power_spectra/galaxy_galaxy/${BOX}/${ISIM}/${IZ}/lightcone${LIGHTCONE}/galaxy_galaxy_power_spectrum_\${amp_name}_\${slope_name}.txt" ] && \
   [ -f "./data_files/power_spectra/kappa_galaxy/${BOX}/${ISIM}/${IZ}/lightcone${LIGHTCONE}/kappa_galaxy_power_spectrum_\${amp_name}_\${slope_name}.txt" ]; then
    echo "Found existing galaxy-galaxy and kappa-galaxy spectra file — skipping unWISE_power_spectra.py"
else
    python3 unWISE_power_spectra.py "\$SLURM_CPUS_PER_TASK" "${BOX}" "${ISIM}" "${IZ}" "\$amp" "\$slope" unlensed True False True True True False "${LIGHTCONE}"
fi

echo "Job 5: Computing the clustering and lensing cross-spectra of FLAMINGO unWISE mock catalogs for box ${BOX}, sim ${ISIM}, lightcone ${LIGHTCONE}, ${IZ} sample and stellar cut with: M_cut(z_mean) = \$amp, n_cut = \$slope."

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
              -o ./batch_files/maximum_likelihood_logs/${BOX}/${ISIM}/${IZ}/lightcone${LIGHTCONE}/job.%j.dump \
              -e ./batch_files/maximum_likelihood_logs/${BOX}/${ISIM}/${IZ}/lightcone${LIGHTCONE}/job.%j.err \
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
echo "    Received arguments: ncpu="\$SLURM_CPUS_PER_TASK", Box='${BOX}', Sim='${ISIM}', Lightcone='${LIGHTCONE}', Sample='${IZ}'"
echo "=============================="

python mock_catalog_likelihood_parallel.py "\$SLURM_CPUS_PER_TASK" "${BOX}" "${ISIM}" "${IZ}" "${LIGHTCONE}"

echo "Job done, info follows."
sacct -j \$SLURM_JOB_ID --format=JobID,JobName,Partition,AveRSS,MaxRSS,AveVMSize,MaxVMSize,Elapsed,ExitCode

EOF
)

echo "Job 6: Computing the maximum likelihood estimates for the M_cut and n_cut values of the mock catalogs for box ${BOX}, sim ${ISIM}, lightcone ${LIGHTCONE}, ${IZ} sample."

# variables you already know in this script
MLE_FILE="./data_files/mle_parameters/${BOX}/${ISIM}/${IZ}/lightcone${LIGHTCONE}/mle_values.txt"

# 7) Launch step 7 after job‐ID=$jid6 succeeds
jid7=$(sbatch --parsable \
              --dependency=afterok:${jid6} \
              --kill-on-invalid-dep=yes \
              --job-name=mle_mock_catalog \
              -c 128 \
              -p cosma8 \
              -A dp004 \
              -t 01:30:00 \
              -o ./batch_files/pipeline_logs/${BOX}/${ISIM}/${IZ}/lightcone${LIGHTCONE}/job.mle_%j.dump \
              -e ./batch_files/pipeline_logs/${BOX}/${ISIM}/${IZ}/lightcone${LIGHTCONE}/job.mle_%j.err \
              <<EOF
#!/usr/bin/env bash
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ARIJCONL@ljmu.ac.uk

set -euo pipefail

# Read MLEs from file written by the MLE job (use only the first matching line)
amp=\$(grep -m1 '^AMP'   "${MLE_FILE}" | cut -d'=' -f2)
slope=\$(grep -m1 '^SLOPE' "${MLE_FILE}" | cut -d'=' -f2)
echo "Read MLEs: amp=\$amp, slope=\$slope"

echo ">>> Launching pipeline with Box='${BOX}' Sim='${ISIM}' Lightcone='${LIGHTCONE}' Sample='${IZ}' M_cut(z_mean)='\$amp' n_cut='\$slope' nsamp='ntotal'"

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
echo "    Received arguments: Box='${BOX}', Sample='${IZ}', M_cut(z_mean)='\$amp', n_cut='\$slope'"
echo "=============================="

python3 stellar_cut_z.py "\$SLURM_CPUS_PER_TASK" "${BOX}" "${ISIM}" "${IZ}" "\$amp" "\$slope" "${LIGHTCONE}"

echo "Job 1: Compute z-dependant stellar cut values for box ${BOX}, ${IZ} sample with: M_cut(z_mean) = \$amp, n_cut = \$slope"

echo "=== Step 2 (Job ID \$SLURM_JOB_ID) starting"
echo "    Received arguments: ncpu='\$SLURM_CPUS_PER_TASK', Box='${BOX}', Sim='${ISIM}', Lightcone='${LIGHTCONE}', Sample='${IZ}', M_cut(z_mean)='\$amp', n_cut='\$slope'"
echo "=============================="

python3 FLAMINGO_halo_lightcones.py "\$SLURM_CPUS_PER_TASK" "${BOX}" "${ISIM}" "${IZ}" "\$amp" "\$slope" "${LIGHTCONE}"

echo "Job 2: Collate halos in lightcone shells for box ${BOX}, sim ${ISIM}, lightcone ${LIGHTCONE}, ${IZ} sample with: M_cut(z_mean) = \$amp, n_cut = \$slope."

echo "=== Step 3 (Job ID \$SLURM_JOB_ID) starting"
echo "    Received arguments: Box='${BOX}', Sim='${ISIM}', Lightcone='${LIGHTCONE}', Sample='${IZ}', M_cut(z_mean)='\$amp', n_cut='\$slope', nsamp='ntotal'"
echo "=============================="

python3 unWISE_data_matching.py "${BOX}" "${ISIM}" "${IZ}" "\$amp" "\$slope" "ntotal" "${LIGHTCONE}"

echo "Job 3: Rescale unWISE dndz curve for box ${BOX}, sim ${ISIM}, lightcone ${LIGHTCONE}, ${IZ} sample and stellar cut with: M_cut(z_mean) = \$amp, n_cut = \$slope."

echo "=== Step 4 (Job ID \$SLURM_JOB_ID) starting"
echo "    Received arguments: ncpu='\$SLURM_CPUS_PER_TASK', Box='${BOX}', Sim='${ISIM}', Lightcone='${LIGHTCONE}', Sample='${IZ}', M_cut(z_mean)='\$amp', n_cut='\$slope'"
echo "=============================="

python3 FLAMINGO_halo_sampling.py "\$SLURM_CPUS_PER_TASK" "${BOX}" "${ISIM}" "${IZ}" "\$amp" "\$slope" "${LIGHTCONE}"

echo "Job 4: Sampling FLAMINGO halo lightcones using rescaled unWISE dndz curve for box ${BOX}, sim ${ISIM}, lightcone ${LIGHTCONE}, ${IZ} sample and stellar cut with: M_cut(z_mean) = \$amp, n_cut = \$slope."

echo "=== Step 5 (Job ID \$SLURM_JOB_ID) starting"
echo "    Received arguments: ncpu='\$SLURM_CPUS_PER_TASK', Box='${BOX}', Sim='${ISIM}', Lightcone='${LIGHTCONE}', Sample='${IZ}', M_cut(z_mean)='\$amp', n_cut='\$slope'"
echo "=============================="

python3 unWISE_power_spectra.py "\$SLURM_CPUS_PER_TASK" "${BOX}" "${ISIM}" "${IZ}" "\$amp" "\$slope" unlensed True False True True True False "${LIGHTCONE}"

echo "Job 5: Computing the clustering and lensing cross-spectra of FLAMINGO unWISE mock catalogs for box ${BOX}, sim ${ISIM}, lightcone ${LIGHTCONE}, ${IZ} sample and stellar cut with: M_cut(z_mean) = \$amp, n_cut = \$slope."

echo "Job done, info follows."
sacct -j \$SLURM_JOB_ID --format=JobID,JobName,Partition,AveRSS,MaxRSS,AveVMSize,MaxVMSize,Elapsed,ExitCode

EOF
)

echo "Job 7: Computing mock catalog and power spectra with MLE stellar cut for box ${BOX}, sim ${ISIM}, lightcone ${LIGHTCONE}, ${IZ} sample."

echo "Chain: $ARR_ID (array) -> $jid6 (MLE) -> $jid7 (final catalog generation)"