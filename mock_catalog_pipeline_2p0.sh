#!/usr/bin/env bash
set -euo pipefail

BOX="${1:?usage: $0 BOX ISIM IZ}"
ISIM="${2:?usage: $0 BOX ISIM IZ}"
IZ="${3:?usage: $0 BOX ISIM IZ}"
LIGHTCONE="${4:-0}"

rm ./batch_files/pipeline_logs/${BOX}/${ISIM}/${IZ}/lightcone${LIGHTCONE}/job.*_2p0.dump || true
rm ./batch_files/pipeline_logs/${BOX}/${ISIM}/${IZ}/lightcone${LIGHTCONE}/job.*_2p0.err || true

ARR_ID=$(sbatch --parsable \
                --array=0-109%10 \
                --job-name=mle_pipeline_array \
                -c 16 \
                -p cosma8 \
                -A dp004 \
                -t 24:00:00 \
                -o ./batch_files/pipeline_logs/${BOX}/${ISIM}/${IZ}/lightcone${LIGHTCONE}/job.%j_2p0.dump \
                -e ./batch_files/pipeline_logs/${BOX}/${ISIM}/${IZ}/lightcone${LIGHTCONE}/job.%j_2p0.err \
                <<EOF
#!/usr/bin/env bash
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ARIJCONL@ljmu.ac.uk

set -euo pipefail

# grid sizes
A_SIZE=11    # number of amplitude values: 10.7, 10.8
S_SIZE=10   # number of slope values: 0.0..1.0 step 0.1

# decode array index -> (amp, slope)
A_IDX=\$(( SLURM_ARRAY_TASK_ID / S_SIZE ))
S_IDX=\$(( SLURM_ARRAY_TASK_ID % S_SIZE ))

amp=\$(awk -v i="\$A_IDX" 'BEGIN{printf "%.1f", 10.3 + 0.1*i}')
slope=\$(awk -v i="\$S_IDX" 'BEGIN{printf "%.1f", 1.1 + 0.1*i}')

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

echo "Chain: $ARR_ID (array)"
