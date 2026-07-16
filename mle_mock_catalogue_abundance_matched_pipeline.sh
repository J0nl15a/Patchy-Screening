#!/usr/bin/env bash
set -euo pipefail

BOX="${1:?usage: $0 BOX ISIM IZ [LIGHTCONE]}"
ISIM="${2:?usage: $0 BOX ISIM IZ [LIGHTCONE]}"
IZ="${3:?usage: $0 BOX ISIM IZ [LIGHTCONE]}"
LIGHTCONE="${4:-0}"

CLI_AMP="${5:-}"
CLI_SLOPE="${6:-}"

# Hard-coded stellar-cut grid
AMP_MIN="10.3"
AMP_MAX="11.3"
AMP_STEP="0.1"

SLOPE_MIN="0.0"
SLOPE_STEP="0.1"

if [ "$IZ" = "Green" ]; then
    SLOPE_MAX=1.0
elif [ "$IZ" = "Blue" ]; then
    SLOPE_MAX=2.0
else
    echo "ERROR: IZ must be 'Green' or 'Blue' (got '$IZ')" >&2
    exit 1
fi

ABUNDANCE_CUT="0.5"

# Fixed number of shell caches
NSHELL=60
NAMP=$(seq "$AMP_MIN" "$AMP_STEP" "$AMP_MAX" | wc -l)
NSLOPE=$(seq "$SLOPE_MIN" "$SLOPE_STEP" "$SLOPE_MAX" | wc -l)
NGRID=$(( NAMP * NSLOPE ))

if [ -d "./batch_files/caching_logs/${BOX}/${ISIM}/${IZ}/lightcone${LIGHTCONE}" ] && \
   [ -d "./batch_files/pipeline_logs/${BOX}/${ISIM}/${IZ}/lightcone${LIGHTCONE}" ] && \
   [ -d "./batch_files/maximum_likelihood_logs/${BOX}/${ISIM}/${IZ}/lightcone${LIGHTCONE}" ]; then
    echo "Directory exists."
else
    echo "Directory does not exist."
    mkdir -p "./batch_files/caching_logs/${BOX}/${ISIM}/${IZ}/lightcone${LIGHTCONE}"
    mkdir -p "./batch_files/pipeline_logs/${BOX}/${ISIM}/${IZ}/lightcone${LIGHTCONE}"
    mkdir -p "./batch_files/maximum_likelihood_logs/${BOX}/${ISIM}/${IZ}/lightcone${LIGHTCONE}"
fi

rm ./batch_files/caching_logs/${BOX}/${ISIM}/${IZ}/lightcone${LIGHTCONE}/job.*.dump || true
rm ./batch_files/caching_logs/${BOX}/${ISIM}/${IZ}/lightcone${LIGHTCONE}/job.*.err || true
rm ./batch_files/pipeline_logs/${BOX}/${ISIM}/${IZ}/lightcone${LIGHTCONE}/job.*.dump || true
rm ./batch_files/pipeline_logs/${BOX}/${ISIM}/${IZ}/lightcone${LIGHTCONE}/job.*.err || true
rm ./batch_files/maximum_likelihood_logs/${BOX}/${ISIM}/${IZ}/lightcone${LIGHTCONE}/job.*.dump || true
rm ./batch_files/maximum_likelihood_logs/${BOX}/${ISIM}/${IZ}/lightcone${LIGHTCONE}/job.*.err || true

USE_DIRECT_PARAMS=false
if [ -n "$CLI_AMP" ] && [ -n "$CLI_SLOPE" ]; then
    USE_DIRECT_PARAMS=true
elif [ -n "$CLI_AMP" ] || [ -n "$CLI_SLOPE" ]; then
    echo "ERROR: provide both AMP and SLOPE, or neither." >&2
    exit 1
fi

# ============================================================
# Step 0: shell caching
# ============================================================
jid0=$(sbatch --parsable \
              --export=ALL,BOX="${BOX}",ISIM="${ISIM}",IZ="${IZ}",LIGHTCONE="${LIGHTCONE}",NSHELL="${NSHELL}" \
              --job-name=shell_caching_stellar_cuts \
              -c 16 \
              -p cosma8 \
              -A dp004 \
              -t 01:00:00 \
              --array=0-$((NSHELL-1))%20 \
              -o "./batch_files/caching_logs/${BOX}/${ISIM}/${IZ}/lightcone${LIGHTCONE}/job.%j.dump" \
              -e "./batch_files/caching_logs/${BOX}/${ISIM}/${IZ}/lightcone${LIGHTCONE}/job.%j.err" \
              <<EOF
#!/usr/bin/env bash
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ARIJCONL@ljmu.ac.uk
set -euo pipefail

echo ">>> Launching caching with Box='${BOX}' Sim='${ISIM}' Lightcone='${LIGHTCONE}'"

module purge
set +u
[ -f "$HOME/.bashrc" ] && . "$HOME/.bashrc"
mamba activate patchy_screening
set -u

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

echo "=== Step 0 (Job ID \$SLURM_JOB_ID) starting"
echo "    Received arguments: ncpu='\$SLURM_CPUS_PER_TASK', Box='${BOX}', Sim='${ISIM}', Lightcone='${LIGHTCONE}'"
echo "=============================="

all_found=true
for i in \$(seq -f "%03g" 0 $((NSHELL - 1))); do
    [ -f "./data_files/shell_caches/${BOX}/${ISIM}/lightcone${LIGHTCONE}/shell_\${i}.parquet" ] || { all_found=false; break; }
done

if [ "\$all_found" = true ]; then
    echo "All shell cache files found — skipping build_cache.py"
else
    python3 FLAMINGO_galaxies.py "\$SLURM_CPUS_PER_TASK" "${BOX}" "${ISIM}" "\$SLURM_ARRAY_TASK_ID" "${LIGHTCONE}"
fi

echo "Job 0: Caching the galaxies from the FLAMINGO lightcone shells and catalogues with box ${BOX}, sim ${ISIM} & lightcone ${LIGHTCONE}."

echo "Job done, info follows."
sacct -j \$SLURM_JOB_ID --format=JobID,JobName,Partition,AveRSS,MaxRSS,AveVMSize,MaxVMSize,Elapsed,ExitCode

EOF
)

echo "Job 0: Caching lightcone shells for box ${BOX}, sim ${ISIM}, lightcone ${LIGHTCONE}, ${IZ} sample."

# Path to MLE output
MLE_FILE="./data_files/mle_parameters/${BOX}/${ISIM}/${IZ}/lightcone${LIGHTCONE}/mle_values.txt"

# ============================================================
# Step 2: stellar_cut_z for MLE values only
# ============================================================
jid2=$(sbatch --parsable \
                --export=ALL,BOX="${BOX}",ISIM="${ISIM}",IZ="${IZ}",LIGHTCONE="${LIGHTCONE}",MLE_FILE="${MLE_FILE}",CLI_AMP="${CLI_AMP}",CLI_SLOPE="${CLI_SLOPE}",USE_DIRECT_PARAMS="${USE_DIRECT_PARAMS}" \
                --dependency=afterok:${jid0} \
                --kill-on-invalid-dep=yes \
                --job-name=mle_stellar_cut \
                -c 1 \
                -p cosma8 \
                -A dp004 \
                -t 00:01:00 \
                -o "./batch_files/pipeline_logs/${BOX}/${ISIM}/${IZ}/lightcone${LIGHTCONE}/job.mle_%j.dump" \
                -e "./batch_files/pipeline_logs/${BOX}/${ISIM}/${IZ}/lightcone${LIGHTCONE}/job.mle_%j.err" \
<<'EOF'
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
[ -f "$HOME/.bashrc" ] && . "$HOME/.bashrc"
mamba activate patchy_screening
set -u

if [ "$USE_DIRECT_PARAMS" = true ]; then
    mle_amp="$CLI_AMP"
    mle_slope="$CLI_SLOPE"
else
    mle_amp=$(grep -m1 '^AMP' "$MLE_FILE" | cut -d'=' -f2 | xargs)
    mle_slope=$(grep -m1 '^SLOPE' "$MLE_FILE" | cut -d'=' -f2 | xargs)
fi

echo "=== Step 9 (Job ID $SLURM_JOB_ID) starting"
echo "    Received arguments: Box='$BOX', Sim='$ISIM', Lightcone='$LIGHTCONE', Sample='$IZ'"
echo "=============================="

if [ -f "./data_files/z_dependant_stellar_cuts/${BOX}/${IZ}/z_stellar_cut_data_${mle_amp//./p}_${mle_slope//./p}.txt" ]; then
    echo "Found stellar cut data for MLE values — skipping FLAMINGO_halo_lightcones_shell.py"
else
    python3 stellar_cut_z.py "$SLURM_CPUS_PER_TASK" "$BOX" "$ISIM" "$IZ" "$mle_amp" "$mle_slope" "$LIGHTCONE"
fi

echo "Job 9: Compute z-dependant stellar cut values for box $BOX, $IZ sample for MLE values only."

echo "Job done, info follows."
sacct -j $SLURM_JOB_ID --format=JobID,JobName,Partition,AveRSS,MaxRSS,AveVMSize,MaxVMSize,Elapsed,ExitCode

EOF
)

# ============================================================
# Step 3: Lightcones for MLE values only
# ============================================================
jid3=$(sbatch --parsable \
                --export=ALL,BOX="${BOX}",ISIM="${ISIM}",IZ="${IZ}",LIGHTCONE="${LIGHTCONE}",MLE_FILE="${MLE_FILE}",CLI_AMP="${CLI_AMP}",CLI_SLOPE="${CLI_SLOPE}",USE_DIRECT_PARAMS="${USE_DIRECT_PARAMS}" \
                --dependency=afterok:${jid2} --kill-on-invalid-dep=yes \
                --job-name=mle_lightcones_shells \
                -c 4 \
                -p cosma8 \
                -A dp004 \
                -t 00:20:00 \
                --array=0-$((NSHELL-1))%20 \
                -o "./batch_files/pipeline_logs/${BOX}/${ISIM}/${IZ}/lightcone${LIGHTCONE}/job.mle_%A_%a.dump" \
                -e "./batch_files/pipeline_logs/${BOX}/${ISIM}/${IZ}/lightcone${LIGHTCONE}/job.mle_%A_%a.err" \
<<'EOF'
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
[ -f "$HOME/.bashrc" ] && . "$HOME/.bashrc"
mamba activate patchy_screening
set -u

if [ "$USE_DIRECT_PARAMS" = true ]; then
    mle_amp="$CLI_AMP"
    mle_slope="$CLI_SLOPE"
else
    mle_amp=$(grep -m1 '^AMP' "$MLE_FILE" | cut -d'=' -f2 | xargs)
    mle_slope=$(grep -m1 '^SLOPE' "$MLE_FILE" | cut -d'=' -f2 | xargs)
fi

echo "=== Step 10 (Job ID $SLURM_JOB_ID) starting"
echo "    Received arguments: ncpu='$SLURM_CPUS_PER_TASK', Box='$BOX', Sim='$ISIM', Lightcone='$LIGHTCONE', Sample='$IZ'"
echo "=============================="

if [ -f "./data_files/halo_totals/${BOX}/${ISIM}/${IZ}/lightcone${LIGHTCONE}/FLAMINGO_halo_totals_${mle_amp//./p}_${mle_slope//./p}.txt" ]; then
    echo "Stellar cut totals found — skipping combine_halo_lightcones.py"
else
    python3 FLAMINGO_halo_lightcones_shell.py \
      "$SLURM_CPUS_PER_TASK" "$BOX" "$ISIM" "$IZ" "$SLURM_ARRAY_TASK_ID" \
      "$mle_amp" "$mle_amp" "0.1" \
      "$mle_slope" "$mle_slope" "0.1" \
      "$LIGHTCONE" False
fi

echo "Job 10: Generate lightcone shells for box $BOX, sim $ISIM, lightcone $LIGHTCONE, $IZ sample for MLE values only for all shells."

echo "Job done, info follows."
sacct -j $SLURM_JOB_ID --format=JobID,JobName,Partition,AveRSS,MaxRSS,AveVMSize,MaxVMSize,Elapsed,ExitCode

EOF
)

# ============================================================
# Step 4: Combine lightcones for MLE values only
# ============================================================
jid4=$(sbatch --parsable \
                --export=ALL,BOX="${BOX}",ISIM="${ISIM}",IZ="${IZ}",LIGHTCONE="${LIGHTCONE}",MLE_FILE="${MLE_FILE}",CLI_AMP="${CLI_AMP}",CLI_SLOPE="${CLI_SLOPE}",USE_DIRECT_PARAMS="${USE_DIRECT_PARAMS}" \
                --dependency=afterok:${jid3} \
                --kill-on-invalid-dep=yes \
                --job-name=mle_combine_lightcones_unWISE_matching \
                -c 4 \
                -p cosma8 \
                -A dp004 \
                -t 01:00:00 \
                -o "./batch_files/pipeline_logs/${BOX}/${ISIM}/${IZ}/lightcone${LIGHTCONE}/job.mle_%j.dump" \
                -e "./batch_files/pipeline_logs/${BOX}/${ISIM}/${IZ}/lightcone${LIGHTCONE}/job.mle_%j.err" \
<<'EOF'
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
[ -f "$HOME/.bashrc" ] && . "$HOME/.bashrc"
mamba activate patchy_screening
set -u

if [ "$USE_DIRECT_PARAMS" = true ]; then
    mle_amp="$CLI_AMP"
    mle_slope="$CLI_SLOPE"
else
    mle_amp=$(grep -m1 '^AMP' "$MLE_FILE" | cut -d'=' -f2 | xargs)
    mle_slope=$(grep -m1 '^SLOPE' "$MLE_FILE" | cut -d'=' -f2 | xargs)
fi

echo "=== Step 11 (Job ID $SLURM_JOB_ID) starting"
echo "    Received arguments: Box='$BOX', Sim='$ISIM', Lightcone='$LIGHTCONE', Sample='$IZ'"
echo "=============================="

if [ -f "./data_files/halo_totals/${BOX}/${ISIM}/${IZ}/lightcone${LIGHTCONE}/FLAMINGO_halo_totals_${mle_amp//./p}_${mle_slope//./p}.txt" ]; then
    echo "Stellar cut totals found — skipping combine_halo_lightcones.py"
else
    python3 combine_halo_lightcones.py \
      "$BOX" "$ISIM" "$IZ" "$LIGHTCONE" False

    echo "Deleting MLE shell_*.txt files..."
    rm ./data_files/halo_totals/${BOX}/${ISIM}/${IZ}/lightcone${LIGHTCONE}/shell_*_mle.txt 2>/dev/null || true
fi

echo "Job 11: Combine halo lightcone shells for box $BOX, sim $ISIM, lightcone $LIGHTCONE, $IZ sample for MLE values only into single halo totals files."

echo "=== Step 12 (Job ID $SLURM_JOB_ID) starting"
echo "    Received arguments: Box='$BOX', Sim='$ISIM', Lightcone='$LIGHTCONE', Sample='$IZ', nsamp='ntotal'"
echo "=============================="

python3 unWISE_data_matching.py "$BOX" "$ISIM" "$IZ" "$mle_amp" "$mle_slope" ntotal "$LIGHTCONE"

echo "Job 12: Match FLAMINGO halo catalogs to unWISE dN/dz for box $BOX, sim $ISIM, lightcone $LIGHTCONE, $IZ sample for MLE values only."

echo "Job done, info follows."
sacct -j $SLURM_JOB_ID --format=JobID,JobName,Partition,AveRSS,MaxRSS,AveVMSize,MaxVMSize,Elapsed,ExitCode

EOF
)

# ============================================================
# Step 5: sampling for MLE values only
# ============================================================
jid5=$(sbatch --parsable \
                --export=ALL,BOX="${BOX}",ISIM="${ISIM}",IZ="${IZ}",LIGHTCONE="${LIGHTCONE}",MLE_FILE="${MLE_FILE}",NSHELL="${NSHELL}",CLI_AMP="${CLI_AMP}",CLI_SLOPE="${CLI_SLOPE}",USE_DIRECT_PARAMS="${USE_DIRECT_PARAMS}" \
                --dependency=afterok:${jid4} \
                --kill-on-invalid-dep=yes \
                --job-name=mle_sampling_shells \
                -c 4 \
                -p cosma8 \
                -A dp004 \
                -t 01:00:00 \
                --array=0-$((NSHELL-1))%20 \
                -o "./batch_files/pipeline_logs/${BOX}/${ISIM}/${IZ}/lightcone${LIGHTCONE}/job.mle_%A_%a.dump" \
                -e "./batch_files/pipeline_logs/${BOX}/${ISIM}/${IZ}/lightcone${LIGHTCONE}/job.mle_%A_%a.err" \
<<'EOF'
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
[ -f "$HOME/.bashrc" ] && . "$HOME/.bashrc"
mamba activate patchy_screening
set -u

if [ "$USE_DIRECT_PARAMS" = true ]; then
    mle_amp="$CLI_AMP"
    mle_slope="$CLI_SLOPE"
else
    mle_amp=$(grep -m1 '^AMP' "$MLE_FILE" | cut -d'=' -f2 | xargs)
    mle_slope=$(grep -m1 '^SLOPE' "$MLE_FILE" | cut -d'=' -f2 | xargs)
fi

echo "=== Step 13 (Job ID $SLURM_JOB_ID) starting"
echo "    Received arguments: ncpu='$SLURM_CPUS_PER_TASK', Box='$BOX', Sim='$ISIM', Lightcone='$LIGHTCONE', Sample='$IZ'"
echo "=============================="

all_found_mle_sampling=true
for i in $(seq -w 0 $((NSHELL - 1))); do
    [ -f "./data_files/mock_halo_catalogs/${BOX}/${ISIM}/${IZ}/lightcone${LIGHTCONE}/mle/shell_${i}_mle.parquet" ] || {
        all_found_mle_sampling=false
        break
    }
done

if [ "$all_found_mle_sampling" = true ]; then
    echo "Found sampled catalogs for MLE values — skipping FLAMINGO_halo_sampling_shell_abundance_matched.py"
else
    python3 FLAMINGO_halo_sampling_shell_abundance_matched.py \
      "$BOX" "$ISIM" "$IZ" "$SLURM_ARRAY_TASK_ID" "$LIGHTCONE"
fi

echo "Job 13: Sample halo lightcone shells for box $BOX, sim $ISIM, lightcone $LIGHTCONE, $IZ sample for MLE values only for all shells into mock galaxy catalogs."

echo "Job done, info follows."
sacct -j $SLURM_JOB_ID --format=JobID,JobName,Partition,AveRSS,MaxRSS,AveVMSize,MaxVMSize,Elapsed,ExitCode

EOF
)

# ============================================================
# Step 6: combine sampling for MLE values only
# ============================================================
jid6=$(sbatch --parsable \
                --export=ALL,BOX="${BOX}",ISIM="${ISIM}",IZ="${IZ}",LIGHTCONE="${LIGHTCONE}",MLE_FILE="${MLE_FILE}",CLI_AMP="${CLI_AMP}",CLI_SLOPE="${CLI_SLOPE}",USE_DIRECT_PARAMS="${USE_DIRECT_PARAMS}" \
                --dependency=afterok:${jid5} \
                --kill-on-invalid-dep=yes \
                --job-name=mle_combine_sampling_power_spectra \
                -c 16 \
                -p cosma8 \
                -A dp004 \
                -t 04:00:00 \
                -o "./batch_files/pipeline_logs/${BOX}/${ISIM}/${IZ}/lightcone${LIGHTCONE}/job.mle_%j.dump" \
                -e "./batch_files/pipeline_logs/${BOX}/${ISIM}/${IZ}/lightcone${LIGHTCONE}/job.mle_%j.err" \
<<'EOF'
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
[ -f "$HOME/.bashrc" ] && . "$HOME/.bashrc"
mamba activate patchy_screening
set -u

if [ "$USE_DIRECT_PARAMS" = true ]; then
    mle_amp="$CLI_AMP"
    mle_slope="$CLI_SLOPE"
else
    mle_amp=$(grep -m1 '^AMP' "$MLE_FILE" | cut -d'=' -f2 | xargs)
    mle_slope=$(grep -m1 '^SLOPE' "$MLE_FILE" | cut -d'=' -f2 | xargs)
fi

echo "=== Step 14 (Job ID $SLURM_JOB_ID) starting"
echo "    Received arguments: Box='$BOX', Sim='$ISIM', Lightcone='$LIGHTCONE', Sample='$IZ'"
echo "=============================="

python3 combine_halo_sampling_abundance_matched.py \
      "$BOX" "$ISIM" "$IZ" "$LIGHTCONE"

dir="./data_files/mock_halo_catalogs/${BOX}/${ISIM}/${IZ}/lightcone${LIGHTCONE}/mle"

if [ -d "$dir" ]; then
    rm -r "$dir"
    echo "Deleted $dir"
fi

echo "Job 14: Combine sampled halo lightcone shells into single mock galaxy catalogs for box $BOX, sim $ISIM, lightcone $LIGHTCONE, $IZ sample for MLE values only."

echo "=== Step 15 (Job ID $SLURM_JOB_ID) starting"
echo "    Received arguments: Box='$BOX', Sim='$ISIM', Lightcone='$LIGHTCONE', Sample='$IZ', nsamp='ntotal'"
echo "=============================="


python3 unWISE_power_spectra.py \
    "$SLURM_CPUS_PER_TASK" "$BOX" "$ISIM" "$IZ" "$mle_amp" "$mle_slope" \
    unlensed True False False True True False False "$LIGHTCONE" False

echo "Job 15: Compute power spectra for box $BOX, sim $ISIM, lightcone $LIGHTCONE, $IZ sample for MLE values only."

echo "Job done, info follows."
sacct -j $SLURM_JOB_ID --format=JobID,JobName,Partition,AveRSS,MaxRSS,AveVMSize,MaxVMSize,Elapsed,ExitCode

EOF
)

echo "Chain:"
echo "${jid0} -> ${jid2} -> ${jid3} -> ${jid4} -> ${jid5} -> ${jid6}"