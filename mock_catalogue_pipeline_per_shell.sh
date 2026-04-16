#!/usr/bin/env bash
set -euo pipefail

BOX="${1:?usage: $0 BOX ISIM IZ [LIGHTCONE]}"
ISIM="${2:?usage: $0 BOX ISIM IZ [LIGHTCONE]}"
IZ="${3:?usage: $0 BOX ISIM IZ [LIGHTCONE]}"
LIGHTCONE="${4:-0}"

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

PRECISION="1"
ABUNDANCE_CUT="0.02"

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

# ============================================================
# Step 0: shell caching
# ============================================================
jid0=$(sbatch --parsable \
              --export=ALL,BOX="${BOX}",ISIM="${ISIM}",IZ="${IZ}",LIGHTCONE="${LIGHTCONE}",AMP_MIN="${AMP_MIN}",AMP_MAX="${AMP_MAX}",AMP_STEP="${AMP_STEP}",SLOPE_MIN="${SLOPE_MIN}",SLOPE_MAX="${SLOPE_MAX}",SLOPE_STEP="${SLOPE_STEP}",NSHELL="${NSHELL}" \
              --job-name=shell_caching_stellar_cuts \
              -c 16 \
              -p cosma8 \
              -A dp203 \
              -t 01:00:00 \
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
conda activate patchy_screening
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
    python3 FLAMINGO_galaxies.py "\$SLURM_CPUS_PER_TASK" "${BOX}" "${ISIM}" "${LIGHTCONE}"
fi

echo "Job 0: Caching the galaxies from the FLAMINGO lightcone shells and catalogues with box ${BOX}, sim ${ISIM} & lightcone ${LIGHTCONE}."

echo "=== Step 1 (Job ID \$SLURM_JOB_ID) starting"
echo "    Received arguments: Box='${BOX}', Sample='${IZ}', Lightcone='${LIGHTCONE}'"
echo "=============================="

for amp in \$(seq "${AMP_MIN}" "${AMP_STEP}" "${AMP_MAX}"); do
  for slope in \$(seq "${SLOPE_MIN}" "${SLOPE_STEP}" "${SLOPE_MAX}"); do
    amp_fmt=\$(printf "%.1f" "\$amp")
    slope_fmt=\$(printf "%.1f" "\$slope")
    amp_name=\${amp_fmt//./p}
    slope_name=\${slope_fmt//./p}

    if [ -f "./data_files/z_dependant_stellar_cuts/${BOX}/${IZ}/z_stellar_cut_data_\${amp_name}_\${slope_name}.txt" ]; then
        echo "Found stellar cut for amp=\$amp_name slope=\$slope_name — skipping stellar_cut_z.py"
    else
        python3 stellar_cut_z.py \$SLURM_CPUS_PER_TASK "${BOX}" "${ISIM}" "${IZ}" "\$amp_fmt" "\$slope_fmt" "${LIGHTCONE}"
    fi
  done
done

echo "Job 1: Compute z-dependant stellar cut values for box ${BOX}, ${IZ} sample"

echo "Job done, info follows."
sacct -j \$SLURM_JOB_ID --format=JobID,JobName,Partition,AveRSS,MaxRSS,AveVMSize,MaxVMSize,Elapsed,ExitCode

EOF
)

echo "Job 0: Caching lightcone shells for box ${BOX}, sim ${ISIM}, lightcone ${LIGHTCONE}, ${IZ} sample."

# ============================================================
# Step 2: Lightcone shell array
# ============================================================
jid2="${jid0}"
all_found_lightcones=true

for amp in $(seq "$AMP_MIN" "$AMP_STEP" "$AMP_MAX"); do
  for slope in $(seq "$SLOPE_MIN" "$SLOPE_STEP" "$SLOPE_MAX"); do
    amp_fmt=$(printf "%.1f" "$amp")
    slope_fmt=$(printf "%.1f" "$slope")

    amp_name=${amp_fmt//./p}
    slope_name=${slope_fmt//./p}

    file="./data_files/halo_totals/${BOX}/${ISIM}/${IZ}/lightcone${LIGHTCONE}/FLAMINGO_halo_totals_${amp_name}_${slope_name}.txt"

    if [ ! -f "$file" ]; then
      all_found_lightcones=false
      echo "Missing: $file"
      break 2
    fi
  done
done

if [ "$all_found_lightcones" = true ]; then
    echo "Found all shell totals — skipping FLAMINGO_halo_lightcones_shell.py"
else
    jid2=$(sbatch --parsable \
                  --export=ALL,BOX="${BOX}",ISIM="${ISIM}",IZ="${IZ}",LIGHTCONE="${LIGHTCONE}",AMP_MIN="${AMP_MIN}",AMP_MAX="${AMP_MAX}",AMP_STEP="${AMP_STEP}",SLOPE_MIN="${SLOPE_MIN}",SLOPE_MAX="${SLOPE_MAX}",SLOPE_STEP="${SLOPE_STEP}" \
                  --dependency=afterok:${jid0} \
                  --kill-on-invalid-dep=yes \
                  --job-name=lightcones_shells \
                  -c 4 \
                  -p cosma8 \
                  -A dp203 \
                  -t 00:20:00 \
                  --array=0-$((NSHELL-1))%20 \
                  -o "./batch_files/pipeline_logs/${BOX}/${ISIM}/${IZ}/lightcone${LIGHTCONE}/job.%A_%a.dump" \
                  -e "./batch_files/pipeline_logs/${BOX}/${ISIM}/${IZ}/lightcone${LIGHTCONE}/job.%A_%a.err" \
<<EOF
#!/usr/bin/env bash
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ARIJCONL@ljmu.ac.uk
set -euo pipefail

module purge
set +u
[ -f "$HOME/.bashrc" ] && . "$HOME/.bashrc"
conda activate patchy_screening
set -u

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

echo "=== Step 2 (Job ID \$SLURM_JOB_ID) starting"
echo "    Received arguments: ncpu='\$SLURM_CPUS_PER_TASK', Box='${BOX}', Sim='${ISIM}', Lightcone='${LIGHTCONE}', Sample='${IZ}'"
echo "=============================="

python3 FLAMINGO_halo_lightcones_shell.py \
    \$SLURM_CPUS_PER_TASK "${BOX}" "${ISIM}" "${IZ}" "\$SLURM_ARRAY_TASK_ID" \
    "${AMP_MIN}" "${AMP_MAX}" "${AMP_STEP}" \
    "${SLOPE_MIN}" "${SLOPE_MAX}" "${SLOPE_STEP}" \
    "${LIGHTCONE}" False

echo "Job 2: Generate lightcone shells for box ${BOX}, sim ${ISIM}, lightcone ${LIGHTCONE}, ${IZ} sample for all shells and stellar cut parameters."

echo "Job done, info follows."
sacct -j \$SLURM_JOB_ID --format=JobID,JobName,Partition,AveRSS,MaxRSS,AveVMSize,MaxVMSize,Elapsed,ExitCode

EOF
    )

    echo "Submitted step 2: ${jid2}"
fi

# ============================================================
# Step 3: Combine halo totals
# ============================================================
jid3="${jid2}"
all_found_dndz=true

for amp in $(seq "$AMP_MIN" "$AMP_STEP" "$AMP_MAX"); do
  for slope in $(seq "$SLOPE_MIN" "$SLOPE_STEP" "$SLOPE_MAX"); do
    amp_fmt=$(printf "%.1f" "$amp")
    slope_fmt=$(printf "%.1f" "$slope")

    amp_name=${amp_fmt//./p}
    slope_name=${slope_fmt//./p}

    file="./data_files/dndz_samples/${BOX}/${ISIM}/${IZ}/lightcone${LIGHTCONE}/dndz_galaxies_sampled_${amp_name}_${slope_name}.txt"

    if [ ! -f "$file" ]; then
      all_found_dndz=false
      echo "Missing: $file"
      break 2
    fi
  done
done

if [ "$all_found_lightcones" = true ] &&
   [ "$all_found_dndz" = true ]; then
        echo "Found matched catalog for — skipping unWISE_data_matching.py"
else
  jid3=$(sbatch --parsable \
                --export=ALL,BOX="${BOX}",ISIM="${ISIM}",IZ="${IZ}",LIGHTCONE="${LIGHTCONE}",AMP_MIN="${AMP_MIN}",AMP_MAX="${AMP_MAX}",AMP_STEP="${AMP_STEP}",SLOPE_MIN="${SLOPE_MIN}",SLOPE_MAX="${SLOPE_MAX}",SLOPE_STEP="${SLOPE_STEP}",NSHELL="${NSHELL}",PREV_ARRAY_ID="${jid2}",all_found_lightcones="${all_found_lightcones}",all_found_dndz="${all_found_dndz}" \
                --dependency=afterok:${jid2} \
                --kill-on-invalid-dep=yes \
                --job-name=combine_lightcones_unWISE_matching \
                -c 4 \
                -p cosma8 \
                -A dp203 \
                -t 01:00:00 \
                -o "./batch_files/pipeline_logs/${BOX}/${ISIM}/${IZ}/lightcone${LIGHTCONE}/job.%j.dump" \
                -e "./batch_files/pipeline_logs/${BOX}/${ISIM}/${IZ}/lightcone${LIGHTCONE}/job.%j.err" \
<<EOF
#!/usr/bin/env bash
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ARIJCONL@ljmu.ac.uk
set -euo pipefail

module purge
set +u
[ -f "$HOME/.bashrc" ] && . "$HOME/.bashrc"
conda activate patchy_screening
set -u

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

echo "=== Step 3 (Job ID \$SLURM_JOB_ID) starting"
echo "    Received arguments: Box='${BOX}', Sim='${ISIM}', Lightcone='${LIGHTCONE}', Sample='${IZ}'"
echo "=============================="

if [ "${all_found_lightcones}" = true ]; then
    echo "Stellar cut totals found — skipping combine_halo_lightcones.py"
else
    python3 combine_halo_lightcones.py \
        "${BOX}" "${ISIM}" "${IZ}" "${LIGHTCONE}" False

    echo "Deleting intermediate shell_*.txt files..."
    rm ./data_files/halo_totals/${BOX}/${ISIM}/${IZ}/lightcone${LIGHTCONE}/shell_*.txt
fi

echo "Job 3: Combine halo lightcone shells for box ${BOX}, sim ${ISIM}, lightcone ${LIGHTCONE}, ${IZ} sample for all stellar cut parameters into single halo totals files."

for ((i=0; i<=NSHELL-2; i++)); do
    rm -f "./batch_files/pipeline_logs/${BOX}/${ISIM}/${IZ}/lightcone${LIGHTCONE}/job.\${PREV_ARRAY_ID}_\${i}."dump
    rm -f "./batch_files/pipeline_logs/${BOX}/${ISIM}/${IZ}/lightcone${LIGHTCONE}/job.\${PREV_ARRAY_ID}_\${i}."err
done

echo "=== Step 3 (Job ID \$SLURM_JOB_ID) starting"
echo "    Received arguments: Box='${BOX}', Sim='${ISIM}', Lightcone='${LIGHTCONE}', Sample='${IZ}', nsamp='ntotal'"
echo "=============================="

if [ "${all_found_dndz}" = true ]; then
    echo "Found matched catalogs — skipping unWISE_data_matching.py"
else
    for amp in \$(seq "${AMP_MIN}" "${AMP_STEP}" "${AMP_MAX}"); do
      for slope in \$(seq "${SLOPE_MIN}" "${SLOPE_STEP}" "${SLOPE_MAX}"); do
        amp_fmt=\$(printf "%.1f" "\$amp")
        slope_fmt=\$(printf "%.1f" "\$slope")
        amp_name=\${amp_fmt//./p}
        slope_name=\${slope_fmt//./p}

        python3 unWISE_data_matching.py "${BOX}" "${ISIM}" "${IZ}" "\$amp_fmt" "\$slope_fmt" ntotal "${LIGHTCONE}"
      done
    done
fi

echo "Job 4: Match FLAMINGO halo catalogs to unWISE dN/dz for box ${BOX}, sim ${ISIM}, lightcone ${LIGHTCONE}, ${IZ} sample for all stellar cut parameters."

echo "Job done, info follows."
sacct -j \$SLURM_JOB_ID --format=JobID,JobName,Partition,AveRSS,MaxRSS,AveVMSize,MaxVMSize,Elapsed,ExitCode

EOF
  )

  echo "Submitted step 3: ${jid3}"
fi

# ============================================================
# Step 5: Halo sampling shell array
# ============================================================
jid5="${jid3}"
all_found_sampling=true

for amp in $(seq "$AMP_MIN" "$AMP_STEP" "$AMP_MAX"); do
  for slope in $(seq "$SLOPE_MIN" "$SLOPE_STEP" "$SLOPE_MAX"); do
    amp_fmt=$(printf "%.1f" "$amp")
    slope_fmt=$(printf "%.1f" "$slope")

    amp_name=${amp_fmt//./p}
    slope_name=${slope_fmt//./p}

    file="./data_files/mock_halo_catalogs/${BOX}/${ISIM}/${IZ}/lightcone${LIGHTCONE}/sampled_halo_data_${amp_name}_${slope_name}.parquet"

    if [ ! -f "$file" ]; then
      all_found_sampling=false
      echo "Missing: $file"
      break 2
    fi
  done
done

if [ "$all_found_sampling" = true ]; then
    echo "Found all sampled catalogs — skipping FLAMINGO_halo_sampling_shell.py"
else
    jid5=$(sbatch --parsable \
                  --export=ALL,BOX="${BOX}",ISIM="${ISIM}",IZ="${IZ}",LIGHTCONE="${LIGHTCONE}",AMP_MIN="${AMP_MIN}",AMP_MAX="${AMP_MAX}",AMP_STEP="${AMP_STEP}",SLOPE_MIN="${SLOPE_MIN}",SLOPE_MAX="${SLOPE_MAX}",SLOPE_STEP="${SLOPE_STEP}" \
                  --dependency=afterok:${jid3} \
                  --kill-on-invalid-dep=yes \
                  --job-name=sampling_shells \
                  -c 4 \
                  -p cosma8 \
                  -A dp203 \
                  -t 01:00:00 \
                  --array=0-$((NSHELL-1))%20 \
                  -o "./batch_files/pipeline_logs/${BOX}/${ISIM}/${IZ}/lightcone${LIGHTCONE}/job.%A_%a.dump" \
                  -e "./batch_files/pipeline_logs/${BOX}/${ISIM}/${IZ}/lightcone${LIGHTCONE}/job.%A_%a.err" \
<<EOF
#!/usr/bin/env bash
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ARIJCONL@ljmu.ac.uk
set -euo pipefail

module purge
set +u
[ -f "$HOME/.bashrc" ] && . "$HOME/.bashrc"
conda activate patchy_screening
set -u

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

echo "=== Step 5 (Job ID \$SLURM_JOB_ID) starting"
echo "    Received arguments: ncpu='\$SLURM_CPUS_PER_TASK', Box='${BOX}', Sim='${ISIM}', Lightcone='${LIGHTCONE}', Sample='${IZ}'"
echo "=============================="

python3 FLAMINGO_halo_sampling_shell.py \
  \$SLURM_CPUS_PER_TASK "${BOX}" "${ISIM}" "${IZ}" "\$SLURM_ARRAY_TASK_ID" \
  "${AMP_MIN}" "${AMP_MAX}" "${AMP_STEP}" \
  "${SLOPE_MIN}" "${SLOPE_MAX}" "${SLOPE_STEP}" \
  "${LIGHTCONE}" False

echo "Job 5: Sample halo lightcone shells for box ${BOX}, sim ${ISIM}, lightcone ${LIGHTCONE}, ${IZ} sample for all shells and stellar cut parameters into mock galaxy catalogs."

echo "Job done, info follows."
sacct -j \$SLURM_JOB_ID --format=JobID,JobName,Partition,AveRSS,MaxRSS,AveVMSize,MaxVMSize,Elapsed,ExitCode

EOF
    )

    echo "Submitted step 5: ${jid5}"
fi

# ============================================================
# Step 6: Combine sampled catalogs
# ============================================================
jid6="${jid5}"

if [ "$all_found_sampling" = true ]; then
    echo "Found all sampled catalogs — skipping combine_halo_sampling.py"
else
    jid6=$(sbatch --parsable \
                  --export=ALL,BOX="${BOX}",ISIM="${ISIM}",IZ="${IZ}",LIGHTCONE="${LIGHTCONE}",AMP_MIN="${AMP_MIN}",AMP_MAX="${AMP_MAX}",AMP_STEP="${AMP_STEP}",SLOPE_MIN="${SLOPE_MIN}",SLOPE_MAX="${SLOPE_MAX}",SLOPE_STEP="${SLOPE_STEP}",NSHELL="${NSHELL}",PREV_ARRAY_ID="${jid5}" \
                  --dependency=afterok:${jid5} \
                  --kill-on-invalid-dep=yes \
                  --job-name=combine_sampling \
                  -c 8 \
                  -p cosma8 \
                  -A dp203 \
                  -t 04:00:00 \
                  -o "./batch_files/pipeline_logs/${BOX}/${ISIM}/${IZ}/lightcone${LIGHTCONE}/job.%j.dump" \
                  -e "./batch_files/pipeline_logs/${BOX}/${ISIM}/${IZ}/lightcone${LIGHTCONE}/job.%j.err" \
<<'EOF'
#!/usr/bin/env bash
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ARIJCONL@ljmu.ac.uk
set -euo pipefail

module purge
set +u
[ -f "$HOME/.bashrc" ] && . "$HOME/.bashrc"
conda activate patchy_screening
set -u

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

echo "=== Step 6 (Job ID \$SLURM_JOB_ID) starting"
echo "    Received arguments: Box='$BOX', Sim='$ISIM', Lightcone='$LIGHTCONE', Sample='$IZ'"
echo "=============================="

python3 combine_halo_sampling.py \
  "$BOX" "$ISIM" "$IZ" "$LIGHTCONE" False

echo "Deleting intermediate sampling shell directories..."

for amp in $(seq "$AMP_MIN" "$AMP_STEP" "$AMP_MAX"); do
  for slope in $(seq "$SLOPE_MIN" "$SLOPE_STEP" "$SLOPE_MAX"); do
    amp_fmt=$(printf "%.1f" "$amp")
    slope_fmt=$(printf "%.1f" "$slope")

    amp_name=${amp_fmt//./p}
    slope_name=${slope_fmt//./p}

    dir="./data_files/mock_halo_catalogs/${BOX}/${ISIM}/${IZ}/lightcone${LIGHTCONE}/${amp_name}_${slope_name}"

    if [ -d "$dir" ]; then
        rm -r "$dir"
        echo "Deleted $dir"
    fi
  done
done

echo "Job 6: Combine sampled halo lightcone shells into single mock galaxy catalogs for box ${BOX}, sim ${ISIM}, lightcone ${LIGHTCONE}, ${IZ} sample for all stellar cut parameters."

for ((i=0; i<=NSHELL-2; i++)); do
    rm -f "./batch_files/pipeline_logs/${BOX}/${ISIM}/${IZ}/lightcone${LIGHTCONE}/job.${PREV_ARRAY_ID}_${i}."dump
    rm -f "./batch_files/pipeline_logs/${BOX}/${ISIM}/${IZ}/lightcone${LIGHTCONE}/job.${PREV_ARRAY_ID}_${i}."err
done

echo "Job done, info follows."
sacct -j $SLURM_JOB_ID --format=JobID,JobName,Partition,AveRSS,MaxRSS,AveVMSize,MaxVMSize,Elapsed,ExitCode

EOF
    )

    echo "Submitted step 6: ${jid6}"
fi

# ============================================================
# Step 7: Power spectra (bash loops)
# ============================================================
jid7="${jid6}"
all_found_spectra=true

for amp in $(seq "$AMP_MIN" "$AMP_STEP" "$AMP_MAX"); do
  for slope in $(seq "$SLOPE_MIN" "$SLOPE_STEP" "$SLOPE_MAX"); do
    amp_fmt=$(printf "%.1f" "$amp")
    slope_fmt=$(printf "%.1f" "$slope")

    amp_name=${amp_fmt//./p}
    slope_name=${slope_fmt//./p}

    file1="./data_files/power_spectra/galaxy_galaxy/${BOX}/${ISIM}/${IZ}/lightcone${LIGHTCONE}/galaxy_galaxy_power_spectrum_${amp_name}_${slope_name}.txt"
    file2="./data_files/power_spectra/kappa_galaxy/${BOX}/${ISIM}/${IZ}/lightcone${LIGHTCONE}/kappa_galaxy_power_spectrum_${amp_name}_${slope_name}.txt"

    if [ ! -f "$file1" ] || [ ! -f "$file2" ]; then
      all_found_spectra=false
      echo "Missing: $file1 or $file2"
      break 2
    fi
  done
done

if [ "$all_found_spectra" = true ]; then
    echo "Found all spectra — skipping unWISE_power_spectra.py"
else
    jid7=$(sbatch --parsable \
                  --export=ALL,BOX="${BOX}",ISIM="${ISIM}",IZ="${IZ}",LIGHTCONE="${LIGHTCONE}",AMP_MIN="${AMP_MIN}",AMP_MAX="${AMP_MAX}",AMP_STEP="${AMP_STEP}",SLOPE_MIN="${SLOPE_MIN}",SLOPE_MAX="${SLOPE_MAX}",SLOPE_STEP="${SLOPE_STEP}",NAMP="${NAMP}",NSLOPE="${NSLOPE}" \
                  --dependency=afterok:${jid6} \
                  --kill-on-invalid-dep=yes \
                  --job-name=power_spectra \
                  -c 8 \
                  -p cosma8 \
                  -A dp203 \
                  -t 02:00:00 \
                  --array=0-$((NGRID-1))%20 \
                  -o "./batch_files/pipeline_logs/${BOX}/${ISIM}/${IZ}/lightcone${LIGHTCONE}/job.%A_%a.dump" \
                  -e "./batch_files/pipeline_logs/${BOX}/${ISIM}/${IZ}/lightcone${LIGHTCONE}/job.%A_%a.err" \
<<'EOF'
#!/usr/bin/env bash
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ARIJCONL@ljmu.ac.uk
set -euo pipefail

module purge
set +u
[ -f "$HOME/.bashrc" ] && . "$HOME/.bashrc"
conda activate patchy_screening
set -u

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# grid sizes
A_SIZE=${NAMP}    # number of amp values: 10.3..11.3 step 0.1
S_SIZE=${NSLOPE}  # number of slope values: 0.0..1.0 step 0.1 

# decode array index -> (amp, slope)
A_IDX=$(( SLURM_ARRAY_TASK_ID / S_SIZE ))
S_IDX=$(( SLURM_ARRAY_TASK_ID % S_SIZE ))

amp=$(awk -v i="$A_IDX" -v amin="${AMP_MIN}" -v astep="${AMP_STEP}" 'BEGIN{printf "%.1f", amin + astep*i}')
slope=$(awk -v i="$S_IDX" -v smin="${SLOPE_MIN}" -v sstep="${SLOPE_STEP}" 'BEGIN{printf "%.1f", smin + sstep*i}')

amp_name=${amp//./p}
slope_name=${slope//./p}

echo "=== Step 7 (Job ID $SLURM_JOB_ID) starting"
echo "    Received arguments: Box='${BOX}', Sim='${ISIM}', Lightcone='${LIGHTCONE}', Sample='${IZ}', nsamp='ntotal'"
echo "=============================="

if [ -f "./data_files/power_spectra/galaxy_galaxy/${BOX}/${ISIM}/${IZ}/lightcone${LIGHTCONE}/galaxy_galaxy_power_spectrum_${amp_name}_${slope_name}.txt" ] && \
   [ -f "./data_files/power_spectra/kappa_galaxy/${BOX}/${ISIM}/${IZ}/lightcone${LIGHTCONE}/kappa_galaxy_power_spectrum_${amp_name}_${slope_name}.txt" ]; then
    echo "Found existing galaxy-galaxy and kappa-galaxy spectra file — skipping unWISE_power_spectra.py"
else 
    python3 unWISE_power_spectra.py \
      "$SLURM_CPUS_PER_TASK" "${BOX}" "${ISIM}" "${IZ}" "$amp" "$slope" \
      unlensed True False False True True False "${LIGHTCONE}"
fi

echo "Job 7: Compute power spectra for box ${BOX}, sim ${ISIM}, lightcone ${LIGHTCONE}, ${IZ} sample for all stellar cut parameters."

echo "Job done, info follows."
sacct -j $SLURM_JOB_ID --format=JobID,JobName,Partition,AveRSS,MaxRSS,AveVMSize,MaxVMSize,Elapsed,ExitCode

EOF
    )

    echo "Submitted step 7: ${jid7}"
fi

# ============================================================
# Step 8: mock likelihood
# ============================================================
jid8=$(sbatch --parsable \
            --export=ALL,BOX="${BOX}",ISIM="${ISIM}",IZ="${IZ}",LIGHTCONE="${LIGHTCONE}",AMP_MIN="${AMP_MIN}",AMP_MAX="${AMP_MAX}",AMP_STEP="${AMP_STEP}",SLOPE_MIN="${SLOPE_MIN}",SLOPE_MAX="${SLOPE_MAX}",SLOPE_STEP="${SLOPE_STEP}",ABUNDANCE_CUT="${ABUNDANCE_CUT}",NGRID="${NGRID}",PREV_ARRAY_ID="${jid7}" \
            --dependency=afterok:${jid7} \
            --kill-on-invalid-dep=yes \
            --job-name=mock_catalog_maximum_likelihood_estimation \
            -c 16 \
            -p cosma8 \
            -A dp203 \
            -t 01:00:00 \
            -o "./batch_files/maximum_likelihood_logs/${BOX}/${ISIM}/${IZ}/lightcone${LIGHTCONE}/job.%j.dump" \
            -e "./batch_files/maximum_likelihood_logs/${BOX}/${ISIM}/${IZ}/lightcone${LIGHTCONE}/job.%j.err" \
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
[ -f "$HOME/.bashrc" ] && . "$HOME/.bashrc"
conda activate patchy_screening
set -u

echo "=== Step 8 (Job ID \$SLURM_JOB_ID) starting"
echo "    Received arguments: ncpu='\$SLURM_CPUS_PER_TASK', Box='${BOX}', Sim='${ISIM}', Lightcone='${LIGHTCONE}', Sample='${IZ}'"
echo "=============================="

python3 mock_catalog_likelihood_parallel.py \
    "\$SLURM_CPUS_PER_TASK" "${BOX}" "${ISIM}" "${IZ}" "${LIGHTCONE}" "${ABUNDANCE_CUT}" \
    "${AMP_MIN}" "${AMP_MAX}" "${AMP_STEP}" \
    "${SLOPE_MIN}" "${SLOPE_MAX}" "${SLOPE_STEP}"

echo "Job 8: Compute maximum likelihood estimates for box ${BOX}, sim ${ISIM}, lightcone ${LIGHTCONE}, ${IZ} sample."

for ((i=0; i<=NGRID-2; i++)); do
    rm -f "./batch_files/pipeline_logs/${BOX}/${ISIM}/${IZ}/lightcone${LIGHTCONE}/job.\${PREV_ARRAY_ID}_\${i}."dump
    rm -f "./batch_files/pipeline_logs/${BOX}/${ISIM}/${IZ}/lightcone${LIGHTCONE}/job.\${PREV_ARRAY_ID}_\${i}."err
done

echo "Job done, info follows."
sacct -j \$SLURM_JOB_ID --format=JobID,JobName,Partition,AveRSS,MaxRSS,AveVMSize,MaxVMSize,Elapsed,ExitCode

EOF
)

# Path to MLE output
MLE_FILE="./data_files/mle_parameters/${BOX}/${ISIM}/${IZ}/lightcone${LIGHTCONE}/mle_values.txt"

# ============================================================
# Step 9: stellar_cut_z for MLE values only
# ============================================================
jid9=$(sbatch --parsable \
                --export=ALL,BOX="${BOX}",ISIM="${ISIM}",IZ="${IZ}",LIGHTCONE="${LIGHTCONE}",MLE_FILE="${MLE_FILE}" \
                --dependency=afterok:${jid8} \
                --kill-on-invalid-dep=yes \
                --job-name=mle_stellar_cut \
                -c 1 \
                -p cosma8 \
                -A dp203 \
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
conda activate patchy_screening
set -u

mle_amp=$(grep -m1 '^AMP' "$MLE_FILE" | cut -d'=' -f2 | xargs)
mle_slope=$(grep -m1 '^SLOPE' "$MLE_FILE" | cut -d'=' -f2 | xargs)

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
# Step 10: Lightcones for MLE values only
# ============================================================
jid10=$(sbatch --parsable \
                --export=ALL,BOX="${BOX}",ISIM="${ISIM}",IZ="${IZ}",LIGHTCONE="${LIGHTCONE}",MLE_FILE="${MLE_FILE}" \
                --dependency=afterok:${jid9} --kill-on-invalid-dep=yes \
                --job-name=mle_lightcones_shells \
                -c 4 \
                -p cosma8 \
                -A dp203 \
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
conda activate patchy_screening
set -u

mle_amp=$(grep -m1 '^AMP' "$MLE_FILE" | cut -d'=' -f2 | xargs)
mle_slope=$(grep -m1 '^SLOPE' "$MLE_FILE" | cut -d'=' -f2 | xargs)

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
      "$LIGHTCONE" True
fi

echo "Job 10: Generate lightcone shells for box $BOX, sim $ISIM, lightcone $LIGHTCONE, $IZ sample for MLE values only for all shells."

echo "Job done, info follows."
sacct -j $SLURM_JOB_ID --format=JobID,JobName,Partition,AveRSS,MaxRSS,AveVMSize,MaxVMSize,Elapsed,ExitCode

EOF
)

# ============================================================
# Step 11: Combine lightcones for MLE values only
# ============================================================
jid11=$(sbatch --parsable \
                --export=ALL,BOX="${BOX}",ISIM="${ISIM}",IZ="${IZ}",LIGHTCONE="${LIGHTCONE}",MLE_FILE="${MLE_FILE}",NSHELL="${NSHELL}",PREV_ARRAY_ID="${jid10}" \
                --dependency=afterok:${jid10} \
                --kill-on-invalid-dep=yes \
                --job-name=mle_combine_lightcones_unWISE_matching \
                -c 4 \
                -p cosma8 \
                -A dp203 \
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
conda activate patchy_screening
set -u

mle_amp=$(grep -m1 '^AMP' "$MLE_FILE" | cut -d'=' -f2 | xargs)
mle_slope=$(grep -m1 '^SLOPE' "$MLE_FILE" | cut -d'=' -f2 | xargs)

echo "=== Step 11 (Job ID $SLURM_JOB_ID) starting"
echo "    Received arguments: Box='$BOX', Sim='$ISIM', Lightcone='$LIGHTCONE', Sample='$IZ'"
echo "=============================="

if [ -f "./data_files/halo_totals/${BOX}/${ISIM}/${IZ}/lightcone${LIGHTCONE}/FLAMINGO_halo_totals_${mle_amp//./p}_${mle_slope//./p}.txt" ]; then
    echo "Stellar cut totals found — skipping combine_halo_lightcones.py"
else
    python3 combine_halo_lightcones.py \
      "$BOX" "$ISIM" "$IZ" "$LIGHTCONE" True

    echo "Deleting MLE shell_*.txt files..."
    rm ./data_files/halo_totals/${BOX}/${ISIM}/${IZ}/lightcone${LIGHTCONE}/shell_*_mle.txt 2>/dev/null || true
fi

echo "Job 11: Combine halo lightcone shells for box $BOX, sim $ISIM, lightcone $LIGHTCONE, $IZ sample for MLE values only into single halo totals files."

for ((i=0; i<=NSHELL-2; i++)); do
    rm -f "./batch_files/pipeline_logs/${BOX}/${ISIM}/${IZ}/lightcone${LIGHTCONE}/job.mle_${PREV_ARRAY_ID}_${i}."dump
    rm -f "./batch_files/pipeline_logs/${BOX}/${ISIM}/${IZ}/lightcone${LIGHTCONE}/job.mle_${PREV_ARRAY_ID}_${i}."err
done

echo "=== Step 12 (Job ID $SLURM_JOB_ID) starting"
echo "    Received arguments: Box='$BOX', Sim='$ISIM', Lightcone='$LIGHTCONE', Sample='$IZ', nsamp='ntotal'"
echo "=============================="

if [ -f "./data_files/dndz_samples/${BOX}/${ISIM}/${IZ}/lightcone${LIGHTCONE}/dndz_galaxies_sampled_${mle_amp//./p}_${mle_slope//./p}.txt" ]; then
    echo "Found matched catalogs — skipping unWISE_data_matching.py"
else
    python3 unWISE_data_matching.py "$BOX" "$ISIM" "$IZ" "$mle_amp" "$mle_slope" ntotal "$LIGHTCONE"
fi

echo "Job 12: Match FLAMINGO halo catalogs to unWISE dN/dz for box $BOX, sim $ISIM, lightcone $LIGHTCONE, $IZ sample for MLE values only."

echo "Job done, info follows."
sacct -j $SLURM_JOB_ID --format=JobID,JobName,Partition,AveRSS,MaxRSS,AveVMSize,MaxVMSize,Elapsed,ExitCode

EOF
)

# ============================================================
# Step 13: sampling for MLE values only
# ============================================================
jid13=$(sbatch --parsable \
                --export=ALL,BOX="${BOX}",ISIM="${ISIM}",IZ="${IZ}",LIGHTCONE="${LIGHTCONE}",MLE_FILE="${MLE_FILE}",NSHELL="${NSHELL}" \
                --dependency=afterok:${jid11} \
                --kill-on-invalid-dep=yes \
                --job-name=mle_sampling_shells \
                -c 4 \
                -p cosma8 \
                -A dp203 \
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
conda activate patchy_screening
set -u

mle_amp=$(grep -m1 '^AMP' "$MLE_FILE" | cut -d'=' -f2 | xargs)
mle_slope=$(grep -m1 '^SLOPE' "$MLE_FILE" | cut -d'=' -f2 | xargs)

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
    echo "Found sampled catalogs for MLE values — skipping FLAMINGO_halo_sampling_shell.py"
else
    python3 FLAMINGO_halo_sampling_shell.py \
      "$SLURM_CPUS_PER_TASK" "$BOX" "$ISIM" "$IZ" "$SLURM_ARRAY_TASK_ID" \
      "$mle_amp" "$mle_amp" "0.1" \
      "$mle_slope" "$mle_slope" "0.1" \
      "$LIGHTCONE" True
fi

echo "Job 13: Sample halo lightcone shells for box $BOX, sim $ISIM, lightcone $LIGHTCONE, $IZ sample for MLE values only for all shells into mock galaxy catalogs."

echo "Job done, info follows."
sacct -j $SLURM_JOB_ID --format=JobID,JobName,Partition,AveRSS,MaxRSS,AveVMSize,MaxVMSize,Elapsed,ExitCode

EOF
)

# ============================================================
# Step 14: combine sampling for MLE values only
# ============================================================
jid14=$(sbatch --parsable \
                --export=ALL,BOX="${BOX}",ISIM="${ISIM}",IZ="${IZ}",LIGHTCONE="${LIGHTCONE}",MLE_FILE="${MLE_FILE}",NSHELL="${NSHELL}",PREV_ARRAY_ID="${jid13}" \
                --dependency=afterok:${jid13} \
                --kill-on-invalid-dep=yes \
                --job-name=mle_combine_sampling_power_spectra \
                -c 16 \
                -p cosma8 \
                -A dp203 \
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
conda activate patchy_screening
set -u

mle_amp=$(grep -m1 '^AMP' "$MLE_FILE" | cut -d'=' -f2 | xargs)
mle_slope=$(grep -m1 '^SLOPE' "$MLE_FILE" | cut -d'=' -f2 | xargs)

echo "=== Step 14 (Job ID $SLURM_JOB_ID) starting"
echo "    Received arguments: Box='$BOX', Sim='$ISIM', Lightcone='$LIGHTCONE', Sample='$IZ'"
echo "=============================="

if [ -f "./data_files/mock_halo_catalogs/${BOX}/${ISIM}/${IZ}/lightcone${LIGHTCONE}/sampled_halo_data_${mle_amp//./p}_${mle_slope//./p}.parquet" ]; then
    echo "Found sampled catalogs for MLE values — skipping combine_halo_sampling.py"
else
    python3 combine_halo_sampling.py \
      "$BOX" "$ISIM" "$IZ" "$LIGHTCONE" True
fi

dir="./data_files/mock_halo_catalogs/${BOX}/${ISIM}/${IZ}/lightcone${LIGHTCONE}/mle"

if [ -d "$dir" ]; then
    rm -r "$dir"
    echo "Deleted $dir"
fi

echo "Job 14: Combine sampled halo lightcone shells into single mock galaxy catalogs for box $BOX, sim $ISIM, lightcone $LIGHTCONE, $IZ sample for MLE values only."

for ((i=0; i<=NSHELL-2; i++)); do
    rm -f "./batch_files/pipeline_logs/${BOX}/${ISIM}/${IZ}/lightcone${LIGHTCONE}/job.mle_${PREV_ARRAY_ID}_${i}."dump
    rm -f "./batch_files/pipeline_logs/${BOX}/${ISIM}/${IZ}/lightcone${LIGHTCONE}/job.mle_${PREV_ARRAY_ID}_${i}."err
done

echo "=== Step 15 (Job ID $SLURM_JOB_ID) starting"
echo "    Received arguments: Box='$BOX', Sim='$ISIM', Lightcone='$LIGHTCONE', Sample='$IZ', nsamp='ntotal'"
echo "=============================="

if [ -f "./data_files/power_spectra/galaxy_galaxy/${BOX}/${ISIM}/${IZ}/lightcone${LIGHTCONE}/galaxy_galaxy_power_spectrum_${mle_amp//./p}_${mle_slope//./p}.txt" ] && \
   [ -f "./data_files/power_spectra/kappa_galaxy/${BOX}/${ISIM}/${IZ}/lightcone${LIGHTCONE}/kappa_galaxy_power_spectrum_${mle_amp//./p}_${mle_slope//./p}.txt" ]; then
    echo "Found existing galaxy-galaxy and kappa-galaxy spectra file for MLE values — skipping unWISE_power_spectra.py"
else
    python3 unWISE_power_spectra.py \
      "$SLURM_CPUS_PER_TASK" "$BOX" "$ISIM" "$IZ" "$mle_amp" "$mle_slope" \
      unlensed True False False True True False "$LIGHTCONE"
fi

echo "Job 15: Compute power spectra for box $BOX, sim $ISIM, lightcone $LIGHTCONE, $IZ sample for MLE values only."

echo "Job done, info follows."
sacct -j $SLURM_JOB_ID --format=JobID,JobName,Partition,AveRSS,MaxRSS,AveVMSize,MaxVMSize,Elapsed,ExitCode

EOF
)

echo "Chain:"
# echo "${jid0} -> ${jid1} -> ${jid2} -> ${jid3} -> ${jid4} -> ${jid5} -> ${jid6} -> ${jid7} -> ${jid8} -> ${jid9} -> ${jid10} -> ${jid11} -> ${jid12} -> ${jid13} -> ${jid14} -> ${jid15}"
echo "${jid0} -> ${jid2} -> ${jid3} -> ${jid5} -> ${jid6} -> ${jid7} -> ${jid8} -> ${jid9} -> ${jid10} -> ${jid11} -> ${jid13} -> ${jid14}"
