#!/bin/bash

# Define paths
PYTHON_FILE="imp_patchy_screening.py"

# Generate a unique filename for this job
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RUN_ID="run_${TIMESTAMP}"
COPIED_SCRIPT="./batch_files/patchy_screening_logs/patchy_screening_${TIMESTAMP}.py"

# Convert the relative path to an absolute path
ABS_SCRIPT_PATH=$(realpath "$COPIED_SCRIPT")

# Copy the current Python script so the submitted job always uses this version
cp "$PYTHON_FILE" "$ABS_SCRIPT_PATH"

# Submit the SLURM job, passing the copied script name
HALO_JOB_ID=$(sbatch -A dp004 \
    --export=PYTHON_SCRIPT=$ABS_SCRIPT_PATH,RUN_ID=$RUN_ID \
    submit_halo_script.sh "$@" | awk '{print $4}')

CMB_JOB_ID=$(sbatch -A dp004 \
    --dependency=afterok:${HALO_JOB_ID} \
    --kill-on-invalid-dep=yes \
    --export=PYTHON_SCRIPT=$ABS_SCRIPT_PATH,RUN_ID=$RUN_ID \
    submit_cmb_script.sh "$@" | awk '{print $4}')

TAU_JOB_ID=$(sbatch -A dp004 \
    --dependency=afterok:${CMB_JOB_ID} \
    --kill-on-invalid-dep=yes \
    --export=PYTHON_SCRIPT=$ABS_SCRIPT_PATH,RUN_ID=$RUN_ID \
    submit_tau_mpi_script.sh "$@" | awk '{print $4}')
    # --dependency=afterok:${CMB_JOB_ID}:${HALO_JOB_ID} \

# CLEAN_UP_JOB_ID=$(sbatch -A dp004 \
#     --dependency=afterok:${TAU_JOB_ID} \
#     --export=PYTHON_SCRIPT=$ABS_SCRIPT_PATH \
#     submit_cleanup_script.sh "$@" | awk '{print $4}')

echo "Submitted CMB job:  $CMB_JOB_ID"
echo "Submitted halo job: $HALO_JOB_ID"
echo "Submitted MPI tau job: $TAU_JOB_ID"
echo "Using Python script: $ABS_SCRIPT_PATH"
