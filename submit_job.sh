#!/bin/bash

# Define paths
PYTHON_FILE="imp_patchy_screening.py"
JOB_SCRIPT="submit_script.sh"

# Generate a unique filename for this job
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
COPIED_SCRIPT="./batch_files/patchy_screening_${TIMESTAMP}.py"

# Convert the relative path to an absolute path
ABS_SCRIPT_PATH=$(realpath "$COPIED_SCRIPT")

# Copy the current Python script so the submitted job always uses this version
cp "$PYTHON_FILE" "$ABS_SCRIPT_PATH"

# Submit the SLURM job, passing the copied script name
JOB_ID=$(sbatch -A dp004 --export=PYTHON_SCRIPT=$ABS_SCRIPT_PATH $JOB_SCRIPT "$@" | awk '{print $4}')

echo "Submitted job $JOB_ID using $ABS_SCRIPT_PATH"
