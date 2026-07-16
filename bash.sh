#!/bin/bash -l

#SBATCH -c 128
#SBATCH -J FLAMINGO_halo_z
#SBATCH -o ./batch_files/misc_logs/job.%J.dump
#SBATCH -e ./batch_files/misc_logs/job.%J.err
#SBATCH -p cosma8
#SBATCH -A dp004
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ARIJCONL@ljmu.ac.uk 
#SBATCH -t 12:00:00

# Queue the job to be restarted
#output=$(sbatch --dependency=afternotok:$SLURM_JOBID $0)
#replacement_id=$(echo $output | awk '{print $4}')

# In your SLURM script or before python
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

PYTHON_SCRIPT="$1"
shift  # Shift removes the first argument ($1), so $@ now contains only the script args

# Generate a unique filename for this job
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
if [ $PYTHON_SCRIPT = "mock_catalog_likelihood.py" ]; then
    COPIED_SCRIPT="./batch_files/mock_catalog_likelihood_${TIMESTAMP}.py"
    # Convert the relative path to an absolute path
    ABS_SCRIPT_PATH=$(realpath "$COPIED_SCRIPT")
    # Copy the current Python script so the submitted job always uses this version
    cp "$PYTHON_SCRIPT" "$ABS_SCRIPT_PATH"
    echo "$ABS_SCRIPT_PATH"
fi

module purge

# Ensure the job uses the correct Python script copy
#echo "Running job with script: $PYTHON_SCRIPT"

# Activate conda environment
source ~/.bashrc 
conda activate patchy_screening
# Run the specified Python script with the remaining arguments
if [ $PYTHON_SCRIPT = "mock_catalog_likelihood.py" ]; then
    python3 "$ABS_SCRIPT_PATH" "$SLURM_CPUS_PER_TASK" "$@"
elif [ $PYTHON_SCRIPT = "stacked_DM_maps.py" ]; then
    python3 "$PYTHON_SCRIPT" 12 "$@"
else
    python3 "$PYTHON_SCRIPT" "$SLURM_CPUS_PER_TASK" "$@"
fi

# ---- CLEANUP AFTER JOB COMPLETION ----
echo "Job completed. Cleaning up temporary files..."

if [ $PYTHON_SCRIPT = "mock_catalog_likelihood.py" ]; then
    # Remove the copied Python script
    rm -f $ABS_SCRIPT_PATH
    # Print completion message
    echo "Cleanup complete. Only SLURM output logs and original Python script remain."
fi

echo "Job done, info follows."
sacct -j $SLURM_JOBID --format=JobID,JobName,Partition,AveRSS,MaxRSS,AveVMSize,MaxVMSize,Elapsed,ExitCode
