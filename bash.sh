#!/bin/bash -l

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH -J FLAMINGO_halo_z
#SBATCH -o ./job.%J.dump
#SBATCH -e ./job.%J.err
#SBATCH -p cosma8
#SBATCH -A dp004
#SBATCH --exclusive
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ARIJCONL@ljmu.ac.uk 
#SBATCH -t 6:00:00

# Queue the job to be restarted
#output=$(sbatch --dependency=afternotok:$SLURM_JOBID $0)
#replacement_id=$(echo $output | awk '{print $4}')

module purge

# Ensure the job uses the correct Python script copy
#echo "Running job with script: $PYTHON_SCRIPT"

# Activate conda environment
conda activate patchy_screening
# Run the copied Python script
python3 FLAMINGO_halo_sampling.py 128 "$@"

# ---- CLEANUP AFTER JOB COMPLETION ----
echo "Job completed. Cleaning up temporary files..."

# Remove the copied Python script
#rm -f $PYTHON_SCRIPT
# Print completion message
#echo "Cleanup complete. Only SLURM output logs and original Python script remain."

# Program exited, so no need to restart
#scancel $replacement_id

#echo "Job done, info follows."
#sacct -j $SLURM_JOBID --format=JobID,JobName,Partition,AveRSS,MaxRSS,AveVMSize,MaxVMSize,Elapsed,ExitCode
