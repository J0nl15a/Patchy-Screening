#!/bin/bash -l

#SBATCH -J patchy_screening
#SBATCH -o ./batch_files/patchy_screening_logs/job.cmb_%J.dump
#SBATCH -e ./batch_files/patchy_screening_logs/job.cmb_%J.err
#SBATCH -p cosma8
#SBATCH -A dp004
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ARIJCONL@ljmu.ac.uk 
#SBATCH -t 01:00:00

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --threads-per-core=1

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK
export POLARS_MAX_THREADS=$SLURM_CPUS_PER_TASK

module purge
module load gnu_comp/14.1.0
module load openmpi/5.0.3

# Ensure the job uses the correct Python script copy
echo "Running job with script: $PYTHON_SCRIPT"

# RUN_ID="$(date +%Y%m%d_%H%M%S)_job${SLURM_JOB_ID}"

# Load the conda environment using mamba or source if available
source ~/.bashrc
mamba activate patchy_screening
module list

# ─── Run your Python script, passing CHECKPOINT_DIR in the environment ───
# python3 $PYTHON_SCRIPT $SLURM_CPUS_PER_TASK "$@" cmb "$RUN_ID" false
python3 $PYTHON_SCRIPT $SLURM_CPUS_PER_TASK "$@" cmb "$RUN_ID" false

echo "Job done, info follows."
sacct -j $SLURM_JOBID --format=JobID,JobName,Partition,AveRSS,MaxRSS,AveVMSize,MaxVMSize,Elapsed,ExitCode
