#!/bin/bash

# Check that exactly two arguments were given
if [ "$#" -ne 4 ]; then
  echo "Usage: $0 <param1> <param2> <param3> <param4>"
  exit 1
fi

param1="$1"
param2="$2"
param3="$3"
param4="$4"

echo ">>> Launching pipeline with param1='${param1}'  param2='${param2}' param3='${param3}' param4='${param4}'"

# 1) Launch step 1 and save its SLURM job ID
jid1=$(sbatch --parsable << EOF
#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH -p cosma8
#SBATCH -A dp004
#SBATCH --exclusive
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ARIJCONL@ljmu.ac.uk
#SBATCH --job-name=FLAMINGO_halo_lightcones
#SBATCH --output=./batch_files/job.FLAMINGO_halo_lightcones.%j.dump
#SBATCH --error=./batch_files/job.FLAMINGO_halo_lightcones.%j.err
#SBATCH --time=02:00:00

module purge
mamba activate patchy_screening

echo "=== Step 1 (Job ID \$SLURM_JOB_ID) starting"
echo "    Received arguments: param1='${param1}', param3='${param3}'"
echo "=============================="

python3 FLAMINGO_halo_lightcones.py "$param1" "$param3"

# Capture exit code to be explicit (though SLURM will honor it anyway)
exit_code=\$?
echo "Step 1 finished with exit code \$exit_code"
exit \$exit_code

echo "Job done, info follows."
sacct -j $SLURM_JOBID --format=JobID,JobName,Partition,AveRSS,MaxRSS,AveVMSize,MaxVMSize,Elapsed,ExitCode

EOF
)

echo "Job 1: Collate halos in lightcone shells for sim ${param1}, with stellar cut of ${param3}."

# 2) Launch step 2, but tell SLURM “do not start until job‐ID=$jid1 finishes OK”
jid2=$(sbatch --parsable \
	      --dependency=afterok:${jid1} \
	      --kill-on-invalid-dep=yes \
	      <<EOF
#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH -p cosma8
#SBATCH -A dp004
#SBATCH --exclusive
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ARIJCONL@ljmu.ac.uk
#SBATCH --job-name=unWISE_data_matching
#SBATCH --output=./batch_files/job.unWISE_data_matching.%j.dump
#SBATCH --error=./batch_files/job.unWISE_data_matching.%j.err
#SBATCH --time=00:01:00

module purge
mamba activate patchy_screening

echo "=== Step 2 (Job ID \$SLURM_JOB_ID) starting"
echo "    Received arguments: param1='${param1}', param2='${param2}', param3='${param3}', param4='${param4}'"
echo "=============================="

python3 unWISE_data_matching.py "$param1" "$param2" "$param3" "$param4"

exit_code=\$?
echo "Step 2 finished with exit code \$exit_code"
exit \$exit_code

echo "Job done, info follows."
sacct -j $SLURM_JOBID --format=JobID,JobName,Partition,AveRSS,MaxRSS,AveVMSize,MaxVMSize,Elapsed,ExitCode

EOF
)

echo "Job 2: Rescale unWISE dndz curve for sim ${param1}, ${param2} sample and stellar cut of ${param3}."

# 3) Launch step 3 after job‐ID=$jid2 succeeds
jid3=$(sbatch --parsable \
	      --dependency=afterok:${jid2} \
	      --kill-on-invalid-dep=yes \
	      <<EOF
#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH -p cosma8
#SBATCH -A dp004
#SBATCH --exclusive
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ARIJCONL@ljmu.ac.uk
#SBATCH --job-name=FLAMINGO_halo_sampling
#SBATCH --output=./batch_files/job.FLAMINGO_halo_sampling.%j.dump
#SBATCH --error=./batch_files/job.FLAMINGO_halo_sampling.%j.err
#SBATCH --time=01:00:00

module purge
mamba activate patchy_screening

echo "=== Step 3 (Job ID \$SLURM_JOB_ID) starting"
echo "    Received arguments: 128, param1='${param1}', param2='${param2}', param3='${param3}'"
echo "=============================="

python3 FLAMINGO_halo_sampling.py 128 "$param1" "$param2" "$param3"

exit_code=\$?
echo "Step 3 finished with exit code \$exit_code"
exit \$exit_code

echo "Job done, info follows."
sacct -j $SLURM_JOBID --format=JobID,JobName,Partition,AveRSS,MaxRSS,AveVMSize,MaxVMSize,Elapsed,ExitCode

EOF
)
       
echo "Job 3: Sampling FLAMINGO halo lightcones using rescaled unWISE dndz curve for sim ${param1}, ${param2} sample and stellar cut of ${param3}."
