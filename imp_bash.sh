#!/bin/bash

# Check that exactly two arguments were given
if [ "$#" -ne 5 ]; then
  echo "Usage: $0 <param1> <param2> <param3> <param4> <param5>"
  exit 1
fi

param1="$1"
param2="$2"
param3="$3"
param4="$4"
param5="$5"

echo ">>> Launching pipeline with Sim='${param1}' Sample='${param2}' M_cut(z_mean)='${param3}' n_cut='${param4}' nsamp='${param5}'"

# 1) Launch step 1 and save its SLURM job ID
jid1=$(sbatch --parsable << EOF
#!/bin/bash
#SBATCH -c 1
#SBATCH -p cosma8
#SBATCH -A dp203
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ARIJCONL@ljmu.ac.uk
#SBATCH --job-name=z_stellar_cut
#SBATCH --output=./batch_files/job.z_stellar_cut.%j.dump
#SBATCH --error=./batch_files/job.z_stellar_cut.%j.err
#SBATCH --time=00:01:00

module purge
mamba activate patchy_screening

echo "=== Step 1 (Job ID \$SLURM_JOB_ID) starting"
echo "    Received arguments: Sample='${param2}', M_cut(z_mean)='${param3}', n_cut='${param4}'"
echo "=============================="

python3 stellar_cut_z.py "$param2" "$param3" "$param4"

exit_code=\$?
echo "Step 1 finished with exit code \$exit_code"
exit \$exit_code

echo "Job done, info follows."
sacct -j $SLURM_JOBID --format=JobID,JobName,Partition,AveRSS,MaxRSS,AveVMSize,MaxVMSize,Elapsed,ExitCode

EOF
)

echo "Job 1: Compute z-dependant stellar cut values for ${param2} sample with: M_cut(z_mean) = ${param3}, n_cut = ${param4}"

# 1) Launch step 1 and save its SLURM job ID
jid2=$(sbatch --parsable \
              --dependency=afterok:${jid1} \
              --kill-on-invalid-dep=yes \
              <<EOF
#!/bin/bash
#SBATCH -c 128
#SBATCH -p cosma8
#SBATCH -A dp203
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ARIJCONL@ljmu.ac.uk
#SBATCH --job-name=halo_lightcones
#SBATCH --output=./batch_files/job.FLAMINGO_halo_lightcones.%j.dump
#SBATCH --error=./batch_files/job.FLAMINGO_halo_lightcones.%j.err
#SBATCH --time=00:30:00

module purge
mamba activate patchy_screening

echo "=== Step 2 (Job ID \$SLURM_JOB_ID) starting"
echo "    Received arguments: ncpu='\$SLURM_CPUS_PER_TASK', Sim='${param1}', Sample='${param2}', M_cut(z_mean)='${param3}', n_cut='${param4}'"
echo "=============================="

python3 FLAMINGO_halo_lightcones.py "\$SLURM_CPUS_PER_TASK" "$param1" "$param2" "$param3" "$param4"

# Capture exit code to be explicit (though SLURM will honor it anyway)
exit_code=\$?
echo "Step 2 finished with exit code \$exit_code"
exit \$exit_code

echo "Job done, info follows."
sacct -j $SLURM_JOBID --format=JobID,JobName,Partition,AveRSS,MaxRSS,AveVMSize,MaxVMSize,Elapsed,ExitCode

EOF
)

echo "Job 2: Collate halos in lightcone shells for sim ${param1}, ${param2} sample with: M_cut(z_mean) = ${param3}, n_cut = ${param4}."

# 3) Launch step 3, but tell SLURM “do not start until job‐ID=$jid2 finishes OK”
jid3=$(sbatch --parsable \
	      --dependency=afterok:${jid2} \
	      --kill-on-invalid-dep=yes \
	      <<EOF
#!/bin/bash
#SBATCH -c 1
#SBATCH -p cosma8
#SBATCH -A dp203
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ARIJCONL@ljmu.ac.uk
#SBATCH --job-name=unWISE_data_matching
#SBATCH --output=./batch_files/job.unWISE_data_matching.%j.dump
#SBATCH --error=./batch_files/job.unWISE_data_matching.%j.err
#SBATCH --time=00:01:00

module purge
mamba activate patchy_screening

echo "=== Step 3 (Job ID \$SLURM_JOB_ID) starting"
echo "    Received arguments: Sim='${param1}', Sample='${param2}', M_cut(z_mean)='${param3}', n_cut='${param4}', nsamp='${param5}'"
echo "=============================="

python3 unWISE_data_matching.py "$param1" "$param2" "$param3" "$param4" "$param5"

exit_code=\$?
echo "Step 3 finished with exit code \$exit_code"
exit \$exit_code

echo "Job done, info follows."
sacct -j $SLURM_JOBID --format=JobID,JobName,Partition,AveRSS,MaxRSS,AveVMSize,MaxVMSize,Elapsed,ExitCode

EOF
)

echo "Job 3: Rescale unWISE dndz curve for sim ${param1}, ${param2} sample and stellar cut with: M_cut(z_mean) = ${param3}, n_cut = ${param4}."

# 4) Launch step 4 after job‐ID=$jid3 succeeds
jid4=$(sbatch --parsable \
	      --dependency=afterok:${jid3} \
	      --kill-on-invalid-dep=yes \
	      <<EOF
#!/bin/bash
#SBATCH -c 128
#SBATCH -p cosma8
#SBATCH -A dp203
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ARIJCONL@ljmu.ac.uk
#SBATCH --job-name=halo_sampling
#SBATCH --output=./batch_files/job.FLAMINGO_halo_sampling.%j.dump
#SBATCH --error=./batch_files/job.FLAMINGO_halo_sampling.%j.err
#SBATCH --time=00:30:00

module purge
mamba activate patchy_screening

echo "=== Step 4 (Job ID \$SLURM_JOB_ID) starting"
echo "    Received arguments: ncpu="\$SLURM_CPUS_PER_TASK", Sim='${param1}', Sample='${param2}', M_cut(z_mean)='${param3}', n_cut='${param4}'"
echo "=============================="

python3 FLAMINGO_halo_sampling.py "\$SLURM_CPUS_PER_TASK" "$param1" "$param2" "$param3" "$param4"

exit_code=\$?
echo "Step 4 finished with exit code \$exit_code"
exit \$exit_code

echo "Job done, info follows."
sacct -j $SLURM_JOBID --format=JobID,JobName,Partition,AveRSS,MaxRSS,AveVMSize,MaxVMSize,Elapsed,ExitCode

EOF
)
       
echo "Job 4: Sampling FLAMINGO halo lightcones using rescaled unWISE dndz curve for sim ${param1}, ${param2} sample and stellar cut with: M_cut(z_mean) = ${param3}, n_cut = ${param4}."
