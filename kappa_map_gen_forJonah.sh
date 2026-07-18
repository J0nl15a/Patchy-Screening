#!/bin/bash
#SBATCH -o batch_files/kappa_map_gen_logs/job.%j_%a.out
#SBATCH -e batch_files/kappa_map_gen_logs/job.%j_%a.err
#SBATCH --job-name=kappa_map_gen_%j_%a    # Job name
#SBATCH --mem=200G ##300G for particle lc, rest 200G
#SBATCH -p cosma8
#SBATCH -A dp004
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ARIJCONL@ljmu.ac.uk
#SBATCH --time=24:00:00
pwd; hostname; date
module purge
conda activate patchy_screening

# START_NUM=$(( ($SLURM_ARRAY_TASK_ID - 1) * 1 + 1 ))
# END_NUM=$(( $SLURM_ARRAY_TASK_ID * 1 ))
# echo This is task $SLURM_ARRAY_TASK_ID, which will do runs $START_NUM to $END_NUM

## python3 kappa_map_gen_forJonah.py $SLURM_ARRAY_TASK_ID $SLURM_ARRAY_TASK_ID > kappa_map_gen_forJonah_$SLURM_ARRAY_TASK_ID.log
# python3 kappa_map_gen_forJonah.py $SLURM_ARRAY_TASK_ID $SLURM_ARRAY_TASK_ID
python3 kappa_map_gen_forJonah.py "$@"

date
