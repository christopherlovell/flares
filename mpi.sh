#!/bin/bash
#SBATCH --ntasks 4
#SBATCH -A dp004
#SBATCH -p cosma6
#SBATCH --job-name=python
##SBATCH --array=0-40%20
#SBATCH -t 0-0:20
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=4
#SBATCH -o logs/std_output.%J
#SBATCH -e logs/std_error.%J


module purge
module load gnu_comp/7.3.0 openmpi/3.0.1 python/3.6.5

#array=(010_z005p000 009_z006p000 008_z007p000 007_z008p000 006_z009p000 005_z010p000)

#set VAR1=38
#set VAR2=${array[$SLURM_ARRAY_TASK_ID]}

mpiexec -n 4 python3 get_GEAGLE.py 38 010_z005p000
##$SLURM_ARRAY_TASK_ID 003_z012p000

echo "Job done, info follows..."
sacct -j $SLURM_JOBID --format=JobID,JobName,Partition,MaxRSS,Elapsed,ExitCode
exit
