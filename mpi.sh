#!/bin/bash
#SBATCH --ntasks 8
#SBATCH -A dp004
#SBATCH -p cosma6
#SBATCH --job-name=get_flares
#SBATCH --array=0-40%20
#SBATCH -t 0-2:00
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=8
#SBATCH -o logs/std_output.%J
#SBATCH -e logs/std_error.%J


module purge
module load gnu_comp/7.3.0 openmpi/3.0.1 python/3.6.5

array=(010_z005p000 009_z006p000 008_z007p000 007_z008p000 006_z009p000 005_z010p000)

#set num=${array[$SLURM_ARRAY_TASK_ID]}

export PY_INSTALL=/cosma/home/dp004/dc-love2/.conda/envs/eagle/bin/python

mpiexec -n 8 $PY_INSTALL get_GEAGLE.py $SLURM_ARRAY_TASK_ID 010_z005p000
mpiexec -n 8 $PY_INSTALL get_GEAGLE.py $SLURM_ARRAY_TASK_ID 009_z006p000 
mpiexec -n 8 $PY_INSTALL get_GEAGLE.py $SLURM_ARRAY_TASK_ID 008_z007p000 
mpiexec -n 8 $PY_INSTALL get_GEAGLE.py $SLURM_ARRAY_TASK_ID 007_z008p000 
mpiexec -n 8 $PY_INSTALL get_GEAGLE.py $SLURM_ARRAY_TASK_ID 006_z009p000 
mpiexec -n 8 $PY_INSTALL get_GEAGLE.py $SLURM_ARRAY_TASK_ID 005_z010p000

echo "Job done, info follows..."
sacct -j $SLURM_JOBID --format=JobID,JobName,Partition,MaxRSS,Elapsed,ExitCode
exit
