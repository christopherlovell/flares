#!/bin/bash
#SBATCH --ntasks 16
#SBATCH -A dp004
#SBATCH -p cosma6
#SBATCH --job-name=get_AGNdt9
#SBATCH --array=0-6%6
#SBATCH -t 0-3:00
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=4
#SBATCH -o logs/std_output.%J
#SBATCH -e logs/std_error.%J

module purge
module load gnu_comp/7.3.0 openmpi/3.0.1 hdf5/1.10.3 python/3.6.5

#array=(010_z005p000 009_z006p000 008_z007p000 007_z008p000 006_z009p000 005_z010p000)
array=(002_z009p993 003_z008p988 004_z008p075 005_z007p050 006_z005p971 008_z005p037)

#export PY_INSTALL=/cosma/home/dp004/dc-love2/.conda/envs/eagle/bin/python

source /cosma7/data/dp004/dc-payy1/my_files/photometry/venv_photo/bin/activate

mpiexec -n 16 python3 download_methods.py $SLURM_ARRAY_TASK_ID ${array[$SLURM_ARRAY_TASK_ID]} AGNdT9

echo "Job done, info follows..."
sacct -j $SLURM_JOBID --format=JobID,JobName,Partition,MaxRSS,Elapsed,ExitCode
exit
