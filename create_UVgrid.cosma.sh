#!/bin/bash
#SBATCH -A dp004
#SBATCH -p cosma6
#SBATCH --job-name=creategrid_UV
#SBATCH -t 0-1:30
#SBATCH --ntasks 20
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=2
#SBATCH -o logs/std_output.%J
#SBATCH -e logs/std_error.%J


module purge
module load python/3.6.5 gnu_comp/7.3.0 openmpi/3.0.1 hdf5/1.10.3

source ./venv_fl/bin/activate

### 3 inputs are (Gridgen or not), (kappa_BC) and (extinction curve: default, Calzetti, SMC, N18)

mpiexec -n 20 python3 creategrid_and_fit.py Gridgen 5 Calzetti

echo "Job done, info follows..."
sacct -j $SLURM_JOBID --format=JobID,JobName,Partition,MaxRSS,Elapsed,ExitCode
exit
