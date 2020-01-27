#!/bin/bash
#SBATCH -A dp004
#SBATCH -p cosma7
#SBATCH --job-name=phot_write
#SBATCH -t 0-10:00
#SBATCH --ntasks 6
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=1
#SBATCH -o logs/std_output.%J
#SBATCH -e logs/std_error.%J


module purge
module load python/3.6.5 gnu_comp/7.3.0 openmpi/3.0.1 parallel_hdf5/1.10.3

source ../photometry/venv_photo/bin/activate

mpiexec -n 6 python3 download_photproperties.py REF Flux
mpiexec -n 6 python3 download_photproperties.py AGNdT9 Flux

echo "Job done, info follows..."
sacct -j $SLURM_JOBID --format=JobID,JobName,Partition,MaxRSS,Elapsed,ExitCode
exit
