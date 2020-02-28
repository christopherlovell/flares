#!/bin/bash
#SBATCH -A dp004
#SBATCH -p cosma6
#SBATCH --job-name=phot_write
#SBATCH -t 0-2:00
#SBATCH --ntasks 6
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=1
#SBATCH -o logs/std_output.%J
#SBATCH -e logs/std_error.%J


module purge
module load python/3.6.5 gnu_comp/7.3.0 openmpi/3.0.1 parallel_hdf5/1.10.3

source ./venv_fl/bin/activate

## Change the argument to the script to Luminosity or Flux; FLARES or REF or AGNdT9
## as required

### For FLARES galaxies, change ntasks as required
mpiexec -n 6 python3 download_photproperties.py FLARES Flux
mpiexec -n 6 python3 download_photproperties.py FLARES Luminosity

### For PERIODIC boxes: REF and AGNdT9, change ntasks as required
# mpiexec -n 6 python3 download_photproperties.py REF Flux
# mpiexec -n 6 python3 download_photproperties.py AGNdT9 Flux

echo "Job done, info follows..."
sacct -j $SLURM_JOBID --format=JobID,JobName,Partition,MaxRSS,Elapsed,ExitCode
exit
