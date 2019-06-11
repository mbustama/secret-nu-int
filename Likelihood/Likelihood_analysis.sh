#!/bin/bash

#SBATCH --job-name=likelihood    # shows up in the output of 'squeue'
#SBATCH --time=4-23:59:59       # specify the requested wall-time
#SBATCH --partition=astro_long  # specify the partition to run on
#SBATCH --nodes=4               # number of nodes allocated for this job
#SBATCH --ntasks-per-node=20    # number of MPI ranks per node
#SBATCH --cpus-per-task=1       # number of OpenMP threads per MPI rank
#SBATCH --mail-type=ALL,TIME_LIMIT_90,TIME_LIMIT,ARRAY_TASKS
#SBATCH --mail-user=vkc652@alumni.ku.dk
#SBATCH -o %A_%a.out # Standard output
#SBATCH -e %A_%a.err # Standard error
##SBATCH --exclude=<node list>  # avoid nodes (e.g. --exclude=node786)


# Move to directory job was submitted from
cd $SLURM_SUBMIT_DIR

# Command to run 
mpiexec -n 4 python Likelihood_analysis_parser.py --E_npts=2 --nu_energy_num_nodes=10 --costhz_npts=2 --log10_energy_dep_npts=5 --epsabs=1.e-1 --epsrel=1.e-1 --n_live_points=5 --evidence_tolerance=0.5 



