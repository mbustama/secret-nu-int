#!/bin/bash

#SBATCH --job-name=likelihood    # shows up in the output of 'squeue'
#SBATCH --time=4-23:59:59       # specify the requested wall-time
#SBATCH --partition=astro_long  # specify the partition to run on
#SBATCH --nodes=1               # number of nodes allocated for this job
#SBATCH --ntasks-per-node=20    # number of MPI ranks per node
#SBATCH --cpus-per-task=1       # number of OpenMP threads per MPI rank
#SBATCH --mail-type=ALL,TIME_LIMIT_90,TIME_LIMIT,ARRAY_TASKS
#SBATCH --mail-user=vkc652@alumni.ku.dk
#SBATCH -o %A.out # Standard output
#SBATCH -e %A.err # Standard error
##SBATCH --exclude=<node list>  # avoid nodes (e.g. --exclude=node786)


# Move to directory job was submitted from
cd $SLURM_SUBMIT_DIR

export LD_LIBRARY_PATH=/groups/astro/vkc652/MultiNest/lib/:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/groups/astro/vkc652/nuSQuIDS/lib/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/groups/astro/vkc652/SQuIDS/lib/

# Command to run
python -u Likelihood_analysis_parser.py --E_npts=10 --costhz_npts=3 --log10_energy_dep_npts=5 --epsabs=1.e-1 --epsrel=1.e-1 --n_live_points=5 --evidence_tolerance=5 --verbose=2

# python -u Likelihood_analysis_parser.py --E_npts=3 --nu_energy_num_nodes=10 --costhz_npts=3 --log10_energy_dep_npts=5 --epsabs=1.e-1 --epsrel=1.e-1 --n_live_points=5 --evidence_tolerance=0.5 --verbose=2

# mpiexec -n 1 python Likelihood_analysis_parser.py --E_npts=2 --nu_energy_num_nodes=10 --costhz_npts=2 --log10_energy_dep_npts=5 --epsabs=1.e-1 --epsrel=1.e-1 --n_live_points=5 --evidence_tolerance=0.5 --verbose=2


