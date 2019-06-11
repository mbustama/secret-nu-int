import json
import numpy
from numpy import *
import scipy.stats, scipy
import pymultinest
from Full_likelihood import *
import argparse 

parser = argparse.ArgumentParser(description='Likelihood Analysis')

parser.add_argument("--z_min", help="Redshift at which flux is generated", type=int, default=0)

parser.add_argument("--z_max", help="Initial value redshift", type=int, default=4)

parser.add_argument("--E_min", help="Minimum energy in array", type=int, default=3)

parser.add_argument("--E_max", help="Maximum energy in array", type=int, default=8)

parser.add_argument("--E_npts", help="Number of energy bins in array", type=int, default=200)

parser.add_argument("--log10_nu_energy_min", help="Default: 2.8", type=float, default=2.8)

parser.add_argument("--log10_nu_energy_max", help="Default: 9.2", type=float, default=9.2)

parser.add_argument("--nu_energy_num_nodes", help="Default: 150", type=int, default=150)

parser.add_argument("--costhz_npts", help="Default: 50", type=int, default=50)

parser.add_argument("--log10_energy_dep_min", help="Default: 3.8", type=float, default=3.8)

parser.add_argument("--log10_energy_dep_max", help="Default: 7.2", type=float, default=7.2)

parser.add_argument("--log10_energy_dep_npts", help="Default: 50", type=int, default=50)

parser.add_argument("--log10_energy_dep_int_min", help="Default: 4.0", type=float, default=4.0)

parser.add_argument("--log10_energy_dep_int_max", help="Default: 7.0", type=float, default=7.0)

parser.add_argument("--time_det_yr", help="Default: 8.0", type=float, default=8.0)

parser.add_argument("--volume_total", help="Default: 6.440e14", type=float, default=6.440e14)

parser.add_argument("--energy_nu_max", help="Default: 1.e8", type=float, default=1.e8)

parser.add_argument("--epsabs", help="Default: 1.e-3", type=float, default=1.e-3)

parser.add_argument("--epsrel", help="Default: 1.e-3", type=float, default=1.e-3)

parser.add_argument("--verbose", help="Default: 0", type=int, default=0)

parser.add_argument("--n_live_points", help="Default: 100", type=int, default=100)

parser.add_argument("--evidence_tolerance", help="Default: 0.1", type=float, default=0.1)


args = parser.parse_args()

z_min = args.z_min
z_max = args.z_max
E_min = args.E_min
E_max = args.E_max
E_npts = args.E_npts
log10_nu_energy_min = args.log10_nu_energy_min
log10_nu_energy_max = args.log10_nu_energy_max
nu_energy_num_nodes = args.nu_energy_num_nodes
costhz_npts = args.costhz_npts
log10_energy_dep_min = args.log10_energy_dep_min
log10_energy_dep_max = args.log10_energy_dep_max
log10_energy_dep_npts = args.log10_energy_dep_npts
log10_energy_dep_int_min = args.log10_energy_dep_int_min
log10_energy_dep_int_max = args.log10_energy_dep_int_max
time_det_yr = args.time_det_yr
volume_total = args.volume_total
energy_nu_max = args.energy_nu_max
epsabs = args.epsabs
epsrel = args.epsrel
verbose = args.verbose
n_live_points = args.n_live_points
evidence_tolerance = args.evidence_tolerance



def Prior(cube, ndim, nparams): 

	#Spectral index. Uniform prior between 2 and 3. 
	cube[0] = cube[0] + 2

	"""
	#Mass of mediator. Log uniform prior between 10^-5 and 10^2
	cube[1] = 10**(cube[1]*7 - 5)

	#Coupling constant. Log uniform prior between 10^-3 and 1. 
	cube[2] = 10**(cube[2]*3 -3)

	#Expected number of astrophysical neutrinos. Uniform distribution between 0 and 80.
	cube[3] = cube[3] * 80 

	#Expected number of conv. atm. neutrinos. Uniform distribution between 0 and 80.
	cube[4] = cube[4] * 80 

	#Expected number of prompt atm. neutrinos. Uniform distribution between 0 and 80.
	cube[5] = cube[5] * 80 

	#Expected number of atm. muons. Uniform distribution between 0 and 80.
	cube[6] = cube[6] * 80 
	"""
	return 0


def Log_Like(cube, ndim, nparams): 

	gamma = cube[0]
	M = 0.01 #cube[1]
	g = 0.03 #cube[2]
	N_a = 20 #cube[3]
	N_conv = 20 #cube[4]
	N_pr = 20 #cube[5]
	N_mu = 20 #cube[6]

	nu_energy_min = 10**log10_nu_energy_min
	nu_energy_max = 10**log10_nu_energy_max

	likelihood = Full_likelihood(N_a, N_conv, N_pr, N_mu, g, M, gamma, nu_energy_min, nu_energy_max, z_min=z_min, z_max=z_max, E_min=E_min, E_max=E_max, E_npts=E_npts,
								nu_energy_num_nodes=nu_energy_num_nodes, costhz_npts=costhz_npts, log10_energy_dep_int_min=log10_energy_dep_int_min, log10_energy_dep_int_max=log10_energy_dep_int_max, 
								log10_energy_dep_min=log10_energy_dep_min, log10_energy_dep_max=log10_energy_dep_max, log10_energy_dep_npts=log10_energy_dep_npts, 
            					time_det_yr=time_det_yr, volume_total=volume_total, energy_nu_max=energy_nu_max, epsabs=epsabs, epsrel=epsrel, verbose=verbose)

	log_l = np.log10(likelihood)

	return log_l




parameters = ["gamma"]#, "M", "g", "N_a", "N_conv", "N_pr", "N_mu"]
n_params = len(parameters)




# Run MultiNest
pymultinest.run(Log_Like, Prior, n_params, outputfiles_basename='Likelihood_out_1D/',
				resume=True, verbose=True, n_live_points=n_live_points, seed=1, 
				evidence_tolerance=evidence_tolerance, importance_nested_sampling=True)

json.dump(parameters, open('Likelihood_out_1D/params.json', 'w')) # Save parameter names