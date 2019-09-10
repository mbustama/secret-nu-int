import json
import numpy
from numpy import *
import scipy.stats, scipy
import pymultinest
from Full_likelihood import *


def Prior(cube, ndim, nparams): 

	#Spectral index. Uniform prior between 2 and 3. 
	cube[0] = cube[0] + 2

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

	return 0


def Log_Like(cube, ndim, nparams): 

	gamma = cube[0]
	M = cube[1]
	g = cube[2]
	N_a = cube[3]
	N_conv = cube[4]
	N_pr = cube[5]
	N_mu = cube[6]

	z_min = 0
	z_max = 4
	E_min = 3
	E_max = 8
	E_npts = 5 #200

	log10_nu_energy_min = 2.8
	log10_nu_energy_max = 9.2

	nu_energy_min = 10**log10_nu_energy_min
	nu_energy_max = 10**log10_nu_energy_max
	nu_energy_num_nodes = 20 #150
	costhz_npts = 2 #50
	log10_energy_dep_int_min = 4
	log10_energy_dep_int_max = 7
	log10_energy_dep_min = 3.8
	log10_energy_dep_max = 7.2
	log10_energy_dep_npts = 10 #50
	time_det_yr = 8
	volume_total = 6.44e14
	energy_nu_max = 1e8
	epsabs = 1e-3
	epsrel = 1e-3
	verbose = 1

	likelihood = Full_likelihood(N_a, N_conv, N_pr, N_mu, g, M, z_min, z_max, E_min, E_max, E_npts, gamma, nu_energy_min, nu_energy_max, nu_energy_num_nodes, 
								costhz_npts, log10_energy_dep_int_min, log10_energy_dep_int_max, log10_energy_dep_min, log10_energy_dep_max, log10_energy_dep_npts, 
            					time_det_yr, volume_total, energy_nu_max, epsabs, epsrel, verbose)

	log_l = np.log10(likelihood)
	print('Log_l=',log_l)
	return log_l




parameters = ["gamma", "M", "g", "N_a", "N_conv", "N_pr", "N_mu"]
n_params = len(parameters)




# Run MultiNest
pymultinest.run(Log_Like, Prior, n_params, outputfiles_basename='out/',
				resume=True, verbose=True, n_live_points=100, seed=1, 
				evidence_tolerance=0.1, importance_nested_sampling=True)

json.dump(parameters, open('out/params.json', 'w')) # Save parameter names

