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


Initialize_All_Cross_Sections(prefix_dsdy_nu_nucleon='dsdy_ct14nn',
            prefix_dsdy_nu_electron='dsdy_electron',
            prefix_cs_nu_electron='cs_electron', kx=1, ky=1, k=1, s=0,
            verbose=verbose)

# ## >>> Initialize atm. fluxes, passing fractions

# lst_costhz_sel, lst_energy_sel, flux_mum_conv, flux_mup_conv, \
#     flux_mum_pr, flux_mup_pr, flux_nue_conv, flux_nuebar_conv, \
#     flux_nue_pr, flux_nuebar_pr, flux_numu_conv, flux_numubar_conv, \
#     flux_numu_pr, flux_numubar_pr, flux_nutau_pr, flux_nutaubar_pr = \
#     Initialize_Atm_Fluxes(flux_set='avg', flag_return_total=False,
#     flag_return_plus_minus=False, kx=1, ky=1, s=0, verbose=verbose)

# # Passing fractions
# # Call the interpolating functions as pf_spl(energy, costhz)
# lst_energy_pf, lst_costhz_pf, pf_nue_conv, pf_nuebar_conv, pf_numu_conv, \
#     pf_numubar_conv, pf_nue_pr, pf_nuebar_pr, pf_numu_pr, pf_numubar_pr = \
#     Initialize_Passing_Fractions(kx=1, ky=1, s=0, verbose=verbose)

# mix_params_data_set ='nufit_4_0_with_sk'
# mix_params_mass_ordering ='no'

# # lst_mix_params is passed to NuSQuIDS and contains the mixing parameters
# # [s12sq_bf, s23sq_bf, s13sq_bf, deltaCP_bf, Delta_m2_21_bf, Delta_m2_31_bf, Delta_m2_32_bf]
# global lst_mix_params
# lst_mix_params = Mixing_Parameters(mix_params_data_set, mix_params_mass_ordering)[0]

# ID_sh, lst_energy_sh, uncertainty_minus_sh, uncertainty_plus_sh, time_sh, declination_sh, RA_sh, Med_sh = Read_Data_File(os.getcwd()+'/'+'data_shower.txt')

# ID_tr, lst_energy_tr, uncertainty_minus_tr, uncertainty_plus_tr, time_tr, declination_tr, RA_tr, Med_tr = Read_Data_File(os.getcwd()+'/'+'data_track.txt')


def Prior(cube, ndim, nparams):

	#Spectral index. Uniform prior between 2 and 3.
	cube[0] = cube[0] + 2

	
	#Mass of mediator. Log uniform prior between 10^-5 and 10^2
	cube[1] = 10**(cube[1]*7 - 5)
	"""
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
	M = cube[1]
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




parameters = ["gamma", "M"]#, "g", "N_a", "N_conv", "N_pr", "N_mu"]
n_params = len(parameters)




# Run MultiNest
pymultinest.run(Log_Like, Prior, n_params, outputfiles_basename='Likelihood_out_2D/',
				resume=True, verbose=True, n_live_points=n_live_points, seed=1,
				evidence_tolerance=evidence_tolerance, importance_nested_sampling=True, log_zero=-300)

json.dump(parameters, open('Likelihood_out_2D/params.json', 'w')) # Save parameter names
