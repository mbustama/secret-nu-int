import json
import numpy
from numpy import *
import scipy
import scipy.stats
import pymultinest
import argparse

from interp_atm_pdf import Initialize_Atmospheric_PDFs
from interp_astro_pdf import Initialize_Interpolator_Astrophysical_PDF
from full_likelihood import Log10_Likelihood

# Recommended run:
# python likelihood_analysis_parser.py --verbose=1 --n_live_points=200 --evidence_tolerance=0.01

# After MultiNest finishes, run this to analyse the results:
# multinest_marginals.py out/likelihood/

parser = argparse.ArgumentParser(description='Likelihood analysis')

parser.add_argument("--n_live_points", help="Default: 100",
	type=int, default=100)

parser.add_argument("--evidence_tolerance", help="Default: 0.1",
	type=float, default=0.1)

parser.add_argument("--resume", help="Resume MultiNest run [default: False]",
	action="store_true")

parser.add_argument("--verbose", help="Default: 0",
	type=int, default=0)

args = parser.parse_args()

n_live_points = args.n_live_points
evidence_tolerance = args.evidence_tolerance
resume = args.resume
verbose = args.verbose


def Prior(cube, ndim, nparams):

	# Spectral index. Uniform prior between 1.8 and 3.
	cube[0] = 1.8+cube[0]*1.2

	# Log10 of mass of mediator [GeV]. Log uniform prior between -3.0 and -1.0.
	cube[1] = -3.0+2.0*cube[1]

	# Log10 of coupling constant. Log uniform prior between -3.0 and -1.0
	cube[2] = -3.0+2.0*cube[2]

	# Expected number of astrophysical neutrinos. Uniform distribution between 0 and 80.
	cube[3] = cube[3]*80

	# Expected number of conv. atm. neutrinos. Uniform distribution between 0 and 80.
	cube[4] = cube[4]*80

	# Expected number of prompt atm. neutrinos. Uniform distribution between 0 and 80.
	cube[5] = cube[5]*80

	# Expected number of atm. muons. Uniform distribution between 0 and 80.
	cube[6] = cube[6]*80

	return 0


def Log10_Likelihood_MultiNest(cube, ndim, nparams):

	gamma = cube[0]
	log10_g = cube[1]
	log10_M = cube[2]
	N_a = cube[3]
	N_conv = cube[4]
	N_pr = cube[5]
	N_mu = cube[6]

	ll = Log10_Likelihood(gamma, log10_g, log10_M, N_a, N_conv, N_pr, N_mu,
            interp_astro_pdf_sh, pdf_atm_conv_sh, pdf_atm_pr_sh,
            interp_astro_pdf_tr, pdf_atm_conv_tr, pdf_atm_pr_tr,
            pdf_atm_muon_tr, num_ic_sh=58, num_ic_tr=22, verbose=verbose)

	return ll


# Initialize the atmospheric PDFs for all of the IceCube events
pdf_atm_conv_sh, pdf_atm_pr_sh, pdf_atm_pr_sh, pdf_atm_muon_sh, \
            pdf_atm_conv_tr, pdf_atm_pr_tr, pdf_atm_muon_tr = \
            Initialize_Atmospheric_PDFs(verbose=verbose)

# Initialize the astrophysical PDF interpolators for all of the IceCube events
interp_astro_pdf_sh, interp_astro_pdf_tr = \
    Initialize_Interpolator_Astrophysical_PDF(verbose=verbose)


parameters = ["gamma", "log10_g", "log10_M", "N_a", "N_conv", "N_pr", "N_mu"]
n_params = len(parameters)


# Run MultiNest
pymultinest.run(Log10_Likelihood_MultiNest, Prior, n_params,
	            outputfiles_basename='out/likelihood/',
				resume=False, verbose=verbose, n_live_points=n_live_points,
				seed=1, evidence_tolerance=evidence_tolerance,
				importance_nested_sampling=True, log_zero=-300.0)
# const_efficiency_mode=True, sampling_efficiency=1)

json.dump(parameters, open('out/likelihood/params.json', 'w')) # Save parameter names


