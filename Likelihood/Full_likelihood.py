from Prob_dist_astro import *
from Prob_dist_atm import *

ID_sh, lst_energy_sh, uncertainty_minus_sh, uncertainty_plus_sh, time_sh, declination_sh, RA_sh, Med_sh = Read_Data_File(os.getcwd()+'/'+'data_shower.txt')

ID_tr, lst_energy_tr, uncertainty_minus_tr, uncertainty_plus_tr, time_tr, declination_tr, RA_tr, Med_tr = Read_Data_File(os.getcwd()+'/'+'data_track.txt')

def Partial_likelihood_showers(N_a, N_conv, N_pr, N_mu, g, M, z_min, z_max, E_min, E_max, E_npts, gamma, nu_energy_min, nu_energy_max, nu_energy_num_nodes,
			costhz_val, costhz_npts, energy_dep, log10_energy_dep_int_min, log10_energy_dep_int_max, log10_energy_dep_min, log10_energy_dep_max, log10_energy_dep_npts,
            time_det_yr, volume_total, energy_nu_max, epsabs, epsrel, verbose):

	pdastro_sh = Prob_dist_astro(g, M, z_min, z_max, E_min, E_max, E_npts, gamma, nu_energy_min, nu_energy_max, nu_energy_num_nodes, costhz_val, costhz_npts, energy_dep,
            log10_energy_dep_int_min, log10_energy_dep_int_max, log10_energy_dep_min, log10_energy_dep_max, log10_energy_dep_npts,
            time_det_yr, volume_total, energy_nu_max, epsabs, epsrel, verbose, flag_compute_shower_rate = True, flag_compute_track_rate = False)

	pdatm_conv_sh = Prob_dist_atm_conv_pr(nu_energy_min, nu_energy_max, nu_energy_num_nodes, costhz_val, costhz_npts, energy_dep,log10_energy_dep_int_min, log10_energy_dep_int_max,
			log10_energy_dep_min, log10_energy_dep_max, log10_energy_dep_npts, time_det_yr, volume_total, energy_nu_max, epsabs, epsrel, verbose,
			flag_use_atm_fluxes_conv = True, flag_use_atm_fluxes_pr = False, flag_apply_self_veto = True, flag_compute_shower_rate = True, flag_compute_track_rate = False)

	pdatm_pr_sh = Prob_dist_atm_conv_pr(nu_energy_min, nu_energy_max, nu_energy_num_nodes, costhz_val, costhz_npts, energy_dep,log10_energy_dep_int_min, log10_energy_dep_int_max,
			log10_energy_dep_min, log10_energy_dep_max, log10_energy_dep_npts, time_det_yr, volume_total, energy_nu_max, epsabs, epsrel, verbose,
			flag_use_atm_fluxes_conv = False, flag_use_atm_fluxes_pr = True, flag_apply_self_veto = True, flag_compute_shower_rate = True, flag_compute_track_rate = False)


	pdatm_muon_sh = 0 #Probabillity distribution of atmospheric muons (showers)

	likelihood = N_a * pdastro_sh + N_conv * pdatm_conv_sh + N_pr * pdatm_pr_sh + N_mu * pdatm_muon_sh

	return likelihood



def Partial_likelihood_tracks(N_a, N_conv, N_pr, N_mu, g, M, z_min, z_max, E_min, E_max, E_npts, gamma, nu_energy_min, nu_energy_max, nu_energy_num_nodes,
			costhz_val, costhz_npts, energy_dep, log10_energy_dep_int_min, log10_energy_dep_int_max, log10_energy_dep_min, log10_energy_dep_max, log10_energy_dep_npts,
            time_det_yr, volume_total, energy_nu_max, epsabs, epsrel, verbose):

	pdastro_tr = Prob_dist_astro(g, M, z_min, z_max, E_min, E_max, E_npts, gamma, nu_energy_min, nu_energy_max, nu_energy_num_nodes, costhz_val, costhz_npts, energy_dep,
            log10_energy_dep_int_min, log10_energy_dep_int_max, log10_energy_dep_min, log10_energy_dep_max, log10_energy_dep_npts,
            time_det_yr, volume_total, energy_nu_max, epsabs, epsrel, verbose, flag_compute_shower_rate = False, flag_compute_track_rate = True)

	pdatm_conv_tr = Prob_dist_atm_conv_pr(nu_energy_min, nu_energy_max, nu_energy_num_nodes, costhz_val, costhz_npts, energy_dep,log10_energy_dep_int_min, log10_energy_dep_int_max,
			log10_energy_dep_min, log10_energy_dep_max, log10_energy_dep_npts, time_det_yr, volume_total, energy_nu_max, epsabs, epsrel, verbose,
			flag_use_atm_fluxes_conv = True, flag_use_atm_fluxes_pr = False, flag_apply_self_veto = True, flag_compute_shower_rate = False, flag_compute_track_rate = True)


	pdatm_pr_tr = Prob_dist_atm_conv_pr(nu_energy_min, nu_energy_max, nu_energy_num_nodes, costhz_val, costhz_npts, energy_dep,log10_energy_dep_int_min, log10_energy_dep_int_max,
			log10_energy_dep_min, log10_energy_dep_max, log10_energy_dep_npts, time_det_yr, volume_total, energy_nu_max, epsabs, epsrel, verbose,
			flag_use_atm_fluxes_conv = False, flag_use_atm_fluxes_pr = True, flag_apply_self_veto = True, flag_compute_shower_rate = False, flag_compute_track_rate = True)


	pdatm_muon_tr = Prob_dist_atm_muon(energy_dep, log10_energy_dep_int_min, log10_energy_dep_int_max, log10_energy_dep_min, log10_energy_dep_max, log10_energy_dep_npts, epsabs, epsrel, verbose)

	likelihood = N_a * pdastro_tr + N_conv * pdatm_conv_tr + N_pr * pdatm_pr_tr + N_mu * pdatm_muon_tr

	return likelihood



def Full_likelihood(N_a, N_conv, N_pr, N_mu, g, M, gamma, nu_energy_min, nu_energy_max, z_min, z_max, E_min, E_max, E_npts, nu_energy_num_nodes,
			costhz_npts, log10_energy_dep_int_min, log10_energy_dep_int_max, log10_energy_dep_min, log10_energy_dep_max, log10_energy_dep_npts,
            time_det_yr, volume_total, energy_nu_max, epsabs, epsrel, verbose):


	FL_sh = 1
	for i in range(3): #len(lst_energy_sh)):
		costhz_val = np.cos((declination_sh[i] + 90)*np.pi/180)
		energy_dep = lst_energy_sh[i]*1000

		FL_sh = FL_sh * Partial_likelihood_showers(N_a, N_conv, N_pr, N_mu, g, M, z_min, z_max, E_min, E_max, E_npts, gamma, nu_energy_min, nu_energy_max, nu_energy_num_nodes,
							costhz_val, costhz_npts, energy_dep, log10_energy_dep_int_min, log10_energy_dep_int_max, log10_energy_dep_min, log10_energy_dep_max, log10_energy_dep_npts,
            				time_det_yr, volume_total, energy_nu_max, epsabs, epsrel, verbose)


	FL_tr = 1

	for i in range(3): #len(lst_energy_tr)):
		costhz_val = np.cos((declination_tr[i] + 90)*np.pi/180)
		energy_dep = lst_energy_tr[i]*1000

		FL_tr = FL_tr * Partial_likelihood_tracks(N_a, N_conv, N_pr, N_mu, g, M, z_min, z_max, E_min, E_max, E_npts, gamma, nu_energy_min, nu_energy_max, nu_energy_num_nodes,
							costhz_val, costhz_npts, energy_dep, log10_energy_dep_int_min, log10_energy_dep_int_max, log10_energy_dep_min, log10_energy_dep_max, log10_energy_dep_npts,
            				time_det_yr, volume_total, energy_nu_max, epsabs, epsrel, verbose)

	FL = np.exp(- N_a - N_conv - N_pr - N_mu) * FL_sh * FL_tr

	return FL

"""
log10_nu_energy_min = 2.8
log10_nu_energy_max = 9.2

test = Full_likelihood(20, 20, 20, 20, 0.03, 0.01, 2, nu_energy_min = 10**log10_nu_energy_min, nu_energy_max = 10**log10_nu_energy_max,
			z_min = 0, z_max = 4, E_min = 3, E_max = 8, E_npts = 10, nu_energy_num_nodes = 150,
			costhz_npts = 2, log10_energy_dep_int_min = 4, log10_energy_dep_int_max = 7, log10_energy_dep_min = 3.8, log10_energy_dep_max = 7.2, log10_energy_dep_npts = 50,
            time_det_yr = 8, volume_total = 6.44e14, energy_nu_max = 1e8, epsabs =1e-3, epsrel = 1e-3, verbose=1)
print(test)

"""
