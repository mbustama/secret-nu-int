from __future__ import division
import numpy as np
import os

from Neutrino_Flux_Earth import *
from global_defs import *
from global_tools import *
from event_rate import *

"""
Parameters:
   N_{a, nu, p, mu}:
       Total number of expected events from astrophysical,
       conv. atmospheric, prompt atmospheric and atmospheric muon neutrinos.
   N_obs_{sh, tr}:
       Total number of observed showers and tracks.
   E_dep:
       Deposited energy in IceCube.
   DNDE_dep:
       Event rate spectra.
   g:
       Coupling constant.
   M:
       Mass of mediator.
   gamma:
       Spectral index (E^-gamma).
"""


def Prob_dist_astro_den(nu_energy_min, nu_energy_max, nu_energy_num_nodes, costhz_npts, Nu_Fluxes_Initial, log10_energy_dep_int_min,
    log10_energy_dep_int_max, log10_energy_dep_min, log10_energy_dep_max, log10_energy_dep_npts, time_det_yr, volume_total, energy_nu_max, epsabs, epsrel, verbose):
    
    mix_params_data_set ='nufit_4_0_with_sk'
    mix_params_mass_ordering ='no'

    # lst_mix_params is passed to NuSQuIDS and contains the mixing parameters
    # [s12sq_bf, s23sq_bf, s13sq_bf, deltaCP_bf, Delta_m2_21_bf, Delta_m2_31_bf, Delta_m2_32_bf]
    lst_mix_params = Mixing_Parameters(mix_params_data_set, mix_params_mass_ordering)[0]

    # Array of cos(theta_z) to compute
    lst_costhz = np.linspace(-1.0, 1.0, costhz_npts)

    lst_flux_det, lst_flux_earth = \
        Nu_Fluxes_At_Detector(nu_energy_min, nu_energy_max, nu_energy_num_nodes, lst_costhz, Nu_Fluxes_Initial, 
            error_rel=1.e-7, error_abs=1.e-7, h_max=500., verbose_level=verbose, lst_mix_params=lst_mix_params, epsabs=1.e-8, epsrel=1.e-8, flag_return_nusquids_format=True)

    log10_energy_dep_int_step = log10_energy_dep_int_max - log10_energy_dep_int_min

    Event_spec = \
        Generate_Event_Spectrum_All_Sky(lst_flux_det, lst_costhz, filename_data_out_suffix='',
            flag_save_data=False, flag_plot_histogram=False, flag_initialize_cross_sections=False,
            flag_compute_shower_rate=True, flag_compute_track_rate=True, 
            flag_sh_nux_nc=True, flag_sh_nue_cc=True, flag_sh_nutau_cc=True, flag_sh_nul_electron_to_electron=True,
            flag_sh_nuebar_electron_to_tau=False, lag_sh_nuebar_electron_to_hadrons=True, flag_sh_nutau_electron_to_tau=False,
            flag_tr_numu_cc=True, flag_tr_nutau_cc=True, flag_tr_nuebar_electron_to_tau=False, 
            flag_tr_nutau_electron_to_tau=False, flag_tr_nuebar_electron_to_muon=False, flag_tr_numu_electron_to_muon=False,
            log10_energy_dep_min=log10_energy_dep_min, log10_energy_dep_max=log10_energy_dep_max, log10_energy_dep_npts=log10_energy_dep_npts,
            log10_energy_dep_int_min=log10_energy_dep_int_min, log10_energy_dep_int_max=log10_energy_dep_int_max, log10_energy_dep_int_step=log10_energy_dep_int_step,
            time_det_yr=time_det_yr, volume_total=volume_total, energy_nu_max=energy_nu_max, integration_method='quad',
            miniter=1, maxiter=500, epsabs=epsabs, epsrel=epsrel, s=0, lst_headers_argparse=None, verbose=verbose)

    print("Event=", Event_spec[0][2][:])

    Denominator = np.sum(Event_spec[0][2][:] + Event_spec[1][2][:] + Event_spec[2][2][:] + Event_spec[3][2][:] + Event_spec[4][2][:] + Event_spec[5][2][:])

    print("Denominator=", Denominator)

    return Denominator 


def Prob_dist_astro_num(nu_energy_min, nu_energy_max, nu_energy_num_nodes, costhz_val, Nu_Fluxes_Initial, energy_dep,
        time_det_yr, volume_total, energy_nu_max, epsabs, epsrel, verbose, flag_compute_shower_rate, flag_compute_track_rate): 

    mix_params_data_set ='nufit_4_0_with_sk'
    mix_params_mass_ordering ='no'

    lst_mix_params = Mixing_Parameters(mix_params_data_set, mix_params_mass_ordering)[0]

    lst_costhz = np.array([costhz_val])

    lst_flux_det, lst_flux_earth = \
        Nu_Fluxes_At_Detector(nu_energy_min, nu_energy_max, nu_energy_num_nodes, lst_costhz, Nu_Fluxes_Initial, 
            error_rel=1.e-7, error_abs=1.e-7, h_max=500., verbose_level=verbose, lst_mix_params=lst_mix_params, epsabs=1.e-8, epsrel=1.e-8, flag_return_nusquids_format=True)


    if (verbose > 0): print("costhz = "+str(costhz_val))

    if (verbose > 0): print("  Building flux splines... ", end='')

    lst_energy_nu = lst_flux_det[0][0] # [GeV]

    lst_flux_det_nue = lst_flux_det[0][1] # nu_e [GeV^{-1} cm^{-2} s^{-1} sr^{-1}]
    
    lst_flux_det_nuebar = lst_flux_det[0][2] # nu_e-bar

    lst_flux_det_numu = lst_flux_det[0][3] # nu_mu

    lst_flux_det_numubar = lst_flux_det[0][4] # nu_mu-bar

    lst_flux_det_nutau = lst_flux_det[0][5] # nu_tau

    lst_flux_det_nutaubar = lst_flux_det[0][6] # nu_tau-bar


    # Interpolating functions [GeV^{-1} cm^{-2} s^{-1} sr^{-1}]
    interp_flux_det_nue = interp1d(lst_energy_nu, lst_flux_det_nue,
        kind='linear', bounds_error=False, fill_value='extrapolate')
    interp_flux_det_nuebar = interp1d(lst_energy_nu, lst_flux_det_nuebar,
        kind='linear', bounds_error=False, fill_value='extrapolate')
    interp_flux_det_numu = interp1d(lst_energy_nu, lst_flux_det_numu,
        kind='linear', bounds_error=False, fill_value='extrapolate')
    interp_flux_det_numubar = interp1d(lst_energy_nu, lst_flux_det_numubar,
        kind='linear', bounds_error=False, fill_value='extrapolate')
    interp_flux_det_nutau = interp1d(lst_energy_nu, lst_flux_det_nutau,
        kind='linear', bounds_error=False, fill_value='extrapolate')
    interp_flux_det_nutaubar = interp1d(lst_energy_nu, lst_flux_det_nutaubar,
        kind='linear', bounds_error=False, fill_value='extrapolate')

    # Flux functions to integrate in E_dep [GeV^{-1} cm^{-2} s^{-1} sr^{-1}]
    Flux_NuE_At_Detector = lambda energy_nu: \
        interp_flux_det_nue(energy_nu)
    Flux_NuEBar_At_Detector = lambda energy_nu: \
        interp_flux_det_nuebar(energy_nu)
    Flux_NuMu_At_Detector = lambda energy_nu: \
        interp_flux_det_numu(energy_nu)
    Flux_NuMuBar_At_Detector = lambda energy_nu: \
        interp_flux_det_numubar(energy_nu)
    Flux_NuTau_At_Detector = lambda energy_nu: \
        interp_flux_det_nutau(energy_nu)
    Flux_NuTauBar_At_Detector = lambda energy_nu: \
        interp_flux_det_nutaubar(energy_nu)
    lst_Flux_Nu_At_Detector = [Flux_NuE_At_Detector, Flux_NuMu_At_Detector,
        Flux_NuTau_At_Detector, Flux_NuEBar_At_Detector,
        Flux_NuMuBar_At_Detector, Flux_NuTauBar_At_Detector]

    if (verbose > 0): print("Done")

    # Detection time
    time_det = time_det_yr*365.*24.*60.*60. # [s]

    # === Shower rate ===

    if (flag_compute_shower_rate == True):

        # Calculate the shower rate
        if (verbose > 0): print("  Calculating the shower rate... ", \
            end='')
        # For each value of energy_dep, Diff_Shower_Rate_Dep_Energy_Total returns
        # [diff_shower_rate_nue_nuebar, diff_shower_rate_numu_numubar,
        #       diff_shower_rate_nutau_nutaubar]
        lst_diff_rate = \
            Diff_Shower_Rate_Dep_Energy_Total(energy_dep, time_det, lst_Flux_Nu_At_Detector, volume_total=volume_total,
                energy_nu_max=energy_nu_max, integration_method='quad',
                miniter=1, maxiter=500, epsabs=epsabs, epsrel=epsrel,
                flag_nux_nc=True, flag_nue_cc=True, flag_nutau_cc=True,
                flag_nul_electron_to_electron= True, flag_nuebar_electron_to_tau= False,
                flag_nuebar_electron_to_hadrons= True, flag_nutau_electron_to_tau= False) 

    # === Track rate ===

    elif (flag_compute_track_rate == True):

        # Calculate the track rate
        if (verbose > 0): print("  Calculating the track rate... ", end='')
        # For each value of energy_dep, Diff_Track_Rate_Dep_Energy_Total returns
        # [diff_track_rate_nue_nuebar, diff_track_rate_numu_numubar,
        #       diff_track_rate_nutau_nutaubar]
        lst_diff_rate = \
            Diff_Track_Rate_Dep_Energy_Total(energy_dep, time_det, lst_Flux_Nu_At_Detector, volume_total=volume_total,
                energy_nu_max=energy_nu_max, integration_method='quad',
                miniter=1, maxiter=500, epsabs=epsabs, epsrel=epsrel,
                flag_numu_cc=True, flag_nutau_cc=True, flag_nuebar_electron_to_tau=False,
                flag_nutau_electron_to_tau=False, flag_nuebar_electron_to_muon= False, flag_numu_electron_to_muon= False) 

    Numerator = np.array([lst_diff_rate[0] + lst_diff_rate[1] + lst_diff_rate[2]]) 

    return Numerator
  

def Prob_dist_astro(g, M, z_min, z_max, E_min, E_max, E_npts, gamma, nu_energy_min, nu_energy_max, nu_energy_num_nodes, costhz_val, costhz_npts, energy_dep, 
            log10_energy_dep_int_min, log10_energy_dep_int_max, log10_energy_dep_min, log10_energy_dep_max, log10_energy_dep_npts, 
            time_det_yr, volume_total, energy_nu_max, epsabs, epsrel, verbose, flag_compute_shower_rate, flag_compute_track_rate): 

    flux_array = Neutrino_Flux(g, M, z_min, z_max, E_min, E_max, E_npts, gamma, m=1.e-10)

    lst_energy_nu = flux_array[:,0] 

    lst_nu_flux = flux_array[:,1]

    interp_nu_flux = interp1d(lst_energy_nu, lst_nu_flux, kind='linear', bounds_error=False, fill_value='extrapolate')

    Nu_Fluxes_Initial = lambda lst_energy_nu, **kwargs: Nu_Fluxes_Initial_Format_NuSQuIDS(lst_energy_nu, interp_nu_flux, **kwargs)

    num = \
        Prob_dist_astro_num(nu_energy_min, nu_energy_max, nu_energy_num_nodes, costhz_val, Nu_Fluxes_Initial, energy_dep, 
           time_det_yr, volume_total, energy_nu_max, epsabs, epsrel, verbose, flag_compute_shower_rate = flag_compute_shower_rate, flag_compute_track_rate = flag_compute_track_rate)

    den = \
        Prob_dist_astro_den(nu_energy_min, nu_energy_max, nu_energy_num_nodes, costhz_npts, Nu_Fluxes_Initial, log10_energy_dep_int_min,
            log10_energy_dep_int_max, log10_energy_dep_min, log10_energy_dep_max, log10_energy_dep_npts, time_det_yr, volume_total, energy_nu_max, epsabs, epsrel, verbose)

    prob = num/den 

    print("Prob_astro=", prob)

    return prob 




"""
external_flux_filename='in/fluxes/Freestreaming200.txt'

lst_energy_nu, lst_nu_flux = Read_Data_File(os.getcwd()+'/'+external_flux_filename)

interp_nu_flux = interp1d(lst_energy_nu, lst_nu_flux, kind='linear', bounds_error=False, fill_value='extrapolate')

test_flux_initial = Nu_Fluxes_Initial = lambda lst_energy_nu, **kwargs: Nu_Fluxes_Initial_Format_NuSQuIDS(lst_energy_nu, interp_nu_flux, **kwargs)

Initialize_All_Cross_Sections(prefix_dsdy_nu_nucleon='dsdy_ct14nn',
            prefix_dsdy_nu_electron='dsdy_electron',
            prefix_cs_nu_electron='cs_electron', kx=1, ky=1, k=1, s=0,
            verbose=0)

log10_nu_energy_min = 2.8
log10_nu_energy_max = 9.2

test = Prob_dist_astro(nu_energy_min = 10**log10_nu_energy_min, nu_energy_max = 10**log10_nu_energy_max, nu_energy_num_nodes = 150, costhz = 0.5, costhz_npts = 2, Nu_Fluxes_Initial = test_flux_initial, 
            energy_dep = 1e5, log10_energy_dep_int_min = 4, log10_energy_dep_int_max = 7, log10_energy_dep_min = 3.8, log10_energy_dep_max = 7.2, log10_energy_dep_npts = 50, 
            time_det_yr = 8, volume_total = 6.44e14, energy_nu_max = 1e8, epsabs =1e-3, epsrel = 1e-3, verbose=1, flag_compute_shower_rate = True, flag_compute_track_rate = False)

np.savetxt('test_prob.txt', test)


def Full_likelihood(g, M, external_flux_filename, z_min, z_max, E_min, E_max, E_npts, gamma): 
    
    flux_array = Neutrino_Flux(g, M, external_flux_filename, z_min, z_max, E_min, E_max, E_npts, gamma, m=1.e-10)

    lst_energy_nu, lst_nu_flux = Read_Data_File(os.getcwd()+'/'+external_flux_filename)

    interp_nu_flux = interp1d(lst_energy_nu, lst_nu_flux, kind='linear', bounds_error=False, fill_value='extrapolate')

    Nu_Fluxes_Initial = lambda lst_energy_nu, **kwargs: Nu_Fluxes_Initial_Format_NuSQuIDS(lst_energy_nu, interp_nu_flux, **kwargs)

    # Initialize all ds/dy and cs cross sections.  These are loaded as global
    # interpolating functions

        Initialize_All_Cross_Sections(prefix_dsdy_nu_nucleon='dsdy_ct14nn',
            prefix_dsdy_nu_electron='dsdy_electron',
            prefix_cs_nu_electron='cs_electron', kx=kx, ky=ky, k=k, s=s,
            verbose=verbose)

        # Detection time
    time_det_yr = 8 

"""
