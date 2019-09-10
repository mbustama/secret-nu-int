from __future__ import division
import numpy as np
import os

from flux_atm import *
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


## >>> Initialize atm. fluxes, passing fractions

lst_costhz_sel, lst_energy_sel, flux_mum_conv, flux_mup_conv, \
    flux_mum_pr, flux_mup_pr, flux_nue_conv, flux_nuebar_conv, \
    flux_nue_pr, flux_nuebar_pr, flux_numu_conv, flux_numubar_conv, \
    flux_numu_pr, flux_numubar_pr, flux_nutau_pr, flux_nutaubar_pr = \
    Initialize_Atm_Fluxes(flux_set='avg', flag_return_total=False,
    flag_return_plus_minus=False, kx=1, ky=1, s=0, verbose=0)

# Passing fractions
# Call the interpolating functions as pf_spl(energy, costhz)
lst_energy_pf, lst_costhz_pf, pf_nue_conv, pf_nuebar_conv, pf_numu_conv, \
    pf_numubar_conv, pf_nue_pr, pf_nuebar_pr, pf_numu_pr, pf_numubar_pr = \
    Initialize_Passing_Fractions(kx=1, ky=1, s=0, verbose=0)

mix_params_data_set ='nufit_4_0_with_sk'
mix_params_mass_ordering ='no'

# lst_mix_params is passed to NuSQuIDS and contains the mixing parameters
# [s12sq_bf, s23sq_bf, s13sq_bf, deltaCP_bf, Delta_m2_21_bf, Delta_m2_31_bf, Delta_m2_32_bf]
lst_mix_params = Mixing_Parameters(mix_params_data_set, mix_params_mass_ordering)[0]


def Prob_dist_atm_conv_pr_den(nu_energy_min, nu_energy_max, nu_energy_num_nodes, costhz_npts, log10_energy_dep_int_min, log10_energy_dep_int_max, log10_energy_dep_min,
    log10_energy_dep_max, log10_energy_dep_npts, time_det_yr, volume_total, energy_nu_max, epsabs, epsrel, verbose, flag_use_atm_fluxes_conv, flag_use_atm_fluxes_pr, flag_apply_self_veto):

    log10_energy_dep_int_step = log10_energy_dep_int_max - log10_energy_dep_int_min


    # Array of cos(theta_z) to compute
    lst_costhz = np.linspace(-1.0, 1.0, costhz_npts)

        # [Adim], [GeV], [GeV cm^{-2} s^{-1} sr^{-1}], ...


    lst_flux_earth = []
    lst_flux_det = []

    for costhz in lst_costhz:

        # Build interpolating functions

        if (flag_use_atm_fluxes_conv == True and \
            flag_use_atm_fluxes_pr == False):

            # Use the conventional flux only
            # [GeV cm^{-2} s^{-1} sr^{-1}]

            nue_flux = lambda energy_nu: \
                flux_nue_conv(costhz, energy_nu)[0][0]
            nuebar_flux = lambda energy_nu: \
                flux_nuebar_conv(costhz, energy_nu)[0][0]
            numu_flux = lambda energy_nu: \
                flux_numu_conv(costhz, energy_nu)[0][0]
            numubar_flux = lambda energy_nu: \
                flux_numubar_conv(costhz, energy_nu)[0][0]
            nutau_flux = lambda energy_nu: 0.0
            nutaubar_flux = lambda energy_nu: 0.0

        elif (flag_use_atm_fluxes_conv == False and \
            flag_use_atm_fluxes_pr == True):

            # Use the prompt flux only
            # [GeV cm^{-2} s^{-1} sr^{-1}]

            nue_flux = lambda energy_nu: \
                flux_nue_pr(costhz, energy_nu)[0][0]
            nuebar_flux = lambda energy_nu: \
                flux_nuebar_pr(costhz, energy_nu)[0][0]
            numu_flux = lambda energy_nu: \
                flux_numu_pr(costhz, energy_nu)[0][0]
            numubar_flux = lambda energy_nu: \
                flux_numubar_pr(costhz, energy_nu)[0][0]
            nutau_flux = lambda energy_nu: \
                flux_nutau_pr(costhz, energy_nu)[0][0]
            nutaubar_flux = lambda energy_nu: \
                flux_nutaubar_pr(costhz, energy_nu)[0][0]

        elif (flag_use_atm_fluxes_conv == True and \
            flag_use_atm_fluxes_pr == True):

            print("flag_use_atm_fluxes_conv and flag_use_atm_fluxes_pr cannot "+ \
                "both be True")
            quit()

        # Nu_Fluxes_Initial is fed to NuSQuIDS
        # [GeV^{-1} cm^{-2} s^{-1} sr^{-1}]
        Nu_Fluxes_Initial = lambda lst_energy, **kwargs: \
            Nu_Fluxes_Initial_Individual_Species_Format_NuSQuIDS(lst_energy,
                nue_flux, nuebar_flux, numu_flux, numubar_flux, nutau_flux,
                nutaubar_flux, flag_divide_by_energy_sq=True, **kwargs)

        # Compute the fluxes at the detector for the current value of costhz
        # [GeV^{-1} cm^{-2} s^{-1} sr^{-1}]
        # lst_flux_det and lst_flux_earth are:
        #   [[lst_nu_energy_out, flux_nu_e, flux_nu_e_bar, flux_nu_m, flux_nu_m_bar,
        #       flux_nu_t, flux_nu_t_bar]_costhz0,
        #    [lst_nu_energy_out, flux_nu_e, flux_nu_e_bar, flux_nu_m, flux_nu_m_bar,
        #       flux_nu_t, flux_nu_t_bar]_costhz1,
        #    ...
        #    [lst_nu_energy_out, flux_nu_e, flux_nu_e_bar, flux_nu_m, flux_nu_m_bar,
        #       flux_nu_t, flux_nu_t_bar]_costhzN,
        flux_det, flux_earth = \
            Nu_Fluxes_At_Detector(nu_energy_min, nu_energy_max,
                nu_energy_num_nodes, [costhz], Nu_Fluxes_Initial,
                error_rel=1.e-7, error_abs=1.e-7, h_max=500.,
                verbose_level=verbose, lst_mix_params=lst_mix_params)

        # Concatenate the fluxes for the current costhz into the global list
        lst_flux_det.append(flux_det[0])
        lst_flux_earth.append(np.array(flux_earth[0]))


    ###############################################################################
    # Apply the atmospheric self-veto after propagating the fluxes to the detector
    ###############################################################################

    if (((flag_use_atm_fluxes_conv == True or flag_use_atm_fluxes_pr == True)) and \
    flag_apply_self_veto == True):

        tmp = []

        # Apply the veto only to lst_flux_det
        for i, costhz in enumerate(lst_costhz):

            lst_nu_energy_out = lst_flux_det[i][0]
            flux_nu_e = lst_flux_det[i][1]
            flux_nu_e_bar = lst_flux_det[i][2]
            flux_nu_m = lst_flux_det[i][3]
            flux_nu_m_bar = lst_flux_det[i][4]
            flux_nu_t = lst_flux_det[i][5]
            flux_nu_t_bar = lst_flux_det[i][6]

            flux_nu_e_tmp = []
            flux_nu_e_bar_tmp = []
            flux_nu_m_tmp = []
            flux_nu_m_bar_tmp = []
            flux_nu_t_tmp = []
            flux_nu_t_bar_tmp = []

            if (flag_use_atm_fluxes_conv == True):

                flux_nu_e_tmp = \
                    [pf_nue_conv(lst_nu_energy_out[j], costhz)[0][0] * \
                    flux_nu_e[j] for j in range(len(lst_nu_energy_out))]
                flux_nu_e_bar_tmp = \
                    [pf_nuebar_conv(lst_nu_energy_out[j], costhz)[0][0] * \
                    flux_nu_e_bar[j] for j in range(len(lst_nu_energy_out))]
                flux_nu_m_tmp = \
                    [pf_numu_conv(lst_nu_energy_out[j], costhz)[0][0] * \
                    flux_nu_m[j] for j in range(len(lst_nu_energy_out))]
                flux_nu_m_bar_tmp = \
                    [pf_numubar_conv(lst_nu_energy_out[j], costhz)[0][0] * \
                    flux_nu_m_bar[j] for j in range(len(lst_nu_energy_out))]
                flux_nu_t_tmp = [0.0 for j in range(len(lst_nu_energy_out))]
                flux_nu_t_bar_tmp = [0.0 for j in range(len(lst_nu_energy_out))]

            elif (flag_use_atm_fluxes_pr == True):

                flux_nu_e_tmp = \
                    [pf_nue_pr(lst_nu_energy_out[j], costhz)[0][0] * \
                    flux_nu_e[j] for j in range(len(lst_nu_energy_out))]
                flux_nu_e_bar_tmp = \
                    [pf_nuebar_pr(lst_nu_energy_out[j], costhz)[0][0] * \
                    flux_nu_e_bar[j] for j in range(len(lst_nu_energy_out))]
                flux_nu_m_tmp = \
                    [pf_numu_pr(lst_nu_energy_out[j], costhz)[0][0] * \
                    flux_nu_m[j] for j in range(len(lst_nu_energy_out))]
                flux_nu_m_bar_tmp = \
                    [pf_numubar_pr(lst_nu_energy_out[j], costhz)[0][0] * \
                    flux_nu_m_bar[j] for j in range(len(lst_nu_energy_out))]
                # We do not self-veto on prompt nu_tau (IceCube does not either)
                flux_nu_t_tmp = [x for x in flux_nu_t]
                flux_nu_t_bar_tmp = [x for x in flux_nu_t_bar]

            tmp.append([lst_nu_energy_out, flux_nu_e_tmp, flux_nu_e_bar_tmp,
                flux_nu_m_tmp, flux_nu_m_bar_tmp, flux_nu_t_tmp,
                flux_nu_t_bar_tmp])

        lst_flux_det = [x for x in tmp]


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

    Denominator = np.sum(Event_spec[0][2] + Event_spec[1][2] + Event_spec[2][2] + Event_spec[3][2] + Event_spec[4][2] + Event_spec[5][2])

    print("Denominator_atm=", Denominator)

    return Denominator


def Prob_dist_atm_conv_pr_num(nu_energy_min, nu_energy_max, nu_energy_num_nodes, costhz_val, energy_dep, time_det_yr, volume_total, energy_nu_max, epsabs, epsrel, verbose,
    flag_compute_shower_rate, flag_compute_track_rate, flag_use_atm_fluxes_conv, flag_use_atm_fluxes_pr, flag_apply_self_veto):

    lst_costhz = np.array([costhz_val])

    lst_flux_earth = []
    lst_flux_det = []

    for costhz in lst_costhz:

        # Build interpolating functions

        if (flag_use_atm_fluxes_conv == True and \
            flag_use_atm_fluxes_pr == False):

            # Use the conventional flux only
            # [GeV cm^{-2} s^{-1} sr^{-1}]

            nue_flux = lambda energy_nu: \
                flux_nue_conv(costhz, energy_nu)[0][0]
            nuebar_flux = lambda energy_nu: \
                flux_nuebar_conv(costhz, energy_nu)[0][0]
            numu_flux = lambda energy_nu: \
                flux_numu_conv(costhz, energy_nu)[0][0]
            numubar_flux = lambda energy_nu: \
                flux_numubar_conv(costhz, energy_nu)[0][0]
            nutau_flux = lambda energy_nu: 0.0
            nutaubar_flux = lambda energy_nu: 0.0

        elif (flag_use_atm_fluxes_conv == False and \
            flag_use_atm_fluxes_pr == True):

            # Use the prompt flux only
            # [GeV cm^{-2} s^{-1} sr^{-1}]

            nue_flux = lambda energy_nu: \
                flux_nue_pr(costhz, energy_nu)[0][0]
            nuebar_flux = lambda energy_nu: \
                flux_nuebar_pr(costhz, energy_nu)[0][0]
            numu_flux = lambda energy_nu: \
                flux_numu_pr(costhz, energy_nu)[0][0]
            numubar_flux = lambda energy_nu: \
                flux_numubar_pr(costhz, energy_nu)[0][0]
            nutau_flux = lambda energy_nu: \
                flux_nutau_pr(costhz, energy_nu)[0][0]
            nutaubar_flux = lambda energy_nu: \
                flux_nutaubar_pr(costhz, energy_nu)[0][0]

        elif (flag_use_atm_fluxes_conv == True and \
            flag_use_atm_fluxes_pr == True):

            print("flag_use_atm_fluxes_conv and flag_use_atm_fluxes_pr cannot "+ \
                "both be True")
            quit()

        # Nu_Fluxes_Initial is fed to NuSQuIDS
        # [GeV^{-1} cm^{-2} s^{-1} sr^{-1}]
        Nu_Fluxes_Initial = lambda lst_energy, **kwargs: \
            Nu_Fluxes_Initial_Individual_Species_Format_NuSQuIDS(lst_energy,
                nue_flux, nuebar_flux, numu_flux, numubar_flux, nutau_flux,
                nutaubar_flux, flag_divide_by_energy_sq=True, **kwargs)

        # Compute the fluxes at the detector for the current value of costhz
        # [GeV^{-1} cm^{-2} s^{-1} sr^{-1}]
        # lst_flux_det and lst_flux_earth are:
        #   [[lst_nu_energy_out, flux_nu_e, flux_nu_e_bar, flux_nu_m, flux_nu_m_bar,
        #       flux_nu_t, flux_nu_t_bar]_costhz0,
        #    [lst_nu_energy_out, flux_nu_e, flux_nu_e_bar, flux_nu_m, flux_nu_m_bar,
        #       flux_nu_t, flux_nu_t_bar]_costhz1,
        #    ...
        #    [lst_nu_energy_out, flux_nu_e, flux_nu_e_bar, flux_nu_m, flux_nu_m_bar,
        #       flux_nu_t, flux_nu_t_bar]_costhzN,
        flux_det, flux_earth = \
            Nu_Fluxes_At_Detector(nu_energy_min, nu_energy_max,
                nu_energy_num_nodes, [costhz], Nu_Fluxes_Initial,
                error_rel=1.e-7, error_abs=1.e-7, h_max=500.,
                verbose_level=verbose, lst_mix_params=lst_mix_params)

        # Concatenate the fluxes for the current costhz into the global list
        lst_flux_det.append(flux_det[0])
        lst_flux_earth.append(np.array(flux_earth[0]))


    ###############################################################################
    # Apply the atmospheric self-veto after propagating the fluxes to the detector
    ###############################################################################

    if (((flag_use_atm_fluxes_conv == True or flag_use_atm_fluxes_pr == True)) and \
    flag_apply_self_veto == True):

        tmp = []

        # Apply the veto only to lst_flux_det
        for i, costhz in enumerate(lst_costhz):

            lst_nu_energy_out = lst_flux_det[i][0]
            flux_nu_e = lst_flux_det[i][1]
            flux_nu_e_bar = lst_flux_det[i][2]
            flux_nu_m = lst_flux_det[i][3]
            flux_nu_m_bar = lst_flux_det[i][4]
            flux_nu_t = lst_flux_det[i][5]
            flux_nu_t_bar = lst_flux_det[i][6]

            flux_nu_e_tmp = []
            flux_nu_e_bar_tmp = []
            flux_nu_m_tmp = []
            flux_nu_m_bar_tmp = []
            flux_nu_t_tmp = []
            flux_nu_t_bar_tmp = []

            if (flag_use_atm_fluxes_conv == True):

                flux_nu_e_tmp = \
                    [pf_nue_conv(lst_nu_energy_out[j], costhz)[0][0] * \
                    flux_nu_e[j] for j in range(len(lst_nu_energy_out))]
                flux_nu_e_bar_tmp = \
                    [pf_nuebar_conv(lst_nu_energy_out[j], costhz)[0][0] * \
                    flux_nu_e_bar[j] for j in range(len(lst_nu_energy_out))]
                flux_nu_m_tmp = \
                    [pf_numu_conv(lst_nu_energy_out[j], costhz)[0][0] * \
                    flux_nu_m[j] for j in range(len(lst_nu_energy_out))]
                flux_nu_m_bar_tmp = \
                    [pf_numubar_conv(lst_nu_energy_out[j], costhz)[0][0] * \
                    flux_nu_m_bar[j] for j in range(len(lst_nu_energy_out))]
                flux_nu_t_tmp = [0.0 for j in range(len(lst_nu_energy_out))]
                flux_nu_t_bar_tmp = [0.0 for j in range(len(lst_nu_energy_out))]

            elif (flag_use_atm_fluxes_pr == True):

                flux_nu_e_tmp = \
                    [pf_nue_pr(lst_nu_energy_out[j], costhz)[0][0] * \
                    flux_nu_e[j] for j in range(len(lst_nu_energy_out))]
                flux_nu_e_bar_tmp = \
                    [pf_nuebar_pr(lst_nu_energy_out[j], costhz)[0][0] * \
                    flux_nu_e_bar[j] for j in range(len(lst_nu_energy_out))]
                flux_nu_m_tmp = \
                    [pf_numu_pr(lst_nu_energy_out[j], costhz)[0][0] * \
                    flux_nu_m[j] for j in range(len(lst_nu_energy_out))]
                flux_nu_m_bar_tmp = \
                    [pf_numubar_pr(lst_nu_energy_out[j], costhz)[0][0] * \
                    flux_nu_m_bar[j] for j in range(len(lst_nu_energy_out))]
                # We do not self-veto on prompt nu_tau (IceCube does not either)
                flux_nu_t_tmp = [x for x in flux_nu_t]
                flux_nu_t_bar_tmp = [x for x in flux_nu_t_bar]

            tmp.append([lst_nu_energy_out, flux_nu_e_tmp, flux_nu_e_bar_tmp,
                flux_nu_m_tmp, flux_nu_m_bar_tmp, flux_nu_t_tmp,
                flux_nu_t_bar_tmp])

        lst_flux_det = [x for x in tmp]



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



def Prob_dist_atm_conv_pr(nu_energy_min, nu_energy_max, nu_energy_num_nodes, costhz_val, costhz_npts, energy_dep, log10_energy_dep_int_min, log10_energy_dep_int_max, log10_energy_dep_min, log10_energy_dep_max, log10_energy_dep_npts,
        time_det_yr, volume_total, energy_nu_max, epsabs, epsrel, verbose, flag_use_atm_fluxes_conv, flag_use_atm_fluxes_pr, flag_apply_self_veto, flag_compute_shower_rate, flag_compute_track_rate):

    num = \
        Prob_dist_atm_conv_pr_num(nu_energy_min, nu_energy_max, nu_energy_num_nodes, costhz_val, energy_dep, time_det_yr, volume_total, energy_nu_max, epsabs, epsrel, verbose,
            flag_compute_shower_rate = flag_compute_shower_rate, flag_compute_track_rate = flag_compute_track_rate,
            flag_use_atm_fluxes_conv = flag_use_atm_fluxes_conv, flag_use_atm_fluxes_pr = flag_use_atm_fluxes_pr, flag_apply_self_veto = flag_apply_self_veto)

    den = \
        Prob_dist_atm_conv_pr_den(nu_energy_min, nu_energy_max, nu_energy_num_nodes, costhz_npts, log10_energy_dep_int_min, log10_energy_dep_int_max,
            log10_energy_dep_min, log10_energy_dep_max, log10_energy_dep_npts, time_det_yr, volume_total, energy_nu_max, epsabs, epsrel, verbose,
            flag_use_atm_fluxes_conv = flag_use_atm_fluxes_conv, flag_use_atm_fluxes_pr = flag_use_atm_fluxes_pr, flag_apply_self_veto = flag_apply_self_veto)

    prob = num/den

    print("Prob_atm=", prob)

    return prob


def Prob_dist_atm_muon_den(log10_energy_dep_int_min, log10_energy_dep_int_max, log10_energy_dep_min, log10_energy_dep_max, log10_energy_dep_npts, epsabs, epsrel, verbose):

    # Array of values of E_dep used to build the interpolation function of
    # dN/dE_dep that is later integrated in E_dep
    lst_log10_energy_dep = np.linspace(log10_energy_dep_min,
        log10_energy_dep_max, log10_energy_dep_npts)
    lst_energy_dep = [10.**x for x in lst_log10_energy_dep] # [GeV]

    # Arrays defining the bins of E_dep in which to integrate
    log10_energy_dep_int_step = log10_energy_dep_int_max - log10_energy_dep_int_min
    log10_energy_dep_int_npts = \
        int((log10_energy_dep_int_max-log10_energy_dep_int_min) / \
            log10_energy_dep_int_step)
    lst_log10_energy_dep_int_min = \
        np.linspace(log10_energy_dep_int_min,
            log10_energy_dep_int_max-log10_energy_dep_int_step,
            log10_energy_dep_int_npts)
    lst_log10_energy_dep_int_max = \
        np.linspace(log10_energy_dep_int_min+log10_energy_dep_int_step,
        log10_energy_dep_int_max, log10_energy_dep_int_npts)
    lst_energy_dep_int_min = [10.**x for x in lst_log10_energy_dep_int_min]
    lst_energy_dep_int_max = [10.**x for x in lst_log10_energy_dep_int_max]

    lst_int_rate = []

    gamma_mu = np.log10(21)/np.log10(60/28) + 1

    lst_diff_rate = np.power(lst_energy_dep, -gamma_mu)

    # Build the interpolating function dN/dE_dep for each flavor
    if (verbose > 0):
        print("  Building interpolating functions of dN/dE_dep... ", \
                end='')
    interp_diff_rate = interp1d(lst_energy_dep,
            lst_diff_rate, kind='linear', bounds_error=False,
            fill_value='extrapolate')
    if (verbose > 0): print("Done")

    # Integrate dN/dE_dep in each bin of E_dep, for each flavor
    if (verbose > 0):
        print("  Integrating dN/dE_dep in each E_dep bin... ", end='')
    res = [Energy_Integrated_Event_Rate(interp_diff_rate,
        lst_energy_dep_int_min[j], lst_energy_dep_int_max[j],
        flag_input_is_interpolating_function=True,
        integration_method='quad',
        miniter=1, maxiter=500,
        epsabs=epsabs, epsrel=epsrel) \
        for j in range(len(lst_energy_dep_int_min))]
    lst_int_rate.append(res)

    if (verbose > 0): print("Done")

    Denominator = lst_int_rate

    return Denominator



def Prob_dist_atm_muon(energy_dep, log10_energy_dep_int_min, log10_energy_dep_int_max, log10_energy_dep_min, log10_energy_dep_max, log10_energy_dep_npts, epsabs, epsrel, verbose):

    gamma_mu = np.log10(21)/np.log10(60/28) + 1

    num = np.power(energy_dep, -gamma_mu)

    den = Prob_dist_atm_muon_den(log10_energy_dep_int_min, log10_energy_dep_int_max, log10_energy_dep_min, log10_energy_dep_max, log10_energy_dep_npts, epsabs, epsrel, verbose)

    prob = num/den

    print(prob)

    return prob


