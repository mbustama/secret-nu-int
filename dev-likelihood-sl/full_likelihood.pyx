import os
import sys

cimport numpy as np
import numpy as np

from libc.math cimport log
from libc.math cimport exp
from global_tools import Write_Data_File


# ID_sh, lst_energy_sh, uncertainty_minus_sh, uncertainty_plus_sh, time_sh, \
#     declination_sh, RA_sh, Med_sh = \
#     Read_Data_File(os.getcwd()+'/ic_data/data_shower.txt')

# ID_tr, lst_energy_tr, uncertainty_minus_tr, uncertainty_plus_tr, time_tr, \
#     declination_tr, RA_tr, Med_tr = \
#     Read_Data_File(os.getcwd()+'/ic_data/data_track.txt')

# ll_den = 273.673 # Log(80!)


cdef double Partial_Likelihood_Showers(int event_index, double gamma,
    double log10_g, double log10_M, double N_a, double N_conv, double N_pr,
    double N_mu, list interp_astro_pdf_sh, list pdf_atm_conv_sh,
    list pdf_atm_pr_sh, int verbose=0):

    # Call Initialize_Interpolator_Astrophysical_PDF and
    # Initialize_Atmospheric_PDFs (once) before calling this function

    cdef double pdf_astro_sh
    cdef double likelihood

    if (verbose > 1): print("Sh #"+str(event_index))

    pdf_astro_sh = interp_astro_pdf_sh[event_index]((gamma, log10_g, log10_M))

    likelihood = N_a*pdf_astro_sh + N_conv*pdf_atm_conv_sh[event_index] + \
                    N_pr*pdf_atm_pr_sh[event_index] #+ \
                    # N_mu*pdf_atm_muon_sh[event_index]

    return likelihood


cdef double Partial_Likelihood_Tracks(int event_index, double gamma,
    double log10_g, double log10_M, double N_a, double N_conv, double N_pr,
    double N_mu, list interp_astro_pdf_tr, list pdf_atm_conv_tr,
    list pdf_atm_pr_tr, list pdf_atm_muon_tr, int verbose=0):

    # Call Initialize_Interpolator_Astrophysical_PDF and
    # Initialize_Atmospheric_PDFs (once) before calling this function

    cdef double pdf_astro_tr
    cdef double likelihood

    if (verbose > 1): print("Tr #"+str(event_index))

    pdf_astro_tr = interp_astro_pdf_tr[event_index]((gamma, log10_g, log10_M))

    likelihood = N_a*pdf_astro_tr + N_conv*pdf_atm_conv_tr[event_index] + \
                    N_pr*pdf_atm_pr_tr[event_index] + \
                    N_mu*pdf_atm_muon_tr[event_index]

    return likelihood


def Log_Likelihood(double gamma, double log10_g, double log10_M,
    double N_a, double N_conv, double N_pr, double N_mu,
    list interp_astro_pdf_sh, list pdf_atm_conv_sh, list pdf_atm_pr_sh,
    list interp_astro_pdf_tr, list pdf_atm_conv_tr, list pdf_atm_pr_tr,
    list pdf_atm_muon_tr, int num_ic_sh=58, int num_ic_tr=22, int verbose=0):

    # Call Initialize_Interpolator_Astrophysical_PDF and
    # Initialize_Atmospheric_PDFs (once) before calling this function

    cdef double fl_sh
    cdef double fl_tr
    cdef double log_likelihood
    cdef int i

    # Showers
    fl_sh = sum([Partial_Likelihood_Showers(i, gamma, log10_g, log10_M, N_a,
                N_conv, N_pr, N_mu, interp_astro_pdf_sh, pdf_atm_conv_sh,
                pdf_atm_pr_sh, verbose=verbose) for i in range(num_ic_sh)])

    # Tracks
    fl_tr = sum([Partial_Likelihood_Tracks(i, gamma, log10_g, log10_M, N_a,
                N_conv, N_pr, N_mu, interp_astro_pdf_tr, pdf_atm_conv_tr,
                pdf_atm_pr_tr, pdf_atm_muon_tr, verbose=verbose) \
                for i in range(num_ic_tr)])


    log_likelihood = log(exp(-N_a-N_conv-N_pr-N_mu)*fl_sh*fl_tr)

    return log_likelihood



