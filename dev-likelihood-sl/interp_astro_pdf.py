# -*- coding: utf-8 -*-

__author__ = "Mauricio Bustamante"
__email__ = "mbustamante@nbi.ku.dk"


"""
interp_astro_pdf.py:
    Return the interpolated astrophysical PDFs for IceCube HESE events,
    as a function of gamma, g, and M, based on pre-computed look-up
    tables.

Created: 2019/11/10 11:04
Last modified: 2019/11/10 11:04
"""


import os
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.interpolate import interp1d
# from pylab import *
# from matplotlib import *
# import matplotlib as mpl
# import matplotlib.ticker as ticker

from global_tools import *


def Initialize_Interpolator_Astrophysical_PDF(verbose=0):

    # global interp_astro_pdf_sh
    # global interp_astro_pdf_tr

    if (verbose > 0): print("Initializing astrophysical PDFs for IceCube events...")

    PATH_IN = os.getcwd()+'/in/ic_astro_pdf/'
    filename_coupling = PATH_IN+'coupling_values.dat'
    filename_mass = PATH_IN+'mass_values.dat'
    filename_gamma = PATH_IN+'gamma_values.dat'
    filename_astro_pdf_sh_base = PATH_IN+'astro_pdf_sh_'
    num_events_astro_pdf_sh = 58
    filename_astro_pdf_tr_base = PATH_IN+'astro_pdf_tr_'
    num_events_astro_pdf_tr = 22

    # Read in the mediator coupling values
    log10_coupling = list(Read_Data_File(filename_coupling)[0])
    log10_coupling_npts = len(log10_coupling)

    # Read in the mediator mass values
    log10_mass = list(Read_Data_File(filename_mass)[0]) # [GeV]
    log10_mass_npts = len(log10_mass)

    # Read in the gamma values
    gamma = list(Read_Data_File(filename_gamma)[0] )# [GeV]
    gamma_npts = len(gamma)

    # Grid points
    points = [gamma, log10_coupling, log10_mass]

    # In the file astro_pdf_sh_*.dat, line i contains the flux computed
    # with the i-th choice of values of (M,g,gamma)

    # Showers:
    # The list interp_astro_pdf_sh contains the interpolation
    # functions of the astrophysical PDF of the i-th shower as a
    # function of gamma, g, M.
    interp_astro_pdf_sh = []
    for i in range(num_events_astro_pdf_sh):
        filename = filename_astro_pdf_sh_base+str(i)+'.dat'
        astro_pdf_data = Read_Data_File(filename)[0]
        interp_astro_pdf_sh.append(\
            RegularGridInterpolator(points,
                                    astro_pdf_data.reshape( \
                                        (   log10_coupling_npts,
                                            log10_mass_npts,
                                            gamma_npts)),
                                    method='nearest',
                                    bounds_error=False,
                                    fill_value=None))

    # Tracks:
    # The list interp_astro_pdf_tr contains the interpolation
    # functions of the astrophysical PDF of the i-th track as a
    # function of gamma, g, M.
    interp_astro_pdf_tr = []
    for i in range(num_events_astro_pdf_tr):
        filename = filename_astro_pdf_tr_base+str(i)+'.dat'
        astro_pdf_data = Read_Data_File(filename)[0]
        interp_astro_pdf_tr.append(\
            RegularGridInterpolator(points,
                                    astro_pdf_data.reshape( \
                                        (   log10_coupling_npts,
                                            log10_mass_npts,
                                            gamma_npts)),
                                    method='nearest',
                                    bounds_error=False,
                                    fill_value=None))

    return interp_astro_pdf_sh, interp_astro_pdf_tr
