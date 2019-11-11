# -*- coding: utf-8 -*-

__author__ = "Mauricio Bustamante"
__email__ = "mbustamante@nbi.ku.dk"


"""
interp_atm_pdf.py:
    Return the interpolated astrophysical PDFs for IceCube HESE events,
    as a function of gamma, g, and M, based on pre-computed look-up
    tables.

Created: 2019/11/10 11:04
Last modified: 2019/11/10 11:04
"""


import os
import numpy as np


from global_tools import *


def Initialize_Atmospheric_PDFs(verbose=0):

    # global pdf_atm_conv_sh
    # global pdf_atm_pr_sh
    # global pdf_atm_muon_sh
    # global pdf_atm_conv_tr
    # global pdf_atm_pr_tr
    # global pdf_atm_muon_tr

    if (verbose > 0): print("Initializing atmospheric PDFs for IceCube events...")

    ID_sh, log10_pdf_atm_conv_sh, log10_pdf_atm_pr_sh, log10_pdf_atm_muon_sh = \
        Read_Data_File(os.getcwd()+'/in/ic_atm_pdf/atm_pdf_ic_sh.dat')
    pdf_atm_conv_sh = [10.**x for x in log10_pdf_atm_conv_sh]
    pdf_atm_pr_sh = [10.**x for x in log10_pdf_atm_pr_sh]
    pdf_atm_muon_sh = [10.**x for x in log10_pdf_atm_muon_sh]

    ID_tr, log10_pdf_atm_conv_tr, log10_pdf_atm_pr_tr, log10_pdf_atm_muon_tr = \
        Read_Data_File(os.getcwd()+'/in/ic_atm_pdf/atm_pdf_ic_tr.dat')
    pdf_atm_conv_tr = [10.**x for x in log10_pdf_atm_conv_tr]
    pdf_atm_pr_tr = [10.**x for x in log10_pdf_atm_pr_tr]
    pdf_atm_muon_tr = [10.**x for x in log10_pdf_atm_muon_tr]

    return pdf_atm_conv_sh, pdf_atm_pr_sh, pdf_atm_pr_sh, pdf_atm_muon_sh, \
            pdf_atm_conv_tr, pdf_atm_pr_tr, pdf_atm_muon_tr


