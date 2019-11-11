from interp_atm_pdf import Initialize_Atmospheric_PDFs
from interp_astro_pdf import Initialize_Interpolator_Astrophysical_PDF
from full_likelihood import Log10_Likelihood

import time
import random

verbose = 0

pdf_atm_conv_sh, pdf_atm_pr_sh, pdf_atm_pr_sh, pdf_atm_muon_sh, \
            pdf_atm_conv_tr, pdf_atm_pr_tr, pdf_atm_muon_tr = \
            Initialize_Atmospheric_PDFs(verbose=verbose)

interp_astro_pdf_sh, interp_astro_pdf_tr = \
    Initialize_Interpolator_Astrophysical_PDF(verbose=verbose)

random.seed(11)
gamma = 1.8
log10_g = -3.0
log10_M = -3.0 # [GeV]
N_a = 20
N_conv = 20
N_pr = 20
N_mu = 20


start = time.time()
for i in range(1000):
    ll = Log10_Likelihood(gamma+1.2*random.random(), log10_g+2.0*random.random(),
            log10_M+2.0*random.random(), N_a, N_conv, N_pr, N_mu,
            interp_astro_pdf_sh, pdf_atm_conv_sh, pdf_atm_pr_sh,
            interp_astro_pdf_tr, pdf_atm_conv_tr, pdf_atm_pr_tr, pdf_atm_muon_tr,
            num_ic_sh=58, num_ic_tr=22, verbose=verbose)
# print(ll)
stop = time.time()
print((stop-start)/1000)

