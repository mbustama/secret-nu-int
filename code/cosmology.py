#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "Mauricio Bustamante"
__email__ = "mbustamante@nbi.ku.dk"


"""
cosmology.py:
    Contains routines to compute various cosmological quantities

Created: 2018/03/14 17:33
Last modified: 2018/10/02 14:32
"""

from numpy import *
from pylab import *
import numpy as np
from matplotlib import *
import matplotlib as mpl
from scipy.integrate import quad
from scipy import interpolate
import os
import sys

from timeit import default_timer as timer

from global_defs import *


def SFR(z):
    """
    Returns the adimensional star-formation rate (SFR), normalized so that
    SFR(0)=1.

    Ref.: H. Yuksel, M. Kistler, J. Beacom, and A. Hopkins, ApJ 683, L5 (2008)
    [arXiv:0804.4008]

    Parameters
    ----------

    z : flt
        [Adim] Redshift

    Returns
    -------

    SFR : flt
        [Adim] Star-formation rate at the requested redshift
    """

    if (z < 1.0):
        return pow(1.0+z, 3.4)
    elif ((z >= 1.0) and (z <4.0)):
        return 13.*pow(1.0+z, -0.3)
    else:
        return 2241.*pow(1.0+z, -3.5)

    return


def Adimensional_Hubble_Parameter(x):
    """
    Returns the adimensional Hubble parameters evaluated at redshift x=1+z; the
    adimensional matter densities are imported from global_defs.py

    Ref.: C. Giunti & C.W. Kim, Fundamentals of Neutrino Physics and
    Astrophysics (2007), Eqs. (16.90)-(16.103)


    Parameters
    ----------

    z : flt
        [Adim] Redshift

    Returns
    -------

    Adimensional_Hubble_Parameter : flt
        [Adim] Value of the adimensional Hubble parameter h(z)
	"""

    # print(np.sqrt( OmegaM/x + OmegaL*x*x ) / x)
    if (OmegaK == 0.0):
	    return np.sqrt( OmegaM/x + OmegaL*x*x ) / x # Slightly faster
    else:
        return np.sqrt( OmegaM/x + OmegaK + OmegaL*x*x ) / x


def DzDt(z):
    """
    Returns the derivate dz/dt.

    Parameters
    ----------

    z : flt
        [Adim] Redshift

    Returns
    -------

    DzDt : flt
        [s^{-1}] Value of dz/dt
    """

    return H0_inv_s*Adimensional_Hubble_Parameter(1.0+z)


def Hubble_Horizon(z, flag_use_precomputed=True):
    """
    Returns the Hubble, or particle, horizon; the adimensional matter
    densities are imported from global_defs.py

    Ref.: C. Giunti & C.W. Kim, Fundamentals of Neutrino Physics and
    Astrophysics (2007), Eq. (16.103)

    Parameters
    ----------

    z : flt
        [Adim] Redshift

    Returns
    -------

    Hubble_Horizon : flt
        [Gpc] Comoving Hubble, or particle, horizon d_H(z)
	"""

    if (flag_use_precomputed == True):
        if (z == 0.00): return 14.458709329436418
        if (z == 0.10): return 12.751758572580052
        if (z == 0.20): return 11.34706554083828
        if (z == 0.25): return 10.735502010706949
        if (z == 0.30): return 10.175152250279107
        if (z == 0.40): return 9.186032316784678
        if (z == 0.50): return 8.342785553985482
        if (z == 0.60): return 7.617543505122876
        if (z == 0.70): return 6.988885323772868
        if (z == 0.75): return 6.705402506304734
        if (z == 0.80): return 6.440091977697317
        if (z == 0.90): return 5.957941827311957
        if (z == 1.00): return 5.531858882244094
        if (z == 1.10): return 5.1532975514994455
        if (z == 1.20): return 4.815290078947334
        if (z == 1.25): return 4.659663051639454
        if (z == 1.30): return 4.512108411088476
        if (z == 1.40): return 4.239008123330833
        if (z == 1.50): return 3.9920321792397555
        if (z == 1.60): return 3.7678589531385533
        if (z == 1.70): return 3.5636834153463157
        if (z == 1.75): return 3.4683359929568187
        if (z == 1.80): return 3.3771234431044093
        if (z == 1.90): return 3.206145359648288
        if (z == 2.00): return 3.0490043221137686
        if (z == 2.10): return 2.904196271758557
        if (z == 2.20): return 2.7704189564567354
        if (z == 2.25): return 2.707307102005293
        if (z == 2.30): return 2.646540122430766
        if (z == 2.40): return 2.5315714091632056
        if (z == 2.50): return 2.424646809687774
        if (z == 2.60): return 2.3250048071027654
        if (z == 2.70): return 2.2319734879405226
        if (z == 2.75): return 2.1877482531820305
        if (z == 2.80): return 2.144958078931741
        if (z == 2.90): return 2.0634304666248315
        if (z == 3.00): return 1.9869203472661787
        if (z == 3.10): return 1.9150077232525173
        if (z == 3.20): return 1.847316516758493
        if (z == 3.25): return 1.8149470709410604
        if (z == 3.30): return 1.7835091141541126
        if (z == 3.40): return 1.7232816890758504
        if (z == 3.50): return 1.6663601794234453
        if (z == 3.60): return 1.6124968155911543
        if (z == 3.70): return 1.5614671150414705
        if (z == 3.75): return 1.5369504559291804
        if (z == 3.80): return 1.5130672727685082
        if (z == 3.90): return 1.4671118889612507
        if (z == 4.00): return 1.423431984797989
        if (z == 4.10): return 1.3818732652044514
        if (z == 4.20): return 1.342294593919542
        if (z == 4.25): return 1.3232069358497804
        if (z == 4.30): return 1.3045666515999885
        if (z == 4.40): return 1.2685707521670475
        if (z == 4.50): return 1.234197796323375
        if (z == 4.60): return 1.2013473442810902
        if (z == 4.70): return 1.169926792351732
        if (z == 4.75): return 1.1547257509734647
        if (z == 4.80): return 1.1398506402433668
        if (z == 4.90): return 1.1110398377609558
        if (z == 5.00): return 1.0834212011715456
        if (z == 5.10): return 1.056926890823794
        if (z == 5.20): return 1.031493942740645
        if (z == 5.25): return 1.0191570320665366
        if (z == 5.30): return 1.0070638478668035
        if (z == 5.40): return 0.983582173476268
        if (z == 5.50): return 0.9609982219498835
        if (z == 5.60): return 0.9392647227398553
        if (z == 5.70): return 0.9183375538591511
        if (z == 5.75): return 0.9081633535486844
        if (z == 5.80): return 0.8981754896843493
        if (z == 5.90): return 0.8787399722499588
        if (z == 6.00): return 0.8599949035494744

    return Hubble_horizon_0 / (1.0+z) * \
        quad(lambda x: 1.0/Adimensional_Hubble_Parameter(x)/x/x, \
        	0.0, 1./(1.0+z), epsabs=1.e-2, limit=100, full_output=1)[0]


def Lookback_Distance(z):
    """
    Returns the lookback distance; the adimensional matter densities are
    imported from global_defs.py

    Ref.: C. Giunti & C.W. Kim, Fundamentals of Neutrino Physics and
    Astrophysics (2007), Eq. (16.95)

    Parameters
    ----------

    z : flt
        [Adim] Redshift

    Returns
    -------

    Lookback_Distance : flt
        [Gpc] Lookback distance d_lb(z)
	"""

    return Hubble_horizon_0 * \
        quad(lambda x: 1.0/Adimensional_Hubble_Parameter(x)/x, \
        	1./(1.0+z), 1.0, epsabs=1.e-2, limit=100, full_output=1)[0]


def Diff_Lookback_Distance(z):
    """
    Returns the differential lookback distance d(d_lb)/dz; the adimensional
    matter densities are imported from global_defs.py

    Ref.: C. Giunti & C.W. Kim, Fundamentals of Neutrino Physics and
    Astrophysics (2007), Eq. (16.95)

    Parameters
    ----------

    z : flt
        [Adim] Redshift

    Returns
    -------

    Diff_Lookback_Distance : flt
        [Gpc] Differential lookback distance d_lb(z)/dz
	"""

    return Hubble_horizon_0/(1.0+z)/Adimensional_Hubble_Parameter(z)


def Diff_Lookback_Distance_Interp(z):
    """
    Returns the *interpolated* differential lookback distance d(d_lb)/dz.
    Data to calculate the interpolation was generated with the routine
    Diff_Lookback_Distance.

    Parameters
    ----------

    z : flt
        [Adim] Redshift

    Returns
    -------

    Diff_Lookback_Distance_Interp : flt
        [Gpc] Differential lookback distance d_lb(z)/dz
	"""

    return interp_diff_distance([z])[0]


def Comoving_Distance(z):
    """
    Returns the comoving distance; the adimensional matter densities
    are imported from global_defs.py.  Assumes Omega_k = 0.

    Ref.: D. Hogg, astro-ph/9905116

    Parameters
    ----------

    z : flt
        [Adim] Redshift

    Returns
    -------

    Comoving_Distance : flt
        [Gpc] Comoving distance d_com(z)
    """

    # DL = (1+z)*DM = (1+z)*DC

    def h(zz):
        zp1 = 1.0+zz
        return sqrt(OmegaM*pow(zp1,3.0) + OmegaL)

    return Hubble_horizon_0 * \
            quad(lambda zz: 1.0/h(zz), 0.0, z, \
                epsabs=1.e-2, limit=100, full_output=1)[0]


def Luminosity_Distance(z):
    """
    Returns the luminosity distance; the adimensional matter densities
    are imported from global_defs.py.  Assumes Omega_k = 0.

    Ref.: D. Hogg, astro-ph/9905116

    Parameters
    ----------

    z : flt
        [Adim] Redshift

    Returns
    -------

    Luminosity_Distance : flt
        [Gpc] Luminosity distance d_lum(z)
    """

    # DL = (1+z)*DM = (1+z)*DC

    def h(zz):
        zp1 = 1.0+zz
        return sqrt(OmegaM*pow(zp1,3.0) + OmegaL)

    return Hubble_horizon_0 * (1.0+z) * \
            quad(lambda zz: 1.0/h(zz), 0.0, z, \
                epsabs=1.e-2, limit=100, full_output=1)[0]


def Differential_Comoving_Volume(z):
    """
    Returns the differential comoving volume dV/dz; the adimensional
    matter densities are imported from global_defs.py.  Assumes
    Omega_k = 0.

    Ref.: D. Hogg, astro-ph/9905116

    Parameters
    ----------

    z : flt
        [Adim] Redshift

    Returns
    -------

    Differential_Comoving_Volume : flt
        [Gpc^3] Differential comoving volume dV/dz
    """

    # DL = (1+z)*DM = (1+z)*DC

    def h(zz):
        zp1 = 1.0+zz
        return sqrt(OmegaM*pow(zp1,3.0) + OmegaL)

    d_com = Comoving_Distance(z)

    return 4.0*np.pi*Hubble_horizon_0/h(z)*d_com*d_com

