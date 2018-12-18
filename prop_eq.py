#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "Mauricio Bustamante"
__email__ = "mbustamante@nbi.ku.dk"


"""
prop_eq.py:
    Routines to calculate terms on the right-hand side of the neutrino
    propagation equation. Some constants are defined inside global_defs.py.
Created: 2018/10/03 21:17
Last modified: 2018/12/05 01:11
"""

import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import quad
from scipy.integrate import simps
from scipy.integrate import trapz
from scipy.integrate import fixed_quad

from cosmology import *
from cross_section import*
from global_defs import *


def Adiabatic_Energy_Losses(z, energy_nu, nu_density):
    """
    Returns the term for adiabatic energy losses in the propagation equation.
    Parameters
    ----------
    z : flt
        [Adim] Redshift
    energy_nu : flt
        [GeV] Neutrino energy at redshift z
    nu_density : flt
        [GeV^{-1} cm^{-3}] Comoving neutrino density
    Returns
    -------
    Adiabatic_Energy_Losses : flt
        [GeV^{-1} cm^{-3} s^{-1}] Value of the adiabatic energy losses
    """

    # Adimensional_Hubble_Parameter takes x=1+z as input
    # H0_inv_s is the Hubble constant [s^{-1}]

    return H0_inv_s*Adimensional_Hubble_Parameter(1.0+z)*nu_density


def Differential_Number_Luminosity_Per_Source(energy_nu, norm=1.0,
    gamma=2.0, energy_nu_max=1.e7):
    """
    Returns the differential neutrino number luminosity per source, L_0.
    Assumes a power law with exponential cut-off.
    Parameters
    ----------
    energy_nu : flt
        [GeV] Neutrino energy at redshift z
    lum_norm : flt
        [GeV^{-1} s^{-1}] Normalization constant
    lum_gamma : flt
        [Adim] Spectral index of the power law
    lum_energy_nu_max : flt
        [GeV] Cut-off neutrino energy
    Returns
    -------
    Differential_Number_Luminosity_Per_Source : flt
        [GeV^{-1} s^{-1}] Value of the luminosity
    """

    return norm * pow(energy_nu,-gamma) * exp(-energy_nu/energy_nu_max)


def Neutrino_Injection(z, energy_nu, lum_norm=1.0, lum_gamma=2.0,
    lum_energy_nu_max=1.e7):
    """
    Returns the neutrino injection term in the propagation equation
    Parameters
    ----------
    z : flt
        [Adim] Redshift
    energy_nu : flt
        [GeV] Neutrino energy at redshift z
    lum_norm : flt
        [GeV^{-1} s^{-1}] Normalization constant, passed to
        Differential_Number_Luminosity_Per_Source
    lum_gamma : flt
        [Adim] Spectral index of the power law, passed to
        Differential_Number_Luminosity_Per_Source
    lum_energy_nu_max : flt
        [GeV] Cut-off neutrino energy, passed to
        Differential_Number_Luminosity_Per_Source
    Returns
    -------
    Adiabatic_Energy_Losses : flt
        [GeV^{-1} cm^{-3} s^{-1}] Value of injection term
    """
    L0 = Differential_Number_Luminosity_Per_Source(energy_nu,
            norm=lum_norm, gamma=lum_gamma,
            energy_nu_max=lum_energy_nu_max) # [GeV^{-1} s^{-1}]

    # SFR(z) returns the adimensional star-formation rate.  We do not care
    # about the normalization of the SFR, since we will fit the resulting
    # neutrino flux to IceCube data, which will fix the overall normalization.
    # So, for the sake of this computation, we can assume that SFR has
    # units of [cm^{-3}], in order for Neutrino_Injection to have the right
    # units.

    return SFR(z)*L0


def Number_Density_Relic_Nu(z):
    """
    Returns the number density of relic neutrinos at redshift z.
    Parameters
    ----------
    z : flt
        [Adim] Redshift
    Returns
    -------
    Number_Density_Relic_Nu : flt
        [cm^{-3}] Number density of relic neutrinos
    """

    return 56.*pow(1.0+z, 3.0)


def Interaction_Rate(z, energy_nu, lst_energy_nu, mass_mediator,
                        coupling_mediator, mass_neutrino=1.e-10, 
                        flag_use_integ_diff_cs=True):
    """
    Returns the interaction rate of neutrinos moving on the relic neutrino
    background, due to secret nu-nu interactions
    Parameters
    ----------
    z : flt
        [Adim] Redshift
    energy_nu : flt
        [GeV] Neutrino energy at redshift z
    mass_mediator : flt
        [GeV] Mass of the new mediator, passed to
        Cross_Section_Nu_Nu_S_Channel_Scalar
    coupling_mediator : flt
        [GeV] Coupling of the new mediator, passed to
        Cross_Section_Nu_Nu_S_Channel_Scalar
    mass_neutrino : flt
        [GeV] Mass of the neutrino, passed to
        Cross_Section_Nu_Nu_S_Channel_Scalar
    Returns
    -------
    Interaction_Rate : flt
        [s^{-1]} Interaction rate of propagating neutrinos
    """
    # [Number_Density_Relic_Nu] = cm^{-3}
    # [Cross_Section_Nu_Nu_S_Channel_Scalar] = cm^2

    if (flag_use_integ_diff_cs == False):

        int_rate = Number_Density_Relic_Nu(z) * \
                    Cross_Section_Nu_Nu_S_Channel_Scalar(energy_nu,
                        mass_mediator, coupling_mediator,
                        mass_neutrino=mass_neutrino) # [cm^{-1}]
                    
    else: 
        index_sel = lst_energy_nu.index(energy_nu)
        int_rate = 0.0
        for i in range(index_sel, len(lst_energy_nu)-1, 1):
            h = lst_energy_nu[i+1]-lst_energy_nu[i]
            
            int_rate += Diff_Cross_Section_Nu_Nu_S_Channel_Scalar( \
                        energy_nu, lst_energy_nu[i], mass_mediator,
                        coupling_mediator, mass_neutrino=mass_neutrino)*h
        
        int_rate = int_rate * Number_Density_Relic_Nu(z)

    int_rate = int_rate * speed_light_cm_per_s # [s^{-1}]

    return int_rate


def Integrand_Regeneration(energy_nu_in, energy_nu_out, interp_nu_density,
    mass_mediator=0.03, coupling_mediator=0.01, mass_neutrino=1.e-10):

    """
    Returns the integrand of the energy integral to compute the regeneration
    term in the propagation equation.
    Parameters
    ----------
    energy_nu_in : flt
        [GeV] Incoming neutrino energy at redshift z
    energy_nu_out : flt
        [GeV] Outgoing neutrino energy at redshift z
    interp_nu_density : flt
        [GeV^{-1} cm^{-3}] Interpolating function of comoving neutrino density
    mass_mediator : flt
        [GeV] Mass of the new mediator, passed to
        Cross_Section_Nu_Nu_S_Channel_Scalar
    coupling_mediator : flt
        [GeV] Coupling of the new mediator, passed to
        Cross_Section_Nu_Nu_S_Channel_Scalar
    mass_neutrino : flt
        [GeV] Mass of the neutrino, passed to
        Cross_Section_Nu_Nu_S_Channel_Scalar
    Returns
    -------
    Integrand_Regeneration : flt
        [GeV^{-2} cm^{-1}] Integrand of the energy integral to compute
        regeneration
    """

    if type(energy_nu_in) is np.ndarray:
        energy_nu_in = energy_nu_in[0]

    diff_cs = Diff_Cross_Section_Nu_Nu_S_Channel_Scalar(energy_nu_in,
                energy_nu_out, mass_mediator, coupling_mediator,
                mass_neutrino=mass_neutrino) # [cm^2 GeV^{-1}]

    # [integrand] = [GeV^{-1} cm^{-3}] * [cm^2 GeV^{-1}] = [GeV^{-2} cm^{-1}]
    integrand = interp_nu_density(energy_nu_in) * diff_cs

    return integrand # [GeV^{-2} cm^{-1}]


def Propagation_Eq_RHS(z, nu_density, energy_nu,
    lst_energy_nu, lst_nu_density,
    mass_mediator=0.03, coupling_mediator=0.01, mass_neutrino=1.e-10,
    lum_norm=1.0, lum_gamma=2.0, lum_energy_nu_max=1.e7,
    flag_regeneration_integ_method='fixed_quad',
    flag_include_adiabatic_losses=True,
    flag_include_source_injection=True,
    flag_include_attenuation=True,
    flag_use_integ_diff_cs=True, 
    flag_include_regeneration=True):
    """
    Returns the right-hand side of the neutrino propagation equation
    Parameters
    ----------
    z : flt
        [Adim] Redshift
    energy_nu : flt
        [GeV] Neutrino energy at redshift z
    nu_density : flt
        [GeV^{-1} cm^{-3}] Comoving neutrino density
    mass_mediator : flt
        [GeV] Mass of the new mediator, passed to
        Cross_Section_Nu_Nu_S_Channel_Scalar
    coupling_mediator : flt
        [GeV] Coupling of the new mediator, passed to
        Cross_Section_Nu_Nu_S_Channel_Scalar
    mass_neutrino : flt
        [GeV] Mass of the neutrino, passed to
        Cross_Section_Nu_Nu_S_Channel_Scalar
    lum_norm : flt
        [GeV^{-1} s^{-1}] Normalization constant, passed to
        Differential_Number_Luminosity_Per_Source
    lum_gamma : flt
        [Adim] Spectral index of the power law, passed to
        Differential_Number_Luminosity_Per_Source
    lum_energy_nu_max : flt
        [GeV] Cut-off neutrino energy, passed to
        Differential_Number_Luminosity_Per_Source
    flag_regeneration_integ_method : str
        [Adim] Integration method used to calculate the energy integral
        in the regeneration term.  Choose from 'quad', 'fixed_quad', 'trapz',
        'simps'.
    flag_include_adiabatic_losses : bool
        [Adim] If True, include the term for adiabatic energy losses
    flag_include_source_injection : bool
        [Adim] If True, include the term for neutrino injection from sources
    flag_include_attenuation : bool
        [Adim] If True, include the term for attenuation due to nu-nu
        interactions
    flag_include_attenuation : bool
        [Adim] If True, include the term for regeneration due to nu-nu
        interactions
    Returns
    -------
    Propagation_Eq_RHS : flt
        [GeV^{-1} cm^{-3}] Right-hand side of the propagation equation
    """

    rhs = 0.0 # [GeV^{-1} cm^{-3} s^{-1}]

    if flag_include_adiabatic_losses:
        rhs += Adiabatic_Energy_Losses(z, energy_nu, nu_density)

    if flag_include_source_injection:
        rhs += Neutrino_Injection(z, energy_nu, lum_norm=lum_norm,
                lum_gamma=lum_gamma, lum_energy_nu_max=lum_energy_nu_max)

    if flag_include_attenuation:
        rhs += -Interaction_Rate(z, energy_nu, lst_energy_nu, mass_mediator,
                 coupling_mediator, mass_neutrino=1.e-10,
                 flag_use_integ_diff_cs=True) * \
                 nu_density
                     

    if flag_include_regeneration:
        interp_nu_density = interp1d(lst_energy_nu, lst_energy_nu,
                            kind='linear', bounds_error=False,
                            fill_value='extrapolate')
        # [Integrand_Regneration] = [GeV^{-2} cm^{-1}]
        # [regen] = [GeV^{-1} cm^{-1}]
        if (flag_regeneration_integ_method == 'quad'):
            regen = quad(lambda energy_nu_in: \
                Integrand_Regeneration(energy_nu_in,
                energy_nu, interp_nu_density, mass_mediator=mass_neutrino,
                coupling_mediator=coupling_mediator,
                mass_neutrino=mass_neutrino),
                energy_nu, np.inf, epsabs=1.e-1, epsrel=1.e-1)[0]
        elif (flag_regeneration_integ_method == 'fixed_quad'):
            regen = list(fixed_quad(lambda energy_nu_in: \
                Integrand_Regeneration(energy_nu_in,
                energy_nu, interp_nu_density, mass_mediator=mass_neutrino,
                coupling_mediator=coupling_mediator,
                mass_neutrino=mass_neutrino),
                energy_nu, 1.e8, n=3))[0]
        elif (flag_regeneration_integ_method == 'simps'):
            # Energies at which to evaluate the energy integral [GeV]
            log10_energy_nu_integral_lo = log10(energy_nu)
            log10_energy_nu_integral_hi = 14.0
            log10_energy_nu_integral_npts = 10
            lst_log10_energy_nu_integral = \
                np.linspace(log10_energy_nu_integral_lo,
                            log10_energy_nu_integral_hi,
                            log10_energy_nu_integral_npts)
            lst_energy_nu_integral = \
                [10.**x for x in lst_log10_energy_nu_integral] # [GeV]
            # Differential cross section [cm^2 GeV^{-1}]
            lst_diff_cs = \
                [Diff_Cross_Section_Nu_Nu_S_Channel_Scalar(energy_nu_in,
                energy_nu, mass_mediator, coupling_mediator,
                mass_neutrino=mass_neutrino) \
                for energy_nu_in in lst_energy_nu_integral]
            # Integrand [GeV^{-1} cm^{-3}]*[cm^2 GeV^{-1}]=[GeV^{-2} cm^{-1}]
            lst_integrand_regeneration = \
                [interp_nu_density(lst_energy_nu_integral[i])*lst_diff_cs[i] \
                for i in range(log10_energy_nu_integral_npts)]
            # Integrate in incoming neutrino energy
            regen = simps(lst_energy_nu_integral, lst_integrand_regeneration)
        elif (flag_regeneration_integ_method == 'trapz'):
            # Energies at which to evaluate the energy integral [GeV]
            log10_energy_nu_integral_lo = log10(energy_nu)
            log10_energy_nu_integral_hi = 14.0
            log10_energy_nu_integral_npts = 10
            lst_log10_energy_nu_integral = \
                np.linspace(log10_energy_nu_integral_lo,
                            log10_energy_nu_integral_hi,
                            log10_energy_nu_integral_npts)
            lst_energy_nu_integral = \
                [10.**x for x in lst_log10_energy_nu_integral] # [GeV]
            # Differential cross section [cm^2 GeV^{-1}]
            lst_diff_cs = \
                [Diff_Cross_Section_Nu_Nu_S_Channel_Scalar(energy_nu_in,
                energy_nu, mass_mediator, coupling_mediator,
                mass_neutrino=mass_neutrino) \
                for energy_nu_in in lst_energy_nu_integral]
            # Integrand [GeV^{-1} cm^{-3}]*[cm^2 GeV^{-1}]=[GeV^{-2} cm^{-1}]
            lst_integrand_regeneration = \
                [interp_nu_density(lst_energy_nu_integral[i])*lst_diff_cs[i] \
                for i in range(log10_energy_nu_integral_npts)]
            # Integrate in incoming neutrino energy
            regen = trapz(lst_energy_nu_integral, lst_integrand_regeneration)
        elif (flag_regeneration_integ_method == 'euler'):
            index_sel = lst_energy_nu.index(energy_nu)
            # Differential cross section [cm^2 GeV^{-1}]
            regen = 0.0
            for i in range(index_sel, len(lst_energy_nu)-1, 1):
                h = lst_energy_nu[i+1]-lst_energy_nu[i]
                regen += -Diff_Cross_Section_Nu_Nu_S_Channel_Scalar( \
                            energy_nu, lst_energy_nu[i], mass_mediator,
                            coupling_mediator, mass_neutrino=mass_neutrino) * \
                            lst_nu_density[index_sel]*h
                regen += Diff_Cross_Section_Nu_Nu_S_Channel_Scalar( \
                            lst_energy_nu[i], energy_nu, mass_mediator,
                            coupling_mediator, mass_neutrino=mass_neutrino) * \
                            lst_nu_density[i]*h

        # [rhs] = [cm^{-3}] * [cm s^{-1}] * [GeV^{-2} cm^{-1}]
        #       = [GeV^{-1} cm^{-3} s^{-1}]
        rhs += Number_Density_Relic_Nu(z) * speed_light_cm_per_s * regen

    # [GeV^{-1} cm^{-3} s^{-1}] / [s^{-1}] = [GeV^{-1} cm^{-3}]
    rhs = rhs/(-(1.0+z)*H0_inv_s*Adimensional_Hubble_Parameter(1.0+z))


    return rhs # [GeV^{-1} cm^{-3}]
