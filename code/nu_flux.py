#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "Mauricio Bustamante"
__email__ = "mbustamante@nbi.ku.dk"


"""
nu_flux.py:
    Routines to calculate the flux of high-energy astrophysical
    neutrinos at Earth.  Some constants are defined inside global_defs.py.

Created: 2018/10/02 14:21
Last modified: 2018/12/05 01:11
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.integrate import ode

from prop_eq import *
from cosmology import *
from cross_section import*
from global_defs import *


def Neutrino_Flux(lst_energy_nu, mass_mediator=0.03, coupling_mediator=0.01,
    mass_neutrino=1.e-10, z_min=0.0, z_max=6.0, nu_density_z_max=0.0,
    lum_norm=1.0, lum_gamma=2.0, lum_energy_nu_max=1.e7,
    flag_include_adiabatic_losses=True,
    flag_include_source_injection=True,
    flag_include_attenuation=True,
    flag_include_regeneration=True,
    delta_z=1.e-2,
    atol=1.e-4, rtol=1.e-4, sol_method='dop853', sol_nsteps=1000,
    flag_regeneration_integ_method='simps'):
    """
    Returns the neutrino flux at redshift z_min, evaluated at energy energy_nu.
    The propagation equation is solved from z_max down to z_min.

    Parameters
    ----------

    energy_nu : flt
        [GeV] Neutrino energy at redshift z

    z_min : flt
        [Adim] Minimum redshift, i.e., redshift at which the flux is returned

    z_max : flt
        [Adim] Maximum redshift, i.e., redshift where initial conditions are
        given

    nu_density_z_max : flt
        [GeV^{-1} cm^{-3}] Initial, comoving neutrino density, given at
        redshift z_max

    lum_norm : flt
        [GeV^{-1} s^{-1}] Normalization constant, passed to
        Differential_Number_Luminosity_Per_Source

    lum_gamma : flt
        [Adim] Spectral index of the power law, passed to
        Differential_Number_Luminosity_Per_Source

    lum_energy_nu_max : flt
        [GeV] Cut-off neutrino energy, passed to
        Differential_Number_Luminosity_Per_Source

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

    atol : flt
        [Adim] Absolute tolerance parameter of the ODE solver

    rtol : flt
        [Adim] Relative tolerance parameter of the ODE solver

    Returns
    -------

    Neutrino_Flux : flt
        [GeV^{-1} cm^{-2} s^{-1} sr^{-1}] Neutrino flux evaluated at z_min and
        energy energy_nu
    """
    def Integrand(z, nu_density, energy_nu, interp_nu_density):
        return Propagation_Eq_RHS(z, nu_density, energy_nu,
                lst_energy_nu, lst_nu_density,
                mass_mediator=mass_mediator,
                coupling_mediator=coupling_mediator,
                mass_neutrino=mass_neutrino,
                lum_norm=lum_norm,
                lum_gamma=lum_gamma,
                lum_energy_nu_max=lum_energy_nu_max,
                flag_regeneration_integ_method=flag_regeneration_integ_method,
                flag_include_adiabatic_losses=flag_include_adiabatic_losses,
                flag_include_source_injection=flag_include_source_injection,
                flag_include_attenuation=flag_include_attenuation,
                flag_include_regeneration=flag_include_regeneration)


    """
    sol = solve_ivp(Integrand, t_span=[z_max, z_min], y0=nu_density_z_max,
            method='RK45', t_eval=[z_min], atol=atol, rtol=rtol)

    print(sol.t)
    print(sol.y)
    """

    if (sol_method == 'dop853'):
        # Apparently, nsteps=500 by default
        solver = ode(Integrand, jac=None).set_integrator('dop853', atol=atol,
            rtol=rtol, nsteps=sol_nsteps, max_step=1.e-3, verbosity=1)
    elif (sol_method == 'dopri5'):
        # Apparently, nsteps=500 by default
        solver = ode(Integrand, jac=None).set_integrator('dopri5', atol=atol,
            rtol=rtol, nsteps=sol_nsteps, max_step=1.e-3, verbosity=1)
    elif (sol_method == 'vode'):
        solver = ode(Integrand, jac=None).set_integrator('vode',
            method='bdf', atol=atol, rtol=rtol, nsteps=sol_nsteps, max_step=1.e-5,
            with_jacobian=False)

    # Initialize density at z = z_max
    lst_nu_density = [0.0]*len(lst_energy_nu)

    # Redshift step
    dz = delta_z

    z = z_max
    # while(solver.successful() and solver.t > z_min):
    while (z > z_min):
        print(z)
        lst_nu_density_new = []
        for i in range(len(lst_energy_nu)):
            # print("  i = ", i, ",  log10(energy_nu/GeV) = ", log10(lst_energy_nu[i]))
            solver.set_initial_value(lst_nu_density[i], z)
            solver.set_f_params(lst_energy_nu[i], lst_nu_density)
            sol = solver.integrate(solver.t-dz)
            lst_nu_density_new.append(sol)
            # print('  %g' % solver.t)
        lst_nu_density = [x for x in lst_nu_density_new]
        z = z-dz

    return lst_nu_density

"""
lst_energy_nu = [1.e5, 1.e6]
lst_nu_density = Neutrino_Flux(lst_energy_nu, mass_mediator=0.03, coupling_mediator=0.01,
    mass_neutrino=1.e-10, z_min=0.0, z_max=6.0, nu_density_z_max=0.0,
    lum_norm=1.0, lum_gamma=2.0, lum_energy_nu_max=1.e7,
    flag_include_adiabatic_losses=True,
    flag_include_source_injection=True,
    flag_include_attenuation=True,
    flag_include_regeneration=True,
    atol=1.e-4, rtol=1.e-4, sol_method='dop853', sol_nsteps=1000)
print(lst_nu_density)
"""
