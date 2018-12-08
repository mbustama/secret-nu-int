#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "Mauricio Bustamante"
__email__ = "mbustamante@nbi.ku.dk"


"""
nu_flux_plots.py:
    Routines to plots the flux of high-energy astrophysical
    neutrinos at Earth.  Some constants are defined inside global_defs.py.

Created: 2018/10/03 20:54
Last modified: 2018/12/05 01:11
"""

from nu_flux import *


def Plot_Nu_Flux_Earth_Compare_Terms(mass_mediator=0.1,
    coupling_mediator=0.3, mass_neutrino=1.e-10, sol_method='dop853',
    output_format='pdf'):

    print("Plot_Nu_Flux_Earth_Compare_Terms: "+ \
        "Plotting neutrino flux, comparing terms in the propagation...")

    # Open the plot and format it
    mpl.rcParams['xtick.labelsize']=26
    mpl.rcParams['ytick.labelsize']=26
    mpl.rcParams['legend.fontsize']=20
    mpl.rcParams['legend.borderpad']=0.4
    mpl.rcParams['axes.labelpad']=10
    mpl.rcParams['ps.fonttype']=42
    mpl.rcParams['pdf.fonttype']=42

    # Neutrino energy [GeV]
    log10_energy_nu_min = 3.0 # [GeV]
    log10_energy_nu_max = 9.0 # [GeV]
    log10_energy_nu_npts = 100
    lst_log10_energy_nu = \
        np.linspace(log10_energy_nu_min, log10_energy_nu_max, \
            log10_energy_nu_npts) # [GeV]
    lst_energy_nu = [10.**x for x in lst_log10_energy_nu] # [GeV]

    fig = plt.figure(figsize=[9,9])
    ax = fig.add_subplot(1,1,1)

    ax.set_xlabel(r'Neutrino energy $E$ [GeV]', fontsize=25)
    ax.set_ylabel(r'$\nu + \bar{\nu}$ flux at Earth $E^2 J$ '+ \
        '[$10^{-8}$ GeV cm$^{-2}$ s$^{-1}$ sr$^{-1}$]', fontsize=25)

    ax.tick_params('both', length=10, width=2, which='major')
    ax.tick_params('both', length=5, width=1, which='minor')
    ax.tick_params(axis='both', which='major', pad=10, direction='in')
    ax.tick_params(axis='both', which='minor', pad=10, direction='in')
    ax.tick_params(axis='y',which='minor', left='on')
    ax.tick_params(axis='y',which='minor', right='on')
    ax.tick_params(bottom=True, top=True, left=True, right=True)

    pylab.xlim([1.e3, 1.e8])
    ax.set_xscale('log')

    pylab.ylim([0.0, 2.0]) # [0.0, 2.5]
    # ax.set_yscale('log')

    # ax_yticks_major = np.array([1.e-34, 1.e-33, 1.e-32, 1.e-31, 1.e-30, 1.e-29, 1.e-28, 1.e-27, \
    #                     1.e-26, 1.e-25, 1.e-24, 1.e-23, 1.e-22, 1.e-21])
    # ax.yaxis.set_major_locator(matplotlib.ticker.FixedLocator( \
    #     ax_yticks_major))


    # """"
    ###########################################################################
    # Free streaming
    ###########################################################################

    lst_nu_flux = Neutrino_Flux(lst_energy_nu,
                    mass_mediator=mass_mediator,
                    coupling_mediator=coupling_mediator,
                    mass_neutrino=mass_neutrino,
                    z_min=0.0, z_max=4.0,
                    nu_density_z_max=0.0, lum_norm=1.0, lum_gamma=2.0,
                    lum_energy_nu_max=1.e7,
                    flag_include_adiabatic_losses=True,
                    flag_include_source_injection=True,
                    flag_include_attenuation=False,
                    flag_include_regeneration=False,
                    delta_z=1.e-1,
                    atol=1.e-1, rtol=1.e-1, sol_method=sol_method,
                    sol_nsteps=500)
    lst_nu_flux = [lst_energy_nu[i]*lst_energy_nu[i]*lst_nu_flux[i] \
                    for i in range(len(lst_energy_nu))]
    norm = 1./lst_nu_flux[0]
    lst_nu_flux = [norm*nu_flux for nu_flux in lst_nu_flux]
    ax.plot(lst_energy_nu, lst_nu_flux, c='C0', ls=':', lw=2.0,
        label='Free streaming', zorder=10)
    # """

    # """
    ###########################################################################
    # With attenuation
    ###########################################################################

    lst_nu_flux = Neutrino_Flux(lst_energy_nu,
                    mass_mediator=mass_mediator,
                    coupling_mediator=coupling_mediator,
                    mass_neutrino=mass_neutrino,
                    z_min=0.0, z_max=4.0,
                    nu_density_z_max=0.0, lum_norm=1.0, lum_gamma=2.0,
                    lum_energy_nu_max=1.e7,
                    flag_include_adiabatic_losses=True,
                    flag_include_source_injection=True,
                    flag_include_attenuation=True,
                    flag_include_regeneration=False,
                    delta_z=1.e-1,
                    atol=1.e-1, rtol=1.e-1, sol_method=sol_method,
                    sol_nsteps=500)
    lst_nu_flux = [lst_energy_nu[i]*lst_energy_nu[i]*lst_nu_flux[i] \
                    for i in range(len(lst_energy_nu))]
    lst_nu_flux = [norm*nu_flux for nu_flux in lst_nu_flux]
    ax.plot(lst_energy_nu, lst_nu_flux, c='C1', ls='--', lw=2.0,
        label='With attenuation', zorder=5)
    # """

    # """
    ###########################################################################
    # With attenuation + regeneration
    ###########################################################################

    lst_nu_flux = Neutrino_Flux(lst_energy_nu,
                    mass_mediator=mass_mediator,
                    coupling_mediator=coupling_mediator,
                    mass_neutrino=mass_neutrino,
                    z_min=0.0, z_max=4.0,
                    nu_density_z_max=0.0, lum_norm=1.0, lum_gamma=2.0,
                    lum_energy_nu_max=1.e7,
                    flag_include_adiabatic_losses=True,
                    flag_include_source_injection=True,
                    flag_include_attenuation=True,
                    flag_include_regeneration=True,
                    delta_z=1.e-1,
                    atol=1.e-1, rtol=1.e-1, sol_method=sol_method,
                    sol_nsteps=500, flag_regeneration_integ_method='euler')
    lst_nu_flux = [lst_energy_nu[i]*lst_energy_nu[i]*lst_nu_flux[i] \
                    for i in range(len(lst_energy_nu))]
    lst_nu_flux = [norm*nu_flux for nu_flux in lst_nu_flux]
    ax.plot(lst_energy_nu, lst_nu_flux, c='C2', ls='-', lw='2.0', \
        label='With attenuation + regeneration', zorder=7)
    # """

    # Legend
    ax.legend(loc='upper right')

    # Save the plot
    filename_out = os.getcwd()+'/output/plots/nu_flux_compare_terms'
    pylab.savefig(filename_out+'.'+output_format, \
        bbox_inches='tight', dpi=300)
    plt.close()

    return


Plot_Nu_Flux_Earth_Compare_Terms(mass_mediator=0.01, coupling_mediator=0.03,
    mass_neutrino=1.e-10, sol_method='dop853', output_format='pdf')
