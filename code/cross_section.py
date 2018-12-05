#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "Mauricio Bustamante"
__email__ = "mbustamante@nbi.ku.dk"


"""
cross_section.py:
    Routines to calculate neutrino-neutrino cross sections

Created: 2018/09/30 18:25
Last modified: 2018/09/30 18:25
"""


import numpy as np
from pylab import *
from matplotlib import *
import matplotlib.pyplot as plt
import matplotlib as mpl

from global_defs import *


def Cross_Section_Nu_Nu_S_Channel_Scalar(energy_nu, mass_mediator, \
    coupling_mediator, mass_neutrino=1.e-10):

    prefactor = pow(coupling_mediator, 4.0)/(4.*np.pi)
    mass_mediator_sq = mass_mediator*mass_mediator
    width_mediator = \
        coupling_mediator*coupling_mediator*mass_mediator / (4.*np.pi)
    s = 2.*energy_nu*mass_neutrino

    den = pow(s-mass_mediator_sq, 2.0) + \
            mass_mediator_sq*width_mediator*width_mediator

    cs = hbar_c_sq * prefactor * s / den

    return cs # [cm^2]


def Diff_Cross_Section_Nu_Nu_S_Channel_Scalar(energy_nu_in, energy_nu_out, \
    mass_mediator, coupling_mediator, mass_neutrino=1.e-10):

    if (energy_nu_in < energy_nu_out):
        return 0.0

    ###########################################################################
    # Calculate the cross section here to save the call to another function
    ###########################################################################

    # prefactor = pow(coupling_mediator, 4.0)/(4.*np.pi)
    # mass_mediator_sq = mass_mediator*mass_mediator
    # width_mediator = \
    #     coupling_mediator*coupling_mediator*mass_mediator / (4.*np.pi)
    # s = 2.*energy_nu_in*mass_neutrino

    # den = pow(s-mass_mediator_sq, 2.0) + \
    #         mass_mediator_sq*width_mediator*width_mediator

    # cs = hbar_c_sq * prefactor * s / den # [cm^2]

    ###########################################################################
    # Distribution function of outgoing neutrinos
    ###########################################################################

    cs = Cross_Section_Nu_Nu_S_Channel_Scalar(energy_nu_in, mass_mediator,
            coupling_mediator, mass_neutrino=mass_neutrino) # [cm^2]
    energy_nu_ratio = energy_nu_out/energy_nu_in
    dist_fun = 3.0/energy_nu_in * \
                (pow(energy_nu_ratio, 2.0) + pow(1.0-energy_nu_ratio, 2.0))

    return cs*dist_fun


def Diff_Cross_Section_Nu_Nu_S_Channel_Scalar_v0(energy_nu, mass_mediator, \
    coupling_mediator, mass_neutrino=1.e-10):

    ###########################################################################
    # Calculate the cross section here to save the call to another function
    ###########################################################################

    prefactor = pow(coupling_mediator, 4.0)/(4.*np.pi)
    mass_mediator_sq = mass_mediator*mass_mediator
    width_mediator = \
        coupling_mediator*coupling_mediator*mass_mediator / (4.*np.pi)
    s = 2.*energy_nu*mass_neutrino

    den = pow(s-mass_mediator_sq, 2.0) + \
            mass_mediator_sq*width_mediator*width_mediator

    cs = hbar_c_sq * prefactor * s / den # [cm^2]

    ###########################################################################
    # Calculate the differential cross section
    ###########################################################################

    t = 2.0 / prefactor / hbar_c_sq * (s-mass_mediator_sq)
    diff_cs = cs/energy_nu * (1.0 - t*cs)

    return diff_cs


def Diff_Cross_Section_Nu_Nu_S_Channel_Scalar_v1(energy_nu_in, energy_nu_out, \
    mass_mediator, coupling_mediator, mass_neutrino=1.e-10):

    energy_min = mass_neutrino*energy_nu_in / (2.*energy_nu_in+mass_neutrino)

    if ((energy_nu_out < energy_min) and (energy_nu_out > energy_in_min)):
        return 0.0

    ###########################################################################
    # Calculate the cross section here to save the call to another function
    ###########################################################################

    prefactor = pow(coupling_mediator, 4.0)/(4.*np.pi)
    mass_mediator_sq = mass_mediator*mass_mediator
    width_mediator = \
        coupling_mediator*coupling_mediator*mass_mediator / (4.*np.pi)
    s = 2.*energy_nu_in*mass_neutrino

    den = pow(s-mass_mediator_sq, 2.0) + \
            mass_mediator_sq*width_mediator*width_mediator

    cs = hbar_c_sq * prefactor * s / den # [cm^2]

    ###########################################################################
    # Calculate the differential cross section
    ###########################################################################

    den = energy_nu_in-energy_min

    return cs/den


def Plot_Cross_Section_Nu_Nu_S_Channel_Scalar(lst_mass_mediator=[0.1], \
    lst_coupling_mediator=[0.3], lst_legends=[''], mass_neutrino=1.e-10, \
    output_format='pdf'):

    print("Plot_Cross_Section_Nu_Nu_S_Channel_Scalar: "+ \
        "Plotting cross section nu-nu s-channel, scalar mediator...")

    # Open the plot and format it
    mpl.rcParams['xtick.labelsize']=26
    mpl.rcParams['ytick.labelsize']=26
    mpl.rcParams['legend.fontsize']=18
    mpl.rcParams['legend.borderpad']=0.4
    mpl.rcParams['axes.labelpad']=10
    mpl.rcParams['ps.fonttype']=42
    mpl.rcParams['pdf.fonttype']=42

    # Neutrino energy [GeV]
    log10_energy_nu_min = 3.0 # [GeV]
    log10_energy_nu_max = 9.0 # [GeV]
    log10_energy_nu_npts = 1000
    lst_log10_energy_nu = \
        np.linspace(log10_energy_nu_min, log10_energy_nu_max, \
            log10_energy_nu_npts) # [GeV]
    lst_energy_nu = [10.**x for x in lst_log10_energy_nu] # [GeV]

    fig = plt.figure(figsize=[9,9])
    ax = fig.add_subplot(1,1,1)

    ax.set_xlabel(r'Neutrino energy $E$ [GeV]', fontsize=25)
    ax.set_ylabel(r'Total $\nu\nu$ $s$-channel cross section $\sigma$ [cm$^2$]', fontsize=25)

    ax.tick_params('both', length=10, width=2, which='major')
    ax.tick_params('both', length=5, width=1, which='minor')
    ax.tick_params(axis='both', which='major', pad=10, direction='in')
    ax.tick_params(axis='both', which='minor', pad=10, direction='in')
    ax.tick_params(axis='y',which='minor', left='on')
    ax.tick_params(axis='y',which='minor', right='on')
    ax.tick_params(bottom=True, top=True, left=True, right=True)

    pylab.xlim([10.**log10_energy_nu_min, 10.**log10_energy_nu_max])
    ax.set_xscale('log')

    pylab.ylim([1e-31, 1e-20])
    ax.set_yscale('log')

    ax_yticks_major = np.array([1.e-31, 1.e-30, 1.e-29, 1.e-28, 1.e-27, \
                        1.e-26, 1.e-25, 1.e-24, 1.e-23, 1.e-22, 1.e-21, \
                        1.e-20])
    ax.yaxis.set_major_locator(matplotlib.ticker.FixedLocator( \
        ax_yticks_major))

    lst_colors = ['C0', 'C1', 'C2', 'C3']
    lst_ls = ['-', '-', '-', '-']
    for i in range(len(lst_mass_mediator)):
        mass_mediator = lst_mass_mediator[i]
        coupling_mediator = lst_coupling_mediator[i]
        ls = lst_ls[i]
        color = lst_colors[i]
        legend = lst_legends[i]
        # print(mass_mediator)
        # print(coupling_mediator)
        # print(ls)
        # print(color)
        # print(legend)
        # Total cross section, sigma [cm^2]
        lst_cs = [Cross_Section_Nu_Nu_S_Channel_Scalar(
                    energy_nu,
                    mass_mediator=mass_mediator,
                    coupling_mediator=coupling_mediator,
                    mass_neutrino=mass_neutrino)
                    for energy_nu in lst_energy_nu]
        # Cross section at the resonant energy
        energy_nu_res = mass_mediator*mass_mediator/(2.*mass_neutrino)
        cs_res = hbar_c_sq*4.*np.pi/(mass_mediator*mass_mediator)
        print(cs_res)
        # Introduce the resonant point in the lists at the right spot
        for j in range(1, len(lst_cs), 1):
            if (lst_cs[j] < lst_cs[j-1]):
                lst_cs.insert(j, cs_res)
                lst_energy_nu.insert(j, energy_nu_res)
                break

        # print(len(lst_energy_nu), len(lst_cs))
        # print()
        # Plot the data
        ax.plot(lst_energy_nu, lst_cs, c=color, ls=ls, lw='2.0', label=legend)

    # Legend
    ax.legend(loc='upper right')

    # Save the plot
    filename_out = os.getcwd()+'/output/plots/cs_nu_nu_s_channel_scalar'
    pylab.savefig(filename_out+'.'+output_format, \
        bbox_inches='tight', dpi=300)
    plt.close()

    return


def Plot_Diff_Cross_Section_Nu_Nu_S_Channel_Scalar(lst_mass_mediator=[0.1], \
    lst_coupling_mediator=[0.3], lst_legends=[''], mass_neutrino=1.e-10, \
    output_format='pdf'):

    print("Plot_Diff_Cross_Section_Nu_Nu_S_Channel_Scalar: "+ \
        "Plotting differential cross section nu-nu s-channel, scalar mediator...")

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
    log10_energy_nu_npts = 1000
    lst_log10_energy_nu = \
        np.linspace(log10_energy_nu_min, log10_energy_nu_max, \
            log10_energy_nu_npts) # [GeV]
    lst_energy_nu = [10.**x for x in lst_log10_energy_nu] # [GeV]

    fig = plt.figure(figsize=[9,9])
    ax = fig.add_subplot(1,1,1)

    ax.set_xlabel(r'Neutrino energy $E$ [GeV]', fontsize=25)
    ax.set_ylabel(r'Diff. $\nu\nu$ $s$-channel cross section ${\rm d}\sigma/{\rm d}E$ [cm$^2$ GeV$^{-1}$]', fontsize=25)

    ax.tick_params('both', length=10, width=2, which='major')
    ax.tick_params('both', length=5, width=1, which='minor')
    ax.tick_params(axis='both', which='major', pad=10, direction='in')
    ax.tick_params(axis='both', which='minor', pad=10, direction='in')
    ax.tick_params(axis='y',which='minor', left='on')
    ax.tick_params(axis='y',which='minor', right='on')
    ax.tick_params(bottom=True, top=True, left=True, right=True)

    pylab.xlim([10.**log10_energy_nu_min, 10.**log10_energy_nu_max])
    ax.set_xscale('log')

    pylab.ylim([1e-34, 1e-25])
    ax.set_yscale('log')

    ax_yticks_major = np.array([1.e-34, 1.e-33, 1.e-32, 1.e-31, 1.e-30, 1.e-29, 1.e-28, 1.e-27, \
                        1.e-26, 1.e-25, 1.e-24, 1.e-23, 1.e-22, 1.e-21])
    ax.yaxis.set_major_locator(matplotlib.ticker.FixedLocator( \
        ax_yticks_major))

    lst_colors = ['C0', 'C1', 'C2', 'C3']
    lst_ls = ['-', '-', '-', '-']
    for i in range(len(lst_mass_mediator)):
        mass_mediator = lst_mass_mediator[i]
        coupling_mediator = lst_coupling_mediator[i]
        ls = lst_ls[i]
        color = lst_colors[i]
        legend = lst_legends[i]
        # Differential cross section, sigma [cm^2 GeV^{-1}]
        lst_diff_cs = [Diff_Cross_Section_Nu_Nu_S_Channel_Scalar(
                        energy_nu,
                        mass_mediator=mass_mediator,
                        coupling_mediator=coupling_mediator,
                        mass_neutrino=mass_neutrino)
                        for energy_nu in lst_energy_nu]
        # Cross section at the resonant energy
        energy_nu_res = mass_mediator*mass_mediator/(2.*mass_neutrino)
        diff_cs_res = hbar_c_sq/(mass_mediator*mass_mediator)/energy_nu_res
        # Introduce the resonant point in the lists at the right spot
        for j in range(1, len(lst_diff_cs), 1):
            if (lst_diff_cs[j] < lst_diff_cs[j-1]):
                lst_diff_cs.insert(j, diff_cs_res)
                lst_energy_nu.insert(j, energy_nu_res)
                break

        # Plot the data
        ax.plot(lst_energy_nu, lst_diff_cs, c=color, ls=ls, lw='2.0', \
            label=legend)

    # Legend
    ax.legend(loc='upper right')

    # Save the plot
    filename_out = os.getcwd()+'/output/plots/diff_cs_nu_nu_s_channel_scalar'
    pylab.savefig(filename_out+'.'+output_format, \
        bbox_inches='tight', dpi=300)
    plt.close()

    return


def Plot_Diff_Cross_Section_Nu_Nu_S_Channel_Scalar_v1(lst_mass_mediator=[0.1], \
    lst_coupling_mediator=[0.3], lst_legends=[''], mass_neutrino=1.e-10, \
    output_format='pdf'):

    print("Plot_Diff_Cross_Section_Nu_Nu_S_Channel_Scalar_v1: "+ \
        "Plotting differential cross section nu-nu s-channel, scalar mediator...")

    # Open the plot and format it
    mpl.rcParams['xtick.labelsize']=26
    mpl.rcParams['ytick.labelsize']=26
    mpl.rcParams['legend.fontsize']=20
    mpl.rcParams['legend.borderpad']=0.4
    mpl.rcParams['axes.labelpad']=10
    mpl.rcParams['ps.fonttype']=42
    mpl.rcParams['pdf.fonttype']=42

    # Neutrino energy [GeV]
    log10_energy_nu_out_min = 3.0 # [GeV]
    log10_energy_nu_out_max = 9.0 # [GeV]
    log10_energy_nu_out_npts = 1000
    lst_log10_energy_nu_out = \
        np.linspace(log10_energy_nu_out_min, log10_energy_nu_out_max, \
            log10_energy_nu_out_npts) # [GeV]
    lst_energy_nu_out = [10.**x for x in lst_log10_energy_nu_out] # [GeV]

    fig = plt.figure(figsize=[9,9])
    ax = fig.add_subplot(1,1,1)

    ax.set_xlabel(r'Neutrino energy $E$ [GeV]', fontsize=25)
    ax.set_ylabel(r'Diff. $\nu\nu$ $s$-channel cross section ${\rm d}\sigma/{\rm d}E$ [cm$^2$ GeV$^{-1}$]', fontsize=25)

    ax.tick_params('both', length=10, width=2, which='major')
    ax.tick_params('both', length=5, width=1, which='minor')
    ax.tick_params(axis='both', which='major', pad=10, direction='in')
    ax.tick_params(axis='both', which='minor', pad=10, direction='in')
    ax.tick_params(axis='y',which='minor', left='on')
    ax.tick_params(axis='y',which='minor', right='on')
    ax.tick_params(bottom=True, top=True, left=True, right=True)

    pylab.xlim([10.**log10_energy_nu_out_min, 10.**log10_energy_nu_out_max])
    ax.set_xscale('log')

    # pylab.ylim([1e-34, 1e-25])
    ax.set_yscale('log')

    ax_yticks_major = np.array([1.e-34, 1.e-33, 1.e-32, 1.e-31, 1.e-30, 1.e-29, 1.e-28, 1.e-27, \
                        1.e-26, 1.e-25, 1.e-24, 1.e-23, 1.e-22, 1.e-21])
    ax.yaxis.set_major_locator(matplotlib.ticker.FixedLocator( \
        ax_yticks_major))

    lst_colors = ['C0', 'C1', 'C2', 'C3']
    lst_ls = ['-', '-', '-', '-']
    energy_nu_in = 1.e6
    for i in range(len(lst_mass_mediator)):
        mass_mediator = lst_mass_mediator[i]
        coupling_mediator = lst_coupling_mediator[i]
        ls = lst_ls[i]
        color = lst_colors[i]
        legend = lst_legends[i]
        # Differential cross section, sigma [cm^2 GeV^{-1}]
        lst_diff_cs = [Diff_Cross_Section_Nu_Nu_S_Channel_Scalar_v1(
                        energy_nu_in, energy_nu_out,
                        mass_mediator=mass_mediator,
                        coupling_mediator=coupling_mediator,
                        mass_neutrino=mass_neutrino)
                        for energy_nu_out in lst_energy_nu_out]
        # Cross section at the resonant energy
        energy_nu_res = mass_mediator*mass_mediator/(2.*mass_neutrino)
        diff_cs_res = hbar_c_sq/(mass_mediator*mass_mediator)/energy_nu_res
        # Introduce the resonant point in the lists at the right spot
        for j in range(1, len(lst_diff_cs), 1):
            if (lst_diff_cs[j] < lst_diff_cs[j-1]):
                lst_diff_cs.insert(j, diff_cs_res)
                lst_energy_nu_out.insert(j, energy_nu_res)
                break

        # Plot the data
        ax.plot(lst_energy_nu_out, lst_diff_cs, c=color, ls=ls, lw='2.0', \
            label=legend)

    # Legend
    ax.legend(loc='upper right')

    # Save the plot
    filename_out = os.getcwd()+'/output/plots/diff_cs_nu_nu_s_channel_scalar_v1'
    pylab.savefig(filename_out+'.'+output_format, \
        bbox_inches='tight', dpi=300)
    plt.close()

    return


"""
lst_mass_mediator = [1.e-1] # [GeV]
lst_coupling_mediator = [0.3]
lst_legends=['']
mass_neutrino = 1.e-10 # [GeV]

Plot_Diff_Cross_Section_Nu_Nu_S_Channel_Scalar_v1(
    lst_mass_mediator=lst_mass_mediator,
    lst_coupling_mediator=lst_coupling_mediator,
    lst_legends=lst_legends,
    mass_neutrino=mass_neutrino,
    output_format='pdf')
"""

"""
lst_mass_mediator = [1.e-1, 1.e-2, 3.e-3, 1.e-3] # [GeV]
lst_coupling_mediator = [0.3, 0.03, 0.03, 0.01]
mass_neutrino = 1.e-10 # [GeV]
lst_legends = [
    r'A: $M$ = 100 MeV, $g = 0.3$',
    r'B: $M$ = 10 MeV, $g = 0.03$',
    r'C: $M$ = 3 MeV, $g = 0.03$',
    r'D: $M$ = 1 MeV, $g = 0.01$']

Plot_Cross_Section_Nu_Nu_S_Channel_Scalar(
    lst_mass_mediator=lst_mass_mediator,
    lst_coupling_mediator=lst_coupling_mediator,
    lst_legends=lst_legends,
    mass_neutrino=mass_neutrino,
    output_format='pdf')

Plot_Cross_Section_Nu_Nu_S_Channel_Scalar(
    lst_mass_mediator=lst_mass_mediator,
    lst_coupling_mediator=lst_coupling_mediator,
    lst_legends=lst_legends,
    mass_neutrino=mass_neutrino,
    output_format='png')

Plot_Diff_Cross_Section_Nu_Nu_S_Channel_Scalar(
    lst_mass_mediator=lst_mass_mediator,
    lst_coupling_mediator=lst_coupling_mediator,
    lst_legends=lst_legends,
    mass_neutrino=mass_neutrino,
    output_format='pdf')

Plot_Diff_Cross_Section_Nu_Nu_S_Channel_Scalar(
    lst_mass_mediator=lst_mass_mediator,
    lst_coupling_mediator=lst_coupling_mediator,
    lst_legends=lst_legends,
    mass_neutrino=mass_neutrino,
    output_format='png')
"""
