#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "Mauricio Bustamante"
__email__ = "mbustamante@nbi.ku.dk"


"""
interaction_length.py:
    Routines to calculate neutrino-neutrino interaction length

Created: 2018/09/30 18:25
Last modified: 2018/09/30 18:25
"""


import numpy as np
from pylab import *
from matplotlib import *
import matplotlib.pyplot as plt
import matplotlib as mpl

from cross_section import *
from global_defs import *
from cosmology import *

def Interaction_Length_Nu_Nu(energy_nu, z, mass_mediator, \
    coupling_mediator, mass_neutrino=1.e-10):

    # [cm^2]
    cross_section = Cross_Section_Nu_Nu_S_Channel_Scalar(energy_nu,
        mass_mediator, coupling_mediator, mass_neutrino=mass_neutrino)

    # [cm^{-3}]
    target_spectrum = 56. * pow(1.0+z, 3.0)

    # [cm]
    int_length = 1.0/(cross_section*target_spectrum)

    return int_length # [cm]



def Plot_Interaction_Length_Nu_Nu_Models_A_B_C_D(\
    log10_energy_nu_min = 5.0, log10_energy_nu_max=8.0,
    log10_energy_nu_npts=100, lst_redshift=[0.0],
    lst_mass_mediator=[0.1], lst_coupling_mediator=[0.3], lst_legends=[''],
    lst_labels=[''], mass_neutrino=1.e-10, filename_out='int_length',
    output_format='pdf'):

    print("Plot_Interaction_Length_Nu_Nu_Models_A_B_C_D: "+ \
        "Plotting interaction length nu-nu s-channel, scalar mediator...")

    # Open the plot and format it
    mpl.rcParams['xtick.labelsize']=26
    mpl.rcParams['ytick.labelsize']=26
    mpl.rcParams['legend.fontsize']=18
    mpl.rcParams['legend.borderpad']=0.4
    mpl.rcParams['axes.labelpad']=10
    mpl.rcParams['ps.fonttype']=42
    mpl.rcParams['pdf.fonttype']=42

    # Neutrino energy [GeV]
    lst_log10_energy_nu = \
        np.linspace(log10_energy_nu_min, log10_energy_nu_max, \
            log10_energy_nu_npts) # [GeV]
    lst_energy_nu_base = [10.**x for x in lst_log10_energy_nu] # [GeV]

    lst_colors = ['C0', 'C1', 'C2', 'C3']
    lst_ls = ['-', '--', ':', '-.']

    fig, axes = plt.subplots(len(lst_mass_mediator), 1, figsize=[9,9],
                sharex=True)
    fig.subplots_adjust(hspace=0.05, wspace=0.38) #0.05

    for i in range(len(axes)):

        ax = axes[i]
        mass_mediator = lst_mass_mediator[i]
        coupling_mediator = lst_coupling_mediator[i]

        # Calculate resonance energy
        energy_nu_res = mass_mediator*mass_mediator/2.0/mass_neutrino
        lst_energy_nu = lst_energy_nu_base + [energy_nu_res]
        lst_energy_nu.sort()

        for j, z in enumerate(lst_redshift):

            color = lst_colors[j]
            ls = lst_ls[j]
            legend = lst_legends[j]

            lst_int_length = [Interaction_Length_Nu_Nu(energy_nu, z,
                                mass_mediator, coupling_mediator,
                                mass_neutrino=mass_neutrino) * \
                                conv_cm_to_Mpc*1.e6 \
                                for energy_nu in lst_energy_nu]

            ax.plot(lst_energy_nu, lst_int_length, color=color, ls=ls,
                label=legend)

        ax.tick_params('both', length=10, width=2, which='major')
        ax.tick_params('both', length=5, width=1, which='minor')
        ax.tick_params(axis='both', which='major', pad=10, direction='in')
        ax.tick_params(axis='both', which='minor', pad=10, direction='in')
        ax.tick_params(axis='y', which='minor', left='on')
        ax.tick_params(axis='y', which='minor', right='on')
        ax.tick_params(axis='x', which='minor', bottom='on')
        ax.tick_params(axis='x', which='minor', top='on')
        ax.tick_params(bottom=True, top=True, left=True, right=True)

        if (i == 0):
            ax.annotate( r'xxxxxxx', \
                xy = (0.825,0.76), \
                xycoords='axes fraction', color=None, alpha=0.0, fontsize=21, \
                horizontalalignment='left', rotation=0, zorder=6,
                bbox=dict(boxstyle='round', fc="wheat", alpha=0.5, ec="k") )
            ax.annotate( lst_labels[i], \
                xy = (0.82,0.74), \
                xycoords='axes fraction', color='k', fontsize=21, \
                horizontalalignment='left', rotation=0, zorder=6 )
        else:
            ax.annotate( r'xxxxxxx', \
                xy = (0.825,0.23), \
                xycoords='axes fraction', color=None, alpha=0.0, fontsize=21, \
                horizontalalignment='left', rotation=0, zorder=6,
                bbox=dict(boxstyle='round', fc="wheat", alpha=0.5, ec="k") )
            ax.annotate( lst_labels[i], \
                xy = (0.82,0.21), \
                xycoords='axes fraction', color='k', fontsize=21, \
                horizontalalignment='left', rotation=0, zorder=6 )

        if (i != 3):
            # ax.xaxis.set_visible(False)
            # ax.ravel().set_axis_off()
            ax.get_xaxis().set_ticklabels([])
            # ax.xaxis.set_major_formatter(plt.NullFormatter())

        # ax.set_xticklabels([])
        if (i == 3):
            ax.set_xlabel(r'Neutrino energy $E$ [GeV]', fontsize=25)
        # if (i != 3):
            # ax.set_xlabel('', fontsize=25)
        if (i == 1):
            ax.set_ylabel(r'Interaction length $\nu\nu$ $s$-channel [pc]', fontsize=25)
            ax.yaxis.set_label_coords(-0.13, -0.025)

        if (i == 0):
            ax.legend(loc='lower left', ncol=2)

        # pylab.xlim([10.**log10_energy_nu_min, 10.**log10_energy_nu_max])
        ax.set_xlim([10.**log10_energy_nu_min, 10.**log10_energy_nu_max])
        ax.set_xscale('log')

        # pylab.ylim([1e-31, 1e-20])
        ax.set_yscale('log')

        log10_int_length_min = floor(log10(min(lst_int_length)))
        log10_int_length_max = ceil(log10(max(lst_int_length)))
        lst_log10_int_length = np.linspace(log10_int_length_min,
                                log10_int_length_max,
                                log10_int_length_max-log10_int_length_min+1)
        ax_yticks_major = [10.**log10_int_length
                            for log10_int_length in lst_log10_int_length[::3]]
        ax_yticks_minor = [10.**log10_int_length
                            for log10_int_length in lst_log10_int_length[::1]]
        # print(lst_log10_int_length)

        # if (i == 0):
            # print(min(lst_int_length), max(lst_int_length))
            # ax_yticks_major = np.array([1.e-31, 1.e-30, 1.e-29, 1.e-28, 1.e-27, \
            #                     1.e-26, 1.e-25, 1.e-24, 1.e-23, 1.e-22, 1.e-21, \
            #                     1.e-20])

        ax.yaxis.set_major_locator(matplotlib.ticker.FixedLocator( \
           ax_yticks_major))
        # ax.yaxis.set_minor_locator(matplotlib.ticker.FixedLocator( \
           # ax_yticks_minor))

    # Save the plot
    filename_out = os.getcwd()+'/output/plots/'+filename_out
    pylab.savefig(filename_out+'.'+output_format, \
        bbox_inches='tight', dpi=300)
    plt.close()

    return


###############################################################################
###############################################################################



def Plot_Lookback_Distance_Between_Redshifts( \
    filename_out='distance_between_redshifts',
    log10_delta_z_min=-10., log10_delta_z_max=0., log10_delta_z_npts=10,
    lst_z_base=[0.01], energy_nu_min=1.e3, energy_nu_max=1.e7,
    output_format='pdf', lst_mass_mediator=[0.1], lst_coupling_mediator=[0.3],
    mass_neutrino=1.e-10):

    print("Plot_Lookback_Distance_Between_Redshifts: "+ \
        "Plotting lookback distance between two redshifts...")

    # Open the plot and format it
    mpl.rcParams['xtick.labelsize']=26
    mpl.rcParams['ytick.labelsize']=26
    mpl.rcParams['legend.fontsize']=18
    mpl.rcParams['legend.borderpad']=0.4
    mpl.rcParams['axes.labelpad']=10
    mpl.rcParams['ps.fonttype']=42
    mpl.rcParams['pdf.fonttype']=42

    # Neutrino energy [GeV]
    lst_log10_delta_z = \
        np.linspace(log10_delta_z_min, log10_delta_z_max, \
            log10_delta_z_npts) # [GeV]
    lst_delta_z = [10.**log10_delta_z for log10_delta_z in lst_log10_delta_z]

    lst_colors = ['C0', 'C1', 'C2', 'C3', 'C4']
    lst_ls = ['-', '--', ':', '-.', (0, (1, 1))]

    fig, ax = plt.subplots(1, 1, figsize=[9,9])

    for i, z_base in enumerate(lst_z_base):

        ls = lst_ls[i]

        lst_delta_z_sel = []
        lst_delta_dist = []
        for delta_z in lst_delta_z:
            if (z_base-delta_z >= 0):
                lst_delta_z_sel.append(delta_z)
                dist_diff = \
                    1.e3*(Lookback_Distance(z_base)-Lookback_Distance(z_base-delta_z)) # [Mpc]
                lst_delta_dist.append(dist_diff)

        # Plot the distance difference
        ax.plot(lst_delta_z_sel, lst_delta_dist, color='k', ls=ls, lw=2.0,
            label=r'$z_i = $'+' '+str(z_base), zorder=4)

        # Plot interaction lengths
        # int_length_energy_nu_min = \
        #     Interaction_Length_Nu_Nu(energy_nu_min, z_base, mass_mediator, \
        #     coupling_mediator, mass_neutrino)*conv_cm_to_Mpc
        # int_length_energy_nu_max = \
        #     Interaction_Length_Nu_Nu(energy_nu_max, z_base, mass_mediator, \
        #     coupling_mediator, mass_neutrino)*conv_cm_to_Mpc
        # ax.fill_between([min(lst_delta_z), max(lst_delta_z)],
        #     [int_length_energy_nu_min, int_length_energy_nu_min],
        #     [int_length_energy_nu_max, int_length_energy_nu_max],
        #     edgecolor=None, facecolor=color, alpha=0.5, zorder=2)

        for j in range(len(lst_mass_mediator)):
            color = lst_colors[j]
            mass_mediator = lst_mass_mediator[j] # [GeV]
            coupling_mediator = lst_coupling_mediator[j]
            energy_nu_res = mass_mediator*mass_mediator/2.0/mass_neutrino # [GeV]
            int_length = \
            Interaction_Length_Nu_Nu(energy_nu_res, z_base, mass_mediator, \
                coupling_mediator, mass_neutrino)*conv_cm_to_Mpc
            ax.plot([min(lst_delta_z), max(lst_delta_z)],
                [int_length, int_length], color=color, ls=ls, lw=2.0, zorder=4)

    for i in range(len(lst_mass_mediator)):
        color = lst_colors[i]
        mass_mediator = lst_mass_mediator[i] # [GeV]
        coupling_mediator = lst_coupling_mediator[i]
        energy_nu_res = mass_mediator*mass_mediator/2.0/mass_neutrino # [GeV]
        int_length_top = \
            Interaction_Length_Nu_Nu(energy_nu_res, lst_z_base[0], \
            mass_mediator, coupling_mediator, mass_neutrino)*conv_cm_to_Mpc
        int_length_bottom = \
            Interaction_Length_Nu_Nu(energy_nu_res, lst_z_base[-1], \
            mass_mediator, coupling_mediator, mass_neutrino)*conv_cm_to_Mpc
        ax.fill_between([min(lst_delta_z), max(lst_delta_z)],
            [int_length_top, int_length_top],
            [int_length_bottom, int_length_bottom],
            edgecolor=None, facecolor=color, alpha=0.4, zorder=2)

    ax.annotate( r'$L(z_i)-L(z_f)$', \
        xy = (0.42,0.58), \
        xycoords='axes fraction', color='k', alpha=1.0, fontsize=21, \
        horizontalalignment='left', rotation=47, zorder=6)
    ax.annotate( r'$L_{\nu\nu, \rm int}(E_{\rm res})$ -- Model A', \
        xy = (0.04,0.90), \
        xycoords='axes fraction', color=lst_colors[0], alpha=1.0, fontsize=21, \
        horizontalalignment='left', rotation=0, zorder=6)
    ax.annotate( r'$L_{\nu\nu, \rm int}(E_{\rm res})$ -- Model D', \
        xy = (0.55,0.35), \
        xycoords='axes fraction', color=lst_colors[1], alpha=1.0, fontsize=21, \
        horizontalalignment='left', rotation=0, zorder=6)

    ax.tick_params('both', length=10, width=2, which='major')
    ax.tick_params('both', length=5, width=1, which='minor')
    ax.tick_params(axis='both', which='major', pad=10, direction='in')
    ax.tick_params(axis='both', which='minor', pad=10, direction='in')
    ax.tick_params(axis='y', which='minor', left='on')
    ax.tick_params(axis='y', which='minor', right='on')
    ax.tick_params(axis='x', which='minor', bottom='on')
    ax.tick_params(axis='x', which='minor', top='on')
    ax.tick_params(bottom=True, top=True, left=True, right=True)

    ax.set_xlabel(r'Redshift difference $\Delta z = z_i - z_f$', fontsize=25)
    ax.set_ylabel(r'Lookback distance $L$ between $z_i$ and $z_f$ [Mpc]', fontsize=25)
    ax.legend(loc='lower right', ncol=1)

    # ax.set_xlim([10.**log10_delta_z_min, 10.**log10_delta_z_max])
    ax.set_xlim([10.**log10_delta_z_min, 1.e-3])
    ax.set_ylim([1.e-9, 2.e-2])
    ax.set_xscale('log')
    ax.set_yscale('log')

    ax_xticks_major = [1.e-10, 1.e-9, 1.e-8, 1.e-7, 1.e-6, 1.e-5, 1.e-4, 1.e-3]
    # ax_xticks_minor = [1.e-9, 1.e-7, 1.e-5, 1.e-3]
    ax.xaxis.set_major_locator(matplotlib.ticker.FixedLocator( \
       ax_xticks_major))
    # ax.xaxis.set_minor_locator(matplotlib.ticker.FixedLocator( \
       # ax_xticks_minor))

    # Save the plot
    filename_out = os.getcwd()+'/output/plots/'+filename_out
    pylab.savefig(filename_out+'.'+output_format, \
        bbox_inches='tight', dpi=300)
    plt.close()

    return



###############################################################################
###############################################################################


Plot_Lookback_Distance_Between_Redshifts( \
    filename_out='distance_between_redshifts_model_A',
    log10_delta_z_min=-10.5, log10_delta_z_max=0., log10_delta_z_npts=10,
    lst_z_base=[0.01, 2.0, 4.0, 6.0], output_format='pdf',
    lst_mass_mediator=[0.1, 0.001],
    lst_coupling_mediator=[0.3, 0.01],
    mass_neutrino=1.e-10,
    energy_nu_min=1.e3, energy_nu_max=1.e8)
    # lst_mass_mediator=[0.1, 0.01, 0.003, 0.001],
    # lst_coupling_mediator=[0.3, 0.03, 0.03, 0.01],


###############################################################################
###############################################################################


"""
Plot_Interaction_Length_Nu_Nu_Models_A_B_C_D(\
    log10_energy_nu_min = 3.0, log10_energy_nu_max=8.0,
    log10_energy_nu_npts=1000, lst_redshift=[0.0, 2.0, 4.0, 6.0],
    lst_mass_mediator=[0.1, 0.01, 0.003, 0.001],
    lst_coupling_mediator=[0.3, 0.03, 0.03, 0.01],
    mass_neutrino=1.e-10, filename_out='int_length', output_format='pdf',
    lst_legends=[r'$z = 0$', r'$z = 2$', r'$z = 4$', r'$z = 6$'],
    lst_labels=['Model A', 'Model B', 'Model C', 'Model D'])
quit()
"""
