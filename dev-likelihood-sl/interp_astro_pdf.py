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

from pylab import *
from matplotlib import *
import matplotlib as mpl
# import matplotlib.ticker as ticker
from matplotlib import ticker, cm


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
                                    method='linear',
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
                                    method='linear',
                                    bounds_error=False,
                                    fill_value=None))

    return interp_astro_pdf_sh, interp_astro_pdf_tr


def Plot_Contour_Astrophysical_PDF(output_filename, output_format='pdf',
    verbose=0):

    # Open the plot and format it
    mpl.rcParams['xtick.labelsize']=26
    mpl.rcParams['ytick.labelsize']=26
    mpl.rcParams['legend.fontsize']=16
    mpl.rcParams['legend.borderpad']=0.4
    mpl.rcParams['axes.labelpad']=10
    mpl.rcParams['ps.fonttype']=42
    mpl.rcParams['pdf.fonttype']=42

    fig, ax = plt.subplots(1, 1, figsize=[9,9])
    fig.subplots_adjust(hspace=0.05, wspace=0.05)

    # Initialize the astrophysical PDF interpolators for all of the
    # IceCube events
    interp_astro_pdf_sh, interp_astro_pdf_tr = \
        Initialize_Interpolator_Astrophysical_PDF(verbose=verbose)

    gamma = 2.5
    log10_g_min = -3.0
    log10_g_max = 1.0
    log10_g_npts = 100
    log10_M_min = -3.0
    log10_M_max = -1.0
    log10_M_npts = 100
    log10_g = np.linspace(log10_g_min, log10_g_max, log10_g_npts)
    log10_M = np.linspace(log10_M_min, log10_M_max, log10_M_npts)
    log10_M_grid, log10_g_grid = np.meshgrid(log10_M, log10_g)

    event_index = 2
    pdf_astro_sh_grid = [[log10(interp_astro_pdf_sh[event_index]((gamma, lg, lM))) \
                        for lM in log10_M] for lg in log10_g]
    print(pdf_astro_sh_grid)
    cs = ax.contourf(log10_M_grid, log10_g_grid, pdf_astro_sh_grid,
            levels=200,
            # locator=ticker.LogLocator(),
            cmap=cm.PuBu_r)
    cbar = fig.colorbar(cs)

    ####################################################################
    # Formatting
    ####################################################################

    ax.set_xlabel(r'$\log_{10}(M/{\rm GeV})$',
        fontsize=25)
    ax.set_ylabel(r'$\log_{10}(g)$',
        fontsize=25)

    # ax.set_xlim([min(energy), max(energy)])
    # ax.set_ylim(1.e-4, 5.0)

    ax.tick_params('both', length=10, width=2, which='major')
    ax.tick_params('both', length=5, width=1, which='minor')
    ax.tick_params(axis='both', which='major', pad=10, direction='in')
    ax.tick_params(axis='both', which='minor', pad=10, direction='in')
    ax.tick_params(axis='x', which='minor', bottom=True, top=True)
    ax.tick_params(axis='y', which='minor', left=True, right=True)
    ax.tick_params(bottom=True, top=True, left=True, right=True)

    # leg = ax.legend(loc='upper left', ncol=9)

    ####################################################################
    # Save the plot
    ####################################################################

    pylab.savefig(output_filename+'_sh_'+str(event_index)+'.'+output_format,
        bbox_inches='tight',
        dpi=300)
    plt.close()


    return


def Plot_1D_Astrophysical_PDF(topology, event_index,
    output_filename, output_format='pdf', verbose=0):

    # Open the plot and format it
    mpl.rcParams['xtick.labelsize']=26
    mpl.rcParams['ytick.labelsize']=26
    mpl.rcParams['legend.fontsize']=16
    mpl.rcParams['legend.borderpad']=0.4
    mpl.rcParams['axes.labelpad']=10
    mpl.rcParams['ps.fonttype']=42
    mpl.rcParams['pdf.fonttype']=42

    fig, ax = plt.subplots(1, 1, figsize=[9,9])
    fig.subplots_adjust(hspace=0.05, wspace=0.05)

    # Initialize the astrophysical PDF interpolators for all of the
    # IceCube events
    interp_astro_pdf_sh, interp_astro_pdf_tr = \
        Initialize_Interpolator_Astrophysical_PDF(verbose=verbose)

    log10_g_min = -3.0
    log10_g_max = 1.0
    log10_g_npts = 100
    log10_M_min = -4.0
    log10_M_max = 0.0
    log10_M_npts = 100
    log10_g = [-3.0, -2.0, -1.0]#np.linspace(log10_g_min, log10_g_max, log10_g_npts)
    log10_M = np.linspace(log10_M_min, log10_M_max, log10_M_npts)

    color = ['C0', 'C1', 'C2', 'C3']
    style = ['-', '--', ':']
    label = [r'$\log_{10}g = -3$', r'$\log_{10}g = -2$', r'$\log_{10}g = -1$',
            r'$\log_{10}g = 0$']

    gamma = 2.8

    for i, lg in enumerate(log10_g):
        if topology == 'sh':
            pdf_astro = [log10(interp_astro_pdf_sh[event_index]((gamma, lg, lM))) \
                            for lM in log10_M]
        elif topology == 'tr':
            pdf_astro = [log10(interp_astro_pdf_tr[event_index]((gamma, lg, lM))) \
                            for lM in log10_M]
        ax.plot(log10_M, pdf_astro, c=color[i], label=label[i], ls=style[i],
            lw=2.0)

    ####################################################################
    # Annotations
    ####################################################################

    if topology == 'sh':
        ax.annotate( r'Shower '+str(event_index), xy = (0.05, 0.95),
            ha='left', xycoords='axes fraction', color='k', fontsize=25,
            zorder=6)
    elif topology == 'tr':
        ax.annotate( r'Track '+str(event_index), xy = (0.05, 0.95),
            ha='left', xycoords='axes fraction', color='k', fontsize=25,
            zorder=6)
    ax.annotate( r'$\gamma = 2.8$', xy = (0.05, 0.90),
        ha='left', xycoords='axes fraction', color='k', fontsize=25,
        zorder=6)

    ####################################################################
    # Formatting
    ####################################################################

    ax.set_xlabel(r'$\log_{10}(M/{\rm GeV})$',
        fontsize=25)
    ax.set_ylabel(r'Astrophysical partial likelihood $\log_{10}(P_{{\rm ast},i})$',
        fontsize=25)

    ax.set_xlim(log10_M_min, log10_M_max)
    ax.set_ylim(-8.8, -5.2)

    ax.tick_params('both', length=10, width=2, which='major')
    ax.tick_params('both', length=5, width=1, which='minor')
    ax.tick_params(axis='both', which='major', pad=10, direction='in')
    ax.tick_params(axis='both', which='minor', pad=10, direction='in')
    ax.tick_params(axis='x', which='minor', bottom=True, top=True)
    ax.tick_params(axis='y', which='minor', left=True, right=True)
    ax.tick_params(bottom=True, top=True, left=True, right=True)

    leg = ax.legend(loc='lower right', ncol=1)

    ####################################################################
    # Save the plot
    ####################################################################

    if topology == 'sh':
        pylab.savefig(output_filename+'_sh_'+str(event_index)+'.'+output_format,
            bbox_inches='tight',
            dpi=300)
    elif topology == 'tr':
        pylab.savefig(output_filename+'_tr_'+str(event_index)+'.'+output_format,
            bbox_inches='tight',
            dpi=300)
    plt.close()


    return


# for i in range(58):
#     Plot_1D_Astrophysical_PDF('sh', i,
#         os.getcwd()+'/out/plots/astro_pdf/linear/1d/astro_pdf_1d',
#         output_format='png', verbose=1)

# for i in range(22):
#     Plot_1D_Astrophysical_PDF('tr', i,
#         os.getcwd()+'/out/plots/astro_pdf/linear/1d/astro_pdf_1d',
#         output_format='png', verbose=1)


# Plot_Contour_Astrophysical_PDF( \
    #os.getcwd()+'/out/plots/astro_pdf/2d/astro_pdf_2d',
    # output_format='png', verbose=1)



