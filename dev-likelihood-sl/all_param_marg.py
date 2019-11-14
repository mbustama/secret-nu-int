# Show plots inline, and load main getdist plot module and samples class
from __future__ import print_function
#%matplotlib inline
#%config InlineBackend.figure_format = 'retina'
import sys
# sys.path.insert(0,os.path.realpath(os.path.join(os.getcwd(),'..')))
import os
from getdist import plots, MCSamples
import getdist
# use this *after* importing getdist if you want to use interactive plots
# %matplotlib notebook
import matplotlib.pyplot as plt
# import IPython
import numpy as np
from global_tools import *



names = ["gamma", "log_{10}(g)", "log_{10}(M[GeV])", "N_a", "N_{conv}", "N_{pr}", "N_{mu}"]
labels = ["gamma", "log_{10}(g)", "log_{10}(M[GeV])", "N_a", "N_{conv}", "N_{pr}", "N_{mu}"]

gamma, log10_g, log10_M, N_a, N_conv, N_pr, N_mu, something_1, something_2, \
    something_3 = Read_Data_File(os.getcwd()+'/out/likelihood/ev.dat')

samps = np.array([gamma, log10_g, log10_M, N_a, N_conv, N_pr, N_mu]).T


# names = ["gamma", "N_a", "N_{conv}", "N_{pr}", "N_{mu}"]
# labels = ["gamma", "N_a", "N_{conv}", "N_{pr}", "N_{mu}"]

# gamma, N_a, N_conv, N_pr, N_mu, something_1, something_2, \
#     something_3 = Read_Data_File(os.getcwd()+'/out/likelihood/ev.dat')

# samps = np.array([gamma, N_a, N_conv, N_pr, N_mu]).T


samples = MCSamples(samples = samps, names = names, labels = labels)


plt.figure()
g = plots.get_subplot_plotter()
samples.updateSettings({'contours': [0.68, 0.90], 'fine_bins_2D': 40, 'smooth_scale_2D': 0.6})
g.settings.num_plot_contours = 3
g.triangle_plot(samples, shaded=True)
#plt.plot(-2.37, -1.80, 'X', color = 'black', markersize=14)
# g.export(os.getcwd()+'/out/plots/all_param_marg.pdf')
g.export(os.getcwd()+'/out/plots/all_param_marg.png')
