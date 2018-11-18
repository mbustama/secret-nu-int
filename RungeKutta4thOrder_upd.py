from __future__ import division
import numpy as np
import scipy.integrate as integrate
from scipy.integrate import solve_ivp
from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt
import math

c=299792458*100  # cm/s

def nt(z): # cm^-3
    return 56*(1+z)**3

def L0(E, z, k=1, gamma=2, E_max=1.0e7):
    return k*np.power(E,-gamma)*np.exp(-E/E_max)


def W(z, a=3.4 , b=-0.3 , c1=-3.5 , B=5000 , C=9 , eta=-10):
    return ((1+z)**(a*eta)+((1+z)/B)**(b*eta)+((1+z)/C)**(c1*eta))**(1/eta)

def L(z,E):
    return W(z)*L0(E, z)

def H(z, H0=0.678/(9.777752*3.16*1e16), OM=0.308, OL=0.692):  # s^-1
    return H0*np.sqrt(OM*(1.+z)**3. + OL)


def sigma(E, g, M, m):  # cm^2
    return (g**4/(16*np.pi))*(2*E*m)/((2*E*m-M**2)**2+((M**4*g**4)/(16*np.pi**2)))* 0.389379e-27

def dsigma(Ep, E, g, M, m): # cm^2*GeV^-1
    if Ep < E:
        return 0
    else:
        return sigma(Ep, g, M, m)*3/Ep*((E/Ep)**2+(1-(E/Ep))**2)

def dx(x, delta_x):
    return (10**(x+delta_x)-10**(x-delta_x))/(2*delta_x)

def Re(E, z, g, M, m, n, eps=1.e-8):
    def integrand(Ep, E, z, g, M, m, n):
        return dsigma(Ep, E, g, M, m) * n
    def integrator(E, z, g, M, m, n):
        return integrate.quad(lambda Ep: integrand(Ep, E, z, g, M, m, n), E, np.inf, epsabs=eps, epsrel=eps)[0]
    return integrator(E, z, g, M ,m, n)


def rhs_fun(E, z, g, M, m, n):

    t0 = H(z)*n
    t1 = L(z, E)
    t2 = - c*nt(z)*sigma(E, g, M, m)*n
    t3 = c*nt(z)*Re(E, z, g, M, m, n, eps=1.e-3)
    # t3 = 0.0

    return (t0+t1+t2+t3)/((1.0+z)*H(z))


def rhs_fun_alt(E, z, g, M, m, n):

    Ep = (1.0+z)
    t1 = (1.0+z)*L(z, Ep)
    t2 = - c*nt(z)*sigma(Ep, g, M, m)*n
    # t3 = c*nt(z)*Re(E, z, g, M, m, n)
    t3 = 0.0

    return (t1+t2+t3)/(-(1. + z)*H(z))



def Runge_Kutta(E, h, g, M, m):
    n_new = np.zeros(E.size)
    n_previous = np.zeros(E.size)
    # k1 = np.zeros(E.size)
    # k2 = np.zeros(E.size)
    # k3 = np.zeros(E.size)
    # k4 = np.zeros(E.size)
    z = 6-h
    while z >= 0:
        for i in np.arange(E.size):

            # k1[i] = h/((1 + z)*H(z))*(H(z)*n_previous[i] + L(z, E[i]) \
            #      - c*nt(z)*sigma(E[i], g, M, m)*n_previous[i]) #+ c*nt(z)*Re(E[i], z, g, M, m, n_previous[i])

            # k2[i] = h/((1 + (z + 0.5*h))*H(z + 0.5*h))*(H(z + 0.5*h)*(n_previous[i] + 0.5*k1[i]) + L(z + 0.5*h, E[i]) \
            #      - c*nt(z + 0.5*h)*sigma(E[i], g, M, m)*(n_previous[i] + 0.5*k1[i]))  # \
            #   # + c*nt(z + 0.5*h)*Re(E[i], z + 0.5*h, g, M, m, (n_previous[i] + 0.5*k1[i]))

            # k3[i] = h/((1 + (z + 0.5*h))*H(z + 0.5*h))*(H(z + 0.5*h)*(n_previous[i] + 0.5*k2[i]) + L(z + 0.5*h, E[i]) \
            #      - c*nt(z + 0.5*h)*sigma(E[i], g, M, m)*(n_previous[i] + 0.5*k2[i]))  # \
            #   # + c*nt(z + 0.5*h) * Re(E[i], z + 0.5*h, g, M, m, (n_previous[i] + 0.5*k2[i]))

            # k4[i] = h/((1 + (z + h))*H(z + h))*(H(z + h)*(n_previous[i] + k3[i])+ L(z + h, E[i]) \
            #      - c*nt(z + h)*sigma(E[i], g, M, m)*(n_previous[i] + k3[i]))  # \
            #   # + c*nt(z + h)* Re(E[i], z + h, g, M, m, (n_previous[i] + k3[i]))

            k1 = h* rhs_fun(E[i], z, g, M, m, n_previous[i])
            k2 = h* rhs_fun(E[i], z+0.5*h, g, M, m, n_previous[i])
            k3 = h* rhs_fun(E[i], z+0.5*h, g, M, m, n_previous[i]+0.5*k2)
            k4 = h* rhs_fun(E[i], z+h, g, M, m, n_previous[i]+k3)

            # n_new[i] = n_previous[i] + 1/6*k1[i] + 1/3*k2[i] + 1/3*k3[i] + 1/6*k4[i]
            n_new[i] = n_previous[i] + k1/6. + k2/3. + k3/3. + k4/6.

#            if math.isnan(n_new[i]) and z>0 :
#                print(z, E[i], n_new[i])


            n_previous = np.copy(n_new)

            z = z-h

    return n_new

x_min = 3
x_max = 8
npts = 500
#test=np.linspace(x_min, x_max, npts)
#test=np.append(test, [ 5.698970005])
#test=np.sort(test)
test=np.power(10, np.linspace(x_min, x_max, npts))
test=np.append(test, [5e3, 4.5e4, 5e5, 5e7 ])
test=np.sort(test)
flux = c/(4*np.pi)*test*test * Runge_Kutta(test, 1e-5, 0.03, 0.01, 1e-10)
renorm_flux = max(flux)
flux = flux/renorm_flux
print(flux)


plt.rcParams['xtick.labelsize']=26
plt.rcParams['ytick.labelsize']=26
plt.rcParams['legend.fontsize']=18
plt.rcParams['legend.borderpad']=0.4
plt.rcParams['axes.labelpad']=10
plt.rcParams['ps.fonttype']=42
plt.rcParams['pdf.fonttype']=42

fig = plt.figure(figsize=[9,9])
ax = fig.add_subplot(1,1,1)

ax.plot(test, flux, label='B: g = 0.03, M = 0.01 ')
ax.set_xlabel(r'Neutrino energy $E$ [GeV]', fontsize=25)
ax.set_ylabel(r'Neutrino flux [$10^{-8}$ GeV cm$^{-2}$ s$^{-1}$ sr$^{-1}$]', fontsize=25)
plt.xscale('log')
#ax.set_xlim((10.**3., 10.**8.))
#ax.set_ylim((0.0, 2.0))
plt.legend(loc='upper right')
plt.savefig('RungeKutta4thOrder_500p.png', bbox_inches='tight', dpi=300)
