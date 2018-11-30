from __future__ import division
import numpy as np
import scipy.integrate as integrate
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
    
def Re(E, z, g, M, m, n, eps=1.e-8):  
    def integrand(Ep, E, z, g, M, m, n):
        return dsigma(Ep, E, g, M, m) * n
    def integrator(E, z, g, M, m, n):
        return integrate.quad(lambda Ep: integrand(Ep, E, z, g, M, m, n), E, np.inf, epsabs=eps, epsrel=eps)[0]
    return integrator(E, z, g, M ,m, n)


def rhs_fun(E, z, g, M, m, n):

    t0 = H(z)*n # Energy loss rate due to redshift
    t1 = L(z, E) # Luminosity density
    t2 = - c*nt(z)*sigma(E, g, M, m)*n # Attenuation term
    #t3 = c*nt(z)*Re(E, z, g, M, m, n, eps=1.e-3) # Regeneration term
    t3 = 0.0
    return (t0+t1+t2+t3)/((1.0+z)*H(z))

from decimal import *
getcontext().prec = 30
def Runge_Kutta(E, h, g, M, m, err):
    n_new = np.zeros(E.size).astype('float64')
    n_previous = np.zeros(E.size).astype('float64')
    n_5order = np.zeros(E.size).astype('float64')
    z = 6
#    h_arr = np.zeros(E.size)*h
#    z = np.zeros(E.size)*6-h_arr
#    s = np.zeros(E.size)
    for i in np.arange(E.size):
        count=0
        while z >= 0:

        
            k1 = h* rhs_fun(E[i], z, g, M, m, n_previous[i])
            k2 = h* rhs_fun(E[i], z + h/4, g, M, m, n_previous[i] + k1/4)
            k3 = h* rhs_fun(E[i], z + 3*h/8, g, M, m, n_previous[i] + 3*k1/32 + 9*k2/32)
            k4 = h* rhs_fun(E[i], z + 12*h/13, g, M, m, n_previous[i] + 1932*k1/2197 - 7200*k2/2197 + 7296*k3/2197)
            k5 = h* rhs_fun(E[i], z + h, g, M, m, n_previous[i] + 439*k1/216 - 8*k2 + 3680*k3/513 - 845*k4/4104)
            k6 = h* rhs_fun(E[i], z + h/2, g, M, m, n_previous[i] - 8*k1/27 + 2*k2 - 3544*k3/2565 + 1859*k4/4104 - 11*k5/40)
            
            print(k1,k2,k3,k4,k5,k6)
            
            
            a = Decimal(25*k1/216) + Decimal(1408*k3/2565) + Decimal(2197*k4/4104) - Decimal(k5/5)
            b = Decimal(16*k1/135) + Decimal(6656*k3/12825) + Decimal(28561*k4/56430) - Decimal(9*k5/50) + Decimal(2*k6/55)
            
            n_new[i] = n_previous[i] + 25*k1/216 + 1408*k3/2565 + 2197*k4/4104 - k5/5
            
            n_5order[i] = n_previous[i] + 16*k1/135 + 6656*k3/12825 + 28561*k4/56430 - 9*k5/50 +2*k6/55
            
            print("Inidivial first", 25*k1/216, 1408*k3/2565, 2197*k4/4104, k5/5)
            print("Sum first", 25*k1/216 + 1408*k3/2565 + 2197*k4/4104 - k5/5)
            
            print("Individual second",16*k1/135, 6656*k3/12825, 28561*k4/56430, 9*k5/50, 2*k6/55)
            print("Sum second", 16*k1/135 + 6656*k3/12825 + 28561*k4/56430 - 9*k5/50 +2*k6/55)
            
            print("Explicit difference:", 25*k1/216 + 1408*k3/2565 + 2197*k4/4104 - k5/5 - (16*k1/135 + 6656*k3/12825 + 28561*k4/56430 - 9*k5/50 +2*k6/55))
            print(repr(n_5order[i]), repr(n_new[i]))
            
            print(err, h, b-a)
            if count == 0:
                s = h
            else:
                #s = (err*h/(2*(abs(n_5order[i] - n_new[i]))))**0.25 
                s = (err*h/(2*abs(float(b-a))))**0.25 
                print("Difference:", n_5order[i] - n_new[i])
            print("s*h", s*h)
            
            z = z-h*s
            
            n_previous[i] = np.copy(n_new[i])
            
            print("Count:",count)
            count = count+1
    return n_new




x_min = 3
x_max = 8
npts = 100
test=np.power(10, np.linspace(x_min, x_max, npts))
test=np.append(test, [5e3, 4.5e4, 5e5, 5e7 ])
test=np.sort(test)
flux = c/(4*np.pi)*test*test * Runge_Kutta(test, 5e-5, 0.03, 0.01, 1e-10, 1e-3)
renorm_flux = flux[0]
flux = flux/renorm_flux

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
plt.legend(loc='upper right')
#plt.savefig('RungeKutta4thOrder_200p_R10_2.png', bbox_inches='tight', dpi=300)


