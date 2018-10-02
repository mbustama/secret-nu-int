from __future__ import division
import numpy as np
import scipy.integrate as integrate
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

Es=np.power(10., np.linspace(3.,8., 600)) #GeV
Es=np.append(Es, [5e3,  4.5e4, 5e5, 5e7 ])
Es=np.sort(Es)

def L0(E, k=1, gamma=2, E_max=1.0e7):
    return k*np.power(E,-gamma)*np.exp(-E/E_max)

def W(z, rho=5e-5, a=3.4 , b=-0.3 , c=-3.5 , B=5000 , C=9 , eta=-10 , LH=3.89e3, OM=0.3089, OL=0.6911):
	return ((1+z)**(a*eta)+((1+z)/B)**(b*eta)+((1+z)/C)**(c*eta))**(1/eta)

def L(z,E):
    return W(z)*L0(E)

def H(z, H0=70*1e5/3.085677581e24, OM=0.3089, OL=0.6911):
    return H0*np.sqrt(OM*(1.+z)**3. + OL)

#Free streaming case
def fun(z, E, n):
    return (H(z)*n+L(z,E))/(-(1+z)*H(z))

ns=np.array([])

#Free streaming case
for E in Es:
    sol=solve_ivp(fun=lambda z, n: fun(z,E,n),t_span=[4, 0], y0=[4, 0], rtol=1e-8, atol=1e-8)

    ns=np.append(ns,sol.y[1,-1]) #cm^-3*GeV^-1

c=299792458*100  # cm/s
J=ns*c/(4*np.pi)  # cm^-2*GeV^-1*s^-1*sr^-1
flux=Es**2*J  # cm^-2*GeV*s^-1*sr^-1
Nflux=flux/max(flux)   # 10^-8 cm^-2*GeV^2*s^-1*sr^-1

#Free streaming
def F(z, E, n):  # cm^-3*GeV^-1*s^-1
    return (H(z)*n+L(z,E))

#Attenuation
def sigma(E, g, M, m):  # cm^2
    return (g**4/(4*np.pi))*(2*E*m)/((2*E*m-M**2)**2+((M**4*g**4)/(16*np.pi**2)))* 0.389376e-27

def A(z, E, n, g, M, m): # cm^-3*GeV^-1*s^-1
    return c*56*(1+z)**3*sigma(E, g, M, m)*n

#Regeneration
def theta(a, b):
    z=np.array([])
    for i in np.arange(a.size):
        if a[i]-b >= 0:
            z=np.append(z,1) 
        else: 
            z=np.append(z,0)
    return z
        
    
def E_min(Ep, m=1e-10):
    return (m*Ep)/(2*Ep+m)
#
def dsigma(Ep, E, g, M, m): 
    return sigma(Ep, g, M, m)*(1-theta(E,Ep))/(Ep-E_min(Ep))*theta(E,E_min(Ep)) + sigma(m, g, M, m)*(1-theta(E, m))/(m-E_min(m))*theta(E,E_min(m)) 
    
    
#def dsigma(E, g, M, m):  # cm^2/GeV
#    return (g**4/(4*np.pi))*(-8*E**2*m**3+2*m*M**4+((g**4*M**4*m)/(8*np.pi**2)))/(((2*E*m-M**2)**2+((M**4*g**4)/(16*np.pi**2)))**2)*0.389376e-27 \
#             + (g**4/(4*np.pi))*(8*(Ep-E)**2*m**3-2*m*M**4-(g**4*M**4*m/(8*np.pi**2)))/(((2*(Ep-E)*m-M**2)**2+M**4*g**4/(16*np.pi**2))**2)*0.389376e-27
        
#def dsigma2(Ep, E, g, M, m):
#    return (g**4/(4*np.pi))*(8*(Ep-E)**2*m**3-2*m*M**4-(g**4*M**4*m/(8*np.pi**2)))/(((2*(Ep-E)*m-M**2)**2+M**4*g**4/(16*np.pi**2))**2)*0.389376e-27
#
    
def IEn(E, n, g, M, m):
    ints = np.zeros(n.size)
    def integrand(x, E, n, g, M, m):
        return n*dsigma(x, E, g, M, m)
    for i in np.arange(n.size):
        ints[i] = integrate.quad(integrand, E, np.inf, args=(E, n[i],g,M,m))[0]
    return ints

def IEE(E, n, g, M, m):
    ints = np.zeros(E.size)
    def integrand(x, E, n, g, M, m):
        return n*dsigma(x, E, g, M, m)
    for i in np.arange(E.size):
        ints[i] = integrate.quad(integrand, E[i], np.inf, args=(E[i],n[i],g,M,m))[0]
    return ints

def R(z, E, n, g, M, m): # cm^-3*GeV^-1*s^-1
    return c*56*(1+z)**3*IEn(E, n, g, M, m)


#Interaction case
def fun_full(z, E, n, g, M, m):
    return (F(z, E, n)+R(z, E, n, g, M, m))/(-(1+z)*H(z))

def n_full(E, g, M, m):
    sol_array=np.zeros(Es.size)
    for i in np.arange(Es.size):
        print(i+1,"out of",Es.size)
        sol_array[i]=solve_ivp(fun=lambda z, n: fun_full(z, Es[i], n, g, M, m),t_span=[4, 0], y0=[4, 0], rtol=1e-6, atol=1e-6).y[1,-1]

    return sol_array

#g = 0.03
#M = 0.01
#m = 1e-10
#n = (n_full(Es, g, M, m))
#z=0
#
#Fres=F(z, Es, ns)
##print(n)
#Rres =c*56*(1+z)**3*(IEE(Es, n, g, M, m))
##print(Rres)
##Ares = A(z, Es, n, g, M, m)
##print(Ares)
#
#
#plt.loglog(Es, abs(Fres) ,  label = 'w/o interactions')
##plt.loglog(Es, L(z,Es), label='...')
#plt.loglog(Es, abs(Rres),       label = 'Regeneration')
##plt.loglog(Es, abs(Ares),              label = 'Attenuation')
#plt.show()
#plt.legend()
#plt.ylabel(' $dn/dz [GeV^{-1} cm^{-3}]$')
#plt.xlabel('Neutrino energy $E[GeV]$')
#plt.savefig('dndz.png')


#plt.figure()
#plt.plot(Es,sigma(Es,0.3,0.1, m), label='A: M=100 MeV, g=0.3')
#plt.plot(Es,sigma(Es,0.03,0.01, m), label= 'B: M=10 MeV, g=0.03')
#plt.plot(Es,sigma(Es,0.03,0.003, m) , label='C: M=3 MeV, g=0.03')
#plt.plot(Es,sigma(Es,0.01,0.001, m), label='D: M=1 MeV, g=0.01')
#plt.legend()
#plt.xscale('log')
#plt.ylabel(' $\sigma[cm^2]$')
#plt.xlabel('Neutrino energy $E[GeV]$')
#plt.yscale('log')
#plt.ylim(1e-31, 1e-20)
#plt.savefig('CrossSection.png')


#plt.figure()
#plt.plot(Es,dsigma(Es, 0.3, 0.1, 1e-10 ), label='A: M=100 MeV, g=0.3')
#plt.plot(Es,dsigma(Es, 0.03, 0.01, 1e-10 ), label='B: M=10 MeV, g=0.03')
#plt.plot(Es,dsigma(Es, 0.03, 0.003, 1e-10 ), label='C: M=3 MeV, g=0.03')
#plt.plot(Es,dsigma(Es, 0.01, 0.001, 1e-10 ), label='D: M=1 MeV, g=0.01')
#plt.legend()
#plt.xscale('log')
#plt.yscale('log')
#plt.ylabel(' $d\sigma/dE [cm^2/GeV]$')
#plt.xlabel('Neutrino energy $E[GeV]$')
#plt.ylim(1e-34, 1e-23)
#plt.savefig('DifferentialCrossSection.png')


###
plt.figure()
plt.plot(Es,dsigma(1e9,Es, 0.3, 0.1, 1e-10 ), label='A: M=100 MeV, g=0.3')
plt.plot(Es,dsigma(1e9, Es, 0.03, 0.01, 1e-10 ), label='B: M=10 MeV, g=0.03')
plt.plot(Es,dsigma(1e9, Es, 0.03, 0.003, 1e-10 ), label='C: M=3 MeV, g=0.03')
plt.plot(Es,dsigma(1e9, Es, 0.01, 0.001, 1e-10 ), label='D: M=1 MeV, g=0.01')
plt.legend()
plt.xscale('log')
plt.yscale('log')
plt.ylabel(' $d\sigma/dE [cm^2/GeV]$')
plt.xlabel('Neutrino energy $E[GeV]$')
#plt.ylim(1e-34, 1e-23)
plt.savefig('DiffCS.png')



#J_full=n_full(Es,0.03,0.01, 1e-10)*c/(4*np.pi)
#flux_full=Es**2*J_full
#Nflux_full=flux_full/max(flux)

#Free streaming case
#plt.figure()
#plt.plot(Es,Nflux)
#plt.xlabel('Neutrino energy $E[GeV]$')
#plt.ylabel('$ E^2 J \ [10^{-8}GeV \ cm^{-2} s^{-1} sr^{-1}]$')
#plt.xscale('log')
#plt.savefig('Freestreaming.png')
#
#plt.figure()
#plt.plot(Es,Nflux_full)
#plt.xlabel('Neutrino energy $E[GeV]$')
#plt.ylabel('$ E^2 J \ [10^{-8}GeV \ cm^{-2} s^{-1} sr^{-1}]$')
#plt.xscale('log')
#plt.savefig('Interactions.png')

