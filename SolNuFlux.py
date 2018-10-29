from __future__ import division
import numpy as np
import scipy.integrate as integrate
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt

Es=np.power(10., np.linspace(3.,8., 10))#GeV
Es=np.append(Es, [5e3,  4.5e4, 5e5, 5e7 ])
Es=np.sort(Es)

def L0(E, z, k=1, gamma=2, E_max=1.0e7):
    return k*np.power(E*(1+z),-gamma)*np.exp(-E*(1+z)/E_max*(1+z))


def W(z, a=3.4 , b=-0.3 , c1=-3.5 , B=5000 , C=9 , eta=-10):
    return ((1+z)**(a*eta)+((1+z)/B)**(b*eta)+((1+z)/C)**(c1*eta))**(1/eta)

def L(z,E):
    return W(z)*L0(E, z)

def H(z, H0=70*1e5/3.085677581e24, OM=0.3089, OL=0.6911):
    return H0*np.sqrt(OM*(1.+z)**3. + OL)


def sigma(E, z, g, M, m):  # cm^2
    return (g**4/(4*np.pi))*(2*E*(1+z)*m)/((2*E*(1+z)*m-M**2)**2+((M**4*g**4)/(16*np.pi**2)))* 0.389376e-27


def dsigma(Ep, E, z, g, M, m):  # cm^2/GeV
        return (g**4/(4*np.pi))*(-8*(E*(1+z))**2*m**3+2*m*M**4+((g**4*M**4*m)/(8*np.pi**2)))/(((2*(E*(1+z))*m-M**2)**2+((M**4*g**4)/(16*np.pi**2)))**2)*0.389376e-27 \
                 + (g**4/(4*np.pi))*(8*(Ep-(E*(1+z)))**2*m**3-2*m*M**4-(g**4*M**4*m/(8*np.pi**2)))/(((2*(Ep-E*(1+z))*m-M**2)**2+M**4*g**4/(16*np.pi**2))**2)*0.389376e-27
             
c=299792458*100  # cm/s


#Free streaming case
def fun_free(z, E, n):
    return (H(z)*n+L(z,E))/(-(1+z)*H(z))

ns=np.array([])
for E in Es:
    sol_free=solve_ivp(fun=lambda z, n: fun_free(z,E,n),t_span=[4, 0], y0=[4, 0], rtol=1e-8, atol=1e-8)

    ns=np.append(ns,sol_free.y[1,-1]) #cm^-3*GeV^-1

J=ns*c/(4*np.pi)*Es**2  # cm^-2*GeV*s^-1*sr^-1
flux=J/max(J)   # 10^-8 cm^-2*GeV^2*s^-1*sr^-1

def A(z, E, n, g, M, m): # cm^-3*GeV^-1*s^-1
    return c*56*(1+z)**3*sigma(E, z, g, M, m)*n


def fun(z, E, n, g, M, m):
    return (-A(z, E, n, g, M, m)+(1+z)*L(z,E))/(-(1+z)*H(z))


def First_guess(E, g, M, m):
    sol=np.zeros(E.size)
    for i in np.arange(E.size):
        sol[i]=solve_ivp(fun=lambda z, n: fun(z, E[i], n, g, M, m),t_span=[4, 0], y0=[4, 0], rtol=1e-6, atol=1e-6).y[1,-1]
    return sol

F1 = First_guess(Es, 0.03, 0.01, 1e-10)         
interpF1 = UnivariateSpline(Es, F1, k=1)#bounds_error = False, fill_value = 'extrapolate')


print(interpF1(0.8e10))


def Final_sol(E, g, M, m):

    def First_guess(E, g, M, m):
        sol=np.zeros(E.size)
        for i in np.arange(E.size):
            sol[i]=solve_ivp(fun=lambda z, n: fun(z, E[i], n, g, M, m),t_span=[4, 0], y0=[4, 0], rtol=1e-6, atol=1e-6).y[1,-1]
        return sol
    
    F1 = First_guess(E, g, M, m)         
    interpF1 = interp1d(E, F1, bounds_error = False, fill_value = 'extrapolate')
        
    def Re(E, z, g, M, m):
        def integrand(x, E, z, g, M, m):
            return dsigma(x, E, z, g, M, m) *interpF1(x)
        def integrator(E, z, g, M, m): 
            return integrate.quad(lambda x: integrand(x, E, z,g, M, m), E*(1+z), 1e9, epsabs=1e-2, epsrel=1e-2)[0]
        return (1+z)*c*56*(1+z)**3*integrator(E, z, g, M ,m)
    
    def fun_full(z, E, n, g, M, m):
        return (-A(z, E, n, g, M, m)+(1+z)*L(z,E)+Re(E, z, g, M, m))/(-(1+z)*H(z))
    
    def Second_guess(E, g, M, m):
        sol_array=np.zeros(E.size)
        for i in np.arange(E.size):
            sol_array[i]=solve_ivp(fun=lambda z, n: fun_full(z, E[i], n, g, M, m),t_span=[4, 0], y0=[4, 0], rtol=1e-6, atol=1e-6).y[1,-1]
        return sol_array 
    
 
    F2 = Second_guess(E, g, M, m)
    interpF2 = interp1d(Es, F2, bounds_error = False, fill_value = 'extrapolate')
    
    error = sum(np.subtract(F2,F1)**2)
    print(error)
    count = 0
    while error > 1 and count < 10:
    
        def Re2(E, z, g, M, m):
            def integrand(x, E, z, g, M, m):
                return dsigma(x, E, z, g, M, m) * interpF2(x)
            def integrator(E, z, g, M, m): 
                return integrate.quad(lambda x: integrand(x, E, z,g, M, m), E, 1e9, epsabs=1e-2, epsrel=1e-2)[0]
            return (1+z)*c*56*(1+z)**3*integrator(E, z, g, M ,m)
        
        def fun_full2(z, E, n, g, M, m):
            return (-A(z, E, n, g, M, m)+(1+z)*L(z,E)+Re2(E, z, g, M, m))/(-(1+z)*H(z))
        
        def Final_guess(E, g, M, m):
            sol_array=np.zeros(E.size)
            for i in np.arange(E.size):
                sol_array[i]=solve_ivp(fun=lambda z, n: fun_full2(z, E[i], n, g, M, m),t_span=[4, 0], y0=[4, 0], rtol=1e-6, atol=1e-6).y[1,-1]
            return sol_array 
        
        F3 = Final_guess(E, g, M, m)
        
        error = sum(np.subtract(F3,F2)**2)
        print(error)
        
        F2 = F3
        interpF2 = interp1d(E, F2, bounds_error = False, fill_value = 'extrapolate')
        
        count = count + 1
               
    return Final_guess(E, g, M, m)


E_test=np.power(10., np.linspace(2.,10., 100))
E_test=np.append(E_test, [5e3,  4.5e4, 5e5, 5e7 ])
E_test=np.sort(E_test)

plt.figure()
plt.plot(E_test,interpF1(E_test))
plt.xlabel('Neutrino energy $E[GeV]$')
plt.ylabel('$ Differential \ number \ density  \ [ cm^{-3} GeV^{-1} ]$')
plt.xscale('log')
plt.yscale('log')
plt.savefig('First_sol.png')


