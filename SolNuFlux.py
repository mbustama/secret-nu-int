from __future__ import division
import numpy as np
import scipy.integrate as integrate
from scipy.integrate import solve_ivp
from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt

Es_npts = 10
log10_Es_min = 2.0
log10_Es_max = 10.0
Es=np.power(10., np.linspace(log10_Es_min, log10_Es_max, Es_npts))#GeV
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


def sigma(E, g, M, m):  # cm^2
    return (g**4/(4*np.pi))*(2*E*m)/((2*E*m-M**2)**2+((M**4*g**4)/(16*np.pi**2)))* 0.389376e-27


#def dsigma(Ep, E, z, g, M, m):  # cm^2/GeV
#        return (g**4/(4*np.pi))*(-8*(E*(1+z))**2*m**3+2*m*M**4+((g**4*M**4*m) \
#                      /(8*np.pi**2)))/(((2*(E*(1+z))*m-M**2)**2+((M**4*g**4)/(16*np.pi**2)))**2)*0.389376e-27 \
#                 + (g**4/(4*np.pi))*(8*(Ep-(E*(1+z)))**2*m**3-2*m*M**4-(g**4*M**4*m/(8*np.pi**2))) \
#                 /(((2*(Ep-E*(1+z))*m-M**2)**2+M**4*g**4/(16*np.pi**2))**2)*0.389376e-27
                 
                 
def dsigma(Ep, E, g, M, m): 
    if Ep < E: 
        return 0
    else: 
        return sigma(Ep, g, M, m)*3/Ep*((E/Ep)**2+(1-(E/Ep))**2)
                 
#log10_E_test_min = 3.0
#log10_E_test_max = 8.0
#E_test_npts = 100
#E_test=np.power(10., np.linspace(log10_E_test_min, log10_E_test_max, E_test_npts))
#E_test=np.append(E_test, [5e3,  4.5e4, 5e5, 5e7 ])
#E_test=np.sort(E_test)
#
#plt.figure()
#plt.plot(E_test,dsigma(1e9,E_test, 0, 0.3, 0.1, 1e-10 ), label='A: M=100 MeV, g=0.3')
#plt.plot(E_test,dsigma(1e9, E_test, 0, 0.03, 0.01, 1e-10 ), label='B: M=10 MeV, g=0.03')
#plt.plot(E_test,dsigma(1e9, E_test, 0, 0.03, 0.003, 1e-10 ), label='C: M=3 MeV, g=0.03')
#plt.plot(E_test,dsigma(1e9, E_test, 0, 0.01, 0.001, 1e-10 ), label='D: M=1 MeV, g=0.01')
#plt.legend()
#plt.xscale('log')
#plt.yscale('log')
#plt.ylabel(' $d\sigma/dE [cm^2/GeV]$')
#plt.xlabel('Neutrino energy $E[GeV]$')  
           

                
c=299792458*100  # cm/s


#Free streaming case
def fun_free(z, E, n):
    return (H(z)*n+L(z,E))/(-(1+z)*H(z))

ns=np.array([])
for E in Es:
    sol_free=solve_ivp(fun=lambda z, n: fun_free(z,E,n),t_span=[4, 0], y0=[4, 0], rtol=1e-8, atol=1e-8)

    ns=np.append(ns,sol_free.y[-1,-1]) #cm^-3*GeV^-1

J=ns*c/(4*np.pi)*Es**2  # cm^-2*GeV*s^-1*sr^-1
flux=J/max(J)   # 10^-8 cm^-2*GeV^2*s^-1*sr^-1

def A(z, E, n, g, M, m): # cm^-3*GeV^-1*s^-1
    return c*56*(1+z)**3*sigma(E*(1+z), g, M, m)*n


def fun(z, E, n, g, M, m):
    return (-A(z, E, n, g, M, m)+(1+z)*L(z,E))/(-(1+z)*H(z))


#def First_guess(E, g, M, m):
#    sol=np.zeros(E.size)
#    for i in np.arange(E.size):
#        sol_return=solve_ivp(fun=lambda z, n: fun(z, E[i], n, g, M, m),t_span=[4.0, 0.0], y0=[0.0], t_eval=[0.0],
#            rtol=1e-8, atol=1e-8, method='LSODA') #method='RK45'
#        sol[i] = sol_return.y[-1,-1]
#    return sol
#
#F1 = First_guess(Es, 0.03, 0.01, 1e-10)         
#interpF1_linear = UnivariateSpline(Es, F1, k=1, ext=0)
#interpF1_quadratic = UnivariateSpline(Es, F1, k=2, ext=0)
#interpF1_cubic = UnivariateSpline(Es, F1, k=3, ext=0)
#
#print(interpF1_linear(0.8e10), interpF1_quadratic(0.8e10), interpF1_cubic(0.8e10))



def Final_sol(E, g, M, m):

    def First_guess(E, g, M, m):
        sol=np.zeros(E.size)
        for i in np.arange(E.size):
            sol_return=solve_ivp(fun=lambda z, n: fun(z, E[i], n, g, M, m),t_span=[4.0, 0.0], y0=[0.0], t_eval=[0.0],
                rtol=1e-8, atol=1e-8, method='LSODA') #method='RK45'
            sol[i] = sol_return.y[-1,-1]
        return sol
    
    F1 = First_guess(E, g, M, m)         
    interpF1_linear = UnivariateSpline(E, F1, k=1, ext=0)
        
    def Re(E, z, g, M, m):
        def integrand(x, E, z, g, M, m):
            return dsigma(x, E*(1+z), g, M, m) *interpF1_linear(x)
        def integrator(E, z, g, M, m):
            return integrate.quad(lambda x: integrand(x, E, z,g, M, m), E*(1+z), 1e9, epsabs=1e-2, epsrel=1e-2)[0]
        return (1+z)*c*56*(1+z)**3*integrator(E, z, g, M ,m)
    
    def fun_full(z, E, n, g, M, m):
        return (-A(z, E, n, g, M, m)+(1+z)*L(z,E)+Re(E, z, g, M, m))/(-(1+z)*H(z))
    
    def Second_guess(E, g, M, m):
        sol_array=np.zeros(E.size)
        for i in np.arange(E.size):
            sol_return=solve_ivp(fun=lambda z, n: fun_full(z, E[i], n, g, M, m),t_span=[4.0, 0.0], y0=[0.0], t_eval=[0.0],
                rtol=1e-8, atol=1e-8, method='LSODA') #method='RK45'
            sol_array[i]=sol_return.y[-1,-1]
        return sol_array 
    
 
    F2 = Second_guess(E, g, M, m)
    interpF2_linear = UnivariateSpline(E, F2, k=1, ext=0)
    
    error = sum(np.subtract(F2,F1)**2)
    print(error)
    count = 0
    while error > 1 and count < 10:
    
        def Re2(E, z, g, M, m):
            def integrand(x, E, z, g, M, m):
                return dsigma(x, E*(1+z), g, M, m) * interpF2_linear(x)
            def integrator(E, z, g, M, m): 
                return integrate.quad(lambda x: integrand(x, E, z,g, M, m), E*(1+z), 1e9, epsabs=1e-2, epsrel=1e-2)[0]
            return (1+z)*c*56*(1+z)**3*integrator(E, z, g, M ,m)
        
        def fun_full2(z, E, n, g, M, m):
            return (-A(z, E, n, g, M, m)+(1+z)*L(z,E)+Re2(E, z, g, M, m))/(-(1+z)*H(z))
        
        def Final_guess(E, g, M, m):
            sol_array=np.zeros(E.size)
            for i in np.arange(E.size):
                sol_return=solve_ivp(fun=lambda z, n: fun_full2(z, E[i], n, g, M, m),t_span=[4.0, 0.0], y0=[0.0], t_eval=[0.0],
                    rtol=1e-8, atol=1e-8, method='LSODA') #method='RK45'
                sol_array[i]=sol_return.y[-1,-1]
            return sol_array 
        
        F3 = Final_guess(E, g, M, m)
        
        error = sum(np.subtract(F3,F2)**2)
        print(error)
        
        F2 = F3
        interpF2_linear = UnivariateSpline(E, F2, k=1, ext=0)
        
        count = count + 1
               
    return Final_guess(E, g, M, m)


Final = Final_sol(Es, 0.03, 0.01, 1e-10) 


#log10_E_test_min = 3.0
#log10_E_test_max = 8.0
#E_test_npts = 100
#E_test=np.power(10., np.linspace(log10_E_test_min, log10_E_test_max, E_test_npts))
#E_test=np.append(E_test, [5e3,  4.5e4, 5e5, 5e7 ])
#E_test=np.sort(E_test)


#plt.figure()
#plt.plot(E_test, interpF1_cubic(E_test))
#plt.xlabel(r'Neutrino energy $E$ [GeV]')
#plt.ylabel(r'Differential number density [cm$^{-3}$ GeV$^{-1}$]')
#plt.xscale('log')
#plt.yscale('log')
#plt.savefig('First_sol.png')

#plt.figure()
#plt.plot(E_test, interpF1_cubic(E_test))
#plt.xlabel(r'Neutrino energy $E$ [GeV]')
#plt.ylabel(r'Differential number density [cm$^{-3}$ GeV$^{-1}$]')
#plt.xscale('log')
#plt.yscale('log')
#plt.savefig('First_sol.png')


#lst_flux_linear = [E_nu*E_nu*interpF1_linear(E_nu) for E_nu in E_test]
#renorm_flux_linear = max(lst_flux_linear)
#lst_flux_linear = [flux/renorm_flux_linear for flux in lst_flux_linear]
#
#lst_flux_quadratic = [E_nu*E_nu*interpF1_quadratic(E_nu) for E_nu in E_test]
#renorm_flux_quadratic = max(lst_flux_quadratic)
#lst_flux_quadratic = [flux/renorm_flux_quadratic for flux in lst_flux_quadratic]
#
#lst_flux_cubic = [E_nu*E_nu*interpF1_cubic(E_nu) for E_nu in E_test]
#renorm_flux_cubic = max(lst_flux_cubic)
#lst_flux_cubic = [flux/renorm_flux_cubic for flux in lst_flux_cubic]
#
#
#plt.rcParams['xtick.labelsize']=26
#plt.rcParams['ytick.labelsize']=26
#plt.rcParams['legend.fontsize']=18
#plt.rcParams['legend.borderpad']=0.4
#plt.rcParams['axes.labelpad']=10
#plt.rcParams['ps.fonttype']=42
#plt.rcParams['pdf.fonttype']=42
#
#fig = plt.figure(figsize=[9,9])
#ax = fig.add_subplot(1,1,1)
#
#ax.plot(E_test, lst_flux_linear, label="Linear interpolation")
#ax.plot(E_test, lst_flux_quadratic, label="Quadratic interpolation")
#ax.plot(E_test, lst_flux_cubic, label="Cubic interpolation")
#ax.set_xlabel(r'Neutrino energy $E$ [GeV]', fontsize=25)
#ax.set_ylabel(r'Neutrino flux [$10^{-8}$ GeV cm$^{-2}$ s$^{-1}$ sr$^{-1}$]', fontsize=25)
#plt.xscale('log')
#ax.set_xlim((10.**log10_E_test_min, 10.**log10_E_test_max))
#ax.set_ylim((0.0, 1.0))
#plt.legend(loc='upper right')
#plt.savefig('First_sol_flux.png', bbox_inches='tight', dpi=300)




