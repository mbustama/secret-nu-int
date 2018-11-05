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

c=299792458*100  # cm/s

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

def dsigma(Ep, E, g, M, m):
    if Ep < E:
        return 0
    else:
        return sigma(Ep, g, M, m)*3/Ep*((E/Ep)**2+(1-(E/Ep))**2)


def A(z, E, n, g, M, m): # cm^-3*GeV^-1*s^-1
    return c*56*(1+z)**3*sigma(E*(1+z), g, M, m)*n


def fun(z, E, n, g, M, m):
    return (-A(z, E, n, g, M, m)+(1+z)*L(z,E))/(-(1+z)*H(z))
    # return L(z,E)/(-H(z))


def First_guess(E, g, M, m):
    sol=np.zeros(E.size)
    for i in np.arange(E.size):
        sol_return=solve_ivp(fun=lambda z, n: fun(z, E[i], n, g, M, m),t_span=[4.0, 0.0], y0=[0.0], t_eval=[0.0],
            rtol=1e-3, atol=1e-3, method='LSODA') #method='RK45'
        sol[i] = max(sol_return.y[-1,-1], 1.e-200)

    return sol


def Re(E, z, g, M, m, interp):
    def integrand(x, E, z, g, M, m):
        return dsigma(x, E*(1+z), g, M, m) *interp(x)
    def integrator(E, z, g, M, m):
        return integrate.quad(lambda x: integrand(x, E, z,g, M, m), E*(1+z), np.inf, epsabs=1e-3, epsrel=1e-3)[0]
    # >>> We introduced an arbitrary factor of 100 below just to test the term
    return 100.*(1+z)*c*56*(1+z)**3*integrator(E, z, g, M ,m)

def fun_full(z, E, n, g, M, m, interp):
    return (-A(z, E, n, g, M, m)+(1+z)*L(z,E)+Re(E, z, g, M, m, interp))/(-(1+z)*H(z))
    # return ((1+z)*L(z,E)+Re(E, z, g, M, m, interp))/(-(1+z)*H(z))


def final_sol(E, g, M, m):
    F1=First_guess(E, g, M, m)
    interpF1_linear = UnivariateSpline(E, F1, k=1, ext=1)
    # interpF1_linear = UnivariateSpline(E, F1, k=1, ext=0)
    sol_array=np.zeros(E.size)
    count = 0

    while 1:

        print(count)

        for i in np.arange(E.size):
            # sol_return=solve_ivp(fun=lambda z, n: fun_full(z, E[i], n, g, M, m, interpF1_linear),t_span=[4.0, 0.0], y0=[0.0], t_eval=[0.0],
            #     rtol=1e-3, atol=1e-3, method='LSODA') #method='RK45'
            sol_return=solve_ivp(fun=lambda z, n: fun_full(z, E[i], n, g, M, m, interpF1_linear),t_span=[4.0, 0.0], y0=[0.0], t_eval=[0.0], rtol=1e-3, atol=1e-3, method='LSODA') #method='RK45'
            sol_array[i]=max(sol_return.y[-1,-1], 1.e-200)

        print(F1)

        # error = sum(np.subtract(sol_array,F1)**2)
        error = sum([(sol_array[i]/F1[i]-1.0)**2.0 for i in range(len(F1))])
        print(error)

        F1 = np.copy(sol_array)
        interpF1_linear = UnivariateSpline(E, F1, k=1, ext=0)

        count = count + 1

        if error < 1.e-8 or count > 10 :
        # if count > 20:
            break

    return sol_array


# Final = final_sol(Es, 0.03, 0.01, 1e-10)

# """
log10_E_test_min = 2.5
log10_E_test_max = 8.5
E_test_npts = 26
E_test=np.power(10., np.linspace(log10_E_test_min, log10_E_test_max, E_test_npts))
E_test=np.append(E_test, [5e3, 4.5e4, 5e5, 5e7 ])
E_test=np.sort(E_test)

flux = E_test*E_test * final_sol(E_test, 0.03, 0.01, 1e-10)
interp_flux = UnivariateSpline(E_test, flux, k=1, ext=0)
renorm_flux = interp_flux(1.e3)#max(flux)
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

ax.plot(E_test, flux, label="Flux")
ax.set_xlabel(r'Neutrino energy $E$ [GeV]', fontsize=25)
ax.set_ylabel(r'Neutrino flux [$10^{-8}$ GeV cm$^{-2}$ s$^{-1}$ sr$^{-1}$]', fontsize=25)
plt.xscale('log')
ax.set_xlim((10.**3., 10.**8.))
ax.set_ylim((0.0, 2.0))
plt.legend(loc='upper right')
plt.savefig('flux.png', bbox_inches='tight', dpi=300)
# """



