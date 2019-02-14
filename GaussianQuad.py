from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
from scipy.integrate import ode
from scipy.interpolate import UnivariateSpline
from scipy.interpolate import interp1d

c=299792458*100  # cm/s

def nt(z): # cm^-3
    return 56*(1+z)**3

def L0(energy_nu, z, k=1, gamma=2, E_max=1.0e7):
    return k*np.power(energy_nu,-gamma)*np.exp(-energy_nu/E_max)


def W(z, a=3.4 , b=-0.3 , c1=-3.5 , B=5000 , C=9 , eta=-10):
    return ((1+z)**(a*eta)+((1+z)/B)**(b*eta)+((1+z)/C)**(c1*eta))**(1/eta)

def L(z, energy_nu):
    return W(z)*L0(energy_nu, z)

def H(z, H0=0.678/(9.777752*3.16*1e16), OM=0.308, OL=0.692):  # s^-1
    return H0*np.sqrt(OM*(1.+z)**3. + OL)


def sigma(energy_nu, g, M, m=1.e-10):  # cm^2
    return (g**4/(16*np.pi))*(2*energy_nu*m)/((2*energy_nu*m-M**2)**2+((M**4*g**4)/(16*np.pi**2)))* 0.389379e-27

def Adiabatic_Energy_Losses(z, energy_nu, nu_density):
    return H(z)*nu_density

def Attenuation(z, energy_nu, nu_density, g, M, m=1.e-10):
    return -c*nt(z)*sigma(energy_nu, g, M, m)*nu_density


    
def Regeneration(z, energy_nu, interp_nu_density, g, M, m=1.e-10):
    
    def Integrand(z, energy_nu_in, energy_nu, interp_nu_density, g, M, m=1.e-10):
        return interp_nu_density(energy_nu_in)*sigma(energy_nu_in, g, M, m=1.e-10)
    
    regen = 0   
    regen = integrate.fixed_quad(lambda energy_nu_in: Integrand(z, energy_nu_in, energy_nu, interp_nu_density, g, M, m=1.e-10), energy_nu, 1e14, n=5)[0]
    
    regen=c*nt(z)*regen/(energy_nu)
    
    return regen

def Propagation_Eq(z, nu_density, energy_nu, lst_energy_nu, lst_nu_density, interp_nu_density, g, M, m=1.e-10):
    rhs = 0
    
    rhs += Adiabatic_Energy_Losses(z, energy_nu, nu_density)
    rhs += L(z, energy_nu)
    rhs += Attenuation(z, energy_nu, nu_density, g, M, m=1.e-10)
    rhs += Regeneration(z, energy_nu, interp_nu_density, g, M, m=1.e-10)
    
    print(Attenuation(z, energy_nu, nu_density, g, M, m=1.e-10)/Regeneration(z, energy_nu, interp_nu_density, g, M, m=1.e-10))
    
    rhs = rhs/(-(1+z)*H(z))

    
    return rhs


def Neutrino_Flux(z_min, z_max, lst_energy_nu, g, M, m=1.e-10):
    
    def Integrand(z, nu_density, energy_nu, interp_nu_density):
        return Propagation_Eq(z, nu_density, energy_nu, lst_energy_nu, lst_nu_density, interp_nu_density, g, M, m=1.e-10)
    
    solver = ode(Integrand, jac=None).set_integrator('dop853', atol=1.e-4, rtol=1.e-4, nsteps=500, max_step=1.e-3, verbosity=1)
    
    lst_nu_density = [0.0]*len(lst_energy_nu)
    

    dz = 1.e-1 

    z = z_max

    while (z > z_min):
        lst_nu_density_new = np.zeros(lst_energy_nu.size)
        #interp_nu_density = interp1d(lst_energy_nu, lst_nu_density, kind='linear', bounds_error=False, fill_value='extrapolate')
        interp_nu_density = UnivariateSpline(lst_energy_nu, lst_nu_density, k=3, ext=0)
       # print(interp_nu_density(lst_energy_nu))
        for i in range(len(lst_energy_nu)):
            
            solver.set_initial_value(lst_nu_density[i], z)
            solver.set_f_params(lst_energy_nu[i], interp_nu_density)
            sol = solver.integrate(solver.t-dz)
            
            lst_nu_density_new[i]=sol
            #print(lst_nu_density_new)
            
        lst_nu_density = [x for x in lst_nu_density_new]
        
        #print(lst_nu_density)
        z = z-dz
   
    return lst_nu_density


log10_E_min = 2
log10_E_max = 8
E_npts = 100
log10_E=np.power(10, np.linspace(log10_E_min, log10_E_max, E_npts))
Flux = Neutrino_Flux(0, 4, log10_E, 0.03 , 0.01, 1e-10)
Flux = [log10_E[i]*log10_E[i]*Flux[i] for i in range(len(log10_E))]
norm = 1/Flux[0]
Flux = [norm*nu_flux for nu_flux in Flux]

plt.figure()
plt.plot(log10_E,Flux)
plt.xlabel('Neutrino energy $E[GeV]$')
plt.ylabel('$ E^2 J \ [10^{-8}GeV \ cm^{-2} s^{-1} sr^{-1}]$')
plt.xscale('log')
#plt.ylim([0.0, 1.5])
plt.xlim([10**3.0, 10**8.0])
#plt.savefig('test.png')
