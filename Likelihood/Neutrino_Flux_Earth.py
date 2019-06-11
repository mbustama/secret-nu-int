from __future__ import division
import numpy as np
from scipy.integrate import ode

c=299792458*100  

def nt(z): 
    return 56*(1+z)**3

def L0(energy_nu, z,  gamma, k=1, E_max=1.0e7):
    return k*np.power(energy_nu,-gamma)*np.exp(-energy_nu/E_max)


def W(z, a=3.4 , b=-0.3 , c1=-3.5 , B=5000 , C=9 , eta=-10):
    return ((1+z)**(a*eta)+((1+z)/B)**(b*eta)+((1+z)/C)**(c1*eta))**(1/eta)

def L(z, energy_nu, gamma):
    return W(z)*L0(energy_nu, z, gamma)

def H(z, H0=0.678/(9.777752*3.16*1e16), OM=0.308, OL=0.692):  
    return H0*np.sqrt(OM*(1.+z)**3. + OL)


def sigma(energy_nu, g, M, m=1.e-10):  
    return (g**4/(16*np.pi))*(2*energy_nu*m)/((2*energy_nu*m-M**2)**2+((M**4*g**4)/(16*np.pi**2)))* 0.389379e-27


def Adiabatic_Energy_Losses(z, energy_nu, nu_density, lst_energy_nu, lst_nu_density):
    index = list(lst_energy_nu).index(energy_nu)

    if index < len(lst_energy_nu)-1: 
        diff = (lst_nu_density[index+1]-lst_nu_density[index])/(lst_energy_nu[index+1]-lst_energy_nu[index])
    else: 
        diff = 0
    return H(z)*(nu_density + energy_nu*diff)

def Attenuation(z, energy_nu, nu_density, g, M, m=1.e-10):
    return -c*nt(z)*sigma(energy_nu, g, M, m)*nu_density
    
def Regeneration(z, energy_nu, lst_energy_nu, lst_nu_density, g, M, m=1.e-10):
    regen = 0
    index = list(lst_energy_nu).index(energy_nu)
    
    for j in range (index, len(lst_energy_nu)-1):
        regen += sigma(lst_energy_nu[j], g, M, m)*lst_nu_density[j]*(lst_energy_nu[j+1]-lst_energy_nu[j])
        
    regen=c*nt(z)*regen/(energy_nu)
    
    return regen

def Propagation_Eq(z, nu_density, energy_nu, lst_energy_nu, lst_nu_density, g, M, gamma, m=1.e-10):
    rhs = 0
    
    rhs += Adiabatic_Energy_Losses(z, energy_nu, nu_density, lst_energy_nu, lst_nu_density)
    rhs += L(z, energy_nu, gamma)
    rhs += Attenuation(z, energy_nu, nu_density, g, M, m=1.e-10)
    rhs += Regeneration(z, energy_nu, lst_energy_nu, lst_nu_density, g, M, m=1.e-10)
    
    rhs = rhs/(-(1+z)*H(z))
    
    return rhs

def Neutrino_Flux(g, M, z_min, z_max, E_min, E_max, E_npts, gamma, m=1.e-10):
    
    def Integrand(z, nu_density, energy_nu):
        return Propagation_Eq(z, nu_density, energy_nu, lst_energy_nu, lst_nu_density, g, M, gamma, m=1.e-10)
    
    solver = ode(Integrand, jac=None).set_integrator('dop853', atol=1.e-4, rtol=1.e-4, nsteps=500, max_step=1.e-3, verbosity=1)
    
    lst_energy_nu=np.power(10, np.linspace(E_min, E_max, E_npts))
    
    lst_nu_density = [0.0]*len(lst_energy_nu)
    
    dz = 1.e-1 

    z = z_max

    while (z > z_min):
        lst_nu_density_new = np.zeros(lst_energy_nu.size)
        
        for i in range(len(lst_energy_nu)):
            
            solver.set_initial_value(lst_nu_density[i], z)
            solver.set_f_params(lst_energy_nu[i])
            sol = solver.integrate(solver.t-dz)
            
            lst_nu_density_new[i]=sol
            
        lst_nu_density = [x for x in lst_nu_density_new]

        z = z-dz
        
    save_array = np.zeros([E_npts, 2])
    save_array[:,0] = lst_energy_nu
    save_array[:,1] = lst_nu_density

    #np.savetxt(external_flux_filename, save_array)
 
    return save_array
