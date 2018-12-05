# -*- coding: utf-8 -*-

#################################################################################
#
# global_defs.py
#
# Contains global definitions, constants, and tools.
#
# Created: 21/03/2016 13:47
# Last modified: 08/05/2018 19:23
#
#################################################################################


# User-defined colors

myOrange = '#FFAA00'
myBlue = '#5e81b5'
myLightBlue = '#DEF0FF'
myPink = '#FF8C66'
myRed = '#EB6235'
myLightRed = '#FFAAAA'
myDarkRed = '9c402b'
myGreen = '#8FB032'
myDarkGreen = '#4A6500'
myLightPurple = '#c681ad'
myPurple = '#ac4283'
myBrightPurple = '#713059'
myYellow = '#F8DB00'

# Fundamental constants
hbar_c_sq = 0.3893793656 * 1.e-27 # GeV^2 mbarn

# Earth's radius

REarth = 6371.0 # [km]

# Average electron fraction inside the Earth

e_fraction_earth_avg = 0.494

# Masses

mass_electron = 0.510998928e-3 # [GeV]
mass_muon = 105.6583715e-3 # [GeV]
mass_proton = 938.272046e-3 # [GeV]
mass_neutron = 939.565379e-3 # [GeV]
mass_isonucleon = ( mass_proton+mass_neutron ) / 2.0 # [GeV]
mass_W = 80.385 # [GeV]
mass_sun = 1.98855e30 # [kg]

# Decay widths

width_W = 2.085 # [GeV]

# Branching ratios

branch_W_to_had = 0.6741
branch_W_to_muon = 0.1063

# Fermi constant

GF = 1.1663787e-5 # [GeV^-2]

# Gravitational constant

GN = 6.70861e-39 # [GeV^-2]

# Light speed

speed_light_km_per_s = 299792458.e-3 # [km s^-1]
speed_light_cm_per_s = 299792458.e2 # [km s^-1]

# Average inelasticity <y> for neutrino-nucleon DIS, above Enu ~ 100 TeV

avg_inelasticity = 0.25

# Density of ice

density_ice = 0.92 # [g cm^-3]

# Molar density of ice

molar_density_ice = 1./18.01528 # [mol g^-1]

# Effective volume of IceCube

Veff = 1.e15 # [cm^3]

# Conversion factors

conv_GeV_to_gram =  1.782661907e-24 # GeV --> g
conv_gram_to_GeV = 1./conv_GeV_to_gram # g --> GeV
conv_gram_to_eV = conv_gram_to_GeV*1.e9 # g --> eV
conv_erg_to_GeV = 624.151 # erg --> GeV
conv_inv_GeV_to_cm = 1.98e-14 # GeV^-1 --> cm
conv_cm_to_inv_GeV = 1./conv_inv_GeV_to_cm # cm --> GeV^-1
conv_cm_to_inv_eV = conv_cm_to_inv_GeV*1.e-9 # cm --> eV^-1
conv_cm3_to_inv_eV3 = conv_cm_to_inv_eV**3.0
conv_GeV_to_inv_s = 2.41799e23 # GeV --> s^-1
conv_Gpc_to_inv_GeV = 2.48876e40 # Gpc --> GeV^-1
conv_Gpc_to_inv_eV = conv_Gpc_to_inv_GeV*1.e-9 # Gpc --> eV^-1
conv_kpc_to_inv_eV = conv_Gpc_to_inv_eV*1.e-6 # kpc --> eV^-1
conv_pc_to_km = 3.0857e13 # pc --> km
conv_kpc_to_cm = conv_pc_to_km*1.e3*1.e5 # kpc --> cm
conv_cm_to_Mpc = 1.e-3/conv_kpc_to_cm
conv_kpc_to_inv_eV = conv_kpc_to_cm*conv_cm_to_inv_eV
conv_inv_eV_to_kpc = 1./conv_kpc_to_inv_eV
conv_inv_eV_to_km = conv_inv_eV_to_kpc*1.e3*conv_pc_to_km
conv_Msun_per_pc3_to_ne_per_cm3 = \
    (mass_sun*1.e3)*conv_gram_to_GeV/mass_proton * \
    pow(conv_pc_to_km*1.e5,-3.0) # Msun pc^-3 --> n_electrons cm^-3

# Avogadro number

NAv = 6.022140857e23 # [mol^-1]

# Random seed

ran_seed = 1234

# Cosmology

h = 0.678
OmegaM = 0.308 # Adimensional matter energy fraction
OmegaL = 0.692 # Adimensional vacuum energy fraction
OmegaK = 0.00 # Adimensional curvature energy fraction
OmegaB = 0.02226/h/h # Adimensional baryon energy fraction [PDG]
H0 = 100.0 * h # Hubble constant [km s^-1 Mpc^-1]
H0_inv_s = h / (9.777752e9*365.*24.*60.*60.) # Hubble constant [s^-1]
Hubble_horizon_0 = speed_light_km_per_s / H0 * 1.e-3 # [Gpc]



