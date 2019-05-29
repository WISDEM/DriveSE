# -*- coding: utf-8 -*-
"""
drivese_constants.py
Created on Tue May 28 11:17:27 2019

@author: gscott
"""

# Unit conversions

U_KNM_INLB = 8850.745454036  # 1 kN-m = 8850.74577 lb-in
U_IN_M = 0.0254000508001  # 1 in = 0.0254 m

# Physical constants

G_GRAV = 9.81 # m-s^-2

# Material properties

E_CAST_IRON = 169e9 # Young's modulus of cast iron in N/m^2
DENSITY_CAST_IRON = 7100.0 # density of cast iron in kg/m^3

# Shaft material properties - note mix of metric and English units

E_STEEL_LSS = 210e9 # Young's modulus of shaft steel in N/m^2
DENSITY_STEEL_LSS = 7800.0 # density of steel in kg/m^3
SY_STEEL_LSS = 66000  # *self.S_ut/700e6 #66000 #psi # approx tensile strength of steel in psi (about 4.55E5 kPa or 4.55E8 N-m^-2)
