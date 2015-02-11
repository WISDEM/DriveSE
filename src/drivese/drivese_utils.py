"""
driveSE_components.py
New components for hub, low speed shaft, main bearings, gearbox, bedplate and yaw bearings, as well as modified components from NacelleSE

Created by Taylor Parsons 2013. Edited by Taylor Parsons 2014
Copyright (c) NREL. All rights reserved.
"""

import numpy as np
from math import pi, cos, sqrt, radians, sin, exp, log10, log, floor, ceil
import algopy
import scipy as scp

# returns FW, mass for bearings without fatigue analysis
def resize_for_bearings(D_shaft,type):
# assume low load rating for bearing
  if type == 'CARB': #p = Fr, so X=1, Y=0
    return [D_shaft,.2663*D_shaft+.0435,1561.4*D_shaft**2.6007]

  elif type == 'SRB':
    return [D_shaft,.2762*D_shaft,876.7*D_shaft**1.7195]

  elif type == 'TRB1':
    return [D_shaft,.0740,92.863*D_shaft**.8399]

  elif type == 'CRB':
    return [D_shaft,.1136*D_shaft,304.19*D_shaft**1.8885]

  elif type == 'TRB2':
    return [D_shaft,.1499*D_shaft,543.01*D_shaft**1.9043]

  elif type == 'RB': #factors depend on ratio Fa/C0, C0 depends on bearing... TODO: add this functionality
    return [D_shaft,.0839,229.47*D_shaft**1.8036]

# fatigue analysis for bearings
def fatigue_for_bearings(D_shaft,F_r,F_a,N_array,life_bearing,type):

  if type == 'CARB': #p = Fr, so X=1, Y=0
    if (np.max(F_a)) > 0:
      print '---------------------------------------------------------'
      print "error: axial loads too large for CARB bearing application"
      print '---------------------------------------------------------'
    else:
      e = 1
      Y1 = 0.
      X2 = 1.
      Y2 = 0.
      p = 10./3
    C_min = C_calc(F_a,F_r,N_array,p,e,Y1,Y2,life_bearing)
    if C_min > 13980*D_shaft**1.5602:
        return [D_shaft,0.4299*D_shaft+0.0382,3682.8*D_shaft**2.7676]
    else:
        return [D_shaft,.2663*D_shaft+.0435,1561.4*D_shaft**2.6007]

  elif type == 'SRB':
    e = 0.32
    Y1 = 2.1
    X2 = 0.67
    Y2 = 3.1
    p = 10./3
    C_min = C_calc(F_a,F_r,N_array,p,e,Y1,Y2,life_bearing)
    if C_min >  13878*D_shaft**1.0796:
        return [D_shaft,.4801*D_shaft,2688.3*D_shaft**1.8877]
    else:
        return [D_shaft,.2762*D_shaft,876.7*D_shaft**1.7195]

  elif type == 'TRB1':
    e = .37
    Y1 = 0
    X2 = .4
    Y2 = 1.6
    p = 10./3
    C_min = C_calc(F_a,F_r,N_array,p,e,Y1,Y2,life_bearing)
    if C_min >  670*D_shaft+1690:
        return [D_shaft,.1335,269.83*D_shaft**.441]
    else:
        return [D_shaft,.0740,92.863*D_shaft**.8399]

  elif type == 'CRB':
    if (np.max(F_a)/np.max(F_r)>=.5) or (np.min(F_a)/(np.min(F_r))>=.5):
      print '--------------------------------------------------------'
      print "error: axial loads too large for CRB bearing application"
      print '--------------------------------------------------------'
    else:
        e = 0.2
        Y1 = 0
        X2 = 0.92
        Y2 = 0.6
        p = 10./3
        C_min = C_calc(F_a,F_r,N_array,p,e,Y1,Y2,life_bearing)
        if C_min > 4526.5*D_shaft**.9556 :
            return [D_shaft,.2603*D_shaft,1070.8*D_shaft**1.8278]
        else:
            return [D_shaft,.1136*D_shaft,304.19*D_shaft**1.8885]

  elif type == 'TRB2':
    e = 0.4
    Y1 = 2.5
    X2 = 0.4
    Y2 = 1.75
    p = 10./3
    C_min = C_calc(F_a,F_r,N_array,p,e,Y1,Y2,life_bearing)
    if C_min > 6579.9*D_shaft**.8592 :
        return [D_shaft,.3689*D_shaft,1442.6*D_shaft**1.8932]
    else:
        return [D_shaft,.1499*D_shaft,543.01*D_shaft**1.9043]

  elif type == 'RB': #factors depend on ratio Fa/C0, C0 depends on bearing... TODO: add this functionality
    e = 0.4
    Y1 = 1.6
    X2 = 0.75
    Y2 = 2.15
    p = 3.
    C_min = C_calc(F_a,F_r,N_array,p,e,Y1,Y2)
    if C_min > 884.5*D_shaft**.9964 :
        return [D_shaft,.1571,646.46*D_shaft**2.]
    else:
        return [D_shaft,.0839,229.47*D_shaft**1.8036]



#calculate required dynamic load rating, C
def C_calc(F_a,F_r,N_array,p,e,Y1,Y2,life_bearing):
  Fa_ref = np.max(F_a) #used in comparisons Fa/Fr <e
  Fr_ref = np.max(F_r)

  if Fa_ref/Fr_ref <=e:
    P = F_r + Y1*F_a
  else:
    P = X2*F_r + Y2*F_a

  P_eq = ((scp.integrate.simps((P**p),x=N_array,even='avg'))/(N_array[-1]-N_array[0]))**(1/p)
  C_min = P_eq*(life_bearing/1e6)**(1./p)/1000 #kN
  return C_min


# -------------------------------------------------

# def fatigue2_for_bearings(D_shaft,type,Fx,n_Fx,Fy_Fy,n_Fy,Fz_Fz,n_Fz,Fz_My,n_My,Fy_Mz,n_Mz,life_bearing):
# #takes in the effects of individual forces and moments on the radial and axial bearing forces, computes C from sum of bearing life reductions

#   if type == 'CARB': #p = Fr, so X=1, Y=0
#     e = 1
#     Y1 = 0.
#     X2 = 1.
#     Y2 = 0.
#     p = 10./3

#   elif type == 'SRB':
#     e = 0.32
#     Y1 = 2.1
#     X2 = 0.67
#     Y2 = 3.1
#     p = 10./3

#   elif type == 'TRB1':
#     e = .37
#     Y1 = 0
#     X2 = .4
#     Y2 = 1.6
#     p = 10./3

#   elif type == 'CRB':
#     e = 0.2
#     Y1 = 0
#     X2 = 0.92
#     Y2 = 0.6
#     p = 10./3

#   elif type == 'TRB2':
#     e = 0.4
#     Y1 = 2.5
#     X2 = 0.4
#     Y2 = 1.75
#     p = 10./3

#   elif type == 'RB': #factors depend on ratio Fa/C0, C0 depends on bearing... TODO: add this functionality?
#   #idea: select bearing based off of bore, then calculate Fa/C0, see if life is feasible, if not, iterate?
#     e = 0.4
#     Y1 = 1.6
#     X2 = 0.75
#     Y2 = 2.15
#     p = 3.

#   #Dynamic load rating calculation:
#   #reference axial and radial force to find which calculation factor to use-- assume this ratio is relatively consistent across bearing life
#   Fa_ref = np.max(Fx)
#   Fr_ref = ((np.max(Fy_Fy)+np.max(Fy_Mz))**2+(np.max(Fz_Fz)+np.max(Fz_My))**2)**.5

#   if Fa_ref/Fr_ref <=e:
#     #P = F_r + Y1*F_a
#     P_fx =Y1*Fx #equivalent P due to Fx
#     P_fy =Fy_Fy #equivalent P due to Fy... etc
#     P_fz =Fz_Fz
#     P_my =Fz_My
#     P_mz =Fy_Mz
#   else:
#     #P = X2*F_r + Y2*F_a
#     P_fx =Y2*Fx #equivalent P due to Fx
#     P_fy =X2*Fy_Fy #equivalent P due to Fy... etc
#     P_fz =X2*Fz_Fz
#     P_my =X2*Fz_My
#     P_mz =X2*Fy_Mz

#   P_eq = ((scp.integrate.simps((P_fx**p),x=n_Fx,even='avg'))/(np.max(n_Fx)-np.min(n_Fx)))**(1/p)\
#   +((scp.integrate.simps((P_fy**p),x=n_Fy,even='avg'))/(np.max(n_Fy)-np.min(n_Fy)))**(1/p)\
#   +((scp.integrate.simps((P_fz**p),x=n_Fz,even='avg'))/(np.max(n_Fz)-np.min(n_Fz)))**(1/p)\
#   +((scp.integrate.simps((P_my**p),x=n_My,even='avg'))/(np.max(n_My)-np.min(n_My)))**(1/p)\
#   +((scp.integrate.simps((P_mz**p),x=n_Mz,even='avg'))/(np.max(n_Mz)-np.min(n_Mz)))**(1/p)

#   C_min = P_eq*(life_bearing/1e6)**(1./p)/1000 #kN


#   if type == 'CARB': #p = Fr, so X=1, Y=0
#     if C_min > 13980*D_shaft**1.5602:
#         return [D_shaft,0.4299*D_shaft+0.0382,3682.8*D_shaft**2.7676]
#     else:
#         return [D_shaft,.2663*D_shaft+.0435,1561.4*D_shaft**2.6007]

#   elif type == 'SRB':
#     if C_min >  13878*D_shaft**1.0796:
#         return [D_shaft,.4801*D_shaft,2688.3*D_shaft**1.8877]
#     else:
#         return [D_shaft,.2762*D_shaft,876.7*D_shaft**1.7195]

#   elif type == 'TRB1':
#     if C_min >  670*D_shaft+1690:
#         return [D_shaft,.1335,269.83*D_shaft**.441]
#     else:
#         return [D_shaft,.0740,92.863*D_shaft**.8399]

#   elif type == 'CRB':
#     if C_min > 4526.5*D_shaft**.9556 :
#         return [D_shaft,.2603*D_shaft,1070.8*D_shaft**1.8278]
#     else:
#         return [D_shaft,.1136*D_shaft,304.19*D_shaft**1.8885]

#   elif type == 'TRB2':
#     if C_min > 6579.9*D_shaft**.8592 :
#         return [D_shaft,.3689*D_shaft,1442.6*D_shaft**1.8932]
#     else:
#         return [D_shaft,.1499*D_shaft,543.01*D_shaft**1.9043]

#   elif type == 'RB': #factors depend on ratio Fa/C0, C0 depends on bearing... TODO: add this functionality
#     if C_min > 884.5*D_shaft**.9964:
#         return [D_shaft,.1571,646.46*D_shaft**2.]
#     else:
#         return [D_shaft,.0839,229.47*D_shaft**1.8036]

# -------------------------------------------------


def get_rotor_mass(machine_rating): #if user inputs forces and zero rotor mass
    return 23.566*machine_rating


def get_L_rb(rotor_diameter):
    return 0.007835*rotor_diameter+0.9642

def get_My(rotor_mass,L_rb): #moments taken to scale approximately with force (rotor mass) and distance (L_rb)
    if L_rb == 0:
      L_rb = get_L_rb((rotor_mass+49089)/1170.6) #approximate rotor diameter from rotor mass
    return 59.7*rotor_mass*L_rb

def get_Mz(rotor_mass,L_rb): #moments taken to scale roughly with force (rotor mass) and distance (L_rb)
    if L_rb == 0:
      L_rb = get_L_rb((rotor_mass-49089)/1170.6) #approximate rotor diameter from rotor mass
    return 53.846*rotor_mass*L_rb
