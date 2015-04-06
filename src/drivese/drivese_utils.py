"""
driveSE_utilis.py
Utilities and functions used in DriveSE

Created by Taylor Parsons 2014.
Copyright (c) NREL. All rights reserved.
"""

from openmdao.main.api import Component
from openmdao.main.datatypes.api import Float, Array
import numpy as np
from math import pi, cos, sqrt, radians, sin, exp, log10, log, floor, ceil
import algopy
import scipy as scp

class blade_moment_transform(Component): 
    ''' Blade_Moment_Transform class          
          The Blade_Moment_Transform class is used to transform moments from the WISDEM rotor models to driveSE.
    '''
    # variables
    # ensure angles are in radians. Azimuth is 3-element array with blade azimuths; b1, b2, b3 are 3-element arrays for each blade moment (Mx, My, Mz); pitch and cone are floats
    azimuth_angle = Array(np.array([0,2*pi/3,4*pi/3]),iotype='in',units='rad',desc='azimuth angles for each blade')
    pitch_angle = Float(iotype ='in', units = 'rad', desc = 'pitch angle at each blade, assumed same')
    cone_angle = Float(iotype='in', units='rad', desc='cone angle at each blade, assumed same')
    b1 = Array(iotype='in', units='N*m', desc='moments in x,y,z directions along local blade coordinate system')
    b2 = Array(iotype='in', units='N*m', desc='moments in x,y,z directions along local blade coordinate system')
    b3 = Array(iotype='in', units='N*m', desc='moments in x,y,z directions along local blade coordinate system')

    # returns
    Mx = Float(iotype='out',units='N*m', desc='rotor moment in x-direction')
    My = Float(iotype='out',units='N*m', desc='rotor moment in y-direction')
    Mz = Float(iotype='out',units='N*m', desc='rotor moment in z-direction')
    
    def __init__(self):
        
        super(blade_moment_transform, self).__init__()
    
    def execute(self):
        # print "input blade loads:"
        # i=0
        # while i<3:
        #   print 'b1:', self.b1[i]
        #   print 'b2:', self.b2[i]
        #   print 'b3:', self.b3[i]
        #   i+=1
        # print

        #nested function for transformations
        def trans(alpha,con,phi,bMx,bMy,bMz):
            Mx = bMx*cos(con)*cos(alpha) - bMy*(sin(con)*cos(alpha)*sin(phi)-sin(alpha)*cos(phi)) + bMz*(sin(con)*cos(alpha)*cos(phi)-sin(alpha)*sin(phi))
            My = bMx*cos(con)*sin(alpha) - bMy*(sin(con)*sin(alpha)*sin(phi)+cos(alpha)*cos(phi)) + bMz*(sin(con)*sin(alpha)*cos(phi)+cos(alpha)*sin(phi))
            Mz = bMx*(-sin(alpha)) - bMy*(-cos(alpha)*sin(phi)) + bMz*(cos(alpha)*cos(phi))
            # print 
            # print Mx
            # print My
            # print Mz
            # print
            return [Mx,My,Mz]

        C_moment = 1.1 #scaling factor based off of IEC recommendation. Set to operational conditions

        [b1Mx,b1My,b1Mz] = trans(self.pitch_angle,self.cone_angle,self.azimuth_angle[0],self.b1[0],self.b1[1]*C_moment,self.b1[2]*C_moment)
        [b2Mx,b2My,b2Mz] = trans(self.pitch_angle,self.cone_angle,self.azimuth_angle[1],self.b2[0],self.b2[1]*C_moment,self.b2[2]*C_moment)
        [b3Mx,b3My,b3Mz] = trans(self.pitch_angle,self.cone_angle,self.azimuth_angle[2],self.b3[0],self.b3[1]*C_moment,self.b3[2]*C_moment)

        self.Mx = b1Mx+b2Mx+b3Mx
        self.My = b1My+b2My+b3My
        self.Mz = b1Mz+b2Mz+b3Mz

        # print 'azimuth:', self.azimuth_angle/pi*180.
        # print 'pitch:', self.pitch_angle/pi*180.
        # print 'cone:', self.cone_angle/pi*180.

        # print "Total Moments:"
        # print self.Mx
        # print self.My
        # print self.Mz
        # print

class blade_force_transform(Component): 
    ''' Blade_Force_Transform class          
          The Blade_Force_Transform class is used to transform forces from the WISDEM rotor models to driveSE.
    '''
    # variables
    # ensure angles are in radians. Azimuth is 3-element array with blade azimuths; b1, b2, b3 are 3-element arrays for each blade moment (Mx, My, Mz); pitch and cone are floats
    azimuth_angle = Array(np.array([0,2*pi/3,4*pi/3]),iotype='in',units='rad',desc='azimuth angles for each blade')
    pitch_angle = Float(iotype ='in', units = 'rad', desc = 'pitch angle at each blade, assumed same')
    cone_angle = Float(iotype='in', units='rad', desc='cone angle at each blade, assumed same')
    b1 = Array(iotype='in', units='N', desc='forces in x,y,z directions along local blade coordinate system')
    b2 = Array(iotype='in', units='N', desc='forces in x,y,z directions along local blade coordinate system')
    b3 = Array(iotype='in', units='N', desc='forces in x,y,z directions along local blade coordinate system')

    # returns
    Fx = Float(iotype='out',units='N', desc='rotor force in x-direction')
    Fy = Float(iotype='out',units='N', desc='rotor force in y-direction')
    Fz = Float(iotype='out',units='N', desc='rotor force in z-direction, not including rotor mass (accounted for in component models')
    
    def __init__(self):
        
        super(blade_force_transform, self).__init__()
    
    def execute(self):
        # print "input blade loads:"
        # i=0
        # while i<3:
        #   print 'b1:', self.b1[i]
        #   print 'b2:', self.b2[i]
        #   print 'b3:', self.b3[i]
        #   i+=1
        # print

        #nested function for transformations
        def trans(alpha,con,phi,bFx,bFy,bFz):
            Fx = bFx*cos(con)*cos(alpha) - bFy*(sin(con)*cos(alpha)*sin(phi)-sin(alpha)*cos(phi)) + bFz*(sin(con)*cos(alpha)*cos(phi)-sin(alpha)*sin(phi))
            Fy = bFx*cos(con)*sin(alpha) - bFy*(sin(con)*sin(alpha)*sin(phi)+cos(alpha)*cos(phi)) + bFz*(sin(con)*sin(alpha)*cos(phi)+cos(alpha)*sin(phi))
            Fz = bFx*(-sin(alpha)) - bFy*(-cos(alpha)*sin(phi)) + bFz*(cos(alpha)*cos(phi))
            # print 
            # print Fx
            # print Fy
            # print Fz
            # print
            return [Fx,Fy,Fz]

        C_force = 1.3 #scaling factor based off of IEC recommendation. Set to operational conditions

        [b1Fx,b1Fy,b1Fz] = trans(self.pitch_angle,self.cone_angle,self.azimuth_angle[0],self.b1[0],self.b1[1]*C_force,self.b1[2]*C_force)
        [b2Fx,b2Fy,b2Fz] = trans(self.pitch_angle,self.cone_angle,self.azimuth_angle[1],self.b2[0],self.b2[1]*C_force,self.b2[2]*C_force)
        [b3Fx,b3Fy,b3Fz] = trans(self.pitch_angle,self.cone_angle,self.azimuth_angle[2],self.b3[0],self.b3[1]*C_force,self.b3[2]*C_force)

        self.Fx = b1Fx+b2Fx+b3Fx
        self.Fy = b1Fy+b2Fy+b3Fy
        self.Fz = b1Fz+b2Fz+b3Fz

        # print 'azimuth:', self.azimuth_angle/pi*180.
        # print 'pitch:', self.pitch_angle/pi*180.
        # print 'cone:', self.cone_angle/pi*180.

        # print "Total forces:"
        # print self.Fx
        # print self.Fy
        # print self.Fz
        # print


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
