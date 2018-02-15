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


def standardrange(N, N_f, Beta, k_b):
    F_delta = (Beta * (log10(N_f) - log10(N))) + 0.18
    if F_delta >= 2 * k_b:
        F_delta = 0.
    return F_delta


def Ninterp(S, a, b):
    return (S / a)**(1 / b)


def Goodman(S_alt, S_mean, Sut):
    return S_alt / (1 - (S_mean / Sut))


# functions used in bedplate sizing
def midDeflection(totalLength, loadLength, load, E, I):
    defl = load * loadLength**2.0 * \
        (3.0 * totalLength - loadLength) / (6.0 * E * I)
    return defl

    # tip deflection for distributed load


def distDeflection(totalLength, distWeight, E, I):
    defl = distWeight * totalLength**4.0 / (8.0 * E * I)
    return defl


class blade_moment_transform(Component):
    ''' Blade_Moment_Transform class          
          The Blade_Moment_Transform class is used to transform moments from the WISDEM rotor models to driveSE.
    '''
    # variables
    # ensure angles are in radians. Azimuth is 3-element array with blade
    # azimuths; b1, b2, b3 are 3-element arrays for each blade moment (Mx, My,
    # Mz); pitch and cone are floats
    self.add_param('azimuth_angle', val=np.array(
        [0, 2 * pi / 3, 4 * pi / 3]), units='rad', desc='azimuth angles for each blade')
    self.add_param('pitch_angle', val=0.0, units='rad',
                   desc='pitch angle at each blade, assumed same')
    self.add_param('cone_angle', val=0.0, units='rad',
                   desc='cone angle at each blade, assumed same')
    self.add_param('b1', val=np.array([]), units='N*m',
                   desc='moments in x,y,z directions along local blade coordinate system')
    self.add_param('b2', val=np.array([]), units='N*m',
                   desc='moments in x,y,z directions along local blade coordinate system')
    self.add_param('b3', val=np.array([]), units='N*m',
                   desc='moments in x,y,z directions along local blade coordinate system')

    # returns
    self.add_output('Mx', val=0.0, units='N*m',
                    desc='rotor moment in x-direction')
    self.add_output('My', val=0.0, units='N*m',
                    desc='rotor moment in y-direction')
    self.add_output('Mz', val=0.0, units='N*m',
                    desc='rotor moment in z-direction')

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

        # nested function for transformations
        def trans(alpha, con, phi, bMx, bMy, bMz):
            Mx = bMx * cos(con) * cos(alpha) - bMy * (sin(con) * cos(alpha) * sin(phi) - sin(
                alpha) * cos(phi)) + bMz * (sin(con) * cos(alpha) * cos(phi) - sin(alpha) * sin(phi))
            My = bMx * cos(con) * sin(alpha) - bMy * (sin(con) * sin(alpha) * sin(phi) + cos(
                alpha) * cos(phi)) + bMz * (sin(con) * sin(alpha) * cos(phi) + cos(alpha) * sin(phi))
            Mz = bMx * (-sin(alpha)) - bMy * (-cos(alpha) *
                                              sin(phi)) + bMz * (cos(alpha) * cos(phi))
            # print
            # print Mx
            # print My
            # print Mz
            # print
            return [Mx, My, Mz]

        [b1Mx, b1My, b1Mz] = trans(self.pitch_angle, self.cone_angle, self.azimuth_angle[
                                   0], self.b1[0], self.b1[1], self.b1[2])
        [b2Mx, b2My, b2Mz] = trans(self.pitch_angle, self.cone_angle, self.azimuth_angle[
                                   1], self.b2[0], self.b2[1], self.b2[2])
        [b3Mx, b3My, b3Mz] = trans(self.pitch_angle, self.cone_angle, self.azimuth_angle[
                                   2], self.b3[0], self.b3[1], self.b3[2])

        self.Mx = b1Mx + b2Mx + b3Mx
        self.My = b1My + b2My + b3My
        self.Mz = b1Mz + b2Mz + b3Mz

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
    # ensure angles are in radians. Azimuth is 3-element array with blade
    # azimuths; b1, b2, b3 are 3-element arrays for each blade force (Fx, Fy,
    # Fz); pitch and cone are floats
    self.add_param('azimuth_angle', val=np.array(
        [0, 2 * pi / 3, 4 * pi / 3]), units='rad', desc='azimuth angles for each blade')
    self.add_param('pitch_angle', val=0.0, units='rad',
                   desc='pitch angle at each blade, assumed same')
    self.add_param('cone_angle', val=0.0, units='rad',
                   desc='cone angle at each blade, assumed same')
    self.add_param('b1', val=np.array([]), units='N',
                   desc='forces in x,y,z directions along local blade coordinate system')
    self.add_param('b2', val=np.array([]), units='N',
                   desc='forces in x,y,z directions along local blade coordinate system')
    self.add_param('b3', val=np.array([]), units='N',
                   desc='forces in x,y,z directions along local blade coordinate system')

    # returns
    self.add_output('Fx', val=0.0, units='N',
                    desc='rotor force in x-direction')
    self.add_output('Fy', val=0.0, units='N',
                    desc='rotor force in y-direction')
    self.add_output('Fz', val=0.0, units='N',
                    desc='rotor force in z-direction')

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

        # nested function for transformations
        def trans(alpha, con, phi, bFx, bFy, bFz):
            Fx = bFx * cos(con) * cos(alpha) - bFy * (sin(con) * cos(alpha) * sin(phi) - sin(
                alpha) * cos(phi)) + bFz * (sin(con) * cos(alpha) * cos(phi) - sin(alpha) * sin(phi))
            Fy = bFx * cos(con) * sin(alpha) - bFy * (sin(con) * sin(alpha) * sin(phi) + cos(
                alpha) * cos(phi)) + bFz * (sin(con) * sin(alpha) * cos(phi) + cos(alpha) * sin(phi))
            Fz = bFx * (-sin(alpha)) - bFy * (-cos(alpha) *
                                              sin(phi)) + bFz * (cos(alpha) * cos(phi))
            # print
            # print Fx
            # print Fy
            # print Fz
            # print
            return [Fx, Fy, Fz]

        [b1Fx, b1Fy, b1Fz] = trans(self.pitch_angle, self.cone_angle, self.azimuth_angle[
                                   0], self.b1[0], self.b1[1], self.b1[2])
        [b2Fx, b2Fy, b2Fz] = trans(self.pitch_angle, self.cone_angle, self.azimuth_angle[
                                   1], self.b2[0], self.b2[1], self.b2[2])
        [b3Fx, b3Fy, b3Fz] = trans(self.pitch_angle, self.cone_angle, self.azimuth_angle[
                                   2], self.b3[0], self.b3[1], self.b3[2])

        self.Fx = b1Fx + b2Fx + b3Fx
        self.Fy = b1Fy + b2Fy + b3Fy
        self.Fz = b1Fz + b2Fz + b3Fz


# returns FW, mass for bearings without fatigue analysis
def resize_for_bearings(D_shaft, type, deriv):
    # assume low load rating for bearing
    if type == 'CARB':  # p = Fr, so X=1, Y=0
        out = [D_shaft, .2663 * D_shaft + .0435, 1561.4 * D_shaft**2.6007]
        if deriv == True:
            out.extend([1., .2663, 1561.4 * 2.6007 * D_shaft**1.6007])
    elif type == 'SRB':
        out = [D_shaft, .2762 * D_shaft, 876.7 * D_shaft**1.7195]
        if deriv == True:
            out.extend([1., .2762, 876.7 * 1.7195 * D_shaft**0.7195])
    elif type == 'TRB1':
        out = [D_shaft, .0740, 92.863 * D_shaft**.8399]
        if deriv == True:
            out.extend([1., 0., 92.863 * 0.8399 * D_shaft**(0.8399 - 1.)])
    elif type == 'CRB':
        out = [D_shaft, .1136 * D_shaft, 304.19 * D_shaft**1.8885]
        if deriv == True:
            out.extend([1., .1136, 304.19 * 1.8885 * D_shaft**0.8885])
    elif type == 'TRB2':
        out = [D_shaft, .1499 * D_shaft, 543.01 * D_shaft**1.9043]
        if deriv == True:
            out.extend([1., .1499, 543.01 * 1.9043 * D_shaft**.9043])
    elif type == 'RB':  # factors depend on ratio Fa/C0, C0 depends on bearing... TODO: add this functionality
        out = [D_shaft, .0839, 229.47 * D_shaft**1.8036]
        if deriv == True:
            out.extend([1.0, 0.0, 229.47 * 1.8036 * D_shaft**0.8036])

    # shaft diameter, FW, mass. if deriv==True, provides derivatives.
    return out


# fatigue analysis for bearings
def fatigue_for_bearings(D_shaft, F_r, F_a, N_array, life_bearing, type, deriv):
    # deriv is boolean, defines if derivatives are returned
    if type == 'CARB':  # p = Fr, so X=1, Y=0
        if (np.max(F_a)) > 0:
            print '---------------------------------------------------------'
            print "error: axial loads too large for CARB bearing application"
            print '---------------------------------------------------------'
        else:
            e = 1
            Y1 = 0.
            X2 = 1.
            Y2 = 0.
            p = 10. / 3
        C_min = C_calc(F_a, F_r, N_array, p, e, Y1, Y2, X2, life_bearing)
        if C_min > 13980 * D_shaft**1.5602:
            out = [D_shaft, 0.4299 * D_shaft +
                   0.0382, 3682.8 * D_shaft**2.7676]
            if deriv:
                out.extend([1., 0.4299, 3682.8 * 2.7676 * D_shaft**1.7676])
        else:
            out = [D_shaft, .2663 * D_shaft + .0435, 1561.4 * D_shaft**2.6007]
            if deriv:
                out.extend([1., .2663, 1561.4 * 2.6007 * D_shaft**1.6007])

    elif type == 'SRB':
        e = 0.32
        Y1 = 2.1
        X2 = 0.67
        Y2 = 3.1
        p = 10. / 3
        C_min = C_calc(F_a, F_r, N_array, p, e, Y1, Y2, X2, life_bearing)
        if C_min > 13878 * D_shaft**1.0796:
            out = [D_shaft, .4801 * D_shaft, 2688.3 * D_shaft**1.8877]
            if deriv:
                out.extend([1., .4801, 2688.3 * 1.8877 * D_shaft**0.8877])
        else:
            out = [D_shaft, .2762 * D_shaft, 876.7 * D_shaft**1.7195]
            if deriv:
                out.extend([1., .2762, 876.7 * 1.7195 * D_shaft**0.7195])

    elif type == 'TRB1':
        e = .37
        Y1 = 0
        X2 = .4
        Y2 = 1.6
        p = 10. / 3
        C_min = C_calc(F_a, F_r, N_array, p, e, Y1, Y2, X2, life_bearing)
        if C_min > 670 * D_shaft + 1690:
            out = [D_shaft, .1335, 269.83 * D_shaft**.441]
            if deriv:
                out.extend([1., 0., 269.83 * 0.441 * D_shaft**(0.441 - 1.)])
        else:
            out = [D_shaft, .0740, 92.863 * D_shaft**.8399]
            if deriv:
                out.extend([1., 0., 92.863 * 0.8399 * D_shaft**(0.8399 - 1.)])

    elif type == 'CRB':
        if (np.max(F_a) / np.max(F_r) >= .5) or (np.min(F_a) / (np.min(F_r)) >= .5):
            print '--------------------------------------------------------'
            print "error: axial loads too large for CRB bearing application"
            print '--------------------------------------------------------'
        else:
            e = 0.2
            Y1 = 0
            X2 = 0.92
            Y2 = 0.6
            p = 10. / 3
            C_min = C_calc(F_a, F_r, N_array, p, e, Y1, Y2, X2, life_bearing)
            if C_min > 4526.5 * D_shaft**.9556:
                out = [D_shaft, .2603 * D_shaft, 1070.8 * D_shaft**1.8278]
                if deriv:
                    out.extend([1., .2603, 1070.8 * 1.8278 * D_shaft**0.8278])
            else:
                out = [D_shaft, .1136 * D_shaft, 304.19 * D_shaft**1.8885]
                if deriv:
                    out.extend([1., .1136, 304.19 * 1.8885 * D_shaft**0.8885])

    elif type == 'TRB2':
        e = 0.4
        Y1 = 2.5
        X2 = 0.4
        Y2 = 1.75
        p = 10. / 3
        C_min = C_calc(F_a, F_r, N_array, p, e, Y1, Y2, X2, life_bearing)
        if C_min > 6579.9 * D_shaft**.8592:
            out = [D_shaft, .3689 * D_shaft, 1442.6 * D_shaft**1.8932]
            if deriv:
                out.extend([1., .3689, 1442.6 * 1.8932 * D_shaft**.8932])
        else:
            out = [D_shaft, .1499 * D_shaft, 543.01 * D_shaft**1.9043]
            if deriv:
                out.extend([1., .1499, 543.01 * 1.9043 * D_shaft**.9043])

    elif type == 'RB':  # factors depend on ratio Fa/C0, C0 depends on bearing... TODO: add this functionality
        e = 0.4
        Y1 = 1.6
        X2 = 0.75
        Y2 = 2.15
        p = 3.
        C_min = C_calc(F_a, F_r, N_array, p, e, Y1, Y2, X2, life_bearing)
        if C_min > 884.5 * D_shaft**.9964:
            out = [D_shaft, .1571, 646.46 * D_shaft**2.]
            if deriv:
                out.extend([1., 0., 646.46 * 2. * D_shaft])
        else:
            out = [D_shaft, .0839, 229.47 * D_shaft**1.8036]
            if deriv:
                out.extend([1.0, 0.0, 229.47 * 1.8036 * D_shaft**0.8036])

    return out


# calculate required dynamic load rating, C
def C_calc(F_a, F_r, N_array, p, e, Y1, Y2, X2, life_bearing):
    Fa_ref = np.max(F_a)  # used in comparisons Fa/Fr <e
    Fr_ref = np.max(F_r)

    if Fa_ref / Fr_ref <= e:
        P = F_r + Y1 * F_a
    else:
        P = X2 * F_r + Y2 * F_a

    P_eq = ((scp.integrate.simps((P**p), x=N_array, even='avg')) /
            (N_array[-1] - N_array[0]))**(1 / p)
    C_min = P_eq * (life_bearing / 1e6)**(1. / p) / 1000  # kN
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


def get_rotor_mass(machine_rating, deriv):  # if user inputs forces and zero rotor mass
    out = [23.566 * machine_rating]
    if deriv:
        out.extend([23.566])
    return out


def get_L_rb(rotor_diameter, deriv=False):
    out = [0.007835 * rotor_diameter + 0.9642]
    if deriv:
        out.extend([.007835])
    return out


# moments taken to scale approximately with force (rotor mass) and
# distance (L_rb)
def get_My(rotor_mass, L_rb):
    if L_rb == 0:
        # approximate rotor diameter from rotor mass
        L_rb = get_L_rb((rotor_mass + 49089) / 1170.6)
    return 59.7 * rotor_mass * L_rb


def get_Mz(rotor_mass, L_rb):  # moments taken to scale roughly with force (rotor mass) and distance (L_rb)
    if L_rb == 0:
        # approximate rotor diameter from rotor mass
        L_rb = get_L_rb((rotor_mass - 49089) / 1170.6)
    return 53.846 * rotor_mass * L_rb


def sys_print(nace):
    print
    print '-------------Nacelle system model results--------------------'

    print 'Low speed shaft %8.1f kg %6.2f m %6.2f Ixx %6.2f Iyy %6.2f Izz %6.2f CGx %6.2f CGy %6.2f CGz '\
          % (nace.lowSpeedShaft.mass - nace.lowSpeedShaft.shrink_disc_mass, nace.lowSpeedShaft.length, nace.lowSpeedShaft.I[0], nace.lowSpeedShaft.I[1], nace.lowSpeedShaft.I[2], nace.lowSpeedShaft.cm[0], nace.lowSpeedShaft.cm[1], nace.lowSpeedShaft.cm[2])
    print 'LSS diameters:', 'upwind', nace.lowSpeedShaft.diameter1, 'downwind', nace.lowSpeedShaft.diameter2, 'inner', nace.lowSpeedShaft.diameter1 * nace.shaft_ratio
    print 'Main bearing upwind   %8.1f kg. cm %8.1f %8.1f %8.1f' % (nace.mainBearing.mass, nace.mainBearing.cm[0], nace.mainBearing.cm[1], nace.mainBearing.cm[2])
    print 'Second bearing downwind   %8.1f kg. cm %8.1f %8.1f %8.1f' % (nace.secondBearing.mass, nace.secondBearing.cm[0], nace.secondBearing.cm[1], nace.secondBearing.cm[2])
    print 'Gearbox         %8.1f kg %6.2f Ixx %6.2f Iyy %6.2f Izz %6.2f CGx %6.2f CGy %6.2f CGz' \
          % (nace.gearbox.mass, nace.gearbox.I[0], nace.gearbox.I[1], nace.gearbox.I[2], nace.gearbox.cm[0], nace.gearbox.cm[1], nace.gearbox.cm[2])
    print '     gearbox stage masses: %8.1f kg  %8.1f kg %8.1f kg' % (nace.gearbox.stage_masses[0], nace.gearbox.stage_masses[1], nace.gearbox.stage_masses[2])
    print 'High speed shaft & brakes  %8.1f kg %6.2f Ixx %6.2f Iyy %6.2f Izz %6.2f CGx %6.2f CGy %6.2f CGz' \
          % (nace.highSpeedSide.mass, nace.highSpeedSide.I[0], nace.highSpeedSide.I[1], nace.highSpeedSide.I[2], nace.highSpeedSide.cm[0], nace.highSpeedSide.cm[1], nace.highSpeedSide.cm[2])
    print 'Generator       %8.1f kg %6.2f Ixx %6.2f Iyy %6.2f Izz %6.2f CGx %6.2f CGy %6.2f CGz' \
          % (nace.generator.mass, nace.generator.I[0], nace.generator.I[1], nace.generator.I[2], nace.generator.cm[0], nace.generator.cm[1], nace.generator.cm[2])
    print 'Variable speed electronics %8.1f kg' % (nace.above_yaw_massAdder.vs_electronics_mass)
    print 'Transformer mass %8.1f kg' % (nace.transformer.mass)
    print 'Overall mainframe %8.1f kg' % (nace.above_yaw_massAdder.mainframe_mass)
    print 'Bedplate     %8.1f kg %8.1f m length %6.2f Ixx %6.2f Iyy %6.2f Izz %6.2f CGx %6.2f CGy %6.2f CGz' \
        % (nace.bedplate.mass, nace.bedplate.length, nace.bedplate.I[0], nace.bedplate.I[1], nace.bedplate.I[2], nace.bedplate.cm[0], nace.bedplate.cm[1], nace.bedplate.cm[2])
    print 'electrical connections  %8.1f kg' % (nace.above_yaw_massAdder.electrical_mass)
    print 'HVAC system     %8.1f kg' % (nace.above_yaw_massAdder.hvac_mass)
    print 'Nacelle cover:   %8.1f kg %6.2f m Height %6.2f m Width %6.2f m Length' % (nace.above_yaw_massAdder.cover_mass, nace.above_yaw_massAdder.height, nace.above_yaw_massAdder.width, nace.above_yaw_massAdder.length)
    print 'Yaw system      %8.1f kg' % (nace.yawSystem.mass)
    print 'Overall nacelle:  %8.1f kg .cm %6.2f %6.2f %6.2f I %6.2f %6.2f %6.2f' % (nace.nacelle_mass, nace.nacelle_cm[0], nace.nacelle_cm[1], nace.nacelle_cm[2], nace.nacelle_I[0], nace.nacelle_I[1], nace.nacelle_I[2])
    # print
    # print 'Mx:', nace.rotor_torque
    # print 'My:',nace.rotor_bending_moment_y
    # print 'Mz:',nace.rotor_bending_moment_z
    # print
