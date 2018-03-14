"""
driveSE_components.py
New components for low speed shaft, main bearings, gearbox, bedplate and yaw bearings, as well as modified components from NacelleSE

Created by Ryan King 2013. Edited by Taylor Parsons 2014
Copyright (c) NREL. All rights reserved.
"""

from openmdao.api import Component

import numpy as np
from math import pi, cos, sqrt, radians, sin, exp, log10, log, floor, ceil
import algopy
import scipy as scp
import scipy.optimize as opt
from scipy import integrate

from drivese_utils import fatigue_for_bearings, resize_for_bearings, get_rotor_mass, get_L_rb, get_My, get_Mz

#-------------------------------------------------------------------------


class Bearing_drive(Component):
    ''' MainBearings class
          The MainBearings class is used to represent the main bearing components of a wind turbine drivetrain. It contains two subcomponents (main bearing and second bearing) which also inherit from the SubComponent class.
          It contains the general properties for a wind turbine component as well as additional design load and dimentional attributes as listed below.
          It contains an update method to determine the mass, mass properties, and dimensions of the component.
    '''

    def __init__(self):
        super(Bearing_drive, self).__init__()

        # variables
        self.add_param('bearing_type', val='', desc='Main bearing type: CARB, TRB1 or SRB')
        self.add_param('bearing_mass', val=0.0, units='kg', desc='bearing mass from LSS model')
        self.add_param('lss_diameter', val=0.0, units='m', desc='lss outer diameter at main bearing')
        self.add_param('lss_design_torque', val=0.0, units='N*m', desc='lss design torque')
        self.add_param('rotor_diameter', val=0.0, units='m', desc='rotor diameter')
        self.add_param('location', val=np.array([0., 0., 0.]), units='m', desc='x,y,z location from shaft model')

        # returns
        self.add_output('mass', val=0.0, units='kg', desc='overall component mass')
        self.add_output('cm', val=np.array([0.0, 0.0, 0.0]), desc='center of mass of the component in [x,y,z] for an arbitrary coordinate system')
        self.add_output('I', val=np.array([0.0, 0.0, 0.0]), desc=' moments of Inertia for the component [Ixx, Iyy, Izz] around its center of mass')

    def solve_nonlinear(self, params, unknowns, resids):
        self.mass = self.bearing_mass
        self.mass += self.mass * (8000.0 / 2700.0)  # add housing weight


class MainBearing_drive(Bearing_drive):
    ''' MainBearings class
          The MainBearings class is used to represent the main bearing components of a wind turbine drivetrain. It contains two subcomponents (main bearing and second bearing) which also inherit from the SubComponent class.
          It contains the general properties for a wind turbine component as well as additional design load and dimentional attributes as listed below.
          It contains an update method to determine the mass, mass properties, and dimensions of the component.
    '''

    def __init__(self):
        super(MainBearing_drive, self).__init__()

    def solve_nonlinear(self, params, unknowns, resids):

        super(MainBearing_drive, self).execute()

        # calculate mass properties
        inDiam = self.lss_diameter
        depth = (inDiam * 1.5)

        if self.location[0] != 0.0:
            self.cm = self.location

        else:
            cmMB = np.array([0.0, 0.0, 0.0])
            cmMB = ([- (0.035 * self.rotor_diameter),  0.0, 0.025 * self.rotor_diameter])
            self.cm = cmMB

        b1I0 = (self.mass * inDiam ** 2) / 4.0
        self.I = ([b1I0, b1I0 / 2.0, b1I0 / 2.0])

#-------------------------------------------------------------------------


class SecondBearing_drive(Bearing_drive):
    ''' MainBearings class
          The MainBearings class is used to represent the main bearing components of a wind turbine drivetrain. It contains two subcomponents (main bearing and second bearing) which also inherit from the SubComponent class.
          It contains the general properties for a wind turbine component as well as additional design load and dimentional attributes as listed below.
          It contains an update method to determine the mass, mass properties, and dimensions of the component.
    '''

    def __init__(self):
        ''' Initializes second bearing component
        '''

        super(SecondBearing_drive, self).__init__()

    def solve_nonlinear(self, params, unknowns, resids):

        super(SecondBearing_drive, self).execute()

        # calculate mass properties
        inDiam = self.lss_diameter
        depth = (inDiam * 1.5)

        if self.mass > 0 and self.location[0] != 0.0:
            self.cm = self.location
        else:
            self.cm = np.array([0, 0, 0])
            self.mass = 0.

        b2I0 = (self.mass * inDiam ** 2) / 4.0
        self.I = ([b2I0, b2I0 / 2.0, b2I0 / 2.0])

#-------------------------------------------------------------------------


class Gearbox_drive(Component):
    ''' Gearbox class
          The Gearbox class is used to represent the gearbox component of a wind turbine drivetrain.
          It contains the general properties for a wind turbine component as well as additional design load and dimentional attributes as listed below.
          It contains an update method to determine the mass, mass properties, and dimensions of the component.
    '''

    def __init__(self):
        super(Gearbox_drive, self).__init__()

        # variables
        self.add_param('gear_ratio', val=0.0 desc='overall gearbox speedup ratio')
        self.add_param('Np', val=np.array([0.0, 0.0, 0.0, ]), desc='number of planets in each stage')
        self.add_param('rotor_speed', val=0.0 desc='rotor rpm at rated power')
        self.add_param('rotor_diameter', val=0.0 desc='rotor diameter')
        self.add_param('rotor_torque', val=0.0, units='N*m', desc='rotor torque at rated power')
        self.add_param('cm_input', val=0.00, units='m', desc='gearbox position along x-axis')

        # parameters
        self.add_param('#name', val=desc='gearbox name')
        self.add_param('gear_configuration', val=desc='string that represents the configuration of the gearbox (stage number and types)')
        self.add_param('#eff', val=0.0 desc='drivetrain efficiency')
        self.add_param('ratio_type', val=desc='optimal or empirical stage ratios')
        self.add_param('shaft_type', val=desc='normal or short shaft length')

        # outputs
        self.add_output('stage_masses', val=np.array([0.0, 0.0, 0.0, 0.0]), units='kg', desc='individual gearbox stage masses')
        self.add_output('mass', val=0.0, units='kg', desc='overall component mass')
        self.add_output('cm', val=np.array([0.0, 0.0, 0.0]), desc='center of mass of the component in [x,y,z] for an arbitrary coordinate system')
        self.add_output('I', val=np.array([0.0, 0.0, 0.0]), desc=' moments of Inertia for the component [Ixx, Iyy, Izz] around its center of mass')
        self.add_output('length', val=0.0, units='m', desc='gearbox length')
        self.add_output('height', val=0.0, units='m', desc='gearbox height')
        self.add_output('diameter', val=0.0, units='m', desc='gearbox diameter')

    def solve_nonlinear(self, params, unknowns, resids):

        self.stageRatio = np.zeros([3, 1])

        # filled in when ebxWeightEst is called
        self.stageTorque = np.zeros([len(self.stageRatio), 1])
        # filled in when ebxWeightEst is called
        self.stageMass = np.zeros([len(self.stageRatio), 1])
        self.stageType = self.stageTypeCalc(self.gear_configuration)
        # print self.gear_ratio
        # print self.Np
        # print self.ratio_type
        # print self.gear_configuration
        self.stageRatio = self.stageRatioCalc(
            self.gear_ratio, self.Np, self.ratio_type, self.gear_configuration)
        # print self.stageRatio

        m = self.gbxWeightEst(self.gear_configuration, self.gear_ratio,            self.Np, self.ratio_type, self.shaft_type, self.rotor_torque)
        self.mass = float(m)
        self.stage_masses = self.stageMass
        # calculate mass properties

        self.length = (0.012 * self.rotor_diameter)
        self.height = (0.015 * self.rotor_diameter)
        self.diameter = (0.75 * self.height)

        cm0 = self.cm_input
        cm1 = 0.0
        # TODO validate or adjust factor. origin is modified to be above
        # bedplate top
        cm2 = 0.4 * self.height
        self.cm = np.array([cm0, cm1, cm2])

        I0 = self.mass * (self.diameter ** 2) / 8 + \
                          (self.mass / 2) * (self.height ** 2) / 8
        I1 = self.mass * (0.5 * (self.diameter ** 2) + (2 / 3)        * (self.length ** 2) + 0.25 * (self.height ** 2)) / 8
        I2 = I1
        self.I = np.array([I0, I1, I2])

        '''def rotor_torque():
            tq = self.gbxPower*1000 / self.eff / \
                (self.rotor_speed * (pi / 30.0))
            return tq
        '''

    def stageTypeCalc(self, config):
        temp = []
        for character in config:
                if character == 'e':
                    temp.append(2)
                if character == 'p':
                    temp.append(1)
        return temp

    def stageMassCalc(self, indStageRatio, indNp, indStageType):
        '''
        Computes the mass of an individual gearbox stage.

        Parameters
        ----------
        indStageRatio : str
          Speedup ratio of the individual stage in question.
        indNp : int
          Number of planets for the individual stage.
        indStageType : int
          Type of gear.  Use '1' for parallel and '2' for epicyclic.
        '''

        # Application factor to include ring/housing/carrier weight
        Kr = 0.4
        Kgamma = 1.1

        if indNp == 3:
            Kgamma = 1.1
        elif indNp == 4:
            Kgamma = 1.1
        elif indNp == 5:
            Kgamma = 1.35

        if indStageType == 1:
            indStageMass = 1.0 + indStageRatio + \
                indStageRatio**2 + (1.0 / indStageRatio)

        elif indStageType == 2:
            sunRatio = 0.5 * indStageRatio - 1.0
            indStageMass = Kgamma * ((1 / indNp) + (1 / (indNp * sunRatio)) + sunRatio + sunRatio**2 + Kr * (
                (indStageRatio - 1)**2) / indNp + Kr * ((indStageRatio - 1)**2) / (indNp * sunRatio))

        return indStageMass

    def gbxWeightEst(self, config, overallRatio, Np, ratio_type, shaft_type, torque):
        '''
        Computes the gearbox weight based on a surface durability criteria.
        '''

        ## Define Application Factors ##
        # Application factor for weight estimate
        Ka = 0.6
        Kshaft = 0.0
        Kfact = 0.0

        # K factor for pitting analysis
        if self.rotor_torque < 200.0:
            Kfact = 850.0
        elif self.rotor_torque < 700.0:
            Kfact = 950.0
        else:
            Kfact = 1100.0

        # Unit conversion from Nm to inlb and vice-versa
        Kunit = 8.029

        # Shaft length factor
        if self.shaft_type == 'normal':
            Kshaft = 1.0
        elif self.shaft_type == 'short':
            Kshaft = 1.25

        # Individual stage torques
        torqueTemp = self.rotor_torque
        for s in range(len(self.stageRatio)):
            # print torqueTemp
            # print self.stageRatio[s]
            self.stageTorque[s] = torqueTemp / self.stageRatio[s]
            torqueTemp = self.stageTorque[s]
            self.stageMass[s] = Kunit * Ka / Kfact * self.stageTorque[s] * \
                self.stageMassCalc(self.stageRatio[s], self.Np[                 s], self.stageType[s])

        gbxWeight = (sum(self.stageMass)) * Kshaft

        return gbxWeight

    def stageRatioCalc(self, overallRatio, Np, ratio_type, config):
        '''
        Calculates individual stage ratios using either empirical relationships from the Sunderland model or a SciPy constrained optimization routine.
        '''

        K_r = 0

        # Assumes we can model everything w/Sunderland model to estimate speed
        # ratio
        if ratio_type == 'empirical':
            if config == 'p':
                x = [overallRatio]
            if config == 'e':
                x = [overallRatio]
            elif config == 'pp':
                x = [overallRatio**0.5, overallRatio**0.5]
            elif config == 'ep':
                x = [overallRatio / 2.5, 2.5]
            elif config == 'ee':
                x = [overallRatio**0.5, overallRatio**0.5]
            elif config == 'eep':
                x = [(overallRatio / 3)**0.5, (overallRatio / 3)**0.5, 3]
            elif config == 'epp':
                x = [overallRatio**(1.0 / 3.0), overallRatio **
                                    (1.0 / 3.0), overallRatio**(1.0 / 3.0)]
            elif config == 'eee':
                x = [overallRatio**(1.0 / 3.0), overallRatio **
                                    (1.0 / 3.0), overallRatio**(1.0 / 3.0)]
            elif config == 'ppp':
                x = [overallRatio**(1.0 / 3.0), overallRatio **
                                    (1.0 / 3.0), overallRatio**(1.0 / 3.0)]

        elif ratio_type == 'optimal':
            x = np.zeros([3, 1])

            if config == 'eep':
                x0 = [overallRatio**(1.0 / 3.0), overallRatio **
                                     (1.0 / 3.0), overallRatio**(1.0 / 3.0)]
                B_1 = Np[0]
                B_2 = Np[1]
                K_r1 = 0
                K_r2 = 0  # 2nd stage structure weight coefficient

                def volume(x):
                    return (1.0 / (x[0])) * ((1.0 / B_1) + (1.0 / (B_1 * ((x[0] / 2.0) - 1.0))) + (x[0] / 2.0 - 1.0) +
                    (x[0] / 2.0 - 1)**2 + K_r1 * ((x[0] - 1.0)**2) / B_1 + K_r1 * ((x[0] - 1.0)**2) / (B_1 * (x[0] / 2.0 - 1.0))) + \
                    (1.0 / (x[0] * x[1])) * ((1.0 / B_2) + (1 / (B_2 * ((x[1] / 2.0) - 1.0))) + (x[1] / 2.0 - 1.0) + (x[1] / 2.0 - 1.0)**2.0 + K_r2 * ((x[1] - 1.0)**2.0) / B_2 +
                     K_r2 * ((x[1] - 1.0)**2.0) / (B_2 * (x[1] / 2.0 - 1.0))) + (1.0 / (x[0] * x[1] * x[2])) * (1.0 + (1.0 / x[2]) + x[2] + x[2]**2)

                def constr1(x, overallRatio):
                    return x[0] * x[1] * x[2] - overallRatio

                def constr2(x, overallRatio):
                    return overallRatio - x[0] * x[1] * x[2]

                x = opt.fmin_cobyla(volume, x0, [constr1, constr2], consargs=[
                                    overallRatio], rhoend=1e-7, iprint=0)

            elif config == 'eep_3':
                # fixes last stage ratio at 3
                x0 = [overallRatio**(1.0 / 3.0), overallRatio **
                                     (1.0 / 3.0), overallRatio**(1.0 / 3.0)]
                B_1 = Np[0]
                B_2 = Np[1]
                K_r1 = 0
                K_r2 = 0.8  # 2nd stage structure weight coefficient

                def volume(x):
                    return (1.0 / (x[0])) * ((1.0 / B_1) + (1.0 / (B_1 * ((x[0] / 2.0) - 1.0))) + (x[0] / 2.0 - 1.0) + (x[0] / 2.0 - 1)**2 + K_r1 * ((x[0] - 1.0)**2) / B_1 + K_r1 * ((x[0] - 1.0)**2) / (B_1 * (x[0] / 2.0 - 1.0))) + (1.0 / (x[0] * x[1])) * ((1.0 / B_2) + (1 / (B_2 * ((x[1] / 2.0) - 1.0))) + (x[1] / 2.0 - 1.0) + (x[1] / 2.0 - 1.0)**2.0 + K_r2 * ((x[1] - 1.0)**2.0) / B_2 + K_r2 * ((x[1] - 1.0)**2.0) / (B_2 * (x[1] / 2.0 - 1.0))) + (1.0 / (x[0] * x[1] * x[2])) * (1.0 + (1.0 / x[2]) + x[2] + x[2]**2)

                def constr1(x, overallRatio):
                    return x[0] * x[1] * x[2] - overallRatio

                def constr2(x, overallRatio):
                    return overallRatio - x[0] * x[1] * x[2]

                def constr3(x, overallRatio):
                    return x[2] - 3.0

                def constr4(x, overallRatio):
                    return 3.0 - x[2]

                x = opt.fmin_cobyla(volume, x0, [constr1, constr2, constr3, constr4], consargs=[
                                    overallRatio], rhoend=1e-7, iprint=0)

            elif config == 'eep_2':
                # fixes final stage ratio at 2
                x0 = [overallRatio**(1.0 / 3.0), overallRatio **
                                     (1.0 / 3.0), overallRatio**(1.0 / 3.0)]
                B_1 = Np[0]
                B_2 = Np[1]
                K_r1 = 0
                K_r2 = 1.6  # 2nd stage structure weight coefficient

                def volume(x):
                    return (1.0 / (x[0])) * ((1.0 / B_1) + (1.0 / (B_1 * ((x[0] / 2.0) - 1.0))) + (x[0] / 2.0 - 1.0) + (x[0] / 2.0 - 1)**2 + K_r1 * ((x[0] - 1.0)**2) / B_1 + K_r1 * ((x[0] - 1.0)**2) / (B_1 * (x[0] / 2.0 - 1.0))) + (1.0 / (x[0] * x[1])) * ((1.0 / B_2) + (1 / (B_2 * ((x[1] / 2.0) - 1.0))) + (x[1] / 2.0 - 1.0) + (x[1] / 2.0 - 1.0)**2.0 + K_r2 * ((x[1] - 1.0)**2.0) / B_2 + K_r2 * ((x[1] - 1.0)**2.0) / (B_2 * (x[1] / 2.0 - 1.0))) + (1.0 / (x[0] * x[1] * x[2])) * (1.0 + (1.0 / x[2]) + x[2] + x[2]**2)

                def constr1(x, overallRatio):
                    return x[0] * x[1] * x[2] - overallRatio

                def constr2(x, overallRatio):
                    return overallRatio - x[0] * x[1] * x[2]

                x = opt.fmin_cobyla(volume, x0, [constr1, constr2], consargs=[
                                    overallRatio], rhoend=1e-7, iprint=0)
            elif config == 'epp':
                # fixes last stage ratio at 3
                x0 = [overallRatio**(1.0 / 3.0), overallRatio **
                                     (1.0 / 3.0), overallRatio**(1.0 / 3.0)]
                B_1 = Np[0]
                B_2 = Np[1]
                K_r = 0

                def volume(x):
                    return (1.0 / (x[0])) * ((1.0 / B_1) + (1.0 / (B_1 * ((x[0] / 2.0) - 1.0))) + (x[0] / 2.0 - 1.0) + (x[0] / 2.0 - 1)**2 +
                    K_r * ((x[0] - 1.0)**2) / B_1 + K_r * ((x[0] - 1.0)**2) / (B_1 * (x[0] / 2.0 - 1.0))) + \
                    (1.0 / (x[0] * x[1])) * (1.0 + (1.0 / x[1]) + x[1] + x[1]**2) \
                    + (1.0 / (x[0] * x[1] * x[2])) * \
                       (1.0 + (1.0 / x[2]) + x[2] + x[2]**2)

                def constr1(x, overallRatio):
                    return x[0] * x[1] * x[2] - overallRatio

                def constr2(x, overallRatio):
                    return overallRatio - x[0] * x[1] * x[2]

                x = opt.fmin_cobyla(volume, x0, [constr1, constr2], consargs=[
                                    overallRatio], rhoend=1e-7, iprint=0)

            else:  # what is this subroutine for?  Yi on 04/16/2014
                x0 = [overallRatio**(1.0 / 3.0), overallRatio **
                                     (1.0 / 3.0), overallRatio**(1.0 / 3.0)]
                B_1 = Np[0]
                K_r = 0.0

                def volume(x):
                    return (1.0 / (x[0])) * ((1.0 / B_1) + (1.0 / (B_1 * ((x[0] / 2.0) - 1.0))) + (x[0] / 2.0 - 1) + (x[0] / 2.0 - 1.0)**2 + K_r * ((x[0] - 1.0)**2) / B_1 + K_r * ((x[0] - 1)**2) / (B_1 * (x[0] / 2.0 - 1.0))) + (1.0 / (x[0] * x[1])) * (1.0 + (1.0 / x[1]) + x[1] + x[1]**2) + (1.0 / (x[0] * x[1] * x[2])) * (1.0 + (1.0 / x[2]) + x[2] + x[2]**2)

                def constr1(x, overallRatio):
                   return x[0] * x[1] * x[2] - overallRatio

                def constr2(x, overallRatio):
                    return overallRatio - x[0] * x[1] * x[2]

                x = opt.fmin_cobyla(volume, x0, [constr1, constr2], consargs=[
                                    overallRatio], rhoend=1e-7, iprint=0)
        else:
            x = 'fail'

        return x

#-------------------------------------------------------------------------


class Bedplate_drive(Component):
    ''' Bedplate class
          The Bedplate class is used to represent the bedplate of a wind turbine drivetrain.
          It contains the general properties for a wind turbine component as well as additional design load and dimentional attributes as listed below.
          It contains an update method to determine the mass, mass properties, and dimensions of the component.
    '''

    def __init__(self):
        super(Bedplate_drive, self).__init__()

        # variables
        self.add_param('gbx_length', val=0.0, units='m', desc='gearbox length')
        self.add_param('gbx_location', val=0.0, units='m', desc='gearbox CM location')
        self.add_param('gbx_mass', val=0.0, units='kg', desc='gearbox mass')
        self.add_param('hss_location', val=0.0, units='m', desc='HSS CM location')
        self.add_param('hss_mass', val=0.0, units='kg', desc='HSS mass')
        self.add_param('generator_location', val=0.0, units='m', desc='generator CM location')
        self.add_param('generator_mass', val=0.0, units='kg', desc='generator mass')
        self.add_param('lss_location', val=0.0, units='m', desc='LSS CM location')
        self.add_param('lss_mass', val=0.0, units='kg', desc='LSS mass')
        self.add_param('lss_length', val=0.0, units='m', desc='LSS length')
        self.add_param('mb1_location', val=0.0, units='m', desc='Upwind main bearing CM location')
        self.add_param('FW_mb1', val=0.0, units='m', desc='Upwind main bearing facewidth')
        self.add_param('mb1_mass', val=0.0, units='kg', desc='Upwind main bearing mass')
        self.add_param('mb2_location', val=0.0, units='m', desc='Downwind main bearing CM location')
        self.add_param('mb2_mass', val=0.0, units='kg', desc='Downwind main bearing mass')
        self.add_param('transformer_mass', val=0.0, units='kg', desc='Transformer mass')
        self.add_param('transformer_location', val=0.0, units='m', desc='transformer CM location')
        self.add_param('tower_top_diameter', val=0.0, units='m', desc='diameter of the top tower section at the yaw gear')
        self.add_param('rotor_diameter', val=0.0, units='m', desc='rotor diameter')
        self.add_param('machine_rating', val=0.0, units='kW', desc='machine_rating machine rating of the turbine')
        self.add_param('rotor_mass', val=0.0, units='kg', desc='rotor mass')
        self.add_param('rotor_bending_moment_y', val=0.0, units='N*m', desc='The bending moment about the y axis')
        self.add_param('rotor_force_z', val=0.0, units='N', desc='The force along the z axis applied at hub center')
        self.add_param('flange_length', val=0.0, units='m', desc='flange length')
        self.add_param('L_rb', val=0.0, units='m', desc='length between rotor center and upwind main bearing')
        self.add_param('overhang', val=0.0, units='m', desc='Overhang distance')

        # parameters
        self.add_param('uptower_transformer', val=desc='Boolean stating if transformer is uptower')

        # outputs
        self.add_output('mass', val=0.0, units='kg', desc='overall component mass')
        self.add_output('cm', val=np.array([0.0, 0.0, 0.0]), desc='center of mass of the component in [x,y,z] for an arbitrary coordinate system')
        self.add_output('I', val=np.array([0.0, 0.0, 0.0]), desc=' moments of Inertia for the component [Ixx, Iyy, Izz] around its center of mass')
        self.add_output('length', val=0.0, units='m', desc='length of bedplate')
        self.add_output('height', val=0.0, units='m',  desc='max height of bedplate')
        self.add_output('width', val=0.0, units='m', desc='width of bedplate')
        
    def characterize_Bedplate_Rear(self):
      self.bi = (self.b0 - self.tw) / 2.0
      self.hi = self.h0 - 2.0 * self.tf
      self.I_b = self.b0 * self.h0**3 / 12.0 - 2 * self.bi * self.hi**3 / 12.0
      self.A = self.b0 * self.h0 - 2.0 * self.bi * self.hi
      self.w = self.A * self.density
      # Tip Deflection for load not at end

      self.hssTipDefl = midDeflection(
          self.rearTotalLength, self.hss_location, self.hss_mass * self.g / 2, self.E, self.I_b)
      self.genTipDefl = midDeflection(
          self.rearTotalLength, self.generator_location, self.generator_mass * self.g / 2, self.E, self.I_b)
      self.convTipDefl = midDeflection(
          self.rearTotalLength, self.convLoc, self.convMass * self.g / 2, self.E, self.I_b)
      self.transTipDefl = midDeflection(
          self.rearTotalLength, self.transLoc, self.transformer_mass * self.g / 2, self.E, self.I_b)
      self.gbxTipDefl = midDeflection(
          self.rearTotalLength, self.gbx_location, self.gbx_mass * self.g / 2, self.E, self.I_b)
      self.selfTipDefl = distDeflection(
          self.rearTotalLength, self.w * self.g, self.E, self.I_b)

      self.totalTipDefl = self.hssTipDefl + self.genTipDefl + self.convTipDefl + \
          self.transTipDefl + self.selfTipDefl + self.gbxTipDefl

      # root stress
      self.totalBendingMoment = (self.hss_location * self.hss_mass + self.generator_location * self.generator_mass +
                                 self.convLoc * self.convMass + self.transLoc * self.transformer_mass + self.w * self.rearTotalLength**2 / 2.0) * self.g
      self.rootStress = self.totalBendingMoment * self.h0 / (2. * self.I_b)

      # mass
      self.steelVolume = self.A * self.rearTotalLength
      self.steelMass = self.steelVolume * self.density

      # 2 parallel I beams
      self.totalSteelMass = 2.0 * self.steelMass

      self.rearTotalTipDefl = self.totalTipDefl
      self.rearBendingStress = self.rootStress

    def characterize_Bedplate_Front(self):
      self.bi = (self.b0 - self.tw) / 2.0
      self.hi = self.h0 - 2.0 * self.tf
      self.I_b = self.b0 * self.h0**3 / 12.0 - 2 * self.bi * self.hi**3 / 12.0
      self.A = self.b0 * self.h0 - 2.0 * self.bi * self.hi
      self.w = self.A * self.castDensity

      # Tip Deflection for load not at end
      self.gbxTipDefl = midDeflection(
          self.frontTotalLength, self.gbx_mass, self.gbx_mass * self.g / 2.0, self.E, self.I_b)
      self.mb1TipDefl = midDeflection(
          self.frontTotalLength, self.mb1_location, self.mb1_mass * self.g / 2.0, self.E, self.I_b)
      self.mb2TipDefl = midDeflection(
          self.frontTotalLength, self.mb2_location, self.mb2_mass * self.g / 2.0, self.E, self.I_b)
      self.lssTipDefl = midDeflection(
          self.frontTotalLength, self.lss_location, self.lss_mass * self.g / 2.0, self.E, self.I_b)
      self.rotorTipDefl = midDeflection(
          self.frontTotalLength, self.rotorLoc, self.rotor_mass * self.g / 2.0, self.E, self.I_b)
      self.rotorFzTipDefl = midDeflection(
          self.frontTotalLength, self.rotorLoc, self.rotorFz / 2.0, self.E, self.I_b)
      self.selfTipDefl = distDeflection(
          self.frontTotalLength, self.w * self.g, self.E, self.I_b)
      self.rotorMyTipDefl = self.rotorMy / 2.0 * \
          self.frontTotalLength**2 / (2.0 * self.E * self.I_b)

      self.totalTipDefl = self.mb1TipDefl + self.mb2TipDefl + self.lssTipDefl  + self.rotorTipDefl + self.selfTipDefl +\
        self.rotorMyTipDefl + self.rotorFzTipDefl + self.gbxTipDefl

      # root stress
      self.totalBendingMoment = (self.mb1_location * self.mb1_mass / 2.0 + self.mb2_location * self.mb2_mass / 2.0 + self.lss_location *
        self.lss_mass / 2.0 + self.w * self.frontTotalLength**2 / 2.0 + self.rotorLoc * self.rotor_mass / 2.0) * self.g + self.rotorLoc *\
        self.rotorFz / 2.0 + self.rotorMy / 2.0
      self.rootStress = self.totalBendingMoment * self.h0 / 2 / self.I_b

      # mass
      self.castVolume = self.A * self.frontTotalLength
      self.castMass = self.castVolume * self.castDensity

      # 2 parallel I-beams
      self.totalCastMass = 2.0 * self.castMass
      self.frontTotalTipDefl = self.totalTipDefl
      self.frontBendingStress = self.rootStress

    def solve_nonlinear(self, params, unknowns, resids):
          # Model bedplate as 2 parallel I-beams with a rear steel frame and a front cast frame
          # Deflection constraints applied at each bedplate end
          # Stress constraint checked at root of front and rear bedplate
          # sections

          self.g = 9.81
          self.E = 2.1e11
          self.density = 7800

          if self.L_rb > 0:
              L_rb = self.L_rb
          else:
              [L_rb] = get_L_rb(self.rotor_diameter, False)

          # component weights and locations
          if self.transformer_mass > 0:  # only if uptower transformer
              self.transLoc = self.transformer_location.item()
              self.convMass = 0.3 * self.transformer_mass
          else:
              self.transLoc = 0
              # (transformer mass * .3)
              self.convMass = (2.4445 * (self.machine_rating) + 1599.0) * 0.3

          self.convLoc = self.generator_location * 2.0
          # TODO: removed self. since this are from connections but not sure if
          # that disruptes upstream usage
          # abs(self.gbx_length/2.0) + abs(self.lss_length)
          mb1_location = abs(self.mb1_location)
          mb2_location = abs(self.mb2_location)  # abs(self.gbx_length/2.0)
          lss_location = abs(self.lss_location)

          if self.transLoc > 0:
            self.rearTotalLength = self.transLoc * 1.1
          else:
            self.rearTotalLength = self.generator_location * 4.237 / \
                2.886 - self.tower_top_diameter / 2.0  # scaled off of GE1.5

          self.frontTotalLength = mb1_location + self.FW_mb1 / 2.

          # rotor weights and loads
          self.rotorLoc = mb1_location + L_rb
          self.rotorFz = abs(self.rotor_force_z)
          self.rotorMy = abs(self.rotor_bending_moment_y)

          # If user does not know important moment, crude approx
          if self.rotor_mass > 0 and self.rotorMy == 0:
              self.rotorMy = get_My(self.rotor_mass, L_rb)

          if self.rotorFz == 0 and self.rotor_mass > 0:
              self.rotorFz = self.rotor_mass * self.g

          # initial I-beam dimensions
          self.tf = 0.01905
          self.tw = 0.0127
          self.h0 = 0.6096
          self.b0 = self.h0 / 2.0

          # Rear Steel Frame:
          if self.gbx_location == 0:
              self.gbx_location = 0
              self.gbx_mass = 0
          else:
              self.gbx_location = self.gbx_location
              self.gbx_mass = self.gbx_mass

          self.rootStress = 250e6
          self.totalTipDefl = 1.0
          self.stressTol = 5e5
          self.deflTol = 1e-4
          self.defl_denom = 1500.  # factor in deflection check
          self.stress_mult = 8.  # modified to fit industry data

          self.stressMax = 620e6  # yeild of alloy steel
          self.deflMax = self.rearTotalLength / self.defl_denom

          counter = 0
          while self.rootStress * self.stress_mult - self.stressMax > self.stressTol or self.totalTipDefl - self.deflMax > self.deflTol:

              counter += 1

              self.characterize_Bedplate_Rear()

              self.tf += 0.002
              self.tw += 0.002
              self.b0 += 0.006
              self.h0 += 0.006
              rearCounter = counter

        self.rearHeight = self.h0
        
        # Front cast section:
        if self.gbx_location < 0:
            self.gbx_location = abs(self.gbx_location)
            self.gbx_mass = self.gbx_mass
        else: 
            self.gbx_location = 0
            self.gbx_mass = 0
        self.E=169e9 #EN-GJS-400-18-LT http://www.claasguss.de/html_e/pdf/THBl2_engl.pdf
        self.castDensity = 7100
        
        self.tf = 0.01905
        self.tw = 0.0127
        self.h0 = 0.6096
        self.b0 = self.h0/2.0
        
        self.rootStress = 250e6
        self.totalTipDefl = 1.0
        
        self.deflMax = self.frontTotalLength/self.defl_denom
        self.stressMax = 200e6
        
        counter = 0
        
        while self.rootStress*self.stress_mult - self.stressMax >  self.stressTol or self.totalTipDefl - self.deflMax >  self.deflTol:
            counter += 1
            characterize_Bedplate_Front(self)
            self.tf += 0.002 
            self.tw += 0.002
            self.b0 += 0.006
            self.h0 += 0.006
            
            frontCounter=counter
          

      self.frontHeight = self.h0

      # frame multiplier for front support
      self.support_multiplier = 1.1+5e13*self.rotor_diameter**(-8) # based on solidworks estimates for GRC and GE bedplates. extraneous mass percentage decreases for larger machines
      # print self.rotor_diameter
      # print support_multiplier
      self.totalCastMass *= self.support_multiplier
      self.totalSteelMass *= self.support_multiplier
      self.mass = self.totalCastMass+ self.totalSteelMass

      # print 'rotor mass', self.rotor_mass
      # print 'rotor bending moment_y', self.rotor_bending_moment_y
      # print 'rotor fz', self.rotor_force_z 
      # print 'rear bedplate length: ', rearTotalLength
      # print 'front bedplate length: ', frontTotalLength
      # print'rear bedplate tip deflection', rearTotalTipDefl
      # print'front bedplate tip deflection', frontTotalTipDefl
      # print 'bending stress [MPa] at root of rear bedplate:', rearBendingStress/1.0e6
      # print 'bending stress [MPa] at root of front bedplate:', frontBendingStress/1.0e6
      # print 'front bedplate bedplate mass [kg]:', totalCastMass
      # print 'rear bedplate mass [kg]:', totalSteelMass
      # print 'total bedplate mass:', totalSteelMass+ totalCastMass

      self.length = self.frontTotalLength + self.rearTotalLength
      self.width = self.b0 + self.tower_top_diameter
      if self.rearHeight >= self.frontHeight:
          self.height = self.rearHeight
      else:
          self.height = self.frontHeight

      # calculate mass properties
      cm = np.array([0.0,0.0,0.0])
      cm[0] = (self.totalSteelMass*self.rearTotalLength/2 - self.totalCastMass*self.frontTotalLength/2)/(self.mass) #previously 0.
      cm[1] = 0.0
      cm[2] = -self.height/2.
      self.cm = cm

      self.depth = (self.length / 2.0)

      I = np.array([0.0, 0.0, 0.0])
      I[0]  = self.mass * (self.width ** 2 + self.depth ** 2) / 8
      I[1]  = self.mass * (self.depth ** 2 + self.width ** 2 + (4/3) * self.length ** 2) / 16
      I[2]  = I[1]
      self.I = I


        
#---------------------------------------------------------------------------------------------------------------

class YawSystem_drive(Component):
    ''' YawSystem class
          The YawSystem class is used to represent the yaw system of a wind turbine drivetrain.
          It contains the general properties for a wind turbine component as well as additional design load and dimentional attributes as listed below.
          It contains an update method to determine the mass, mass properties, and dimensions of the component.
    '''


    def __init__(self):
        super(YawSystem_drive, self).__init__()

        # variables
        self.add_param('rotor_diameter', val=0.0, units='m', desc='rotor diameter')
        self.add_param('rotor_thrust', val=0.0, units='N', desc='maximum rotor thrust')
        self.add_param('tower_top_diameter', val=0.0, units='m', desc='tower top diameter')
        self.add_param('above_yaw_mass', val=0.0, units='kg', desc='above yaw mass')
        self.add_param('bedplate_height', val=0.0, units='m', desc='bedplate height')

        # parameters
        self.add_param('yaw_motors_number', val=00, desc='number of yaw motors')

        # outputs
        self.add_output('mass', val=0.0, units='kg', desc='overall component mass')
        self.add_output('cm', val=np.array([0.0, 0.0, 0.0]), desc='center of mass of the component in [x,y,z] for an arbitrary coordinate system')
        self.add_output('I', val=np.array([0.0, 0.0, 0.0]), desc=' moments of Inertia for the component [Ixx, Iyy, Izz] around its center of mass')    

    def solve_nonlinear(self, params, unknowns, resids):

      if self.yaw_motors_number == 0 :
        if self.rotor_diameter < 90.0 :
          self.yaw_motors_number = 4
        elif self.rotor_diameter < 120.0 :
          self.yaw_motors_number = 6
        else:
          self.yaw_motors_number = 8

      # Assume friction plate surface width is 1/10 the diameter
      # Assume friction plate thickness scales with rotor diameter
      frictionPlateVol=pi*self.tower_top_diameter*(self.tower_top_diameter*0.10)*(self.rotor_diameter/1000.0)
      steelDensity=8000.0
      frictionPlateMass=frictionPlateVol*steelDensity

      # Assume same yaw motors as Vestas V80 for now: Bonfiglioli 709T2M
      yawMotorMass=190.0

      totalYawMass=frictionPlateMass + (self.yaw_motors_number*yawMotorMass)
      self.mass= totalYawMass

      # calculate mass properties
      # yaw system assumed to be collocated to tower top center
      cm = np.array([0.0,0.0,0.0])
      cm[2] = -self.bedplate_height
      self.cm = cm

      # assuming 0 MOI for yaw system (ie mass is nonrotating)
      I = np.array([0.0, 0.0, 0.0])
      self.I = I


        #-------------------------------------------------------------------------------

class Transformer_drive(Component):
    ''' Transformer class
            The transformer class is used to represent the transformer of a wind turbine drivetrain.
            It contains the general properties for a wind turbine component as well as additional design load and dimentional attributes as listed below.
            It contains an update method to determine the mass, mass properties, and dimensions of the component if it is in fact uptower'''

    def __init__(self):
        super(Transformer_drive, self).__init__()

        self.missing_deriv_policy = 'assume_zero'

        # inputs
        self.add_param('machine_rating', val=0.0, units='kW', desc='machine rating of the turbine')
        self.add_param('uptower_transformer', val=desc='uptower or downtower transformer')
        self.add_param('tower_top_diameter', val=0.0, units='m', desc='tower top diameter for comparision of nacelle CM')
        self.add_param('rotor_mass', val=0.0, units='kg', desc='rotor mass')
        self.add_param('overhang', val=0.0, units='m', desc='rotor overhang distance')
        self.add_param('generator_cm', val=np.array([]), desc='center of mass of the generator in [x,y,z]')
        self.add_param('rotor_diameter', val=0.0, units='m', desc='rotor diameter of turbine')
        self.add_param('RNA_mass', val=0.0, units='kg', desc='mass of total RNA')
        self.add_param('RNA_cm', val=0.0, units='m', desc='RNA CM along x-axis')

        # outputs
        self.add_output('mass', val=0.0, units='kg', desc='overall component mass')
        self.add_output('cm', val=np.array([0.0, 0.0, 0.0]), desc='center of mass of the component in [x,y,z] for an arbitrary coordinate system')
        self.add_output('I', val=np.array([0.0, 0.0, 0.0]), desc=' moments of Inertia for the component [Ixx, Iyy, Izz] around its center of mass')    

    def solve_nonlinear(self, params, unknowns, resids):

        if self.uptower_transformer == True:
            # function places transformer where tower top CM is within tower bottom OD to reduce tower moments
            if self.rotor_mass:
                rotor_mass = self.rotor_mass
            else:
                [rotor_mass] = get_rotor_mass(self.machine_rating,False)

            bottom_OD = self.tower_top_diameter*1.7 #approximate average from industry data

            self.mass = 2.4445*(self.machine_rating) + 1599.0
            
            if self.RNA_cm <= -(bottom_OD)/2: #upwind of acceptable. Most likely
                transformer_x = (bottom_OD/2.*(self.RNA_mass+self.mass) - (self.RNA_mass*self.RNA_cm))/(self.mass)
                if transformer_x > self.generator_cm[0]*3:
                    print '\n ---------transformer location manipulation not suitable for overall Nacelle CM changes: rear distance excessively large------- \n'
                    transformer_x = self.generator_cm[0] + (1.6 * 0.015 * self.rotor_diameter) #assuming generator and transformer approximately same length
                else:
                    transformer_x = self.generator_cm[0] + (1.8 * 0.015 * self.rotor_diameter) #assuming generator and transformer approximately same length

            cm = np.array([0.,0.,0.])
            cm[0] = transformer_x
            cm[1] = self.generator_cm[1]
            cm[2] = self.generator_cm[2]/.75*.5 #same height as gearbox CM
            self.cm = cm
            
            width = self.tower_top_diameter+.5
            height = 0.016*self.rotor_diameter #similar to gearbox
            length = .012*self.rotor_diameter #similar to gearbox
            
            def get_I(d1,d2,mass):
                return mass*(d1**2 + d2**2)/12.
            
            I = np.array([0.,0.,0.])
            I[0] = get_I(height,width,self.mass)
            I[1] = get_I(length, height, self.mass)
            I[2] = get_I(length, width, self.mass)
            self.I = I
            
        else:
            self.cm = np.array([0.,0.,0.])
            self.I = self.cm.copy()
            self.mass = 0.

#-------------------------------------------------------------------

class HighSpeedSide_drive(Component):
    '''
    HighSpeedShaft class
          The HighSpeedShaft class is used to represent the high speed shaft and mechanical brake components of a wind turbine drivetrain.
          It contains the general properties for a wind turbine component as well as additional design load and dimentional attributes as listed below.
          It contains an update method to determine the mass, mass properties, and dimensions of the component.
    '''

    def __init__(self):
        super(HighSpeedSide_drive, self).__init__()

        # controls what happens if derivatives are missing
        self.missing_deriv_policy = 'assume_zero'

        # variables
        self.add_param('rotor_diameter', val=0.0, units='m', desc='rotor diameter')
        self.add_param('rotor_torque', val=0.0, units='N*m', desc='rotor torque at rated power')
        self.add_param('gear_ratio', val=0.0 desc='overall gearbox ratio')
        self.add_param('lss_diameter', val=0.0, units='m', desc='low speed shaft outer diameter')
        self.add_param('gearbox_length', val=0.0, units='m', desc='gearbox length')
        self.add_param('gearbox_height', val=0.0, units='m', desc='gearbox height')
        self.add_param('gearbox_cm', val=np.array([]), units='m', desc='gearbox cm [x,y,z]')
        self.add_param('length_in', val=0.0, units='m', desc='high speed shaft length determined by user. Default 0.5m')

        # returns
        self.add_output('mass', val=0.0, units='kg', desc='overall component mass')
        self.add_output('cm', val=np.array([0.0, 0.0, 0.0]), desc='center of mass of the component in [x,y,z] for an arbitrary coordinate system')
        self.add_output('I', val=np.array([0.0, 0.0, 0.0]), desc=' moments of Inertia for the component [Ixx, Iyy, Izz] around its center of mass')
        self.add_output('length', val=0.0 desc='length of high speed shaft')

    def solve_nonlinear(self, params, unknowns, resids):

      # compute masses, dimensions and cost
      design_torque = self.rotor_torque / self.gear_ratio               # design torque [Nm] based on rotor torque and Gearbox ratio
      massFact = 0.025                                 # mass matching factor default value
      highSpeedShaftMass = (massFact * design_torque)

      mechBrakeMass = (0.5 * highSpeedShaftMass)      # relationship derived from HSS multiplier for University of Sunderland model compared to NREL CSM for 750 kW and 1.5 MW turbines

      self.mass = (mechBrakeMass + highSpeedShaftMass)

      diameter = (1.5 * self.lss_diameter)                     # based on WindPACT relationships for full HSS / mechanical brake assembly
      if self.length_in == 0:
          self.length = 0.5+self.rotor_diameter/127.
      else:
          self.length = self.length_in
      length = self.length

      matlDensity = 7850. # material density kg/m^3

      # calculate mass properties
      cm = np.array([0.0,0.0,0.0])
      cm[0]   = self.gearbox_cm[0]+self.gearbox_length/2+length/2
      cm[1]   = self.gearbox_cm[1]
      cm[2]   = self.gearbox_cm[2]+self.gearbox_height*0.2
      self.cm = cm

      I = np.array([0.0, 0.0, 0.0])
      I[0]    = 0.25 * length * 3.14159 * matlDensity * (diameter ** 2) * (self.gear_ratio**2) * (diameter ** 2) / 8.
      I[1]    = self.mass * ((3/4.) * (diameter ** 2) + (length ** 2)) / 12.
      I[2]    = I[1]
      self.I = I


#----------------------------------------------------------------------------------------------

class Generator_drive(Component):
    '''Generator class
          The Generator class is used to represent the generator of a wind turbine drivetrain.
          It contains the general properties for a wind turbine component as well as additional design load and dimentional attributes as listed below.
          It contains an update method to determine the mass, mass properties, and dimensions of the component.
    '''

    def __init__(self):
        super(Generator_drive, self).__init__()

        # controls what happens if derivatives are missing
        self.missing_deriv_policy = 'assume_zero'

        # variables
        self.add_param('rotor_diameter', val=0.0, units='m', desc='rotor diameter')
        self.add_param('machine_rating', val=0.0, units='kW', desc='machine rating of generator')
        self.add_param('gear_ratio', val=0.0 desc='overall gearbox ratio')
        self.add_param('highSpeedSide_length', val=0.0, units='m', desc='length of high speed shaft and brake')
        self.add_param('highSpeedSide_cm', val=np.array([0.0,0.0,0.0]), units='m', desc='cm of high speed shaft and brake')
        self.add_param('rotor_speed', val=0.0, units='rpm', desc='Speed of rotor at rated power')

        # parameters
        self.add_param('drivetrain_design', val='geared', ('geared', 'single_stage', 'multi_drive', 'pm_direct_drive'))
        #returns
        self.add_output('mass', val=0.0, units='kg', desc='overall component mass')
        self.add_output('cm', val=np.array([0.0, 0.0, 0.0]), desc='center of mass of the component in [x,y,z] for an arbitrary coordinate system')
        self.add_output('I', val=np.array([0.0, 0.0, 0.0]), desc=' moments of Inertia for the component [Ixx, Iyy, Izz] around its center of mass')
        
    def solve_nonlinear(self, params, unknowns, resids):

      massCoeff = [None, 6.4737, 10.51 ,  5.34  , 37.68  ]
      massExp   = [None, 0.9223, 0.9223,  0.9223, 1      ]

      if self.rotor_speed !=0:
        CalcRPM = self.rotor_speed
      else:
        CalcRPM    = 80 / (self.rotor_diameter*0.5*pi/30)
      CalcTorque = (self.machine_rating*1.1) / (CalcRPM * pi/30)

      if self.drivetrain_design == 'geared':
          drivetrain_design = 1
      elif self.drivetrain_design == 'single_stage':
          drivetrain_design = 2
      elif self.drivetrain_design == 'multi_drive':
          drivetrain_design = 3
      elif self.drivetrain_design == 'pm_direct_drive':
          drivetrain_design = 4

      if (drivetrain_design < 4):
          self.mass = (massCoeff[drivetrain_design] * self.machine_rating ** massExp[drivetrain_design])
      else:  # direct drive
          self.mass = (massCoeff[drivetrain_design] * CalcTorque ** massExp[drivetrain_design])

      # calculate mass properties
      length = (1.8 * 0.015 * self.rotor_diameter)
      d_length_d_rotor_diameter = 1.8*.015

      depth = (0.015 * self.rotor_diameter)
      d_depth_d_rotor_diameter = 0.015

      width = (0.5 * depth)
      d_width_d_depth = 0.5

      # print self.highSpeedSide_cm
      cm = np.array([0.0,0.0,0.0])
      cm[0]  = self.highSpeedSide_cm[0] + self.highSpeedSide_length/2. + length/2.
      cm[1]  = self.highSpeedSide_cm[1]
      cm[2]  = self.highSpeedSide_cm[2]
      self.cm = cm

      I = np.array([0.0, 0.0, 0.0])
      I[0]   = ((4.86 * (10. ** (-5))) * (self.rotor_diameter ** 5.333)) + (((2./3.) * self.mass) * (depth ** 2 + width ** 2) / 8.)
      I[1]   = (I[0] / 2.) / (self.gear_ratio ** 2) + ((1./3.) * self.mass * (length ** 2) / 12.) + (((2. / 3.) * self.mass) * \
                 (depth ** 2. + width ** 2. + (4./3.) * (length ** 2.)) / 16. )
      I[2]   = I[1]
      self.I = I 



#-------------------------------------------------------------------------------

class AboveYawMassAdder_drive(Component):

    def __init__(self):
        super(AboveYawMassAdder_drive, self).__init__()

        # controls what happens if derivatives are missing
        self.missing_deriv_policy = 'assume_zero'

        # variables
        self.add_param('machine_rating', val=0.0, units='kW', desc='machine rating')
        self.add_param('lss_mass', val=0.0, units='kg', desc='component mass')
        self.add_param('main_bearing_mass', val=0.0, units='kg', desc='component mass')
        self.add_param('second_bearing_mass', val=0.0, units='kg', desc='component mass')
        self.add_param('gearbox_mass', val=0.0, units='kg', desc='component mass')
        self.add_param('hss_mass', val=0.0, units='kg', desc='component mass')
        self.add_param('generator_mass', val=0.0, units='kg', desc='component mass')
        self.add_param('bedplate_mass', val=0.0, units='kg', desc='component mass')
        self.add_param('bedplate_length', val=0.0, units='m', desc='component length')
        self.add_param('bedplate_width', val=0.0, units='m', desc='component width')
        self.add_param('transformer_mass', val=0.0, units='kg', desc='component mass')

        # parameters
        self.add_param('crane', val=desc='flag for presence of crane')

        # returns
        self.add_output('electrical_mass', val=0.0, units='kg', desc='component mass')
        self.add_output('vs_electronics_mass', val=0.0, units='kg', desc='component mass')
        self.add_output('hvac_mass', val=0.0, units='kg', desc='component mass')
        self.add_output('controls_mass', val=0.0, units='kg', desc='component mass')
        self.add_output('platforms_mass', val=0.0, units='kg', desc='component mass')
        self.add_output('crane_mass', val=0.0, units='kg', desc='component mass')
        self.add_output('mainframe_mass', val=0.0, units='kg', desc='component mass')
        self.add_output('cover_mass', val=0.0, units='kg', desc='component mass')
        self.add_output('above_yaw_mass', val=0.0, units='kg', desc='total mass above yaw system')
        self.add_output('length', val=0.0, units='m', desc='component length')
        self.add_output('width', val=0.0, units='m', desc='component width')
        self.add_output('height', val=0.0, units='m', desc='component height')
        
    def solve_nonlinear(self, params, unknowns, resids):
        # electronic systems, hydraulics and controls
        self.electrical_mass = 0.0
        
        self.vs_electronics_mass = 0 #2.4445*self.machine_rating + 1599.0 accounted for in transformer calcs
        
        self.hvac_mass = 0.08 * self.machine_rating
        
        self.controls_mass     = 0.0
        
        # mainframe system including bedplate, platforms, crane and miscellaneous hardware
        self.platforms_mass = 0.125 * self.bedplate_mass
        
        if (self.crane):
            self.crane_mass =  3000.0
        else:
            self.crane_mass = 0.0
            
        self.mainframe_mass  = self.bedplate_mass + self.crane_mass + self.platforms_mass
        
        nacelleCovArea      = 2 * (self.bedplate_length ** 2)              # this calculation is based on Sunderland
        self.cover_mass = (84.1 * nacelleCovArea) / 2          # this calculation is based on Sunderland - divided by 2 in order to approach CSM
        
        # yaw system weight calculations based on total system mass above yaw system
        self.above_yaw_mass =  (self.lss_mass + 
                                self.main_bearing_mass + self.second_bearing_mass + 
                                self.gearbox_mass + 
                                self.hss_mass + 
                                self.generator_mass + 
                                self.mainframe_mass + 
                                self.transformer_mass +
                                self.electrical_mass + 
                                self.vs_electronics_mass + 
                                self.hvac_mass +
                                self.cover_mass)

        self.length      = self.bedplate_length                              # nacelle length [m] based on bedplate length
        self.width       = self.bedplate_width                        # nacelle width [m] based on bedplate width
        self.height      = (2.0 / 3.0) * self.length                         # nacelle height [m] calculated based on cladding area


#--------------------------------------------
class RNASystemAdder_drive(Component):
    ''' RNASystem class
          This analysis is only to be used in placing the transformer of the drivetrain.
          The Rotor-Nacelle-Group class is used to represent the RNA of the turbine without the transformer and bedplate (to resolve circular dependency issues).
          It contains the general properties for a wind turbine component as well as additional design load and dimentional attributes as listed below.
          It contains an update method to determine the mass, mass properties, and dimensions of the component. 
    '''

    def __init__(self):
        super(RNASystemAdder_drive , self).__init__()

        # controls what happens if derivatives are missing
        self.missing_deriv_policy = 'assume_zero'

        # inputs
        self.add_param('yawMass', val=0.0, units='kg', desc='mass of yaw system')
        self.add_param('lss_mass', val=0.0, units='kg', desc='component mass')
        self.add_param('main_bearing_mass', val=0.0, units='kg', desc='component mass')
        self.add_param('second_bearing_mass', val=0.0, units='kg', desc='component mass')
        self.add_param('gearbox_mass', val=0.0, units='kg', desc='component mass')
        self.add_param('hss_mass', val=0.0, units='kg', desc='component mass')
        self.add_param('generator_mass', val=0.0, units='kg', desc='component mass')
        self.add_param('lss_cm', val=np.array([0.0,0.0,0.0]), units='kg', desc='component CM')
        self.add_param('main_bearing_cm', val=np.array([0.0,0.0,0.0]), units='kg', desc='component CM')
        self.add_param('second_bearing_cm', val=np.array([0.0,0.0,0.0]), units='kg', desc='component CM')
        self.add_param('gearbox_cm', val=np.array([0.0,0.0,0.0]), units='kg', desc='component CM')
        self.add_param('hss_cm', val=np.array([0.0,0.0,0.0]), units='kg', desc='component CM')
        self.add_param('generator_cm', val=np.array([0.0,0.0,0.0]), units='kg', desc='component CM')
        self.add_param('overhang', val=0.0, units='m', desc='nacelle overhang')
        self.add_param('rotor_mass', val=0.0, units='kg', desc='component mass')
        self.add_param('machine_rating', val=0.0, units='kW', desc='machine rating')

        # returns
        self.add_output('RNA_mass', val=0.0, units='kg', desc='mass of total RNA')
        self.add_output('RNA_cm', val=0.0, units='m', desc='RNA CM along x-axis')
        
    def solve_nonlinear(self, params, unknowns, resids):

        if self.rotor_mass>0:
            rotor_mass = self.rotor_mass
        else:
            [rotor_mass] = get_rotor_mass(self.machine_rating,False)

        masses = np.array([rotor_mass, self.yawMass, self.lss_mass, self.main_bearing_mass,self.second_bearing_mass,self.gearbox_mass,self.hss_mass,self.generator_mass])
        cms = np.array([(-self.overhang), 0.0, self.lss_cm[0], self.main_bearing_cm[0], self.second_bearing_cm[0], self.gearbox_cm[0], self.hss_cm[0], self.generator_cm[0]])
        
        self.RNA_mass = np.sum(masses)
        self.RNA_cm = np.sum(masses*cms)/np.sum(masses)
        

#--------------------------------------------
class NacelleSystemAdder_drive(Component): #added to drive to include transformer
    ''' NacelleSystem class
          The Nacelle class is used to represent the overall nacelle of a wind turbine.
          It contains the general properties for a wind turbine component as well as additional design load and dimentional attributes as listed below.
          It contains an update method to determine the mass, mass properties, and dimensions of the component.
    '''

    def __init__(self):
        super(NacelleSystemAdder_drive , self).__init__()

        # controls what happens if derivatives are missing
        self.missing_deriv_policy = 'assume_zero'

        # variables
        self.add_param('above_yaw_mass', val=0.0, units='kg', desc='mass above yaw system')
        self.add_param('yawMass', val=0.0, units='kg', desc='mass of yaw system')
        self.add_param('lss_mass', val=0.0, units='kg', desc='component mass')
        self.add_param('main_bearing_mass', val=0.0, units='kg', desc='component mass')
        self.add_param('second_bearing_mass', val=0.0, units='kg', desc='component mass')
        self.add_param('gearbox_mass', val=0.0, units='kg', desc='component mass')
        self.add_param('hss_mass', val=0.0, units='kg', desc='component mass')
        self.add_param('generator_mass', val=0.0, units='kg', desc='component mass')
        self.add_param('bedplate_mass', val=0.0, units='kg', desc='component mass')
        self.add_param('mainframe_mass', val=0.0, units='kg', desc='component mass')
        self.add_param('lss_cm', val=np.array([0.0,0.0,0.0]), units='kg', desc='component CM')
        self.add_param('main_bearing_cm', val=np.array([0.0,0.0,0.0]), units='kg', desc='component CM')
        self.add_param('second_bearing_cm', val=np.array([0.0,0.0,0.0]), units='kg', desc='component CM')
        self.add_param('gearbox_cm', val=np.array([0.0,0.0,0.0]), units='kg', desc='component CM')
        self.add_param('hss_cm', val=np.array([0.0,0.0,0.0]), units='kg', desc='component CM')
        self.add_param('generator_cm', val=np.array([0.0,0.0,0.0]), units='kg', desc='component CM')
        self.add_param('bedplate_cm', val=np.array([0.0,0.0,0.0]), units='kg', desc='component CM')
        self.add_param('lss_I', val=np.array([0.0,0.0,0.0]), units='kg', desc='component I')
        self.add_param('main_bearing_I', val=np.array([0.0,0.0,0.0]), units='kg', desc='component I')
        self.add_param('second_bearing_I', val=np.array([0.0,0.0,0.0]), units='kg', desc='component I')
        self.add_param('gearbox_I', val=np.array([0.0,0.0,0.0]), units='kg', desc='component I')
        self.add_param('hss_I', val=np.array([0.0,0.0,0.0]), units='kg', desc='component I')
        self.add_param('generator_I', val=np.array([0.0,0.0,0.0]), units='kg', desc='component I')
        self.add_param('bedplate_I', val=np.array([0.0,0.0,0.0]), units='kg', desc='component I')
        self.add_param('transformer_mass', val=0.0, units='kg', desc='component mass')
        self.add_param('transformer_cm', val=np.array([0.0,0.0,0.0]), units='kg', desc='component CM')
        self.add_param('transformer_I', val=np.array([0.0,0.0,0.0]), units='kg', desc='component I')

        # returns
        self.add_output('nacelle_mass', val=0.0, units='kg', desc='overall component mass')
        self.add_output('nacelle_cm', val=np.array([0.0, 0.0, 0.0]), units='m', desc='center of mass of the component in [x,y,z] for an arbitrary coordinate system')
        self.add_output('nacelle_I', val=np.array([0.0, 0.0, 0.0]), units='kg*m**2', desc=' moments of Inertia for the component [Ixx, Iyy, Izz] around its center of mass')
        
    def solve_nonlinear(self, params, unknowns, resids):

      # aggregation of nacelle mass
      self.nacelle_mass = (self.above_yaw_mass + self.yawMass)

      # calculation of mass center and moments of inertia
      cm = np.array([0.0,0.0,0.0])
      for i in (range(0,3)):
          # calculate center of mass (use mainframe_mass in place of bedplate_mass - assume lumped around bedplate_cm)
          cm[i] = ( (self.lss_mass * self.lss_cm[i] + self.transformer_cm[i] * self.transformer_mass + 
                     self.main_bearing_mass * self.main_bearing_cm[i] + self.second_bearing_mass * self.second_bearing_cm[i] + 
                     self.gearbox_mass * self.gearbox_cm[i] + self.hss_mass * self.hss_cm[i] + 
                     self.generator_mass * self.generator_cm[i] + self.mainframe_mass * self.bedplate_cm[i] ) / 
                    (self.lss_mass + self.main_bearing_mass + self.second_bearing_mass + 
                     self.gearbox_mass + self.hss_mass + self.generator_mass + self.mainframe_mass) )
      self.nacelle_cm = cm

      I = np.zeros(6)
      for i in (range(0,3)):                        # calculating MOI, at nacelle center of gravity with origin at tower top center / yaw mass center, ignoring masses of non-drivetrain components / auxiliary systems
          # calculate moments around CM
          # sum moments around each components CM (adjust for mass of mainframe) # TODO: add yaw MMI
          I[i]  =  self.lss_I[i] + self.main_bearing_I[i] + self.second_bearing_I[i] + self.gearbox_I[i] + self.transformer_I[i] +\
                        self.hss_I[i] + self.generator_I[i] + self.bedplate_I[i] * (self.mainframe_mass / self.bedplate_mass)
          # translate to nacelle CM using parallel axis theorem (use mass of mainframe en lieu of bedplate to account for auxiliary equipment)
          for j in (range(0,3)):
              if i != j:
                  I[i] += (self.lss_mass * (self.lss_cm[j] - cm[j]) ** 2 + 
                           self.main_bearing_mass * (self.main_bearing_cm[j] - cm[j]) ** 2 + 
                           self.second_bearing_mass * (self.second_bearing_cm[j] - cm[j]) ** 2 + 
                           self.gearbox_mass * (self.gearbox_cm[j] - cm[j]) ** 2 + 
                           self.transformer_mass * (self.transformer_cm[j] - cm[j]) ** 2 + 
                           self.hss_mass * (self.hss_cm[j] - cm[j]) ** 2 + 
                           self.generator_mass * (self.generator_cm[j] - cm[j]) ** 2 + 
                           self.mainframe_mass * (self.bedplate_cm[j] - cm[j]) ** 2 )
      self.nacelle_I = I


if __name__ == '__main__':
     pass
