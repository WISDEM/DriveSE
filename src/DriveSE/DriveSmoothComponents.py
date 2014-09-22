#!/usr/bin/env python
# encoding: utf-8
"""
DriveSmoothComponents.py

Created by Andrew Ning on 2014-02-15.
Copyright (c) NREL. All rights reserved.
"""

from openmdao.main.api import Component, Assembly
from openmdao.main.datatypes.api import Float, Array, Enum, Str, Int, Bool
import numpy as np
from math import pi, sin, cos, radians
from scipy.optimize import fmin_cobyla
import algopy

from akima import Akima
from commonse.utilities import smooth_abs, vstack
from drivewpact.drive import NacelleBase
from drivewpact.drive import HighSpeedSide, Generator, AboveYawMassAdder, NacelleSystemAdder
from fusedwind.interface import implement_base

@implement_base(NacelleBase)
class NacelleTS(Assembly):

    # Base Variables
    # variables
    rotor_diameter = Float(iotype='in', units='m', desc='rotor diameter')
    rotor_mass = Float(iotype='in', units='kg', desc='rotor mass')
    rotor_torque = Float(iotype='in', units='N*m', desc='rotor torque at rated power')
    rotor_thrust = Float(iotype='in', units='N', desc='maximum rotor thrust')
    rotor_speed = Float(iotype='in', units='rpm', desc='rotor speed at rated')
    machine_rating = Float(iotype='in', units='kW', desc='machine rating of generator')
    gear_ratio = Float(iotype='in', desc='overall gearbox ratio')
    tower_top_diameter = Float(iotype='in', units='m', desc='diameter of tower top')
    rotor_bending_moment = Float(iotype='in', units='N*m', desc='maximum aerodynamic bending moment')

    # parameters
    drivetrain_design = Int(iotype='in', desc='type of gearbox based on drivetrain type: 1 = standard 3-stage gearbox, 2 = single-stage, 3 = multi-gen, 4 = direct drive', deriv_ignore=True)
    crane = Bool(iotype='in', desc='flag for presence of crane', deriv_ignore=True)
    bevel = Int(0, iotype='in', desc='Flag for the presence of a bevel stage - 1 if present, 0 if not')
    gear_configuration = Str(iotype='in', desc='tring that represents the configuration of the gearbox (stage number and types)')

    # outputs
    nacelle_mass = Float(iotype='out', units='kg', desc='nacelle mass')
    nacelle_cm = Array(iotype='out', units='m', desc='center of mass of nacelle from tower top in yaw-aligned coordinate system')
    nacelle_I = Array(iotype='out', units='kg*m**2', desc='mass moments of inertia for nacelle [Ixx, Iyy, Izz, Ixy, Ixz, Iyz] about its center of mass')
    low_speed_shaft_mass = Float(iotype='out', units='kg', desc='component mass')
    main_bearing_mass = Float(iotype='out', units='kg', desc='component mass')
    second_bearing_mass = Float(iotype='out', units='kg', desc='component mass')
    gearbox_mass = Float(iotype='out', units='kg', desc='component mass')
    high_speed_side_mass = Float(iotype='out', units='kg', desc='component mass')
    generator_mass = Float(iotype='out', units='kg', desc='component mass')
    bedplate_mass = Float(iotype='out', units='kg', desc='component mass')
    yaw_system_mass = Float(iotype='out', units='kg', desc='component mass')

    # design variables
    L_ms = Float(iotype='in')  # lengths in low-speed shaft
    L_mb = Float(iotype='in')
    # L_ms_gb = Float(iotype='in')

    # tf_rear = Float(iotype='in')  # Ibeam sizing in bedplate
    # tw_rear = Float(iotype='in')
    h0_rear = Float(iotype='in')

    # tf_front = Float(iotype='in')
    # tw_front = Float(iotype='in')
    h0_front = Float(iotype='in')

    # parameters
    Np = Array(np.array([0, 0, 0]), iotype='in', dtype=np.int, desc='number of planets in each stage')
    ratio_type = Str(iotype='in', desc='optimal or empirical stage ratios')
    shaft_type = Str(iotype='in', desc='normal or short shaft length')
    shaft_angle = Float(iotype='in', units='deg', desc='Angle of the LSS inclindation with respect to the horizontal')
    shaft_ratio = Float(iotype='in', desc='Ratio of inner diameter to outer diameter.  Leave zero for solid LSS')
    shrink_disc_mass = Float(iotype='in', units='kg', desc='Mass of the shrink disc')
    mb1Type = Str(iotype='in', desc='Main bearing type: CARB, TRB or SRB')
    mb2Type = Str(iotype='in', desc='Second bearing type: CARB, TRB or SRB')
    yaw_motors_number = Float(iotype='in', desc='number of yaw motors')
    g = Float(9.81, iotype='in', units='m/s**2', desc='acceleration of gravity')

    # constraints
    sizing_constraints = Array(iotype='out')  # sizing constraints in low speed shaft.  all must <= 0
    rootStress_margin_rear = Float(iotype='out')  # bedplate constraints
    totalTipDefl_margin_rear = Float(iotype='out')
    rootStress_margin_front = Float(iotype='out')
    totalTipDefl_margin_front = Float(iotype='out')


    def configure(self):

        # select components
        self.add('gearbox', GearboxSmooth())
        self.add('lowSpeedShaft', LowSpeedShaftDrive4ptSmooth())
        self.add('mainBearing', BearingSmooth())
        self.add('secondBearing', BearingSmooth())
        self.add('highSpeedSide', HighSpeedSide())
        self.add('generator', Generator())
        self.add('bedplate', BedplateSmooth())
        self.add('above_yaw_massAdder', AboveYawMassAdder())
        self.add('yawSystem', YawSystemSmooth())
        self.add('nacelleSystem', NacelleSystemAdder())

        self.driver.workflow.add(['gearbox', 'lowSpeedShaft', 'mainBearing', 'secondBearing', 'highSpeedSide', 'generator', 'bedplate', 'above_yaw_massAdder', 'yawSystem', 'nacelleSystem'])

        # connections to gearbox
        self.connect('rotor_diameter', 'gearbox.rotor_diameter')
        self.connect('rotor_torque', 'gearbox.rotor_torque')
        self.connect('Np', 'gearbox.Np')
        self.connect('gear_ratio', 'gearbox.gear_ratio')
        self.connect('gear_configuration', 'gearbox.gear_configuration')
        self.connect('ratio_type', 'gearbox.ratio_type')
        self.connect('shaft_type', 'gearbox.shaft_type')

        # connections to lowSpeedShaft
        self.connect('rotor_mass', 'lowSpeedShaft.rotor_mass')
        self.connect('rotor_diameter', 'lowSpeedShaft.rotor_diameter')
        self.connect('rotor_thrust', 'lowSpeedShaft.rotor_force_x')
        self.connect('-rotor_mass * g', 'lowSpeedShaft.rotor_force_z')
        self.connect('rotor_torque', 'lowSpeedShaft.rotor_bending_moment_x')
        self.lowSpeedShaft.rotor_force_y = 0.0
        self.lowSpeedShaft.rotor_bending_moment_y = 0.0
        self.lowSpeedShaft.rotor_bending_moment_z = 0.0
        self.connect('gearbox.mass', 'lowSpeedShaft.gearbox_mass')
        self.connect('shaft_angle', 'lowSpeedShaft.shaft_angle')
        self.connect('shaft_ratio', 'lowSpeedShaft.shaft_ratio')
        self.connect('shrink_disc_mass', 'lowSpeedShaft.shrink_disc_mass')
        self.connect('mb1Type', 'lowSpeedShaft.mb1Type')
        self.connect('mb2Type', 'lowSpeedShaft.mb2Type')
        self.connect('L_ms', 'lowSpeedShaft.L_ms')
        self.connect('L_mb', 'lowSpeedShaft.L_mb')
        # self.connect('L_ms_gb', 'lowSpeedShaft.L_ms_gb')

        # connections to mainBearing
        self.connect('mb1Type', 'mainBearing.bearing_type')
        self.connect('lowSpeedShaft.diameter1', 'mainBearing.lss_diameter')
        self.connect('rotor_diameter', 'mainBearing.rotor_diameter')
        self.mainBearing.bearing_switch = 'main'

        # connections to secondBearing
        self.connect('mb2Type', 'secondBearing.bearing_type')
        self.connect('lowSpeedShaft.diameter2', 'secondBearing.lss_diameter')
        self.connect('rotor_diameter', 'secondBearing.rotor_diameter')
        self.mainBearing.bearing_switch = 'second'

        # connections to highSpeedSide
        self.connect('rotor_diameter', 'highSpeedSide.rotor_diameter')
        self.connect('rotor_torque', 'highSpeedSide.rotor_torque')
        self.connect('gear_ratio', 'highSpeedSide.gear_ratio')
        self.connect('lowSpeedShaft.diameter1', 'highSpeedSide.lss_diameter')

        # connections to generator
        self.connect('rotor_diameter', 'generator.rotor_diameter')
        self.connect('machine_rating', 'generator.machine_rating')
        self.connect('gear_ratio', 'generator.gear_ratio')
        self.connect('drivetrain_design', 'generator.drivetrain_design')

        # connections to bedplate
        self.connect('rotor_diameter', 'bedplate.rotor_diameter')
        self.connect('machine_rating', 'bedplate.machine_rating')
        self.connect('rotor_mass', 'bedplate.rotor_mass')
        self.connect('tower_top_diameter', 'bedplate.tower_top_diameter')
        self.connect('highSpeedSide.cm[0]', 'bedplate.hss_location')
        self.connect('highSpeedSide.mass', 'bedplate.hss_mass')
        self.connect('generator.cm[0]', 'bedplate.generator_location')
        self.connect('generator.mass', 'bedplate.generator_mass')
        self.connect('lowSpeedShaft.cm[0]', 'bedplate.lss_location')
        self.connect('lowSpeedShaft.mass', 'bedplate.lss_mass')
        self.connect('mainBearing.cm[0]', 'bedplate.mb1_location')
        self.connect('mainBearing.mass', 'bedplate.mb1_mass')
        self.connect('secondBearing.cm[0]', 'bedplate.mb2_location')
        self.connect('secondBearing.mass', 'bedplate.mb2_mass')
        self.connect('-rotor_mass * g', 'bedplate.rotor_force_z')
        # self.connect('tf_rear', 'bedplate.tf_rear')
        # self.connect('tw_rear', 'bedplate.tw_rear')
        self.connect('h0_rear', 'bedplate.h0_rear')
        # self.connect('tf_front', 'bedplate.tf_front')
        # self.connect('tw_front', 'bedplate.tw_front')
        self.connect('h0_front', 'bedplate.h0_front')
        self.rotor_bending_moment_y = 0.0  # TODO: for now

        # connections to above_yaw_massAdder
        self.connect('machine_rating', 'above_yaw_massAdder.machine_rating')
        self.connect('crane', 'above_yaw_massAdder.crane')
        self.connect('lowSpeedShaft.mass', 'above_yaw_massAdder.lss_mass')
        self.connect('mainBearing.mass', 'above_yaw_massAdder.main_bearing_mass')
        self.connect('secondBearing.mass', 'above_yaw_massAdder.second_bearing_mass')
        self.connect('gearbox.mass', 'above_yaw_massAdder.gearbox_mass')
        self.connect('highSpeedSide.mass', 'above_yaw_massAdder.hss_mass')
        self.connect('generator.mass', 'above_yaw_massAdder.generator_mass')
        self.connect('bedplate.mass', 'above_yaw_massAdder.bedplate_mass')
        self.connect('bedplate.length', 'above_yaw_massAdder.bedplate_length')
        self.connect('bedplate.width', 'above_yaw_massAdder.bedplate_width')

        # connections to yawSystem
        self.connect('rotor_diameter', 'yawSystem.rotor_diameter')
        # self.connect('rotor_thrust', 'yawSystem.rotor_thrust')
        self.connect('tower_top_diameter', 'yawSystem.tower_top_diameter')
        self.connect('yaw_motors_number', 'yawSystem.yaw_motors_number')
        # self.connect('above_yaw_massAdder.above_yaw_mass', 'yawSystem.above_yaw_mass')

        # connections to nacelle system
        self.connect('lowSpeedShaft.mass', 'nacelleSystem.lss_mass')
        self.connect('mainBearing.mass', 'nacelleSystem.main_bearing_mass')
        self.connect('secondBearing.mass', 'nacelleSystem.second_bearing_mass')
        self.connect('gearbox.mass', 'nacelleSystem.gearbox_mass')
        self.connect('highSpeedSide.mass', 'nacelleSystem.hss_mass')
        self.connect('generator.mass', 'nacelleSystem.generator_mass')
        self.connect('bedplate.mass', 'nacelleSystem.bedplate_mass')
        self.connect('above_yaw_massAdder.mainframe_mass', 'nacelleSystem.mainframe_mass')
        self.connect('yawSystem.mass', 'nacelleSystem.yawMass')
        self.connect('above_yaw_massAdder.above_yaw_mass', 'nacelleSystem.above_yaw_mass')
        self.connect('lowSpeedShaft.cm', 'nacelleSystem.lss_cm')
        self.connect('mainBearing.cm', 'nacelleSystem.main_bearing_cm')
        self.connect('secondBearing.cm', 'nacelleSystem.second_bearing_cm')
        self.connect('gearbox.cm', 'nacelleSystem.gearbox_cm')
        self.connect('highSpeedSide.cm', 'nacelleSystem.hss_cm')
        self.connect('generator.cm', 'nacelleSystem.generator_cm')
        self.connect('bedplate.cm', 'nacelleSystem.bedplate_cm')
        self.connect('lowSpeedShaft.I', 'nacelleSystem.lss_I')
        self.connect('mainBearing.I', 'nacelleSystem.main_bearing_I')
        self.connect('secondBearing.I', 'nacelleSystem.second_bearing_I')
        self.connect('gearbox.I', 'nacelleSystem.gearbox_I')
        self.connect('highSpeedSide.I', 'nacelleSystem.hss_I')
        self.connect('generator.I', 'nacelleSystem.generator_I')
        self.connect('bedplate.I', 'nacelleSystem.bedplate_I')

        # connections to outputs
        self.connect('lowSpeedShaft.sizing_constraints', 'sizing_constraints')
        self.connect('bedplate.rootStress_margin_rear', 'rootStress_margin_rear')
        self.connect('bedplate.totalTipDefl_margin_rear', 'totalTipDefl_margin_rear')
        self.connect('bedplate.rootStress_margin_front', 'rootStress_margin_front')
        self.connect('bedplate.totalTipDefl_margin_front', 'totalTipDefl_margin_front')
        self.connect('lowSpeedShaft.mass', 'low_speed_shaft_mass')
        self.connect('mainBearing.mass', 'main_bearing_mass')
        self.connect('secondBearing.mass', 'second_bearing_mass')
        self.connect('gearbox.mass', 'gearbox_mass')
        self.connect('highSpeedSide.mass', 'high_speed_side_mass')
        self.connect('generator.mass', 'generator_mass')
        self.connect('bedplate.mass', 'bedplate_mass')
        self.connect('yawSystem.mass', 'yaw_system_mass')
        self.connect('nacelleSystem.nacelle_mass', 'nacelle_mass')
        self.connect('nacelleSystem.nacelle_cm', 'nacelle_cm')
        self.connect('nacelleSystem.nacelle_I', 'nacelle_I')



class BearingSmooth(Component):

    # variables
    bearing_type = Str(iotype='in', desc='Main bearing type: CARB, TRB or SRB')
    lss_diameter = Float(iotype='in', units='m', desc='lss outer diameter at main bearing')
    rotor_diameter = Float(iotype='in', units='m', desc='rotor diameter')
    bearing_switch = Enum('main', ('main', 'second'), iotype='in')

    # returns
    mass = Float(0.0, iotype='out', units='kg', desc='overall component mass')
    cm = Array(np.array([0.0, 0.0, 0.0]), iotype='out', desc='center of mass of the component in [x,y,z] for an arbitrary coordinate system')
    I = Array(np.array([0.0, 0.0, 0.0]), iotype='out', desc=' moments of Inertia for the component [Ixx, Iyy, Izz] around its center of mass')

    def execute(self):

        # setup spline for mass as function of diameter
        dpt = [0.0, 0.1, 0.2, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95, 1.05, 1.15, 1.2, 1.3, 1.4]

        if self.bearing_type == 'CARB':
            mpt = [120.0, 120.0, 120.0, 120.0, 145.0, 225.0, 390.0, 645.0, 860.0, 1200.0, 1570.0, 2000.0, 2740.0, 2740.0, 2740.0, 2740.0]

        elif self.bearing_type == 'SRB':
            mpt = [128.7, 128.7, 128.7, 128.7, 220.0, 440.0, 715.0, 1200.0, 1600.0, 2000.0, 2350.0, 2700.0, 2960.0, 2960.0, 2960.0, 2960.0]

        # TODO: TRB bearing type

        spline = Akima(dpt, mpt, delta_x=0.0)
        self.mass, self.dmass_dd = spline.interp(self.lss_diameter)

        # add housing weight
        self.mass += self.mass*(8000.0/2700.0)

        # calculate mass properties
        if self.bearing_switch == 'main':
            c1 = 0.035
        elif self.bearing_switch == 'second':
            c1 = 0.01
        self.cm = np.array([- (c1 * self.rotor_diameter), 0.0, 0.025 * self.rotor_diameter])
        self.c1 = c1

        b1I0 = (self.mass * self.lss_diameter** 2) / 4.0
        self.I = np.array([b1I0, b1I0 / 2.0, b1I0 / 2.0])


    def list_deriv_vars(self):

        inputs = ('lss_diameter', 'rotor_diameter')
        outputs = ('mass', 'cm', 'I')

        return inputs, outputs

    def provideJ(self):

        dmass = np.array([self.dmass_dd*(1 + 8000.0/2700.0), 0.0])
        dcm = np.array([[0.0, -self.c1], [0.0, 0.0], [0.0, 0.025]])
        db1I0 = (self.mass*2*self.lss_diameter + dmass[0]*self.lss_diameter**2)/4.0
        dI = np.array([[db1I0, 0.0], [db1I0/2.0, 0.0], [db1I0/2.0, 0.0]])
        J = vstack([dmass, dcm, dI])

        return J


class BedplateSmooth(Component):
    ''' Bedplate class
          The Bedplate class is used to represent the bedplate of a wind turbine drivetrain.
          It contains the general properties for a wind turbine component as well as additional design load and dimentional attributes as listed below.
          It contains an update method to determine the mass, mass properties, and dimensions of the component.
    '''

    #variables
    hss_location = Float(iotype='in', units='m', desc='HSS CM location')
    hss_mass = Float(iotype='in', units='kg', desc='HSS mass')
    generator_location = Float(iotype='in', units='m', desc='generator CM location')
    generator_mass = Float(iotype='in', units='kg', desc='generator mass')
    lss_location = Float(iotype='in', units='m', desc='LSS CM location')
    lss_mass = Float(iotype='in', units='kg', desc='LSS mass')
    mb1_location = Float(iotype='in', units='m', desc='Upwind main bearing CM location')
    mb1_mass = Float(iotype='in', units='kg', desc='Upwind main bearing mass')
    mb2_location = Float(iotype='in', units='m', desc='Downwind main bearing CM location')
    mb2_mass = Float(iotype='in', units='kg', desc='Downwind main bearing mass')
    tower_top_diameter = Float(iotype='in', units='m', desc='diameter of the top tower section at the yaw gear')
    rotor_diameter = Float(iotype='in', units='m', desc='rotor diameter')
    machine_rating = Float(iotype='in', units='kW', desc='machine_rating machine rating of the turbine')
    rotor_mass = Float(iotype='in', units='kg', desc='rotor mass')
    rotor_bending_moment_y = Float(iotype='in', units='N*m', desc='The bending moment about the y axis')
    rotor_force_z = Float(iotype='in', units='N', desc='The force along the z axis applied at hub center')

    # tf_rear = Float(0.01905, iotype='in')
    # tw_rear = Float(0.0127, iotype='in')
    h0_rear = Float(0.6096, iotype='in')

    # tf_front = Float(0.01905, iotype='in')
    # tw_front = Float(0.0127, iotype='in')
    h0_front = Float(0.6096, iotype='in')

    #parameters
    # uptower_transformer = Bool(iotype='in', desc='Boolean stating if transformer is uptower')

    #outputs
    mass = Float(0.0, iotype='out', units='kg', desc='overall component mass')
    cm = Array(np.array([0.0, 0.0, 0.0]), iotype='out', desc='center of mass of the component in [x,y,z] for an arbitrary coordinate system')
    I = Array(np.array([0.0, 0.0, 0.0]), iotype='out', desc=' moments of Inertia for the component [Ixx, Iyy, Izz] around its center of mass')
    length = Float(iotype='out', units='m', desc='length of bedplate')
    width = Float(iotype='out', units='m', desc='width of bedplate')

    rootStress_margin_rear = Float(iotype='out')
    totalTipDefl_margin_rear = Float(iotype='out')

    rootStress_margin_front = Float(iotype='out')
    totalTipDefl_margin_front = Float(iotype='out')



    def myexec(self, x):

        #Model bedplate as 2 parallel I-beams with a rear steel frame and a front cast frame
        #Deflection constraints applied at each bedplate end
        #Stress constraint checked at root of front and rear bedplate sections

        hss_location = x[0]
        hss_mass = x[1]
        generator_location = x[2]
        generator_mass = x[3]
        lss_location = x[4]
        lss_mass = x[5]
        mb1_location = x[6]
        mb1_mass = x[7]
        mb2_location = x[8]
        mb2_mass = x[9]
        tower_top_diameter = x[10]
        rotor_diameter = x[11]
        machine_rating = x[12]
        rotor_mass = x[13]
        rotor_bending_moment_y = x[14]
        rotor_force_z = x[15]
        h0_rear = x[16]
        h0_front = x[17]

        g = 9.81
        E_rear = 2.1e11
        density = 7800.0

        #rear component weights and locations
        transLoc = 3.0*generator_location
        transformer_mass = 2.4445*(machine_rating) + 1599.0
        convLoc = 2.0*generator_location
        convMass = 0.3*transformer_mass

        rearTotalLength = 0.0

        if transLoc > 0:
            rearTotalLength = transLoc + 1.0
        else:
            rearTotalLength = generator_location + 1.0

        #component masses and locations
        # mb1_location, _ = smooth_abs(mb1_location)
        # mb2_location, _ = smooth_abs(mb2_location)
        # lss_location, _ = smooth_abs(lss_location)

        frontTotalLength = mb1_location + 0.2

        #rotor weights and loads
        rotorLoc = frontTotalLength
        # rotorFz, _ = smooth_abs(rotor_force_z)
        # rotorMy, _ = smooth_abs(rotor_bending_moment_y)
        rotorFz = rotor_force_z
        rotorMy = rotor_bending_moment_y
        rotorLoc=frontTotalLength

        #initial I-beam dimensions
        # h0 = h0_rear
        b0_rear = h0_rear/2.0
        tw_rear = h0_rear/5.0  # /48.0
        tf_rear = h0_rear/5.0  # /32.0

        stressTol = 1e6
        deflTol = 3.5e-3  # todo: model SUPER sensitive to this parameter... modified to achieve agreement with 5 MW turbine for now

        # rootStress = 250e6
        # totalTipDefl = 1.0
        # maxstress = 50e6
        maxstress = 9e6

        def midDeflection(totalLength, loadLength, load, E, I):
            defl = load*loadLength**2.0*(3.0*totalLength - loadLength)/(6.0*E*I)
            return defl

          #tip deflection for distributed load
        def distDeflection(totalLength, distWeight, E, I):
            defl = distWeight*totalLength**4.0/(8.0*E*I)
            return defl


        # while rootStress - 50e6 > stressTol or totalTipDefl - 0.0001 > deflTol:
        bi_rear = (b0_rear-tw_rear)/2.0
        hi_rear = h0_rear-2.0*tf_rear
        I_rear = b0_rear*h0_rear**3/12.0 - 2*bi_rear*hi_rear**3/12.0
        A_rear = b0_rear*h0_rear - 2.0*bi_rear*hi_rear
        w_rear = A_rear*density
        #Tip Deflection for load not at end


        hssTipDefl = midDeflection(rearTotalLength, hss_location, hss_mass*g/2, E_rear, I_rear)
        genTipDefl = midDeflection(rearTotalLength, generator_location, generator_mass*g/2, E_rear, I_rear)
        convTipDefl = midDeflection(rearTotalLength, convLoc, convMass*g/2, E_rear, I_rear)
        transTipDefl = midDeflection(rearTotalLength, transLoc, transformer_mass*g/2, E_rear, I_rear)
        selfTipDefl_rear = distDeflection(rearTotalLength, w_rear*g, E_rear, I_rear)

        totalTipDefl_rear = hssTipDefl + genTipDefl + convTipDefl + transTipDefl + selfTipDefl_rear

        totalTipDefl_margin_rear = (totalTipDefl_rear - 0.0001 - deflTol)/0.0001

        #root stress
        totalBendingMoment_rear = (hss_location*hss_mass + generator_location*generator_mass + convLoc*convMass + transLoc*transformer_mass + w_rear*rearTotalLength**2/2.0)*g
        rootStress_rear = totalBendingMoment_rear*h0_rear/2/I_rear

        rootStress_margin_rear = (rootStress_rear - maxstress - stressTol)/maxstress

        #mass
        steelVolume = A_rear*rearTotalLength
        steelMass = steelVolume*density

        #2 parallel I beams
        totalSteelMass = 2.0*steelMass


        #front cast section

        E_front = 169e9  # EN-GJS-400-18-LT http://www.claasguss.de/html_e/pdf/THBl2_engl.pdf
        castDensity = 7100.0

        # h0 = h0_front
        b0_front = h0_front/2.0
        tw_front = h0_front/5.0  # /48.0
        tf_front = h0_front/5.0  # /32.0


        # rootStress = 250e6
        # totalTipDefl = 1.0
        # counter = 0

        # while rootStress - 50e6 > stressTol or totalTipDefl - 0.0001 > deflTol:

        bi_front = (b0_front-tw_front)/2.0
        hi_front = h0_front-2.0*tf_front
        I_front = b0_front*h0_front**3/12.0 - 2*bi_front*hi_front**3/12.0
        A_front = b0_front*h0_front - 2.0*bi_front*hi_front
        w_front = A_front*castDensity
        #Tip Deflection for load not at end


        mb1TipDefl = midDeflection(frontTotalLength, mb1_location, mb1_mass*g/2.0, E_front, I_front)
        mb2TipDefl = midDeflection(frontTotalLength, mb2_location, mb2_mass*g/2.0, E_front, I_front)
        lssTipDefl = midDeflection(frontTotalLength, lss_location, lss_mass*g/2.0, E_front, I_front)
        rotorTipDefl = midDeflection(frontTotalLength, rotorLoc, rotor_mass*g/2.0, E_front, I_front)
        rotorFzTipDefl = midDeflection(frontTotalLength, rotorLoc, rotorFz/2.0, E_front, I_front)
        selfTipDefl_front = distDeflection(frontTotalLength, w_front*g, E_front, I_front)
        rotorMyTipDefl = rotorMy/2.0*frontTotalLength**2/(2.0*E_front*I_front)

        totalTipDefl_front = mb1TipDefl + mb2TipDefl + lssTipDefl + rotorTipDefl + selfTipDefl_front + rotorMyTipDefl + rotorFzTipDefl

        totalTipDefl_margin_front = (totalTipDefl_front - 0.0001 - deflTol)/0.0001

        #root stress
        totalBendingMoment_front=(mb1_location*mb1_mass/2.0 + mb2_location*mb2_mass/2.0 + lss_location*lss_mass/2.0 + w_front*frontTotalLength**2/2.0 + rotorLoc*rotor_mass/2.0)*g + rotorLoc*rotorFz/2.0 +rotorMy/2.0
        rootStress_front = totalBendingMoment_front*h0_front/2.0/I_front

        rootStress_margin_front = (rootStress_front - maxstress - stressTol)/maxstress

        #mass
        castVolume = A_front*frontTotalLength
        castMass = castVolume*castDensity

        #2 parallel I-beams
        totalCastMass = 2.0*castMass

        front_frame_support_multiplier = 1.33
        totalCastMass *= front_frame_support_multiplier
        mass = totalCastMass + totalSteelMass
        length = frontTotalLength + rearTotalLength
        width = b0_front + tower_top_diameter


        # calculate mass properties
        # cm = np.array([0.0, 0.0, 0.0])
        cm0 = 0.0
        cm1 = 0.0
        cm2 = 0.0122 * rotor_diameter                             # half distance from shaft to yaw axis
        # self.cm = cm

        depth = (length / 2.0)

        # I = np.array([0.0, 0.0, 0.0])
        I0 = mass * (width**2 + depth**2) / 8.0
        I1 = mass * (depth**2 + width**2 + (4.0/3) * length**2) / 16.0
        I2 = I1
        # self.I = I

        out = algopy.zeros(13, dtype=x)
        out[0] = mass
        out[1] = cm0
        out[2] = cm1
        out[3] = cm2
        out[4] = I0
        out[5] = I1
        out[6] = I2
        out[7] = length
        out[8] = width
        out[9] = rootStress_margin_rear
        out[10] = totalTipDefl_margin_rear
        out[11] = rootStress_margin_front
        out[12] = totalTipDefl_margin_front

        return out



    def execute(self):

        mb1_location, _ = smooth_abs(self.mb1_location)
        mb2_location, _ = smooth_abs(self.mb2_location)
        lss_location, _ = smooth_abs(self.lss_location)
        rotor_force_z, _ = smooth_abs(self.rotor_force_z)
        rotor_bending_moment_y, _ = smooth_abs(self.rotor_bending_moment_y)

        x = [self.hss_location, self.hss_mass, self.generator_location, self.generator_mass,
            lss_location, self.lss_mass, mb1_location, self.mb1_mass, mb2_location,
            self.mb2_mass, self.tower_top_diameter, self.rotor_diameter, self.machine_rating,
            self.rotor_mass, rotor_bending_moment_y, rotor_force_z, self.h0_rear, self.h0_front]
        out = self.myexec(x)
        self.mass = out[0]
        self.cm = out[1:4]
        self.I = out[4:7]
        self.length = out[7]
        self.width = out[8]
        self.rootStress_margin_rear = out[9]
        self.totalTipDefl_margin_rear = out[10]
        self.rootStress_margin_front = out[11]
        self.totalTipDefl_margin_front = out[12]



    def list_deriv_vars(self):

        inputs = ('hss_location', 'hss_mass', 'generator_location', 'generator_mass', 'lss_location', 'lss_mass', 'mb1_location', 'mb1_mass', 'mb2_location', 'mb2_mass', 'tower_top_diameter', 'rotor_diameter', 'machine_rating', 'rotor_mass', 'rotor_bending_moment_y', 'rotor_force_z', 'h0_rear', 'h0_front')
        outputs = ('mass', 'cm', 'I', 'length', 'width', 'rootStress_margin_rear', 'totalTipDefl_margin_rear', 'rootStress_margin_front', 'totalTipDefl_margin_front')

        return inputs, outputs

    def provideJ(self):

        mb1_location, dmb1_dmb1 = smooth_abs(self.mb1_location)
        mb2_location, dmb2_dmb2 = smooth_abs(self.mb2_location)
        lss_location, dlss_dlss = smooth_abs(self.lss_location)
        rotor_force_z, drfz_drfz = smooth_abs(self.rotor_force_z)
        rotor_bending_moment_y, drbmy_drbmy = smooth_abs(self.rotor_bending_moment_y)

        x = algopy.UTPM.init_jacobian([self.hss_location, self.hss_mass, self.generator_location, self.generator_mass,
            lss_location, self.lss_mass, mb1_location, self.mb1_mass, mb2_location,
            self.mb2_mass, self.tower_top_diameter, self.rotor_diameter, self.machine_rating,
            self.rotor_mass, rotor_bending_moment_y, rotor_force_z, self.h0_rear, self.h0_front])

        J = algopy.UTPM.extract_jacobian(self.myexec(x))

        J[:, 6] *= dmb1_dmb1
        J[:, 8] *= dmb2_dmb2
        J[:, 4] *= dlss_dlss
        J[:, 15] *= drfz_drfz
        J[:, 14] *= drbmy_drbmy

        return J





class YawSystemSmooth(Component):
    ''' YawSystem class
          The YawSystem class is used to represent the yaw system of a wind turbine drivetrain.
          It contains the general properties for a wind turbine component as well as additional design load and dimentional attributes as listed below.
          It contains an update method to determine the mass, mass properties, and dimensions of the component.
    '''
    #variables
    rotor_diameter = Float(iotype='in', units='m', desc='rotor diameter')
    # rotor_thrust = Float(iotype='in', units='N', desc='maximum rotor thrust')
    tower_top_diameter = Float(iotype='in', units='m', desc='tower top diameter')
    # above_yaw_mass = Float(iotype='in', units='kg', desc='above yaw mass')

    #parameters
    yaw_motors_number = Float(iotype='in', desc='number of yaw motors')

    #outputs
    mass = Float(0.0, iotype='out', units='kg', desc='overall component mass')
    cm = Array(np.array([0.0, 0.0, 0.0]), iotype='out', desc='center of mass of the component in [x,y,z] for an arbitrary coordinate system')
    I = Array(np.array([0.0, 0.0, 0.0]), iotype='out', desc=' moments of Inertia for the component [Ixx, Iyy, Izz] around its center of mass')


    def execute(self):

        # if self.yaw_motors_number == 0 :
        #   if self.rotor_diameter < 90.0 :
        #     self.yaw_motors_number = 4.0
        #   elif self.rotor_diameter < 120.0 :
        #     self.yaw_motors_number = 6.0
        #   else:
        #     self.yaw_motors_number = 8.0

        #assume friction plate surface width is 1/10 the diameter
        #assume friction plate thickness scales with rotor diameter
        frictionPlateVol = pi*self.tower_top_diameter*(self.tower_top_diameter*0.10)*(self.rotor_diameter/1000.0)
        self.steelDensity = 8000.0
        frictionPlateMass = frictionPlateVol*self.steelDensity

        #Assume same yaw motors as Vestas V80 for now: Bonfiglioli 709T2M
        yawMotorMass = 190.0

        totalYawMass = frictionPlateMass + (self.yaw_motors_number*yawMotorMass)
        self.mass = totalYawMass

        # calculate mass properties
        # yaw system assumed to be collocated to tower top center
        self.cm = np.array([0.0, 0.0, 0.0])

        # assuming 0 MOI for yaw system (ie mass is nonrotating)
        self.I = np.array([0.0, 0.0, 0.0])


    def list_deriv_vars(self):

        inputs = ('rotor_diameter', 'tower_top_diameter')
        outputs = ('mass', 'cm', 'I')

        return inputs, outputs

    def provideJ(self):

        dfpm_drotord = pi*self.tower_top_diameter*self.tower_top_diameter*0.10/1000.0*self.steelDensity
        dfpm_dtowertd = pi*2*self.tower_top_diameter*0.10*self.rotor_diameter/1000.0*self.steelDensity

        J = np.zeros((7, 2))
        J[0, :] = [dfpm_drotord, dfpm_dtowertd]

        return J


class GearboxSmooth(Component):
    ''' Gearbox class
          The Gearbox class is used to represent the gearbox component of a wind turbine drivetrain.
          It contains the general properties for a wind turbine component as well as additional design load and dimentional attributes as listed below.
          It contains an update method to determine the mass, mass properties, and dimensions of the component.
    '''

    #variables
    #gbxPower = Float(iotype='in', units='kW', desc='gearbox rated power')
    gear_ratio = Float(iotype='in', desc='overall gearbox speedup ratio')
    Np = Array(np.array([0, 0, 0]), iotype='in', dtype=np.int, desc='number of planets in each stage')
    # rotor_speed = Float(iotype='in', desc='rotor rpm at rated power')
    rotor_diameter = Float(iotype='in', desc='rotor diameter')
    rotor_torque = Float(iotype='in', units='N*m', desc='rotor torque at rated power')

    #parameters
    gear_configuration = Str(iotype='in', desc='string that represents the configuration of the gearbox (stage number and types)')
    ratio_type = Str(iotype='in', desc='optimal or empirical stage ratios')
    shaft_type = Str(iotype='in', desc='normal or short shaft length')

    # outputs
    stage_masses = Array(np.array([0.0, 0.0, 0.0, 0.0]), iotype='out', units='kg', desc='individual gearbox stage masses')
    mass = Float(0.0, iotype='out', units='kg', desc='overall component mass')
    cm = Array(np.array([0.0, 0.0, 0.0]), iotype='out', desc='center of mass of the component in [x,y,z] for an arbitrary coordinate system')
    I = Array(np.array([0.0, 0.0, 0.0]), iotype='out', desc=' moments of Inertia for the component [Ixx, Iyy, Izz] around its center of mass')



    def execute(self):

        self.stageRatio=np.zeros([3, 1])

        self.stageTorque = np.zeros([len(self.stageRatio), 1])  # filled in when ebxWeightEst is called
        self.stageMass = np.zeros([len(self.stageRatio), 1])  # filled in when ebxWeightEst is called
        self.stageType = self.stageTypeCalc(self.gear_configuration)
        self.stageRatio = self.stageRatioCalc(self.gear_ratio, self.Np, self.ratio_type, self.gear_configuration)


        m = self.gbxWeightEst(self.gear_configuration, self.gear_ratio, self.Np, self.ratio_type, self.shaft_type, self.rotor_torque)
        self.mass = float(m)
        self.stage_masses=self.stageMass
        # calculate mass properties
        cm0 = 0.0
        cm1 = cm0
        cm2 = 0.025 * self.rotor_diameter
        self.cm = np.array([cm0, cm1, cm2])

        length = (0.012 * self.rotor_diameter)
        height = (0.015 * self.rotor_diameter)
        diameter = (0.75 * height)

        I0 = self.mass * (diameter ** 2) / 8 + (self.mass / 2) * (height ** 2) / 8
        I1 = self.mass * (0.5 * (diameter ** 2) + (2 / 3) * (length ** 2) + 0.25 * (height ** 2)) / 8
        I2 = I1
        self.I = np.array([I0, I1, I2])

        '''def rotor_torque():
            # tq = self.gbxPower*1000 / self.eff / (self.rotor_speed * (pi / 30.0))
            return tq
        '''

    def stageTypeCalc(self, config):
        temp=[]
        for character in config:
            if character == 'e':
                temp.append(2)
            if character == 'p':
                temp.append(1)
        return temp


    def stageRatioCalc(self, overallRatio, Np, ratio_type, config):
        '''
        Calculates individual stage ratios using either empirical relationships from the Sunderland model or a SciPy constrained optimization routine.
        '''
        K_r=0

        #Assumes we can model everything w/Sunderland model to estimate speed ratio
        if ratio_type == 'empirical':
            if config == 'p':
                x = [overallRatio]
            if config == 'e':
                x = [overallRatio]
            elif config == 'pp':
                x = [overallRatio**0.5, overallRatio**0.5]
            elif config == 'ep':
                x = [overallRatio/2.5, 2.5]
            elif config =='ee':
                x = [overallRatio**0.5, overallRatio**0.5]
            elif config == 'eep':
                x = [(overallRatio/3)**0.5, (overallRatio/3)**0.5, 3]
            elif config == 'epp':
                x = [overallRatio**(1.0/3.0), overallRatio**(1.0/3.0), overallRatio**(1.0/3.0)]
            elif config == 'eee':
                x = [overallRatio**(1.0/3.0), overallRatio**(1.0/3.0), overallRatio**(1.0/3.0)]
            elif config == 'ppp':
                x = [overallRatio**(1.0/3.0), overallRatio**(1.0/3.0), overallRatio**(1.0/3.0)]

        elif ratio_type == 'optimal':
            x = np.zeros([3, 1])

            if config == 'eep':
                x0=[overallRatio**(1.0/3.0), overallRatio**(1.0/3.0), overallRatio**(1.0/3.0)]
                B_1=Np[0]
                B_2=Np[1]
                K_r1=0
                K_r2=0  # 2nd stage structure weight coefficient

                def volume(x):
                    return (1.0/(x[0]))*((1.0/B_1)+(1.0/(B_1*((x[0]/2.0)-1.0)))+(x[0]/2.0-1.0)+(x[0]/2.0-1)**2+K_r1*((x[0]-1.0)**2)/B_1 + K_r1*((x[0]-1.0)**2)/(B_1*(x[0]/2.0-1.0))) + (1.0/(x[0]*x[1]))*((1.0/B_2)+(1/(B_2*((x[1]/2.0)-1.0)))+(x[1]/2.0-1.0)+(x[1]/2.0-1.0)**2.0+K_r2*((x[1]-1.0)**2.0)/B_2 + K_r2*((x[1]-1.0)**2.0)/(B_2*(x[1]/2.0-1.0))) + (1.0/(x[0]*x[1]*x[2]))*(1.0+(1.0/x[2])+x[2] + x[2]**2)

                def constr1(x, overallRatio):
                    return x[0]*x[1]*x[2]-overallRatio

                def constr2(x, overallRatio):
                    return overallRatio-x[0]*x[1]*x[2]

                x = fmin_cobyla(volume, x0, [constr1, constr2], consargs=[overallRatio], rhoend=1e-8, iprint=0)

            elif config == 'eep_3':
                #fixes last stage ratio at 3
                x0=[overallRatio**(1.0/3.0), overallRatio**(1.0/3.0), overallRatio**(1.0/3.0)]
                B_1=Np[0]
                B_2=Np[1]
                K_r1=0
                K_r2=0.8  # 2nd stage structure weight coefficient

                def volume(x):
                    return (1.0/(x[0]))*((1.0/B_1)+(1.0/(B_1*((x[0]/2.0)-1.0)))+(x[0]/2.0-1.0)+(x[0]/2.0-1)**2+K_r1*((x[0]-1.0)**2)/B_1 + K_r1*((x[0]-1.0)**2)/(B_1*(x[0]/2.0-1.0))) + (1.0/(x[0]*x[1]))*((1.0/B_2)+(1/(B_2*((x[1]/2.0)-1.0)))+(x[1]/2.0-1.0)+(x[1]/2.0-1.0)**2.0+K_r2*((x[1]-1.0)**2.0)/B_2 + K_r2*((x[1]-1.0)**2.0)/(B_2*(x[1]/2.0-1.0))) + (1.0/(x[0]*x[1]*x[2]))*(1.0+(1.0/x[2])+x[2] + x[2]**2)

                def constr1(x, overallRatio):
                    return x[0]*x[1]*x[2]-overallRatio

                def constr2(x, overallRatio):
                    return overallRatio-x[0]*x[1]*x[2]

                def constr3(x, overallRatio):
                    return x[2]-3.0

                def constr4(x, overallRatio):
                    return 3.0-x[2]

                x = fmin_cobyla(volume,  x0, [constr1, constr2, constr3, constr4], consargs=[overallRatio], rhoend=1e-8, iprint=0)

            elif config == 'eep_2':
                #fixes final stage ratio at 2
                x0=[overallRatio**(1.0/3.0), overallRatio**(1.0/3.0), overallRatio**(1.0/3.0)]
                B_1=Np[0]
                B_2=Np[1]
                K_r1=0
                K_r2=1.6  # 2nd stage structure weight coefficient

                def volume(x):
                    return (1.0/(x[0]))*((1.0/B_1)+(1.0/(B_1*((x[0]/2.0)-1.0)))+(x[0]/2.0-1.0)+(x[0]/2.0-1)**2+K_r1*((x[0]-1.0)**2)/B_1 + K_r1*((x[0]-1.0)**2)/(B_1*(x[0]/2.0-1.0))) + (1.0/(x[0]*x[1]))*((1.0/B_2)+(1/(B_2*((x[1]/2.0)-1.0)))+(x[1]/2.0-1.0)+(x[1]/2.0-1.0)**2.0+K_r2*((x[1]-1.0)**2.0)/B_2 + K_r2*((x[1]-1.0)**2.0)/(B_2*(x[1]/2.0-1.0))) + (1.0/(x[0]*x[1]*x[2]))*(1.0+(1.0/x[2])+x[2] + x[2]**2)

                def constr1(x, overallRatio):
                    return x[0]*x[1]*x[2]-overallRatio

                def constr2(x, overallRatio):
                    return overallRatio-x[0]*x[1]*x[2]

                x = fmin_cobyla(volume, x0, [constr1, constr2], consargs=[overallRatio], rhoend=1e-8, iprint=0)

            else:
                x0=[overallRatio**(1.0/3.0), overallRatio**(1.0/3.0), overallRatio**(1.0/3.0)]
                B_1=Np[0]
                K_r=0.0

                def volume(x):
                    return (1.0/(x[0]))*((1.0/B_1)+(1.0/(B_1*((x[0]/2.0)-1.0)))+(x[0]/2.0-1)+(x[0]/2.0-1.0)**2+K_r*((x[0]-1.0)**2)/B_1 + K_r*((x[0]-1)**2)/(B_1*(x[0]/2.0-1.0))) + (1.0/(x[0]*x[1]))*(1.0+(1.0/x[1])+x[1] + x[1]**2)+ (1.0/(x[0]*x[1]*x[2]))*(1.0+(1.0/x[2])+x[2] + x[2]**2)

                def constr1(x, overallRatio):
                    return x[0]*x[1]*x[2]-overallRatio

                def constr2(x, overallRatio):
                    return overallRatio-x[0]*x[1]*x[2]

                x = fmin_cobyla(volume, x0, [constr1, constr2], consargs=[overallRatio], rhoend=1e-8, iprint=0)
        else:
            x ='fail'

        return x


    def gbxWeightEst(self, config, overallRatio, Np, ratio_type, shaft_type, torque):
        '''
        Computes the gearbox weight based on a surface durability criteria.
        '''

        ## Define Application Factors ##
        #Application factor for weight estimate
        Ka=0.6
        Kshaft=0.0
        Kfact=0.0

        #K factor for pitting analysis
        if self.rotor_torque < 200000.0:
            Kfact = 850.0
        elif self.rotor_torque < 700000.0:
            Kfact = 950.0
        else:
            Kfact = 1100.0

        #Unit conversion from Nm to inlb and vice-versa
        Kunit=8.029

        # Shaft length factor
        if self.shaft_type == 'normal':
            Kshaft = 1.0
        elif self.shaft_type == 'short':
            Kshaft = 1.25

        #Individual stage torques
        torqueTemp=self.rotor_torque
        for s in range(len(self.stageRatio)):
            #print torqueTemp
            #print self.stageRatio[s]
            self.stageTorque[s]=torqueTemp/self.stageRatio[s]
            torqueTemp=self.stageTorque[s]
            self.stageMass[s]=Kunit*Ka/Kfact*self.stageTorque[s]*self.stageMassCalc(self.stageRatio[s], self.Np[s], self.stageType[s])

        gbxWeight=(sum(self.stageMass))*Kshaft

        return gbxWeight


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

        #Application factor to include ring/housing/carrier weight
        Kr=0.4
        Kgamma=1.1

        if indNp == 3:
            Kgamma=1.1
        elif indNp == 4:
            Kgamma=1.1
        elif indNp == 5:
            Kgamma=1.35

        if indStageType == 1:
            indStageMass=1.0+indStageRatio+indStageRatio**2+(1.0/indStageRatio)

        elif indStageType == 2:
            sunRatio=0.5*indStageRatio - 1.0
            indStageMass=Kgamma*((1/indNp)+(1/(indNp*sunRatio))+sunRatio+sunRatio**2+Kr*((indStageRatio-1)**2)/indNp+Kr*((indStageRatio-1)**2)/(indNp*sunRatio))

        return indStageMass




class LowSpeedShaftDrive4ptSmooth(Component):
    ''' LowSpeedShaft class
          The LowSpeedShaft class is used to represent the low speed shaft component of a wind turbine drivetrain.
          It contains the general properties for a wind turbine component as well as additional design load and dimentional attributes as listed below.
          It contains an update method to determine the mass, mass properties, and dimensions of the component.
    '''

    # variables
    rotor_bending_moment_x = Float(iotype='in', units='N*m', desc='The bending moment about the x axis')
    rotor_bending_moment_y = Float(iotype='in', units='N*m', desc='The bending moment about the y axis')
    rotor_bending_moment_z = Float(iotype='in', units='N*m', desc='The bending moment about the z axis')
    rotor_force_x = Float(iotype='in', units='N', desc='The force along the x axis applied at hub center')
    rotor_force_y = Float(iotype='in', units='N', desc='The force along the y axis applied at hub center')
    rotor_force_z = Float(iotype='in', units='N', desc='The force along the z axis applied at hub center')
    rotor_mass = Float(iotype='in', units='kg', desc='rotor mass')
    rotor_diameter = Float(iotype='in', units='m', desc='rotor diameter')
    # machine_rating = Float(iotype='in', units='kW', desc='machine_rating machine rating of the turbine')
    gearbox_mass = Float(iotype='in', units='kg', desc='Gearbox mass')
    # carrier_mass = Float(iotype='in', units='kg', desc='Carrier mass')

    L_ms = Float(iotype='in')
    L_mb = Float(iotype='in')
    # L_ms_gb = Float(iotype='in')

    # parameters
    shrink_disc_mass = Float(iotype='in', units='kg', desc='Mass of the shrink disc')  # shrink disk or flange addtional mass
    shaft_angle = Float(iotype='in', units='deg', desc='Angle of the LSS inclindation with respect to the horizontal')
    shaft_ratio = Float(iotype='in', desc='Ratio of inner diameter to outer diameter.  Leave zero for solid LSS')
    mb1Type = Str(iotype='in', desc='Main bearing type: CARB, TRB or SRB')
    mb2Type = Str(iotype='in', desc='Second bearing type: CARB, TRB or SRB')

    # outputs
    # design_torque = Float(iotype='out', units='N*m', desc='lss design torque')
    # design_bending_load = Float(iotype='out', units='N', desc='lss design bending load')
    length = Float(iotype='out', units='m', desc='lss length')
    diameter1 = Float(iotype='out', units='m', desc='lss outer diameter at main bearing')
    diameter2 = Float(iotype='out', units='m', desc='lss outer diameter at second bearing')
    mass = Float(0.0, iotype='out', units='kg', desc='overall component mass')
    cm = Array(np.array([0.0, 0.0, 0.0]), iotype='out', desc='center of mass of the component in [x,y,z] for an arbitrary coordinate system')
    I = Array(np.array([0.0, 0.0, 0.0]), iotype='out', desc=' moments of Inertia for the component [Ixx, Iyy, Izz] around its center of mass')

    sizing_constraints = Array(iotype='out')


    def execute(self):
        #Hub Forces
        # F_r_x = self.rotor_force_x            # External F_x
        F_r_y = self.rotor_force_y                 # External F_y
        F_r_z = self.rotor_force_z                  # External F_z
        M_r_x = self.rotor_bending_moment_x
        M_r_y = self.rotor_bending_moment_y
        M_r_z = self.rotor_bending_moment_z

        #input parameters
        g=9.81
        gamma=self.shaft_angle  # deg LSS angle wrt horizontal
        # PSF=1


        # initialization for iterations
        L_ms = self.L_ms
        # L_ms_new = 0.0
        # L_ms_0 = 0.5  # main shaft length downwind of main bearing
        # L_ms=L_ms_0
        tol=1e-4
        # check_limit = 1.0
        # dL=0.05
        # counter = 0
        # N_count=100
        # N_count_2=2
        len_pts=101
        D_max=1
        D_min=0.2
        sR = self.shaft_ratio

        #Distances
        L_rb = 1.912        # distance from hub center to main bearing   # to add as an input
        L_bg = 6.11         # distance from hub center to gearbox yokes  # to add as an input
        L_as = L_ms/2.0     # distance from main bearing to shaft center
        L_gb = 0.0          # distance to gbx center from trunnions in x-dir # to add as an input
        H_gb = 1.0          # distance to gbx center from trunnions in z-dir # to add as an input
        # L_gp = 0.825        # distance from gbx coupling to gbx trunnions
        # L_cu = L_ms + 0.5   # distance from upwind main bearing to upwind carrier bearing 0.5 meter is an estimation # to add as an input
        # L_cd = L_cu + 0.5   # distance from upwind main bearing to downwind carrier bearing 0.5 meter is an estimation # to add as an input

        #material properties
        E=2.1e11
        density=7800.0
        n_safety = 2.5  # According to AGMA, takes into account the peak load safety factor
        Sy = 66000  # psi

        #unit conversion
        u_knm_inlb = 8850.745454036
        u_in_m = 0.0254000508001

        #bearing deflection limits
        # MB_limit = 0.026
        # CB_limit = 4.0/60.0/180.0*pi
        TRB_limit = 3.0/60.0/180.0*pi
        n_safety_brg = 1.0

        # while abs(check_limit) > tol and counter <N_count:

        # # counter = counter+1
        # if L_ms_new > 0:
        #     L_ms=L_ms_new
        # else:
        #     L_ms=L_ms_0

        #Distances
        L_as = L_ms/2.0     # distance from main bearing to shaft center
        # L_cu = L_ms + 0.5   # distance from upwind main bearing to upwind carrier bearing 0.5 meter is an estimation # to add as an input
        # L_cd = L_cu + 0.5   # distance from upwind main bearing to downwind carrier bearing 0.5 meter is an estimation # to add as an input

        #Weight properties
        rotorWeight=self.rotor_mass*g                             # rotor weight
        lssWeight = pi/3.0*(D_max**2 + D_min**2 + D_max*D_min)*L_ms*density*g/4.0
        # lss_mass = lssWeight/g
        gbxWeight = self.gearbox_mass*g                               # gearbox weight
        # carrierWeight = self.carrier_mass*g                       # carrier weight
        shrinkDiscWeight = self.shrink_disc_mass*g

        #define LSS
        x_ms = np.linspace(L_rb, L_ms+L_rb, len_pts)
        x_rb = np.linspace(0.0, L_rb, len_pts)
        # y_gp = np.linspace(0, L_gp, len_pts)

        # F_mb_x = -F_r_x - rotorWeight*sin(radians(gamma))
        F_mb_y = M_r_z/L_bg - F_r_y*(L_bg + L_rb)/L_bg
        F_mb_z = (-M_r_y + rotorWeight*(cos(radians(gamma))*(L_rb + L_bg)
            + sin(radians(gamma))*H_gb) + lssWeight*(L_bg - L_as)
            * cos(radians(gamma)) + shrinkDiscWeight*cos(radians(gamma))
            * (L_bg - L_ms) - gbxWeight*cos(radians(gamma))*L_gb - F_r_z*cos(radians(gamma))*(L_bg + L_rb))/L_bg

        # F_gb_x = -(lssWeight+shrinkDiscWeight+gbxWeight)*sin(radians(gamma))
        # F_gb_y = -F_mb_y - F_r_y
        # F_gb_z = -F_mb_z + (shrinkDiscWeight+rotorWeight+gbxWeight + lssWeight)*cos(radians(gamma)) - F_r_z

        My_ms = np.zeros(2*len_pts)
        Mz_ms = np.zeros(2*len_pts)

        for k in range(len_pts):
            My_ms[k] = -M_r_y + rotorWeight*cos(radians(gamma))*x_rb[k] + 0.5*lssWeight/L_ms*x_rb[k]**2 - F_r_z*x_rb[k]
            Mz_ms[k] = -M_r_z - F_r_y*x_rb[k]

        for j in range(len_pts):
            My_ms[j+len_pts] = -F_r_z*x_ms[j] - M_r_y + rotorWeight*cos(radians(gamma))*x_ms[j] - F_mb_z*(x_ms[j]-L_rb) + 0.5*lssWeight/L_ms*x_ms[j]**2
            Mz_ms[j+len_pts] = -M_r_z - F_mb_y*(x_ms[j]-L_rb) -F_r_y*x_ms[j]

        # x_shaft = np.concatenate([x_rb, x_ms])

        MM_max=np.amax((My_ms**2+Mz_ms**2)**0.5)
        # Index=np.argmax((My_ms**2+Mz_ms**2)**0.5)

        MM_min = ((My_ms[-1]**2+Mz_ms[-1]**2)**0.5)
        #Design shaft OD
        MM=MM_max
        D_max=(16.0*n_safety/pi/Sy*(4.0*(MM*u_knm_inlb/1000)**2+3.0*(M_r_x*u_knm_inlb/1000)**2)**0.5)**(1.0/3.0)*u_in_m

        #OD at end
        MM=MM_min
        D_min=(16.0*n_safety/pi/Sy*(4.0*(MM*u_knm_inlb/1000)**2+3.0*(M_r_x*u_knm_inlb/1000)**2)**0.5)**(1.0/3.0)*u_in_m

        #Estimate ID
        D_in=sR*D_max
        D_max = (D_max**4 + D_in**4)**0.25
        D_min = (D_min**4 + D_in**4)**0.25

        lssWeight_new=((pi/3)*(D_max**2+D_min**2+D_max*D_min)*(L_ms)*density/4+(-pi/4*(D_in**2)*density*(L_ms)))*g

        def deflection(F_r_z, W_r, gamma, M_r_y, f_mb_z, L_rb, W_ms, L_ms, z):
            return -F_r_z*z**3/6.0 + W_r*cos(radians(gamma))*z**3/6.0 - M_r_y*z**2/2.0 - f_mb_z*(z-L_rb)**3/6.0 + W_ms/(L_ms + L_rb)/24.0*z**4


        D1 = deflection(F_r_z, rotorWeight, gamma, M_r_y, F_mb_z, L_rb, lssWeight_new, L_ms, L_rb+L_ms)
        D2 = deflection(F_r_z, rotorWeight, gamma, M_r_y, F_mb_z, L_rb, lssWeight_new, L_ms, L_rb)
        C1 = -(D1-D2)/L_ms
        C2 = D2-C1*(L_rb)

        I_2=pi/64.0*(D_max**4 - D_in**4)

        def gx(F_r_z, W_r, gamma, M_r_y, f_mb_z, L_rb, W_ms, L_ms, C1, z):
            return -F_r_z*z**2/2.0 + W_r*cos(radians(gamma))*z**2/2.0 - M_r_y*z - f_mb_z*(z-L_rb)**2/2.0 + W_ms/(L_ms + L_rb)/6.0*z**3 + C1

        theta_y = np.zeros(len_pts)
        d_y = np.zeros(len_pts)

        for kk in range(len_pts):
            theta_y[kk]=gx(F_r_z, rotorWeight, gamma, M_r_y, F_mb_z, L_rb, lssWeight_new, L_ms, C1, x_ms[kk])/E/I_2
            d_y[kk]=(deflection(F_r_z, rotorWeight, gamma, M_r_y, F_mb_z, L_rb, lssWeight_new, L_ms, x_ms[kk])+C1*x_ms[kk]+C2)/E/I_2

        # check_limit = abs(abs(theta_y[-1])-TRB_limit/n_safety_brg)

        self.sizing_constraints = np.array([theta_y[-1] - TRB_limit/n_safety_brg - tol,
            -theta_y[-1] + TRB_limit/n_safety_brg - tol])

        # if check_limit < 0:
        #     L_ms_new = L_ms + dL
        # else:
        #     L_ms_new = L_ms + dL




         #Initialization
        # L_mb=L_ms_new
        # counter_ms=0
        # check_limit_ms=1.0
        # L_mb_new=0.0
        # L_mb_0=L_mb                     # main shaft length
        # L_ms = L_ms_new
        # dL_ms = 0.05
        # dL = 0.0025

        # while abs(check_limit_ms)>tol and counter_ms<N_count:
            # counter_ms = counter_ms + 1
            # if L_mb_new > 0:
            #     L_mb=L_mb_new
            # else:
            #     L_mb=L_mb_0

        # counter = 0.0
        # check_limit=1.0
        # L_ms_gb_new=0.0
        # L_ms_0=0.5  # mainshaft length
        # L_ms = L_ms_0

            # while abs(check_limit) > tol and counter <N_count_2:
            #     counter = counter+1
            #     if L_ms_gb_new>0.0:
            #         L_ms_gb = L_ms_gb_new
            #     else:
            #         L_ms_gb = L_ms_0

        L_mb = self.L_mb
        # L_ms_gb = self.L_ms_gb
        L_ms_gb = 0.5  # this appears to be set in the original version of the code (L_ms_gb_new was never updated).

        #Distances
        L_as = (L_ms_gb+L_mb)/2.0
        # L_cu = (L_ms_gb + L_mb) + 0.5
        # L_cd = L_cu + 0.5

        #Weight
        lssWeight_new=((pi/3)*(D_max**2+D_min**2+D_max*D_min)*(L_ms_gb + L_mb)*density/4+(-pi/4*(D_in**2)*density*(L_ms_gb + L_mb)))*g

        #define LSS
        x_ms = np.linspace(L_rb + L_mb, L_ms_gb + L_mb +L_rb, len_pts)
        x_mb = np.linspace(L_rb, L_mb+L_rb, len_pts)
        x_rb = np.linspace(0.0, L_rb, len_pts)
        # y_gp = np.linspace(0, L_gp, len_pts)

        # F_mb2_x = -F_r_x - rotorWeight*sin(radians(gamma))
        F_mb2_y = -M_r_z/L_mb + F_r_y*(L_rb)/L_mb
        F_mb2_z = (M_r_y - rotorWeight*cos(radians(gamma))*L_rb
            -lssWeight*L_as*cos(radians(gamma)) - shrinkDiscWeight*L_ms*cos(radians(gamma))
            + gbxWeight*cos(radians(gamma))*L_gb + F_r_z*cos(radians(gamma))*L_rb)/L_mb

        # F_mb1_x = 0.0
        F_mb1_y = -F_r_y - F_mb2_y
        F_mb1_z = (rotorWeight + lssWeight + shrinkDiscWeight)*cos(radians(gamma)) - F_r_z - F_mb2_z

        # F_gb_x = -(lssWeight+shrinkDiscWeight+gbxWeight)*sin(radians(gamma))
        # F_gb_y = -F_mb_y - F_r_y
        # F_gb_z = -F_mb_z + (shrinkDiscWeight+rotorWeight+gbxWeight + lssWeight)*cos(radians(gamma)) - F_r_z

        My_ms = np.zeros(3*len_pts)
        Mz_ms = np.zeros(3*len_pts)

        for k in range(len_pts):
            My_ms[k] = -M_r_y + rotorWeight*cos(radians(gamma))*x_rb[k] + 0.5*lssWeight/L_ms*x_rb[k]**2 - F_r_z*x_rb[k]
            Mz_ms[k] = -M_r_z - F_r_y*x_rb[k]

        for j in range(len_pts):
            My_ms[j+len_pts] = -F_r_z*x_mb[j] - M_r_y + rotorWeight*cos(radians(gamma))*x_mb[j] - F_mb1_z*(x_mb[j]-L_rb) + 0.5*lssWeight/L_ms*x_mb[j]**2
            Mz_ms[j+len_pts] = -M_r_z - F_mb1_y*(x_mb[j]-L_rb) -F_r_y*x_mb[j]

        for l in range(len_pts):
            My_ms[l + 2*len_pts] = -F_r_z*x_ms[l] - M_r_y + rotorWeight*cos(radians(gamma))*x_ms[l] - F_mb1_z*(x_ms[l]-L_rb) -F_mb2_z*(x_ms[l] - L_rb - L_mb) + 0.5*lssWeight/L_ms*x_ms[l]**2
            Mz_ms[l + 2*len_pts] = -M_r_z - F_mb_y*(x_ms[l]-L_rb) -F_r_y*x_ms[l]

        # x_shaft = np.concatenate([x_rb, x_mb, x_ms])

        MM_max=np.amax((My_ms**2+Mz_ms**2)**0.5)
        # Index=np.argmax((My_ms**2+Mz_ms**2)**0.5)

        MM_min = ((My_ms[-1]**2+Mz_ms[-1]**2)**0.5)

        MM_med = ((My_ms[-1 - len_pts]**2 + Mz_ms[-1 - len_pts]**2)**0.5)

        #Design Shaft OD using static loading and distortion energy theory
        MM=MM_max
        D_max=(16.0*n_safety/pi/Sy*(4.0*(MM*u_knm_inlb/1000)**2+3.0*(M_r_x*u_knm_inlb/1000)**2)**0.5)**(1.0/3.0)*u_in_m

        #OD at end
        MM=MM_min
        D_min=(16.0*n_safety/pi/Sy*(4.0*(MM*u_knm_inlb/1000)**2+3.0*(M_r_x*u_knm_inlb/1000)**2)**0.5)**(1.0/3.0)*u_in_m

        MM=MM_med
        D_med=(16.0*n_safety/pi/Sy*(4.0*(MM*u_knm_inlb/1000)**2+3.0*(M_r_x*u_knm_inlb/1000)**2)**0.5)**(1.0/3.0)*u_in_m

        #Estimate ID
        D_in=sR*D_max
        D_max = (D_max**4 + D_in**4)**0.25
        D_min = (D_min**4 + D_in**4)**0.25
        D_med = (D_med**4 + D_in**4)**0.25

        lssWeight_new = (density*pi/12.0*L_mb*(D_max**2+D_med**2 + D_max*D_med) - density*pi/4.0*D_in**2*L_mb)*g

        #deflection between mb1 and mb2
        def deflection1(F_r_z, W_r, gamma, M_r_y, f_mb1_z, L_rb, W_ms, L_ms, L_mb, z):
            return -F_r_z*z**3/6.0 + W_r*cos(radians(gamma))*z**3/6.0 - M_r_y*z**2/2.0 - f_mb1_z*(z-L_rb)**3/6.0 + W_ms/(L_ms + L_mb)/24.0*z**4

        D11 = deflection1(F_r_z, rotorWeight, gamma, M_r_y, F_mb1_z, L_rb, lssWeight_new, L_ms, L_mb, L_rb+L_mb)
        D21 = deflection1(F_r_z, rotorWeight, gamma, M_r_y, F_mb1_z, L_rb, lssWeight_new, L_ms, L_mb, L_rb)
        C11 = -(D11-D21)/L_mb
        C21 = -D21-C11*(L_rb)

        I_2=pi/64.0*(D_max**4 - D_in**4)

        def gx1(F_r_z, W_r, gamma, M_r_y, f_mb1_z, L_rb, W_ms, L_ms, L_mb, C11, z):
            return -F_r_z*z**2/2.0 + W_r*cos(radians(gamma))*z**2/2.0 - M_r_y*z - f_mb1_z*(z - L_rb)**2/2.0 + W_ms/(L_ms + L_mb)/6.0*z**3 + C11

        theta_y = np.zeros(2*len_pts)
        d_y = np.zeros(2*len_pts)

        for kk in range(len_pts):
            theta_y[kk]=gx1(F_r_z, rotorWeight, gamma, M_r_y, F_mb1_z, L_rb, lssWeight_new, L_ms, L_mb, C11, x_mb[kk])/E/I_2
            d_y[kk]=(deflection1(F_r_z, rotorWeight, gamma, M_r_y, F_mb1_z, L_rb, lssWeight_new, L_ms, L_mb, x_mb[kk])+C11*x_mb[kk]+C21)/E/I_2


        #Deflection between mb2 and gbx
        def deflection2(F_r_z, W_r, gamma, M_r_y, f_mb1_z, f_mb2_z, L_rb, W_ms, L_ms, L_mb, z):
            return -F_r_z*z**3/6.0 + W_r*cos(radians(gamma))*z**3/6.0 - M_r_y*z**2/2.0 - f_mb1_z*(z-L_rb)**3/6.0 + -f_mb2_z*(z - L_rb - L_mb)**3/6.0 + W_ms/(L_ms + L_mb)/24.0*z**4

        def gx2(F_r_z, W_r, gamma, M_r_y, f_mb1_z, f_mb2_z, L_rb, W_ms, L_ms, L_mb, z):
            return -F_r_z*z**2/2.0 + W_r*cos(radians(gamma))*z**2/2.0 - M_r_y*z - f_mb1_z*(z - L_rb)**2/2.0 - f_mb2_z*(z - L_rb - L_mb)**2/2.0 + W_ms/(L_ms + L_mb)/6.0*z**3

        D12 = deflection2(F_r_z, rotorWeight, gamma, M_r_y, F_mb1_z, F_mb2_z, L_rb, lssWeight_new, L_ms, L_mb, L_rb+L_mb)
        D22 = gx2(F_r_z, rotorWeight, gamma, M_r_y, F_mb1_z, F_mb2_z, L_rb, lssWeight_new, L_ms, L_mb, L_rb+L_mb)
        C12 = gx1(F_r_z, rotorWeight, gamma, M_r_y, F_mb1_z, L_rb, lssWeight_new, L_ms, L_mb, C11, x_mb[-1])-D22
        C22 = -D12-C12*(L_rb + L_mb)

        for kk in range(len_pts):
            theta_y[kk + len_pts]=(gx2(F_r_z, rotorWeight, gamma, M_r_y, F_mb1_z, F_mb2_z, L_rb, lssWeight_new, L_ms, L_mb, x_ms[kk]) + C12)/E/I_2
            d_y[kk + len_pts]=(deflection2(F_r_z, rotorWeight, gamma, M_r_y, F_mb1_z, F_mb2_z, L_rb, lssWeight_new, L_ms, L_mb, x_ms[kk])+C12*x_ms[kk]+C22)/E/I_2

        self.sizing_constraints = np.concatenate([self.sizing_constraints,
            [theta_y[-1] - TRB_limit/n_safety_brg - tol,
            -theta_y[-1] + TRB_limit/n_safety_brg - tol]])

        self.sizing_constraints /= TRB_limit  # for normalization


        # check_limit = abs(abs(theta_y[-1])-TRB_limit/n_safety_brg)

        # if check_limit < 0:
        #     L_ms_gb_new = L_ms_gb + dL
        # else:
        #     L_ms_gb_new = L_ms_gb + dL

        # check_limit_ms = abs(abs(theta_y[-1]) - TRB_limit/n_safety_brg)

        # if check_limit_ms < 0:
        #     L_mb_new = L_mb + dL_ms
        # else:
        #     L_mb_new = L_mb + dL_ms

        [D_max_a, FW_max] = resize_for_bearings(D_max, self.mb1Type)

        [D_med_a, FW_med] = resize_for_bearings(D_med, self.mb2Type)

        lss_mass_new=(pi/3)*(D_max_a**2+D_med_a**2+D_max_a*D_med_a)*(L_mb-(FW_max+FW_med)/2)*density/4+ \
            (pi/4)*(D_max_a**2-D_in**2)*density*FW_max+ \
            (pi/4)*(D_med_a**2-D_in**2)*density*FW_med- \
            (pi/4)*(D_in**2)*density*(L_mb+(FW_max+FW_med)/2)
        lss_mass_new *= 1.3  # add flange and shrink disk mass
        self.length= L_mb + (FW_max+FW_med)/2  # TODO: create linear relationship based on power rating

        self.D_outer=D_max
        self.D_inner=D_in
        self.mass=lss_mass_new
        self.diameter1= D_max_a
        self.diameter2= D_med_a

        # calculate mass properties
        cm = np.array([0.0, 0.0, 0.0])
        cm[0] = -(0.035 - 0.01) * self.rotor_diameter            # cm based on WindPACT work - halfway between locations of two main bearings
        cm[1] = 0.0
        cm[2] = 0.025 * self.rotor_diameter
        self.cm = cm

        I = np.array([0.0, 0.0, 0.0])
        I[0] = self.mass * (self.D_inner**2.0 + self.D_outer**2.0) / 8.0
        I[1] = self.mass * (self.D_inner**2.0 + self.D_outer**2.0 + (4.0 / 3.0) * (self.length**2.0)) / 16.0
        I[2] = I[1]
        self.I = I



def resize_for_bearings(D_mb, mbtype):
    # Internal function to resize shaft for bearings - for Yi to add content (using lookup table etc)
    # To add bearing load capacity check later
    '''D_mb1 = 1.25
      D_mb2 = 0.75
      FW_mb1=0.45
      FW_mb2=0.5
    '''

    d_pt = [0.0, 0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.9, 0.95, 1.1, 1.2, 1.3, 1.4]

    if mbtype == 'CARB':
        da_pt = [0.3, 0.3, 0.3, 0.3, 0.4, 0.5, 0.6, 0.71, 0.8, 0.95, 1.0, 1.25, 1.25, 1.25, 1.25]
        # fwpt = [0.2, 0.2, 0.2, 0.2, 0.2, 0.325, 0.375, 0.345, 0.375, 0.3, 0.375, 0.45, 0.45, 0.45, 0.45]
        fw_pt = [0.2, 0.2, 0.2, 0.2, 0.2, 0.325, 0.375, 0.375, 0.375, 0.375, 0.375, 0.45, 0.45, 0.45, 0.45]

    elif mbtype == 'SRB':

        # dapt = [0.3, 0.3, 0.3, 0.3, 0.4, 0.5, 0.6, 0.75, 0.8, 0.95, 1, 1.25, 1.25, 1.25, 1.25]
        da_pt = [0.3, 0.3, 0.3, 0.3, 0.4, 0.5, 0.6, 0.71, 0.8, 0.95, 1, 1.25, 1.25, 1.25, 1.25]
        # fwpt = [0.2, 0.2, 0.2, 0.2, 0.25, 0.325, 0.375, 0.44, 0.475, 0.525, 0.5, 0.5, 0.5, 0.5, 0.5]
        fw_pt = [0.2, 0.2, 0.2, 0.2, 0.25, 0.325, 0.375, 0.44, 0.475, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]

    da_spline = Akima(d_pt, da_pt, delta_x=0.0)
    D_mb_a, ddmba_ddmb = da_spline.interp(D_mb)

    fw_spline = Akima(d_pt, fw_pt, delta_x=0.0)
    FW_mb, dfwmb_ddmb = fw_spline.interp(D_mb)

    return D_mb_a, FW_mb

