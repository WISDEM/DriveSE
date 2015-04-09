"""
drivesmooth.py
smooth components for low speed shaft/main bearings, gearbox, bedplate and yaw bearings, as well as modified components from NacelleSE
Created by Taylor Parsons on 4/7/2015.
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
from drivewpact.drive import AboveYawMassAdder, NacelleSystemAdder #TODO remove after implementing these
from drivese_utils import sys_print, size_Generator, size_HighSpeedSide, size_YawSystem, size_LowSpeedShaft, \
    setup_Bedplate_Front, setup_Bedplate, size_Bedplate, characterize_Bedplate_Front, characterize_Bedplate_Rear, \
    size_Transformer, add_RNA, add_Nacelle, add_AboveYawMass, size_LSS_3pt, get_Damage_Brng1, get_Damage_Brng2, \
    setup_Fatigue_Loads
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
    drivetrain_design = Enum('geared', ('geared', 'single_stage', 'multiSmooth', 'pm_directSmooth'), iotype='in')
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
        self.add('generator', GeneratorSmooth())
        self.add('bedplate', BedplateSmooth())
        self.add('above_yaw_massAdder', AboveYawMassAdderSmooth())
        self.add('yawSystem', YawSystemSmooth())
        self.add('nacelleSystem', NacelleSystemAdderSmooth())
        self.add('transformer',TransformerSmooth())
        self.add('rna',RNASystemAddeSmooth())

        self.driver.workflow.add(['gearbox', 'lowSpeedShaft', 'mainBearing', 'secondBearing', 'highSpeedSide', \
            'generator', 'bedplate', 'above_yaw_massAdder', 'yawSystem', 'nacelleSystem','transformer','rna'])

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

class AboveYawMassAdderSmooth(Component):

    # variables
    machine_rating = Float(iotype = 'in', units='kW', desc='machine rating')
    lss_mass = Float(iotype = 'in', units='kg', desc='component mass')
    main_bearing_mass = Float(iotype = 'in', units='kg', desc='component mass')
    second_bearing_mass = Float(iotype = 'in', units='kg', desc='component mass')
    gearbox_mass = Float(iotype = 'in', units='kg', desc='component mass')
    hss_mass = Float(iotype = 'in', units='kg', desc='component mass')
    generator_mass = Float(iotype = 'in', units='kg', desc='component mass')
    bedplate_mass = Float(iotype = 'in', units='kg', desc='component mass')
    bedplate_length = Float(iotype = 'in', units='m', desc='component length')
    bedplate_width = Float(iotype = 'in', units='m', desc='component width')
    transformer_mass = Float(iotype = 'in', units='kg', desc='component mass')

    # parameters
    crane = Bool(iotype='in', desc='flag for presence of crane')

    # returns
    electrical_mass = Float(iotype = 'out', units='kg', desc='component mass')
    vs_electronics_mass = Float(iotype = 'out', units='kg', desc='component mass')
    hvac_mass = Float(iotype = 'out', units='kg', desc='component mass')
    controls_mass = Float(iotype = 'out', units='kg', desc='component mass')
    platforms_mass = Float(iotype = 'out', units='kg', desc='component mass')
    crane_mass = Float(iotype = 'out', units='kg', desc='component mass')
    mainframe_mass = Float(iotype = 'out', units='kg', desc='component mass')
    cover_mass = Float(iotype = 'out', units='kg', desc='component mass')
    above_yaw_mass = Float(iotype = 'out', units='kg', desc='total mass above yaw system')
    length = Float(iotype = 'out', units='m', desc='component length')
    width = Float(iotype = 'out', units='m', desc='component width')
    height = Float(iotype = 'out', units='m', desc='component height')

    def __init__(self):
        ''' Initialize above yaw mass adder component
        '''

        super(AboveYawMassAdder_drive, self).__init__()

        #controls what happens if derivatives are missing
        self.missing_deriv_policy = 'assume_zero'

    def execute(self):

        add_AboveYawMass(self)

class NacelleSystemAdderSmooth(Component): #added to drive to include transformer
    ''' NacelleSystem class
          The Nacelle class is used to represent the overall nacelle of a wind turbine.
          It contains the general properties for a wind turbine component as well as additional design load and dimentional attributes as listed below.
          It contains an update method to determine the mass, mass properties, and dimensions of the component.
    '''
    # variables
    above_yaw_mass = Float(iotype='in', units='kg', desc='mass above yaw system')
    yawMass = Float(iotype='in', units='kg', desc='mass of yaw system')
    lss_mass = Float(iotype = 'in', units='kg', desc='component mass')
    main_bearing_mass = Float(iotype = 'in', units='kg', desc='component mass')
    second_bearing_mass = Float(iotype = 'in', units='kg', desc='component mass')
    gearbox_mass = Float(iotype = 'in', units='kg', desc='component mass')
    hss_mass = Float(iotype = 'in', units='kg', desc='component mass')
    generator_mass = Float(iotype = 'in', units='kg', desc='component mass')
    bedplate_mass = Float(iotype = 'in', units='kg', desc='component mass')
    mainframe_mass = Float(iotype = 'in', units='kg', desc='component mass')
    lss_cm = Array(np.array([0.0,0.0,0.0]),iotype = 'in', units='kg', desc='component CM')
    main_bearing_cm = Array(np.array([0.0,0.0,0.0]),iotype = 'in', units='kg', desc='component CM')
    second_bearing_cm = Array(np.array([0.0,0.0,0.0]),iotype = 'in', units='kg', desc='component CM')
    gearbox_cm = Array(np.array([0.0,0.0,0.0]),iotype = 'in', units='kg', desc='component CM')
    hss_cm = Array(np.array([0.0,0.0,0.0]),iotype = 'in', units='kg', desc='component CM')
    generator_cm = Array(np.array([0.0,0.0,0.0]),iotype = 'in', units='kg', desc='component CM')
    bedplate_cm = Array(np.array([0.0,0.0,0.0]),iotype = 'in', units='kg', desc='component CM')
    lss_I = Array(np.array([0.0,0.0,0.0]),iotype = 'in', units='kg', desc='component I')
    main_bearing_I = Array(np.array([0.0,0.0,0.0]),iotype = 'in', units='kg', desc='component I')
    second_bearing_I = Array(np.array([0.0,0.0,0.0]),iotype = 'in', units='kg', desc='component I')
    gearbox_I = Array(np.array([0.0,0.0,0.0]),iotype = 'in', units='kg', desc='component I')
    hss_I = Array(np.array([0.0,0.0,0.0]),iotype = 'in', units='kg', desc='component I')
    generator_I = Array(np.array([0.0,0.0,0.0]),iotype = 'in', units='kg', desc='component I')
    bedplate_I = Array(np.array([0.0,0.0,0.0]),iotype = 'in', units='kg', desc='component I')
    transformer_mass = Float(iotype = 'in', units='kg', desc='component mass')
    transformer_cm = Array(np.array([0.0,0.0,0.0]),iotype = 'in', units='kg', desc='component CM')
    transformer_I = Array(np.array([0.0,0.0,0.0]),iotype = 'in', units='kg', desc='component I')

    # returns
    nacelle_mass = Float(0.0, iotype='out', units='kg', desc='overall component mass')
    nacelle_cm = Array(np.array([0.0, 0.0, 0.0]), units='m', iotype='out', desc='center of mass of the component in [x,y,z] for an arbitrary coordinate system')
    nacelle_I = Array(np.array([0.0, 0.0, 0.0]), units='kg*m**2', iotype='out', desc=' moments of Inertia for the component [Ixx, Iyy, Izz] around its center of mass')

    def __init__(self):
        ''' Initialize above yaw mass adder component
        '''

        super(NacelleSystemAdder_drive , self).__init__()

        #controls what happens if derivatives are missing
        self.missing_deriv_policy = 'assume_zero'

    def execute(self):
        add_Nacelle(self)


class RNASystemAddeSmooth(Component):
    ''' RNASystem class
          This analysis is only to be used in placing the transformer of the drivetrain.
          The Rotor-Nacelle-Assembly class is used to represent the RNA of the turbine without the transformer and bedplate (to resolve circular dependency issues).
          It contains the general properties for a wind turbine component as well as additional design load and dimentional attributes as listed below.
          It contains an update method to determine the mass, mass properties, and dimensions of the component. 
    '''
    #inputs
    yawMass = Float(iotype='in', units='kg', desc='mass of yaw system')
    lss_mass = Float(iotype = 'in', units='kg', desc='component mass')
    main_bearing_mass = Float(iotype = 'in', units='kg', desc='component mass')
    second_bearing_mass = Float(iotype = 'in', units='kg', desc='component mass')
    gearbox_mass = Float(iotype = 'in', units='kg', desc='component mass')
    hss_mass = Float(iotype = 'in', units='kg', desc='component mass')
    generator_mass = Float(iotype = 'in', units='kg', desc='component mass')
    lss_cm = Array(np.array([0.0,0.0,0.0]),iotype = 'in', units='kg', desc='component CM')
    main_bearing_cm = Array(np.array([0.0,0.0,0.0]),iotype = 'in', units='kg', desc='component CM')
    second_bearing_cm = Array(np.array([0.0,0.0,0.0]),iotype = 'in', units='kg', desc='component CM')
    gearbox_cm = Array(np.array([0.0,0.0,0.0]),iotype = 'in', units='kg', desc='component CM')
    hss_cm = Array(np.array([0.0,0.0,0.0]),iotype = 'in', units='kg', desc='component CM')
    generator_cm = Array(np.array([0.0,0.0,0.0]),iotype = 'in', units='kg', desc='component CM')
    overhang = Float(iotype = 'in', units='m', desc='nacelle overhang')
    rotor_mass = Float(iotype = 'in', units='kg', desc='component mass')
    machine_rating = Float(iotype = 'in', units = 'kW', desc = 'machine rating ')

    #returns
    RNA_mass = Float(iotype = 'out', units='kg', desc='mass of total RNA')
    RNA_cm = Float(iotype='out', units='m', desc='RNA CM along x-axis')

    def __init__(self):
        ''' Initialize RNA Adder component
        '''

        super(RNASystemAdder_drive , self).__init__()

        #controls what happens if derivatives are missing
        self.missing_deriv_policy = 'assume_zero'

    def execute(self):

        add_RNA(self)

'''-----------------------------------------------------------------------------------------------------------------------------------------------------'''

class TransformerSmooth(Component):
    ''' Transformer class
            The transformer class is used to represent the transformer of a wind turbine drivetrain.
            It contains the general properties for a wind turbine component as well as additional design load and dimentional attributes as listed below.
            It contains an update method to determine the mass, mass properties, and dimensions of the component if it is in fact uptower'''

    #inputs
    machine_rating = Float(iotype='in', units='kW', desc='machine rating of the turbine')
    uptower_transformer = Bool(iotype='in', desc = 'uptower or downtower transformer')
    tower_top_diameter = Float(iotype = 'in', units = 'm', desc = 'tower top diameter for comparision of nacelle CM')
    rotor_mass = Float(iotype='in', units='kg', desc='rotor mass')
    overhang = Float(iotype='in', units='m', desc='rotor overhang distance')
    generator_cm = Array(iotype='in', desc='center of mass of the generator in [x,y,z]')
    rotor_diameter = Float(iotype='in',units='m', desc='rotor diameter of turbine')
    RNA_mass = Float(iotype = 'in', units='kg', desc='mass of total RNA')
    RNA_cm = Float(iotype='in', units='m', desc='RNA CM along x-axis')

    #outputs
    mass = Float(0.0, iotype='out', units='kg', desc='overall component mass')
    cm = Array(np.array([0.0, 0.0, 0.0]), iotype='out', desc='center of mass of the component in [x,y,z] for an arbitrary coordinate system')
    I = Array(np.array([0.0, 0.0, 0.0]), iotype='out', desc=' moments of Inertia for the component [Ixx, Iyy, Izz] around its center of mass')    

    def __init__(self):
        '''
        Initializes transformer component
        '''

        super(Transformer_drive, self).__init__()

        self.missing_deriv_policy = 'assume_zero'

    def execute(self):

        size_Transformer(self)

class GeneratorSmooth(Component):
    '''Generator class
          The Generator class is used to represent the generator of a wind turbine drivetrain.
          It contains the general properties for a wind turbine component as well as additional design load and dimensional attributes as listed below.
          It contains an update method to determine the mass, mass properties, and dimensions of the component.
    '''

    # variables
    rotor_diameter = Float(iotype='in', units='m', desc='rotor diameter')
    machine_rating = Float(iotype='in', units='kW', desc='machine rating of generator')
    gear_ratio = Float(iotype='in', desc='overall gearbox ratio')
    highSpeedSide_length = Float( iotype = 'in', units = 'm', desc='length of high speed shaft and brake')
    highSpeedSide_cm = Array(np.array([0.0,0.0,0.0]), iotype = 'in', units = 'm', desc='cm of high speed shaft and brake')
    rotor_speed = Float(iotype='in', units='rpm', desc='Speed of rotor at rated power')

    # parameters
    drivetrain_design = Enum('geared', ('geared', 'single_stage', 'multiSmooth', 'pm_directSmooth'), iotype='in')

    # returns
    mass = Float(0.0, iotype='out', units='kg', desc='overall component mass')
    cm = Array(np.array([0.0, 0.0, 0.0]), iotype='out', desc='center of mass of the component in [x,y,z] for an arbitrary coordinate system')
    I = Array(np.array([0.0, 0.0, 0.0]), iotype='out', desc=' moments of Inertia for the component [Ixx, Iyy, Izz] around its center of mass')

    def __init__(self):
        '''
        Initializes generator component
        '''

        super(GeneratorSmooth, self).__init__()

        #controls what happens if derivatives are missing
        self.missing_deriv_policy = 'assume_zero'

    def execute(self):

        size_Generator(self)

        # derivatives
        if (drivetrain_design < 4):
            self.d_mass_d_rotor_diameter = 0.0
        else:  # direct drive
            self.d_mass_d_rotor_diameter = massExp[drivetrain_design] * (massCoeff[drivetrain_design] * CalcTorque ** (massExp[drivetrain_design] - 1)) * (self.machine_rating * 1.1 * 0.5 / 80)
            # if self.rotor_speed !=0:
            #     self.d_mass_d_rotor_speed = massCoeff[drivetrain_design] * massExp[drivetrain_design] * CalcTorque ** (massExp[drivetrain_design] - 1))

        if (drivetrain_design < 4):
            self.d_mass_d_machine_rating = massExp[drivetrain_design] * (massCoeff[drivetrain_design] * self.machine_rating ** (massExp[drivetrain_design]-1))
        else:  # direct drive
            self.d_mass_d_machine_rating = massExp[drivetrain_design] * (massCoeff[drivetrain_design] * CalcTorque ** (massExp[drivetrain_design] - 1)) * (self.rotor_diameter * 1.1 * 0.5 / 80)

        self.d_cm_d_rotor_diameter = np.array([])

        self.d_I_d_rotor_diameter = np.array([0.0, 0.0, 0.0])
        self.d_I_d_rotor_diameter[0] = ((4.86*(10.**(-5)))*(self.rotor_diameter**4.333)) * 5.333 + \
                                (1./8.) * (2./3.) * self.d_mass_d_rotor_diameter * (depth ** 2 + width ** 2) + \
                                (1./8.) * (2./3.) * self.mass * (2.*depth*0.015 + 2.*width*0.5*0.015)
        self.d_I_d_rotor_diameter[1] = (1./(2.*self.gear_ratio**2))*(self.d_I_d_rotor_diameter[0]) + \
                                 self.d_mass_d_rotor_diameter * (((1./3.) * (length ** 2) / 12.) + ((2. / 3.) * (depth ** 2 + width ** 2 + (4./3.) * (length ** 2)) / 16. )) + \
                                 self.mass * ((1./3.) * (1./12.) * (2. * length * 1.6 * 0.015) + (2./3.) * (1./16.) * (2.*depth*0.015 + 2.*width*0.5*0.015 + (4./3.)*2.*length*1.6*0.015))
        self.d_I_d_rotor_diameter[2] = self.d_I_d_rotor_diameter[1]

        self.d_I_d_machine_rating = np.array([0.0, 0.0, 0.0])
        self.d_I_d_machine_rating[0] = (1./8.) * (2./3.) * self.d_mass_d_machine_rating * (depth ** 2 + width ** 2)
        self.d_I_d_machine_rating[1] = (1/(2.*self.gear_ratio**2))*self.d_I_d_machine_rating[0] + \
                                       ((1/3.) * self.d_mass_d_machine_rating * (length ** 2) / 12.) + \
                                       (((2 / 3.) * self.d_mass_d_machine_rating) * (depth ** 2 + width ** 2 + (4/3.) * (length ** 2)) / 16. )
        self.d_I_d_machine_rating[2] = self.d_I_d_machine_rating[1]

        self.d_I_d_gear_ratio = np.array([0.0, 0.0, 0.0])
        self.d_I_d_gear_ratio[1] = (1/2.) * self.I[0] * (-2.) * (self.gear_ratio**(-3))
        self.d_I_d_gear_ratio[2] = self.d_I_d_gear_ratio[1]


    def list_deriv_vars(self):

        inputs = ['rotor_diameter','machine_rating', 'gear_ratio','highSpeedSide_length','highSpeedSide_cm','rotor_speed']
        outputs = ['mass', 'cm', 'I']

        return inputs, outputs

    def provideJ(self):

        # Jacobian
        self.J = np.array([[self.d_mass_d_rotor_diameter, self.d_mass_d_machine_rating, 0], \
                           [self.d_cm_d_rotor_diameter[0], 0, 0], \
                           [self.d_cm_d_rotor_diameter[1], 0, 0], \
                           [self.d_cm_d_rotor_diameter[2], 0, 0], \
                           [self.d_I_d_rotor_diameter[0], self.d_I_d_machine_rating[0], self.d_I_d_gear_ratio[0]], \
                           [self.d_I_d_rotor_diameter[1], self.d_I_d_machine_rating[1], self.d_I_d_gear_ratio[1]], \
                           [self.d_I_d_rotor_diameter[2], self.d_I_d_machine_rating[2], self.d_I_d_gear_ratio[2]]])

        return self.J

#---------------------------------------------------------------------------------------------------------------

class HighSpeedSideSmooth(Component):
    '''
    HighSpeedShaft class
          The HighSpeedShaft class is used to represent the high speed shaft and mechanical brake components of a wind turbine drivetrain.
          It contains the general properties for a wind turbine component as well as additional design load and dimentional attributes as listed below.
          It contains an update method to determine the mass, mass properties, and dimensions of the component.
    '''

    # variables
    rotor_diameter = Float(iotype='in', units='m', desc='rotor diameter')
    rotor_torque = Float(iotype='in', units='N*m', desc='rotor torque at rated power')
    gear_ratio = Float(iotype='in', desc='overall gearbox ratio')
    lss_diameter = Float(iotype='in', units='m', desc='low speed shaft outer diameter')
    gearbox_length = Float(iotype = 'in', units = 'm', desc='gearbox length')
    gearbox_height = Float(iotype='in', units = 'm', desc = 'gearbox height')
    gearbox_cm = Array(iotype = 'in', units = 'm', desc = 'gearbox cm [x,y,z]')
    length_in = Float(iotype = 'in', units = 'm', desc = 'high speed shaft length determined by user. Default 0.5m')

    # returns
    mass = Float(0.0, iotype='out', units='kg', desc='overall component mass')
    cm = Array(np.array([0.0, 0.0, 0.0]), iotype='out', desc='center of mass of the component in [x,y,z] for an arbitrary coordinate system')
    I = Array(np.array([0.0, 0.0, 0.0]), iotype='out', desc=' moments of Inertia for the component [Ixx, Iyy, Izz] around its center of mass')
    length = Float(iotype='out', desc='length of high speed shaft')

    def __init__(self):
        '''
        Initializes high speed side component
        '''

        super(HighSpeedSideSmooth, self).__init__()

        #controls what happens if derivatives are missing
        self.missing_deriv_policy = 'assume_zero'

    def execute(self):

        size_HighSpeedSide(self)

#---------------------------------------------------------------------------------------------------------------

class YawSystemSmooth(Component):
    ''' YawSystem class
          The YawSystem class is used to represent the yaw system of a wind turbine drivetrain.
          It contains the general properties for a wind turbine component as well as additional design load and dimentional attributes as listed below.
          It contains an update method to determine the mass, mass properties, and dimensions of the component.
    '''
    #variables
    rotor_diameter = Float(iotype='in', units='m', desc='rotor diameter')
    rotor_thrust = Float(iotype='in', units='N', desc='maximum rotor thrust')
    tower_top_diameter = Float(iotype='in', units='m', desc='tower top diameter')
    above_yaw_mass = Float(iotype='in', units='kg', desc='above yaw mass')
    bedplate_height = Float(iotype = 'in', units = 'm', desc = 'bedplate height')

    #parameters
    yaw_motors_number = Int(0,iotype='in', desc='number of yaw motors')

    #outputs
    mass = Float(0.0, iotype='out', units='kg', desc='overall component mass')
    cm = Array(np.array([0.0, 0.0, 0.0]), iotype='out', desc='center of mass of the component in [x,y,z] for an arbitrary coordinate system')
    I = Array(np.array([0.0, 0.0, 0.0]), iotype='out', desc=' moments of Inertia for the component [Ixx, Iyy, Izz] around its center of mass')    


    def __init__(self):
        ''' Initializes yaw system
        '''
        super(YawSystemSmooth, self).__init__()

    def execute(self):
        
        size_YawSystem(self)

        #-------------------------------------------------------------------------------

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


    def execute(self):
        #Model bedplate as 2 parallel I-beams with a rear steel frame and a front cast frame
        #Deflection constraints applied at each bedplate end
        #Stress constraint checked at root of front and rear bedplate sections

        mb1_location, _ = smooth_abs(self.mb1_location)
        mb2_location, _ = smooth_abs(self.mb2_location)
        lss_location, _ = smooth_abs(self.lss_location)
        rotor_force_z, _ = smooth_abs(self.rotor_force_z)
        rotor_bending_moment_y, _ = smooth_abs(self.rotor_bending_moment_y)

        setup_Bedplate(self)

        #rear steel section
        characterize_Bedplate_Rear(self)

        totalTipDefl_margin_rear = (self.totalTipDefl_rear - 0.0001 - self.deflTol)/0.0001
        rootStress_margin_rear = (self.rootStress_rear - self.maxstress - self.stressTol)/self.maxstress

        #front cast section
        characterize_Bedplate_Front(self)

        totalTipDefl_margin_front = (self.totalTipDefl_front - 0.0001 - self.deflTol)/0.0001
        rootStress_margin_front = (self.rootStress_front - self.maxstress - self.stressTol)/self.maxstress

        #determine bedplate sizing
        size_Bedplate(self)


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

class LowSpeedShaft3ptSmooth(Component):
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
    machine_rating = Float(iotype='in', units='kW', desc='machine_rating machine rating of the turbine')
    gearbox_mass = Float(iotype='in', units='kg', desc='Gearbox mass')
    carrier_mass = Float(iotype='in', units='kg', desc='Carrier mass')
    overhang = Float(iotype='in', units='m', desc='Overhang distance')

    # parameters
    shrink_disc_mass = Float(iotype='in', units='kg', desc='Mass of the shrink disc')
    gearbox_cm = Array(iotype = 'in', units = 'm', desc = 'center of mass of gearbox')
    gearbox_length = Float(iotype='in', units='m', desc='gearbox length')
    flange_length = Float(iotype ='in', units='m', desc ='flange length')
    shaft_angle = Float(iotype='in', units='rad', desc='Angle of the LSS inclindation with respect to the horizontal')
    shaft_ratio = Float(iotype='in', desc='Ratio of inner diameter to outer diameter.  Leave zero for solid LSS')
    mb1Type = Str(iotype='in',desc='Main bearing type: CARB, TRB1 or SRB')
    mb2Type = Str(iotype='in',desc='Second bearing type: CARB, TRB1 or SRB')

    L_rb = Float(iotype='in', units='m', desc='distance between hub center and upwind main bearing')
    check_fatigue = Enum(0,(0,1,2),iotype = 'in', desc = 'turns on and off fatigue check')
    fatigue_exponent = Float(iotype = 'in', desc = 'fatigue exponent of material')
    S_ut = Float(700e6,iotype = 'in', units = 'Pa', desc = 'ultimate tensile strength of material')
    weibull_A = Float(iotype = 'in', units = 'm/s', desc = 'weibull scale parameter "A" of 10-minute windspeed probability distribution')
    weibull_k = Float(iotype = 'in', desc = 'weibull shape parameter "k" of 10-minute windspeed probability distribution')
    blade_number = Float(iotype = 'in', desc = 'number of blades on rotor, 2 or 3')
    cut_in = Float(iotype = 'in', units = 'm/s', desc = 'cut-in windspeed')
    cut_out = Float(iotype = 'in', units = 'm/s', desc = 'cut-out windspeed')
    Vrated = Float(iotype = 'in', units = 'm/s', desc = 'rated windspeed')
    T_life = Float(iotype = 'in', units = 'yr', desc = 'design life')
    IEC_Class = Str(iotype='in',desc='IEC class letter: A, B, or C')
    DrivetrainEfficiency = Float(iotype = 'in', desc = 'overall drivettrain efficiency')
    rotor_freq = Float(iotype = 'in', units = 'rpm', desc='rated rotor speed')
    availability = Float(.95,iotype = 'in', desc = 'turbine availability')

    rotor_thrust_distribution = Array(iotype='in', units ='N', desc = 'thrust distribution across turbine life')
    rotor_thrust_count = Array(iotype='in', desc = 'corresponding cycle array for thrust distribution')
    rotor_Fy_distribution = Array(iotype='in', units ='N', desc = 'Fy distribution across turbine life')
    rotor_Fy_count = Array(iotype='in', desc = 'corresponding cycle array for Fy distribution')
    rotor_Fz_distribution = Array(iotype='in', units ='N', desc = 'Fz distribution across turbine life')
    rotor_Fz_count = Array(iotype='in', desc = 'corresponding cycle array for Fz distribution') 
    rotor_torque_distribution = Array(iotype='in', units ='N*m', desc = 'torque distribution across turbine life')
    rotor_torque_count = Array(iotype='in', desc = 'corresponding cycle array for torque distribution') 
    rotor_My_distribution = Array(iotype='in', units ='N*m', desc = 'My distribution across turbine life')
    rotor_My_count = Array(iotype='in', desc = 'corresponding cycle array for My distribution') 
    rotor_Mz_distribution = Array(iotype='in', units ='N*m', desc = 'Mz distribution across turbine life')
    rotor_Mz_count = Array(iotype='in', desc = 'corresponding cycle array for Mz distribution') 
   
    # outputs
    design_torque = Float(iotype='out', units='N*m', desc='lss design torque')
    design_bending_load = Float(iotype='out', units='N', desc='lss design bending load')
    length = Float(iotype='out', units='m', desc='lss length')
    diameter1 = Float(iotype='out', units='m', desc='lss outer diameter at main bearing')
    diameter2 = Float(iotype='out', units='m', desc='lss outer diameter at second bearing')
    mass = Float(0.0, iotype='out', units='kg', desc='overall component mass')
    cm = Array(np.array([0.0, 0.0, 0.0]), iotype='out', desc='center of mass of the component in [x,y,z] for an arbitrary coordinate system')
    I = Array(np.array([0.0, 0.0, 0.0]), iotype='out', desc=' moments of Inertia for the component [Ixx, Iyy, Izz] around its center of mass')
    FW_mb = Float(iotype='out', units='m', desc='facewidth of main bearing')    
    bearing_mass1 = Float(iotype='out', units = 'kg', desc='main bearing mass')
    bearing_mass2 = Float(0., iotype='out', units = 'kg', desc='main bearing mass') #zero for 3-pt model
    bearing_location1 = Array(np.array([0,0,0]),iotype='out', units = 'm', desc = 'main bearing 1 center of mass')
    bearing_location2 = Array(np.array([0,0,0]),iotype='out', units = 'm', desc = 'main bearing 2 center of mass')

    def __init__(self):
        '''
        Initializes low speed shaft component  
        '''

        super(LowSpeedShaft3ptSmooth, self).__init__()
    
    def execute(self):

        #input parameters
        if self.flange_length == 0:
            self.flange_length = 0.3*(self.rotor_diameter/100.0)**2.0 - 0.1 * (self.rotor_diameter / 100.0) + 0.4

        if self.L_rb == 0: #distance from hub center to main bearing
            self.L_rb = get_L_rb(self.rotor_diameter, False)

        #If user does not know important moments, crude approx
        if self.rotor_mass > 0 and self.rotor_bending_moment_y == 0: 
            self.rotor_bending_moment_y=get_My(self.rotor_mass,self.L_rb)

        if self.rotor_mass > 0 and self.rotor_bending_moment_z == 0:
            self.rotor_bending_moment_z=get_Mz(self.rotor_mass,self.L_rb)

        self.g = 9.81 #m/s
        self.density = 7850.0


        self.L_ms_new = 0.0
        self.L_ms_0=0.5 # main shaft length downwind of main bearing
        self.L_ms=self.L_ms_0
        tol=1e-4 
        check_limit = 1.0
        dL=0.05
        self.D_max = 1.0
        self.D_min = 0.2
        # self.D_in=self.shaft_ratio*self.D_max

        T=self.rotor_bending_moment_x/1000.0

        #Main bearing defelection check
        MB_limit=0.026;
        CB_limit=4.0/60.0/180.0*pi;
        TRB1_limit=3.0/60.0/180.0*pi;
        self.n_safety_brg = 1.0
        self.n_safety=2.5
        self.Sy = 66000#*self.S_ut/700e6 #psi
        self.E=2.1e11  
        N_count=50    
          
        self.u_knm_inlb = 8850.745454036
        self.u_in_m = 0.0254000508001
        counter=0
        length_max = self.overhang - self.L_rb + (self.gearbox_cm[0] -self.gearbox_length/2.) #modified length limit 7/29

        while abs(check_limit) > tol and self.L_ms_new < length_max:
            counter =counter+1
            if self.L_ms_new > 0:
                 self.L_ms=self.L_ms_new
            else:
                  self.L_ms=self.L_ms_0

            #-----------------------
            size_LSS_3pt(self)
            #-----------------------

            check_limit = abs(abs(self.theta_y[-1])-TRB1_limit/self.n_safety_brg)
            #print 'deflection slope'
            #print TRB1_limit
            #print 'threshold'
            #print theta_y[-1]
            self.L_ms_new = self.L_ms + dL        

        # fatigue check Taylor Parsons 6/2014
        if self.check_fatigue == 1 or 2:
          #start_time = time.time()
          #material properties 34CrNiMo6 steel +QT, large diameter
          self.E=2.1e11
          self.density=7800.0
          self.n_safety = 2.5
          if self.S_ut <= 0:
            self.S_ut=700.0e6 #Pa
          Sm=0.9*self.S_ut #for bending situations, material strength at 10^3 cycles
          C_size=0.6 #diameter larger than 10"
          C_surf=4.51*(self.S_ut/1e6)**-.265 #machined surface 272*(self.S_ut/1e6)**-.995 #forged
          C_temp=1 #normal operating temps
          C_reliab=0.814 #99% reliability
          C_envir=1. #enclosed environment
          Se=C_size*C_surf*C_temp*C_reliab*C_envir*.5*self.S_ut #modified endurance limit for infinite life

          if self.fatigue_exponent!=0:
            if self.fatigue_exponent > 0:
                self.SN_b = - self.fatigue_exponent
            else:
                self.SN_b = self.fatigue_exponent
          else:
            Nfinal = 5e8 #point where fatigue limit occurs under hypothetical S-N curve TODO adjust to fit actual data
            z=log10(1e3)-log10(Nfinal)  #assuming no endurance limit (high strength steel)
            self.SN_b=1/z*log10(Sm/Se)
          self.SN_a=Sm/(1000.**self.SN_b)
          # print 'm:', -1/self.SN_b
          # print 'a:', self.SN_a

          if self.check_fatigue == 1:
              #checks to make sure all inputs are reasonable
              if self.rotor_mass < 100:
                  [self.rotor_mass] = get_rotor_mass(self.machine_rating,False)

              #Rotor Loads calculations using DS472
              setup_Fatigue_Loads(self)

              #upwind bearing calculations
              iterationstep=0.001
              diameter_limit = 1.5
              while True:
                  get_Damage_Brng1(self)

                  # print 'Bearing Diameter:', self.D_max
                  # print 'self.Damage:', self.Damage
                  if self.Damage < 1 or self.D_max >= diameter_limit:
                      # print 'Bearing Diameter:', self.D_max
                      # print 'self.Damage:', self.Damage
                      #print (time.time() - start_time), 'seconds of total simulation time'
                      break
                  else:
                      self.D_max+=iterationstep

              #begin bearing calculations
              N_bearings = self.N/self.blade_number #rotation number

              Fz1stoch = (-self.My_stoch)/(self.L_ms)
              Fy1stoch = self.Mz_stoch/self.L_ms
              Fz1determ = (self.weightGbx*self.L_gb - self.LssWeight*.5*self.L_ms - self.rotorWeight*(self.L_ms+self.L_rb)) / (self.L_ms)

              Fr_range = ((abs(Fz1stoch)+abs(Fz1determ))**2 +Fy1stoch**2)**.5 #radial stochastic + deterministic mean
              Fa_range = self.Fx_stoch*cos(self.shaft_angle) + (self.rotorWeight+self.LssWeight)*sin(self.shaft_angle) #axial stochastic + mean

              life_bearing = self.N_f/self.blade_number

              [self.D_max_a,FW_max,bearingmass] = fatigue_for_bearings(self.D_max, Fr_range, Fa_range, N_bearings, life_bearing, self.mb1Type,False)
         
        #resize bearing if no fatigue check
        if self.check_fatigue == 0:
            [self.D_max_a,FW_max,bearingmass] = resize_for_bearings(self.D_max,  self.mb1Type,False)

        [self.D_min_a,FW_min,trash] = resize_for_bearings(self.D_min,  self.mb2Type,False) #mb2 is a representation of the gearbox connection
            
        lss_mass_new=(pi/3)*(self.D_max_a**2+self.D_min_a**2+self.D_max_a*self.D_min_a)*(self.L_ms-(FW_max+FW_min)/2)*self.density/4+ \
                         (pi/4)*(self.D_max_a**2-self.D_in**2)*self.density*FW_max+\
                         (pi/4)*(self.D_min_a**2-self.D_in**2)*self.density*FW_min-\
                         (pi/4)*(self.D_in**2)*self.density*(self.L_ms+(FW_max+FW_min)/2)
        lss_mass_new *= 1.35 # add flange and shrink disk mass
        self.length=self.L_ms_new + (FW_max+FW_min)/2 + self.flange_length
        #print ("self.L_ms: {0}").format(self.L_ms)
        #print ("LSS length, m: {0}").format(self.length)
        self.D_outer=self.D_max
        #print ("Upwind MB OD, m: {0}").format(self.D_max_a)
        #print ("CB OD, m: {0}").format(self.D_min_a)
        #print ("self.D_min: {0}").format(self.D_min)
        self.D_in=self.D_in
        self.mass=lss_mass_new
        self.diameter1= self.D_max_a
        self.diameter2= self.D_min_a 
        #self.length=self.L_ms
        #print self.length
        self.D_outer=self.D_max_a
        self.diameter=self.D_max_a

         # calculate mass properties
        downwind_location = np.array([self.gearbox_cm[0]-self.gearbox_length/2. , self.gearbox_cm[1] , self.gearbox_cm[2] ])

        bearing_location1 = np.array([0.,0.,0.]) #upwind
        bearing_location1[0] = downwind_location[0] - self.L_ms*cos(self.shaft_angle)
        bearing_location1[1] = downwind_location[1]
        bearing_location1[2] = downwind_location[2] + self.L_ms*sin(self.shaft_angle)
        self.bearing_location1 = bearing_location1

        self.bearing_location2 = np.array([0.,0.,0.]) #downwind does not exist

        cm = np.array([0.0,0.0,0.0])
        cm[0] = downwind_location[0] - 0.65*self.length*cos(self.shaft_angle) #From solid models, center of mass with flange (not including shrink disk) very nearly .65*total_length
        cm[1] = downwind_location[1]
        cm[2] = downwind_location[2] + 0.65*self.length*sin(self.shaft_angle)

        #including shrink disk mass
        self.cm[0] = (cm[0]*self.mass + downwind_location[0]*self.shrink_disc_mass) / (self.mass+self.shrink_disc_mass) 
        self.cm[1] = cm[1]
        self.cm[2] = (cm[2]*self.mass + downwind_location[2]*self.shrink_disc_mass) / (self.mass+self.shrink_disc_mass)
        # print 'shaft before shrink disk:', self.mass
        self.mass+=self.shrink_disc_mass

        I = np.array([0.0, 0.0, 0.0])
        I[0]  = self.mass * (self.D_in ** 2.0 + self.D_outer ** 2.0) / 8.0
        I[1]  = self.mass * (self.D_in ** 2.0 + self.D_outer ** 2.0 + (4.0 / 3.0) * (self.length ** 2.0)) / 16.0
        I[2]  = I[1]
        self.I = I

        # print 'self.L_rb %8.f' %(self.L_rb) #*(self.machine_rating/5.0e3)   #distance from hub center to main bearing scaled off NREL 5MW
        # print 'L_bg %8.f' %(L_bg) #*(self.machine_rating/5.0e3)         #distance from hub center to gearbox yokes
        # print 'L_as %8.f' %(L_as) #distance from main bearing to shaft center
      
        self.FW_mb=FW_max
        self.bearing_mass1 = bearingmass
        self.bearing_mass2 = 0.


class LowSpeedShaft4ptSmooth(Component):
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


def nacelle_example_5MW_baseline_3pt():

    # NREL 5 MW Rotor Variables
    print '----- NREL 5 MW Turbine - 3 Point Suspension -----'
    nace = Drive3pt()
    nace.rotor_diameter = 126.0 # m
    nace.rotor_speed = 12.1 # #rpm m/s
    nace.machine_rating = 5000.0
    nace.DrivetrainEfficiency = 0.95
    nace.rotor_torque =  1.5 * (nace.machine_rating * 1000 / nace.DrivetrainEfficiency) / (nace.rotor_speed * (pi / 30)) # 
    nace.rotor_thrust = 599610.0 # N
    nace.rotor_mass = 0.0 #accounted for in F_z # kg
    nace.rotor_speed = 12.1 #rpm
    nace.rotor_bending_moment = -16665000.0 # Nm same as rotor_bending_moment_y
    nace.rotor_bending_moment_x = 330770.0# Nm
    nace.rotor_bending_moment_y = -16665000.0 # Nm
    nace.rotor_bending_moment_z = 2896300.0 # Nm
    nace.rotor_force_x = 599610.0 # N
    nace.rotor_force_y = 186780.0 # N
    nace.rotor_force_z = -842710.0 # N

    # NREL 5 MW Drivetrain variables
    nace.drivetrain_design = 'geared' # geared 3-stage Gearbox with induction generator machine
    nace.machine_rating = 5000.0 # kW
    nace.gear_ratio = 96.76 # 97:1 as listed in the 5 MW reference document
    nace.gear_configuration = 'eep' # epicyclic-epicyclic-parallel
    nace.crane = True # onboard crane present
    nace.shaft_angle = 5.0 #deg
    nace.shaft_ratio = 0.10
    nace.Np = [3,3,1]
    nace.ratio_type = 'optimal'
    nace.shaft_type = 'normal'
    nace.uptower_transformer=True
    nace.shrink_disc_mass = 333.3*nace.machine_rating/1000.0 # estimated
    nace.carrier_mass = 8000.0 # estimated
    nace.mb1Type = 'SRB'
    nace.mb2Type = 'SRB'
    nace.flange_length = 0.5
    nace.overhang = 5.0
    nace.L_rb = 1.912 # length from hub center to main bearing, leave zero if unknow

    nace.check_fatigue = 0 #0 if no fatigue check, 1 if parameterized fatigue check, 2 if known loads inputs

    #variables if check_fatigue = 1:
    nace.blade_number=3
    nace.cut_in=3. #cut-in m/s
    nace.cut_out=25. #cut-out m/s
    nace.Vrated=11.4 #rated windspeed m/s
    nace.weibull_k = 2.2 # windepeed distribution shape parameter
    nace.weibull_A = 9. # windspeed distribution scale parameter
    nace.T_life=20. #design life in years
    nace.IEC_Class_Letter = 'A'

    #variables if check_fatigue =2:
    #test distribution
    p_o = 6444.24
    R = nace.rotor_diameter*.5
    count = np.logspace(log10(328),log10(288984470), endpoint=True , num=100)
    standard = count.copy()
    for i in range(len(count)):
        standard[i] = .11*2.5*.28*13.4*(log10(288984470)-log10(count[i]))+0.18
    Fx_factor = (.3649*log(nace.rotor_diameter)-1.074)
    Mx_factor = (.0799*log(nace.rotor_diameter)-.2577)
    My_factor = (.172*log(nace.rotor_diameter)-.5943)
    Mz_factor = (.1659*log(nace.rotor_diameter)-.5795)
    nace.rotor_thrust_distribution = standard.copy()**0.5*p_o*(R)*Fx_factor
    nace.rotor_thrust_count = np.logspace(log10(328),log10(288984470), endpoint=True , num=100)
    nace.rotor_Fy_distribution = np.zeros(100)
    nace.rotor_Fy_count = nace.rotor_thrust_count.copy()
    nace.rotor_Fz_distribution = np.zeros(100)
    nace.rotor_Fz_count = nace.rotor_thrust_count.copy()
    nace.rotor_torque_distribution = standard.copy()*0.45*p_o*(R)**2*Mx_factor
    nace.rotor_torque_count = nace.rotor_thrust_count.copy()
    nace.rotor_My_distribution = standard.copy()*0.33*p_o*.8*(R)**2*My_factor
    nace.rotor_My_count = nace.rotor_thrust_count.copy()
    nace.rotor_Mz_distribution = standard.copy()*0.33*p_o*.8*(R)**2*Mz_factor
    nace.rotor_Mz_count = nace.rotor_thrust_count.copy() 

    # NREL 5 MW Tower Variables
    nace.tower_top_diameter = 3.78 # m

    nace.run()

    #cm_print(nace)
    sys_print(nace)

def nacelle_example_5MW_baseline_4pt():

    #DLC7.1a_0001_Land_38.0V0_352ny_S01.out from Table 42 in 
    #"Effect of Tip Velocity Constraints on the Optimized Design of a Wind Turbine"

    # NREL 5 MW Rotor Variables
    print '----- NREL 5 MW Turbine - 4 Point Suspension -----'
    nace = Drive4pt()
    nace.rotor_diameter = 126.0 # m
    nace.rotor_speed = 12.1 # #rpm m/s
    nace.machine_rating = 5000.0
    nace.DrivetrainEfficiency = 0.95
    nace.rotor_torque =  1.5 * (nace.machine_rating * 1000 / nace.DrivetrainEfficiency) / (nace.rotor_speed * (pi / 30)) # 6.35e6 #4365248.74 # Nm
    nace.rotor_thrust = 599610.0 # N
    nace.rotor_mass = 0.0 #accounted for in F_z # kg
    nace.rotor_speed = 12.1 #rpm
    nace.rotor_bending_moment = -16665000.0 # Nm same as rotor_bending_moment_y
    nace.rotor_bending_moment_x = 330770.0# Nm
    nace.rotor_bending_moment_y = -16665000.0 # Nm
    nace.rotor_bending_moment_z = 2896300.0 # Nm
    nace.rotor_force_x = 599610.0 # N
    nace.rotor_force_y = 186780.0 # N
    nace.rotor_force_z = -842710.0 # N

    # NREL 5 MW Drivetrain variables
    nace.drivetrain_design = 'geared' # geared 3-stage Gearbox with induction generator machine
    nace.machine_rating = 5000.0 # kW
    nace.gear_ratio = 96.76 # 97:1 as listed in the 5 MW reference document
    nace.gear_configuration = 'eep' # epicyclic-epicyclic-parallel
    nace.crane = True # onboard crane present
    nace.shaft_angle = 5.0 #deg
    nace.shaft_ratio = 0.10
    nace.Np = [3,3,1]
    nace.ratio_type = 'optimal'
    nace.shaft_type = 'normal'
    nace.uptower_transformer=True
    nace.shrink_disc_mass = 333.3*nace.machine_rating/1000.0 # estimated
    nace.carrier_mass = 8000.0 # estimated
    nace.mb1Type = 'CARB'
    nace.mb2Type = 'SRB'
    nace.flange_length = 0.5 #m
    nace.overhang = 5.0
    nace.gearbox_cm = 0.1
    nace.hss_length = 1.5
    nace.L_rb = 1.912 # length from hub center to main bearing, leave zero if unknown

    nace.check_fatigue = 0 #0 if no fatigue check, 1 if parameterized fatigue check, 2 if known loads inputs

    #variables if check_fatigue = 1:
    nace.blade_number=3
    nace.cut_in=3. #cut-in m/s
    nace.cut_out=25. #cut-out m/s
    nace.Vrated=11.4 #rated windspeed m/s
    nace.weibull_k = 2.2 # windepeed distribution shape parameter
    nace.weibull_A = 9. # windspeed distribution scale parameter
    nace.T_life=20. #design life in years
    nace.IEC_Class_Letter = 'A'

    nace.tower_top_diameter = 3.78 # m

    nace.run()

    #cm_print(nace)
    sys_print(nace)


if __name__ == '__main__':
    ''' Main runs through tests of both drivetrain configurations'''

    nacelle_example_5MW_baseline_3pt()

    nacelle_example_5MW_baseline_4pt()

