"""
DriveSE.py

Created by Yi Guo, Taylor Parsons and Ryan King2014.
Copyright (c) NREL. All rights reserved.
"""

from openmdao.main.api import Component, Assembly
from openmdao.main.datatypes.api import Float, Bool, Int, Str, Array, Enum
import numpy as np
from math import pi, cos, sqrt, radians, sin, exp, log10, log, floor, ceil
import algopy
import scipy as scp
import scipy.optimize as opt
from scipy import integrate

from fusedwind.interface import implement_base
from drivewpact.drive import NacelleBase
from drivese_components import LowSpeedShaft_drive, Gearbox_drive, MainBearing_drive, SecondBearing_drive, Bedplate_drive, YawSystem_drive, LowSpeedShaft_drive3pt, \
    LowSpeedShaft_drive4pt, Transformer_drive, HighSpeedSide_drive, Generator_drive, NacelleSystemAdder_drive, AboveYawMassAdder_drive, RNASystemAdder_drive

@implement_base(NacelleBase)
class Drive3pt(Assembly):
    '''
       DriveSE class
          The DriveSE3pt class is used to represent the nacelle system of a wind turbine with a single main bearing
    '''

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
    drivetrain_design = Enum('geared', ('geared', 'single_stage', 'multi_drive', 'pm_direct_drive'), iotype='in')
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
    transformer_mass = Float(iotype='out', units='kg', desc='component mass')

    # outputs for hub CM calcuations
    MB1_location = Array(iotype = 'out', units = 'm', desc = 'center of mass of main bearing in [x,y,z] for an arbitrary coordinate system')    

    # new variables
    rotor_bending_moment_x = Float(iotype='in', units='N*m', desc='The bending moment about the x axis')
    rotor_bending_moment_y = Float(iotype='in', units='N*m', desc='The bending moment about the y axis')
    rotor_bending_moment_z = Float(iotype='in', units='N*m', desc='The bending moment about the z axis')
    rotor_force_x = Float(iotype='in', units='N', desc='The force along the x axis applied at hub center')
    rotor_force_y = Float(iotype='in', units='N', desc='The force along the y axis applied at hub center')
    rotor_force_z = Float(iotype='in', units='N', desc='The force along the z axis applied at hub center')
    shaft_angle = Float(iotype='in', units='deg', desc='Angle of the LSS inclindation with respect to the horizontal')
    blade_root_diameter = Float(iotype='in', units='m', desc='blade root diameter')
    shaft_ratio = Float(iotype='in', desc='Ratio of inner diameter to outer diameter.  Leave zero for solid LSS')
    rotor_speed = Float(iotype='in', units='rpm', desc='Speed of rotor at rated power')
    ratio_type=Str(iotype='in', desc='optimal or empirical stage ratios')
    shaft_type = Str(iotype='in', desc = 'normal or short shaft length')
    uptower_transformer = Bool(iotype = 'in', desc = 'Boolean stating if transformer is uptower')
    shrink_disc_mass = Float(iotype='in',desc='Mass of the shrink disc')
    carrier_mass = Float(iotype='in', units='kg', desc='Carrier mass')
    flange_length = Float(iotype='in', units='m', desc='flange length')    
    L_rb = Float(iotype='in', units='m', desc='distance between hub center and upwind main bearing')
    overhang = Float(iotype='in', units='m', desc='Overhang distance')
    gearbox_cm = Float(0.0,iotype = 'in', units = 'm', desc = 'distance from tower-top center to gearbox cm--negative for upwind')
    hss_length = Float(iotype = 'in', units = 'm', desc = 'optional high speed shaft length determined by user')

    # new parameters
    Np = Array(np.array([0.0,0.0,0.0,]), iotype='in', desc='number of planets in each stage')
    mb1Type = Str(iotype='in',desc='Main bearing type: CARB, TRB or SRB')
    mb2Type = Str(iotype='in',desc= 'Carrier bearing type: CRB, TRB or RB')

    #Fatigue Parameters
    check_fatigue = Int(iotype = 'in', desc = 'turns on and off fatigue check. 0 if no fatigue check, 1 if unknown loads, 2 if known loads')
    fatigue_exponent = Float(iotype = 'in', desc = 'fatigue exponent of material')
    S_ut = Float(iotype = 'in', units = 'Pa', desc = 'ultimate tensile strength of shaft material')
    weibull_A = Float(iotype = 'in', desc = 'weibull scale parameter "A" of 10-minute windspeed probability distribution')
    weibull_k = Float(iotype = 'in', desc = 'weibull shape parameter "k" of 10-minute windspeed probability distribution')
    blade_number = Int(3,iotype = 'in', desc = 'number of blades on rotor, 2 or 3')
    cut_in = Float(iotype = 'in', units = 'm/s', desc = 'cut-in windspeed')
    cut_out = Float(iotype = 'in', units = 'm/s', desc = 'cut-out windspeed')
    Vrated = Float(iotype = 'in', units = 'm/s', desc = 'rated windspeed')
    T_life = Float(iotype = 'in', units = 'yr', desc = 'cut-in windspeed')
    IEC_Class = Str(iotype='in',desc='IEC class letter: A, B, or C')
    DrivetrainEfficiency = Float(iotype = 'in', desc = 'overall drivettrain efficiency')
    availability = Float(.95,iotype = 'in', desc = 'turbine availability')

    #for use if check_fatigue  = 2:
    rotor_thrust_distribution = Array(iotype='in', units ='N', desc = 'thrust distribution across turbine life')
    rotor_thrust_count = Array(iotype='in', desc = 'corresponding cycle-count array for thrust distribution')
    rotor_Fy_distribution = Array(iotype='in', units ='N', desc = 'Fy distribution across turbine life')
    rotor_Fy_count = Array(iotype='in', desc = 'corresponding cycle-count array for Fy distribution')
    rotor_Fz_distribution = Array(iotype='in', units ='N', desc = 'Fz distribution across turbine life')
    rotor_Fz_count = Array(iotype='in', desc = 'corresponding cycle-count array for Fz distribution') 
    rotor_torque_distribution = Array(iotype='in', units ='N*m', desc = 'torque distribution across turbine life')
    rotor_torque_count = Array(iotype='in', desc = 'corresponding cycle-count array for torque distribution') 
    rotor_My_distribution = Array(iotype='in', units ='N*m', desc = 'My distribution across turbine life')
    rotor_My_count = Array(iotype='in', desc = 'corresponding cycle-count array for My distribution') 
    rotor_Mz_distribution = Array(iotype='in', units ='N*m', desc = 'Mz distribution across turbine life')
    rotor_Mz_count = Array(iotype='in', desc = 'corresponding cycle-count array for Mz distribution') 

    def configure(self):

        # select components
        self.add('above_yaw_massAdder', AboveYawMassAdder_drive())
        self.add('nacelleSystem', NacelleSystemAdder_drive())
        self.add('lowSpeedShaft', LowSpeedShaft_drive3pt())
        self.add('mainBearing', MainBearing_drive())
        self.add('secondBearing',SecondBearing_drive())
        self.add('gearbox', Gearbox_drive())
        self.add('highSpeedSide', HighSpeedSide_drive())
        self.add('generator', Generator_drive())
        self.add('bedplate', Bedplate_drive())
        self.add('yawSystem', YawSystem_drive())
        self.add('transformer',Transformer_drive())
        self.add('rna', RNASystemAdder_drive())

        # workflow
        self.driver.workflow.add(['above_yaw_massAdder', 'nacelleSystem', 'lowSpeedShaft', 'mainBearing', 'secondBearing', 'gearbox', 'highSpeedSide', 'transformer','generator', 'bedplate', 'yawSystem','rna'])

        # connect inputs
        self.connect('rotor_diameter', ['lowSpeedShaft.rotor_diameter', 'mainBearing.rotor_diameter', 'secondBearing.rotor_diameter', 'gearbox.rotor_diameter', 'highSpeedSide.rotor_diameter', \
                     'generator.rotor_diameter', 'bedplate.rotor_diameter', 'yawSystem.rotor_diameter','transformer.rotor_diameter'])
        self.connect('rotor_bending_moment_x', ['lowSpeedShaft.rotor_bending_moment_x'])
        self.connect('rotor_bending_moment_y', ['bedplate.rotor_bending_moment_y','lowSpeedShaft.rotor_bending_moment_y'])
        self.connect('rotor_bending_moment_z', 'lowSpeedShaft.rotor_bending_moment_z')
        self.connect('rotor_force_x', 'lowSpeedShaft.rotor_force_x')
        self.connect('rotor_force_y', 'lowSpeedShaft.rotor_force_y')
        self.connect('rotor_force_z', ['bedplate.rotor_force_z','lowSpeedShaft.rotor_force_z'])
        self.connect('rotor_torque', ['gearbox.rotor_torque', 'highSpeedSide.rotor_torque']) # Need to address internal torque calculations...
        self.connect('rotor_mass', ['bedplate.rotor_mass','lowSpeedShaft.rotor_mass','transformer.rotor_mass'])
        self.connect('rotor_thrust', 'yawSystem.rotor_thrust')
        self.connect('tower_top_diameter', ['bedplate.tower_top_diameter', 'yawSystem.tower_top_diameter'])
        self.connect('machine_rating', ['bedplate.machine_rating', 'generator.machine_rating', 'above_yaw_massAdder.machine_rating', 'lowSpeedShaft.machine_rating','transformer.machine_rating'])
        self.connect('drivetrain_design', 'generator.drivetrain_design')
        self.connect('gear_ratio', ['gearbox.gear_ratio', 'generator.gear_ratio', 'highSpeedSide.gear_ratio'])
        self.connect('gear_configuration', 'gearbox.gear_configuration')
        self.connect('crane', 'above_yaw_massAdder.crane')
        self.connect('shaft_angle', 'lowSpeedShaft.shaft_angle')
        self.connect('shaft_ratio', 'lowSpeedShaft.shaft_ratio')
        self.connect('shrink_disc_mass', 'lowSpeedShaft.shrink_disc_mass')
        self.connect('carrier_mass', 'lowSpeedShaft.carrier_mass')
        self.connect('mb1Type', ['mainBearing.bearing_type', 'lowSpeedShaft.mb1Type'])
        self.connect('mb2Type', ['secondBearing.bearing_type', 'lowSpeedShaft.mb2Type'])
        self.connect('Np', 'gearbox.Np')
        self.connect('ratio_type', 'gearbox.ratio_type')
        self.connect('shaft_type', 'gearbox.shaft_type')
        self.connect('flange_length', ['bedplate.flange_length','lowSpeedShaft.flange_length'])
        self.connect('overhang',['lowSpeedShaft.overhang','bedplate.overhang'])
        self.connect('hss_length', 'highSpeedSide.length_in')
        self.connect('L_rb', ['lowSpeedShaft.L_rb','bedplate.L_rb'])


        self.connect('check_fatigue', 'lowSpeedShaft.check_fatigue')
        self.connect('weibull_A', 'lowSpeedShaft.weibull_A')
        self.connect('weibull_k', 'lowSpeedShaft.weibull_k')
        self.connect('blade_number', ['lowSpeedShaft.blade_number'])
        self.connect('cut_in', 'lowSpeedShaft.cut_in')
        self.connect('cut_out', 'lowSpeedShaft.cut_out')
        self.connect('Vrated', 'lowSpeedShaft.Vrated')
        self.connect('T_life', 'lowSpeedShaft.T_life')
        self.connect('IEC_Class', 'lowSpeedShaft.IEC_Class')
        self.connect('DrivetrainEfficiency', 'lowSpeedShaft.DrivetrainEfficiency')
        self.connect('rotor_speed', ['lowSpeedShaft.rotor_freq','generator.rotor_speed'])
        self.connect('rotor_thrust_distribution', 'lowSpeedShaft.rotor_thrust_distribution')
        self.connect('rotor_thrust_count', 'lowSpeedShaft.rotor_thrust_count')
        self.connect('rotor_Fy_distribution', 'lowSpeedShaft.rotor_Fy_distribution')
        self.connect('rotor_Fy_count', 'lowSpeedShaft.rotor_Fy_count')
        self.connect('rotor_Fz_distribution', 'lowSpeedShaft.rotor_Fz_distribution')
        self.connect('rotor_Fz_count', 'lowSpeedShaft.rotor_Fz_count')
        self.connect('rotor_torque_distribution', 'lowSpeedShaft.rotor_torque_distribution')
        self.connect('rotor_torque_count', 'lowSpeedShaft.rotor_torque_count')
        self.connect('rotor_My_distribution', 'lowSpeedShaft.rotor_My_distribution')
        self.connect('rotor_My_count', 'lowSpeedShaft.rotor_My_count')
        self.connect('rotor_Mz_distribution', 'lowSpeedShaft.rotor_Mz_distribution')
        self.connect('rotor_Mz_count', 'lowSpeedShaft.rotor_Mz_count')
        self.connect('fatigue_exponent', 'lowSpeedShaft.fatigue_exponent')
        self.connect('S_ut', 'lowSpeedShaft.S_ut')
        self.connect('availability', 'lowSpeedShaft.availability')

        # connect components
        self.connect('lowSpeedShaft.design_torque', ['mainBearing.lss_design_torque', 'secondBearing.lss_design_torque'])
        self.connect('lowSpeedShaft.diameter1', ['mainBearing.lss_diameter', 'highSpeedSide.lss_diameter'])
        self.connect('lowSpeedShaft.diameter2', 'secondBearing.lss_diameter')
        self.connect('lowSpeedShaft.length', 'bedplate.lss_length')
        self.connect('gearbox.length', 'bedplate.gbx_length')
        self.connect('bedplate.length', 'above_yaw_massAdder.bedplate_length')
        self.connect('bedplate.width', 'above_yaw_massAdder.bedplate_width')
        self.connect('bedplate.height', 'yawSystem.bedplate_height')
        self.connect('gearbox.height', ['highSpeedSide.gearbox_height'])
        self.connect('lowSpeedShaft.FW_mb', 'bedplate.FW_mb1')
        self.connect('mainBearing.cm[0]', 'bedplate.mb1_location')

        self.connect('lowSpeedShaft.bearing_mass1',['mainBearing.bearing_mass'])
        self.connect('lowSpeedShaft.bearing_mass2',['secondBearing.bearing_mass'])
        self.connect('lowSpeedShaft.mass', ['bedplate.lss_mass','above_yaw_massAdder.lss_mass', 'nacelleSystem.lss_mass', 'low_speed_shaft_mass'])
        self.connect('mainBearing.mass', ['bedplate.mb1_mass','above_yaw_massAdder.main_bearing_mass', 'nacelleSystem.main_bearing_mass', 'main_bearing_mass'])
        self.connect('secondBearing.mass', ['bedplate.mb2_mass','above_yaw_massAdder.second_bearing_mass', 'nacelleSystem.second_bearing_mass', 'second_bearing_mass'])
        self.connect('gearbox.mass', ['lowSpeedShaft.gearbox_mass', 'above_yaw_massAdder.gearbox_mass', 'nacelleSystem.gearbox_mass', 'gearbox_mass'])
        self.connect('highSpeedSide.mass', ['bedplate.hss_mass','above_yaw_massAdder.hss_mass', 'nacelleSystem.hss_mass', 'high_speed_side_mass'])
        self.connect('generator.mass', ['bedplate.generator_mass','above_yaw_massAdder.generator_mass', 'nacelleSystem.generator_mass', 'generator_mass'])
        self.connect('bedplate.mass', ['above_yaw_massAdder.bedplate_mass', 'nacelleSystem.bedplate_mass', 'bedplate_mass'])
        self.connect('transformer.mass', ['above_yaw_massAdder.transformer_mass', 'transformer_mass'])
        self.connect('above_yaw_massAdder.mainframe_mass', 'nacelleSystem.mainframe_mass')
        self.connect('yawSystem.mass', ['nacelleSystem.yawMass', 'yaw_system_mass'])
        self.connect('above_yaw_massAdder.above_yaw_mass', ['yawSystem.above_yaw_mass', 'nacelleSystem.above_yaw_mass'])

        self.connect('lowSpeedShaft.bearing_location2', 'secondBearing.location')
        self.connect('lowSpeedShaft.bearing_location1', 'mainBearing.location')
        self.connect('lowSpeedShaft.cm', ['nacelleSystem.lss_cm'])
        self.connect('mainBearing.cm', 'nacelleSystem.main_bearing_cm')
        self.connect('secondBearing.cm', 'nacelleSystem.second_bearing_cm')
        self.connect('gearbox.cm', ['nacelleSystem.gearbox_cm'])
        self.connect('highSpeedSide.cm', ['nacelleSystem.hss_cm','generator.highSpeedSide_cm'])
        self.connect('highSpeedSide.length', 'generator.highSpeedSide_length')
        self.connect('generator.cm', ['nacelleSystem.generator_cm'])
        self.connect('generator.cm[0]','bedplate.generator_location')
        self.connect('bedplate.cm', ['nacelleSystem.bedplate_cm'])
        self.connect('gearbox_cm','gearbox.cm_input')
        self.connect('gearbox.cm',['lowSpeedShaft.gearbox_cm','highSpeedSide.gearbox_cm'])
        self.connect('gearbox.length',['lowSpeedShaft.gearbox_length','highSpeedSide.gearbox_length'])

        self.connect('lowSpeedShaft.I', ['nacelleSystem.lss_I'])
        self.connect('mainBearing.I', 'nacelleSystem.main_bearing_I')
        self.connect('secondBearing.I', 'nacelleSystem.second_bearing_I')
        self.connect('gearbox.I', ['nacelleSystem.gearbox_I'])
        self.connect('highSpeedSide.I', ['nacelleSystem.hss_I'])
        self.connect('generator.I', ['nacelleSystem.generator_I'])
        self.connect('bedplate.I', ['nacelleSystem.bedplate_I'])

        #transformer and RNA analysis
        self.connect('uptower_transformer', ['bedplate.uptower_transformer','transformer.uptower_transformer'])
        self.connect('tower_top_diameter', 'transformer.tower_top_diameter')
        self.connect('overhang', 'transformer.overhang')
        self.connect('generator.cm', 'transformer.generator_cm')
        self.connect('lowSpeedShaft.mass', 'rna.lss_mass')
        self.connect('mainBearing.mass', 'rna.main_bearing_mass')
        self.connect('secondBearing.mass', 'rna.second_bearing_mass')
        self.connect('gearbox.mass', 'rna.gearbox_mass')
        self.connect('highSpeedSide.mass', 'rna.hss_mass')
        self.connect('generator.mass', 'rna.generator_mass')
        self.connect('overhang', 'rna.overhang')
        self.connect('rotor_mass', 'rna.rotor_mass')
        self.connect('lowSpeedShaft.cm', 'rna.lss_cm')
        self.connect('mainBearing.cm', 'rna.main_bearing_cm')
        self.connect('secondBearing.cm', 'rna.second_bearing_cm')
        self.connect('gearbox.cm', 'rna.gearbox_cm')
        self.connect('highSpeedSide.cm', 'rna.hss_cm')
        self.connect('generator.cm', 'rna.generator_cm')
        self.connect('machine_rating', 'rna.machine_rating')
        self.connect('rna.RNA_mass', 'transformer.RNA_mass')
        self.connect('rna.RNA_cm','transformer.RNA_cm')
        self.connect('transformer.cm[0]', 'bedplate.transformer_location')
        self.connect('transformer.mass', ['bedplate.transformer_mass','nacelleSystem.transformer_mass'])
        self.connect('transformer.cm','nacelleSystem.transformer_cm')
        self.connect('transformer.I', 'nacelleSystem.transformer_I')

        # create passthroughs
        self.connect('nacelleSystem.nacelle_mass', 'nacelle_mass')
        self.connect('nacelleSystem.nacelle_cm', 'nacelle_cm')
        self.connect('nacelleSystem.nacelle_I', 'nacelle_I')
        
        # passthroughs for hub
        self.connect('mainBearing.cm','MB1_location')

#------------------------------------------------------------------
@implement_base(NacelleBase)
class Drive4pt(Assembly):
    '''
       DriveSE class
          The DriveSE4pt class is used to represent the nacelle system of a wind turbine with two main bearings
    '''

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
    transformer_mass = Float(iotype='out', units='kg', desc='component mass')

    # parameters
    drivetrain_design = Enum('geared', ('geared', 'single_stage', 'multi_drive', 'pm_direct_drive'), iotype='in')
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

    # outputs for hub CM calcuations
    MB1_location = Array(iotype = 'out', units = 'm', desc = 'center of mass of main bearing in [x,y,z] for an arbitrary coordinate system')

    # new variables
    rotor_bending_moment_x = Float(iotype='in', units='N*m', desc='The bending moment about the x axis')
    rotor_bending_moment_y = Float(iotype='in', units='N*m', desc='The bending moment about the y axis')
    rotor_bending_moment_z = Float(iotype='in', units='N*m', desc='The bending moment about the z axis')
    rotor_force_x = Float(iotype='in', units='N', desc='The force along the x axis applied at hub center')
    rotor_force_y = Float(iotype='in', units='N', desc='The force along the y axis applied at hub center')
    rotor_force_z = Float(iotype='in', units='N', desc='The force along the z axis applied at hub center')
    blade_root_diameter = Float(iotype='in', units='m', desc='blade root diameter')
    shaft_angle = Float(iotype='in', units='deg', desc='Angle of the LSS inclindation with respect to the horizontal') # Bedplate tilting angle
    shaft_ratio = Float(iotype='in', desc='Ratio of inner diameter to outer diameter.  Leave zero for solid LSS')
    rotor_speed = Float(iotype='in', units='rpm', desc='Speed of rotor at rated power')
    flange_length = Float(iotype='in', units='m', desc='flange length')
    overhang = Float(iotype='in', units='m', desc='Overhang distance')
    gearbox_cm = Float(0.0,iotype = 'in', units = 'm', desc = 'distance from tower-top center to gearbox cm--negative for upwind')

    # new variables and parameters
    Np = Array(np.array([0.0,0.0,0.0,]), iotype='in', desc='number of planets in each stage')
    ratio_type=Str(iotype='in', desc='optimal or empirical stage ratios')
    shaft_type = Str(iotype='in', desc = 'normal or short shaft length')
    uptower_transformer = Bool(iotype = 'in', desc = 'Boolean stating if transformer is uptower')
    shrink_disc_mass = Float(iotype='in',  desc='Mass of the shrink disc')
    carrier_mass = Float(iotype='in', units='kg', desc='Carrier mass')
    mb1Type = Str(iotype='in',desc='Main bearing type: CARB, TRB or SRB')
    mb2Type = Str(iotype='in',desc='Second bearing type: CARB, TRB or SRB')
    L_rb = Float(iotype='in', units='m', desc='distance between hub center and upwind main bearing')
    hss_length = Float(iotype = 'in', units = 'm', desc = 'high speed shaft length determined by user. Default 0.5m')

    # fatigue check
    check_fatigue = Int(iotype = 'in', desc = 'turns on and off fatigue check. 0 if no fatigue check, 1 if unknown loads, 2 if known loads')
    fatigue_exponent = Float(iotype = 'in', desc = 'fatigue exponent of shaft material')
    S_ut = Float(iotype = 'in', units = 'Pa', desc = 'ultimate tensile strength of shaft material')
    weibull_A = Float(iotype = 'in', desc = 'weibull scale parameter "A" of 10-minute windspeed probability distribution')
    weibull_k = Float(iotype = 'in', desc = 'weibull shape parameter "k" of 10-minute windspeed probability distribution')
    blade_number = Int(3,iotype = 'in', desc = 'number of blades on rotor, 2 or 3')
    cut_in = Float(iotype = 'in', units = 'm/s', desc = 'cut-in windspeed')
    cut_out = Float(iotype = 'in', units = 'm/s', desc = 'cut-out windspeed')
    Vrated = Float(iotype = 'in', units = 'm/s', desc = 'rated windspeed')
    T_life = Float(iotype = 'in', units = 'yr', desc = 'cut-in windspeed')
    IEC_Class = Str(iotype='in',desc='IEC class letter: A, B, or C')
    DrivetrainEfficiency = Float(iotype = 'in', desc = 'overall drivettrain efficiency')
    availability = Float(.95,iotype = 'in', desc = 'turbine availability')

    #for use if check_fatigue  = 2:
    rotor_thrust_distribution = Array(iotype='in', units ='N', desc = 'thrust distribution across turbine life')
    rotor_thrust_count = Array(iotype='in', desc = 'corresponding cycle-count array for thrust distribution')
    rotor_Fy_distribution = Array(iotype='in', units ='N', desc = 'Fy distribution across turbine life')
    rotor_Fy_count = Array(iotype='in', desc = 'corresponding cycle-count array for Fy distribution')
    rotor_Fz_distribution = Array(iotype='in', units ='N', desc = 'Fz distribution across turbine life')
    rotor_Fz_count = Array(iotype='in', desc = 'corresponding cycle-count array for Fz distribution') 
    rotor_torque_distribution = Array(iotype='in', units ='N*m', desc = 'torque distribution across turbine life')
    rotor_torque_count = Array(iotype='in', desc = 'corresponding cycle-count array for torque distribution') 
    rotor_My_distribution = Array(iotype='in', units ='N*m', desc = 'My distribution across turbine life')
    rotor_My_count = Array(iotype='in', desc = 'corresponding cycle-count array for My distribution') 
    rotor_Mz_distribution = Array(iotype='in', units ='N*m', desc = 'Mz distribution across turbine life')
    rotor_Mz_count = Array(iotype='in', desc = 'corresponding cycle-count array for Mz distribution') 

    def configure(self):

        # select components
        self.add('above_yaw_massAdder', AboveYawMassAdder_drive())
        self.add('nacelleSystem', NacelleSystemAdder_drive())
        self.add('lowSpeedShaft', LowSpeedShaft_drive4pt())
        self.add('mainBearing', MainBearing_drive())
        self.add('secondBearing',SecondBearing_drive())
        self.add('gearbox', Gearbox_drive())
        self.add('highSpeedSide', HighSpeedSide_drive())
        self.add('generator', Generator_drive())
        self.add('bedplate', Bedplate_drive())
        self.add('yawSystem', YawSystem_drive())
        self.add('transformer', Transformer_drive())
        self.add('rna', RNASystemAdder_drive())

        # workflow
        self.driver.workflow.add(['above_yaw_massAdder', 'nacelleSystem', 'lowSpeedShaft', 'mainBearing', 'secondBearing', 'gearbox', 'highSpeedSide', 'generator', 'bedplate', 'yawSystem','transformer','rna'])

        # connect inputs
        self.connect('rotor_diameter', ['lowSpeedShaft.rotor_diameter', 'mainBearing.rotor_diameter', 'secondBearing.rotor_diameter', 'gearbox.rotor_diameter', 'highSpeedSide.rotor_diameter', \
                     'generator.rotor_diameter', 'bedplate.rotor_diameter', 'yawSystem.rotor_diameter','transformer.rotor_diameter'])
        self.connect('rotor_bending_moment_x', ['lowSpeedShaft.rotor_bending_moment_x'])
        self.connect('rotor_bending_moment_y', ['bedplate.rotor_bending_moment_y','lowSpeedShaft.rotor_bending_moment_y'])
        self.connect('rotor_bending_moment_z', 'lowSpeedShaft.rotor_bending_moment_z')
        self.connect('rotor_force_x', 'lowSpeedShaft.rotor_force_x')
        self.connect('rotor_force_y', 'lowSpeedShaft.rotor_force_y')
        self.connect('rotor_force_z', ['bedplate.rotor_force_z','lowSpeedShaft.rotor_force_z'])
        self.connect('rotor_torque', ['gearbox.rotor_torque', 'highSpeedSide.rotor_torque']) # Need to address internal torque calculations...
        self.connect('rotor_mass', ['bedplate.rotor_mass','lowSpeedShaft.rotor_mass','transformer.rotor_mass'])
        self.connect('rotor_thrust', 'yawSystem.rotor_thrust')
        self.connect('tower_top_diameter', ['bedplate.tower_top_diameter', 'yawSystem.tower_top_diameter'])
        self.connect('machine_rating', ['bedplate.machine_rating', 'generator.machine_rating', 'above_yaw_massAdder.machine_rating', 'lowSpeedShaft.machine_rating','transformer.machine_rating'])
        self.connect('drivetrain_design', 'generator.drivetrain_design')
        self.connect('gear_ratio', ['gearbox.gear_ratio', 'generator.gear_ratio', 'highSpeedSide.gear_ratio'])
        self.connect('gear_configuration', 'gearbox.gear_configuration')
        self.connect('crane', 'above_yaw_massAdder.crane')
        self.connect('shaft_angle', ['lowSpeedShaft.shaft_angle'])
        self.connect('shaft_ratio', 'lowSpeedShaft.shaft_ratio')
        self.connect('shrink_disc_mass', 'lowSpeedShaft.shrink_disc_mass')
        self.connect('carrier_mass', 'lowSpeedShaft.carrier_mass')
        self.connect('mb1Type', ['mainBearing.bearing_type', 'lowSpeedShaft.mb1Type'])
        self.connect('mb2Type', ['secondBearing.bearing_type', 'lowSpeedShaft.mb2Type'])
        self.connect('Np', 'gearbox.Np')
        self.connect('ratio_type', 'gearbox.ratio_type')
        self.connect('shaft_type', 'gearbox.shaft_type')
        self.connect('flange_length', ['bedplate.flange_length','lowSpeedShaft.flange_length'])
        self.connect('L_rb', ['lowSpeedShaft.L_rb','bedplate.L_rb'])
        self.connect('gearbox_cm','gearbox.cm_input')
        self.connect('hss_length', 'highSpeedSide.length_in')
        self.connect('availability', 'lowSpeedShaft.availability')
        self.connect('check_fatigue', 'lowSpeedShaft.check_fatigue')
        self.connect('fatigue_exponent', 'lowSpeedShaft.fatigue_exponent')
        self.connect('S_ut', ['lowSpeedShaft.S_ut'])
        self.connect('weibull_A', 'lowSpeedShaft.weibull_A')
        self.connect('weibull_k', 'lowSpeedShaft.weibull_k')
        self.connect('blade_number', ['lowSpeedShaft.blade_number'])
        self.connect('cut_in', 'lowSpeedShaft.cut_in')
        self.connect('cut_out', 'lowSpeedShaft.cut_out')
        self.connect('Vrated', 'lowSpeedShaft.Vrated')
        self.connect('T_life', 'lowSpeedShaft.T_life')
        self.connect('IEC_Class', 'lowSpeedShaft.IEC_Class')
        self.connect('DrivetrainEfficiency', 'lowSpeedShaft.DrivetrainEfficiency')
        self.connect('rotor_speed', ['lowSpeedShaft.rotor_freq','generator.rotor_speed'])
        self.connect('rotor_thrust_distribution', 'lowSpeedShaft.rotor_thrust_distribution')
        self.connect('rotor_thrust_count', 'lowSpeedShaft.rotor_thrust_count')
        self.connect('rotor_Fy_distribution', 'lowSpeedShaft.rotor_Fy_distribution')
        self.connect('rotor_Fy_count', 'lowSpeedShaft.rotor_Fy_count')
        self.connect('rotor_Fz_distribution', 'lowSpeedShaft.rotor_Fz_distribution')
        self.connect('rotor_Fz_count', 'lowSpeedShaft.rotor_Fz_count')
        self.connect('rotor_torque_distribution', 'lowSpeedShaft.rotor_torque_distribution')
        self.connect('rotor_torque_count', 'lowSpeedShaft.rotor_torque_count')
        self.connect('rotor_My_distribution', 'lowSpeedShaft.rotor_My_distribution')
        self.connect('rotor_My_count', 'lowSpeedShaft.rotor_My_count')
        self.connect('rotor_Mz_distribution', 'lowSpeedShaft.rotor_Mz_distribution')
        self.connect('rotor_Mz_count', 'lowSpeedShaft.rotor_Mz_count')

        # connect components
        self.connect('lowSpeedShaft.design_torque', ['mainBearing.lss_design_torque', 'secondBearing.lss_design_torque'])
        self.connect('lowSpeedShaft.diameter1', ['mainBearing.lss_diameter', 'highSpeedSide.lss_diameter'])
        self.connect('lowSpeedShaft.diameter2', 'secondBearing.lss_diameter')
        self.connect('lowSpeedShaft.bearing_location2', 'secondBearing.location')
        self.connect('lowSpeedShaft.bearing_location1', 'mainBearing.location')
        self.connect('lowSpeedShaft.length', 'bedplate.lss_length')
        self.connect('gearbox.length', ['bedplate.gbx_length','highSpeedSide.gearbox_length','lowSpeedShaft.gearbox_length'])
        self.connect('gearbox.height', ['highSpeedSide.gearbox_height'])
        self.connect('bedplate.height', 'yawSystem.bedplate_height')
        self.connect('bedplate.length', 'above_yaw_massAdder.bedplate_length')
        self.connect('bedplate.width', 'above_yaw_massAdder.bedplate_width')
        self.connect('overhang',['lowSpeedShaft.overhang','bedplate.overhang'])
        self.connect('lowSpeedShaft.FW_mb1', 'bedplate.FW_mb1')

        self.connect('lowSpeedShaft.bearing_mass1',['mainBearing.bearing_mass'])
        self.connect('lowSpeedShaft.bearing_mass2',['secondBearing.bearing_mass'])
        self.connect('lowSpeedShaft.mass', ['bedplate.lss_mass','above_yaw_massAdder.lss_mass', 'nacelleSystem.lss_mass', 'low_speed_shaft_mass'])
        self.connect('mainBearing.mass', ['bedplate.mb1_mass','above_yaw_massAdder.main_bearing_mass', 'nacelleSystem.main_bearing_mass', 'main_bearing_mass'])
        self.connect('secondBearing.mass', ['bedplate.mb2_mass','above_yaw_massAdder.second_bearing_mass', 'nacelleSystem.second_bearing_mass', 'second_bearing_mass'])
        self.connect('gearbox.mass', ['lowSpeedShaft.gearbox_mass', 'above_yaw_massAdder.gearbox_mass', 'nacelleSystem.gearbox_mass', 'gearbox_mass'])
        self.connect('highSpeedSide.mass', ['bedplate.hss_mass','above_yaw_massAdder.hss_mass', 'nacelleSystem.hss_mass', 'high_speed_side_mass'])
        self.connect('generator.mass', ['bedplate.generator_mass','above_yaw_massAdder.generator_mass', 'nacelleSystem.generator_mass', 'generator_mass'])
        self.connect('bedplate.mass', ['above_yaw_massAdder.bedplate_mass', 'nacelleSystem.bedplate_mass', 'bedplate_mass'])
        self.connect('transformer.mass', ['above_yaw_massAdder.transformer_mass', 'transformer_mass'])
        self.connect('above_yaw_massAdder.mainframe_mass', 'nacelleSystem.mainframe_mass')
        self.connect('yawSystem.mass', ['nacelleSystem.yawMass', 'yaw_system_mass'])
        self.connect('above_yaw_massAdder.above_yaw_mass', ['yawSystem.above_yaw_mass', 'nacelleSystem.above_yaw_mass'])


        self.connect('lowSpeedShaft.cm', ['nacelleSystem.lss_cm'])
        self.connect('lowSpeedShaft.cm[0]', 'bedplate.lss_location')
        self.connect('mainBearing.cm', 'nacelleSystem.main_bearing_cm')
        self.connect('mainBearing.cm[0]', 'bedplate.mb1_location')
        self.connect('secondBearing.cm', 'nacelleSystem.second_bearing_cm')
        self.connect('secondBearing.cm[0]', 'bedplate.mb2_location')
        self.connect('gearbox.cm', ['nacelleSystem.gearbox_cm','highSpeedSide.gearbox_cm','lowSpeedShaft.gearbox_cm'])
        self.connect('highSpeedSide.cm', ['nacelleSystem.hss_cm','generator.highSpeedSide_cm'])
        self.connect('highSpeedSide.length', 'generator.highSpeedSide_length')
        self.connect('highSpeedSide.cm[0]','bedplate.hss_location')
        self.connect('generator.cm', ['nacelleSystem.generator_cm'])
        self.connect('generator.cm[0]','bedplate.generator_location')
        self.connect('bedplate.cm', ['nacelleSystem.bedplate_cm'])

        self.connect('lowSpeedShaft.I', ['nacelleSystem.lss_I'])
        self.connect('mainBearing.I', 'nacelleSystem.main_bearing_I')
        self.connect('secondBearing.I', 'nacelleSystem.second_bearing_I')
        self.connect('gearbox.I', ['nacelleSystem.gearbox_I'])
        self.connect('highSpeedSide.I', ['nacelleSystem.hss_I'])
        self.connect('generator.I', ['nacelleSystem.generator_I'])
        self.connect('bedplate.I', ['nacelleSystem.bedplate_I'])

        #transformer and RNA analysis
        self.connect('uptower_transformer', ['bedplate.uptower_transformer','transformer.uptower_transformer'])
        self.connect('tower_top_diameter', 'transformer.tower_top_diameter')
        self.connect('overhang', 'transformer.overhang')
        self.connect('generator.cm', 'transformer.generator_cm')
        self.connect('lowSpeedShaft.mass', 'rna.lss_mass')
        self.connect('mainBearing.mass', 'rna.main_bearing_mass')
        self.connect('secondBearing.mass', 'rna.second_bearing_mass')
        self.connect('gearbox.mass', 'rna.gearbox_mass')
        self.connect('highSpeedSide.mass', 'rna.hss_mass')
        self.connect('generator.mass', 'rna.generator_mass')
        self.connect('overhang', 'rna.overhang')
        self.connect('rotor_mass', 'rna.rotor_mass')
        self.connect('lowSpeedShaft.cm', 'rna.lss_cm')
        self.connect('mainBearing.cm', 'rna.main_bearing_cm')
        self.connect('secondBearing.cm', 'rna.second_bearing_cm')
        self.connect('gearbox.cm', 'rna.gearbox_cm')
        self.connect('highSpeedSide.cm', 'rna.hss_cm')
        self.connect('generator.cm', 'rna.generator_cm')
        self.connect('machine_rating', 'rna.machine_rating')
        self.connect('rna.RNA_mass', 'transformer.RNA_mass')
        self.connect('rna.RNA_cm','transformer.RNA_cm')
        self.connect('transformer.cm[0]', 'bedplate.transformer_location')
        self.connect('transformer.mass', ['bedplate.transformer_mass','nacelleSystem.transformer_mass'])
        self.connect('transformer.cm','nacelleSystem.transformer_cm')
        self.connect('transformer.I', 'nacelleSystem.transformer_I')

        # create passthroughs
        self.connect('nacelleSystem.nacelle_mass', 'nacelle_mass')
        self.connect('nacelleSystem.nacelle_cm', 'nacelle_cm')
        self.connect('nacelleSystem.nacelle_I', 'nacelle_I')

        # passthroughs for hub
        self.connect('mainBearing.cm','MB1_location')

#------------------------------------------------------------------
#NacelleSE drive with simplified low speed shaft and windpact models
#@implement_base(NacelleBase)
# class NacelleSE_drive(Assembly):
#     '''
#        NacelleSE class
#           The NacelleSE class is used to represent the nacelle system of a wind turbine.
#     '''

#     # parameters
#     rotor_bending_moment = Float(iotype='in', units='N*m', desc='rotor aerodynamic bending moment')
#     shaft_angle = Float(iotype='in', units='deg', desc='Angle of the LSS inclindation with respect to the horizontal')
#     shaft_length = Float(iotype='in', units='m', desc='length of low speed shaft')
#     shaftD1 = Float(iotype='in', desc='Fraction of LSS distance from gearbox to downwind main bearing')
#     shaftD2 = Float(iotype='in', desc='raction of LSS distance from gearbox to upwind main bearing')
#     shaft_ratio = Float(iotype='in', desc='Ratio of inner diameter to outer diameter.  Leave zero for solid LSS')
#     rotor_speed = Float(iotype='in', units='rpm', desc='Speed of rotor at rated power')
#     #gbxPower = Float(iotype='in', units='kW', desc='gearbox rated power')
#     #eff = Float(iotype='in', desc='Fraction of LSS distance from gearbox to downwind main bearing')
#     Np = Array(np.array([0.0,0.0,0.0,]), iotype='in', desc='number of planets in each stage')
#     ratio_type=Str(iotype='in', desc='optimal or empirical stage ratios')
#     shaft_type = Str(iotype='in', desc = 'normal or short shaft length')
#     uptower_transformer = Bool(iotype = 'in', desc = 'Boolean stating if transformer is uptower')

#     def configure(self):

#         # select components
#         self.add('above_yaw_massAdder', AboveYawMassAdder())
#         self.add('nacelleSystem', NacelleSystemAdder())
#         self.add('lowSpeedShaft', LowSpeedShaft_drive())
#         self.add('mainBearing', MainBearing())
#         self.add('secondBearing',SecondBearing())
#         self.add('gearbox', Gearbox())
#         self.add('highSpeedSide', HighSpeedSide())
#         self.add('generator', Generator())
#         self.add('bedplate', Bedplate_drive())
#         self.add('yawSystem', YawSystem_drive())


#         # workflow
#         self.driver.workflow.add(['above_yaw_massAdder', 'nacelleSystem', 'lowSpeedShaft', 'mainBearing', 'secondBearing', 'gearbox', 'highSpeedSide', 'generator', 'bedplate', 'yawSystem'])


#         # connect inputs
#         self.connect('rotor_diameter', ['lowSpeedShaft.rotor_diameter', 'mainBearing.rotor_diameter', 'secondBearing.rotor_diameter', 'gearbox.rotor_diameter', 'highSpeedSide.rotor_diameter', \
#                      'generator.rotor_diameter', 'bedplate.rotor_diameter', 'yawSystem.rotor_diameter'])
#         self.connect('rotor_bending_moment', 'lowSpeedShaft.rotor_bending_moment')

#         self.connect('rotor_torque', ['lowSpeedShaft.rotor_torque', 'gearbox.rotor_torque', 'highSpeedSide.rotor_torque'])

#         self.connect('rotor_mass', 'lowSpeedShaft.rotor_mass')
#         self.connect('rotor_speed', ['mainBearing.rotor_speed', 'secondBearing.rotor_speed'])

#         self.connect('rotor_thrust', 'yawSystem.rotor_thrust')
#         self.connect('tower_top_diameter', ['bedplate.tower_top_diameter', 'yawSystem.tower_top_diameter'])
#         self.connect('machine_rating', ['generator.machine_rating', 'above_yaw_massAdder.machine_rating', 'lowSpeedShaft.machine_rating'])

#         self.connect('drivetrain_design', 'generator.drivetrain_design')
#         self.connect('gear_ratio', ['gearbox.gear_ratio', 'generator.gear_ratio', 'highSpeedSide.gear_ratio'])
#         self.connect('gear_configuration', 'gearbox.gear_configuration')

#         self.connect('crane', 'above_yaw_massAdder.crane')
#         self.connect('shaft_angle', ['lowSpeedShaft.shaft_angle','mainBearing.shaft_angle','secondBearing.shaft_angle'])
#         self.connect('shaft_length', ['lowSpeedShaft.shaft_length', 'bedplate.shaft_length'])
#         self.connect('shaftD1', 'lowSpeedShaft.shaftD1')
#         self.connect('shaftD2', 'lowSpeedShaft.shaftD2')
#         self.connect('shaft_ratio', 'lowSpeedShaft.shaft_ratio')
#         self.connect('rotor_speed', 'lowSpeedShaft.rotor_speed')


#         # connect components
#         self.connect('lowSpeedShaft.design_torque', ['mainBearing.lss_design_torque', 'secondBearing.lss_design_torque'])
#         self.connect('lowSpeedShaft.diameter', ['mainBearing.lss_diameter', 'secondBearing.lss_diameter', 'highSpeedSide.lss_diameter'])
#         self.connect('bedplate.length', 'above_yaw_massAdder.bedplate_length')
#         self.connect('bedplate.width', 'above_yaw_massAdder.bedplate_width')

#         self.connect('lowSpeedShaft.mass', ['mainBearing.lss_mass', 'secondBearing.lss_mass', 'above_yaw_massAdder.lss_mass', 'nacelleSystem.lss_mass'])
#         self.connect('mainBearing.mass', ['above_yaw_massAdder.main_bearing_mass', 'nacelleSystem.main_bearing_mass'])
#         self.connect('secondBearing.mass', ['above_yaw_massAdder.second_bearing_mass', 'nacelleSystem.second_bearing_mass'])
#         self.connect('gearbox.mass', ['above_yaw_massAdder.gearbox_mass', 'nacelleSystem.gearbox_mass'])
#         self.connect('highSpeedSide.mass', ['above_yaw_massAdder.hss_mass', 'nacelleSystem.hss_mass'])
#         self.connect('generator.mass', ['above_yaw_massAdder.generator_mass', 'nacelleSystem.generator_mass'])
#         self.connect('bedplate.mass', ['above_yaw_massAdder.bedplate_mass', 'nacelleSystem.bedplate_mass'])
#         self.connect('above_yaw_massAdder.mainframe_mass', 'nacelleSystem.mainframe_mass')
#         self.connect('yawSystem.mass', ['nacelleSystem.yawMass'])
#         self.connect('above_yaw_massAdder.above_yaw_mass', ['yawSystem.above_yaw_mass', 'nacelleSystem.above_yaw_mass'])

#         self.connect('gearbox.cm', ['nacelleSystem.gearbox_cm','highSpeedSide.gearbox_cm','lowSpeedShaft.gearbox_cm','transformer.gearbox_cm',])
#         self.connect('gearbox.dimensions', ['highSpeedSide.gearbox_dimensions','lowSpeedShaft.gearbox_dimensions'])
#         self.connect('lowSpeedShaft.cm', ['nacelleSystem.lss_cm','mainBearing.lss_cm','secondBearing.lss_cm'])
#         self.connect('lowSpeedShaft.dimensions', ['mainBearing.lss_dimensions','secondBearing.lss_dimensions'])
#         self.connect('mainBearing.cm', 'nacelleSystem.main_bearing_cm')
#         self.connect('secondBearing.cm', 'nacelleSystem.second_bearing_cm')
#         self.connect('bedplate.cm', ['nacelleSystem.bedplate_cm'])
#         self.connect('highSpeedSide.cm', ['nacelleSystem.hss_cm'])
#         self.connect('generator.cm', ['nacelleSystem.generator_cm'])

#         self.connect('lowSpeedShaft.I', ['nacelleSystem.lss_I'])
#         self.connect('mainBearing.I', 'nacelleSystem.main_bearing_I')
#         self.connect('secondBearing.I', 'nacelleSystem.second_bearing_I')
#         self.connect('gearbox.I', ['nacelleSystem.gearbox_I'])
#         self.connect('highSpeedSide.I', ['nacelleSystem.hss_I'])
#         self.connect('generator.I', ['nacelleSystem.generator_I'])
#         self.connect('bedplate.I', ['nacelleSystem.bedplate_I'])

#         # create passthroughs
#         self.connect('nacelleSystem.nacelle_mass', 'nacelle_mass')
#         self.connect('nacelleSystem.nacelle_cm', 'nacelle_cm')
#         self.connect('nacelleSystem.nacelle_I', 'nacelle_I')


#------------------------------------------------------------------
#examples

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

    #variables if check_fatigue =2:
    #test distribution
    # p_o = 6444.24
    # R = nace.rotor_diameter*.5
    # count = np.logspace(log10(328),log10(288984470), endpoint=True , num=100)
    # standard = count.copy()
    # for i in range(len(count)):
    #     standard[i] = .11*2.5*.28*13.4*(log10(288984470)-log10(count[i]))+0.18
    # Fx_factor = (.3649*log(nace.rotor_diameter)-1.074)
    # Mx_factor = (.0799*log(nace.rotor_diameter)-.2577)
    # My_factor = (.172*log(nace.rotor_diameter)-.5943)
    # Mz_factor = (.1659*log(nace.rotor_diameter)-.5795)
    # nace.rotor_thrust_distribution = standard.copy()**0.5*p_o*(R)*Fx_factor
    # nace.rotor_thrust_count = np.logspace(log10(328),log10(288984470), endpoint=True , num=100)
    # nace.rotor_Fy_distribution = np.zeros(100)
    # nace.rotor_Fy_count = nace.rotor_thrust_count.copy()
    # nace.rotor_Fz_distribution = np.zeros(100)
    # nace.rotor_Fz_count = nace.rotor_thrust_count.copy()
    # nace.rotor_torque_distribution = standard.copy()*0.45*p_o*(R)**2*Mx_factor
    # nace.rotor_torque_count = nace.rotor_thrust_count.copy()
    # nace.rotor_My_distribution = standard.copy()*0.33*p_o*.8*(R)**2*My_factor
    # nace.rotor_My_count = nace.rotor_thrust_count.copy()
    # nace.rotor_Mz_distribution = standard.copy()*0.33*p_o*.8*(R)**2*Mz_factor
    # nace.rotor_Mz_count = nace.rotor_thrust_count.copy()
    # NREL 5 MW Tower Variables

    nace.tower_top_diameter = 3.78 # m

    nace.run()

    #cm_print(nace)
    sys_print(nace)

def nacelle_example_1p5MW_3pt():
    
    # test of module for turbine data set

    # 1.5 MW Rotor Variables 
    print '----- NREL 1p5MW  Drivetrain - 3 Point Suspension-----'
    nace = Drive3pt()
    nace.rotor_diameter = 77 # m
    nace.rotor_speed = 16.18  #rpm
    nace.DrivetrainEfficiency = 0.95
    nace.machine_rating = 1500
    nace.rotor_torque =  1.5 * (nace.machine_rating * 1000 / nace.DrivetrainEfficiency) / (nace.rotor_speed * (pi / 30)) # 6.35e6 #4365248.74 # Nm
    nace.rotor_thrust = 2.6204e5 #
    nace.rotor_mass = 0.0 #
    nace.rotor_speed = 16.18 #rpm
    nace.rotor_bending_moment = 2.7795e6 #
    nace.rotor_bending_moment_x = 8.4389e5 #
    nace.rotor_bending_moment_y = -2.6758e6 #
    nace.rotor_bending_moment_z = 7.5222e2 #
    nace.rotor_force_x = 2.6204e5 #
    nace.rotor_force_y = 2.8026e4 #
    nace.rotor_force_z = -3.4763e5 #


    # 1p5MW  Drivetrain variables
    nace.drivetrain_design = 'geared' # geared 3-stage Gearbox with induction generator machine
    nace.machine_rating = 1500.0 # kW
    nace.gear_ratio = 78 # 
    nace.gear_configuration = 'epp' # epicyclic-parallel-parallel
    nace.crane = False # onboard crane not present
    nace.shaft_angle = 5.0 #deg
    nace.shaft_ratio = 0.10
    nace.Np = [3,1,1]
    nace.ratio_type = 'optimal'
    nace.shaft_type = 'normal'
    nace.uptower_transformer=False #True
    nace.shrink_disc_mass = 333.3*nace.machine_rating/1000.0 # estimated
    nace.carrier_mass = 2000.0 # estimated
    nace.mb1Type = 'SRB'
    nace.mb2Type = 'SRB'
    nace.flange_length = 0.285 #m
    nace.overhang = 3.3
    nace.L_rb = 1.535 # length from hub center to main bearing, leave zero if unknown

    nace.check_fatigue = 0 #0 if no fatigue check, 1 if parameterized fatigue check, 2 if known loads inputs

    #variables if check_fatigue = 1:
    nace.blade_number=3
    nace.cut_in=3.5 #cut-in m/s
    nace.cut_out=20. #cut-out m/s
    nace.Vrated=11.5 #rated windspeed m/s
    nace.weibull_k = 2.2 # windepeed distribution shape parameter
    nace.weibull_A = 9. # windspeed distribution scale parameter
    nace.T_life=20. #design life in years
    nace.IEC_Class_Letter = 'B'

    #variables if check_fatigue =2:
    # nace.rotor_thrust_distribution = 
    # nace.rotor_thrust_count = 
    # nace.rotor_Fy_distribution = 
    # nace.rotor_Fy_count = 
    # nace.rotor_Fz_distribution = 
    # nace.rotor_Fz_count = 
    # nace.rotor_torque_distribution = 
    # nace.rotor_torque_count = 
    # nace.rotor_My_distribution = 
    # nace.rotor_My_count = 
    # nace.rotor_Mz_distribution = 
    # nace.rotor_Mz_count =     

    # 1p5MW Tower Variables
    nace.tower_top_diameter = 2.3 # m

    nace.run()

    sys_print(nace)

def nacelle_example_1p5MW_4pt():
    
    # test of module for turbine data set

    print '----- NREL 1p5MW  Drivetrain - 4 Point Suspension-----'
    nace = Drive4pt()
    nace.rotor_diameter = 77 # m
    nace.rotor_speed = 16.18  #rpm
    nace.DrivetrainEfficiency = 0.95
    nace.machine_rating = 1500
    nace.rotor_torque =  1.5 * (nace.machine_rating * 1000 / nace.DrivetrainEfficiency) / (nace.rotor_speed * (pi / 30)) # 6.35e6 #4365248.74 # Nm
    nace.rotor_thrust = 2.6204e5 #
    nace.rotor_mass = 0.0 #
    nace.rotor_speed = 16.18 #rpm
    nace.rotor_bending_moment = 2.7795e6 #
    nace.rotor_bending_moment_x = 8.4389e5 #
    nace.rotor_bending_moment_y = -2.6758e6 #
    nace.rotor_bending_moment_z = 7.5222e2 #
    nace.rotor_force_x = 2.6204e5 #
    nace.rotor_force_y = 2.8026e4 #
    nace.rotor_force_z = -3.4763e5 #

    # 1p5MW  Drivetrain variables
    nace.drivetrain_design = 'geared' # geared 3-stage Gearbox with induction generator machine
    nace.machine_rating = 1500.0 # kW
    nace.gear_ratio = 78 # 
    nace.gear_configuration = 'epp' # epicyclic-parallel-parallel
    nace.crane = False #True # onboard crane present
    nace.shaft_angle = 5.0 #deg
    nace.shaft_ratio = 0.10
    nace.Np = [3,1,1]
    nace.ratio_type = 'optimal'
    nace.shaft_type = 'normal'
    nace.uptower_transformer=False #True
    nace.shrink_disc_mass = 333.3*nace.machine_rating/1000.0 # estimated
    nace.carrier_mass = 2000.0 # estimated
    nace.mb1Type = 'CARB'
    nace.mb2Type = 'SRB'
    nace.flange_length = 0.285 #m
    nace.overhang = 4
    nace.L_rb =  1.3# length from hub center to main bearing, leave zero if unknown
    nace.gearbox_cm = 0.0

    nace.check_fatigue = 0 #0 if no fatigue check, 1 if parameterized fatigue check, 2 if known loads inputs

    #variables if check_fatigue = 1:
    nace.blade_number=3
    nace.cut_in=3.5 #cut-in m/s
    nace.cut_out=20. #cut-out m/s
    nace.Vrated=11.5 #rated windspeed m/s
    nace.weibull_k = 2.2 # windepeed distribution shape parameter
    nace.weibull_A = 9. # windspeed distribution scale parameter
    nace.T_life=20. #design life in years
    nace.IEC_Class_Letter = 'B'

    #variables if check_fatigue =2:
    # nace.rotor_thrust_distribution = 
    # nace.rotor_thrust_count = 
    # nace.rotor_Fy_distribution = 
    # nace.rotor_Fy_count = 
    # nace.rotor_Fz_distribution = 
    # nace.rotor_Fz_count = 
    # nace.rotor_torque_distribution = 
    # nace.rotor_torque_count = 
    # nace.rotor_My_distribution = 
    # nace.rotor_My_count = 
    # nace.rotor_Mz_distribution = 
    # nace.rotor_Mz_count = 

    # 1p5MW Tower Variables
    nace.tower_top_diameter = 2.3 # m

    nace.run()

    #cm_print(nace)
    sys_print(nace)

def nacelle_example_p75_3pt():

    # test of module for turbine data set
    print '----- NREL 750kW Design - 3 Point Suspension----'
    # 0.75MW Rotor Variables
    nace = Drive3pt()
    nace.rotor_diameter = 48.2 # m
    nace.rotor_speed = 22.0 # #rpm m/s
    nace.DrivetrainEfficiency = 0.95
    nace.machine_rating = 750
    nace.rotor_torque =  1.5 * (nace.machine_rating * 1000 / nace.DrivetrainEfficiency) / (nace.rotor_speed * (pi / 30)) # 6.35e6 #4365248.74 # Nm
 
    nace.rotor_thrust = 143000.0 # # N
    nace.rotor_mass = 0.0 # kg
    nace.rotor_speed = 22.0 #rpm
    nace.rotor_bending_moment = 495.6e3 #
    nace.rotor_bending_moment_x = 401.0e3 #
    nace.rotor_bending_moment_y = 495.6e3 #
    nace.rotor_bending_moment_z = -443.0e3 #
    nace.rotor_force_x = 143000.0 #
    nace.rotor_force_y = -12600.0 #
    nace.rotor_force_z = -142.0e3 #

    # NREL 750 kW Drivetrain variables
    nace.drivetrain_design = 'geared' # geared 3-stage Gearbox with induction generator machine
    nace.machine_rating = 750 # kW
    nace.gear_ratio = 81.491 #  
    nace.gear_configuration = 'epp' # epicyclic-parallel-parallel
    nace.crane = False #True if onboard crane present
    nace.shaft_angle = 5.0 #deg
    nace.shaft_length = 2.1 #m
    nace.shaft_ratio = 0.10
    nace.Np = [3,1,1]
    nace.ratio_type = 'optimal'
    nace.shaft_type = 'normal'
    nace.uptower_transformer=False
    nace.shrink_disc_mass = 333.3*nace.machine_rating/1000.0 # estimated
    nace.carrier_mass = 250. # estimated
    nace.mb1Type = 'SRB'
    nace.mb2Type = 'TRB2'
    nace.flange_length = 0.285 #m
    nace.overhang = 2.26
    nace.L_rb = 1.22 # length from hub center to main bearing, leave zero if unknown
    nace.gearbox_cm = 0.8
    nace.blade_root_diameter= 1.6

    nace.check_fatigue = 0 #0 if no fatigue check, 1 if parameterized fatigue check, 2 if known loads inputs

    #variables if check_fatigue = 1:
    nace.blade_number=3
    nace.cut_in=3. #cut-in m/s
    nace.cut_out=25. #cut-out m/s
    nace.Vrated=16. #rated windspeed m/s
    nace.weibull_k = 2.2 # windepeed distribution shape parameter
    nace.weibull_A = 9. # windspeed distribution scale parameter
    nace.T_life=20. #design life in years
    nace.IEC_Class_Letter = 'A'


    #variables if check_fatigue =2:
    # nace.rotor_thrust_distribution = 
    # nace.rotor_thrust_count = 
    # nace.rotor_Fy_distribution = 
    # nace.rotor_Fy_count = 
    # nace.rotor_Fz_distribution = 
    # nace.rotor_Fz_count = 
    # nace.rotor_torque_distribution = 
    # nace.rotor_torque_count = 
    # nace.rotor_My_distribution = 
    # nace.rotor_My_count = 
    # nace.rotor_Mz_distribution = 
    # nace.rotor_Mz_count = 

    # 0.75MW Tower Variables
    nace.tower_top_diameter = 2.21 # m

    nace.run()
    #cm_print(nace)
    sys_print(nace)
       
def nacelle_example_p75_4pt():

    # test of module for turbine data set
    print '----- NREL 750kW Design - 4 Point Suspension----'
    # 0.75MW Rotor Variables
    nace = Drive4pt()
    nace.rotor_diameter = 48.2 # m
    nace.rotor_speed = 22.0 # #rpm 
    nace.DrivetrainEfficiency = 0.95
    nace.machine_rating = 750
    nace.rotor_torque =  1.5 * (nace.machine_rating * 1000 / nace.DrivetrainEfficiency) / (nace.rotor_speed * (pi / 30)) # 6.35e6 #4365248.74 # Nm
    #nace.rotor_torque = 6.37e6 #
    nace.rotor_thrust = 143000.0 #
    nace.rotor_mass = 0.0 #
    nace.rotor_speed = 22.0 #rpm
    nace.rotor_bending_moment = 495.6e3 #
    nace.rotor_bending_moment_x = 401.0e3 #
    nace.rotor_bending_moment_y = 495.6e3 #
    nace.rotor_bending_moment_z = -443.0e3 #
    nace.rotor_force_x = 143000.0 #
    nace.rotor_force_y = -12600.0 #
    nace.rotor_force_z = -142.0e3 #

    # NREL 750 kW Drivetrain variables
    nace.drivetrain_design = 'geared' # geared 3-stage Gearbox with induction generator machine
    nace.machine_rating = 750 # kW
    nace.gear_ratio = 81.491 #  as listed in the 5 MW reference document
    nace.gear_configuration = 'epp' # epicyclic-parallel-parallel
    nace.crane = False #True # onboard crane present
    nace.shaft_angle = 5.0 #deg
    nace.shaft_length = 2.1 #m
    nace.shaft_ratio = 0.10
    nace.Np = [3,1,1]
    nace.ratio_type = 'optimal'
    nace.shaft_type = 'normal'
    nace.uptower_transformer=False #True
    nace.shrink_disc_mass = 333.3*nace.machine_rating/1000.0 # estimated
    nace.carrier_mass = 1000.0 # estimated
    nace.mb1Type = 'SRB'
    nace.mb2Type = 'TRB2'
    nace.flange_length = 0.338 #m
    nace.overhang = 2.26
    nace.L_rb = 1.22 # 0.007835*rotor_diameter+0.9642 length from hub center to main bearing, leave zero if unknown
    nace.gearbox_cm = 0.90

    nace.check_fatigue = 0 #0 if no fatigue check, 1 if parameterized fatigue check, 2 if known loads inputs

    #variables if check_fatigue = 1:
    nace.blade_number=3
    nace.cut_in=3. #cut-in m/s
    nace.cut_out=25. #cut-out m/s
    nace.Vrated=16. #rated windspeed m/s
    nace.weibull_k = 2.2 # windepeed distribution shape parameter
    nace.weibull_A = 9. # windspeed distribution scale parameter
    nace.T_life=20. #design life in years
    nace.IEC_Class_Letter = 'A'

    #variables if check_fatigue =2:
    # nace.rotor_thrust_distribution = 
    # nace.rotor_thrust_count = 
    # nace.rotor_Fy_distribution = 
    # nace.rotor_Fy_count = 
    # nace.rotor_Fz_distribution = 
    # nace.rotor_Fz_count = 
    # nace.rotor_torque_distribution = 
    # nace.rotor_torque_count = 
    # nace.rotor_My_distribution = 
    # nace.rotor_My_count = 
    # nace.rotor_Mz_distribution = 
    # nace.rotor_Mz_count = 

    # 0.75MW Tower Variables
    nace.tower_top_diameter = 2.21 # m

    nace.run()
    sys_print(nace)  

def cm_print(nace):
    print
    print '------------------Component CM and Mass:----------------------'
    print 'gearbox:         ', nace.gearbox.cm, '\n                 ',nace.gearbox.mass, '\n--------------------------------------------------------------'
    print 'highSpeedSide:   ', nace.highSpeedSide.cm,'\n                 ', nace.highSpeedSide.mass,'\n--------------------------------------------------------------'
    print 'generator:       ', nace.generator.cm,'\n                 ', nace.generator.mass,'\n--------------------------------------------------------------'
    print 'lowSpeed:        ', nace.lowSpeedShaft.cm,'\n                 ', nace.lowSpeedShaft.mass,'\n--------------------------------------------------------------'
    print 'bearing1:        ', nace.mainBearing.cm,'\n                 ', nace.mainBearing.mass,'\n--------------------------------------------------------------'
    print 'bearing2:        ', nace.secondBearing.cm,'\n                 ', nace.secondBearing.mass,'\n--------------------------------------------------------------'
    print 'yawSystem:       ', nace.yawSystem.cm,'\n                 ', nace.yawSystem.mass,'\n--------------------------------------------------------------'
    print 'bedplate:        ', nace.bedplate.cm,'\n                 ', nace.bedplate.mass,'\n--------------------------------------------------------------'
    print 'transformer:     ', nace.transformer.cm,'\n                 ', nace.transformer.mass,'\n--------------------------------------------------------------'
    print 'nacelleSystem:   ', nace.nacelleSystem.nacelle_cm,'\n                 ', nace.nacelleSystem.nacelle_mass,'\n--------------------------------------------------------------'

def sys_print(nace):
    print
    print '-------------Nacelle system model results--------------------'

    print 'Low speed shaft %8.1f kg %6.2f m %6.2f Ixx %6.2f Iyy %6.2f Izz %6.2f CGx %6.2f CGy %6.2f CGz '\
          % (nace.lowSpeedShaft.mass-nace.lowSpeedShaft.shrink_disc_mass , nace.lowSpeedShaft.length, nace.lowSpeedShaft.I[0], nace.lowSpeedShaft.I[1], nace.lowSpeedShaft.I[2], nace.lowSpeedShaft.cm[0], nace.lowSpeedShaft.cm[1], nace.lowSpeedShaft.cm[2])
    print 'LSS diameters:', 'upwind', nace.lowSpeedShaft.diameter1   , 'downwind', nace.lowSpeedShaft.diameter2 , 'inner', nace.lowSpeedShaft.diameter1*nace.shaft_ratio
    print 'Main bearing upwind   %8.1f kg. cm %8.1f %8.1f %8.1f' % (nace.mainBearing.mass ,nace.mainBearing.cm[0],nace.mainBearing.cm[1],nace.mainBearing.cm[2])
    print 'Second bearing downwind   %8.1f kg. cm %8.1f %8.1f %8.1f' % (nace.secondBearing.mass ,nace.secondBearing.cm[0],nace.secondBearing.cm[1],nace.secondBearing.cm[2])
    print 'Gearbox         %8.1f kg %6.2f Ixx %6.2f Iyy %6.2f Izz %6.2f CGx %6.2f CGy %6.2f CGz' \
          % (nace.gearbox.mass, nace.gearbox.I[0], nace.gearbox.I[1], nace.gearbox.I[2], nace.gearbox.cm[0], nace.gearbox.cm[1], nace.gearbox.cm[2] )
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
    print 'HVAC system     %8.1f kg' % (nace.above_yaw_massAdder.hvac_mass )
    print 'Nacelle cover:   %8.1f kg %6.2f m Height %6.2f m Width %6.2f m Length' % (nace.above_yaw_massAdder.cover_mass , nace.above_yaw_massAdder.height, nace.above_yaw_massAdder.width, nace.above_yaw_massAdder.length)
    print 'Yaw system      %8.1f kg' % (nace.yawSystem.mass )
    print 'Overall nacelle:  %8.1f kg .cm %6.2f %6.2f %6.2f I %6.2f %6.2f %6.2f' % (nace.nacelle_mass, nace.nacelle_cm[0], nace.nacelle_cm[1], nace.nacelle_cm[2], nace.nacelle_I[0], nace.nacelle_I[1], nace.nacelle_I[2]  )
    print



if __name__ == '__main__':
    ''' Main runs through tests of several drivetrain configurations with known component masses and dimensions '''

    nacelle_example_5MW_baseline_3pt()

    nacelle_example_5MW_baseline_4pt()

    # nacelle_example_1p5MW_3pt()

    # nacelle_example_1p5MW_4pt()

    # nacelle_example_p75_3pt()

    # nacelle_example_p75_4pt()

