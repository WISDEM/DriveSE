# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 12:28:29 2019

@author: gscott

Set up and run a DriveSE model using the new spherical hub and mainshaft flange - 2019 07 22
  HubSE, HubMassOnlySE, Hub_CM_Adder_OM are now imported from sph_hubse_omdao.py (instead of hubse_omdao.py)
"""

import numpy as np
import sys

#from drivese.drivese_components import LowSpeedShaft4pt, LowSpeedShaft3pt, Gearbox, MainBearing, Bedplate, YawSystem, \
#                                       Transformer, HighSpeedSide, Generator, NacelleSystemAdder, AboveYawMassAdder, RNASystemAdder
from drivese.drivese_components_clean import LowSpeedShaft4pt, LowSpeedShaft3pt, Gearbox, MainBearing, Bedplate, YawSystem, \
                                       Transformer, HighSpeedSide, Generator, NacelleSystemAdder, AboveYawMassAdder, RNASystemAdder
from drivese.sph_hubse_omdao import HubSE, HubMassOnlySE, Hub_CM_Adder_OM
from openmdao.api import Group, Component, IndepVarComp, Problem
from openmdao.api import view_model

#from drivese.drivese_omdao import Drive3pt
from drivese.drivese_omdao import LowSpeedShaft3pt_OM, LowSpeedShaft4pt_OM, \
                                  MainBearing_OM, Gearbox_OM, HighSpeedSide_OM, Generator_OM, \
                                  Bedplate_OM, Transformer_OM, RNASystemAdder_OM, AboveYawMassAdder_OM, YawSystem_OM, \
                                  NacelleSystemAdder_OM

#-------------------------------------------------------------------------
# Groups
#   (were Assemblies in OpenMDAO 0.x)
#-------------------------------------------------------------------------
    
class Drive3pt(Group):
    ''' Class Drive3pt defines an OpenMDAO group that represents a wind turbine drivetrain with a 3-point suspension.
        This Group can serve as the root of an OpenMDAO Problem.
    It contains the following components:
        HubMassOnlySE(blade_number)
        LowSpeedShaft3pt_OM(mb1Type, IEC_Class)
        MainBearing_OM('main')
        Hub_CM_Adder_OM()
        Gearbox_OM(gear_configuration, shaft_factor)
        HighSpeedSide_OM()
        Generator_OM(drivetrain_design)
        Bedplate_OM(uptower_transformer)
        Transformer_OM(uptower_transformer)
        RNASystemAdder_OM()
        AboveYawMassAdder_OM(crane)
        YawSystem_OM(yaw_motors_number)
        NacelleSystemAdder_OM()
    '''

    def __init__(self, mb1Type, IEC_Class, gear_configuration, shaft_factor, drivetrain_design, 
                 uptower_transformer, yaw_motors_number, crane, blade_number, debug=False):
        super(Drive3pt, self).__init__()

        # Add common inputs for drivetrain
        self.add('gear_ratio',        IndepVarComp('gear_ratio', 0.0),        promotes=['*'])
        self.add('shaft_angle',       IndepVarComp('shaft_angle', 0.0),       promotes=['*'])
        self.add('shaft_ratio',       IndepVarComp('shaft_ratio', 0.0),       promotes=['*'])
        self.add('shrink_disc_mass',  IndepVarComp('shrink_disc_mass', 0.0),  promotes=['*'])
        self.add('carrier_mass',      IndepVarComp('carrier_mass', 0.0),      promotes=['*'])
        self.add('flange_length',     IndepVarComp('flange_length', 0.0),     promotes=['*'])
        self.add('overhang',          IndepVarComp('overhang', 0.0),          promotes=['*'])
        self.add('distance_hub2mb',   IndepVarComp('distance_hub2mb', 0.0),   promotes=['*'])
        self.add('gearbox_input_xcm', IndepVarComp('gearbox_input_xcm', 0.0), promotes=['*'])
        self.add('hss_input_length',  IndepVarComp('hss_input_length', 0.0),  promotes=['*'])
        self.add('planet_numbers',    IndepVarComp('planet_numbers', np.array([0, 0, 0]), pass_by_obj=True), promotes=['*'])
        #self.add('drivetrain_efficiency', IndepVarComp('drivetrain_efficiency', 0.0), promotes=['*'])

        # Add common inputs for tower
        self.add('tower_top_diameter',IndepVarComp([('tower_top_diameter', 0.0)]), promotes=['*'])

        # Add some more IndepVarComps to get rid of 'no associated unknowns' message
        self.add('rotor_diameter',         IndepVarComp('rotor_diameter',         0.0), promotes=['*'])
        self.add('rotor_rpm',              IndepVarComp('rotor_rpm',              0.0), promotes=['*'])
        self.add('rotor_torque',           IndepVarComp('rotor_torque',           0.0), promotes=['*'])
        self.add('rotor_thrust',           IndepVarComp('rotor_thrust',           0.0), promotes=['*'])
        self.add('rotor_bending_moment_x', IndepVarComp('rotor_bending_moment_x', 0.0), promotes=['*'])
        self.add('rotor_bending_moment_y', IndepVarComp('rotor_bending_moment_y', 0.0), promotes=['*'])
        self.add('rotor_bending_moment_z', IndepVarComp('rotor_bending_moment_z', 0.0), promotes=['*'])
        self.add('rotor_force_y',          IndepVarComp('rotor_force_y',          0.0), promotes=['*'])
        self.add('rotor_force_z',          IndepVarComp('rotor_force_z',          0.0), promotes=['*'])
        self.add('blade_mass',             IndepVarComp('blade_mass',             0.0), promotes=['*'])
        self.add('blade_root_diameter',    IndepVarComp('blade_root_diameter',    0.0), promotes=['*'])
        self.add('blade_length',           IndepVarComp('blade_length',           0.0), promotes=['*'])
        self.add('drivetrain_efficiency',  IndepVarComp('drivetrain_efficiency',  0.0), promotes=['*'])
        self.add('machine_rating',         IndepVarComp('machine_rating',         0.0), promotes=['*'])
        
        # Create 3 pt drivetrain group
        self.add('hub',                 HubMassOnlySE(blade_number, debug=debug),                  promotes=['*'])
        self.add('lowSpeedShaft',       LowSpeedShaft3pt_OM(mb1Type, IEC_Class, debug=debug),      promotes=['*'])
        self.add('mainBearing',         MainBearing_OM('main'),                                    promotes=['lss_design_torque','rotor_diameter']) #need to make explicit connections for main bearing
        self.add('hubCM',               Hub_CM_Adder_OM(),                                         promotes=['*'])
        self.add('gearbox',             Gearbox_OM(gear_configuration, shaft_factor, debug=debug), promotes=['*'])
        self.add('highSpeedSide',       HighSpeedSide_OM(),                                        promotes=['*'])
        self.add('generator',           Generator_OM(drivetrain_design),                           promotes=['*'])
        self.add('bedplate',            Bedplate_OM(uptower_transformer, debug=debug),             promotes=['*'])
        self.add('transformer',         Transformer_OM(uptower_transformer),                       promotes=['*'])
        self.add('rna',                 RNASystemAdder_OM(),                                       promotes=['*'])
        self.add('above_yaw_massAdder', AboveYawMassAdder_OM(crane, debug=debug),                  promotes=['*'])
        self.add('yawSystem',           YawSystem_OM(yaw_motors_number),                           promotes=['*'])
        self.add('nacelleSystem',       NacelleSystemAdder_OM(),                                   promotes=['*'])

        # Connect components where explicit connections needed (for main bearing)
        self.connect('lss_mb1_mass', ['mainBearing.bearing_mass'])
        self.connect('lss_diameter1', ['mainBearing.lss_diameter', 'lss_diameter'])
        self.connect('lss_mb1_cm', ['mainBearing.lss_mb_cm'])
        self.connect('mainBearing.mb_mass', ['mb1_mass'])
        self.connect('mainBearing.mb_cm', ['mb1_cm', 'MB1_location'])
        self.connect('mainBearing.mb_I', ['mb1_I'])

        self.connect('lss_cm','lss_location',src_indices=[0])
        self.connect('hss_cm','hss_location',src_indices=[0])
        self.connect('gearbox_cm','gearbox_location',src_indices=[0])
        self.connect('generator_cm','generator_location',src_indices=[0])
        
#------------------------------------------------------------------
   
class Drive4pt(Group):
    ''' Class Drive4pt defines an OpenMDAO group that represents a wind turbine drivetrain with a 4-point suspension.
        This Group can serve as the root of an OpenMDAO Problem.
    '''

    def __init__(self, mb1Type, mb2Type, IEC_Class, gear_configuration, shaft_factor, drivetrain_design, 
                 uptower_transformer, yaw_motors_number, crane, blade_number, debug=False):
        super(Drive4pt, self).__init__()
        
        # Add common inputs for drivetrain
        self.add('gear_ratio',        IndepVarComp('gear_ratio',        0.0), promotes=['*'])
        self.add('shaft_angle',       IndepVarComp('shaft_angle',       0.0), promotes=['*'])
        self.add('shaft_ratio',       IndepVarComp('shaft_ratio',       0.0), promotes=['*'])
        self.add('shrink_disc_mass',  IndepVarComp('shrink_disc_mass',  0.0), promotes=['*'])
        self.add('carrier_mass',      IndepVarComp('carrier_mass',      0.0), promotes=['*'])
        self.add('flange_length',     IndepVarComp('flange_length',     0.0), promotes=['*'])
        self.add('overhang',          IndepVarComp('overhang',          0.0), promotes=['*'])
        self.add('distance_hub2mb',   IndepVarComp('distance_hub2mb',   0.0), promotes=['*'])
        self.add('gearbox_input_xcm', IndepVarComp('gearbox_input_xcm', 0.0), promotes=['*'])
        self.add('hss_input_length',  IndepVarComp('hss_input_length',  0.0), promotes=['*'])
        self.add('planet_numbers',    IndepVarComp('planet_numbers', np.array([0, 0, 0]), pass_by_obj=True), promotes=['*'])
        #self.add('drivetrain_efficiency', IndepVarComp('drivetrain_efficiency', 0.0), promotes=['*'])
        
        # Add common inputs for tower
        #self.add('tower_top_diameter',IndepVarComp([('tower_top_diameter', 0.0)]), promotes=['*'])
        self.add('tower_top_diameter',IndepVarComp('tower_top_diameter', 0.0), promotes=['*'])

        # Add some more IndepVarComps to get rid of 'no associated unknowns' message
        self.add('rotor_diameter',         IndepVarComp('rotor_diameter',         0.0), promotes=['*'])
        self.add('rotor_rpm',              IndepVarComp('rotor_rpm',              0.0), promotes=['*'])
        self.add('rotor_torque',           IndepVarComp('rotor_torque',           0.0), promotes=['*'])
        self.add('rotor_thrust',           IndepVarComp('rotor_thrust',           0.0), promotes=['*'])
        self.add('rotor_bending_moment_x', IndepVarComp('rotor_bending_moment_x', 0.0), promotes=['*'])
        self.add('rotor_bending_moment_y', IndepVarComp('rotor_bending_moment_y', 0.0), promotes=['*'])
        self.add('rotor_bending_moment_z', IndepVarComp('rotor_bending_moment_z', 0.0), promotes=['*'])
        self.add('rotor_force_y',          IndepVarComp('rotor_force_y',          0.0), promotes=['*'])
        self.add('rotor_force_z',          IndepVarComp('rotor_force_z',          0.0), promotes=['*'])
        self.add('blade_mass',             IndepVarComp('blade_mass',             0.0), promotes=['*'])
        self.add('blade_root_diameter',    IndepVarComp('blade_root_diameter',    0.0), promotes=['*'])
        self.add('blade_length',           IndepVarComp('blade_length',           0.0), promotes=['*'])
        self.add('drivetrain_efficiency',  IndepVarComp('drivetrain_efficiency',  0.0), promotes=['*'])
        self.add('machine_rating',         IndepVarComp('machine_rating',         0.0), promotes=['*'])
        
        # select components
        self.add('hub',                 HubMassOnlySE(blade_number, debug=debug),                      promotes=['*'])
        self.add('lowSpeedShaft',       LowSpeedShaft4pt_OM(mb1Type, mb2Type, IEC_Class, debug=debug), promotes=['*'])
        self.add('mainBearing',         MainBearing_OM('main'),                           promotes=['lss_design_torque','rotor_diameter']) #explicit connections for bearings
        self.add('secondBearing',       MainBearing_OM('second'),                         promotes=['lss_design_torque','rotor_diameter']) #explicit connections for bearings
        self.add('hubCM',               Hub_CM_Adder_OM(),                                promotes=['*'])
        self.add('gearbox',             Gearbox_OM(gear_configuration, shaft_factor, debug=debug),     promotes=['*'])
        self.add('highSpeedSide',       HighSpeedSide_OM(),                               promotes=['*'])
        self.add('generator',           Generator_OM(drivetrain_design),                  promotes=['*'])
        self.add('bedplate',            Bedplate_OM(uptower_transformer, debug=debug),                 promotes=['*'])
        self.add('transformer',         Transformer_OM(uptower_transformer),              promotes=['*'])
        self.add('rna',                 RNASystemAdder_OM(),                              promotes=['*'])
        self.add('above_yaw_massAdder', AboveYawMassAdder_OM(crane, debug=debug),                      promotes=['*'])
        self.add('yawSystem',           YawSystem_OM(yaw_motors_number),                  promotes=['*'])
        self.add('nacelleSystem',       NacelleSystemAdder_OM(),                          promotes=['*'])

        # Connect components where explicit connections needed (for main bearings)
        self.connect('lss_mb1_mass',          ['mainBearing.bearing_mass'])
        self.connect('lss_diameter1',         ['mainBearing.lss_diameter', 'lss_diameter'])
        self.connect('lss_mb1_cm',            ['mainBearing.lss_mb_cm'])
        self.connect('mainBearing.mb_mass',   ['mb1_mass'])
        self.connect('mainBearing.mb_cm',     ['mb1_cm', 'MB1_location'])
        self.connect('mainBearing.mb_I',      ['mb1_I'])

        self.connect('lss_mb2_mass',          ['secondBearing.bearing_mass'])
        self.connect('lss_diameter2',         ['secondBearing.lss_diameter'])
        self.connect('lss_mb2_cm',            ['secondBearing.lss_mb_cm'])
        self.connect('secondBearing.mb_mass', ['mb2_mass'])
        self.connect('secondBearing.mb_cm',   ['mb2_cm'])
        self.connect('secondBearing.mb_I',    ['mb2_I'])
                                              
        self.connect('lss_cm',       'lss_location',       src_indices=[0])
        self.connect('hss_cm',       'hss_location',       src_indices=[0])
        self.connect('gearbox_cm',   'gearbox_location',   src_indices=[0])
        self.connect('generator_cm', 'generator_location', src_indices=[0])

        
#------------------------------------------------------------------
# examples

def sph_nacelle_example_5MW_baseline_3pt(debug=False):

    # NREL 5 MW Drivetrain variables
    # geared 3-stage Gearbox with induction generator machine
    drivetrain_design = 'geared'
    gear_configuration = 'eep'  # epicyclic-epicyclic-parallel
    mb1Type = 'SRB'
    IEC_Class = 'B'
    shaft_factor = 'normal'
    uptower_transformer = True
    crane = True  # onboard crane present
    yaw_motors_number = 0 # default value - will be internally calculated
    blade_number = 3

    runid = 'N5_sph_3pt'
    modid = ''
    
    prob=Problem(root=Drive3pt(mb1Type, IEC_Class, gear_configuration, shaft_factor, drivetrain_design, 
                               uptower_transformer, yaw_motors_number, crane, blade_number,
                               debug=debug))
    prob.setup()
    if False:
        view_model(prob, show_browser=True)
        return

    # Rotor and load inputs
    prob['rotor_diameter'] = 126.0  # m
    prob['rotor_rpm'] = 12.1  # rpm m/s
    prob['machine_rating'] = 5000.0
    prob['drivetrain_efficiency'] = 0.95
    prob['rotor_torque'] = 1.5 * (prob['machine_rating'] * 1000 / prob['drivetrain_efficiency']) \
                              / (prob['rotor_rpm'] * (np.pi / 30))
    #prob['rotor_thrust'] = 599610.0  # N
    prob['rotor_mass'] = 0.0  # accounted for in F_z # kg
    prob['rotor_bending_moment_x'] =    330770.0  # Nm
    prob['rotor_bending_moment_y'] = -16665000.0  # Nm
    prob['rotor_bending_moment_z'] =   2896300.0  # Nm
    prob['rotor_thrust'] =   599610.0  # N
    prob['rotor_force_y'] =  186780.0  # N
    prob['rotor_force_z'] = -842710.0  # N

    # Drivetrain inputs
    prob['machine_rating'] = 5000.0  # kW
    prob['gear_ratio'] = 96.76  # 97:1 as listed in the 5 MW reference document
    prob['shaft_angle'] = 5.0*np.pi / 180.0  # rad
    prob['shaft_ratio'] = 0.10
    prob['planet_numbers'] = [3, 3, 1]
    prob['shrink_disc_mass'] = 333.3 * prob['machine_rating'] / 1000.0  # estimated
    prob['carrier_mass'] = 8000.0  # estimated
    prob['flange_length'] = 0 # 0.5
    prob['overhang'] = 5.0
    prob['distance_hub2mb'] = 0 #1.912  # length from hub center to main bearing, leave zero if unknown
    prob['gearbox_input_xcm'] = 0.1
    prob['hss_input_length'] = 1.5

    # try this:
    prob['blade_mass'] = 17740.0
    prob['blade_root_diameter'] = 2.5
    prob['blade_length'] = 60.0
    
    prob['hub_thickness'] = 0.010
    
    # Tower inputs
    prob['tower_top_diameter'] = 3.78  # m

    # test options
    #prob['rotor_force_z'] = 0; modid += '_RFZ0'

    prob.run()

    if debug:
        print('----- NREL 5 MW Turbine - Spherical hub - 3 Point Suspension -----')
        #print(prob.root.unknowns.dump())
        
        # dump to file - gns 2019 04 29
        #ofname = 'N5_sph_3pt_dump.txt'
        ofname = '{}{}_dump.txt'.format(runid, modid)
        ofh = open(ofname, 'w')
        ofh.write('----- NREL 5 MW Turbine - Spherical hub - 3 Point Suspension -----\n')
        prob.root.unknowns.dump(out_stream=ofh)
        ofh.close()
        sys.stderr.write('Dumped unknowns to {}\n'.format(ofname))

    return prob

#-------------------------------
    
def sph_nacelle_example_5MW_baseline_4pt(debug=False):

    # NREL 5 MW Drivetrain variables
    # geared 3-stage Gearbox with induction generator machine
    drivetrain_design = 'geared'
    gear_configuration = 'eep'  # epicyclic-epicyclic-parallel
    mb1Type = 'CARB'
    mb2Type = 'SRB'
    IEC_Class = 'B'
    shaft_factor = 'normal'
    uptower_transformer = True
    crane = True  # onboard crane present
    yaw_motors_number = 0 # default value - will be internally calculated
    blade_number = 3

    runid = 'N5_sph_4pt'
    modid = ''
    
    prob=Problem(root=Drive4pt(mb1Type, mb2Type, IEC_Class, gear_configuration, shaft_factor, drivetrain_design, 
                               uptower_transformer, yaw_motors_number, crane, blade_number,
                               debug=debug))
    prob.setup()
    #view_model(prob, show_browser=True)
    #return

    # Rotor and load inputs
    prob['rotor_diameter'] = 126.0  # m
    prob['rotor_rpm'] = 12.1  # rpm m/s
    prob['machine_rating'] = 5000.0
    prob['drivetrain_efficiency'] = 0.95
    prob['rotor_torque'] = 1.5 * (prob['machine_rating'] * 1000 / \
                             prob['drivetrain_efficiency']) / (prob['rotor_rpm'] * (np.pi / 30))
    prob['rotor_mass'] = 0.0  # accounted for in F_z # kg
    prob['rotor_bending_moment_x'] =    330770.0  # Nm
    prob['rotor_bending_moment_y'] = -16665000.0  # Nm
    prob['rotor_bending_moment_z'] =   2896300.0  # Nm
    prob['rotor_thrust'] =   599610.0  # N
    prob['rotor_force_y'] =  186780.0  # N
    prob['rotor_force_z'] = -842710.0  # N

    # Drivetrain inputs
    prob['machine_rating'] = 5000.0  # kW
    prob['gear_ratio'] = 96.76  # 97:1 as listed in the 5 MW reference document
    prob['shaft_angle'] = 5.0*np.pi / 180.0  # rad
    prob['shaft_ratio'] = 0.10
    prob['planet_numbers'] = [3, 3, 1]
    prob['shrink_disc_mass'] = 333.3 * prob['machine_rating'] / 1000.0  # estimated
    prob['carrier_mass'] = 8000.0  # estimated
    prob['flange_length'] = 0.5
    prob['overhang'] = 5.0
    prob['distance_hub2mb'] = 1.912  # length from hub center to main bearing, leave zero if unknown
    prob['gearbox_input_xcm'] = 0.1
    prob['hss_input_length'] = 1.5

    # try this:
    prob['blade_mass'] = 17740.0
    prob['blade_root_diameter'] = 2.5
    prob['blade_length'] = 60.0
    
    # Tower inputs
    prob['tower_top_diameter'] = 3.78  # m

    prob.run()

    if debug:
        print('----- NREL 5 MW Turbine - 4 Point Suspension -----')
        print(prob.root.unknowns.dump())
        
        # dump to file - gns 2019 04 29
        #ofname = 'N5_sph_4pt_dump.txt'
        ofname = '{}{}_dump.txt'.format(runid, modid)
        ofh = open(ofname, 'w')
        ofh.write('----- NREL 5 MW Turbine - 4 Point Suspension -----\n')
        prob.root.unknowns.dump(out_stream=ofh)
        ofh.close()
        sys.stderr.write('Dumped unknowns to {}\n'.format(ofname))

    return prob
#%%-------------------------------
    
if __name__ == '__main__':
    ''' Main runs through tests of several drivetrain configurations with known component masses and dimensions '''

    debug = True
    #debug = False
    
    #sph_nacelle_example_5MW_baseline_3pt(debug=debug)
    sph_nacelle_example_5MW_baseline_4pt(debug=debug)

