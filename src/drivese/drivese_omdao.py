"""
DriveSE.py

Created by Yi Guo, Taylor Parsons and Ryan King 2014.
Copyright (c) NREL. All rights reserved.
"""

from openmdao.api import Group, Problem
import numpy as np

# FUSED wrapper
from fusedwind.fused_openmdao import FUSED_Component, FUSED_Group, FUSED_add, FUSED_connect, FUSED_print, \
                                     FUSED_Problem, FUSED_setup, FUSED_run, FUSED_VarComp

# FUSED drivetrain components
from fused_drivese import FUSED_Gearbox, FUSED_MainBearing, FUSED_Bedplate, FUSED_YawSystem, FUSED_LowSpeedShaft3pt, \
    FUSED_LowSpeedShaft4pt, FUSED_Transformer, FUSED_HighSpeedSide, FUSED_Generator, FUSED_NacelleSystemAdder, FUSED_AboveYawMassAdder, FUSED_RNASystemAdder

def Drive3pt(drive_group, mb1Type, IEC_Class, gear_configuration, shaft_factor, drivetrain_design, uptower_transformer, yaw_motors_number, crane):

    # Add common inputs for rotor
    FUSED_add(drive_group, 'rotorvars',FUSED_VarComp([('rotor_diameter', 0.0),
								    																 ('rotor_bending_moment_x', 0.0),
								    																 ('rotor_bending_moment_y', 0.0),
								    																 ('rotor_bending_moment_z', 0.0),
								    																 ('rotor_thrust', 0.0),
								    																 ('rotor_force_y',0.0),
								    																 ('rotor_force_z', 0.0),
								    																 ('rotor_torque', 0.0),
								    																 ('rotor_mass', 0.0),								    																 
								    																 ]),['*'])

    # Add common inputs for drivetrain
    FUSED_add(drive_group, 'drivevars',FUSED_VarComp([('machine_rating', 0.0),
								    																 ('gear_ratio', 0.0),
								    																 ('flange_length', 0.0),
								    																 ('overhang', 0.0),
								    																 ('distance_hub2mb', 0.0),
								    																 ('gearbox_input_cm', 0.0),
								    																 ('hss_input_length', 0.0),
								    																 ]),['*'])

    # Add common inputs for tower
    FUSED_add(drive_group, 'towervars',FUSED_VarComp([('tower_top_diameter', 0.0)]),['*'])

    # Create 3 pt drivetrain group
    FUSED_add(drive_group, 'lowSpeedShaft', FUSED_Component(FUSED_LowSpeedShaft3pt(mb1Type, IEC_Class)), ['*'])
    FUSED_add(drive_group, 'mainBearing', FUSED_Component(FUSED_MainBearing('main')),['lss_design_torque','rotor_diameter']) #need to make explicit connections for main bearing
    FUSED_add(drive_group, 'gearbox', FUSED_Component(FUSED_Gearbox(gear_configuration, shaft_factor)), ['*'])
    FUSED_add(drive_group, 'highSpeedSide', FUSED_Component(FUSED_HighSpeedSide()), ['*'])
    FUSED_add(drive_group, 'generator', FUSED_Component(FUSED_Generator(drivetrain_design)), ['*'])
    FUSED_add(drive_group, 'bedplate', FUSED_Component(FUSED_Bedplate(uptower_transformer)), ['*'])
    FUSED_add(drive_group, 'transformer', FUSED_Component(FUSED_Transformer(uptower_transformer)), ['*'])
    FUSED_add(drive_group, 'rna', FUSED_Component(FUSED_RNASystemAdder()), ['*'])
    FUSED_add(drive_group, 'above_yaw_massAdder', FUSED_Component(FUSED_AboveYawMassAdder(crane)), ['*'])
    FUSED_add(drive_group, 'yawSystem', FUSED_Component(FUSED_YawSystem(yaw_motors_number)), ['*'])
    FUSED_add(drive_group, 'nacelleSystem', FUSED_Component(FUSED_NacelleSystemAdder()), ['*'])

    # Connect components where explicit connections needed (for main bearing)
    FUSED_connect(drive_group, 'lss_mb1_mass', ['mainBearing.bearing_mass'])
    FUSED_connect(drive_group, 'lss_diameter1', ['mainBearing.lss_diameter', 'lss_diameter'])
    FUSED_connect(drive_group, 'lss_mb1_cm', ['mainBearing.lss_mb_cm'])
    FUSED_connect(drive_group, 'mainBearing.mb_mass', ['mb1_mass'])
    FUSED_connect(drive_group, 'mainBearing.mb_cm', ['mb1_cm'])
    FUSED_connect(drive_group, 'mainBearing.mb_I', ['mb1_I'])

#------------------------------------------------------------------
def Drive4pt(drive_group, mb1Type, mb2Type, IEC_Class, gear_configuration, shaft_factor, drivetrain_design, uptower_transformer, yaw_motors_number, crane):

    # Add common inputs for rotor
    FUSED_add(drive_group, 'rotorvars',FUSED_VarComp([('rotor_diameter', 0.0),
								    																 ('rotor_bending_moment_x', 0.0),
								    																 ('rotor_bending_moment_y', 0.0),
								    																 ('rotor_bending_moment_z', 0.0),
								    																 ('rotor_thrust', 0.0),
								    																 ('rotor_force_y',0.0),
								    																 ('rotor_force_z', 0.0),
								    																 ('rotor_torque', 0.0),
								    																 ('rotor_mass', 0.0),								    																 
								    																 ]),['*'])

    # Add common inputs for drivetrain
    FUSED_add(drive_group, 'drivevars',FUSED_VarComp([('machine_rating', 0.0),
								    																 ('gear_ratio', 0.0),
								    																 ('flange_length', 0.0),
								    																 ('overhang', 0.0),
								    																 ('distance_hub2mb', 0.0),
								    																 ('gearbox_input_cm', 0.0),
								    																 ('hss_input_length', 0.0),
								    																 ]),['*'])

    # Add common inputs for tower
    FUSED_add(drive_group, 'towervars',FUSED_VarComp([('tower_top_diameter', 0.0)]),['*'])

    # select components
    FUSED_add(drive_group, 'lowSpeedShaft', FUSED_Component(FUSED_LowSpeedShaft4pt(mb1Type, mb2Type, IEC_Class)), ['*'])
    FUSED_add(drive_group, 'mainBearing', FUSED_Component(FUSED_MainBearing('main')), ['lss_design_torque','rotor_diameter']) #explicit connections for bearings
    FUSED_add(drive_group, 'secondBearing', FUSED_Component(FUSED_MainBearing('second')), ['lss_design_torque','rotor_diameter']) #explicit connections for bearings
    FUSED_add(drive_group, 'gearbox', FUSED_Component(FUSED_Gearbox(gear_configuration, shaft_factor)), ['*'])
    FUSED_add(drive_group, 'highSpeedSide', FUSED_Component(FUSED_HighSpeedSide()), ['*'])
    FUSED_add(drive_group, 'generator', FUSED_Component(FUSED_Generator(drivetrain_design)), ['*'])
    FUSED_add(drive_group, 'bedplate', FUSED_Component(FUSED_Bedplate(uptower_transformer)), ['*'])
    FUSED_add(drive_group, 'transformer', FUSED_Component(FUSED_Transformer(uptower_transformer)), ['*'])
    FUSED_add(drive_group, 'rna', FUSED_Component(FUSED_RNASystemAdder()), ['*'])
    FUSED_add(drive_group, 'above_yaw_massAdder', FUSED_Component(FUSED_AboveYawMassAdder(crane)), ['*'])
    FUSED_add(drive_group, 'yawSystem', FUSED_Component(FUSED_YawSystem(yaw_motors_number)), ['*'])
    FUSED_add(drive_group, 'nacelleSystem', FUSED_Component(FUSED_NacelleSystemAdder()), ['*'])

    # Connect components where explicit connections needed (for main bearings)
    FUSED_connect(drive_group, 'lss_mb1_mass', ['mainBearing.bearing_mass'])
    FUSED_connect(drive_group, 'lss_diameter1', ['mainBearing.lss_diameter', 'lss_diameter'])
    FUSED_connect(drive_group, 'lss_mb1_cm', ['mainBearing.lss_mb_cm'])
    FUSED_connect(drive_group, 'mainBearing.mb_mass', ['mb1_mass'])
    FUSED_connect(drive_group, 'mainBearing.mb_cm', ['mb1_cm'])
    FUSED_connect(drive_group, 'mainBearing.mb_I', ['mb1_I'])

    FUSED_connect(drive_group, 'lss_mb2_mass', ['secondBearing.bearing_mass'])
    FUSED_connect(drive_group, 'lss_diameter2', ['secondBearing.lss_diameter'])
    FUSED_connect(drive_group, 'lss_mb2_cm', ['secondBearing.lss_mb_cm'])
    FUSED_connect(drive_group, 'secondBearing.mb_mass', ['mb2_mass'])
    FUSED_connect(drive_group, 'secondBearing.mb_cm', ['mb2_cm'])
    FUSED_connect(drive_group, 'secondBearing.mb_I', ['mb2_I'])

#------------------------------------------------------------------
# examples

def nacelle_example_5MW_baseline_3pt():

    # NREL 5 MW Drivetrain variables
    # geared 3-stage Gearbox with induction generator machine
    drivetrain_design='geared'
    gear_configuration='eep'  # epicyclic-epicyclic-parallel
    mb1Type='SRB'
    IEC_Class='B'
    shaft_factor='normal'
    uptower_transformer=True
    crane=True  # onboard crane present
    yaw_motors_number = 0 # default value - will be internally calculated

    nace=FUSED_Group()
    Drive3pt(nace, mb1Type, IEC_Class, gear_configuration, shaft_factor, drivetrain_design, uptower_transformer, yaw_motors_number, crane)

    prob=FUSED_Problem(nace)
    FUSED_setup(prob)

    # Rotor and load inputs
    prob['rotor_diameter']=126.0  # m
    prob['rotor_speed']=12.1  # rpm m/s
    prob['machine_rating']=5000.0
    prob['drivetrain_efficiency']=0.95
    prob['rotor_torque']=1.5 * (prob['machine_rating'] * 1000 / \
                             prob['drivetrain_efficiency']) / (prob['rotor_speed'] * (np.pi / 30))
    prob['rotor_thrust']=599610.0  # N
    prob['rotor_mass']=0.0  # accounted for in F_z # kg
    prob['rotor_speed']=12.1  # rpm
    prob['rotor_bending_moment_x']=330770.0  # Nm
    prob['rotor_bending_moment_y']=-16665000.0  # Nm
    prob['rotor_bending_moment_z']=2896300.0  # Nm
    prob['rotor_thrust']=599610.0  # N
    prob['rotor_force_y']=186780.0  # N
    prob['rotor_force_z']=-842710.0  # N

    # Drivetrain inputs
    prob['machine_rating']=5000.0  # kW
    prob['gear_ratio']=96.76  # 97:1 as listed in the 5 MW reference document
    prob['shaft_angle']=5.0*np.pi / 180.0  # rad
    prob['shaft_ratio']=0.10
    prob['planet_numbers']=[3, 3, 1]
    prob['shrink_disc_mass']=333.3 * prob['machine_rating'] / 1000.0  # estimated
    prob['carrier_mass']=8000.0  # estimated
    prob['flange_length']=0.5
    prob['overhang']=5.0
    prob['distance_hub2mb']=1.912  # length from hub center to main bearing, leave zero if unknown
    prob['gearbox_input_cm'] = 0.1
    prob['hss_input_length'] = 1.5

    # Tower inputs
    prob['tower_top_diameter']=3.78  # m

    FUSED_run(prob)

    print('----- NREL 5 MW Turbine - 3 Point Suspension -----')
    FUSED_print(nace)

def nacelle_example_5MW_baseline_4pt():

    # NREL 5 MW Drivetrain variables
    # geared 3-stage Gearbox with induction generator machine
    drivetrain_design='geared'
    gear_configuration='eep'  # epicyclic-epicyclic-parallel
    mb1Type='CARB'
    mb2Type='SRB'
    IEC_Class='B'
    shaft_factor='normal'
    uptower_transformer=True
    crane=True  # onboard crane present
    yaw_motors_number = 0 # default value - will be internally calculated

    nace=FUSED_Group()
    Drive4pt(nace, mb1Type, mb2Type, IEC_Class, gear_configuration, shaft_factor, drivetrain_design, uptower_transformer, yaw_motors_number, crane)

    prob=FUSED_Problem(nace)
    FUSED_setup(prob)

    # Rotor and load inputs
    prob['rotor_diameter']=126.0  # m
    prob['rotor_speed']=12.1  # rpm m/s
    prob['machine_rating']=5000.0
    prob['drivetrain_efficiency']=0.95
    prob['rotor_torque']=1.5 * (prob['machine_rating'] * 1000 / \
                             prob['drivetrain_efficiency']) / (prob['rotor_speed'] * (np.pi / 30))
    prob['rotor_thrust']=599610.0  # N
    prob['rotor_mass']=0.0  # accounted for in F_z # kg
    prob['rotor_speed']=12.1  # rpm
    prob['rotor_bending_moment_x']=330770.0  # Nm
    prob['rotor_bending_moment_y']=-16665000.0  # Nm
    prob['rotor_bending_moment_z']=2896300.0  # Nm
    prob['rotor_thrust']=599610.0  # N
    prob['rotor_force_y']=186780.0  # N
    prob['rotor_force_z']=-842710.0  # N

    # Drivetrain inputs
    prob['machine_rating']=5000.0  # kW
    prob['gear_ratio']=96.76  # 97:1 as listed in the 5 MW reference document
    prob['shaft_angle']=5.0*np.pi / 180.0  # rad
    prob['shaft_ratio']=0.10
    prob['planet_numbers']=[3, 3, 1]
    prob['shrink_disc_mass']=333.3 * prob['machine_rating'] / 1000.0  # estimated
    prob['carrier_mass']=8000.0  # estimated
    prob['flange_length']=0.5
    prob['overhang']=5.0
    prob['distance_hub2mb']=1.912  # length from hub center to main bearing, leave zero if unknown
    prob['gearbox_input_cm'] = 0.1
    prob['hss_input_length'] = 1.5

    # Tower inputs
    prob['tower_top_diameter']=3.78  # m

    FUSED_run(prob)

    print('----- NREL 5 MW Turbine - 4 Point Suspension -----')
    FUSED_print(nace)


'''
#Need to update for new structure of drivetrain
def nacelle_example_1p5MW_3pt():

    # test of module for turbine data set

    # 1.5 MW Rotor Variables
    print('----- NREL 1p5MW  Drivetrain - 3 Point Suspension-----')
    nace=Group()
    Drive3pt(nace)
    prob=Problem(nace)
    prob.setup()
    nace.rotor_diameter=77  # m
    nace.rotor_speed=16.18  # rpm
    nace.drivetrain_efficiency=0.95
    nace.machine_rating=1500
    nace.rotor_torque=1.5 * (nace.machine_rating * 1000 / nace.drivetrain_efficiency) / \
                             (nace.rotor_speed * (pi / 30)
                              )  # 6.35e6 #4365248.74 # Nm
    nace.rotor_thrust=2.6204e5
    nace.rotor_mass=0.0
    nace.rotor_speed=16.18  # rpm
    nace.rotor_bending_moment=2.7795e6
    nace.rotor_bending_moment_x=8.4389e5
    nace.rotor_bending_moment_y=-2.6758e6
    nace.rotor_bending_moment_z=7.5222e2
    nace.rotor_thrust=2.6204e5
    nace.rotor_force_y=2.8026e4
    nace.rotor_force_z=-3.4763e5


    # 1p5MW  Drivetrain variables
    # geared 3-stage Gearbox with induction generator machine
    nace.drivetrain_design='geared'
    nace.machine_rating=1500.0  # kW
    nace.gear_ratio=78
    nace.gear_configuration='epp'  # epicyclic-parallel-parallel
    nace.crane=False  # onboard crane not present
    nace.shaft_angle=5.0  # deg
    nace.shaft_ratio=0.10
    nace.Np=[3, 1, 1]
    nace.shaft_type='normal'
    nace.uptower_transformer=False  # True
    nace.shrink_disc_mass=333.3 * nace.machine_rating / 1000.0  # estimated
    nace.carrier_mass=2000.0  # estimated
    nace.mb1Type='SRB'
    nace.mb2Type='SRB'
    nace.flange_length=0.285  # m
    nace.overhang=3.3
    nace.distance_hub2mb=1.535  # length from hub center to main bearing, leave zero if unknown

    # 0 if no fatigue check, 1 if parameterized fatigue check, 2 if known
    # loads inputs
    nace.check_fatigue=0

    # variables if check_fatigue = 1:
    nace.blade_number=3
    nace.cut_in=3.5  # cut-in m/s
    nace.cut_out=20.  # cut-out m/s
    nace.Vrated=11.5  # rated windspeed m/s
    nace.weibull_k=2.2  # windepeed distribution shape parameter
    nace.weibull_A=9.  # windspeed distribution scale parameter
    nace.T_life=20.  # design life in years
    nace.IEC_Class_Letter='B'

    # variables if check_fatigue =2:
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
    nace.tower_top_diameter=2.3  # m

    prob.run()

    sys_print(nace)

def nacelle_example_1p5MW_4pt():

    # test of module for turbine data set

    print('----- NREL 1p5MW  Drivetrain - 4 Point Suspension-----')
    nace=Group()
    Drive4pt(nace)
    prob=Problem(nace)
    prob.setup()
    nace.rotor_diameter=77  # m
    nace.rotor_speed=16.18  # rpm
    nace.drivetrain_efficiency=0.95
    nace.machine_rating=1500
    nace.rotor_torque=1.5 * (nace.machine_rating * 1000 / nace.drivetrain_efficiency) / \
                             (nace.rotor_speed * (pi / 30)
                              )  # 6.35e6 #4365248.74 # Nm
    nace.rotor_thrust=2.6204e5
    nace.rotor_mass=0.0
    nace.rotor_speed=16.18  # rpm
    nace.rotor_bending_moment=2.7795e6
    nace.rotor_bending_moment_x=8.4389e5
    nace.rotor_bending_moment_y=-2.6758e6
    nace.rotor_bending_moment_z=7.5222e2
    nace.rotor_thrust=2.6204e5
    nace.rotor_force_y=2.8026e4
    nace.rotor_force_z=-3.4763e5

    # 1p5MW  Drivetrain variables
    # geared 3-stage Gearbox with induction generator machine
    nace.drivetrain_design='geared'
    nace.machine_rating=1500.0  # kW
    nace.gear_ratio=78
    nace.gear_configuration='epp'  # epicyclic-parallel-parallel
    nace.crane=False  # True # onboard crane present
    nace.shaft_angle=5.0  # deg
    nace.shaft_ratio=0.10
    nace.Np=[3, 1, 1]
    nace.shaft_type='normal'
    nace.uptower_transformer=False  # True
    nace.shrink_disc_mass=333.3 * nace.machine_rating / 1000.0  # estimated
    nace.carrier_mass=2000.0  # estimated
    nace.mb1Type='CARB'
    nace.mb2Type='SRB'
    nace.flange_length=0.285  # m
    nace.overhang=4
    nace.distance_hub2mb=1.3  # length from hub center to main bearing, leave zero if unknown
    nace.gearbox_cm=0.0

    # 0 if no fatigue check, 1 if parameterized fatigue check, 2 if known
    # loads inputs
    nace.check_fatigue=0

    # variables if check_fatigue = 1:
    nace.blade_number=3
    nace.cut_in=3.5  # cut-in m/s
    nace.cut_out=20.  # cut-out m/s
    nace.Vrated=11.5  # rated windspeed m/s
    nace.weibull_k=2.2  # windepeed distribution shape parameter
    nace.weibull_A=9.  # windspeed distribution scale parameter
    nace.T_life=20.  # design life in years
    nace.IEC_Class_Letter='B'

    # variables if check_fatigue =2:
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
    nace.tower_top_diameter=2.3  # m

    prob.run()

    # cm_print(nace)
    sys_print(nace)

def nacelle_example_p75_3pt():

    # test of module for turbine data set
    print('----- NREL 750kW Design - 3 Point Suspension----')
    # 0.75MW Rotor Variables
    nace=Group()
    Drive3pt(nace)
    prob=Problem(nace)
    prob.setup()
    nace.rotor_diameter=48.2  # m
    nace.rotor_speed=22.0  # rpm m/s
    nace.drivetrain_efficiency=0.95
    nace.machine_rating=750
    nace.rotor_torque=1.5 * (nace.machine_rating * 1000 / nace.drivetrain_efficiency) / \
                             (nace.rotor_speed * (pi / 30)
                              )  # 6.35e6 #4365248.74 # Nm

    nace.rotor_thrust=143000.0  # N
    nace.rotor_mass=0.0  # kg
    nace.rotor_speed=22.0  # rpm
    nace.rotor_bending_moment=495.6e3
    nace.rotor_bending_moment_x=401.0e3
    nace.rotor_bending_moment_y=495.6e3
    nace.rotor_bending_moment_z=-443.0e3
    nace.rotor_thrust=143000.0
    nace.rotor_force_y=-12600.0
    nace.rotor_force_z=-142.0e3

    # NREL 750 kW Drivetrain variables
    # geared 3-stage Gearbox with induction generator machine
    nace.drivetrain_design='geared'
    nace.machine_rating=750  # kW
    nace.gear_ratio=81.491
    nace.gear_configuration='epp'  # epicyclic-parallel-parallel
    nace.crane=False  # True if onboard crane present
    nace.shaft_angle=5.0  # deg
    nace.shaft_length=2.1  # m
    nace.shaft_ratio=0.10
    nace.Np=[3, 1, 1]
    nace.shaft_type='normal'
    nace.uptower_transformer=False
    nace.shrink_disc_mass=333.3 * nace.machine_rating / 1000.0  # estimated
    nace.carrier_mass=250.  # estimated
    nace.mb1Type='SRB'
    nace.mb2Type='TRB2'
    nace.flange_length=0.285  # m
    nace.overhang=2.26
    nace.distance_hub2mb=1.22  # length from hub center to main bearing, leave zero if unknown
    nace.gearbox_cm=0.8
    nace.blade_root_diameter=1.6

    # 0 if no fatigue check, 1 if parameterized fatigue check, 2 if known
    # loads inputs
    nace.check_fatigue=0

    # variables if check_fatigue = 1:
    nace.blade_number=3
    nace.cut_in=3.  # cut-in m/s
    nace.cut_out=25.  # cut-out m/s
    nace.Vrated=16.  # rated windspeed m/s
    nace.weibull_k=2.2  # windepeed distribution shape parameter
    nace.weibull_A=9.  # windspeed distribution scale parameter
    nace.T_life=20.  # design life in years
    nace.IEC_Class_Letter='A'


    # variables if check_fatigue =2:
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
    nace.tower_top_diameter=2.21  # m

    prob.run()
    # cm_print(nace)
    sys_print(nace)

def nacelle_example_p75_4pt():

    # test of module for turbine data set
    print('----- NREL 750kW Design - 4 Point Suspension----')
    # 0.75MW Rotor Variables
    nace=Group()
    Drive4pt(nace)
    prob=Problem(nace)
    prob.setup()
    nace.rotor_diameter=48.2  # m
    nace.rotor_speed=22.0  # rpm
    nace.drivetrain_efficiency=0.95
    nace.machine_rating=750
    nace.rotor_torque=1.5 * (nace.machine_rating * 1000 / nace.drivetrain_efficiency) / \
                             (nace.rotor_speed * (pi / 30)
                              )  # 6.35e6 #4365248.74 # Nm
    #nace.rotor_torque = 6.37e6 #
    nace.rotor_thrust=143000.0
    nace.rotor_mass=0.0
    nace.rotor_speed=22.0  # rpm
    nace.rotor_bending_moment=459.6e3
    nace.rotor_bending_moment_x=401.0e3
    nace.rotor_bending_moment_y=459.6e3
    nace.rotor_bending_moment_z=-443.0e3
    nace.rotor_thrust=143000.0
    nace.rotor_force_y=-12600.0
    nace.rotor_force_z=-142.0e3

    # NREL 750 kW Drivetrain variables
    # geared 3-stage Gearbox with induction generator machine
    nace.drivetrain_design='geared'
    nace.machine_rating=750  # kW
    nace.gear_ratio=81.491  # as listed in the 5 MW reference document
    nace.gear_configuration='epp'  # epicyclic-parallel-parallel
    nace.crane=False  # True # onboard crane present
    nace.shaft_angle=5.0  # deg
    nace.shaft_length=2.1  # m
    nace.shaft_ratio=0.10
    nace.Np=[3, 1, 1]
    nace.shaft_type='normal'
    nace.uptower_transformer=False  # True
    nace.shrink_disc_mass=333.3 * nace.machine_rating / 1000.0  # estimated
    nace.carrier_mass=1000.0  # estimated
    nace.mb1Type='SRB'
    nace.mb2Type='TRB2'
    nace.flange_length=0.338  # m
    nace.overhang=2.26
    nace.distance_hub2mb=1.22  # 0.007835*rotor_diameter+0.9642 length from hub center to main bearing, leave zero if unknown
    nace.gearbox_cm=0.90

    # 0 if no fatigue check, 1 if parameterized fatigue check, 2 if known
    # loads inputs
    nace.check_fatigue=0

    # variables if check_fatigue = 1:
    nace.blade_number=3
    nace.cut_in=3.  # cut-in m/s
    nace.cut_out=25.  # cut-out m/s
    nace.Vrated=16.  # rated windspeed m/s
    nace.weibull_k=2.2  # windepeed distribution shape parameter
    nace.weibull_A=9.  # windspeed distribution scale parameter
    nace.T_life=20.  # design life in years
    nace.IEC_Class_Letter='A'

    # variables if check_fatigue =2:
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
    nace.tower_top_diameter=2.21  # m

    prob.run()

    sys_print(nace)
'''

if __name__ == '__main__':
    ''' Main runs through tests of several drivetrain configurations with known component masses and dimensions '''

    nacelle_example_5MW_baseline_3pt()

    nacelle_example_5MW_baseline_4pt()
    
    '''
    nacelle_example_1p5MW_3pt()

    nacelle_example_1p5MW_4pt()

    nacelle_example_p75_3pt()

    nacelle_example_p75_4pt()'''