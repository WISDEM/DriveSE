"""
DriveSE.py

Created by Yi Guo, Taylor Parsons and Ryan King 2014.
Copyright (c) NREL. All rights reserved.
"""

import numpy as np

from drivese.fused_drivese import Gearbox_OM, MainBearing_OM, Bedplate_OM, YawSystem_OM, LowSpeedShaft3pt_OM, \
    LowSpeedShaft4pt_OM, Transformer_OM, HighSpeedSide_OM, Generator_OM, NacelleSystemAdder_OM, AboveYawMassAdder_OM, RNASystemAdder_OM
from drivese.fused_hubse import Hub_OM, PitchSystem_OM, Spinner_OM, Hub_System_Adder_OM
from openmdao.api import Group, Component, IndepVarComp, Problem

class Drive3pt(Group):

    def __init__(self, mb1Type, IEC_Class, gear_configuration, shaft_factor, drivetrain_design, uptower_transformer, yaw_motors_number, crane, blade_number):
        super(Drive3pt, self).__init__()


        # Add common inputs for rotor
        self.add('rotor_diameter', IndepVarComp('rotor_diameter', 0.0), promotes=['*'])
        self.add('rotor_bending_moment_x', IndepVarComp('rotor_bending_moment_x', 0.0), promotes=['*'])
        self.add('rotor_bending_moment_y', IndepVarComp('rotor_bending_moment_y', 0.0), promotes=['*'])
        self.add('rotor_bending_moment_z', IndepVarComp('rotor_bending_moment_z', 0.0), promotes=['*'])
        self.add('rotor_thrust', IndepVarComp('rotor_thrust', 0.0), promotes=['*'])
        self.add('rotor_force_y', IndepVarComp('rotor_force_y', 0.0), promotes=['*'])
        self.add('rotor_force_z', IndepVarComp('rotor_force_z', 0.0), promotes=['*'])
        self.add('rotor_torque', IndepVarComp('rotor_torque', 0.0), promotes=['*'])
        self.add('rotor_mass', IndepVarComp('rotor_mass', 0.0), promotes=['*'])

        # Add common inputs for drivetrain
        self.add('gear_ratio', IndepVarComp('gear_ratio', 0.0), promotes=['*'])
        self.add('shaft_angle', IndepVarComp('shaft_angle', 0.0), promotes=['*'])
        self.add('shaft_ratio', IndepVarComp('shaft_ratio', 0.0), promotes=['*'])
        self.add('shrink_disc_mass', IndepVarComp('shrink_disc_mass', 0.0), promotes=['*'])
        self.add('carrier_mass', IndepVarComp('carrier_mass', 0.0), promotes=['*'])
        self.add('flange_length', IndepVarComp('flange_length', 0.0), promotes=['*'])
        self.add('overhang', IndepVarComp('overhang', 0.0), promotes=['*'])
        self.add('distance_hub2mb', IndepVarComp('distance_hub2mb', 0.0), promotes=['*'])
        self.add('gearbox_input_cm', IndepVarComp('gearbox_input_cm', 0.0), promotes=['*'])
        self.add('hss_input_length', IndepVarComp('hss_input_length', 0.0), promotes=['*'])

        # Add common inputs for tower
        self.add('towervars',IndepVarComp([('tower_top_diameter', 0.0)]), promotes=['*'])

        # hub
        #self.add('hub', Hub_OM(blade_number), promotes=['*'])
        #self.add('pitchSystem', PitchSystem_OM(blade_number), promotes=['*'])
        #self.add('spinner', Spinner_OM(), promotes=['*'])
        #self.add('adder', Hub_System_Adder_OM(), promotes=['*'])

        # Create 3 pt drivetrain group
        self.add('lowSpeedShaft', LowSpeedShaft3pt_OM(mb1Type, IEC_Class), promotes=['*'])
        self.add('mainBearing', MainBearing_OM('main'), promotes=['lss_design_torque','rotor_diameter']) #need to make explicit connections for main bearing
        self.add('gearbox', Gearbox_OM(gear_configuration, shaft_factor), promotes=['*'])
        self.add('highSpeedSide', HighSpeedSide_OM(), promotes=['*'])
        self.add('generator', Generator_OM(drivetrain_design), promotes=['*'])
        self.add('bedplate', Bedplate_OM(uptower_transformer), promotes=['*'])
        self.add('transformer', Transformer_OM(uptower_transformer), promotes=['*'])
        self.add('rna', RNASystemAdder_OM(), promotes=['*'])
        self.add('above_yaw_massAdder', AboveYawMassAdder_OM(crane), promotes=['*'])
        self.add('yawSystem', YawSystem_OM(yaw_motors_number), promotes=['*'])
        self.add('nacelleSystem', NacelleSystemAdder_OM(), promotes=['*'])

        # Connect components where explicit connections needed (for main bearing)
        self.connect('lss_mb1_mass', ['mainBearing.bearing_mass'])
        self.connect('lss_diameter1', ['mainBearing.lss_diameter', 'lss_diameter'])
        self.connect('lss_mb1_cm', ['mainBearing.lss_mb_cm'])
        self.connect('mainBearing.mb_mass', ['mb1_mass'])
        self.connect('mainBearing.mb_cm', ['mb1_cm'])
        self.connect('mainBearing.mb_I', ['mb1_I'])

#------------------------------------------------------------------
class Drive4pt(Group):

    def __init__(self, mb1Type, mb2Type, IEC_Class, gear_configuration, shaft_factor, drivetrain_design, uptower_transformer, yaw_motors_number, crane, blade_number):
        super(Drive4pt, self).__init__()

        # Add common inputs for rotor
        self.add('rotor_diameter', IndepVarComp('rotor_diameter', 0.0), promotes=['*'])
        self.add('rotor_bending_moment_x', IndepVarComp('rotor_bending_moment_x', 0.0), promotes=['*'])
        self.add('rotor_bending_moment_y', IndepVarComp('rotor_bending_moment_y', 0.0), promotes=['*'])
        self.add('rotor_bending_moment_z', IndepVarComp('rotor_bending_moment_z', 0.0), promotes=['*'])
        self.add('rotor_thrust', IndepVarComp('rotor_thrust', 0.0), promotes=['*'])
        self.add('rotor_force_y', IndepVarComp('rotor_force_y', 0.0), promotes=['*'])
        self.add('rotor_force_z', IndepVarComp('rotor_force_z', 0.0), promotes=['*'])
        self.add('rotor_torque', IndepVarComp('rotor_torque', 0.0), promotes=['*'])
        self.add('rotor_mass', IndepVarComp('rotor_mass', 0.0), promotes=['*'])

        # Add common inputs for drivetrain
        self.add('machine_rating',IndepVarComp('machine_rating', 0.0), promotes=['*'])
        self.add('gear_ratio', IndepVarComp('gear_ratio', 0.0), promotes=['*'])
        self.add('flange_length', IndepVarComp('flange_length', 0.0), promotes=['*'])
        self.add('overhang', IndepVarComp('overhang', 0.0), promotes=['*'])
        self.add('distance_hub2mb', IndepVarComp('distance_hub2mb', 0.0), promotes=['*'])
        self.add('gearbox_input_cm', IndepVarComp('gearbox_input_cm', 0.0), promotes=['*'])
        self.add('hss_input_length', IndepVarComp('hss_input_length', 0.0), promotes=['*'])

        # Add common inputs for tower
        self.add('towervars',IndepVarComp([('tower_top_diameter', 0.0)]),['*'])

        # hub
        self.add('hub', Hub_OM(blade_number), promotes=['*'])
        self.add('pitchSystem', PitchSystem_OM(blade_number), promotes=['*'])
        self.add('spinner', Spinner_OM(), promotes=['*'])
        self.add('adder', Hub_System_Adder_OM(), promotes=['*'])

        # select components
        self.add('lowSpeedShaft', LowSpeedShaft4pt_OM(mb1Type, mb2Type, IEC_Class), promotes=['*'])
        self.add('mainBearing', MainBearing_OM('main'), promotes=['lss_design_torque','rotor_diameter']) #explicit connections for bearings
        self.add('secondBearing', MainBearing_OM('second'), promotes=['lss_design_torque','rotor_diameter']) #explicit connections for bearings
        self.add('gearbox', Gearbox_OM(gear_configuration, shaft_factor), promotes=['*'])
        self.add('highSpeedSide', HighSpeedSide_OM(), promotes=['*'])
        self.add('generator', Generator_OM(drivetrain_design), promotes=['*'])
        self.add('bedplate', Bedplate_OM(uptower_transformer), promotes=['*'])
        self.add('transformer', Transformer_OM(uptower_transformer), promotes=['*'])
        self.add('rna', RNASystemAdder_OM(), promotes=['*'])
        self.add('above_yaw_massAdder', AboveYawMassAdder_OM(crane), promotes=['*'])
        self.add('yawSystem', YawSystem_OM(yaw_motors_number), promotes=['*'])
        self.add('nacelleSystem', NacelleSystemAdder_OM(), promotes=['*'])

        # Connect components where explicit connections needed (for main bearings)
        self.connect('lss_mb1_mass', ['mainBearing.bearing_mass'])
        self.connect('lss_diameter1', ['mainBearing.lss_diameter', 'lss_diameter'])
        self.connect('lss_mb1_cm', ['mainBearing.lss_mb_cm'])
        self.connect('mainBearing.mb_mass', ['mb1_mass'])
        self.connect('mainBearing.mb_cm', ['mb1_cm'])
        self.connect('mainBearing.mb_I', ['mb1_I'])

        self.connect('lss_mb2_mass', ['secondBearing.bearing_mass'])
        self.connect('lss_diameter2', ['secondBearing.lss_diameter'])
        self.connect('lss_mb2_cm', ['secondBearing.lss_mb_cm'])
        self.connect('secondBearing.mb_mass', ['mb2_mass'])
        self.connect('secondBearing.mb_cm', ['mb2_cm'])
        self.connect('secondBearing.mb_I', ['mb2_I'])

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
    blade_number = 3

    prob=Problem(root=Drive3pt(mb1Type, IEC_Class, gear_configuration, shaft_factor, drivetrain_design, uptower_transformer, yaw_motors_number, crane, blade_number))
    prob.setup()

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

    prob.run()

    print('----- NREL 5 MW Turbine - 3 Point Suspension -----')
    print(prob.root.unknowns.dump())

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
    blade_number = 3


    prob=Problem(root=Drive4pt(mb1Type, mb2Type, IEC_Class, gear_configuration, shaft_factor, drivetrain_design, uptower_transformer, yaw_motors_number, crane, blade_number))
    prob.setup()

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

    prob.run()

    print('----- NREL 5 MW Turbine - 4 Point Suspension -----')
    print(prob.root.unknowns.dump())


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
