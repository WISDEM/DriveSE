
import numpy as np

# FUSED wrapper
from fusedwind.fused_openmdao import FUSED_Component, FUSED_Group, FUSED_add, FUSED_connect, FUSED_print, \
                                     FUSED_Problem, FUSED_setup, FUSED_run, FUSED_VarComp
                                     
from fused_hubse import FUSED_Hub, FUSED_PitchSystem, FUSED_Spinner, FUSED_Hub_System_Adder

#-------------------------------------------------------------------------

def HubSE(hubgroup, blade_number):
    '''
       HubSE class
          The HubSE class is used to represent the hub system of a wind turbine. 
          HubSE integrates the hub, pitch system and spinner / nose cone components for the hub system.
    '''

    # Add common inputs for rotor
    FUSED_add(hubgroup, 'rotorvars',FUSED_VarComp([('rotor_diameter', 0.0),
                                                     ('rotor_bending_moment_y', 0.0),                                                  
                                                     ]),['*'])

    FUSED_add(hubgroup, 'hub', FUSED_Component(FUSED_Hub(blade_number)), ['*'])
    FUSED_add(hubgroup, 'pitchSystem', FUSED_Component(FUSED_PitchSystem(blade_number)), ['*'])
    FUSED_add(hubgroup, 'spinner', FUSED_Component(FUSED_Spinner()), ['*'])
    FUSED_add(hubgroup,'adder', FUSED_Component(FUSED_Hub_System_Adder()), ['*'])

#-------------------------------------------------------------------------
# Examples based on reference turbines including the NREL 5 MW, WindPACT 1.5 MW and the GRC 750 kW system.

def example_5MW_4pt():

    # simple test of module

    blade_number = 3

    hubsys=FUSED_Group()
    HubSE(hubsys, blade_number)

    prob=FUSED_Problem(hubsys)
    FUSED_setup(prob)

    prob['rotor_diameter'] = 126.0  # m
    prob['blade_root_diameter'] = 3.542
    prob['machine_rating'] = 5000.0
    prob['blade_mass'] = 17740.0  # kg
    prob['shaft_angle'] = np.radians(5)
    #prob['distance_hub2mb'] = 0.0
    #prob['MB1_location'] = [0.0, 0.0, 0.0]

    AirDensity = 1.225  # kg/(m^3)
    Solidity = 0.0517
    RatedWindSpeed = 11.05  # m/s
    prob['rotor_bending_moment_y'] = (3.06 * np.pi / 8) * AirDensity * (
        RatedWindSpeed ** 2) * (Solidity * (prob['rotor_diameter'] ** 3)) / blade_number

    FUSED_run(prob)

    print("NREL 5 MW turbine test")
    FUSED_print(hubsys)

'''
# TODO: update other examples for the hub system
def example_1p5MW_4pt():

    # WindPACT 1.5 MW turbine
    prob = Problem()
    prob.root = HubSE()
    prob.setup()

    prob['blade_mass'] = 4470.0
    prob['rotor_diameter'] = 70.0
    prob['blade_number'] = 3
    prob['hub_diameter'] = 3.0
    prob['machine_rating'] = 1500.0
    prob['blade_root_diameter'] = 2.0  # TODO - find actual number

    AirDensity = 1.225
    Solidity = 0.065
    RatedWindSpeed = 12.12
    prob['rotor_bending_moment'] = (3.06 * pi / 8) * AirDensity * (
        RatedWindSpeed ** 2) * (Solidity * (prob['rotor_diameter'] ** 3)) / prob['blade_number']

    prob.run()

    print("WindPACT 1.5 MW turbine test")
    print("Hub FUSED_Objects")
    print('  hub         {0:8.1f} kg'.format(prob['hub_mass']))  # 31644.47
    print('  pitch mech  {0:8.1f} kg'.format(prob['pitch_system_mass']))  # 17003.98
    print('  nose cone   {0:8.1f} kg'.format(prob['spinner_mass']))  # 1810.50
    # 50458.95
    print('Hub system total {0:8.1f} kg'.format(prob['hub_system_mass']))
    print('    cm {0:6.2f} {1:6.2f} {2:6.2f}'.format(prob['hub_system_cm'][0], prob['hub_system_cm'][1], prob['hub_system_cm'][2]))
    print('    I {0:6.1f} {1:6.1f} {2:6.1f}'.format(prob['hub_system_I'][0], prob['hub_system_I'][1], prob['hub_system_I'][2]))
    print()

def example_750kW_4pt():

    # GRC 750 kW turbine
    prob = Problem()
    prob.root = HubSE()
    prob.setup()

    prob['blade_mass'] = 3400.0
    prob['rotor_diameter'] = 48.2
    prob['blade_number'] = 3
    prob['hub_diameter'] = 3.0
    prob['machine_rating'] = 750.0
    prob['blade_root_diameter'] = 1.0  # TODO - find actual number

    AirDensity = 1.225
    Solidity = 0.07  # uknown value
    RatedWindSpeed = 16.0
    prob['rotor_bending_moment'] = (3.06 * pi / 8) * AirDensity * (
        RatedWindSpeed ** 2) * (Solidity * (prob['rotor_diameter'] ** 3)) / prob['blade_number']

    prob.run()

    print("windpact 750 kW turbine test")
    print("Hub FUSED_Objects")
    print('  hub         {0:8.1f} kg'.format(prob['hub_mass']))  # 31644.47
    print('  pitch mech  {0:8.1f} kg'.format(prob['pitch_system_mass']))  # 17003.98
    print('  nose cone   {0:8.1f} kg'.format(prob['spinner_mass']))  # 1810.50
    # 50458.95
    print('Hub system total {0:8.1f} kg'.format(prob['hub_system_mass']))
    print('    cm {0:6.2f} {1:6.2f} {2:6.2f}'.format(prob['hub_system_cm'][0], prob['hub_system_cm'][1], prob['hub_system_cm'][2]))
    print('    I {0:6.1f} {1:6.1f} {2:6.1f}'.format(prob['hub_system_I'][0], prob['hub_system_I'][1], prob['hub_system_I'][2]))
    print()

'''

# Main code to run hub system examples
if __name__ == "__main__":

    example_5MW_4pt()