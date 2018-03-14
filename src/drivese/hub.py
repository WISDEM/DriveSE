"""
hubSE.py

Created by Katherine Dykes 2012.
Copyright (c) NREL. All rights reserved.
"""

from openmdao.api import Group, Component, Problem, IndepVarComp
import numpy as np
from math import pi, cos, sqrt, radians, sin, exp, log10, log, floor, ceil

#from fusedwind.interface import implement_base, base
from drivese.drivese_utils import get_L_rb

# Hub Base Group


#@base
class HubBase(Component):
    def __init__(self):
        super(Hub_Base, self).__init__()

        # variables
        self.add_param('blade_mass', val=0.0, units='kg', desc='mass of one blade')
        self.add_param('rotor_bending_moment', val=0.0, units='N*m', desc='flapwise bending moment at blade root')
        self.add_param('rotor_diameter', val=0.0, units='m', desc='rotor diameter')
        self.add_param('blade_root_diameter', val=0.0, units='m', desc='blade root diameter')

        # parameters
        self.add_param('blade_number', val=3, desc='number of turbine blades', pass_by_obj=True)

        # outputs
        self.add_output('hub_system_mass', val=0.0, units='kg',  desc='overall component mass')
        self.add_output('hub_system_cm', val=np.array([]), desc='center of mass of the hub relative to tower to in yaw-aligned c.s.')
        self.add_output('hub_system_I', val=np.array([]), desc='mass moments of Inertia of hub [Ixx, Iyy, Izz, Ixy, Ixz, Iyz] around its center of mass in yaw-aligned c.s.')

        self.add_output('hub_mass', val=0.0, units='kg')
        self.add_output('pitch_system_mass', val=0.0, units='kg')
        self.add_output('spinner_mass', val=0.0, units='kg')


class Hub_System_Adder_drive(Component):
    ''' Get_hub_cm class
          The Get_hub_cm class is used to pass the hub cm data to upper level models.
          It contains the general properties for a wind turbine component as well as additional design load and dimentional attributes as listed below.
    '''

    def __init__(self):
        super(Hub_System_Adder_drive, self).__init__()

        # controls what happens if derivatives are missing
        self.missing_deriv_policy = 'assume_zero'

        # variables
        self.add_param('rotor_diameter', val=0.0, units='m', desc='rotor diameter')
        self.add_param('L_rb', val=0.0, units='m', desc='distance between hub center and upwind main bearing')
        self.add_param('shaft_angle', val=0.0, units='deg', desc='shaft angle')
        self.add_param('MB1_location', val=np.zeros(3), units='m', desc='center of mass of main bearing in [x,y,z] for an arbitrary coordinate system')
        self.add_param('hub_mass', val=0.0, units='kg', desc='mass of Hub')
        self.add_param('hub_diameter', val=0.03, units='m', desc='hub diameter')
        self.add_param('hub_thickness', val=0.0, units='m', desc='hub thickness')
        self.add_param('pitch_system_mass', val=0.0, units='kg', desc='mass of Pitch System')
        self.add_param('spinner_mass', val=0.0, units='kg', desc='mass of spinner')

        # outputs
        self.add_output('hub_system_cm', val=np.zeros(3), units='m',  desc='center of mass of the hub relative to tower to in yaw-aligned c.s.')
        self.add_output('hub_system_I', val=np.zeros(6), desc='mass moments of Inertia of hub [Ixx, Iyy, Izz, Ixy, Ixz, Iyz] around its center of mass in yaw-aligned c.s.')
        self.add_output('hub_system_mass', val=0.0,  units='kg', desc='mass of hub system')

    def solve_nonlinear(self, params, unknowns, resids):
        # Unpack variables
        hub_diameter      = params['hub_diameter']
        hub_thickness     = params['hub_thickness']
        
        hub_mass          = params['hub_mass']
        spinner_mass      = params['spinner_mass']
        pitch_system_mass = params['pitch_system_mass']
        
        L_rb = params['L_rb']
        if L_rb <= 0: get_L_rb(params['rotor_diameter'])

        cm = np.array([0.0, 0.0, 0.0])
        cm[0] = params['MB1_location'][0] - L_rb
        cm[1] = 0.0
        cm[2] = params['MB1_location'][2] + L_rb * sin(radians(params['shaft_angle']))
        unknowns['hub_system_cm'] = cm

        unknowns['hub_system_mass'] = hub_mass + pitch_system_mass + spinner_mass

        # add I definitions here
        def getI(m, d, t):
            r = 0.5 * d
            I = ( 0.4*m * (r**5 - (r - t)**5) / (r**3 - (r - t)**3) ) * np.ones(3)
            return np.r_[I, np.zeros(3)]

        hub_I = getI(hub_mass, hub_diameter, hub_thickness)

        pitch_system_I = pitch_system_mass * 0.25*(hub_diameter**2) * np.r_[np.ones(3), np.zeros(3)]

        spinner_diameter = 3.3 if hub_diameter == 0 else hub_diameter
        # 0.055 for 1.5 MW outer diameter of 3.3 - using proportional constant
        spinner_thickness = spinner_diameter * (0.055 / 3.30)

        spinner_I = getI(spinner_mass, spinner_diameter, spinner_thickness)

        # add moments of inertia
        # calculating MOI, at nacelle center of gravity with origin at tower
        # top center / yaw mass center, ignoring masses of non-drivetrain
        # components / auxiliary systems
        # calculate moments around CM
        # sum moments around each components CM
        unknowns['hub_system_I'] = hub_I + pitch_system_I + spinner_I

        # translate to hub system CM using parallel axis theorem- because cm is assumed shared, unneeded
        # for j in (range(0,3)):
        #     if i != j:
        #         I[i] +=  (hub_mass * (hub_cm[j] - hub_system_cm[j]) ** 2) + \
        #                       (pitch_system_mass * (pitch_system_cm[j] - hub_system_cm[j]) ** 2) + \
        #                       (spinner_mass * (spinner_cm[j] - hub_system_cm[j]) ** 2)

# -------------------------------------------------


class Hub_drive(Component):
    ''' Hub class    
          The Hub class is used to represent the hub component of a wind turbine. 
          It contains the general properties for a wind turbine component as well as additional design load and dimentional attributes as listed below.
          It contains an update method to determine the mass, mass properties, and dimensions of the component.            
    '''

    def __init__(self):
        super(Hub_drive, self).__init__()

        # variables
        self.add_param('blade_root_diameter', val=0.0, units='m', desc='blade root diameter')
        self.add_param('machine_rating', val=0.0, units='MW', desc='machine rating of turbine')

        # parameters
        self.add_param('blade_number', val=3, desc='number of turbine blades', pass_by_obj=True)

        # outputs
        self.add_output('hub_diameter', val=0.0, units='m', desc='hub diameter')
        self.add_output('hub_thickness', val=0.0, units='m', desc='hub thickness')
        self.add_output('hub_mass', val=0.0, units='kg', desc='overall component mass')

    def solve_nonlinear(self, params, unknowns, resids):

        blade_root_diameter = params['blade_root_diameter']
        if blade_root_diameter <= 0: 2.659 * params['machine_rating']**0.3254
        blade_root_radius = blade_root_diameter

        # Model hub as a cyclinder with holes for blade root and nacelle
        # flange.
        rCyl = 1.1 * blade_root_radius
        hCyl = 2.8 * blade_root_radius
        castThickness = 0.1*rCyl
        approxCylVol = 2.0 * pi * rCyl * castThickness * hCyl
        bladeRootVol = pi * blade_root_radius**2 * castThickness

        # assume nacelle flange opening is similar to blade root opening
        approxCylNetVol = approxCylVol - (1.0 + params['blade_number']) * bladeRootVol
        castDensity = 7200.0  # kg/m^3
        unknowns['hub_mass'] = approxCylNetVol * castDensity

        # calculate mass properties
        unknowns['hub_diameter'] = 2.0 * rCyl
        unknowns['hub_thickness'] = castThickness


class PitchSystem_drive(Component):
    '''
     PitchSystem class
      The PitchSystem class is used to represent the pitch system of a wind turbine.
      It contains the general properties for a wind turbine component as well as additional design load and dimentional attributes as listed below.
      It contains an update method to determine the mass, mass properties, and dimensions of the component.
    '''

    def __init__(self):
        super(PitchSystem_drive, self).__init__()

        # controls what happens if derivatives are missing
        self.missing_deriv_policy = 'assume_zero'

        # variables
        self.add_param('blade_mass', val=0.0, units='kg', desc='mass of one blade')
        self.add_param('rotor_bending_moment', val=0.0, units='N*m', desc='flapwise bending moment at blade root')

        # parameters
        self.add_param('blade_number', val=3, desc='number of turbine blades', pass_by_obj=True)

        # outputs
        self.add_output('pitch_system_mass', val=0.0, units='kg', desc='overall component mass')

    def solve_nonlinear(self, params, unknowns, resids):

        # Sunderland method for calculating pitch system masses
        # density of pitch system material (kg / m^3) - assuming BS1503-622
        # (same material as LSS)
        pitchmatldensity = 7860.0
        # allowable stress of hub material (N / m^2)
        pitchmatlstress = 3710.0

        # default factor is 1.0 (0.54 for modern designs)
        hubpitchFact = 1.0
        #self.mass =hubpitchFact * (0.22 * self.blade_mass * self.blade_number + 12.6 * self.blade_number * self.rotor_bending_moment * (pitchmatldensity / pitchmatlstress))
        # mass of pitch system based on Sunderland model
        unknowns['pitch_system_mass'] = (hubpitchFact * (0.22 * params['blade_mass'] * params['blade_number'] +
                                            12.6 * params['rotor_bending_moment'] * (pitchmatldensity / pitchmatlstress)) )
        # mass of pitch system based on Sunderland model


#-------------------------------------------------------------------------

class Spinner_drive(Component):
    '''
       Spinner class
          The SpinnerClass is used to represent the spinner of a wind turbine.
          It contains the general properties for a wind turbine component as well as additional design load and dimentional attributes as listed below.
          It contains an update method to determine the mass, mass properties, and dimensions of the component.
    '''

    def __init__(self):
        super(Spinner_drive, self).__init__()

        # controls what happens if derivatives are missing
        self.missing_deriv_policy = 'assume_zero'

        # variables
        self.add_param('rotor_diameter', val=0.0, units='m', desc='rotor diameter')

        # outputs
        self.add_output('spinner_mass', val=0.0, units='kg', desc='overall component mass')

    def solve_nonlinear(self, params, unknowns, resids):

        # spinner mass comes from cost and scaling model
        unknowns['spinner_mass'] = 18.5 * params['rotor_diameter'] + (-520.5)


#-------------------------------------------------------------------------


#@implement_base(HubBase)
class HubSE(Group):
    '''
       HubWPACT class
          The HubWPACT class is used to represent the hub system of a wind turbine.
    '''

    def __init__(self):
        super(HubSE, self).__init__()

        # variables
        self.add('blade_mass', IndepVarComp('blade_mass', val=0.0, units='kg', desc='mass of one blade'), promotes=['*'])
        self.add('rotor_bending_moment', IndepVarComp('rotor_bending_moment', val=0.0, units='N*m', desc='flapwise bending moment at blade root'), promotes=['*'])
        self.add('rotor_diameter', IndepVarComp('rotor_diameter', val=0.0, units='m', desc='rotor diameter'), promotes=['*'])
        self.add('blade_root_diameter', IndepVarComp('blade_root_diameter', val=0.0, units='m', desc='blade root diameter'), promotes=['*'])
        self.add('machine_rating', IndepVarComp('machine_rating', val=0.0, units='MW', desc='machine rating of turbine'), promotes=['*'])

        # parameters
        self.add('blade_number', IndepVarComp('blade_number', val=3, desc='number of turbine blades', pass_by_obj=True), promotes=['*'])

        # DUMMY OUTPUTS DO NOT USE. Calculated in hubsystemadderdrive
        #self.add_output('hub_system_mass', val=0.0, units='kg')
        #self.add_output('hub_system_cm', val=np.array([0.0, 0.0, 0.0]), units='m')
        #self.add_output('hub_system_I', val=np.array([0.0, 0.0, 0.0]),)

        # select components
        self.add('hub', Hub_drive(), promotes=['*'])
        self.add('pitchSystem', PitchSystem_drive(), promotes=['*'])
        self.add('spinner', Spinner_drive(), promotes=['*'])
        self.add('adder', Hub_System_Adder_drive(), promotes=['*'])

#-------------------------------------------------------------------------

def example_5MW_4pt():

    # simple test of module

    # NREL 5 MW turbine
    prob = Problem()
    prob.root = HubSE()
    prob.setup()

    prob['rotor_diameter'] = 126.0  # m
    prob['blade_number'] = 3
    prob['blade_root_diameter'] = 3.542
    prob['machine_rating'] = 5000.0
    prob['blade_mass'] = 17740.0  # kg

    AirDensity = 1.225  # kg/(m^3)
    Solidity = 0.0517
    RatedWindSpeed = 11.05  # m/s
    prob['rotor_bending_moment'] = (3.06 * pi / 8) * AirDensity * (
        RatedWindSpeed ** 2) * (Solidity * (prob['rotor_diameter'] ** 3)) / prob['blade_number']

    prob.run()

    print "NREL 5 MW turbine test"
    print "Hub Components"
    print '  hub         {0:8.1f} kg'.format(prob['hub_mass'])  # 31644.47
    print '  pitch mech  {0:8.1f} kg'.format(prob['pitch_system_mass'])  # 17003.98
    print '  nose cone   {0:8.1f} kg'.format(prob['spinner_mass'])  # 1810.50
    # 50458.95
    print 'Hub system total {0:8.1f} kg'.format(prob['hub_system_mass'])
    print '    cm {0:6.2f} {1:6.2f} {2:6.2f}'.format(prob['hub_system_cm'][0], prob['hub_system_cm'][1], prob['hub_system_cm'][2])
    print '    I {0:6.1f} {1:6.1f} {2:6.1f}'.format(prob['hub_system_I'][0], prob['hub_system_I'][1], prob['hub_system_I'][2])
    print


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

    print "WindPACT 1.5 MW turbine test"
    print "Hub Components"
    print '  hub         {0:8.1f} kg'.format(prob['hub_mass'])
    print '  pitch mech  {0:8.1f} kg'.format(prob['pitch_system_mass'])
    print '  nose cone   {0:8.1f} kg'.format(prob['spinner_mass'])
    print 'HUB TOTAL     {0:8.1f} kg'.format(prob['hub_system_mass'])
    print 'cm {0:6.2f} {1:6.2f} {2:6.2f}'.format(prob['hub_system_cm'][0], prob['hub_system_cm'][1], prob['hub_system_cm'][2])
    print 'I {0:6.1f} {1:6.1f} {2:6.1f}'.format(prob['hub_system_I'][0], prob['hub_system_I'][1], prob['hub_system_I'][2])
    print


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

    print "windpact 750 kW turbine test"
    print "Hub Components"
    print '  hub         {0:8.1f} kg'.format(prob['hub_mass'])
    print '  pitch mech  {0:8.1f} kg'.format(prob['pitch_system_mass'])
    print '  nose cone   {0:8.1f} kg'.format(prob['spinner_mass'])
    print 'HUB TOTAL     {0:8.1f} kg'.format(prob['hub_system_mass'])
    print 'cm {0:6.2f} {1:6.2f} {2:6.2f}'.format(prob['hub_system_cm'][0], prob['hub_system_cm'][1], prob['hub_system_cm'][2])
    print 'I {0:6.1f} {1:6.1f} {2:6.1f}'.format(prob['hub_system_I'][0], prob['hub_system_I'][1], prob['hub_system_I'][2])
    print

if __name__ == "__main__":

    example_5MW_4pt()

    example_1p5MW_4pt()

    example_750kW_4pt()
