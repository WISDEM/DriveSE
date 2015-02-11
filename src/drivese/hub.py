"""
hubSE.py

Created by Katherine Dykes 2012.
Copyright (c) NREL. All rights reserved.
"""

from openmdao.main.api import Assembly, Component
from openmdao.main.datatypes.api import Float, Int, Array
import numpy as np
from math import pi, cos, sqrt, radians, sin, exp, log10, log, floor, ceil

from fusedwind.interface import implement_base
from drivewpact.hub import HubBase, HubSystemAdder
from drivese.drivese_utils import get_L_rb

@implement_base(HubBase)
class HubSE(Assembly):
    '''
       HubWPACT class
          The HubWPACT class is used to represent the hub system of a wind turbine.
    '''

    # variables
    blade_mass = Float(iotype='in', units='kg', desc='mass of one blade')
    rotor_bending_moment = Float(iotype='in', units='N*m', desc='flapwise bending moment at blade root')
    rotor_diameter = Float(iotype='in', units='m', desc='rotor diameter')
    blade_root_diameter = Float(iotype='in', units='m', desc='blade root diameter')
    L_rb = Float(iotype='in', units = 'm', desc = 'distance between hub center and upwind main bearing')
    gamma = Float(iotype = 'in', units = 'deg', desc = 'shaft angle')
    MB1_location = Array(iotype = 'in', units = 'm', desc = 'center of mass of main bearing in [x,y,z] for an arbitrary coordinate system')
    machine_rating = Float(iotype = 'in', units = 'MW', desc = 'machine rating of turbine')
    
    # parameters
    blade_number = Int(3, iotype='in', desc='number of turbine blades')

    # outputs
    hub_system_mass = Float(0.0, iotype='out', units='kg', desc='overall component mass')
    hub_system_cm = Array(iotype='out', desc='center of mass of the hub relative to tower to in yaw-aligned c.s.')
    hub_system_I = Array(iotype='out', desc='mass moments of Inertia of hub [Ixx, Iyy, Izz, Ixy, Ixz, Iyz] around its center of mass in yaw-aligned c.s.')

    hub_mass = Float(0.0, iotype='out', units='kg')
    pitch_system_mass = Float(0.0, iotype='out', units='kg')
    spinner_mass = Float(0.0, iotype='out', units='kg')

    def configure(self):

        # select components
        self.add('hubSystem', HubSystemAdder())
        self.add('hub', Hub_drive())
        self.add('pitchSystem', PitchSystem_drive())
        self.add('spinner', Spinner_drive())

        # workflow
        self.driver.workflow.add(['hubSystem', 'hub', 'pitchSystem', 'spinner'])

        # connect inputs
        self.connect('blade_mass', ['pitchSystem.blade_mass'])
        self.connect('rotor_bending_moment', ['pitchSystem.rotor_bending_moment'])
        self.connect('blade_number', ['hub.blade_number', 'pitchSystem.blade_number'])
        self.connect('rotor_diameter', ['hub.rotor_diameter', 'pitchSystem.rotor_diameter', 'spinner.rotor_diameter'])
        self.connect('hub.diameter', ['pitchSystem.hub_diameter', 'spinner.hub_diameter'])
        self.connect('blade_root_diameter', 'hub.blade_root_diameter')
        self.connect('L_rb',['hub.L_rb','pitchSystem.L_rb','spinner.L_rb'])
        self.connect('gamma',['hub.gamma','pitchSystem.gamma','spinner.gamma'])
        self.connect('MB1_location',['hub.MB1_location','pitchSystem.MB1_location','spinner.MB1_location'])
        self.connect('machine_rating','hub.machine_rating')

        # connect components
        self.connect('hub.mass', 'hubSystem.hub_mass')
        self.connect('hub.cm', 'hubSystem.hub_cm')
        self.connect('hub.I', 'hubSystem.hub_I')
        self.connect('pitchSystem.mass', 'hubSystem.pitch_system_mass')
        self.connect('pitchSystem.cm', 'hubSystem.pitch_system_cm')
        self.connect('pitchSystem.I', 'hubSystem.pitch_system_I')
        self.connect('spinner.mass', 'hubSystem.spinner_mass')
        self.connect('spinner.cm', 'hubSystem.spinner_cm')
        self.connect('spinner.I', 'hubSystem.spinner_I')

        # connect outputs
        self.connect('hubSystem.hub_system_mass', 'hub_system_mass')
        self.connect('hubSystem.hub_system_cm', 'hub_system_cm')
        self.connect('hubSystem.hub_system_I', 'hub_system_I')
        self.connect('hub.mass', 'hub_mass')
        self.connect('pitchSystem.mass', 'pitch_system_mass')
        self.connect('spinner.mass', 'spinner_mass')

#-------------------------------------------------------------------------------

# -------------------------------------------------

class Hub_drive(Component):
    ''' Hub class    
          The Hub class is used to represent the hub component of a wind turbine. 
          It contains the general properties for a wind turbine component as well as additional design load and dimentional attributes as listed below.
          It contains an update method to determine the mass, mass properties, and dimensions of the component.            
    '''

    # variables
    blade_root_diameter = Float(iotype='in', units='m', desc='blade root diameter')
    MB1_location = Array(iotype = 'in', units = 'm', desc = 'center of mass of main bearing in [x,y,z] for an arbitrary coordinate system')
    machine_rating = Float(iotype = 'in', units = 'MW', desc = 'machine rating of turbine')
    
    # parameters
    blade_number = Int(3, iotype='in', desc='number of turbine blades')

    # outputs
    diameter = Float(0.0, iotype='out', units='m', desc='hub diameter')
    mass = Float(0.0, iotype='out', units='kg', desc='overall component mass')
    I = Array(np.array([0.0, 0.0, 0.0]), iotype='out', desc=' moments of Inertia for the component [Ixx, Iyy, Izz] around its center of mass')
    
    def __init__(self):
        ''' 
        Initializes hub component  
        '''

        super(Hub_drive, self).__init__()

    def execute(self):

        if self.blade_root_diameter: #added 8/6/14 to allow analysis of hubs for unknown blade roots.
            blade_root_diameter = self.blade_root_diameter
        else:
            blade_root_diameter = 2.659*self.machine_rating**.3254


        #Model hub as a cyclinder with holes for blade root and nacelle flange.
        rCyl=1.1*blade_root_diameter/2.0
        hCyl=2.8*blade_root_diameter/2.0
        castThickness = rCyl/10.0
        approxCylVol=2*pi*rCyl*castThickness*hCyl
        bladeRootVol=pi*(blade_root_diameter/2.0)**2*castThickness

        #assume nacelle flange opening is similar to blade root opening
        approxCylNetVol = approxCylVol - (1.0 + self.blade_number)*bladeRootVol
        castDensity = 7200.0 # kg/m^3
        self.mass=approxCylNetVol*castDensity

        # calculate mass properties
        self.diameter=2*rCyl
        self.thickness=castThickness

        I = np.array([0.0, 0.0, 0.0])

        I[0] = 0.4 * (self.mass) * ((self.diameter / 2) ** 5 - (self.diameter / 2 - self.thickness) ** 5) / \
               ((self.diameter / 2) ** 3 - (self.diameter / 2 - self.thickness) ** 3)
        I[1] = I[0]
        I[2] = I[1]
        self.I = (I)

class PitchSystem_drive(Component):
    '''
     PitchSystem class
      The PitchSystem class is used to represent the pitch system of a wind turbine.
      It contains the general properties for a wind turbine component as well as additional design load and dimentional attributes as listed below.
      It contains an update method to determine the mass, mass properties, and dimensions of the component.
    '''

    # variables
    blade_mass = Float(iotype='in', units='kg', desc='mass of one blade')
    rotor_bending_moment = Float(iotype='in', units='N*m', desc='flapwise bending moment at blade root')
    rotor_diameter = Float(iotype='in', units='m', desc='rotor diameter')
    MB1_location = Array(iotype = 'in', units = 'm', desc = 'center of mass of main bearing in [x,y,z] for an arbitrary coordinate system')
    L_rb = Float(iotype='in', units = 'm', desc = 'distance between hub center and upwind main bearing')
    gamma = Float(iotype = 'in', units = 'deg', desc = 'shaft angle')

    # parameters
    hub_diameter = Float(0.0, iotype='in', units='m', desc='hub diameter')
    blade_number = Int(3, iotype='in', desc='number of turbine blades')

    # outputs
    mass = Float(0.0, iotype='out', units='kg', desc='overall component mass')
    cm = Array(np.array([0.0, 0.0, 0.0]), iotype='out', desc='center of mass of the component in [x,y,z] for an arbitrary coordinate system')
    I = Array(np.array([0.0, 0.0, 0.0]), iotype='out', desc=' moments of Inertia for the component [Ixx, Iyy, Izz] around its center of mass')

    def __init__(self):
        '''
        Initializes pitch system
        '''

        super(PitchSystem_drive, self).__init__()

        #controls what happens if derivatives are missing
        self.missing_deriv_policy = 'assume_zero'

    def execute(self):

        # Sunderland method for calculating pitch system masses
        pitchmatldensity = 7860.0                             # density of pitch system material (kg / m^3) - assuming BS1503-622 (same material as LSS)
        pitchmatlstress  = 371000000.0                              # allowable stress of hub material (N / m^2)

        # Root moment required as input, could be undone
        '''if rotor_bending_moment == 0.0:
            rotor_bending_moment = (3.06 * pi / 8) * AirDensity * (RatedWindSpeed ** 2) * (Solidity * (RotorDiam ** 3)) / BladeNum
                                                            # simplified equation for blade root moment (Sunderland model) if one is not provided'''

        hubpitchFact      = 1.0                                 # default factor is 1.0 (0.54 for modern designs)
        #self.mass =hubpitchFact * (0.22 * self.blade_mass * self.blade_number + 12.6 * self.blade_number * self.rotor_bending_moment * (pitchmatldensity / pitchmatlstress))
                                                            # mass of pitch system based on Sunderland model
        self.mass =hubpitchFact * (0.22 * self.blade_mass * self.blade_number + 12.6 * self.rotor_bending_moment * (pitchmatldensity / pitchmatlstress))
                                                            # mass of pitch system based on Sunderland model

        # calculate mass properties
        if self.hub_diameter == 0:
            self.diameter =(3.30)
        else:
            self.diameter =(self.hub_diameter)

        if self.L_rb:
            L_rb = self.L_rb
        else:
            L_rb = get_L_rb(self.rotor_diameter)

        cm = np.array([0.0,0.0,0.0])
        cm[0]     = self.MB1_location[0] - L_rb
        cm[1]     = 0.0
        cm[2]     = self.MB1_location[2] + L_rb*sin(radians(self.gamma))
        self.cm = (cm)

        I = np.array([0.0, 0.0, 0.0])
        I[0] = self.mass * (self.diameter ** 2) / 4
        I[1] = I[0]
        I[2] = I[1]
        self.I = (I)

#-------------------------------------------------------------------------------

class Spinner_drive(Component):
    '''
       Spinner class
          The SpinnerClass is used to represent the spinner of a wind turbine.
          It contains the general properties for a wind turbine component as well as additional design load and dimentional attributes as listed below.
          It contains an update method to determine the mass, mass properties, and dimensions of the component.
    '''

    # variables
    rotor_diameter = Float(iotype='in', units='m', desc='rotor diameter')
    MB1_location = Array(iotype = 'in', units = 'm', desc = 'center of mass of main bearing in [x,y,z] for an arbitrary coordinate system')
    L_rb = Float(iotype='in', units = 'm', desc = 'distance between hub center and upwind main bearing')
    gamma = Float(iotype = 'in', units = 'deg', desc = 'shaft angle')

    # parameters
    hub_diameter = Float(0.0, iotype='in', units='m', desc='hub diameter')

    # outputs
    mass = Float(0.0, iotype='out', units='kg', desc='overall component mass')
    cm = Array(np.array([0.0, 0.0, 0.0]), iotype='out', desc='center of mass of the component in [x,y,z] for an arbitrary coordinate system')
    I = Array(np.array([0.0, 0.0, 0.0]), iotype='out', desc=' moments of Inertia for the component [Ixx, Iyy, Izz] around its center of mass')

    def __init__(self):
        '''
        Initializes spinner system
        '''

        super(Spinner_drive, self).__init__()

        #controls what happens if derivatives are missing
        self.missing_deriv_policy = 'assume_zero'

    def execute(self):

        self.mass =18.5 * self.rotor_diameter + (-520.5)   # spinner mass comes from cost and scaling model

        # calculate mass properties
        if self.hub_diameter == 0:
            self.diameter =(3.30)
        else:
            self.diameter =(self.hub_diameter)
        self.thickness = self.diameter * (0.055 / 3.30)         # 0.055 for 1.5 MW outer diameter of 3.3 - using proportional constant

        if self.L_rb:
            L_rb = self.L_rb
        else:
            L_rb = get_L_rb(self.rotor_diameter)

        cm = np.array([0.0,0.0,0.0])
        cm[0]     = self.MB1_location[0] - L_rb
        cm[1]     = 0.0
        cm[2]     = self.MB1_location[2] + L_rb*sin(radians(self.gamma))
        self.cm = (cm)

        I = np.array([0.0, 0.0, 0.0])
        I[0] = 0.4 * (self.mass) * ((self.diameter / 2) ** 5 - (self.diameter / 2 - self.thickness) ** 5) / \
               ((self.diameter / 2) ** 3 - (self.diameter / 2 - self.thickness) ** 3)
        I[1] = I[0]
        I[2] = I[1]
        self.I = (I)


def example_5MW_4pt():

    # simple test of module

    # NREL 5 MW turbine
    hub = HubSE()

    hub.rotor_diameter = 126.0 # m
    hub.blade_number  = 3
    hub.blade_root_diameter   = 3.542
    hub.machine_rating = 5000.0
    hub.blade_mass = 17740.0 # kg

    AirDensity= 1.225 # kg/(m^3)
    Solidity  = 0.0517
    RatedWindSpeed = 11.05 # m/s
    hub.rotor_bending_moment = (3.06 * pi / 8) * AirDensity * (RatedWindSpeed ** 2) * (Solidity * (hub.rotor_diameter ** 3)) / hub.blade_number

    hub.gamma = 5.0
    hub.MB1_location = np.array([-3.2, 0.0, 1.0])

    hub.run()

    print "NREL 5 MW turbine test"
    print "Hub Components"
    print '  hub         {0:8.1f} kg'.format(hub.hub.mass)  # 31644.47
    print '  pitch mech  {0:8.1f} kg'.format(hub.pitchSystem.mass) # 17003.98
    print '  nose cone   {0:8.1f} kg'.format(hub.spinner.mass) # 1810.50
    print 'Hub system total {0:8.1f} kg'.format(hub.hub_system_mass) # 50458.95
    print '    cm {0:6.2f} {1:6.2f} {2:6.2f}'.format(hub.hub_system_cm[0], hub.hub_system_cm[1], hub.hub_system_cm[2])
    print '    I {0:6.1f} {1:6.1f} {2:6.1f}'.format(hub.hub_system_I[0], hub.hub_system_I[1], hub.hub_system_I[2])
    print

def example_1p5MW_4pt():

    # WindPACT 1.5 MW turbine
    hub = HubSE()
    hub.blade_mass = 4470.0
    hub.rotor_diameter = 70.0
    hub.bladeNumer  = 3
    hub.hub_diameter   = 3.0
    hub.machine_rating = 1500.0
    hub.blade_root_diameter = 2.0 #TODO - find actual number

    AirDensity= 1.225
    Solidity  = 0.065
    RatedWindSpeed = 12.12
    hub.rotor_bending_moment = (3.06 * pi / 8) * AirDensity * (RatedWindSpeed ** 2) * (Solidity * (hub.rotor_diameter ** 3)) / hub.blade_number

    hub.gamma = 5.0
    hub.MB1_location = np.array([-2.2, 0.0, 0.5])

    hub.run()

    print "WindPACT 1.5 MW turbine test"
    print "Hub Components"
    print '  hub         {0:8.1f} kg'.format(hub.hub.mass)
    print '  pitch mech  {0:8.1f} kg'.format(hub.pitchSystem.mass)
    print '  nose cone   {0:8.1f} kg'.format(hub.spinner.mass)
    print 'HUB TOTAL     {0:8.1f} kg'.format(hub.hub_system_mass)
    print 'cm {0:6.2f} {1:6.2f} {2:6.2f}'.format(hub.hub_system_cm[0], hub.hub_system_cm[1], hub.hub_system_cm[2])
    print 'I {0:6.1f} {1:6.1f} {2:6.1f}'.format(hub.hub_system_I[0], hub.hub_system_I[1], hub.hub_system_I[2])
    print


def example_750kW_4pt():

    # GRC 750 kW turbine
    hub = HubSE()
    hub.blade_mass = 3400.0
    hub.rotor_diameter = 48.2
    hub.bladeNumer = 3
    hub.hub_diameter = 3.0
    hub.machine_rating = 750.0
    hub.blade_root_diameter = 1.0 #TODO - find actual number

    AirDensity = 1.225
    Solidity = 0.07 # uknown value
    RatedWindSpeed = 16.0
    hub.rotor_bending_moment = (3.06 * pi / 8) * AirDensity * (RatedWindSpeed ** 2) * (Solidity * (hub.rotor_diameter ** 3)) / hub.blade_number

    hub.gamma = 5.0
    hub.MB1_location = np.array([-1.2, 0.0, 0.4])

    hub.run()

    print "windpact 750 kW turbine test"
    print "Hub Components"
    print '  hub         {0:8.1f} kg'.format(hub.hub.mass)
    print '  pitch mech  {0:8.1f} kg'.format(hub.pitchSystem.mass)
    print '  nose cone   {0:8.1f} kg'.format(hub.spinner.mass)
    print 'HUB TOTAL     {0:8.1f} kg'.format(hub.hub_system_mass)
    print 'cm {0:6.2f} {1:6.2f} {2:6.2f}'.format(hub.hub_system_cm[0], hub.hub_system_cm[1], hub.hub_system_cm[2])
    print 'I {0:6.1f} {1:6.1f} {2:6.1f}'.format(hub.hub_system_I[0], hub.hub_system_I[1], hub.hub_system_I[2])
    print

if __name__ == "__main__":

    example_5MW_4pt()
    
    example_1p5MW_4pt()

    example_750kW_4pt()