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
from drivewpact.hub import HubBase#, HubSystemAdder
from drivese.drivese_utils import get_L_rb

@implement_base(HubBase)
class Hub_System_Adder_drive(Component):
    ''' Get_hub_cm class
          The Get_hub_cm class is used to pass the hub cm data to upper level models.
          It contains the general properties for a wind turbine component as well as additional design load and dimentional attributes as listed below.
    '''

    # variables
    rotor_diameter = Float(iotype='in', units='m', desc='rotor diameter')
    L_rb = Float(0.0,iotype='in', units = 'm', desc = 'distance between hub center and upwind main bearing')
    shaft_angle = Float(iotype = 'in', units = 'deg', desc = 'shaft angle')
    MB1_location = Array(iotype = 'in', units = 'm', desc = 'center of mass of main bearing in [x,y,z] for an arbitrary coordinate system')
    hub_mass = Float(iotype='in', units='kg',desc='mass of Hub')
    hub_diameter = Float(3.0,iotype='in', units='m', desc='hub diameter')
    hub_thickness = Float(iotype='in', units='m', desc='hub thickness')
    pitch_system_mass = Float(iotype='in', units='kg',desc='mass of Pitch System')
    spinner_mass = Float(iotype='in', units='kg',desc='mass of spinner')

    # outputs
    hub_system_cm = Array(iotype='out', desc='center of mass of the hub relative to tower to in yaw-aligned c.s.')
    hub_system_I = Array(iotype='out', desc='mass moments of Inertia of hub [Ixx, Iyy, Izz, Ixy, Ixz, Iyz] around its center of mass in yaw-aligned c.s.')
    hub_system_mass = Float(iotype='out', units='kg',desc='mass of hub system')

    def __init__(self):
        ''' Initialize Get_hub_cm component
        '''

        super(GetHubCM_drive, self).__init__()

        #controls what happens if derivatives are missing
        self.missing_deriv_policy = 'assume_zero'

    def execute(self):
        if self.L_rb>0:
            L_rb = self.L_rb
        else:
            L_rb = get_L_rb(self.rotor_diameter)

        cm = np.array([0.0,0.0,0.0])
        cm[0]     = self.MB1_location[0] - L_rb
        cm[1]     = 0.0
        cm[2]     = self.MB1_location[2] + L_rb*sin(radians(self.gamma))
        self.hub_system_cm = (cm)

        self.hub_system_mass = self.hub_mass + self.pitch_system_mass + self.spinner_mass

        #add I definitions here
        hub_I = np.array([0.0, 0.0, 0.0])
        hub_I[0] = 0.4 * (self.hub_mass) * ((self.hub_diameter / 2) ** 5 - (self.hub_diameter / 2 - self.hub_thickness) ** 5) / \
               ((self.hub_diameter / 2) ** 3 - (self.hub_diameter / 2 - self.hub_thickness) ** 3)
        hub_I[1] = hub_I[0]
        hub_I[2] = hub_I[1]

        pitch_system_I = np.array([0.0, 0.0, 0.0])
        pitch_system_I[0] = self.pitch_system_mass * (self.diameter ** 2) / 4
        pitch_system_I[1] = pitch_system_I[0]
        pitch_system_I[2] = pitch_system_I[1]


        if self.hub_diameter == 0:
            spinner_diameter =(3.30)
        else:
            spinner_diameter =(self.hub_diameter)
        spinner_thickness = spinner_diameter * (0.055 / 3.30)         # 0.055 for 1.5 MW outer diameter of 3.3 - using proportional constant

        spinner_I = np.array([0.0, 0.0, 0.0])
        spinner_I[0] = 0.4 * (self.spinner_mass) * ((spinner_diameter / 2) ** 5 - (spinner_diameter / 2 - spinner_thickness) ** 5) / \
               ((spinner_diameter / 2) ** 3 - (spinner_diameter / 2 - spinner_thickness) ** 3)
        spinner_I[1] = spinner_I[0]
        spinner_I[2] = spinner_I[1]


        #add moments of inertia
        I = np.zeros(6)
        for i in (range(0,3)):                        # calculating MOI, at nacelle center of gravity with origin at tower top center / yaw mass center, ignoring masses of non-drivetrain components / auxiliary systems
            # calculate moments around CM
            # sum moments around each components CM
            I[i]  =  hub_I[i] + pitch_system_I[i] + spinner_I[i]
            # translate to hub system CM using parallel axis theorem- because cm is assumed shared, unneeded
            # for j in (range(0,3)):
            #     if i != j:
            #         I[i] +=  (self.hub_mass * (self.hub_cm[j] - self.hub_system_cm[j]) ** 2) + \
            #                       (self.pitch_system_mass * (self.pitch_system_cm[j] - self.hub_system_cm[j]) ** 2) + \
            #                       (self.spinner_mass * (self.spinner_cm[j] - self.hub_system_cm[j]) ** 2)
        self.hub_system_I = I

#-------------------------------------------------------------------------------


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
    machine_rating = Float(iotype = 'in', units = 'MW', desc = 'machine rating of turbine')
    
    # parameters
    blade_number = Int(3, iotype='in', desc='number of turbine blades')

    # outputs
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
        self.driver.workflow.add(['hub', 'pitchSystem', 'spinner'])

        # connect inputs
        self.connect('blade_mass', ['pitchSystem.blade_mass'])
        self.connect('rotor_bending_moment', ['pitchSystem.rotor_bending_moment'])
        self.connect('blade_number', ['hub.blade_number', 'pitchSystem.blade_number'])
        self.connect('rotor_diameter', ['hub.rotor_diameter', 'pitchSystem.rotor_diameter', 'spinner.rotor_diameter'])
        self.connect('blade_root_diameter', 'hub.blade_root_diameter')
        self.connect('machine_rating','hub.machine_rating')

        # # connect components #removed and connected at turbine level
        # self.connect('hub.mass', 'hubSystem.hub_mass')
        # self.connect('pitchSystem.mass', 'hubSystem.pitch_system_mass')
        # self.connect('spinner.mass', 'hubSystem.spinner_mass')

        # connect outputs
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
    machine_rating = Float(iotype = 'in', units = 'MW', desc = 'machine rating of turbine')
    
    # parameters
    blade_number = Int(3, iotype='in', desc='number of turbine blades')

    # outputs
    diameter = Float(0.0, iotype='out', units='m', desc='hub diameter')
    thickness = Float(0.0, iotype='out',units='m',desc='hub thickness')
    mass = Float(0.0, iotype='out', units='kg', desc='overall component mass')
    
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

    # parameters
    blade_number = Int(3, iotype='in', desc='number of turbine blades')

    # outputs
    mass = Float(0.0, iotype='out', units='kg', desc='overall component mass')

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

    # outputs
    mass = Float(0.0, iotype='out', units='kg', desc='overall component mass')

    def __init__(self):
        '''
        Initializes spinner system
        '''

        super(Spinner_drive, self).__init__()

        #controls what happens if derivatives are missing
        self.missing_deriv_policy = 'assume_zero'

    def execute(self):

        self.mass =18.5 * self.rotor_diameter + (-520.5)   # spinner mass comes from cost and scaling model


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