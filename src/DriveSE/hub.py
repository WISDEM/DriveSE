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
from drivewpact.hub import HubBase, PitchSystem, Spinner, HubSystemAdder, HubBase

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
        self.add('pitchSystem', PitchSystem())
        self.add('spinner', Spinner())

        # workflow
        self.driver.workflow.add(['hubSystem', 'hub', 'pitchSystem', 'spinner'])

        # connect inputs
        self.connect('blade_mass', ['pitchSystem.blade_mass'])
        self.connect('rotor_bending_moment', ['pitchSystem.rotor_bending_moment'])
        self.connect('blade_number', ['hub.blade_number', 'pitchSystem.blade_number'])
        self.connect('rotor_diameter', ['hub.rotor_diameter', 'pitchSystem.rotor_diameter', 'spinner.rotor_diameter'])
        self.connect('hub.diameter', ['pitchSystem.hub_diameter', 'spinner.hub_diameter'])
        self.connect('blade_root_diameter', 'hub.blade_root_diameter')
        self.connect('L_rb','hub.L_rb')
        self.connect('gamma','hub.gamma')
        self.connect('MB1_location','hub.MB1_location')
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
    rotor_diameter = Float(iotype='in', units='m', desc='rotor diameter')
    blade_root_diameter = Float(iotype='in', units='m', desc='blade root diameter')
    L_rb = Float(iotype='in', units = 'm', desc = 'distance between hub center and upwind main bearing')
    gamma = Float(iotype = 'in', units = 'deg', desc = 'shaft angle')
    MB1_location = Array(iotype = 'in', units = 'm', desc = 'center of mass of main bearing in [x,y,z] for an arbitrary coordinate system')
    machine_rating = Float(iotype = 'in', units = 'MW', desc = 'machine rating of turbine')
    
    # parameters
    blade_number = Int(3, iotype='in', desc='number of turbine blades')

    # outputs
    diameter = Float(0.0, iotype='out', units='m', desc='hub diameter')
    mass = Float(0.0, iotype='out', units='kg', desc='overall component mass')
    cm = Array(np.array([0.0, 0.0, 0.0]), iotype='out', desc='center of mass of the component in [x,y,z] for an arbitrary coordinate system')
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

        if self.L_rb:
            L_rb = self.L_rb
        else:
            L_rb = get_L_rb(self.rotor_diameter)

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


def example():

    # simple test of module

    # NREL 5 MW turbine
    print "NREL 5 MW turbine test"
    hubS = HubSE()
    hubS.rotor_diameter = 126.0 # m
    hubS.blade_number  = 3
    hubS.blade_root_diameter   = 3.542

    hubS.L_rb = 0.5
    hubS.gamma = 5.0
    hubS.MB1_location = np.array([-0.5, 0.0, 0.0])
    hubS.machine_rating = 5000.0

    hubS.run()

    print "Hub Components"
    print '  hub         {0:8.1f} kg'.format(hubS.hub.mass)  # 31644.47
    print '  pitch mech  {0:8.1f} kg'.format(hubS.pitchSystem.mass) # 17003.98
    print '  nose cone   {0:8.1f} kg'.format(hubS.spinner.mass) # 1810.50
    print 'Hub system total {0:8.1f} kg'.format(hubS.hub_system_mass) # 50458.95
    print '    cm {0:6.2f} {1:6.2f} {2:6.2f}'.format(hubS.hub_system_cm[0], hubS.hub_system_cm[1], hubS.hub_system_cm[2])
    print '    I {0:6.1f} {1:6.1f} {2:6.1f}'.format(hubS.hub_system_I[0], hubS.hub_system_I[1], hubS.hub_system_I[2])

if __name__ == "__main__":

    example()