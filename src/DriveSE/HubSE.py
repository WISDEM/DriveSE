"""
hubSE.py

Created by Katherine Dykes 2012.
Copyright (c) NREL. All rights reserved.
"""

from openmdao.main.api import Assembly
from openmdao.main.datatypes.api import Float, Int, Array
from math import pi
import numpy as np

from drivewpact.hub import HubBase, PitchSystem, Spinner, HubSystemAdder
from DriveSE_components import Hub_drive


class HubBase(Assembly):

    # variables
    blade_mass = Float(iotype='in', units='kg', desc='mass of one blade')
    rotor_bending_moment = Float(iotype='in', units='N*m', desc='flapwise bending moment at blade root')
    rotor_diameter = Float(iotype='in', units='m', desc='rotor diameter')
    blade_root_diameter = Float(iotype='in', units='m', desc='blade root diameter')

    # parameters
    blade_number = Int(3, iotype='in', desc='number of turbine blades')

    # outputs
    hub_system_mass = Float(0.0, iotype='out', units='kg', desc='overall component mass')
    hub_system_cm = Array(iotype='out', desc='center of mass of the hub relative to tower to in yaw-aligned c.s.')
    hub_system_I = Array(iotype='out', desc='mass moments of Inertia of hub [Ixx, Iyy, Izz, Ixy, Ixz, Iyz] around its center of mass in yaw-aligned c.s.')

    hub_mass = Float(0.0, iotype='out', units='kg')
    pitch_system_mass = Float(0.0, iotype='out', units='kg')
    spinner_mass = Float(0.0, iotype='out', units='kg')
        
class HubSE(HubBase):
    '''
       HubWPACT class
          The HubWPACT class is used to represent the hub system of a wind turbine.
    '''

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

def example():

    # simple test of module

    # NREL 5 MW turbine
    print "NREL 5 MW turbine test"
    hubS = HubSE()
    hubS.rotor_diameter = 126.0 # m
    hubS.blade_number  = 3
    hubS.blade_root_diameter   = 3.542

    hubS.hub.L_rb = 0.5
    hubS.hub.gamma = 5.0
    hubS.hub.MB1_location = np.array([-0.5, 0.0, 0.0])
    hubS.hub.machine_rating = 5000.0

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