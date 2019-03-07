"""
FUSED versions of HubSE components
"""

import numpy as np

from drivese.hubse_components import Hub, PitchSystem, Spinner, Hub_System_Adder
from openmdao.api import Component, Group

class Hub_System_Adder_OM(Component):
    ''' Get_hub_cm class
          The Get_hub_cm class is used to pass the hub cm data to upper level models.
          It contains the general properties for a wind turbine component as well as additional design load and dimentional attributes as listed below.
    '''

    def __init__(self):

        super(Hub_System_Adder_OM, self).__init__()

        # variables
        self.add_param('rotor_diameter', val=0.0, units='m', desc='rotor diameter')
        self.add_param('distance_hub2mb', val=0.0, units='m', desc='distance between hub center and upwind main bearing')
        self.add_param('shaft_angle', val=0.0, units='rad', desc='shaft angle')
        self.add_param('MB1_location', val=np.zeros(3), shape=(3,), units='m', desc='center of mass of main bearing in [x,y,z] for an arbitrary coordinate system')
        self.add_param('hub_mass', val=0.0, units='kg', desc='mass of Hub')
        self.add_param('hub_diameter', val=0.03, units='m', desc='hub diameter')
        self.add_param('hub_thickness', val=0.0, units='m', desc='hub thickness')
        self.add_param('pitch_system_mass', val=0.0, units='kg', desc='mass of Pitch System')
        self.add_param('spinner_mass', val=0.0, units='kg', desc='mass of spinner')

        # outputs
        self.add_output('hub_system_cm', val=np.zeros(3), shape=(3,), units='m',  desc='center of mass of the hub relative to tower to in yaw-aligned c.s.')
        self.add_output('hub_system_I', val=np.zeros(3), shape=(3,), desc='mass moments of Inertia of hub [Ixx, Iyy, Izz, Ixy, Ixz, Iyz] around its center of mass in yaw-aligned c.s.')
        self.add_output('hub_system_mass', val=0.0,  units='kg', desc='mass of hub system')

        self.hub_adder = Hub_System_Adder()

    def solve_nonlinear(self, inputs, outputs, resid):
    
        (outputs['hub_system_mass'], outputs['hub_system_cm'], outputs['hub_system_I']) \
             = self.hub_adder.compute(inputs['rotor_diameter'], inputs['distance_hub2mb'], inputs['shaft_angle'], inputs['MB1_location'], \
                              inputs['hub_mass'], inputs['hub_diameter'], inputs['hub_thickness'], inputs['pitch_system_mass'], inputs['spinner_mass'])        
        
        return outputs

# -------------------------------------------------

class Hub_OM(Component):
    ''' Hub class    
          The Hub class is used to represent the hub component of a wind turbine. 
          It contains the general properties for a wind turbine component as well as additional design load and dimentional attributes as listed below.
          It contains an update method to determine the mass, mass properties, and dimensions of the component.            
    '''

    def __init__(self, blade_number):

        super(Hub_OM, self).__init__()

        # variables
        self.add_param('blade_root_diameter', val=0.0, units='m', desc='blade root diameter')
        self.add_param('machine_rating', val=0.0, units='MW', desc='machine rating of turbine')

        # outputs
        self.add_output('hub_diameter', val=0.0, units='m', desc='hub diameter')
        self.add_output('hub_thickness', val=0.0, units='m', desc='hub thickness')
        self.add_output('hub_mass', val=0.0, units='kg', desc='overall component mass')

        self.hub = Hub(blade_number)

    def solve_nonlinear(self, inputs, outputs, resid):
    
        (outputs['hub_mass'], outputs['hub_diameter'], outputs['hub_thickness']) \
            = self.hub.compute(inputs['blade_root_diameter'], inputs['machine_rating'])
        
        return outputs


class PitchSystem_OM(Component):
    '''
     PitchSystem class
      The PitchSystem class is used to represent the pitch system of a wind turbine.
      It contains the general properties for a wind turbine component as well as additional design load and dimentional attributes as listed below.
      It contains an update method to determine the mass, mass properties, and dimensions of the component.
    '''

    def __init__(self, blade_number):

        super(PitchSystem_OM, self).__init__()

        # variables
        self.add_param('blade_mass', val=0.0, units='kg', desc='mass of one blade')
        self.add_param('rotor_bending_moment_y', val=0.0, units='N*m', desc='flapwise bending moment at blade root')

        # outputs
        self.add_output('pitch_system_mass', val=0.0, units='kg', desc='overall component mass')

        self.pitch = PitchSystem(blade_number)

    def solve_nonlinear(self, inputs, outputs, resid):
    
        (outputs['pitch_system_mass']) \
            = self.pitch.compute(inputs['blade_mass'], inputs['rotor_bending_moment_y'])
        
        return outputs


#-------------------------------------------------------------------------

class Spinner_OM(Component):
    '''
       Spinner class
          The SpinnerClass is used to represent the spinner of a wind turbine.
          It contains the general properties for a wind turbine component as well as additional design load and dimentional attributes as listed below.
          It contains an update method to determine the mass, mass properties, and dimensions of the component.
    '''

    def __init__(self):

        super(Spinner_OM, self).__init__()

        # variables
        self.add_param('rotor_diameter', val=0.0, units='m', desc='rotor diameter')

        # outputs
        self.add_output('spinner_mass', val=0.0, units='kg', desc='overall component mass')

        self.spinner = Spinner()

    def solve_nonlinear(self, inputs, outputs, resid):

        (outputs['spinner_mass']) \
            =self.spinner.compute(inputs['rotor_diameter'])    
        
        return outputs



# Main code to run hub system examples
if __name__ == "__main__":

    #TODO: fused_hubse examples
    pass
