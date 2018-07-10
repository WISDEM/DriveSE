"""
FUSED versions of HubSE components
"""

import numpy as np

from hubse_components import Hub, PitchSystem, Spinner, Hub_System_Adder

# FUSED helper functions and interface defintions
from fusedwind.fused_wind import FUSED_Object

class FUSED_Hub_System_Adder(FUSED_Object):
    ''' Get_hub_cm class
          The Get_hub_cm class is used to pass the hub cm data to upper level models.
          It contains the general properties for a wind turbine component as well as additional design load and dimentional attributes as listed below.
    '''

    def __init__(self):

        super(FUSED_Hub_System_Adder, self).__init__()

        # variables
        self.add_input(**{'name' : 'rotor_diameter', 'val' : 0.0, 'units' : 'm', 'desc' : 'rotor diameter'})
        self.add_input(**{'name' : 'distance_hub2mb', 'val' : 0.0, 'units' : 'm', 'desc' : 'distance between hub center and upwind main bearing'})
        self.add_input(**{'name' : 'shaft_angle', 'val' : 0.0, 'units' : 'deg', 'desc' : 'shaft angle'})
        self.add_input(**{'name' : 'MB1_location', 'val' : np.zeros(3), 'shape' : (3,), 'units' : 'm', 'desc' : 'center of mass of main bearing in [x,y,z] for an arbitrary coordinate system'})
        self.add_input(**{'name' : 'hub_mass', 'val' : 0.0, 'units' : 'kg', 'desc' : 'mass of Hub'})
        self.add_input(**{'name' : 'hub_diameter', 'val' : 0.03, 'units' : 'm', 'desc' : 'hub diameter'})
        self.add_input(**{'name' : 'hub_thickness', 'val' : 0.0, 'units' : 'm', 'desc' : 'hub thickness'})
        self.add_input(**{'name' : 'pitch_system_mass', 'val' : 0.0, 'units' : 'kg', 'desc' : 'mass of Pitch System'})
        self.add_input(**{'name' : 'spinner_mass', 'val' : 0.0, 'units' : 'kg', 'desc' : 'mass of spinner'})

        # outputs
        self.add_output(**{'name' : 'hub_system_cm', 'val' : np.zeros(3), 'shape' : (3,), 'units' : 'm',  'desc' : 'center of mass of the hub relative to tower to in yaw-aligned c.s.'})
        self.add_output(**{'name' : 'hub_system_I', 'val' : np.zeros(3), 'shape' : (3,), 'desc' : 'mass moments of Inertia of hub [Ixx, Iyy, Izz, Ixy, Ixz, Iyz] around its center of mass in yaw-aligned c.s.'})
        self.add_output(**{'name' : 'hub_system_mass', 'val' : 0.0,  'units' : 'kg', 'desc' : 'mass of hub system'})

        self.hub_adder = Hub_System_Adder()

    def compute(self, inputs, outputs):
    
        (outputs['hub_system_mass'], outputs['hub_system_cm'], outputs['hub_system_I']) \
             = self.hub_adder.compute(inputs['rotor_diameter'], inputs['distance_hub2mb'], inputs['shaft_angle'], inputs['MB1_location'], \
                              inputs['hub_mass'], inputs['hub_diameter'], inputs['hub_thickness'], inputs['pitch_system_mass'], inputs['spinner_mass'])        
        
        return outputs

# -------------------------------------------------

class FUSED_Hub(FUSED_Object):
    ''' Hub class    
          The Hub class is used to represent the hub component of a wind turbine. 
          It contains the general properties for a wind turbine component as well as additional design load and dimentional attributes as listed below.
          It contains an update method to determine the mass, mass properties, and dimensions of the component.            
    '''

    def __init__(self, blade_number):

        super(FUSED_Hub, self).__init__()

        # variables
        self.add_input(**{'name' : 'blade_root_diameter', 'val' : 0.0, 'units' : 'm', 'desc' : 'blade root diameter'})
        self.add_input(**{'name' : 'machine_rating', 'val' : 0.0, 'units' : 'MW', 'desc' : 'machine rating of turbine'})

        # outputs
        self.add_output(**{'name' : 'hub_diameter', 'val' : 0.0, 'units' : 'm', 'desc' : 'hub diameter'})
        self.add_output(**{'name' : 'hub_thickness', 'val' : 0.0, 'units' : 'm', 'desc' : 'hub thickness'})
        self.add_output(**{'name' : 'hub_mass', 'val' : 0.0, 'units' : 'kg', 'desc' : 'overall component mass'})

        self.hub = Hub(blade_number)

    def compute(self, inputs, outputs):
    
        (outputs['hub_mass'], outputs['hub_diameter'], outputs['hub_thickness']) \
            = self.hub.compute(inputs['blade_root_diameter'], inputs['machine_rating'])
        
        return outputs


class FUSED_PitchSystem(FUSED_Object):
    '''
     PitchSystem class
      The PitchSystem class is used to represent the pitch system of a wind turbine.
      It contains the general properties for a wind turbine component as well as additional design load and dimentional attributes as listed below.
      It contains an update method to determine the mass, mass properties, and dimensions of the component.
    '''

    def __init__(self, blade_number):

        super(FUSED_PitchSystem, self).__init__()

        # variables
        self.add_input(**{'name' : 'blade_mass', 'val' : 0.0, 'units' : 'kg', 'desc' : 'mass of one blade'})
        self.add_input(**{'name' : 'rotor_bending_moment_y', 'val' : 0.0, 'units' : 'N*m', 'desc' : 'flapwise bending moment at blade root'})

        # outputs
        self.add_output(**{'name' : 'pitch_system_mass', 'val' : 0.0, 'units' : 'kg', 'desc' : 'overall component mass'})

        self.pitch = PitchSystem(blade_number)

    def compute(self, inputs, outputs):
    
        (outputs['pitch_system_mass']) \
            = self.pitch.compute(inputs['blade_mass'], inputs['rotor_bending_moment_y'])
        
        return outputs


#-------------------------------------------------------------------------

class FUSED_Spinner(FUSED_Object):
    '''
       Spinner class
          The SpinnerClass is used to represent the spinner of a wind turbine.
          It contains the general properties for a wind turbine component as well as additional design load and dimentional attributes as listed below.
          It contains an update method to determine the mass, mass properties, and dimensions of the component.
    '''

    def __init__(self):

        super(FUSED_Spinner, self).__init__()

        # variables
        self.add_input(**{'name' : 'rotor_diameter', 'val' : 0.0, 'units' : 'm', 'desc' : 'rotor diameter'})

        # outputs
        self.add_output(**{'name' : 'spinner_mass', 'val' : 0.0, 'units' : 'kg', 'desc' : 'overall component mass'})

        self.spinner = Spinner()

    def compute(self, inputs, outputs):

        (outputs['spinner_mass']) \
            =self.spinner.compute(inputs['rotor_diameter'])    
        
        return outputs



# Main code to run hub system examples
if __name__ == "__main__":

    #TODO: fused_hubse examples
    pass
