
import numpy as np

from openmdao.api import Component, Group, Problem, IndepVarComp
from drivese.hubse_components import Hub, PitchSystem, Spinner, Hub_System_Adder, Hub_Mass_Adder, Hub_CM_Adder


#-------------------------------------------------------------------------

class Hub_System_Adder_OM(Component):
    ''' 
    Compute hub mass, cm, and I
    '''

    def __init__(self, blade_number):

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
        self.add_param('blade_mass', val=0.0, units='kg', desc='mass of one blade')

        # outputs
        self.add_output('hub_system_cm', val=np.zeros(3), shape=(3,), units='m',  desc='center of mass of the hub relative to tower to in yaw-aligned c.s.')
        self.add_output('hub_system_I', val=np.zeros(6), shape=(3,), desc='mass moments of Inertia of hub [Ixx, Iyy, Izz, Ixy, Ixz, Iyz] around its center of mass in yaw-aligned c.s.')
        self.add_output('hub_system_mass', val=0.0,  units='kg', desc='mass of hub system')
        self.add_output('rotor_mass', val=0.0,  units='kg', desc='mass of rotor')

        self.hub_adder = Hub_System_Adder(blade_number)

    def solve_nonlinear(self, inputs, outputs, resid):
    
        (outputs['rotor_mass'], outputs['hub_system_mass'], outputs['hub_system_cm'], outputs['hub_system_I']) \
             = self.hub_adder.compute(inputs['rotor_diameter'], inputs['blade_mass'], inputs['distance_hub2mb'], inputs['shaft_angle'], inputs['MB1_location'], \
                              inputs['hub_mass'], inputs['hub_diameter'], inputs['hub_thickness'], inputs['pitch_system_mass'], inputs['spinner_mass'])        
        
        return outputs


#-------------------------------------------------------------------------

class Hub_Mass_Adder_OM(Component):
    ''' 
    Compute hub mass and I
    Excluding cm here, because it has a dependency on main bearing location, which can only be calculated once the full rotor mass is set
    '''

    def __init__(self, blade_number):

        super(Hub_Mass_Adder_OM, self).__init__()

        # variables
        self.add_param('hub_mass', val=0.0, units='kg', desc='mass of Hub')
        self.add_param('hub_diameter', val=0.03, units='m', desc='hub diameter')
        self.add_param('hub_thickness', val=0.0, units='m', desc='hub thickness')
        self.add_param('pitch_system_mass', val=0.0, units='kg', desc='mass of Pitch System')
        self.add_param('spinner_mass', val=0.0, units='kg', desc='mass of spinner')
        self.add_param('blade_mass', val=0.0, units='kg', desc='mass of one blade')

        # outputs
        self.add_output('hub_system_I', val=np.zeros(6), shape=(3,), desc='mass moments of Inertia of hub [Ixx, Iyy, Izz, Ixy, Ixz, Iyz] around its center of mass in yaw-aligned c.s.')
        self.add_output('hub_system_mass', val=0.0,  units='kg', desc='mass of hub system')
        self.add_output('rotor_mass', val=0.0,  units='kg', desc='mass of rotor')
        self.add_output('hub_I', val=np.zeros(3), desc='Hub inertia about rotor axis (does not include pitch and spinner masses)')

        self.hub_adder = Hub_Mass_Adder(blade_number)

    def solve_nonlinear(self, inputs, outputs, resid):
    
        (outputs['rotor_mass'], outputs['hub_system_mass'], outputs['hub_system_I'], outputs['hub_I'])\
             = self.hub_adder.compute(inputs['blade_mass'], inputs['hub_mass'], inputs['hub_diameter'], inputs['hub_thickness'], inputs['pitch_system_mass'], inputs['spinner_mass'])        
        
        return outputs


#-------------------------------------------------------------------------

class Hub_CM_Adder_OM(Component):
    ''' 
    Compute hub cm
    Separating cm here, because it has a dependency on main bearing location, which can only be calculated once the full rotor mass is set
    '''

    def __init__(self):

        super(Hub_CM_Adder_OM, self).__init__()

        # variables
        self.add_param('rotor_diameter', val=0.0, units='m', desc='rotor diameter')
        self.add_param('distance_hub2mb', val=0.0, units='m', desc='distance between hub center and upwind main bearing')
        self.add_param('shaft_angle', val=0.0, units='rad', desc='shaft angle')
        self.add_param('MB1_location', val=np.zeros(3), shape=(3,), units='m', desc='center of mass of main bearing in [x,y,z] for an arbitrary coordinate system')

        # outputs
        self.add_output('hub_system_cm', val=np.zeros(3), shape=(3,), units='m',  desc='center of mass of the hub relative to tower to in yaw-aligned c.s.')

        self.hub_adder = Hub_CM_Adder()

    def solve_nonlinear(self, inputs, outputs, resid):
    
        outputs['hub_system_cm'] = self.hub_adder.compute(inputs['rotor_diameter'], inputs['distance_hub2mb'],
                                                          inputs['shaft_angle'], inputs['MB1_location'])
        
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
        self.add_param('machine_rating', val=0.0, units='kW', desc='machine rating of turbine')

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

#-------------------------------------------------------------------------

class HubSE(Group):

    def __init__(self, blade_number):
        super(HubSE, self).__init__()
        '''
           HubSE class
              The HubSE class is used to represent the hub system of a wind turbine. 
              HubSE integrates the hub, pitch system and spinner / nose cone components for the hub system.
        '''
        self.add('hub', Hub_OM(blade_number), ['*'])
        self.add('pitchSystem', PitchSystem_OM(blade_number), ['*'])
        self.add('spinner', Spinner_OM(), ['*'])
        self.add('adder', Hub_System_Adder_OM(blade_number), ['*'])
        
        #-------------------------------------------------------------------------
        # Examples based on reference turbines including the NREL 5 MW, WindPACT 1.5 MW and the GRC 750 kW system.
        

class HubMassOnlySE(Group):

    def __init__(self, blade_number):
        super(HubMassOnlySE, self).__init__()
        '''
           HubSE class
              The HubSE class is used to represent the hub system of a wind turbine. 
              HubSE integrates the hub, pitch system and spinner / nose cone components for the hub system.
              Mass ly here, because it CM has a dependency on main bearing location, which can only be calculated once the full rotor mass is set
        '''
        self.add('hub', Hub_OM(blade_number), ['*'])
        self.add('pitchSystem', PitchSystem_OM(blade_number), ['*'])
        self.add('spinner', Spinner_OM(), ['*'])
        self.add('adder', Hub_Mass_Adder_OM(blade_number), ['*'])
        
        #-------------------------------------------------------------------------
        # Examples based on reference turbines including the NREL 5 MW, WindPACT 1.5 MW and the GRC 750 kW system.
        
def example_5MW_4pt():
    
    # simple test of module
    
    blade_number = 3

    prob=Problem(root=HubSE(blade_number))
    prob.setup()

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

    prob.run()

    print("NREL 5 MW turbine test")
    print(prob.root.unknowns.dump())

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
    print("Hub Objects")
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
    print("Hub Objects")
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
