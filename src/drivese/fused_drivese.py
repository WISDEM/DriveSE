"""
FUSED versions of DriveSE components
"""

import numpy as np

from drivese.drivese_components import LowSpeedShaft4pt, LowSpeedShaft3pt, Gearbox, MainBearing, Bedplate, YawSystem, \
                                       Transformer, HighSpeedSide, Generator, NacelleSystemAdder, AboveYawMassAdder, RNASystemAdder
from openmdao.api import Component, Group

#-------------------------------------------------------------------------
# Components
#-------------------------------------------------------------------------

class LowSpeedShaft4pt_OM(Component):
    ''' LowSpeedShaft class
          The LowSpeedShaft class is used to represent the low speed shaft component of a wind turbine drivetrain. 
          It contains the general properties for a wind turbine component as well as additional design load and dimentional attributes as listed below.
          It contains an update method to determine the mass, mass properties, and dimensions of the component.
    '''

    def __init__(self, mb1Type, mb2Type, IEC_Class):

        super(LowSpeedShaft4pt_OM, self).__init__()

        # variables
        self.add_param('rotor_bending_moment_x', val=0.0, units='N*m', desc='The bending moment about the x axis')
        self.add_param('rotor_bending_moment_y', val=0.0, units='N*m', desc='The bending moment about the y axis')
        self.add_param('rotor_bending_moment_z', val=0.0, units='N*m', desc='The bending moment about the z axis')
        self.add_param('rotor_thrust', val=0.0, units='N', desc='The force along the x axis applied at hub center')
        self.add_param('rotor_force_y', val=0.0, units='N', desc='The force along the y axis applied at hub center')
        self.add_param('rotor_force_z', val=0.0, units='N', desc='The force along the z axis applied at hub center')
        self.add_param('rotor_mass', val=0.0, units='kg', desc='rotor mass')
        self.add_param('rotor_diameter', val=0.0, units='m', desc='rotor diameter')
        self.add_param('machine_rating', val=0.0, units='kW', desc='machine_rating machine rating of the turbine')
        self.add_param('gearbox_mass', val=0.0, units='kg', desc='Gearbox mass')
        self.add_param('carrier_mass', val=0.0, units='kg', desc='Carrier mass')
        self.add_param('overhang', val=0.0, units='m', desc='Overhang distance')
        self.add_param('distance_hub2mb', val=0.0, units='m', desc='distance between hub center and upwind main bearing')
        self.add_param('drivetrain_efficiency', val=0.0, desc='overall drivettrain efficiency')

        # parameters
        self.add_param('shrink_disc_mass', val=0.0, units='kg', desc='Mass of the shrink disc')
        self.add_param('gearbox_cm', val=np.array([0.0, 0.0, 0.0]), units='m', desc='center of mass of gearbox')
        self.add_param('gearbox_length', val=0.0, units='m', desc='gearbox length')
        self.add_param('flange_length', val=0.0, units='m', desc='flange length')
        self.add_param('shaft_angle', val=0.0, units='rad', desc='Angle of the LSS inclindation with respect to the horizontal')
        self.add_param('shaft_ratio', val=0.0, desc='Ratio of inner diameter to outer diameter.  Leave zero for solid LSS')
      
        # outputs
        self.add_output('lss_design_torque', val=0.0,  units='N*m', desc='lss design torque')
        self.add_output('lss_design_bending_load', val=0.0,  units='N', desc='lss design bending load')
        self.add_output('lss_length', val=0.0, units='m', desc='lss length')
        self.add_output('lss_diameter1', val=0.0, units='m',  desc='lss outer diameter at main bearing')
        self.add_output('lss_diameter2', val=0.0, units='m',  desc='lss outer diameter at second bearing')
        self.add_output('lss_mass', val=0.0, units='kg', desc='overall component mass')
        self.add_output('lss_cm', val=np.array([0.0, 0.0, 0.0]), desc='center of mass of the component in [x,y,z] for an arbitrary coordinate system')
        self.add_output('lss_I', val=np.array([0.0, 0.0, 0.0]), desc=' moments of Inertia for the component [Ixx, Iyy, Izz] around its center of mass')
        self.add_output('lss_mb1_facewidth', val=0.0, units='m',  desc='facewidth of upwind main bearing')
        self.add_output('lss_mb2_facewidth', val=0.0, units='m',  desc='facewidth of main bearing')
        self.add_output('lss_mb1_mass', val=0.0,  units='kg', desc='main bearing mass')
        self.add_output('lss_mb2_mass', val=0.0, units='kg',  desc='second bearing mass')
        self.add_output('lss_mb1_cm', val=np.array([0, 0, 0]), units='m', desc='main bearing 1 center of mass')
        self.add_output('lss_mb2_cm', val=np.array([0, 0, 0]), units='m', desc='main bearing 2 center of mass')

        self.lss4pt = LowSpeedShaft4pt(mb1Type, mb2Type, IEC_Class)

    def solve_nonlinear(self, inputs, outputs, resid):

        (outputs['lss_design_torque'], outputs['lss_design_bending_load'], outputs['lss_length'], outputs['lss_diameter1'], outputs['lss_diameter2'], outputs['lss_mass'], outputs['lss_cm'], outputs['lss_I'], \
         outputs['lss_mb1_facewidth'], outputs['lss_mb2_facewidth'], outputs['lss_mb1_mass'], outputs['lss_mb2_mass'], outputs['lss_mb1_cm'], outputs['lss_mb2_cm']) \
                = self.lss4pt.compute(inputs['rotor_diameter'], inputs['rotor_mass'], inputs['rotor_thrust'], inputs['rotor_force_y'], inputs['rotor_force_z'], \
                                    inputs['rotor_bending_moment_x'], inputs['rotor_bending_moment_y'], inputs['rotor_bending_moment_z'], \
                                    inputs['overhang'], inputs['machine_rating'], inputs['drivetrain_efficiency'], \
                                    inputs['gearbox_mass'], inputs['carrier_mass'], inputs['gearbox_cm'], inputs['gearbox_length'], \
                                    inputs['shrink_disc_mass'], inputs['flange_length'], inputs['distance_hub2mb'], inputs['shaft_angle'], inputs['shaft_ratio'])       

        return outputs

#-------------------------------------------------------------------------


class LowSpeedShaft3pt_OM(Component):
    ''' LowSpeedShaft class
          The LowSpeedShaft class is used to represent the low speed shaft component of a wind turbine drivetrain. 
          It contains the general properties for a wind turbine component as well as additional design load and dimentional attributes as listed below.
          It contains an update method to determine the mass, mass properties, and dimensions of the component.
    '''
    def __init__(self, mb1Type, IEC_Class):

        super(LowSpeedShaft3pt_OM, self).__init__()

        # variables
        self.add_param('rotor_bending_moment_x', val=0.0, units='N*m', desc='The bending moment about the x axis')
        self.add_param('rotor_bending_moment_y', val=0.0, units='N*m', desc='The bending moment about the y axis')
        self.add_param('rotor_bending_moment_z', val=0.0, units='N*m', desc='The bending moment about the z axis')
        self.add_param('rotor_thrust', val=0.0, units='N', desc='The force along the x axis applied at hub center')
        self.add_param('rotor_force_y', val=0.0, units='N', desc='The force along the y axis applied at hub center')
        self.add_param('rotor_force_z', val=0.0, units='N', desc='The force along the z axis applied at hub center')
        self.add_param('rotor_mass', val=0.0, units='kg', desc='rotor mass')
        self.add_param('rotor_diameter', val=0.0, units='m', desc='rotor diameter')
        self.add_param('machine_rating', val=0.0, units='kW', desc='machine_rating machine rating of the turbine')
        self.add_param('gearbox_mass', val=0.0, units='kg', desc='Gearbox mass')
        self.add_param('carrier_mass', val=0.0, units='kg', desc='Carrier mass')
        self.add_param('overhang', val=0.0, units='m', desc='Overhang distance')
        self.add_param('distance_hub2mb', val=0.0, units='m', desc='distance between hub center and upwind main bearing')
        self.add_param('drivetrain_efficiency', val=0.0, desc='overall drivettrain efficiency')

        # parameters
        self.add_param('shrink_disc_mass', val=0.0, units='kg', desc='Mass of the shrink disc')
        self.add_param('gearbox_cm', val=np.array([0.0, 0.0, 0.0]), units='m', desc='center of mass of gearbox')
        self.add_param('gearbox_length', val=0.0, units='m', desc='gearbox length')
        self.add_param('flange_length', val=0.0, units='m', desc='flange length')
        self.add_param('shaft_angle', val=0.0, units='rad', desc='Angle of the LSS inclindation with respect to the horizontal')
        self.add_param('shaft_ratio', val=0.0, desc='Ratio of inner diameter to outer diameter.  Leave zero for solid LSS')
        
        # outputs
        self.add_output('lss_design_torque', val=0.0,  units='N*m', desc='lss design torque')
        self.add_output('lss_design_bending_load', val=0.0,  units='N', desc='lss design bending load')
        self.add_output('lss_length', val=0.0, units='m', desc='lss length')
        self.add_output('lss_diameter1', val=0.0, units='m',  desc='lss outer diameter at main bearing')
        self.add_output('lss_diameter2', val=0.0, units='m',  desc='lss outer diameter at second bearing')
        self.add_output('lss_mass', val=0.0, units='kg', desc='overall component mass')
        self.add_output('lss_cm', val=np.array([0.0, 0.0, 0.0]), desc='center of mass of the component in [x,y,z] for an arbitrary coordinate system')
        self.add_output('lss_I', val=np.array([0.0, 0.0, 0.0]), desc=' moments of Inertia for the component [Ixx, Iyy, Izz] around its center of mass')
        self.add_output('lss_mb1_facewidth', val=0.0, units='m',  desc='facewidth of upwind main bearing')
        self.add_output('lss_mb2_facewidth', val=0.0, units='m',  desc='facewidth of main bearing')
        self.add_output('lss_mb1_mass', val=0.0,  units='kg', desc='main bearing mass')
        self.add_output('lss_mb2_mass', val=0.0, units='kg',  desc='second bearing mass')
        self.add_output('lss_mb1_cm', val=np.array([0, 0, 0]), units='m', desc='main bearing 1 center of mass')
        self.add_output('lss_mb2_cm', val=np.array([0, 0, 0]), units='m', desc='main bearing 2 center of mass')

        self.lss3pt = LowSpeedShaft3pt(mb1Type, IEC_Class)

    def solve_nonlinear(self, inputs, outputs, resid):

        (outputs['lss_design_torque'], outputs['lss_design_bending_load'], outputs['lss_length'], outputs['lss_diameter1'], outputs['lss_diameter2'], outputs['lss_mass'], outputs['lss_cm'], outputs['lss_I'], \
         outputs['lss_mb1_facewidth'], outputs['lss_mb2_facewidth'], outputs['lss_mb1_mass'], outputs['lss_mb2_mass'], outputs['lss_mb1_cm'], outputs['lss_mb2_cm']) \
                = self.lss3pt.compute(inputs['rotor_diameter'], inputs['rotor_mass'], inputs['rotor_thrust'], inputs['rotor_force_y'], inputs['rotor_force_z'], \
                                    inputs['rotor_bending_moment_x'], inputs['rotor_bending_moment_y'], inputs['rotor_bending_moment_z'], \
                                    inputs['overhang'], inputs['machine_rating'], inputs['drivetrain_efficiency'], \
                                    inputs['gearbox_mass'], inputs['carrier_mass'], inputs['gearbox_cm'], inputs['gearbox_length'], \
                                    inputs['shrink_disc_mass'], inputs['flange_length'], inputs['distance_hub2mb'], inputs['shaft_angle'], inputs['shaft_ratio'])       

        return outputs

#-------------------------------------------------------------------------

class MainBearing_OM(Component):
    ''' MainBearings class
          The MainBearings class is used to represent the main bearing components of a wind turbine drivetrain. It contains two subcomponents (main bearing and second bearing) which also inherit from the SubComponent class.
          It contains the general properties for a wind turbine component as well as additional design load and dimentional attributes as listed below.
          It contains an update method to determine the mass, mass properties, and dimensions of the component.
    '''

    def __init__(self, bearing_position):

        super(MainBearing_OM, self).__init__()

        # variables
        self.add_param('bearing_mass', val=0.0, units='kg', desc='bearing mass from LSS model')
        self.add_param('lss_diameter', val=0.0, units='m', desc='lss outer diameter at main bearing')
        self.add_param('lss_design_torque', val=0.0, units='N*m', desc='lss design torque')
        self.add_param('rotor_diameter', val=0.0, units='m', desc='rotor diameter')
        self.add_param('lss_mb_cm', val=np.array([0., 0., 0.]), units='m', desc='x,y,z location from shaft model')

        # returns
        self.add_output('mb_mass', val=0.0, units='kg', desc='overall component mass')
        self.add_output('mb_cm', val=np.array([0.0, 0.0, 0.0]), desc='center of mass of the component in [x,y,z] for an arbitrary coordinate system')
        self.add_output('mb_I', val=np.array([0.0, 0.0, 0.0]), desc=' moments of Inertia for the component [Ixx, Iyy, Izz] around its center of mass')

        self.mb = MainBearing(bearing_position)
        
    def solve_nonlinear(self, inputs, outputs, resid):

        (outputs['mb_mass'], outputs['mb_cm'], outputs['mb_I']) \
            = self.mb.compute(inputs['bearing_mass'], inputs['lss_diameter'], inputs['lss_design_torque'], inputs['rotor_diameter'], inputs['lss_mb_cm'])

        return outputs

#-------------------------------------------------------------------------

class Gearbox_OM(Component):
    ''' Gearbox class
          The Gearbox class is used to represent the gearbox component of a wind turbine drivetrain.
          It contains the general properties for a wind turbine component as well as additional design load and dimentional attributes as listed below.
          It contains an update method to determine the mass, mass properties, and dimensions of the component.
    '''

    def __init__(self, gear_configuration, shaft_factor):

        super(Gearbox_OM, self).__init__()

        # variables
        self.add_param('gear_ratio', val=0.0, desc='overall gearbox speedup ratio')
        self.add_param('planet_numbers', val=np.array([0.0, 0.0, 0.0,]), desc='number of planets in each stage')
        self.add_param('rotor_speed', val=0.0, units='rpm', desc='rotor rpm at rated power')
        self.add_param('rotor_diameter', val=0.0, desc='rotor diameter')
        self.add_param('rotor_torque', val=0.0, units='N*m', desc='rotor torque at rated power')
        self.add_param('gearbox_input_cm', val=0.00, units='m', desc='gearbox position along x-axis')

        # outputs
        self.add_output('stage_masses', val=np.array([0.0, 0.0, 0.0]), units='kg', desc='individual gearbox stage gearbox_masses')
        self.add_output('gearbox_mass', val=0.0, units='kg', desc='overall component gearbox_mass')
        self.add_output('gearbox_cm', val=np.array([0.0, 0.0, 0.0]), desc='center of gearbox_mass of the component in [x,y,z] for an arbitrary coordinate system')
        self.add_output('gearbox_I', val=np.array([0.0, 0.0, 0.0]), desc=' moments of gearbox_Inertia for the component [gearbox_Ixx, gearbox_Iyy, gearbox_Izz] around its center of gearbox_mass')
        self.add_output('gearbox_length', val=0.0, units='m', desc='gearbox length')
        self.add_output('gearbox_height', val=0.0, units='m', desc='gearbox height')
        self.add_output('gearbox_diameter', val=0.0, units='m', desc='gearbox diameter')

        self.gearbox = Gearbox(gear_configuration, shaft_factor)

    def solve_nonlinear(self, inputs, outputs, resid):
        
        (outputs['stage_masses'], outputs['gearbox_mass'], outputs['gearbox_cm'], outputs['gearbox_I'], outputs['gearbox_length'], outputs['gearbox_height'], outputs['gearbox_diameter']) \
            = self.gearbox.compute(inputs['gear_ratio'], inputs['planet_numbers'], inputs['rotor_speed'], inputs['rotor_diameter'], inputs['rotor_torque'], inputs['gearbox_input_cm'])

        return outputs



#-------------------------------------------------------------------

class HighSpeedSide_OM(Component):
    '''
    HighSpeedShaft class
          The HighSpeedShaft class is used to represent the high speed shaft and mechanical brake components of a wind turbine drivetrain.
          It contains the general properties for a wind turbine component as well as additional design load and dimentional attributes as listed below.
          It contains an update method to determine the mass, mass properties, and dimensions of the component.
    '''

    def __init__(self):

        super(HighSpeedSide_OM, self).__init__()

        # variables
        self.add_param('rotor_diameter', val=0.0, units='m', desc='rotor diameter')
        self.add_param('rotor_torque', val=0.0, units='N*m', desc='rotor torque at rated power')
        self.add_param('gear_ratio', val=0.0, desc='overall gearbox ratio')
        self.add_param('lss_diameter', val=0.0, units='m', desc='low speed shaft outer diameter')
        self.add_param('gearbox_length', val=0.0, units='m', desc='gearbox length')
        self.add_param('gearbox_height', val=0.0, units='m', desc='gearbox height')
        self.add_param('gearbox_cm', val=np.array([0.0, 0.0, 0.0]), units='m', desc='gearbox cm [x,y,z]')
        self.add_param('hss_input_length', val=0.0, units='m', desc='high speed shaft length determined by user. Default 0.5m')

        # returns
        self.add_output('hss_mass', val=0.0, units='kg', desc='overall component mass')
        self.add_output('hss_cm', val=np.array([0.0, 0.0, 0.0]), desc='center of mass of the component in [x,y,z] for an arbitrary coordinate system')
        self.add_output('hss_I', val=np.array([0.0, 0.0, 0.0]), desc=' moments of Inertia for the component [Ixx, Iyy, Izz] around its center of mass')
        self.add_output('hss_length', val=0.0, desc='length of high speed shaft')

        self.hss = HighSpeedSide()

    def solve_nonlinear(self, inputs, outputs, resid):

        (outputs['hss_mass'], outputs['hss_cm'], outputs['hss_I'], outputs['hss_length']) \
            = self.hss.compute(inputs['rotor_diameter'], inputs['rotor_torque'], inputs['gear_ratio'], inputs['lss_diameter'], inputs['gearbox_length'], inputs['gearbox_height'], inputs['gearbox_cm'], inputs['hss_input_length'])

        return outputs

#----------------------------------------------------------------------------------------------

class Generator_OM(Component):
    '''Generator class
          The Generator class is used to represent the generator of a wind turbine drivetrain.
          It contains the general properties for a wind turbine component as well as additional design load and dimentional attributes as listed below.
          It contains an update method to determine the mass, mass properties, and dimensions of the component.
    '''

    def __init__(self, drivetrain_design):

        super(Generator_OM, self).__init__()

        # variables
        self.add_param('rotor_diameter', val=0.0, units='m', desc='rotor diameter')
        self.add_param('machine_rating', val=0.0, units='kW', desc='machine rating of generator')
        self.add_param('gear_ratio', val=0.0, desc='overall gearbox ratio')
        self.add_param('hss_length', val=0.0, units='m', desc='length of high speed shaft and brake')
        self.add_param('hss_cm', val=np.array([0.0,0.0,0.0]), units='m', desc='cm of high speed shaft and brake')
        self.add_param('rotor_speed', val=0.0, units='rpm', desc='Speed of rotor at rated power')

        #returns
        self.add_output('generator_mass', val=0.0, units='kg', desc='overall component mass')
        self.add_output('generator_cm', val=np.array([0.0, 0.0, 0.0]), desc='center of mass of the component in [x,y,z] for an arbitrary coordinate system')
        self.add_output('generator_I', val=np.array([0.0, 0.0, 0.0]), desc=' moments of Inertia for the component [Ixx, Iyy, Izz] around its center of mass')

        self.gen = Generator(drivetrain_design)
        
    def solve_nonlinear(self, inputs, outputs, resid):

        (outputs['generator_mass'], outputs['generator_cm'], outputs['generator_I']) \
            = self.gen.compute(inputs['rotor_diameter'], inputs['machine_rating'], inputs['gear_ratio'], inputs['hss_length'], inputs['hss_cm'], inputs['rotor_speed'])

        return outputs

#--------------------------------------------
class RNASystemAdder_OM(Component):
    ''' RNASystem class
          This analysis is only to be used in placing the transformer of the drivetrain.
          The Rotor-Nacelle-Group class is used to represent the RNA of the turbine without the transformer and bedplate (to resolve circular dependency issues).
          It contains the general properties for a wind turbine component as well as additional design load and dimentional attributes as listed below.
          It contains an update method to determine the mass, mass properties, and dimensions of the component. 
    '''

    def __init__(self):

        super(RNASystemAdder_OM, self).__init__()

        # inputs
        self.add_param('lss_mass', val=0.0, units='kg', desc='component mass')
        self.add_param('mb1_mass', val=0.0, units='kg', desc='component mass')
        self.add_param('mb2_mass', val=0.0, units='kg', desc='component mass')
        self.add_param('gearbox_mass', val=0.0, units='kg', desc='component mass')
        self.add_param('hss_mass', val=0.0, units='kg', desc='component mass')
        self.add_param('generator_mass', val=0.0, units='kg', desc='component mass')
        self.add_param('lss_cm', val=np.array([0.0,0.0,0.0]), units='m', desc='component CM')
        self.add_param('mb1_cm', val=np.array([0.0,0.0,0.0]), units='m', desc='component CM')
        self.add_param('mb2_cm', val=np.array([0.0,0.0,0.0]), units='m', desc='component CM')
        self.add_param('gearbox_cm', val=np.array([0.0,0.0,0.0]), units='m', desc='component CM')
        self.add_param('hss_cm', val=np.array([0.0,0.0,0.0]), units='m', desc='component CM')
        self.add_param('generator_cm', val=np.array([0.0,0.0,0.0]), units='m', desc='component CM')
        self.add_param('overhang', val=0.0, units='m', desc='nacelle overhang')
        self.add_param('rotor_mass', val=0.0, units='kg', desc='component mass')
        self.add_param('machine_rating', val=0.0, units='kW', desc='machine rating')

        # returns
        self.add_output('RNA_mass', val=0.0, units='kg', desc='mass of total RNA')
        self.add_output('RNA_cm', val=0.0, units='m', desc='RNA CM along x-axis')
        
        self.rnaadder = RNASystemAdder()
        
    def solve_nonlinear(self, inputs, outputs, resid):

        (outputs['RNA_mass'], outputs['RNA_cm']) \
                    = self.rnaadder.compute(inputs['lss_mass'], inputs['mb1_mass'], inputs['mb2_mass'], inputs['gearbox_mass'], inputs['hss_mass'], inputs['generator_mass'], \
                      inputs['lss_cm'], inputs['mb1_cm'], inputs['mb2_cm'], inputs['gearbox_cm'], inputs['hss_cm'], inputs['generator_cm'], inputs['overhang'], inputs['rotor_mass'], inputs['machine_rating'])

        return outputs
        

#-------------------------------------------------------------------------------

class Transformer_OM(Component):
    ''' Transformer class
            The transformer class is used to represent the transformer of a wind turbine drivetrain.
            It contains the general properties for a wind turbine component as well as additional design load and dimentional attributes as listed below.
            It contains an update method to determine the mass, mass properties, and dimensions of the component if it is in fact uptower'''

    def __init__(self, uptower_transformer):

        super(Transformer_OM, self).__init__()

        # inputs
        self.add_param('machine_rating', val=0.0, units='kW', desc='machine rating of the turbine')
        self.add_param('tower_top_diameter', val=0.0, units='m', desc='tower top diameter for comparision of nacelle CM')
        self.add_param('rotor_mass', val=0.0, units='kg', desc='rotor mass')
        self.add_param('overhang', val=0.0, units='m', desc='rotor overhang distance')
        self.add_param('generator_cm', val=np.array([0.0, 0.0, 0.0]), desc='center of mass of the generator in [x,y,z]')
        self.add_param('rotor_diameter', val=0.0, units='m', desc='rotor diameter of turbine')
        self.add_param('RNA_mass', val=0.0, units='kg', desc='mass of total RNA')
        self.add_param('RNA_cm', val=0.0, units='m', desc='RNA CM along x-axis')

        # outputs
        self.add_output('transformer_mass', val=0.0, units='kg', desc='overall component mass')
        self.add_output('transformer_cm', val=np.array([0.0, 0.0, 0.0]), desc='center of mass of the component in [x,y,z] for an arbitrary coordinate system')
        self.add_output('transformer_I', val=np.array([0.0, 0.0, 0.0]), desc=' moments of Inertia for the component [Ixx, Iyy, Izz] around its center of mass')    

        self.transformer = Transformer(uptower_transformer)

    def solve_nonlinear(self, inputs, outputs, resid):

        (outputs['transformer_mass'], outputs['transformer_cm'], outputs['transformer_I']) \
            = self.transformer.compute(inputs['machine_rating'], inputs['tower_top_diameter'], inputs['rotor_mass'], inputs['overhang'], inputs['generator_cm'], inputs['rotor_diameter'], inputs['RNA_mass'], inputs['RNA_cm'])

        return outputs


#-------------------------------------------------------------------------

class Bedplate_OM(Component):
    ''' Bedplate class
          The Bedplate class is used to represent the bedplate of a wind turbine drivetrain.
          It contains the general properties for a wind turbine component as well as additional design load and dimentional attributes as listed below.
          It contains an update method to determine the mass, mass properties, and dimensions of the component.
    '''

    def __init__(self, uptower_transformer):

        super(Bedplate_OM, self).__init__()

        # variables
        self.add_param('gearbox_length', val=0.0, units='m', desc='gearbox length')
        self.add_param('gearbox_location', val=0.0, units='m', desc='gearbox CM location')
        self.add_param('gearbox_mass', val=0.0, units='kg', desc='gearbox mass')
        self.add_param('hss_location', val=0.0, units='m', desc='HSS CM location')
        self.add_param('hss_mass', val=0.0, units='kg', desc='HSS mass')
        self.add_param('generator_location', val=0.0, units='m', desc='generator CM location')
        self.add_param('generator_mass', val=0.0, units='kg', desc='generator mass')
        self.add_param('lss_location', val=0.0, units='m', desc='LSS CM location')
        self.add_param('lss_mass', val=0.0, units='kg', desc='LSS mass')
        self.add_param('lss_length', val=0.0, units='m', desc='LSS length')
        self.add_param('lss_mb1_facewidth', val=0.0, units='m', desc='Upwind main bearing facewidth')
        self.add_param('mb1_cm', val=np.array([0.0, 0.0, 0.0]), units='m', desc='Upwind main bearing CM location')
        self.add_param('mb1_mass', val=0.0, units='kg', desc='Upwind main bearing mass')
        self.add_param('mb2_cm', val=np.array([0.0, 0.0, 0.0]), units='m', desc='Downwind main bearing CM location')
        self.add_param('mb2_mass', val=0.0, units='kg', desc='Downwind main bearing mass')
        self.add_param('transformer_mass', val=0.0, units='kg', desc='Transformer mass')
        self.add_param('transformer_cm', val=np.array([0.0, 0.0, 0.0]), units='m', desc='transformer CM location')
        self.add_param('tower_top_diameter', val=0.0, units='m', desc='diameter of the top tower section at the yaw gear')
        self.add_param('rotor_diameter', val=0.0, units='m', desc='rotor diameter')
        self.add_param('machine_rating', val=0.0, units='kW', desc='machine_rating machine rating of the turbine')
        self.add_param('rotor_mass', val=0.0, units='kg', desc='rotor mass')
        self.add_param('rotor_bending_moment_y', val=0.0, units='N*m', desc='The bending moment about the y axis')
        self.add_param('rotor_force_z', val=0.0, units='N', desc='The force along the z axis applied at hub center')
        self.add_param('flange_length', val=0.0, units='m', desc='flange length')
        self.add_param('distance_hub2mb', val=0.0, units='m', desc='length between rotor center and upwind main bearing')
        self.add_param('overhang', val=0.0, units='m', desc='Overhang distance')

        # outputs
        self.add_output('bedplate_mass', val=0.0, units='kg', desc='overall component bedplate_mass')
        self.add_output('bedplate_cm', val=np.array([0.0, 0.0, 0.0]), desc='center of bedplate_mass of the component in [x,y,z] for an arbitrary coordinate system')
        self.add_output('bedplate_I', val=np.array([0.0, 0.0, 0.0]), desc=' moments of Inertia for the component [Ixx, Iyy, Izz] around its center of bedplate_mass')
        self.add_output('bedplate_length', val=0.0, units='m', desc='length of bedplate')
        self.add_output('bedplate_height', val=0.0, units='m',  desc='max height of bedplate')
        self.add_output('bedplate_width', val=0.0, units='m', desc='width of bedplate')
        
        self.bpl = Bedplate(uptower_transformer)

    def solve_nonlinear(self, inputs, outputs, resid):

        (outputs['bedplate_mass'], outputs['bedplate_cm'], outputs['bedplate_I'], outputs['bedplate_length'], outputs['bedplate_height'], outputs['bedplate_width']) \
            = self.bpl.compute(inputs['gearbox_length'], inputs['gearbox_location'], inputs['gearbox_mass'], inputs['hss_location'], inputs['hss_mass'], inputs['generator_location'], inputs['generator_mass'], \
                      inputs['lss_location'], inputs['lss_mass'], inputs['lss_length'], inputs['mb1_cm'], inputs['lss_mb1_facewidth'], inputs['mb1_mass'], inputs['mb2_cm'], inputs['mb2_mass'], \
                      inputs['transformer_mass'], inputs['transformer_cm'], \
                      inputs['tower_top_diameter'], inputs['rotor_diameter'], inputs['machine_rating'], inputs['rotor_mass'], inputs['rotor_bending_moment_y'], inputs['rotor_force_z'], \
                      inputs['flange_length'], inputs['distance_hub2mb'], inputs['overhang'])

        return outputs

#-------------------------------------------------------------------------------

class AboveYawMassAdder_OM(Component):

    def __init__(self, crane):

        super(AboveYawMassAdder_OM, self).__init__()

        # variables
        self.add_param('machine_rating', val=0.0, units='kW', desc='machine rating')
        self.add_param('lss_mass', val=0.0, units='kg', desc='component mass')
        self.add_param('mb1_mass', val=0.0, units='kg', desc='component mass')
        self.add_param('mb2_mass', val=0.0, units='kg', desc='component mass')
        self.add_param('gearbox_mass', val=0.0, units='kg', desc='component mass')
        self.add_param('hss_mass', val=0.0, units='kg', desc='component mass')
        self.add_param('generator_mass', val=0.0, units='kg', desc='component mass')
        self.add_param('bedplate_mass', val=0.0, units='kg', desc='component mass')
        self.add_param('bedplate_length', val=0.0, units='m', desc='component length')
        self.add_param('bedplate_width', val=0.0, units='m', desc='component width')
        self.add_param('transformer_mass', val=0.0, units='kg', desc='component mass')

        # returns
        self.add_output('electrical_mass', val=0.0, units='kg', desc='component mass')
        self.add_output('vs_electronics_mass', val=0.0, units='kg', desc='component mass')
        self.add_output('hvac_mass', val=0.0, units='kg', desc='component mass')
        self.add_output('controls_mass', val=0.0, units='kg', desc='component mass')
        self.add_output('platforms_mass', val=0.0, units='kg', desc='component mass')
        self.add_output('crane_mass', val=0.0, units='kg', desc='component mass')
        self.add_output('mainframe_mass', val=0.0, units='kg', desc='component mass')
        self.add_output('cover_mass', val=0.0, units='kg', desc='component mass')
        self.add_output('above_yaw_mass', val=0.0, units='kg', desc='total mass above yaw system')
        self.add_output('nacelle_length', val=0.0, units='m', desc='component length')
        self.add_output('nacelle_width', val=0.0, units='m', desc='component width')
        self.add_output('nacelle_height', val=0.0, units='m', desc='component height')
        
        self.aboveyawmass = AboveYawMassAdder(crane)

    def solve_nonlinear(self, inputs, outputs, resid):

        (outputs['electrical_mass'], outputs['vs_electronics_mass'], outputs['hvac_mass'], outputs['controls_mass'], outputs['platforms_mass'], outputs['crane_mass'], \
               outputs['mainframe_mass'], outputs['cover_mass'], outputs['above_yaw_mass'], outputs['nacelle_length'], outputs['nacelle_width'], outputs['nacelle_height']) \
            = self.aboveyawmass.compute(inputs['machine_rating'], inputs['lss_mass'], inputs['mb1_mass'], inputs['mb2_mass'], inputs['gearbox_mass'], \
                      inputs['hss_mass'], inputs['generator_mass'], inputs['bedplate_mass'], inputs['bedplate_length'], inputs['bedplate_width'], inputs['transformer_mass'])

        return outputs

#---------------------------------------------------------------------------------------------------------------

class YawSystem_OM(Component):
    ''' YawSystem class
          The YawSystem class is used to represent the yaw system of a wind turbine drivetrain.
          It contains the general properties for a wind turbine component as well as additional design load and dimentional attributes as listed below.
          It contains an update method to determine the mass, mass properties, and dimensions of the component.
    '''

    def __init__(self, yaw_motors_number):

        super(YawSystem_OM, self).__init__()

        # variables
        self.add_param('rotor_diameter', val=0.0, units='m', desc='rotor diameter')
        self.add_param('rotor_thrust', val=0.0, units='N', desc='maximum rotor thrust')
        self.add_param('tower_top_diameter', val=0.0, units='m', desc='tower top diameter')
        self.add_param('above_yaw_mass', val=0.0, units='kg', desc='above yaw mass')
        self.add_param('bedplate_height', val=0.0, units='m', desc='bedplate height')

        # outputs
        self.add_output('yaw_mass', val=0.0, units='kg', desc='overall component mass')
        self.add_output('yaw_cm', val=np.array([0.0, 0.0, 0.0]), desc='center of mass of the component in [x,y,z] for an arbitrary coordinate system')
        self.add_output('yaw_I', val=np.array([0.0, 0.0, 0.0]), desc=' moments of Inertia for the component [Ixx, Iyy, Izz] around its center of mass')    

        self.yaw = YawSystem(yaw_motors_number)

    def solve_nonlinear(self, inputs, outputs, resid):

        (outputs['yaw_mass'], outputs['yaw_cm'], outputs['yaw_I']) \
            = self.yaw.compute(inputs['rotor_diameter'], inputs['rotor_thrust'], inputs['tower_top_diameter'], inputs['above_yaw_mass'], inputs['bedplate_height'])

        return outputs


#--------------------------------------------
class NacelleSystemAdder_OM(Component): #added to drive to include transformer
    ''' NacelleSystem class
          The Nacelle class is used to represent the overall nacelle of a wind turbine.
          It contains the general properties for a wind turbine component as well as additional design load and dimentional attributes as listed below.
          It contains an update method to determine the mass, mass properties, and dimensions of the component.
    '''

    def __init__(self):
        super(NacelleSystemAdder_OM, self).__init__()

        # variables
        self.add_param('above_yaw_mass', val=0.0, units='kg', desc='mass above yaw system')
        self.add_param('yaw_mass', val=0.0, units='kg', desc='mass of yaw system')
        self.add_param('lss_mass', val=0.0, units='kg', desc='component mass')
        self.add_param('mb1_mass', val=0.0, units='kg', desc='component mass')
        self.add_param('mb2_mass', val=0.0, units='kg', desc='component mass')
        self.add_param('gearbox_mass', val=0.0, units='kg', desc='component mass')
        self.add_param('hss_mass', val=0.0, units='kg', desc='component mass')
        self.add_param('generator_mass', val=0.0, units='kg', desc='component mass')
        self.add_param('bedplate_mass', val=0.0, units='kg', desc='component mass')
        self.add_param('mainframe_mass', val=0.0, units='kg', desc='component mass')
        self.add_param('lss_cm', val=np.array([0.0,0.0,0.0]), units='m', desc='component CM')
        self.add_param('mb1_cm', val=np.array([0.0,0.0,0.0]), units='m', desc='component CM')
        self.add_param('mb2_cm', val=np.array([0.0,0.0,0.0]), units='m', desc='component CM')
        self.add_param('gearbox_cm', val=np.array([0.0,0.0,0.0]), units='m', desc='component CM')
        self.add_param('hss_cm', val=np.array([0.0,0.0,0.0]), units='m', desc='component CM')
        self.add_param('generator_cm', val=np.array([0.0,0.0,0.0]), units='m', desc='component CM')
        self.add_param('bedplate_cm', val=np.array([0.0,0.0,0.0]), units='m', desc='component CM')
        self.add_param('lss_I', val=np.array([0.0,0.0,0.0]), units='kg*m**2', desc='component I')
        self.add_param('mb1_I', val=np.array([0.0,0.0,0.0]), units='kg*m**2', desc='component I')
        self.add_param('mb2_I', val=np.array([0.0,0.0,0.0]), units='kg*m**2', desc='component I')
        self.add_param('gearbox_I', val=np.array([0.0,0.0,0.0]), units='kg*m**2', desc='component I')
        self.add_param('hss_I', val=np.array([0.0,0.0,0.0]), units='kg*m**2', desc='component I')
        self.add_param('generator_I', val=np.array([0.0,0.0,0.0]), units='kg*m**2', desc='component I')
        self.add_param('bedplate_I', val=np.array([0.0,0.0,0.0]), units='kg*m**2', desc='component I')
        self.add_param('transformer_mass', val=0.0, units='kg', desc='component mass')
        self.add_param('transformer_cm', val=np.array([0.0,0.0,0.0]), units='m', desc='component CM')
        self.add_param('transformer_I', val=np.array([0.0,0.0,0.0]), units='kg*m**2', desc='component I')

        # returns
        self.add_output('nacelle_mass', val=0.0, units='kg', desc='overall component mass')
        self.add_output('nacelle_cm', val=np.array([0.0, 0.0, 0.0]), units='m', desc='center of mass of the component in [x,y,z] for an arbitrary coordinate system')
        self.add_output('nacelle_I', val=np.array([0.0, 0.0, 0.0]), units='kg*m**2', desc=' moments of Inertia for the component [Ixx, Iyy, Izz] around its center of mass')

        self.nacelleadder = NacelleSystemAdder()
        
    def solve_nonlinear(self, inputs, outputs, resid):

        (outputs['nacelle_mass'], outputs['nacelle_cm'], outputs['nacelle_I']) \
                    = self.nacelleadder.compute(inputs['above_yaw_mass'], inputs['yaw_mass'], inputs['lss_mass'], inputs['mb1_mass'], inputs['mb2_mass'], inputs['gearbox_mass'], \
                      inputs['hss_mass'], inputs['generator_mass'], inputs['bedplate_mass'], inputs['mainframe_mass'], \
                      inputs['lss_cm'], inputs['mb1_cm'], inputs['mb2_cm'], inputs['gearbox_cm'], inputs['hss_cm'], inputs['generator_cm'], inputs['bedplate_cm'], \
                      inputs['lss_I'], inputs['mb1_I'], inputs['mb2_I'], inputs['gearbox_I'], inputs['hss_I'], inputs['generator_I'], inputs['bedplate_I'], \
                      inputs['transformer_mass'], inputs['transformer_cm'], inputs['transformer_I'])

        return outputs

if __name__ == '__main__':

    '''TODO: add examples'''
    
    pass
