"""
DriveSE.py

Created by Yi Guo, Taylor Parsons and Ryan King 2014.
Copyright (c) NREL. All rights reserved.
"""

import numpy as np

from drivese.drivese_components import LowSpeedShaft4pt, LowSpeedShaft3pt, Gearbox, MainBearing, Bedplate, YawSystem, \
                                       Transformer, HighSpeedSide, Generator, NacelleSystemAdder, AboveYawMassAdder, RNASystemAdder
from drivese.hubse_omdao import HubSE, HubMassOnlySE, Hub_CM_Adder_OM
from openmdao.api import Group, Component, IndepVarComp, Problem



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
        self.add_param('planet_numbers', val=np.array([0, 0, 0,]), desc='number of planets in each stage', pass_by_obj=True)
        self.add_param('rotor_rpm', val=0.0, units='rpm', desc='rotor rpm at rated power')
        self.add_param('rotor_diameter', val=0.0, units='m', desc='rotor diameter')
        self.add_param('rotor_torque', val=0.0, units='N*m', desc='rotor torque at rated power')
        self.add_param('gearbox_input_xcm', val=0.00, units='m', desc='gearbox position along x-axis')

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
            = self.gearbox.compute(inputs['gear_ratio'], inputs['planet_numbers'], inputs['rotor_rpm'], inputs['rotor_diameter'], inputs['rotor_torque'], inputs['gearbox_input_xcm'])

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
        self.add_param('rotor_rpm', val=0.0, units='rpm', desc='Speed of rotor at rated power')

        #returns
        self.add_output('generator_mass', val=0.0, units='kg', desc='overall component mass')
        self.add_output('generator_cm', val=np.array([0.0, 0.0, 0.0]), desc='center of mass of the component in [x,y,z] for an arbitrary coordinate system')
        self.add_output('generator_I', val=np.array([0.0, 0.0, 0.0]), desc=' moments of Inertia for the component [Ixx, Iyy, Izz] around its center of mass')

        self.gen = Generator(drivetrain_design)
        
    def solve_nonlinear(self, inputs, outputs, resid):

        (outputs['generator_mass'], outputs['generator_cm'], outputs['generator_I']) \
            = self.gen.compute(inputs['rotor_diameter'], inputs['machine_rating'], inputs['gear_ratio'], inputs['hss_length'], inputs['hss_cm'], inputs['rotor_rpm'])

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
            = self.transformer.compute(inputs['machine_rating'], inputs['tower_top_diameter'], inputs['rotor_mass'], inputs['generator_cm'], inputs['rotor_diameter'], inputs['RNA_mass'], inputs['RNA_cm'])

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
                      inputs['flange_length'], inputs['distance_hub2mb'])

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
        # print(inputs['machine_rating'], inputs['lss_mass'], inputs['mb1_mass'], inputs['mb2_mass'], inputs['gearbox_mass'],
        #       inputs['hss_mass'], inputs['generator_mass'], inputs['bedplate_mass'], inputs['bedplate_length'],
        #       inputs['bedplate_width'], inputs['transformer_mass'])
        # print(outputs['electrical_mass'], outputs['vs_electronics_mass'], outputs['hvac_mass'], outputs['controls_mass'],
        #       outputs['platforms_mass'], outputs['crane_mass'], outputs['mainframe_mass'], outputs['cover_mass'],
        #       outputs['above_yaw_mass'], outputs['nacelle_length'], outputs['nacelle_width'], outputs['nacelle_height'])

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
        self.add_output('nacelle_cm', val=np.zeros(3), units='m', desc='center of mass of the component in [x,y,z] for an arbitrary coordinate system')
        self.add_output('nacelle_I', val=np.zeros(6), units='kg*m**2', desc=' moments of Inertia for the component [Ixx, Iyy, Izz] around its center of mass')

        self.nacelleadder = NacelleSystemAdder()
        
    def solve_nonlinear(self, inputs, outputs, resid):

        (outputs['nacelle_mass'], outputs['nacelle_cm'], outputs['nacelle_I']) \
                    = self.nacelleadder.compute(inputs['above_yaw_mass'], inputs['yaw_mass'], inputs['lss_mass'], inputs['mb1_mass'], inputs['mb2_mass'], inputs['gearbox_mass'], \
                      inputs['hss_mass'], inputs['generator_mass'], inputs['bedplate_mass'], inputs['mainframe_mass'], \
                      inputs['lss_cm'], inputs['mb1_cm'], inputs['mb2_cm'], inputs['gearbox_cm'], inputs['hss_cm'], inputs['generator_cm'], inputs['bedplate_cm'], \
                      inputs['lss_I'], inputs['mb1_I'], inputs['mb2_I'], inputs['gearbox_I'], inputs['hss_I'], inputs['generator_I'], inputs['bedplate_I'], \
                      inputs['transformer_mass'], inputs['transformer_cm'], inputs['transformer_I'])

        return outputs

#-------------------------------------------------------------------------
# Assemblies
#-------------------------------------------------------------------------
    
class Drive3pt(Group):

    def __init__(self, mb1Type, IEC_Class, gear_configuration, shaft_factor, drivetrain_design, uptower_transformer, yaw_motors_number, crane, blade_number):
        super(Drive3pt, self).__init__()


        # Add common inputs for rotor
        #self.add('rotor_diameter', IndepVarComp('rotor_diameter', 0.0), promotes=['*'])
        #self.add('rotor_bending_moment_x', IndepVarComp('rotor_bending_moment_x', 0.0), promotes=['*'])
        #self.add('rotor_bending_moment_y', IndepVarComp('rotor_bending_moment_y', 0.0), promotes=['*'])
        #self.add('rotor_bending_moment_z', IndepVarComp('rotor_bending_moment_z', 0.0), promotes=['*'])
        #self.add('rotor_thrust', IndepVarComp('rotor_thrust', 0.0), promotes=['*'])
        #self.add('rotor_force_y', IndepVarComp('rotor_force_y', 0.0), promotes=['*'])
        #self.add('rotor_force_z', IndepVarComp('rotor_force_z', 0.0), promotes=['*'])
        #self.add('rotor_torque', IndepVarComp('rotor_torque', 0.0), promotes=['*'])
        #self.add('rotor_rpm', IndepVarComp('rotor_rpm', 0.0), promotes=['*'])
        #self.add('machine_rating',IndepVarComp('machine_rating', 0.0), promotes=['*'])

        # Add common inputs for drivetrain
        self.add('gear_ratio', IndepVarComp('gear_ratio', 0.0), promotes=['*'])
        self.add('shaft_angle', IndepVarComp('shaft_angle', 0.0), promotes=['*'])
        self.add('shaft_ratio', IndepVarComp('shaft_ratio', 0.0), promotes=['*'])
        self.add('shrink_disc_mass', IndepVarComp('shrink_disc_mass', 0.0), promotes=['*'])
        self.add('carrier_mass', IndepVarComp('carrier_mass', 0.0), promotes=['*'])
        self.add('flange_length', IndepVarComp('flange_length', 0.0), promotes=['*'])
        self.add('overhang', IndepVarComp('overhang', 0.0), promotes=['*'])
        self.add('distance_hub2mb', IndepVarComp('distance_hub2mb', 0.0), promotes=['*'])
        self.add('gearbox_input_xcm', IndepVarComp('gearbox_input_xcm', 0.0), promotes=['*'])
        self.add('hss_input_length', IndepVarComp('hss_input_length', 0.0), promotes=['*'])
        self.add('planet_numbers', IndepVarComp('planet_numbers', np.array([0, 0, 0]), pass_by_obj=True), promotes=['*'])
        #self.add('drivetrain_efficiency', IndepVarComp('drivetrain_efficiency', 0.0), promotes=['*'])

        # Add common inputs for tower
        self.add('tower_top_diameter',IndepVarComp([('tower_top_diameter', 0.0)]), promotes=['*'])

        # Create 3 pt drivetrain group
        self.add('hub', HubMassOnlySE(blade_number), promotes=['*'])
        self.add('lowSpeedShaft', LowSpeedShaft3pt_OM(mb1Type, IEC_Class), promotes=['*'])
        self.add('mainBearing', MainBearing_OM('main'), promotes=['lss_design_torque','rotor_diameter']) #need to make explicit connections for main bearing
        self.add('hubCM', Hub_CM_Adder_OM(), promotes=['*'])
        self.add('gearbox', Gearbox_OM(gear_configuration, shaft_factor), promotes=['*'])
        self.add('highSpeedSide', HighSpeedSide_OM(), promotes=['*'])
        self.add('generator', Generator_OM(drivetrain_design), promotes=['*'])
        self.add('bedplate', Bedplate_OM(uptower_transformer), promotes=['*'])
        self.add('transformer', Transformer_OM(uptower_transformer), promotes=['*'])
        self.add('rna', RNASystemAdder_OM(), promotes=['*'])
        self.add('above_yaw_massAdder', AboveYawMassAdder_OM(crane), promotes=['*'])
        self.add('yawSystem', YawSystem_OM(yaw_motors_number), promotes=['*'])
        self.add('nacelleSystem', NacelleSystemAdder_OM(), promotes=['*'])

        # Connect components where explicit connections needed (for main bearing)
        self.connect('lss_mb1_mass', ['mainBearing.bearing_mass'])
        self.connect('lss_diameter1', ['mainBearing.lss_diameter', 'lss_diameter'])
        self.connect('lss_mb1_cm', ['mainBearing.lss_mb_cm'])
        self.connect('mainBearing.mb_mass', ['mb1_mass'])
        self.connect('mainBearing.mb_cm', ['mb1_cm', 'MB1_location'])
        self.connect('mainBearing.mb_I', ['mb1_I'])

        self.connect('lss_cm','lss_location',src_indices=[0])
        self.connect('hss_cm','hss_location',src_indices=[0])
        self.connect('gearbox_cm','gearbox_location',src_indices=[0])
        self.connect('generator_cm','generator_location',src_indices=[0])

#------------------------------------------------------------------
class Drive4pt(Group):

    def __init__(self, mb1Type, mb2Type, IEC_Class, gear_configuration, shaft_factor, drivetrain_design, uptower_transformer, yaw_motors_number, crane, blade_number):
        super(Drive4pt, self).__init__()

        # Add common inputs for rotor
        #self.add('rotor_diameter', IndepVarComp('rotor_diameter', 0.0), promotes=['*'])
        #self.add('rotor_bending_moment_x', IndepVarComp('rotor_bending_moment_x', 0.0), promotes=['*'])
        #self.add('rotor_bending_moment_y', IndepVarComp('rotor_bending_moment_y', 0.0), promotes=['*'])
        #self.add('rotor_bending_moment_z', IndepVarComp('rotor_bending_moment_z', 0.0), promotes=['*'])
        #self.add('rotor_thrust', IndepVarComp('rotor_thrust', 0.0), promotes=['*'])
        #self.add('rotor_force_y', IndepVarComp('rotor_force_y', 0.0), promotes=['*'])
        #self.add('rotor_force_z', IndepVarComp('rotor_force_z', 0.0), promotes=['*'])
        #self.add('rotor_torque', IndepVarComp('rotor_torque', 0.0), promotes=['*'])
        #self.add('rotor_rpm', IndepVarComp('rotor_rpm', 0.0), promotes=['*'])
        #self.add('machine_rating',IndepVarComp('machine_rating', 0.0), promotes=['*'])

        # Add common inputs for drivetrain
        self.add('gear_ratio', IndepVarComp('gear_ratio', 0.0), promotes=['*'])
        self.add('shaft_angle', IndepVarComp('shaft_angle', 0.0), promotes=['*'])
        self.add('shaft_ratio', IndepVarComp('shaft_ratio', 0.0), promotes=['*'])
        self.add('shrink_disc_mass', IndepVarComp('shrink_disc_mass', 0.0), promotes=['*'])
        self.add('carrier_mass', IndepVarComp('carrier_mass', 0.0), promotes=['*'])
        self.add('flange_length', IndepVarComp('flange_length', 0.0), promotes=['*'])
        self.add('overhang', IndepVarComp('overhang', 0.0), promotes=['*'])
        self.add('distance_hub2mb', IndepVarComp('distance_hub2mb', 0.0), promotes=['*'])
        self.add('gearbox_input_xcm', IndepVarComp('gearbox_input_xcm', 0.0), promotes=['*'])
        self.add('hss_input_length', IndepVarComp('hss_input_length', 0.0), promotes=['*'])
        self.add('planet_numbers', IndepVarComp('planet_numbers', np.array([0, 0, 0]), pass_by_obj=True), promotes=['*'])
        #self.add('drivetrain_efficiency', IndepVarComp('drivetrain_efficiency', 0.0), promotes=['*'])
        
        # Add common inputs for tower
        self.add('tower_top_diameter',IndepVarComp([('tower_top_diameter', 0.0)]), promotes=['*'])

        # select components
        self.add('hub', HubMassOnlySE(blade_number), promotes=['*'])
        self.add('lowSpeedShaft', LowSpeedShaft4pt_OM(mb1Type, mb2Type, IEC_Class), promotes=['*'])
        self.add('mainBearing', MainBearing_OM('main'), promotes=['lss_design_torque','rotor_diameter']) #explicit connections for bearings
        self.add('secondBearing', MainBearing_OM('second'), promotes=['lss_design_torque','rotor_diameter']) #explicit connections for bearings
        self.add('hubCM', Hub_CM_Adder_OM(), promotes=['*'])
        self.add('gearbox', Gearbox_OM(gear_configuration, shaft_factor), promotes=['*'])
        self.add('highSpeedSide', HighSpeedSide_OM(), promotes=['*'])
        self.add('generator', Generator_OM(drivetrain_design), promotes=['*'])
        self.add('bedplate', Bedplate_OM(uptower_transformer), promotes=['*'])
        self.add('transformer', Transformer_OM(uptower_transformer), promotes=['*'])
        self.add('rna', RNASystemAdder_OM(), promotes=['*'])
        self.add('above_yaw_massAdder', AboveYawMassAdder_OM(crane), promotes=['*'])
        self.add('yawSystem', YawSystem_OM(yaw_motors_number), promotes=['*'])
        self.add('nacelleSystem', NacelleSystemAdder_OM(), promotes=['*'])

        # Connect components where explicit connections needed (for main bearings)
        self.connect('lss_mb1_mass', ['mainBearing.bearing_mass'])
        self.connect('lss_diameter1', ['mainBearing.lss_diameter', 'lss_diameter'])
        self.connect('lss_mb1_cm', ['mainBearing.lss_mb_cm'])
        self.connect('mainBearing.mb_mass', ['mb1_mass'])
        self.connect('mainBearing.mb_cm', ['mb1_cm', 'MB1_location'])
        self.connect('mainBearing.mb_I', ['mb1_I'])

        self.connect('lss_mb2_mass', ['secondBearing.bearing_mass'])
        self.connect('lss_diameter2', ['secondBearing.lss_diameter'])
        self.connect('lss_mb2_cm', ['secondBearing.lss_mb_cm'])
        self.connect('secondBearing.mb_mass', ['mb2_mass'])
        self.connect('secondBearing.mb_cm', ['mb2_cm'])
        self.connect('secondBearing.mb_I', ['mb2_I'])

        self.connect('lss_cm','lss_location',src_indices=[0])
        self.connect('hss_cm','hss_location',src_indices=[0])
        self.connect('gearbox_cm','gearbox_location',src_indices=[0])
        self.connect('generator_cm','generator_location',src_indices=[0])

#------------------------------------------------------------------
# examples

def nacelle_example_5MW_baseline_3pt():

    # NREL 5 MW Drivetrain variables
    # geared 3-stage Gearbox with induction generator machine
    drivetrain_design='geared'
    gear_configuration='eep'  # epicyclic-epicyclic-parallel
    mb1Type='SRB'
    IEC_Class='B'
    shaft_factor='normal'
    uptower_transformer=True
    crane=True  # onboard crane present
    yaw_motors_number = 0 # default value - will be internally calculated
    blade_number = 3

    prob=Problem(root=Drive3pt(mb1Type, IEC_Class, gear_configuration, shaft_factor, drivetrain_design, uptower_transformer, yaw_motors_number, crane, blade_number))
    prob.setup()

    # Rotor and load inputs
    prob['rotor_diameter']=126.0  # m
    prob['rotor_rpm']=12.1  # rpm m/s
    prob['machine_rating']=5000.0
    prob['drivetrain_efficiency']=0.95
    prob['rotor_torque']=1.5 * (prob['machine_rating'] * 1000 / \
                             prob['drivetrain_efficiency']) / (prob['rotor_rpm'] * (np.pi / 30))
    prob['rotor_thrust']=599610.0  # N
    prob['rotor_mass']=0.0  # accounted for in F_z # kg
    prob['rotor_bending_moment_x']=330770.0  # Nm
    prob['rotor_bending_moment_y']=-16665000.0  # Nm
    prob['rotor_bending_moment_z']=2896300.0  # Nm
    prob['rotor_thrust']=599610.0  # N
    prob['rotor_force_y']=186780.0  # N
    prob['rotor_force_z']=-842710.0  # N

    # Drivetrain inputs
    prob['machine_rating']=5000.0  # kW
    prob['gear_ratio']=96.76  # 97:1 as listed in the 5 MW reference document
    prob['shaft_angle']=5.0*np.pi / 180.0  # rad
    prob['shaft_ratio']=0.10
    prob['planet_numbers']=[3, 3, 1]
    prob['shrink_disc_mass']=333.3 * prob['machine_rating'] / 1000.0  # estimated
    prob['carrier_mass']=8000.0  # estimated
    prob['flange_length']=0.5
    prob['overhang']=5.0
    prob['distance_hub2mb']=1.912  # length from hub center to main bearing, leave zero if unknown
    prob['gearbox_input_xcm'] = 0.1
    prob['hss_input_length'] = 1.5

    # Tower inputs
    prob['tower_top_diameter']=3.78  # m

    prob.run()

    # print('----- NREL 5 MW Turbine - 3 Point Suspension -----')
    # print(prob.root.unknowns.dump())

def nacelle_example_5MW_baseline_4pt():

    # NREL 5 MW Drivetrain variables
    # geared 3-stage Gearbox with induction generator machine
    drivetrain_design='geared'
    gear_configuration='eep'  # epicyclic-epicyclic-parallel
    mb1Type='CARB'
    mb2Type='SRB'
    IEC_Class='B'
    shaft_factor='normal'
    uptower_transformer=True
    crane=True  # onboard crane present
    yaw_motors_number = 0 # default value - will be internally calculated
    blade_number = 3


    prob=Problem(root=Drive4pt(mb1Type, mb2Type, IEC_Class, gear_configuration, shaft_factor, drivetrain_design, uptower_transformer, yaw_motors_number, crane, blade_number))
    prob.setup()

    # Rotor and load inputs
    prob['rotor_diameter']=126.0  # m
    prob['rotor_rpm']=12.1  # rpm m/s
    prob['machine_rating']=5000.0
    prob['drivetrain_efficiency']=0.95
    prob['rotor_torque']=1.5 * (prob['machine_rating'] * 1000 / \
                             prob['drivetrain_efficiency']) / (prob['rotor_rpm'] * (np.pi / 30))
    prob['rotor_thrust']=599610.0  # N
    prob['rotor_mass']=0.0  # accounted for in F_z # kg
    prob['rotor_bending_moment_x']=330770.0  # Nm
    prob['rotor_bending_moment_y']=-16665000.0  # Nm
    prob['rotor_bending_moment_z']=2896300.0  # Nm
    prob['rotor_thrust']=599610.0  # N
    prob['rotor_force_y']=186780.0  # N
    prob['rotor_force_z']=-842710.0  # N

    # Drivetrain inputs
    prob['machine_rating']=5000.0  # kW
    prob['gear_ratio']=96.76  # 97:1 as listed in the 5 MW reference document
    prob['shaft_angle']=5.0*np.pi / 180.0  # rad
    prob['shaft_ratio']=0.10
    prob['planet_numbers']=[3, 3, 1]
    prob['shrink_disc_mass']=333.3 * prob['machine_rating'] / 1000.0  # estimated
    prob['carrier_mass']=8000.0  # estimated
    prob['flange_length']=0.5
    prob['overhang']=5.0
    prob['distance_hub2mb']=1.912  # length from hub center to main bearing, leave zero if unknown
    prob['gearbox_input_xcm'] = 0.1
    prob['hss_input_length'] = 1.5

    # Tower inputs
    prob['tower_top_diameter']=3.78  # m

    prob.run()

    # print('----- NREL 5 MW Turbine - 4 Point Suspension -----')
    # print(prob.root.unknowns.dump())


'''
#Need to update for new structure of drivetrain
def nacelle_example_1p5MW_3pt():

    # test of module for turbine data set

    # 1.5 MW Rotor Variables
    print('----- NREL 1p5MW  Drivetrain - 3 Point Suspension-----')
    nace=Group()
    Drive3pt(nace)
    prob=Problem(nace)
    prob.setup()
    nace.rotor_diameter=77  # m
    nace.rotor_rpm=16.18  # rpm
    nace.drivetrain_efficiency=0.95
    nace.machine_rating=1500
    nace.rotor_torque=1.5 * (nace.machine_rating * 1000 / nace.drivetrain_efficiency) / \
                             (nace.rotor_rpm * (pi / 30)
                              )  # 6.35e6 #4365248.74 # Nm
    nace.rotor_thrust=2.6204e5
    nace.rotor_mass=0.0
    nace.rotor_bending_moment=2.7795e6
    nace.rotor_bending_moment_x=8.4389e5
    nace.rotor_bending_moment_y=-2.6758e6
    nace.rotor_bending_moment_z=7.5222e2
    nace.rotor_thrust=2.6204e5
    nace.rotor_force_y=2.8026e4
    nace.rotor_force_z=-3.4763e5


    # 1p5MW  Drivetrain variables
    # geared 3-stage Gearbox with induction generator machine
    nace.drivetrain_design='geared'
    nace.machine_rating=1500.0  # kW
    nace.gear_ratio=78
    nace.gear_configuration='epp'  # epicyclic-parallel-parallel
    nace.crane=False  # onboard crane not present
    nace.shaft_angle=5.0  # deg
    nace.shaft_ratio=0.10
    nace.Np=[3, 1, 1]
    nace.shaft_type='normal'
    nace.uptower_transformer=False  # True
    nace.shrink_disc_mass=333.3 * nace.machine_rating / 1000.0  # estimated
    nace.carrier_mass=2000.0  # estimated
    nace.mb1Type='SRB'
    nace.mb2Type='SRB'
    nace.flange_length=0.285  # m
    nace.overhang=3.3
    nace.distance_hub2mb=1.535  # length from hub center to main bearing, leave zero if unknown

    # 0 if no fatigue check, 1 if parameterized fatigue check, 2 if known
    # loads inputs
    nace.check_fatigue=0

    # variables if check_fatigue = 1:
    nace.blade_number=3
    nace.cut_in=3.5  # cut-in m/s
    nace.cut_out=20.  # cut-out m/s
    nace.Vrated=11.5  # rated windspeed m/s
    nace.weibull_k=2.2  # windepeed distribution shape parameter
    nace.weibull_A=9.  # windspeed distribution scale parameter
    nace.T_life=20.  # design life in years
    nace.IEC_Class_Letter='B'

    # variables if check_fatigue =2:
    # nace.rotor_thrust_distribution =
    # nace.rotor_thrust_count =
    # nace.rotor_Fy_distribution =
    # nace.rotor_Fy_count =
    # nace.rotor_Fz_distribution =
    # nace.rotor_Fz_count =
    # nace.rotor_torque_distribution =
    # nace.rotor_torque_count =
    # nace.rotor_My_distribution =
    # nace.rotor_My_count =
    # nace.rotor_Mz_distribution =
    # nace.rotor_Mz_count =

    # 1p5MW Tower Variables
    nace.tower_top_diameter=2.3  # m

    prob.run()

    sys_print(nace)

def nacelle_example_1p5MW_4pt():

    # test of module for turbine data set

    print('----- NREL 1p5MW  Drivetrain - 4 Point Suspension-----')
    nace=Group()
    Drive4pt(nace)
    prob=Problem(nace)
    prob.setup()
    nace.rotor_diameter=77  # m
    nace.rotor_rpm=16.18  # rpm
    nace.drivetrain_efficiency=0.95
    nace.machine_rating=1500
    nace.rotor_torque=1.5 * (nace.machine_rating * 1000 / nace.drivetrain_efficiency) / \
                             (nace.rotor_rpm * (pi / 30)
                              )  # 6.35e6 #4365248.74 # Nm
    nace.rotor_thrust=2.6204e5
    nace.rotor_mass=0.0
    nace.rotor_bending_moment=2.7795e6
    nace.rotor_bending_moment_x=8.4389e5
    nace.rotor_bending_moment_y=-2.6758e6
    nace.rotor_bending_moment_z=7.5222e2
    nace.rotor_thrust=2.6204e5
    nace.rotor_force_y=2.8026e4
    nace.rotor_force_z=-3.4763e5

    # 1p5MW  Drivetrain variables
    # geared 3-stage Gearbox with induction generator machine
    nace.drivetrain_design='geared'
    nace.machine_rating=1500.0  # kW
    nace.gear_ratio=78
    nace.gear_configuration='epp'  # epicyclic-parallel-parallel
    nace.crane=False  # True # onboard crane present
    nace.shaft_angle=5.0  # deg
    nace.shaft_ratio=0.10
    nace.Np=[3, 1, 1]
    nace.shaft_type='normal'
    nace.uptower_transformer=False  # True
    nace.shrink_disc_mass=333.3 * nace.machine_rating / 1000.0  # estimated
    nace.carrier_mass=2000.0  # estimated
    nace.mb1Type='CARB'
    nace.mb2Type='SRB'
    nace.flange_length=0.285  # m
    nace.overhang=4
    nace.distance_hub2mb=1.3  # length from hub center to main bearing, leave zero if unknown
    nace.gearbox_cm=0.0

    # 0 if no fatigue check, 1 if parameterized fatigue check, 2 if known
    # loads inputs
    nace.check_fatigue=0

    # variables if check_fatigue = 1:
    nace.blade_number=3
    nace.cut_in=3.5  # cut-in m/s
    nace.cut_out=20.  # cut-out m/s
    nace.Vrated=11.5  # rated windspeed m/s
    nace.weibull_k=2.2  # windepeed distribution shape parameter
    nace.weibull_A=9.  # windspeed distribution scale parameter
    nace.T_life=20.  # design life in years
    nace.IEC_Class_Letter='B'

    # variables if check_fatigue =2:
    # nace.rotor_thrust_distribution =
    # nace.rotor_thrust_count =
    # nace.rotor_Fy_distribution =
    # nace.rotor_Fy_count =
    # nace.rotor_Fz_distribution =
    # nace.rotor_Fz_count =
    # nace.rotor_torque_distribution =
    # nace.rotor_torque_count =
    # nace.rotor_My_distribution =
    # nace.rotor_My_count =
    # nace.rotor_Mz_distribution =
    # nace.rotor_Mz_count =

    # 1p5MW Tower Variables
    nace.tower_top_diameter=2.3  # m

    prob.run()

    # cm_print(nace)
    sys_print(nace)

def nacelle_example_p75_3pt():

    # test of module for turbine data set
    print('----- NREL 750kW Design - 3 Point Suspension----')
    # 0.75MW Rotor Variables
    nace=Group()
    Drive3pt(nace)
    prob=Problem(nace)
    prob.setup()
    nace.rotor_diameter=48.2  # m
    nace.rotor_rpm=22.0  # rpm m/s
    nace.drivetrain_efficiency=0.95
    nace.machine_rating=750
    nace.rotor_torque=1.5 * (nace.machine_rating * 1000 / nace.drivetrain_efficiency) / \
                             (nace.rotor_rpm * (pi / 30)
                              )  # 6.35e6 #4365248.74 # Nm

    nace.rotor_thrust=143000.0  # N
    nace.rotor_mass=0.0  # kg
    nace.rotor_bending_moment=495.6e3
    nace.rotor_bending_moment_x=401.0e3
    nace.rotor_bending_moment_y=495.6e3
    nace.rotor_bending_moment_z=-443.0e3
    nace.rotor_thrust=143000.0
    nace.rotor_force_y=-12600.0
    nace.rotor_force_z=-142.0e3

    # NREL 750 kW Drivetrain variables
    # geared 3-stage Gearbox with induction generator machine
    nace.drivetrain_design='geared'
    nace.machine_rating=750  # kW
    nace.gear_ratio=81.491
    nace.gear_configuration='epp'  # epicyclic-parallel-parallel
    nace.crane=False  # True if onboard crane present
    nace.shaft_angle=5.0  # deg
    nace.shaft_length=2.1  # m
    nace.shaft_ratio=0.10
    nace.Np=[3, 1, 1]
    nace.shaft_type='normal'
    nace.uptower_transformer=False
    nace.shrink_disc_mass=333.3 * nace.machine_rating / 1000.0  # estimated
    nace.carrier_mass=250.  # estimated
    nace.mb1Type='SRB'
    nace.mb2Type='TRB2'
    nace.flange_length=0.285  # m
    nace.overhang=2.26
    nace.distance_hub2mb=1.22  # length from hub center to main bearing, leave zero if unknown
    nace.gearbox_cm=0.8
    nace.blade_root_diameter=1.6

    # 0 if no fatigue check, 1 if parameterized fatigue check, 2 if known
    # loads inputs
    nace.check_fatigue=0

    # variables if check_fatigue = 1:
    nace.blade_number=3
    nace.cut_in=3.  # cut-in m/s
    nace.cut_out=25.  # cut-out m/s
    nace.Vrated=16.  # rated windspeed m/s
    nace.weibull_k=2.2  # windepeed distribution shape parameter
    nace.weibull_A=9.  # windspeed distribution scale parameter
    nace.T_life=20.  # design life in years
    nace.IEC_Class_Letter='A'


    # variables if check_fatigue =2:
    # nace.rotor_thrust_distribution =
    # nace.rotor_thrust_count =
    # nace.rotor_Fy_distribution =
    # nace.rotor_Fy_count =
    # nace.rotor_Fz_distribution =
    # nace.rotor_Fz_count =
    # nace.rotor_torque_distribution =
    # nace.rotor_torque_count =
    # nace.rotor_My_distribution =
    # nace.rotor_My_count =
    # nace.rotor_Mz_distribution =
    # nace.rotor_Mz_count =

    # 0.75MW Tower Variables
    nace.tower_top_diameter=2.21  # m

    prob.run()
    # cm_print(nace)
    sys_print(nace)

def nacelle_example_p75_4pt():

    # test of module for turbine data set
    print('----- NREL 750kW Design - 4 Point Suspension----')
    # 0.75MW Rotor Variables
    nace=Group()
    Drive4pt(nace)
    prob=Problem(nace)
    prob.setup()
    nace.rotor_diameter=48.2  # m
    nace.rotor_rpm=22.0  # rpm
    nace.drivetrain_efficiency=0.95
    nace.machine_rating=750
    nace.rotor_torque=1.5 * (nace.machine_rating * 1000 / nace.drivetrain_efficiency) / \
                             (nace.rotor_rpm * (pi / 30)
                              )  # 6.35e6 #4365248.74 # Nm
    #nace.rotor_torque = 6.37e6 #
    nace.rotor_thrust=143000.0
    nace.rotor_mass=0.0
    nace.rotor_bending_moment=459.6e3
    nace.rotor_bending_moment_x=401.0e3
    nace.rotor_bending_moment_y=459.6e3
    nace.rotor_bending_moment_z=-443.0e3
    nace.rotor_thrust=143000.0
    nace.rotor_force_y=-12600.0
    nace.rotor_force_z=-142.0e3

    # NREL 750 kW Drivetrain variables
    # geared 3-stage Gearbox with induction generator machine
    nace.drivetrain_design='geared'
    nace.machine_rating=750  # kW
    nace.gear_ratio=81.491  # as listed in the 5 MW reference document
    nace.gear_configuration='epp'  # epicyclic-parallel-parallel
    nace.crane=False  # True # onboard crane present
    nace.shaft_angle=5.0  # deg
    nace.shaft_length=2.1  # m
    nace.shaft_ratio=0.10
    nace.Np=[3, 1, 1]
    nace.shaft_type='normal'
    nace.uptower_transformer=False  # True
    nace.shrink_disc_mass=333.3 * nace.machine_rating / 1000.0  # estimated
    nace.carrier_mass=1000.0  # estimated
    nace.mb1Type='SRB'
    nace.mb2Type='TRB2'
    nace.flange_length=0.338  # m
    nace.overhang=2.26
    nace.distance_hub2mb=1.22  # 0.007835*rotor_diameter+0.9642 length from hub center to main bearing, leave zero if unknown
    nace.gearbox_cm=0.90

    # 0 if no fatigue check, 1 if parameterized fatigue check, 2 if known
    # loads inputs
    nace.check_fatigue=0

    # variables if check_fatigue = 1:
    nace.blade_number=3
    nace.cut_in=3.  # cut-in m/s
    nace.cut_out=25.  # cut-out m/s
    nace.Vrated=16.  # rated windspeed m/s
    nace.weibull_k=2.2  # windepeed distribution shape parameter
    nace.weibull_A=9.  # windspeed distribution scale parameter
    nace.T_life=20.  # design life in years
    nace.IEC_Class_Letter='A'

    # variables if check_fatigue =2:
    # nace.rotor_thrust_distribution =
    # nace.rotor_thrust_count =
    # nace.rotor_Fy_distribution =
    # nace.rotor_Fy_count =
    # nace.rotor_Fz_distribution =
    # nace.rotor_Fz_count =
    # nace.rotor_torque_distribution =
    # nace.rotor_torque_count =
    # nace.rotor_My_distribution =
    # nace.rotor_My_count =
    # nace.rotor_Mz_distribution =
    # nace.rotor_Mz_count =

    # 0.75MW Tower Variables
    nace.tower_top_diameter=2.21  # m

    prob.run()

    sys_print(nace)
'''

if __name__ == '__main__':
    ''' Main runs through tests of several drivetrain configurations with known component masses and dimensions '''

    nacelle_example_5MW_baseline_3pt()

    nacelle_example_5MW_baseline_4pt()
    
    '''
    nacelle_example_1p5MW_3pt()

    nacelle_example_1p5MW_4pt()

    nacelle_example_p75_3pt()

    nacelle_example_p75_4pt()'''
