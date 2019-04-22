"""
hubSE components
Copyright (c) NREL. All rights reserved.
"""

import numpy as np
from math import pi, cos, sqrt, sin, exp

from drivese.drivese_utils import get_distance_hub2mb

class Hub_System_Adder(object):
    ''' 
    Compute hub mass, cm, and I
    '''

    def __init__(self, blade_number):

        super(Hub_System_Adder, self).__init__()
        self.mass_adder = Hub_Mass_Adder(blade_number)
        self.cm_adder   = Hub_CM_Adder()

    def compute(self, rotor_diameter, blade_mass, distance_hub2mb, shaft_angle, MB1_location, hub_mass, hub_diameter, hub_thickness, pitch_system_mass, spinner_mass):

        (self.rotor_mass, self.hub_system_mass, self.hub_system_I, self.hub_I) = self.mass_adder.compute(blade_mass, hub_mass, hub_diameter,
                                                                                             hub_thickness, pitch_system_mass, spinner_mass)
        self.hub_system_cm = self.cm_adder.compute(rotor_diameter, distance_hub2mb, shaft_angle, MB1_location)

        return(self.rotor_mass, self.hub_system_mass, self.hub_system_cm, self.hub_system_I, self.hub_I)

# -------------------------------------------------


class Hub_Mass_Adder(object):
    ''' 
    Compute hub mass and I
    Excluding cm here, because it has a dependency on main bearing location, which can only be calculated once the full rotor mass is set
    '''

    def __init__(self, blade_number):

        super(Hub_Mass_Adder, self).__init__()
        self.blade_number = blade_number

    def compute(self, blade_mass, hub_mass, hub_diameter, hub_thickness, pitch_system_mass, spinner_mass):

        # variables
        self.blade_mass = blade_mass #Float(iotype='in', units='kg', desc='mass of one blade')
        self.hub_mass = hub_mass #Float(iotype='in', units='kg',desc='mass of Hub')
        self.hub_diameter = hub_diameter #Float(3.0,iotype='in', units='m', desc='hub diameter')
        self.hub_thickness = hub_thickness #Float(iotype='in', units='m', desc='hub thickness')
        self.pitch_system_mass = pitch_system_mass #Float(iotype='in', units='kg',desc='mass of Pitch System')
        self.spinner_mass = spinner_mass #Float(iotype='in', units='kg',desc='mass of spinner')
        
        # outputs
        self.hub_system_I = np.array([0.0, 0.0, 0.0]) #Array(iotype='out', desc='mass moments of Inertia of hub [Ixx, Iyy, Izz, Ixy, Ixz, Iyz] around its center of mass in yaw-aligned c.s.')
        self.hub_system_mass = 0.0 #Float(iotype='out', units='kg',desc='mass of hub system')
        self.rotor_mass = 0.0

        self.hub_system_mass = self.hub_mass + self.pitch_system_mass + self.spinner_mass
        self.rotor_mass = self.hub_system_mass + self.blade_number*self.blade_mass

        #add I definitions here
        hub_I = np.array([0.0, 0.0, 0.0])
        hub_I[0] = 0.4 * (self.hub_mass) * ((self.hub_diameter / 2) ** 5 - (self.hub_diameter / 2 - self.hub_thickness) ** 5) / \
               ((self.hub_diameter / 2) ** 3 - (self.hub_diameter / 2 - self.hub_thickness) ** 3)
        hub_I[1] = hub_I[0]
        hub_I[2] = hub_I[1]

        pitch_system_I = np.array([0.0, 0.0, 0.0])
        pitch_system_I[0] = self.pitch_system_mass * (self.hub_diameter ** 2) / 4
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
        #I = np.zeros(3)
        #for i in (range(0,3)):                        # calculating MOI, at nacelle center of gravity with origin at tower top center / yaw mass center, ignoring masses of non-drivetrain components / auxiliary systems
            # calculate moments around CM
            # sum moments around each components CM
            #I[i]  =  hub_I[i] + pitch_system_I[i] + spinner_I[i]
        self.hub_system_I = np.r_[hub_I + pitch_system_I + spinner_I, np.zeros(3)]

        return(self.rotor_mass, self.hub_system_mass, self.hub_system_I, hub_I)

# -------------------------------------------------


class Hub_CM_Adder(object):
    ''' 
    Compute hub cm
    Separating cm here, because it has a dependency on main bearing location, which can only be calculated once the full rotor mass is set
    '''

    def __init__(self):

        super(Hub_CM_Adder, self).__init__()

    def compute(self, rotor_diameter, distance_hub2mb, shaft_angle, MB1_location):

        # variables
        self.rotor_diameter = rotor_diameter #Float(iotype='in', units='m', desc='rotor diameter')
        self.distance_hub2mb = distance_hub2mb #Float(0.0,iotype='in', units = 'm', desc = 'distance between hub center and upwind main bearing')
        self.shaft_angle = shaft_angle #Float(iotype = 'in', units = 'deg', desc = 'shaft angle')
        self.MB1_location = MB1_location #Array(iotype = 'in', units = 'm', desc = 'center of mass of main bearing in [x,y,z] for an arbitrary coordinate system')
        
        # outputs
        self.hub_system_cm = np.array([0.0, 0.0, 0.0]) #Array(iotype='out', units='m',desc='center of mass of the hub relative to tower to in yaw-aligned c.s.')
        
        if self.distance_hub2mb>0:
            distance_hub2mb = self.distance_hub2mb
        else:
            distance_hub2mb = get_distance_hub2mb(self.rotor_diameter)

        cm = np.array([0.0,0.0,0.0])
        cm[0]     = self.MB1_location[0] - distance_hub2mb
        cm[1]     = 0.0
        cm[2]     = self.MB1_location[2] + distance_hub2mb*sin(self.shaft_angle)
        self.hub_system_cm = (cm)

        return(self.hub_system_cm)

# -------------------------------------------------

class Hub(object):
    ''' Hub class    
          The Hub class is used to represent the hub component of a wind turbine. 
          It contains the general properties for a wind turbine component as well as additional design load and dimentional attributes as listed below.
          It contains an update method to determine the mass, mass properties, and dimensions of the component.            
    '''

    def __init__(self, blade_number):

        super(Hub, self).__init__()
        
        self.blade_number = blade_number

    def compute(self, blade_root_diameter, machine_rating):

        # variables
        self.blade_root_diameter = blade_root_diameter #Float(iotype='in', units='m', desc='blade root diameter')
        self.machine_rating = 1e3*machine_rating #kw->MW Float(iotype = 'in', units = 'MW', desc = 'machine rating of turbine')
        
        # parameters
        #blade_number = Int(3, iotype='in', desc='number of turbine blades')
    
        # outputs
        self.diameter = 0.0 #Float(0.0, iotype='out', units='m', desc='hub diameter')
        self.thickness = 0.0 #Float(0.0, iotype='out',units='m',desc='hub thickness')
        self.mass = 0.0 #Float(0.0, iotype='out', units='kg', desc='overall component mass')

        if self.blade_root_diameter > 0.0: #added 8/6/14 to allow analysis of hubs for unknown blade roots.
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
        
        return(self.mass, self.diameter, self.thickness)


class PitchSystem(object):
    '''
     PitchSystem class
      The PitchSystem class is used to represent the pitch system of a wind turbine.
      It contains the general properties for a wind turbine component as well as additional design load and dimentional attributes as listed below.
      It contains an update method to determine the mass, mass properties, and dimensions of the component.
    '''

    def __init__(self, blade_number):

        super(PitchSystem, self).__init__()
        
        self.blade_number = blade_number

    def compute(self, blade_mass, rotor_bending_moment_y):

        # variables
        self.blade_mass = blade_mass #Float(iotype='in', units='kg', desc='mass of one blade')
        self.rotor_bending_moment_y = rotor_bending_moment_y #Float(iotype='in', units='N*m', desc='flapwise bending moment at blade root')
    
        # parameters
        #blade_number = Int(3, iotype='in', desc='number of turbine blades')
    
        # outputs
        self.mass = 0.0 #Float(0.0, iotype='out', units='kg', desc='overall component mass')

        # Sunderland method for calculating pitch system masses
        pitchmatldensity = 7860.0                             # density of pitch system material (kg / m^3) - assuming BS1503-622 (same material as LSS)
        pitchmatlstress  = 371000000.0                              # allowable stress of hub material (N / m^2)

        hubpitchFact      = 1.0                                 # default factor is 1.0 (0.54 for modern designs)
        self.mass =hubpitchFact * (0.22 * self.blade_mass * self.blade_number + 12.6 * self.rotor_bending_moment_y * (pitchmatldensity / pitchmatlstress))
                                                            # mass of pitch system based on Sunderland model

        return(self.mass)

#-------------------------------------------------------------------------

class Spinner(object):
    '''
       Spinner class
          The SpinnerClass is used to represent the spinner of a wind turbine.
          It contains the general properties for a wind turbine component as well as additional design load and dimentional attributes as listed below.
          It contains an update method to determine the mass, mass properties, and dimensions of the component.
    '''

    def __init__(self):
        super(Spinner, self).__init__()

    def compute(self, rotor_diameter):

        # variables
        self.rotor_diameter = rotor_diameter #Float(iotype='in', units='m', desc='rotor diameter')
    
        # spinner mass comes from cost and scaling model
        self.mass =18.5 * self.rotor_diameter + (-520.5)   # spinner mass comes from cost and scaling model
        
        return(self.mass)

if __name__ == "__main__":

    # TODO: raw python hub component examples
    pass
