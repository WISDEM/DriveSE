"""
driveSE_components.py
New components for low speed shaft, main bearings, gearbox, bedplate and yaw bearings, as well as modified components from NacelleSE

Created by Ryan King 2013. Edited by Taylor Parsons 2014
Copyright (c) NREL. All rights reserved.
"""

from openmdao.main.api import Component, Assembly
from openmdao.main.datatypes.api import Float, Bool, Int, Str, Array, Enum

import numpy as np
from math import pi, cos, sqrt, radians, sin, exp, log10, log, floor, ceil
import algopy
import scipy as scp
import scipy.optimize as opt
from scipy import integrate

from drivese_utils import fatigue_for_bearings, resize_for_bearings, get_rotor_mass, get_L_rb, get_My, get_Mz,\
    size_Generator, size_HighSpeedSide, size_YawSystem, size_LowSpeedShaft, setup_Bedplate_Front, setup_Bedplate, size_Bedplate,\
    characterize_Bedplate_Front, characterize_Bedplate_Rear, size_Transformer, add_RNA, add_Nacelle, add_AboveYawMass,\
    size_LSS_3pt, get_Damage_Brng1, get_Damage_Brng2, setup_Fatigue_Loads, size_LSS_4pt_Loop_1, size_LSS_4pt_Loop_2 #fatigue2_for_bearings,


#-------------------------------------------------------------------------------
# Components
#-------------------------------------------------------------------------------

class LowSpeedShaft_drive4pt(Component):
    ''' LowSpeedShaft class
          The LowSpeedShaft class is used to represent the low speed shaft component of a wind turbine drivetrain. 
          It contains the general properties for a wind turbine component as well as additional design load and dimentional attributes as listed below.
          It contains an update method to determine the mass, mass properties, and dimensions of the component.
    '''

    # variables
    rotor_bending_moment_x = Float(iotype='in', units='N*m', desc='The bending moment about the x axis')
    rotor_bending_moment_y = Float(iotype='in', units='N*m', desc='The bending moment about the y axis')
    rotor_bending_moment_z = Float(iotype='in', units='N*m', desc='The bending moment about the z axis')
    rotor_force_x = Float(iotype='in', units='N', desc='The force along the x axis applied at hub center')
    rotor_force_y = Float(iotype='in', units='N', desc='The force along the y axis applied at hub center')
    rotor_force_z = Float(iotype='in', units='N', desc='The force along the z axis applied at hub center')
    rotor_mass = Float(iotype='in', units='kg', desc='rotor mass')
    rotor_diameter = Float(iotype='in', units='m', desc='rotor diameter')
    machine_rating = Float(iotype='in', units='kW', desc='machine_rating machine rating of the turbine')
    gearbox_mass = Float(iotype='in', units='kg', desc='Gearbox mass')
    carrier_mass = Float(iotype='in', units='kg', desc='Carrier mass')
    overhang = Float(iotype='in', units='m', desc='Overhang distance')
    gearbox_cm = Array(iotype = 'in', units = 'm', desc = 'center of mass of gearbox')
    gearbox_length = Float(iotype='in', units='m', desc='gearbox length')
    flange_length = Float(iotype ='in', units='m', desc ='flange length')
    L_rb = Float(iotype='in', units='m', desc='distance between hub center and upwind main bearing')
    shaft_angle = Float(iotype='in', units='rad', desc='Angle of the LSS inclindation with respect to the horizontal')
    shaft_ratio = Float(iotype='in', desc='Ratio of inner diameter to outer diameter.  Leave zero for solid LSS')
    DrivetrainEfficiency = Float(iotype = 'in', desc = 'overall drivettrain efficiency')

    #fatigue1 variables
    rotor_freq = Float(iotype = 'in', units = 'rpm', desc='rated rotor speed')
    availability = Float(.95,iotype = 'in', desc = 'turbine availability')
    fatigue_exponent = Float(0,iotype = 'in', desc = 'fatigue exponent of material')
    S_ut = Float(700e6,iotype = 'in', units = 'Pa', desc = 'ultimate tensile strength of material')
    weibull_A = Float(iotype = 'in', units = 'm/s', desc = 'weibull scale parameter "A" of 10-minute windspeed probability distribution')
    weibull_k = Float(iotype = 'in', desc = 'weibull shape parameter "k" of 10-minute windspeed probability distribution')
    blade_number = Float(iotype = 'in', desc = 'number of blades on rotor, 2 or 3')
    cut_in = Float(iotype = 'in', units = 'm/s', desc = 'cut-in windspeed')
    cut_out = Float(iotype = 'in', units = 'm/s', desc = 'cut-out windspeed')
    Vrated = Float(iotype = 'in', units = 'm/s', desc = 'rated windspeed')
    T_life = Float(iotype = 'in', units = 'yr', desc = 'cut-in windspeed')

    # fatigue2 variables
    rotor_thrust_distribution = Array(iotype='in', units ='N', desc = 'thrust distribution across turbine life')
    rotor_thrust_count = Array(iotype='in', desc = 'corresponding cycle array for thrust distribution')
    rotor_Fy_distribution = Array(iotype='in', units ='N', desc = 'Fy distribution across turbine life')
    rotor_Fy_count = Array(iotype='in', desc = 'corresponding cycle array for Fy distribution')
    rotor_Fz_distribution = Array(iotype='in', units ='N', desc = 'Fz distribution across turbine life')
    rotor_Fz_count = Array(iotype='in', desc = 'corresponding cycle array for Fz distribution') 
    rotor_torque_distribution = Array(iotype='in', units ='N*m', desc = 'torque distribution across turbine life')
    rotor_torque_count = Array(iotype='in', desc = 'corresponding cycle array for torque distribution') 
    rotor_My_distribution = Array(iotype='in', units ='N*m', desc = 'My distribution across turbine life')
    rotor_My_count = Array(iotype='in', desc = 'corresponding cycle array for My distribution') 
    rotor_Mz_distribution = Array(iotype='in', units ='N*m', desc = 'Mz distribution across turbine life')
    rotor_Mz_count = Array(iotype='in', desc = 'corresponding cycle array for Mz distribution') 

    # parameters
    shrink_disc_mass = Float(iotype='in', units='kg', desc='Mass of the shrink disc')# shrink disk or flange addtional mass
    mb1Type = Enum('SRB',('CARB','TRB1','TRB2','SRB','CRB','RB'),iotype='in',desc='Main bearing type')
    mb2Type = Enum('SRB',('CARB','TRB1','TRB2','SRB','CRB','RB'),iotype='in',desc='Second bearing type')
    check_fatigue = Enum(0,(0,1,2),iotype = 'in', desc = 'turns on and off fatigue check')
    IEC_Class = Enum('A',('A','B','C'),iotype='in',desc='IEC class letter: A, B, or C')
    
    # outputs
    design_torque = Float(iotype='out', units='N*m', desc='lss design torque')
    design_bending_load = Float(iotype='out', units='N', desc='lss design bending load')
    length = Float(iotype='out', units='m', desc='lss length')
    diameter1 = Float(iotype='out', units='m', desc='lss outer diameter at main bearing')
    diameter2 = Float(iotype='out', units='m', desc='lss outer diameter at second bearing')
    mass = Float(0.0, iotype='out', units='kg', desc='overall component mass')
    cm = Array(np.array([0.0, 0.0, 0.0]), iotype='out', desc='center of mass of the component in [x,y,z] for an arbitrary coordinate system')
    I = Array(np.array([0.0, 0.0, 0.0]), iotype='out', desc=' moments of Inertia for the component [Ixx, Iyy, Izz] around its center of mass')
    FW_mb1 = Float(iotype='out', units='m', desc='facewidth of upwind main bearing') 
    FW_mb2 = Float(iotype='out', units='m', desc='facewidth of main bearing')     
    bearing_mass1 = Float(iotype='out', units = 'kg', desc='main bearing mass')
    bearing_mass2 = Float(iotype='out', units = 'kg', desc='second bearing mass')
    bearing_location1 = Array(np.array([0,0,0]),iotype='out', units = 'm', desc = 'main bearing 1 center of mass')
    bearing_location2 = Array(np.array([0,0,0]),iotype='out', units = 'm', desc = 'main bearing 2 center of mass')

    def __init__(self):

        super(LowSpeedShaft_drive4pt, self).__init__()
    
    def execute(self):

        #input parameters
        self.g=9.81

        if self.L_rb == 0: #distance from hub center to main bearing
          L_rb = 0.007835*self.rotor_diameter+0.9642
        else:
        	L_rb = self.L_rb

        #If user does not know important moments, crude approx
        if self.rotor_mass > 0 and self.rotor_bending_moment_y == 0: 
            self.rotor_bending_moment_y=get_My(self.rotor_mass,L_rb)

        if self.rotor_mass > 0 and self.rotor_bending_moment_z == 0:
            self.rotor_bending_moment_z=get_Mz(self.rotor_mass,L_rb)

        if self.rotor_mass ==0:
          [self.rotor_mass] = get_rotor_mass(self.machine_rating,False)

        if self.flange_length == 0:
            self.flange_length = 0.3*(self.rotor_diameter/100.0)**2.0 - 0.1 * (self.rotor_diameter / 100.0) + 0.4
                
        # initialization for iterations    
        self.L_ms_new = 0.0
        self.L_ms_0=0.5 # main shaft length downwind of main bearing
        self.L_ms=self.L_ms_0
        self.len_pts=101
        self.D_max=1
        self.D_min=0.2

        tol=1e-4 
        check_limit = 1.0
        dL=0.05
        counter = 0
        N_count=50
        N_count_2=2

        #Distances
        self.L_bg = 6.11-L_rb    #distance from first main bearing to gearbox yokes  # to add as an input
        self.L_as = self.L_ms/2.0     #distance from main bearing to shaft center
        self.L_gb = 0.0          #distance to gbx center from trunnions in x-dir # to add as an input
        self.H_gb = 1.0          #distance to gbx center from trunnions in z-dir # to add as an input     
        self.L_gp = 0.825        #distance from gbx coupling to gbx trunnions
        self.L_cu = self.L_ms + 0.5   #distance from upwind main bearing to upwind carrier bearing 0.5 meter is an estimation # to add as an input
        self.L_cd = self.L_cu + 0.5   #distance from upwind main bearing to downwind carrier bearing 0.5 meter is an estimation # to add as an input
        
        #material properties
        self.E=2.1e11
        self.density=7800.0
        self.n_safety = 2.5 # According to AGMA, takes into account the peak load safety factor
        self.Sy = 66000#*self.S_ut/700e6 #66000 #psi

        #unit conversion
        self.u_knm_inlb = 8850.745454036
        self.u_in_m = 0.0254000508001

        #Main bearing defelection check
        if self.mb1Type == 'TRB1' or 'TRB2':
            Bearing_Limit = 3.0/60.0/180.0*pi
        elif self.mb1Type == 'CRB':
            Bearing_Limit = 4.0/60.0/180.0*pi
        elif self.mb1Type == 'SRB' or 'RB':
            Bearing_Limit = 0.078
        elif self.mb1Type == 'RB':
            Bearing_Limit = 0.002
        elif self.mb1Type == 'CARB':
            Bearing_Limit = 0.5/180*pi
        else:
            Bearing_Limit = False

        #Second bearing defelection check
        if self.mb2Type == 'TRB1' or 'TRB2':
            Bearing_Limit2 = 3.0/60.0/180.0*pi
        elif self.mb2Type == 'CRB':
            Bearing_Limit2 = 4.0/60.0/180.0*pi
        elif self.mb2Type == 'SRB' or 'RB':
            Bearing_Limit2 = 0.078
        elif self.mb2Type == 'RB':
            Bearing_Limit2 = 0.002
        elif self.mb2Type == 'CARB':
            Bearing_Limit2 = 0.5/180*pi
        else:
            Bearing_Limit2 = False

        self.n_safety_brg = 1.0

        length_max = self.overhang - L_rb + (self.gearbox_cm[0] -self.gearbox_length/2.) #modified length limit 7/29/14

        while abs(check_limit) > tol and self.L_ms_new < length_max:
            counter = counter+1
            if self.L_ms_new > 0:
                self.L_ms=self.L_ms_new
            else:
                self.L_ms=self.L_ms_0

            size_LSS_4pt_Loop_1(self)

            check_limit = abs(abs(self.theta_y[-1])-Bearing_Limit/self.n_safety_brg)

            if check_limit < 0:
                self.L_ms_new = self.L_ms + dL

            else:
                self.L_ms_new = self.L_ms + dL

         #Initialization
        self.L_mb=self.L_ms_new
        counter_ms=0
        check_limit_ms=1.0
        self.L_mb_new=0.0
        self.L_mb_0=self.L_mb                     #main shaft length
        self.L_ms = self.L_ms_new
        dL_ms = 0.05
        dL = 0.0025

        while abs(check_limit_ms)>tol and self.L_mb_new < length_max:
            counter_ms= counter_ms + 1
            if self.L_mb_new > 0:
                self.L_mb=self.L_mb_new
            else:
                self.L_mb=self.L_mb_0

            counter = 0.0
            check_limit=1.0
            self.L_ms_gb_new=0.0
            self.L_ms_0=0.5 #mainshaft length
            self.L_ms = self.L_ms_0


            while abs(check_limit) > tol and counter <N_count_2:
                counter =counter+1
                if self.L_ms_gb_new>0.0:
                    self.L_ms_gb = self.L_ms_gb_new
                else:
                    self.L_ms_gb = self.L_ms_0

                size_LSS_4pt_Loop_2(self)

                check_limit = abs(abs(self.theta_y[-1])-Bearing_Limit/self.n_safety_brg)

                if check_limit < 0:
                    self.L_ms__gb_new = self.L_ms_gb + dL
                else:
                    self.L_ms__gb_new = self.L_ms_gb + dL

                check_limit_ms = abs(abs(self.theta_y[-1]) - Bearing_Limit2/self.n_safety_brg)

                if check_limit_ms < 0:
                    self.L_mb_new = self.L_mb + dL_ms
                else:
                    self.L_mb_new = self.L_mb + dL_ms

        # fatigue check Taylor Parsons 6/14
        if self.check_fatigue == 1 or self.check_fatigue == 2:
          #start_time = time.time()

          #checks to make sure all inputs are reasonable
          if self.rotor_mass < 100:
              [self.rotor_mass] = get_rotor_mass(self.machine_rating,False)

          #material properties 34CrNiMo6 steel +QT, large diameter
          self.n_safety = 2.5
          if self.S_ut <= 0:
            self.S_ut=700.0e6 #Pa

          #calculate material props for fatigue
          Sm=0.9*self.S_ut #for bending situations, material strength at 10^3 cycles

          if self.fatigue_exponent!=0:
            if self.fatigue_exponent > 0:
                self.SN_b = - self.fatigue_exponent
            else:
                self.SN_b = self.fatigue_exponent
          else:
              C_size=0.6 #diameter larger than 10"
              C_surf=4.51*(self.S_ut/1e6)**-.265 #machined surface 272*(self.S_ut/1e6)**-.995 #forged
              C_temp=1 #normal operating temps
              C_reliab=0.814 #99% reliability
              C_envir=1. #enclosed environment
              Se=C_size*C_surf*C_temp*C_reliab*C_envir*.5*self.S_ut #modified endurance limit for infinite life (should be Sf)\
              Nfinal = 5e8 #point where fatigue limit occurs under hypothetical S-N curve TODO adjust to fit actual data
              z=log10(1e3)-log10(Nfinal)  #assuming no endurance limit (high strength steel)
              self.SN_b=1/z*log10(Sm/Se)
          self.SN_a=Sm/(1000.**self.SN_b)
          # print 'fatigue_exponent:',self.SN_b
          # print 'm:', -1/self.SN_b
          # print 'a:', self.SN_a
          if self.check_fatigue == 1:

              setup_Fatigue_Loads(self)

              #upwind bearing calculations
              iterationstep=0.001
              diameter_limit = 5.0
              while True:
                  get_Damage_Brng1(self)
                  if self.Damage < 1 or self.D_max >= diameter_limit:
                      break
                  else:
                      self.D_max+=iterationstep

              #downwind bearing calculations
              diameter_limit = 5.0
              iterationstep=0.001
              while True:
                  get_Damage_Brng2(self)
                  if self.Damage < 1 or self.D_med >= diameter_limit:
                      break
                  else:
                      self.D_med+=iterationstep

              #begin bearing calculations
              N_bearings = self.N/self.blade_number #counts per rotation (not defined by characteristic frequency 3n_rotor)

              Fr1_range = ((abs(self.Fz1stoch)+abs(self.Fz1determ))**2 +self.Fy1stoch**2)**.5 #radial stochastic + deterministic mean
              Fa1_range = np.zeros(len(self.Fy1stoch))

              #...calculate downwind forces
              lss_weight=self.density*9.81*(((pi/12)*(self.D_max**2+self.D_med**2+self.D_max*self.D_med)*(self.L_mb))-(pi/4*self.L_mb*self.D_in**2))
              Fy2stoch = -self.Mz_stoch/(self.L_mb) #= -Fy1 - Fy_stoch
              Fz2stoch = -(lss_weight*2./3.*self.L_mb-self.My_stoch)/(self.L_mb) + (lss_weight+self.shrinkDiscWeight+self.gbxWeight)*cos(self.shaft_angle) - self.rotorWeight #-Fz1 +Weights*cos(gamma)-Fz_stoch+Fz_mean (Fz_mean is in negative direction)
              Fr2_range = (Fy2stoch**2+(Fz2stoch+abs(-self.rotorWeight*L_rb + 0.5*lss_weight+self.gbxWeight*self.L_gb/self.L_mb))**2)**0.5
              Fa2_range = self.Fx_stoch*cos(self.shaft_angle) + (self.rotorWeight+lss_weight)*sin(self.shaft_angle) #axial stochastic + mean

              life_bearing = self.N_f/self.blade_number

              [self.D_max_a,FW_max,bearing1mass] = fatigue_for_bearings(self.D_max, Fr1_range, Fa1_range, N_bearings, life_bearing, self.mb1Type,False)
              [self.D_med_a,FW_med,bearing2mass] = fatigue_for_bearings(self.D_med, Fr2_range, Fa2_range, N_bearings, life_bearing, self.mb2Type,False)

          # elif self.check_fatigue == 2: # untested and not used currently
          #   Fx = self.rotor_thrust_distribution
          #   n_Fx = self.rotor_thrust_count
          #   Fy = self.rotor_Fy_distribution
          #   n_Fy = self.rotor_Fy_count
          #   Fz = self.rotor_Fz_distribution
          #   n_Fz = self.rotor_Fz_count
          #   Mx = self.rotor_torque_distribution
          #   n_Mx = self.rotor_torque_count
          #   My = self.rotor_My_distribution
          #   n_My = self.rotor_My_count
          #   Mz = self.rotor_Mz_distribution
          #   n_Mz = self.rotor_Mz_count

          #   print n_Fx
          #   print Fx*.5
          #   print Mx*.5
          #   print -1/self.SN_b

          #   def Ninterp(L_ult,L_range,m):
          #       return (L_ult/(.5*L_range))**m #TODO double-check that the input will be the load RANGE instead of load amplitudes. May also include means

          #   #upwind bearing calcs
          #   diameter_limit = 5.0
          #   iterationstep=0.001
          #   #upwind bearing calcs
          #   while True:
          #       self.Damage = 0
          #       Fx_ult = self.SN_a*(pi/4.*(self.D_max**2-self.D_in**2))
          #       Fyz_ult = self.SN_a*(pi*(self.D_max**4-self.D_in**4))/(self.D_max*64.)/self.L_rb
          #       Mx_ult = self.SN_a*(pi*(self.D_max**4-self.D_in**4))/(32*(3)**.5*self.D_max)
          #       Myz_ult = self.SN_a*(pi*(self.D_max**4-self.D_in**4))/(self.D_max*64.)
          #       if Fx_ult !=0 and np.all(n_Fx) != 0:
          #           self.Damage+=scp.integrate.simps(n_Fx/Ninterp(Fx_ult,Fx,-1/self.SN_b),x=n_Fx,even = 'avg')
          #       if Fyz_ult !=0:
          #           if np.all(n_Fy) != 0:
          #               self.Damage+=scp.integrate.simps(abs(n_Fy/Ninterp(Fyz_ult,Fy,-1/self.SN_b)),x=n_Fy,even = 'avg')
          #           if np.all(n_Fz) != 0:
          #               self.Damage+=scp.integrate.simps(abs(n_Fz/Ninterp(Fyz_ult,Fz,-1/self.SN_b)),x=n_Fz,even = 'avg')
          #       if Mx_ult !=0 and np.all(n_Mx) != 0:
          #           self.Damage+=scp.integrate.simps(abs(n_Mx/Ninterp(Mx_ult,Mx,-1/self.SN_b)),x=n_Mx,even = 'avg')
          #       if Myz_ult!=0:
          #           if np.all(n_My) != 0:
          #               self.Damage+=scp.integrate.simps(abs(n_My/Ninterp(Myz_ult,My,-1/self.SN_b)),x=n_My,even = 'avg')
          #           if np.all(n_Mz) != 0:
          #               self.Damage+=scp.integrate.simps(abs(n_Mz/Ninterp(Myz_ult,Mz,-1/self.SN_b)),x=n_Mz,even = 'avg')

          #       print 'Upwind Bearing Diameter:', self.D_max
          #       print 'self.Damage:', self.Damage

          #       if self.Damage <= 1 or self.D_max >= diameter_limit:
          #           # print 'Upwind Bearing Diameter:', self.D_max
          #           # print 'self.Damage:', self.Damage
          #           #print (time.time() - start_time), 'seconds of total simulation time'
          #           break
          #       else:
          #           self.D_max+=iterationstep
          #   #downwind bearing calcs
          #   while True:
          #       self.Damage = 0
          #       Fx_ult = self.SN_a*(pi/4.*(self.D_med**2-self.D_in**2))
          #       Mx_ult = self.SN_a*(pi*(self.D_med**4-self.D_in**4))/(32*(3)**.5*self.D_med)
          #       if Fx_ult !=0:
          #           self.Damage+=scp.integrate.simps(n_Fx/Ninterp(Fx_ult,Fx,-1/self.SN_b),x=n_Fx,even = 'avg')
          #       if Mx_ult !=0:
          #           self.Damage+=scp.integrate.simps(n_Mx/Ninterp(Mx_ult,Mx,-1/self.SN_b),x=n_Mx,even = 'avg')
          #       print 'Downwind Bearing Diameter:', self.D_med
          #       print 'self.Damage:', self.Damage

          #       if self.Damage <= 1 or self.D_med>= diameter_limit:
          #           # print 'Upwind Bearing Diameter:', self.D_max
          #           # print 'self.Damage:', self.Damage
          #           #print (time.time() - start_time), 'seconds of total simulation time'
          #           break
          #       else:
          #           self.D_med+=iterationstep

          #   #bearing calcs
          #   if self.availability != 0 and rotor_freq != 0 and self.T_life != 0 and self.cut_out != 0 and self.weibull_A != 0:
          #       N_rotations = self.availability*rotor_freq/60.*(self.T_life*365*24*60*60)*exp(-(self.cut_in/self.weibull_A)**self.weibull_k)-exp(-(self.cut_out/self.weibull_A)**self.weibull_k)
          #   elif np.max(n_Fx > 1e6):
          #       N_rotations = np.max(n_Fx)/self.blade_number
          #   elif np.max(n_My > 1e6):
          #       N_rotations = np.max(n_My)/self.blade_number
          #   # print 'Upwind bearing calcs'
          #   Fz1_Fz = Fz*(self.L_mb+self.L_rb)/self.L_mb
          #   Fz1_My = My/self.L_mb
          #   Fy1_Fy = -Fy*(self.L_mb+self.L_rb)/self.L_mb
          #   Fy1_Mz = Mz/self.L_mb
          #   [self.D_max_a,FW_max,bearing1mass] = fatigue2_for_bearings(self.D_max,self.mb1Type,np.zeros(2),np.array([1,2]),Fy1_Fy,n_Fy/self.blade_number,Fz1_Fz,n_Fz/self.blade_number,Fz1_My,n_My/self.blade_number,Fy1_Mz,n_Mz/self.blade_number,N_rotations)
          #   # print 'Downwind bearing calcs'
          #   Fz2_Fz = Fz*self.L_rb/self.L_mb
          #   Fz2_My = My/self.L_mb
          #   Fy2_Fy = Fy*self.L_rb/self.L_mb
          #   Fy2_Mz = Mz/self.L_mb
          #   [self.D_med_a,FW_med,bearing2mass] = fatigue2_for_bearings(self.D_med,self.mb2Type,Fx,n_Fx/self.blade_number,Fy2_Fy,n_Fy/self.blade_number,Fz2_Fz,n_Fz/self.blade_number,Fz2_My,n_My/self.blade_number,Fy2_Mz,n_Mz/self.blade_number,N_rotations)

        else: #if fatigue_check is not true, resize based on diameter            
            [self.D_max_a,FW_max,bearing1mass] = resize_for_bearings(self.D_max,  self.mb1Type,False)
            [self.D_med_a,FW_med,bearing2mass] = resize_for_bearings(self.D_med,  self.mb2Type,False)

        # end fatigue code additions 6/2014
            
        lss_mass_new=(pi/3)*(self.D_max_a**2+self.D_med_a**2+self.D_max_a*self.D_med_a)*(self.L_mb-(FW_max+FW_med)/2)*self.density/4+ \
                         (pi/4)*(self.D_max_a**2-self.D_in**2)*self.density*FW_max+\
                         (pi/4)*(self.D_med_a**2-self.D_in**2)*self.density*FW_med-\
                         (pi/4)*(self.D_in**2)*self.density*(self.L_mb+(FW_max+FW_med)/2)

        ## begin bearing routine with updated shaft mass
        self.length=self.L_mb_new + (FW_max+FW_med)/2 + self.flange_length # add facewidths and flange
        # print ("self.L_mb: {0}").format(self.L_mb)
        # print ("LSS length, m: {0}").format(self.length)
        self.D_outer=self.D_max
        # print ("Upwind MB OD, m: {0}").format(self.D_max_a)
        # print ("Dnwind MB OD, m: {0}").format(self.D_med_a)
        # print ("self.D_min: {0}").format(self.D_min)
        self.D_in=self.D_in
        self.mass=lss_mass_new*1.33 # add flange mass
        self.diameter1= self.D_max_a
        self.diameter2= self.D_med_a 

        # calculate mass properties
        downwind_location = np.array([self.gearbox_cm[0]-self.gearbox_length/2. , self.gearbox_cm[1] , self.gearbox_cm[2] ])

        bearing_location1 = np.array([0.,0.,0.]) #upwind
        bearing_location1[0] = downwind_location[0] - (self.L_mb_new + FW_med/2)*cos(self.shaft_angle)
        bearing_location1[1] = downwind_location[1]
        bearing_location1[2] = downwind_location[2] + (self.L_mb_new + FW_med/2)*sin(self.shaft_angle)
        self.bearing_location1 = bearing_location1

        bearing_location2 = np.array([0.,0.,0.]) #downwind
        bearing_location2[0] = downwind_location[0] - FW_med*.5*cos(self.shaft_angle)
        bearing_location2[1] = downwind_location[1]
        bearing_location2[2] = downwind_location[2] + FW_med*.5*sin(self.shaft_angle)
        self.bearing_location2 = bearing_location2

        cm = np.array([0.0,0.0,0.0])
        cm[0] = downwind_location[0] - 0.65*self.length*cos(self.shaft_angle) #From solid models, center of mass with flange (not including shrink disk) very nearly .65*total_length
        cm[1] = downwind_location[1]
        cm[2] = downwind_location[2] + 0.65*self.length*sin(self.shaft_angle)

        #including shrink disk mass
        self.cm[0] = (cm[0]*self.mass + downwind_location[0]*self.shrink_disc_mass) / (self.mass+self.shrink_disc_mass) 
        self.cm[1] = cm[1]
        self.cm[2] = (cm[2]*self.mass + downwind_location[2]*self.shrink_disc_mass) / (self.mass+self.shrink_disc_mass)
        self.mass+=self.shrink_disc_mass

        I = np.array([0.0, 0.0, 0.0])
        I[0]  = self.mass * (self.D_in ** 2.0 + self.D_outer ** 2.0) / 8.0
        I[1]  = self.mass * (self.D_in ** 2.0 + self.D_outer ** 2.0 + (4.0 / 3.0) * (self.length ** 2.0)) / 16.0
        I[2]  = I[1]
        self.I = I

        self.FW_mb1 = FW_max
        self.FW_mb2 = FW_med

        self.bearing_mass1 = bearing1mass
        self.bearing_mass2 = bearing2mass

#-------------------------------------------------------------------------------
class LowSpeedShaft_drive3pt(Component):
    ''' LowSpeedShaft class
          The LowSpeedShaft class is used to represent the low speed shaft component of a wind turbine drivetrain. 
          It contains the general properties for a wind turbine component as well as additional design load and dimentional attributes as listed below.
          It contains an update method to determine the mass, mass properties, and dimensions of the component.
    '''

    # variables
    rotor_bending_moment_x = Float(iotype='in', units='N*m', desc='The bending moment about the x axis')
    rotor_bending_moment_y = Float(iotype='in', units='N*m', desc='The bending moment about the y axis')
    rotor_bending_moment_z = Float(iotype='in', units='N*m', desc='The bending moment about the z axis')
    rotor_force_x = Float(iotype='in', units='N', desc='The force along the x axis applied at hub center')
    rotor_force_y = Float(iotype='in', units='N', desc='The force along the y axis applied at hub center')
    rotor_force_z = Float(iotype='in', units='N', desc='The force along the z axis applied at hub center')
    rotor_mass = Float(iotype='in', units='kg', desc='rotor mass')
    rotor_diameter = Float(iotype='in', units='m', desc='rotor diameter')
    machine_rating = Float(iotype='in', units='kW', desc='machine_rating machine rating of the turbine')
    gearbox_mass = Float(iotype='in', units='kg', desc='Gearbox mass')
    carrier_mass = Float(iotype='in', units='kg', desc='Carrier mass')
    overhang = Float(iotype='in', units='m', desc='Overhang distance')
    L_rb = Float(iotype='in', units='m', desc='distance between hub center and upwind main bearing')

    #fatigue1 variables
    fatigue_exponent = Float(iotype = 'in', desc = 'fatigue exponent of material')
    S_ut = Float(700e6,iotype = 'in', units = 'Pa', desc = 'ultimate tensile strength of material')
    weibull_A = Float(iotype = 'in', units = 'm/s', desc = 'weibull scale parameter "A" of 10-minute windspeed probability distribution')
    weibull_k = Float(iotype = 'in', desc = 'weibull shape parameter "k" of 10-minute windspeed probability distribution')
    blade_number = Float(iotype = 'in', desc = 'number of blades on rotor, 2 or 3')
    cut_in = Float(iotype = 'in', units = 'm/s', desc = 'cut-in windspeed')
    cut_out = Float(iotype = 'in', units = 'm/s', desc = 'cut-out windspeed')
    Vrated = Float(iotype = 'in', units = 'm/s', desc = 'rated windspeed')
    T_life = Float(iotype = 'in', units = 'yr', desc = 'design life')
    DrivetrainEfficiency = Float(iotype = 'in', desc = 'overall drivettrain efficiency')
    rotor_freq = Float(iotype = 'in', units = 'rpm', desc='rated rotor speed')
    availability = Float(.95,iotype = 'in', desc = 'turbine availability')

    #fatigue2 variables
    rotor_thrust_distribution = Array(iotype='in', units ='N', desc = 'thrust distribution across turbine life')
    rotor_thrust_count = Array(iotype='in', desc = 'corresponding cycle array for thrust distribution')
    rotor_Fy_distribution = Array(iotype='in', units ='N', desc = 'Fy distribution across turbine life')
    rotor_Fy_count = Array(iotype='in', desc = 'corresponding cycle array for Fy distribution')
    rotor_Fz_distribution = Array(iotype='in', units ='N', desc = 'Fz distribution across turbine life')
    rotor_Fz_count = Array(iotype='in', desc = 'corresponding cycle array for Fz distribution') 
    rotor_torque_distribution = Array(iotype='in', units ='N*m', desc = 'torque distribution across turbine life')
    rotor_torque_count = Array(iotype='in', desc = 'corresponding cycle array for torque distribution') 
    rotor_My_distribution = Array(iotype='in', units ='N*m', desc = 'My distribution across turbine life')
    rotor_My_count = Array(iotype='in', desc = 'corresponding cycle array for My distribution') 
    rotor_Mz_distribution = Array(iotype='in', units ='N*m', desc = 'Mz distribution across turbine life')
    rotor_Mz_count = Array(iotype='in', desc = 'corresponding cycle array for Mz distribution') 

    # parameters
    shrink_disc_mass = Float(iotype='in', units='kg', desc='Mass of the shrink disc')
    gearbox_cm = Array(iotype = 'in', units = 'm', desc = 'center of mass of gearbox')
    gearbox_length = Float(iotype='in', units='m', desc='gearbox length')
    flange_length = Float(iotype ='in', units='m', desc ='flange length')
    shaft_angle = Float(iotype='in', units='rad', desc='Angle of the LSS inclindation with respect to the horizontal')
    shaft_ratio = Float(iotype='in', desc='Ratio of inner diameter to outer diameter.  Leave zero for solid LSS')
    mb1Type = Enum('SRB',('CARB','TRB1','TRB2','SRB','CRB','RB'),iotype='in',desc='Main bearing type')
    mb2Type = Enum('SRB',('CARB','TRB1','TRB2','SRB','CRB','RB'),iotype='in',desc='Second bearing type')
    check_fatigue = Enum(0,(0,1,2),iotype = 'in', desc = 'turns on and off fatigue check')
    IEC_Class = Enum('A',('A','B','C'),iotype='in',desc='IEC class letter: A, B, or C')
   
    # outputs
    design_torque = Float(iotype='out', units='N*m', desc='lss design torque')
    design_bending_load = Float(iotype='out', units='N', desc='lss design bending load')
    length = Float(iotype='out', units='m', desc='lss length')
    diameter1 = Float(iotype='out', units='m', desc='lss outer diameter at main bearing')
    diameter2 = Float(iotype='out', units='m', desc='lss outer diameter at second bearing')
    mass = Float(0.0, iotype='out', units='kg', desc='overall component mass')
    cm = Array(np.array([0.0, 0.0, 0.0]), iotype='out', desc='center of mass of the component in [x,y,z] for an arbitrary coordinate system')
    I = Array(np.array([0.0, 0.0, 0.0]), iotype='out', desc=' moments of Inertia for the component [Ixx, Iyy, Izz] around its center of mass')
    FW_mb = Float(iotype='out', units='m', desc='facewidth of main bearing')    
    bearing_mass1 = Float(iotype='out', units = 'kg', desc='main bearing mass')
    bearing_mass2 = Float(0., iotype='out', units = 'kg', desc='main bearing mass') #zero for 3-pt model
    bearing_location1 = Array(np.array([0,0,0]),iotype='out', units = 'm', desc = 'main bearing 1 center of mass')
    bearing_location2 = Array(np.array([0,0,0]),iotype='out', units = 'm', desc = 'main bearing 2 center of mass')

    def __init__(self):
        '''
        Initializes low speed shaft component  
        '''

        super(LowSpeedShaft_drive3pt, self).__init__()
    
    def execute(self):

        #input parameters
        if self.flange_length == 0:
            self.flange_length = 0.3*(self.rotor_diameter/100.0)**2.0 - 0.1 * (self.rotor_diameter / 100.0) + 0.4

        if self.L_rb == 0: #distance from hub center to main bearing
           L_rb = get_L_rb(self.rotor_diameter, False)[0]
        else:
        	 L_rb = self.L_rb

        #If user does not know important moments, crude approx
        if self.rotor_mass > 0 and self.rotor_bending_moment_y == 0: 
            self.rotor_bending_moment_y=get_My(self.rotor_mass,L_rb)

        if self.rotor_mass > 0 and self.rotor_bending_moment_z == 0:
            self.rotor_bending_moment_z=get_Mz(self.rotor_mass,L_rb)

        self.g = 9.81 #m/s
        self.density = 7850.0


        self.L_ms_new = 0.0
        self.L_ms_0=0.5 # main shaft length downwind of main bearing
        self.L_ms=self.L_ms_0
        tol=1e-4 
        check_limit = 1.0
        dL=0.05
        self.D_max = 1.0
        self.D_min = 0.2

        T=self.rotor_bending_moment_x/1000.0

        #Main bearing defelection check
        if self.mb1Type == 'TRB1' or 'TRB2':
            Bearing_Limit = 3.0/60.0/180.0*pi
        elif self.mb1Type == 'CRB':
            Bearing_Limit = 4.0/60.0/180.0*pi
        elif self.mb1Type == 'SRB' or 'RB':
            Bearing_Limit = 0.078
        elif self.mb1Type == 'RB':
            Bearing_Limit = 0.002
        elif self.mb1Type == 'CARB':
            Bearing_Limit = 0.5/180*pi
        else:
            Bearing_Limit = False
        
        self.n_safety_brg = 1.0
        self.n_safety=2.5
        self.Sy = 66000#*self.S_ut/700e6 #psi
        self.E=2.1e11  
        N_count=50    
          
        self.u_knm_inlb = 8850.745454036
        self.u_in_m = 0.0254000508001
        counter=0
        length_max = self.overhang - L_rb + (self.gearbox_cm[0] -self.gearbox_length/2.) #modified length limit 7/29

        while abs(check_limit) > tol and self.L_ms_new < length_max:
            counter =counter+1
            if self.L_ms_new > 0:
                 self.L_ms=self.L_ms_new
            else:
                  self.L_ms=self.L_ms_0

            #-----------------------
            size_LSS_3pt(self)
            #-----------------------

            check_limit = abs(abs(self.theta_y[-1])-Bearing_Limit/self.n_safety_brg)
            #print 'deflection slope'
            #print Bearing_Limit
            #print 'threshold'
            #print theta_y[-1]
            self.L_ms_new = self.L_ms + dL        

        # fatigue check Taylor Parsons 6/2014
        if self.check_fatigue == 1 or 2:
          #start_time = time.time()
          #material properties 34CrNiMo6 steel +QT, large diameter
          self.E=2.1e11
          self.density=7800.0
          self.n_safety = 2.5
          if self.S_ut <= 0:
            self.S_ut=700.0e6 #Pa
          Sm=0.9*self.S_ut #for bending situations, material strength at 10^3 cycles
          C_size=0.6 #diameter larger than 10"
          C_surf=4.51*(self.S_ut/1e6)**-.265 #machined surface 272*(self.S_ut/1e6)**-.995 #forged
          C_temp=1 #normal operating temps
          C_reliab=0.814 #99% reliability
          C_envir=1. #enclosed environment
          Se=C_size*C_surf*C_temp*C_reliab*C_envir*.5*self.S_ut #modified endurance limit for infinite life

          if self.fatigue_exponent!=0:
            if self.fatigue_exponent > 0:
                self.SN_b = - self.fatigue_exponent
            else:
                self.SN_b = self.fatigue_exponent
          else:
            Nfinal = 5e8 #point where fatigue limit occurs under hypothetical S-N curve TODO adjust to fit actual data
            z=log10(1e3)-log10(Nfinal)  #assuming no endurance limit (high strength steel)
            self.SN_b=1/z*log10(Sm/Se)
          self.SN_a=Sm/(1000.**self.SN_b)
          # print 'm:', -1/self.SN_b
          # print 'a:', self.SN_a

          if self.check_fatigue == 1:
              #checks to make sure all inputs are reasonable
              if self.rotor_mass < 100:
                  [self.rotor_mass] = get_rotor_mass(self.machine_rating,False)

              #Rotor Loads calculations using DS472
              setup_Fatigue_Loads(self)

              #upwind diameter calculations
              iterationstep=0.001
              diameter_limit = 1.5
              while True:

                  get_Damage_Brng1(self)

                  if self.Damage < 1 or self.D_max >= diameter_limit:
                      break
                  else:
                      self.D_max+=iterationstep

              #begin bearing calculations
              N_bearings = self.N/self.blade_number #rotation number

              Fz1stoch = (-self.My_stoch)/(self.L_ms)
              Fy1stoch = self.Mz_stoch/self.L_ms
              Fz1determ = (self.weightGbx*self.L_gb - self.LssWeight*.5*self.L_ms - self.rotorWeight*(self.L_ms+L_rb)) / (self.L_ms)

              Fr_range = ((abs(Fz1stoch)+abs(Fz1determ))**2 +Fy1stoch**2)**.5 #radial stochastic + deterministic mean
              Fa_range = self.Fx_stoch*cos(self.shaft_angle) + (self.rotorWeight+self.LssWeight)*sin(self.shaft_angle) #axial stochastic + mean

              life_bearing = self.N_f/self.blade_number

              [self.D_max_a,FW_max,bearingmass] = fatigue_for_bearings(self.D_max, Fr_range, Fa_range, N_bearings, life_bearing, self.mb1Type,False)

          # elif self.check_fatigue == 2:
          #   Fx = self.rotor_thrust_distribution
          #   n_Fx = self.rotor_thrust_count
          #   Fy = self.rotor_Fy_distribution
          #   n_Fy = self.rotor_Fy_count
          #   Fz = self.rotor_Fz_distribution
          #   n_Fz = self.rotor_Fz_count
          #   Mx = self.rotor_torque_distribution
          #   n_Mx = self.rotor_torque_count
          #   My = self.rotor_My_distribution
          #   n_My = self.rotor_My_count
          #   Mz = self.rotor_Mz_distribution
          #   n_Mz = self.rotor_Mz_count

          #   # print n_Fx
          #   # print Fx*.5
          #   # print Mx*.5
          #   # print -1/self.SN_b

          #   def Ninterp(L_ult,L_range,m):
          #       return (L_ult/(.5*L_range))**m #TODO double-check that the input will be the load RANGE instead of load amplitudes. Also, may include means?

          #   #upwind bearing calcs
          #   diameter_limit = 5.0
          #   iterationstep=0.001
          #   #upwind bearing calcs
          #   while True:
          #       self.Damage = 0
          #       Fx_ult = self.SN_a*(pi/4.*(self.D_max**2-self.D_in**2))
          #       Fyz_ult = self.SN_a*(pi*(self.D_max**4-self.D_in**4))/(self.D_max*32*self.L_rb)
          #       Mx_ult = self.SN_a*(pi*(self.D_max**4-self.D_in**4))/(32*(3.**.5)*self.D_max)
          #       Myz_ult = self.SN_a*(pi*(self.D_max**4-self.D_in**4))/(self.D_max*64.)
          #       if Fx_ult and np.all(n_Fx):
          #           self.Damage+=scp.integrate.simps(n_Fx/Ninterp(Fx_ult,Fx,-1/self.SN_b),x=n_Fx,even = 'avg')
          #       if Fyz_ult:
          #           if np.all(n_Fy):
          #               self.Damage+=scp.integrate.simps(abs(n_Fy/Ninterp(Fyz_ult,Fy,-1/self.SN_b)),x=n_Fy,even = 'avg')
          #           if np.all(n_Fz):
          #               self.Damage+=scp.integrate.simps(abs(n_Fz/Ninterp(Fyz_ult,Fz,-1/self.SN_b)),x=n_Fz,even = 'avg')
          #       if Mx_ult and np.all(n_Mx):
          #           self.Damage+=scp.integrate.simps(abs(n_Mx/Ninterp(Mx_ult,Mx,-1/self.SN_b)),x=n_Mx,even = 'avg')
          #       if Myz_ult:
          #           if np.all(n_My):
          #               self.Damage+=scp.integrate.simps(abs(n_My/Ninterp(Myz_ult,My,-1/self.SN_b)),x=n_My,even = 'avg')
          #           if np.all(n_Mz):
          #               self.Damage+=scp.integrate.simps(abs(n_Mz/Ninterp(Myz_ult,Mz,-1/self.SN_b)),x=n_Mz,even = 'avg')

          #       print 'Upwind Bearing Diameter:', self.D_max
          #       print 'self.Damage:', self.Damage

          #       if self.Damage <= 1 or self.D_max >= diameter_limit:
          #           # print 'Upwind Bearing Diameter:', self.D_max
          #           # print 'self.Damage:', self.Damage
          #           #print (time.time() - start_time), 'seconds of total simulation time'
          #           break
          #       else:
          #           self.D_max+=iterationstep

          #   #bearing calcs
          #   if self.availability != 0 and rotor_freq != 0 and self.T_life != 0 and self.cut_out != 0 and self.weibull_A != 0:
          #       N_rotations = self.availability*rotor_freq/60.*(self.T_life*365*24*60*60)*exp(-(self.cut_in/self.weibull_A)**self.weibull_k)-exp(-(self.cut_out/self.weibull_A)**self.weibull_k)
          #   elif np.max(n_Fx > 1e6):
          #       N_rotations = np.max(n_Fx)/self.blade_number
          #   elif np.max(n_My > 1e6):
          #       N_rotations = np.max(n_My)/self.blade_number

          #   # Fz1 = (Fz*(self.L_ms+self.L_rb)+My)/self.L_ms
          #   Fz1_Fz = Fz*(self.L_ms+self.L_rb)/self.L_ms #force in z direction due to Fz
          #   Fz1_My = My/self.L_ms #force in z direction due to My
          #   Fy1_Fy = -Fy*(self.L_ms+self.L_rb)/self.L_ms
          #   Fy1_Mz = Mz/self.L_ms
          #   [self.D_max_a,FW_max,bearingmass] = fatigue2_for_bearings(self.D_max,self.mb1Type,np.zeros(2),np.array([1,2]),Fy1_Fy,n_Fy/self.blade_number,Fz1_Fz,n_Fz/self.blade_number,Fz1_My,n_My/self.blade_number,Fy1_Mz,n_Mz/self.blade_number,N_rotations)
         
        #resize bearing if no fatigue check
        if self.check_fatigue == 0:
            [self.D_max_a,FW_max,bearingmass] = resize_for_bearings(self.D_max,  self.mb1Type,False)

        [self.D_min_a,FW_min,trash] = resize_for_bearings(self.D_min,  self.mb2Type,False) #mb2 is a representation of the gearbox connection
            
        lss_mass_new=(pi/3)*(self.D_max_a**2+self.D_min_a**2+self.D_max_a*self.D_min_a)*(self.L_ms-(FW_max+FW_min)/2)*self.density/4+ \
                         (pi/4)*(self.D_max_a**2-self.D_in**2)*self.density*FW_max+\
                         (pi/4)*(self.D_min_a**2-self.D_in**2)*self.density*FW_min-\
                         (pi/4)*(self.D_in**2)*self.density*(self.L_ms+(FW_max+FW_min)/2)
        lss_mass_new *= 1.35 # add flange and shrink disk mass
        self.length=self.L_ms_new + (FW_max+FW_min)/2 + self.flange_length
        #print ("self.L_ms: {0}").format(self.L_ms)
        #print ("LSS length, m: {0}").format(self.length)
        self.D_outer=self.D_max
        #print ("Upwind MB OD, m: {0}").format(self.D_max_a)
        #print ("CB OD, m: {0}").format(self.D_min_a)
        #print ("self.D_min: {0}").format(self.D_min)
        self.D_in=self.D_in
        self.mass=lss_mass_new
        self.diameter1= self.D_max_a
        self.diameter2= self.D_min_a 
        #self.length=self.L_ms
        #print self.length
        self.D_outer=self.D_max_a
        self.diameter=self.D_max_a

         # calculate mass properties
        downwind_location = np.array([self.gearbox_cm[0]-self.gearbox_length/2. , self.gearbox_cm[1] , self.gearbox_cm[2] ])

        bearing_location1 = np.array([0.,0.,0.]) #upwind
        bearing_location1[0] = downwind_location[0] - self.L_ms*cos(self.shaft_angle)
        bearing_location1[1] = downwind_location[1]
        bearing_location1[2] = downwind_location[2] + self.L_ms*sin(self.shaft_angle)
        self.bearing_location1 = bearing_location1

        self.bearing_location2 = np.array([0.,0.,0.]) #downwind does not exist

        cm = np.array([0.0,0.0,0.0])
        cm[0] = downwind_location[0] - 0.65*self.length*cos(self.shaft_angle) #From solid models, center of mass with flange (not including shrink disk) very nearly .65*total_length
        cm[1] = downwind_location[1]
        cm[2] = downwind_location[2] + 0.65*self.length*sin(self.shaft_angle)

        #including shrink disk mass
        self.cm[0] = (cm[0]*self.mass + downwind_location[0]*self.shrink_disc_mass) / (self.mass+self.shrink_disc_mass) 
        self.cm[1] = cm[1]
        self.cm[2] = (cm[2]*self.mass + downwind_location[2]*self.shrink_disc_mass) / (self.mass+self.shrink_disc_mass)
        # print 'shaft before shrink disk:', self.mass
        self.mass+=self.shrink_disc_mass

        I = np.array([0.0, 0.0, 0.0])
        I[0]  = self.mass * (self.D_in ** 2.0 + self.D_outer ** 2.0) / 8.0
        I[1]  = self.mass * (self.D_in ** 2.0 + self.D_outer ** 2.0 + (4.0 / 3.0) * (self.length ** 2.0)) / 16.0
        I[2]  = I[1]
        self.I = I

        # print 'self.L_rb %8.f' %(self.L_rb) #*(self.machine_rating/5.0e3)   #distance from hub center to main bearing scaled off NREL 5MW
        # print 'L_bg %8.f' %(L_bg) #*(self.machine_rating/5.0e3)         #distance from hub center to gearbox yokes
        # print 'L_as %8.f' %(L_as) #distance from main bearing to shaft center
      
        self.FW_mb=FW_max
        self.bearing_mass1 = bearingmass
        self.bearing_mass2 = 0.

#-------------------------------------------------------------------------------

class LowSpeedShaft_drive(Component):
    ''' LowSpeedShaft class
          The LowSpeedShaft class is used to represent the low speed shaft component of a wind turbine drivetrain. This model is outdated and does not contain fatigue analysis
          It contains the general properties for a wind turbine component as well as additional design load and dimentional attributes as listed below.
          It contains an update method to determine the mass, mass properties, and dimensions of the component.
    '''

    # variables
    rotor_torque = Float(iotype='in', units='N*m', desc='The torque load due to aerodynamic forces on the rotor')
    rotor_bending_moment = Float(iotype='in', units='N*m', desc='The bending moment from uneven aerodynamic loads')
    rotor_mass = Float(iotype='in', units='kg', desc='rotor mass')
    rotor_diameter = Float(iotype='in', units='m', desc='rotor diameter')
    rotor_speed = Float(iotype='in', units='rpm', desc='rotor speed at rated power')
    machine_rating = Float(iotype='in', units='kW', desc='machine_rating machine rating of the turbine')

    # parameters
    shaft_angle = Float(iotype='in', units='deg', desc='Angle of the LSS inclindation with respect to the horizontal')
    shaft_length = Float(iotype='in', units='m', desc='length of low speed shaft')
    shaftD1 = Float(iotype='in', units='m', desc='Fraction of LSS distance from gearbox to downwind main bearing')
    shaftD2 = Float(iotype='in', units='m', desc='raction of LSS distance from gearbox to upwind main bearing')
    shaft_ratio = Float(iotype='in', desc='Ratio of inner diameter to outer diameter.  Leave zero for solid LSS')
    
    # outputs
    design_torque = Float(iotype='out', units='N*m', desc='lss design torque')
    design_bending_load = Float(iotype='out', units='N', desc='lss design bending load')
    length = Float(iotype='out', units='m', desc='lss length')
    diameter = Float(iotype='out', units='m', desc='lss outer diameter')
    mass = Float(0.0, iotype='out', units='kg', desc='overall component mass')
    cm = Array(np.array([0.0, 0.0, 0.0]), iotype='out', desc='center of mass of the component in [x,y,z] for an arbitrary coordinate system')
    I = Array(np.array([0.0, 0.0, 0.0]), iotype='out', desc=' moments of Inertia for the component [Ixx, Iyy, Izz] around its center of mass')    

    def __init__(self):
        '''
        Initializes low speed shaft component
        '''

        super(LowSpeedShaft_drive, self).__init__()
    
    def execute(self):    

        size_LowSpeedShaft(self)

#-------------------------------------------------------------------------------

class Bearing_drive(Component): 
    ''' MainBearings class          
          The MainBearings class is used to represent the main bearing components of a wind turbine drivetrain. It contains two subcomponents (main bearing and second bearing) which also inherit from the SubComponent class.
          It contains the general properties for a wind turbine component as well as additional design load and dimentional attributes as listed below.
          It contains an update method to determine the mass, mass properties, and dimensions of the component.           
    '''
    # variables
    bearing_type = Str(iotype='in',desc='Main bearing type: CARB, TRB1 or SRB')
    bearing_mass = Float(iotype ='in', units = 'kg', desc = 'bearing mass from LSS model')
    lss_diameter = Float(iotype='in', units='m', desc='lss outer diameter at main bearing')
    lss_design_torque = Float(iotype='in', units='N*m', desc='lss design torque')
    rotor_diameter = Float(iotype='in', units='m', desc='rotor diameter')
    location = Array(np.array([0.,0.,0.]),iotype = 'in', units = 'm', desc = 'x,y,z location from shaft model')

    # returns
    mass = Float(0.0, iotype='out', units='kg', desc='overall component mass')
    cm = Array(np.array([0.0, 0.0, 0.0]), iotype='out', desc='center of mass of the component in [x,y,z] for an arbitrary coordinate system')
    I = Array(np.array([0.0, 0.0, 0.0]), iotype='out', desc=' moments of Inertia for the component [Ixx, Iyy, Izz] around its center of mass')
    
    def __init__(self):
        
        super(Bearing_drive, self).__init__()
    
    def execute(self):
        self.mass = self.bearing_mass
        self.mass += self.mass*(8000.0/2700.0) #add housing weight

class MainBearing_drive(Bearing_drive): 
    ''' MainBearings class          
          The MainBearings class is used to represent the main bearing components of a wind turbine drivetrain. It contains two subcomponents (main bearing and second bearing) which also inherit from the SubComponent class.
          It contains the general properties for a wind turbine component as well as additional design load and dimentional attributes as listed below.
          It contains an update method to determine the mass, mass properties, and dimensions of the component.           
    '''
    
    def __init__(self):
        ''' Initializes main bearing component 
        '''
        
        super(MainBearing_drive, self).__init__()
    
    def execute(self):

        super(MainBearing_drive, self).execute()
        
        # calculate mass properties
        inDiam  = self.lss_diameter
        depth = (inDiam * 1.5)

        if self.location[0] != 0.0:
            self.cm = self.location

        else:
            cmMB = np.array([0.0,0.0,0.0])
            cmMB = ([- (0.035 * self.rotor_diameter), 0.0, 0.025 * self.rotor_diameter])
            self.cm = cmMB
       
        b1I0 = (self.mass * inDiam ** 2 ) / 4.0 
        self.I = ([b1I0, b1I0 / 2.0, b1I0 / 2.0])

#-------------------------------------------------------------------------------

class SecondBearing_drive(Bearing_drive): 
    ''' MainBearings class          
          The MainBearings class is used to represent the main bearing components of a wind turbine drivetrain. It contains two subcomponents (main bearing and second bearing) which also inherit from the SubComponent class.
          It contains the general properties for a wind turbine component as well as additional design load and dimentional attributes as listed below.
          It contains an update method to determine the mass, mass properties, and dimensions of the component.           
    '''
    
    def __init__(self):
        ''' Initializes second bearing component 
        '''
        
        super(SecondBearing_drive, self).__init__()
    
    def execute(self):

        super(SecondBearing_drive, self).execute()

        # calculate mass properties
        inDiam  = self.lss_diameter
        depth = (inDiam * 1.5)

        if self.mass > 0 and self.location[0] != 0.0:
            self.cm = self.location
        else:
            self.cm = np.array([0,0,0])
            self.mass = 0.


        b2I0  = (self.mass * inDiam ** 2 ) / 4.0 
        self.I = ([b2I0, b2I0 / 2.0, b2I0 / 2.0])

#-------------------------------------------------------------------------------


class Gearbox_drive(Component):
    ''' Gearbox class
          The Gearbox class is used to represent the gearbox component of a wind turbine drivetrain.
          It contains the general properties for a wind turbine component as well as additional design load and dimentional attributes as listed below.
          It contains an update method to determine the mass, mass properties, and dimensions of the component.
    '''

    #variables
    
    gear_ratio = Float(iotype='in', desc='overall gearbox speedup ratio')
    Np = Array(np.array([0.0,0.0,0.0,]), iotype='in', desc='number of planets in each stage')
    rotor_speed = Float(iotype='in', desc='rotor rpm at rated power')
    rotor_diameter = Float(iotype='in', desc='rotor diameter')
    rotor_torque = Float(iotype='in', units='N*m', desc='rotor torque at rated power')
    cm_input = Float(0,iotype = 'in', units='m', desc ='gearbox position along x-axis')

    #parameters
    #name = Str(iotype='in', desc='gearbox name')
    gear_configuration = Str(iotype='in', desc='string that represents the configuration of the gearbox (stage number and types)')
    #eff = Float(iotype='in', desc='drivetrain efficiency')
    ratio_type = Str(iotype='in', desc='optimal or empirical stage ratios')
    shaft_type = Str(iotype='in', desc = 'normal or short shaft length')

    # outputs
    stage_masses = Array(np.array([0.0, 0.0, 0.0, 0.0]), iotype='out', units='kg', desc='individual gearbox stage masses')
    mass = Float(0.0, iotype='out', units='kg', desc='overall component mass')
    cm = Array(np.array([0.0, 0.0, 0.0]), iotype='out', desc='center of mass of the component in [x,y,z] for an arbitrary coordinate system')
    I = Array(np.array([0.0, 0.0, 0.0]), iotype='out', desc=' moments of Inertia for the component [Ixx, Iyy, Izz] around its center of mass')    
    length = Float(iotype='out', units='m', desc='gearbox length')
    height = Float(iotype='out', units='m', desc='gearbox height')
    diameter = Float(iotype='out', units='m', desc='gearbox diameter')


    def __init__(self):
        '''
        Initializes gearbox component
        '''



        super(Gearbox_drive,self).__init__()

    def execute(self):

        self.stageRatio=np.zeros([3,1])

        self.stageTorque = np.zeros([len(self.stageRatio),1]) #filled in when ebxWeightEst is called
        self.stageMass = np.zeros([len(self.stageRatio),1]) #filled in when ebxWeightEst is called
        self.stageType=self.stageTypeCalc(self.gear_configuration)
        #print self.gear_ratio
        #print self.Np
        #print self.ratio_type
        #print self.gear_configuration
        self.stageRatio=self.stageRatioCalc(self.gear_ratio,self.Np,self.ratio_type,self.gear_configuration)
        #print self.stageRatio

        m=self.gbxWeightEst(self.gear_configuration,self.gear_ratio,self.Np,self.ratio_type,self.shaft_type,self.rotor_torque)
        self.mass = float(m)
        self.stage_masses=self.stageMass
        # calculate mass properties

        self.length = (0.012 * self.rotor_diameter)
        self.height = (0.015 * self.rotor_diameter)
        self.diameter = (0.75 * self.height)

        cm0   = self.cm_input
        cm1   = 0.0
        cm2   = 0.4*self.height #TODO validate or adjust factor. origin is modified to be above bedplate top
        self.cm = np.array([cm0, cm1, cm2])

        I0 = self.mass * (self.diameter ** 2 ) / 8 + (self.mass / 2) * (self.height ** 2) / 8
        I1 = self.mass * (0.5 * (self.diameter ** 2) + (2 / 3) * (self.length ** 2) + 0.25 * (self.height ** 2)) / 8
        I2 = I1
        self.I = np.array([I0, I1, I2])

        '''def rotor_torque():
            tq = self.gbxPower*1000 / self.eff / (self.rotor_speed * (pi / 30.0))
            return tq
        '''
     
    def stageTypeCalc(self, config):
        temp=[]
        for character in config:
                if character == 'e':
                    temp.append(2)
                if character == 'p':
                    temp.append(1)
        return temp

    def stageMassCalc(self, indStageRatio,indNp,indStageType):

        '''
        Computes the mass of an individual gearbox stage.

        Parameters
        ----------
        indStageRatio : str
          Speedup ratio of the individual stage in question.
        indNp : int
          Number of planets for the individual stage.
        indStageType : int
          Type of gear.  Use '1' for parallel and '2' for epicyclic.
        '''

        #Application factor to include ring/housing/carrier weight
        Kr=0.4
        Kgamma=1.1

        if indNp == 3:
            Kgamma=1.1
        elif indNp == 4:
            Kgamma=1.1
        elif indNp == 5:
            Kgamma=1.35

        if indStageType == 1:
            indStageMass=1.0+indStageRatio+indStageRatio**2+(1.0/indStageRatio)

        elif indStageType == 2:
            sunRatio=0.5*indStageRatio - 1.0
            indStageMass=Kgamma*((1/indNp)+(1/(indNp*sunRatio))+sunRatio+sunRatio**2+Kr*((indStageRatio-1)**2)/indNp+Kr*((indStageRatio-1)**2)/(indNp*sunRatio))

        return indStageMass
        
    def gbxWeightEst(self, config,overallRatio,Np,ratio_type,shaft_type,torque):


        '''
        Computes the gearbox weight based on a surface durability criteria.
        '''

        ## Define Application Factors ##
        #Application factor for weight estimate
        Ka=0.6
        Kshaft=0.0
        Kfact=0.0

        #K factor for pitting analysis
        if self.rotor_torque < 200000.0:
            Kfact = 850.0
        elif self.rotor_torque < 700000.0:
            Kfact = 950.0
        else:
            Kfact = 1100.0

        #Unit conversion from Nm to inlb and vice-versa
        Kunit=8.029

        # Shaft length factor
        if self.shaft_type == 'normal':
            Kshaft = 1.0
        elif self.shaft_type == 'short':
            Kshaft = 1.25

        #Individual stage torques
        torqueTemp=self.rotor_torque
        for s in range(len(self.stageRatio)):
            #print torqueTemp
            #print self.stageRatio[s]
            self.stageTorque[s]=torqueTemp/self.stageRatio[s]
            torqueTemp=self.stageTorque[s]
            self.stageMass[s]=Kunit*Ka/Kfact*self.stageTorque[s]*self.stageMassCalc(self.stageRatio[s],self.Np[s],self.stageType[s])
        
        gbxWeight=(sum(self.stageMass))*Kshaft
        
        return gbxWeight

    def stageRatioCalc(self, overallRatio,Np,ratio_type,config):
        '''
        Calculates individual stage ratios using either empirical relationships from the Sunderland model or a SciPy constrained optimization routine.
        '''

        K_r=0
                    
        #Assumes we can model everything w/Sunderland model to estimate speed ratio
        if ratio_type == 'empirical':
            if config == 'p': 
                x=[overallRatio]
            if config == 'e':
                x=[overallRatio]
            elif config == 'pp':
                x=[overallRatio**0.5,overallRatio**0.5]
            elif config == 'ep':
                x=[overallRatio/2.5,2.5]
            elif config =='ee':
                x=[overallRatio**0.5,overallRatio**0.5]
            elif config == 'eep':
                x=[(overallRatio/3)**0.5,(overallRatio/3)**0.5,3]
            elif config == 'epp':
                x=[overallRatio**(1.0/3.0),overallRatio**(1.0/3.0),overallRatio**(1.0/3.0)]
            elif config == 'eee':
                x=[overallRatio**(1.0/3.0),overallRatio**(1.0/3.0),overallRatio**(1.0/3.0)]
            elif config == 'ppp':
                x=[overallRatio**(1.0/3.0),overallRatio**(1.0/3.0),overallRatio**(1.0/3.0)]
        
        elif ratio_type == 'optimal':
            x=np.zeros([3,1])

            if config == 'eep':
                x0=[overallRatio**(1.0/3.0),overallRatio**(1.0/3.0),overallRatio**(1.0/3.0)]
                B_1=Np[0]
                B_2=Np[1]
                K_r1=0
                K_r2=0 #2nd stage structure weight coefficient

                def volume(x):
                    return (1.0/(x[0]))*((1.0/B_1)+(1.0/(B_1*((x[0]/2.0)-1.0)))+(x[0]/2.0-1.0)+ \
                    (x[0]/2.0-1)**2+K_r1*((x[0]-1.0)**2)/B_1 + K_r1*((x[0]-1.0)**2)/(B_1*(x[0]/2.0-1.0))) + \
                    (1.0/(x[0]*x[1]))*((1.0/B_2)+(1/(B_2*((x[1]/2.0)-1.0)))+(x[1]/2.0-1.0)+(x[1]/2.0-1.0)**2.0+K_r2*((x[1]-1.0)**2.0)/B_2 + \
                     K_r2*((x[1]-1.0)**2.0)/(B_2*(x[1]/2.0-1.0))) + (1.0/(x[0]*x[1]*x[2]))*(1.0+(1.0/x[2])+x[2] + x[2]**2)                              
                
                def constr1(x,overallRatio):
                    return x[0]*x[1]*x[2]-overallRatio
        
                def constr2(x,overallRatio):
                    return overallRatio-x[0]*x[1]*x[2]

                x=opt.fmin_cobyla(volume, x0,[constr1,constr2],consargs=[overallRatio],rhoend=1e-7, iprint = 0)
        
            elif config == 'eep_3':
                #fixes last stage ratio at 3
                x0=[overallRatio**(1.0/3.0),overallRatio**(1.0/3.0),overallRatio**(1.0/3.0)]
                B_1=Np[0]
                B_2=Np[1]
                K_r1=0
                K_r2=0.8 #2nd stage structure weight coefficient

                def volume(x):
                    return (1.0/(x[0]))*((1.0/B_1)+(1.0/(B_1*((x[0]/2.0)-1.0)))+(x[0]/2.0-1.0)+(x[0]/2.0-1)**2+K_r1*((x[0]-1.0)**2)/B_1 + K_r1*((x[0]-1.0)**2)/(B_1*(x[0]/2.0-1.0))) + (1.0/(x[0]*x[1]))*((1.0/B_2)+(1/(B_2*((x[1]/2.0)-1.0)))+(x[1]/2.0-1.0)+(x[1]/2.0-1.0)**2.0+K_r2*((x[1]-1.0)**2.0)/B_2 + K_r2*((x[1]-1.0)**2.0)/(B_2*(x[1]/2.0-1.0))) + (1.0/(x[0]*x[1]*x[2]))*(1.0+(1.0/x[2])+x[2] + x[2]**2)                              
                
                def constr1(x,overallRatio):
                    return x[0]*x[1]*x[2]-overallRatio
        
                def constr2(x,overallRatio):
                    return overallRatio-x[0]*x[1]*x[2]
                
                def constr3(x,overallRatio):
                    return x[2]-3.0
                
                def constr4(x,overallRatio):
                    return 3.0-x[2]

                x=opt.fmin_cobyla(volume, x0,[constr1,constr2,constr3,constr4],consargs=[overallRatio],rhoend=1e-7,iprint=0)
            
            elif config == 'eep_2':
                #fixes final stage ratio at 2
                x0=[overallRatio**(1.0/3.0),overallRatio**(1.0/3.0),overallRatio**(1.0/3.0)]
                B_1=Np[0]
                B_2=Np[1]
                K_r1=0
                K_r2=1.6 #2nd stage structure weight coefficient

                def volume(x):
                    return (1.0/(x[0]))*((1.0/B_1)+(1.0/(B_1*((x[0]/2.0)-1.0)))+(x[0]/2.0-1.0)+(x[0]/2.0-1)**2+K_r1*((x[0]-1.0)**2)/B_1 + K_r1*((x[0]-1.0)**2)/(B_1*(x[0]/2.0-1.0))) + (1.0/(x[0]*x[1]))*((1.0/B_2)+(1/(B_2*((x[1]/2.0)-1.0)))+(x[1]/2.0-1.0)+(x[1]/2.0-1.0)**2.0+K_r2*((x[1]-1.0)**2.0)/B_2 + K_r2*((x[1]-1.0)**2.0)/(B_2*(x[1]/2.0-1.0))) + (1.0/(x[0]*x[1]*x[2]))*(1.0+(1.0/x[2])+x[2] + x[2]**2)                              
                
                def constr1(x,overallRatio):
                    return x[0]*x[1]*x[2]-overallRatio
        
                def constr2(x,overallRatio):
                    return overallRatio-x[0]*x[1]*x[2]

                x=opt.fmin_cobyla(volume, x0,[constr1,constr2],consargs=[overallRatio],rhoend=1e-7, iprint = 0)
            elif config == 'epp':
                #fixes last stage ratio at 3
                x0=[overallRatio**(1.0/3.0),overallRatio**(1.0/3.0),overallRatio**(1.0/3.0)]
                B_1=Np[0]
                B_2=Np[1]
                K_r=0
               
                def volume(x):
                    return (1.0/(x[0]))*((1.0/B_1)+(1.0/(B_1*((x[0]/2.0)-1.0)))+(x[0]/2.0-1.0)+(x[0]/2.0-1)**2+ \
                    K_r*((x[0]-1.0)**2)/B_1 + K_r*((x[0]-1.0)**2)/(B_1*(x[0]/2.0-1.0))) + \
                    (1.0/(x[0]*x[1]))*(1.0+(1.0/x[1])+x[1] + x[1]**2) \
                    + (1.0/(x[0]*x[1]*x[2]))*(1.0+(1.0/x[2])+x[2] + x[2]**2)                              
                
                def constr1(x,overallRatio):
                    return x[0]*x[1]*x[2]-overallRatio
        
                def constr2(x,overallRatio):
                    return overallRatio-x[0]*x[1]*x[2]
                
                x=opt.fmin_cobyla(volume, x0,[constr1,constr2],consargs=[overallRatio],rhoend=1e-7,iprint=0)
                
            else:  # what is this subroutine for?  Yi on 04/16/2014
                x0=[overallRatio**(1.0/3.0),overallRatio**(1.0/3.0),overallRatio**(1.0/3.0)]
                B_1=Np[0]
                K_r=0.0
                def volume(x):
                    return (1.0/(x[0]))*((1.0/B_1)+(1.0/(B_1*((x[0]/2.0)-1.0)))+(x[0]/2.0-1)+(x[0]/2.0-1.0)**2+K_r*((x[0]-1.0)**2)/B_1 + K_r*((x[0]-1)**2)/(B_1*(x[0]/2.0-1.0))) + (1.0/(x[0]*x[1]))*(1.0+(1.0/x[1])+x[1] + x[1]**2)+ (1.0/(x[0]*x[1]*x[2]))*(1.0+(1.0/x[2])+x[2] + x[2]**2)
                                  
                def constr1(x,overallRatio):
                   return x[0]*x[1]*x[2]-overallRatio
        
                def constr2(x,overallRatio):
                    return overallRatio-x[0]*x[1]*x[2]

                x=opt.fmin_cobyla(volume, x0,[constr1,constr2],consargs=[overallRatio],rhoend=1e-7, iprint = 0)
        else:
            x='fail'
                  
        return x
        
#---------------------------------------------------------------------------------------------------------------

class Bedplate_drive(Component):
    ''' Bedplate class
          The Bedplate class is used to represent the bedplate of a wind turbine drivetrain.
          It contains the general properties for a wind turbine component as well as additional design load and dimentional attributes as listed below.
          It contains an update method to determine the mass, mass properties, and dimensions of the component.
    '''

    #variables
    gbx_length = Float(iotype = 'in', units = 'm', desc = 'gearbox length')
    gbx_location = Float(iotype = 'in', units = 'm', desc = 'gearbox CM location')
    gbx_mass = Float(iotype = 'in', units = 'kg', desc = 'gearbox mass')
    hss_location = Float(iotype ='in', units = 'm', desc='HSS CM location')
    hss_mass = Float(iotype ='in', units = 'kg', desc='HSS mass')
    generator_location = Float(iotype ='in', units = 'm', desc='generator CM location')
    generator_mass = Float(iotype ='in', units = 'kg', desc='generator mass')
    lss_location = Float(iotype ='in', units = 'm', desc='LSS CM location')
    lss_mass = Float(iotype ='in', units = 'kg', desc='LSS mass')
    lss_length = Float(iotype = 'in', units = 'm', desc = 'LSS length')
    mb1_location = Float(iotype ='in', units = 'm', desc='Upwind main bearing CM location')
    FW_mb1 = Float(iotype = 'in', units = 'm', desc = 'Upwind main bearing facewidth')
    mb1_mass = Float(iotype ='in', units = 'kg', desc='Upwind main bearing mass')
    mb2_location = Float(iotype ='in', units = 'm', desc='Downwind main bearing CM location')
    mb2_mass = Float(iotype ='in', units = 'kg', desc='Downwind main bearing mass')
    transformer_mass = Float(iotype ='in', units = 'kg', desc='Transformer mass')
    transformer_location = Float(iotype = 'in', units = 'm', desc = 'transformer CM location')
    tower_top_diameter = Float(iotype ='in', units = 'm', desc='diameter of the top tower section at the yaw gear')
    rotor_diameter = Float(iotype = 'in', units = 'm', desc='rotor diameter')
    machine_rating = Float(iotype='in', units='kW', desc='machine_rating machine rating of the turbine')
    rotor_mass = Float(iotype='in', units='kg', desc='rotor mass')
    rotor_bending_moment_y = Float(iotype='in', units='N*m', desc='The bending moment about the y axis')
    rotor_force_z = Float(iotype='in', units='N', desc='The force along the z axis applied at hub center')
    flange_length = Float(iotype='in', units='m', desc='flange length')
    L_rb = Float(iotype = 'in', units = 'm', desc = 'length between rotor center and upwind main bearing')
    overhang = Float(iotype='in', units='m', desc='Overhang distance')

    #parameters
    uptower_transformer = Bool(iotype = 'in', desc = 'Boolean stating if transformer is uptower')

    #outputs
    mass = Float(0.0, iotype='out', units='kg', desc='overall component mass')
    cm = Array(np.array([0.0, 0.0, 0.0]), iotype='out', desc='center of mass of the component in [x,y,z] for an arbitrary coordinate system')
    I = Array(np.array([0.0, 0.0, 0.0]), iotype='out', desc=' moments of Inertia for the component [Ixx, Iyy, Izz] around its center of mass')    
    length = Float(iotype='out', units='m', desc='length of bedplate')
    height = Float(iotype='out', units='m', desc='max height of bedplate')
    width = Float(iotype='out', units='m', desc='width of bedplate')

    def __init__(self):
        ''' Initializes bedplate component
        '''

        super(Bedplate_drive,self).__init__()

    def execute(self):
        #Model bedplate as 2 parallel I-beams with a rear steel frame and a front cast frame
        #Deflection constraints applied at each bedplate end
        #Stress constraint checked at root of front and rear bedplate sections
        setup_Bedplate(self)
        counter = 0
        while self.rootStress*self.stress_mult - self.stressMax >  self.stressTol or self.totalTipDefl - self.deflMax >  self.deflTol:

          counter += 1

          characterize_Bedplate_Rear(self)

          self.tf += 0.002 
          self.tw += 0.002
          self.b0 += 0.006
          self.h0 += 0.006
          rearCounter = counter

        self.rearHeight = self.h0

        #Front cast section:
        setup_Bedplate_Front(self)

        counter = 0

        while self.rootStress*self.stress_mult - self.stressMax >  self.stressTol or self.totalTipDefl - self.deflMax >  self.deflTol:
          counter += 1
          characterize_Bedplate_Front(self)
          self.tf += 0.002 
          self.tw += 0.002
          self.b0 += 0.006
          self.h0 += 0.006

          frontCounter=counter

        size_Bedplate(self)
        
        # print 'front length and mass:', frontTotalLength, totalCastMass
        # print 'rear length and mass:', rearTotalLength, totalSteelMass 
        
#---------------------------------------------------------------------------------------------------------------

class YawSystem_drive(Component):
    ''' YawSystem class
          The YawSystem class is used to represent the yaw system of a wind turbine drivetrain.
          It contains the general properties for a wind turbine component as well as additional design load and dimentional attributes as listed below.
          It contains an update method to determine the mass, mass properties, and dimensions of the component.
    '''
    #variables
    rotor_diameter = Float(iotype='in', units='m', desc='rotor diameter')
    rotor_thrust = Float(iotype='in', units='N', desc='maximum rotor thrust')
    tower_top_diameter = Float(iotype='in', units='m', desc='tower top diameter')
    above_yaw_mass = Float(iotype='in', units='kg', desc='above yaw mass')
    bedplate_height = Float(iotype = 'in', units = 'm', desc = 'bedplate height')

    #parameters
    yaw_motors_number = Int(0,iotype='in', desc='number of yaw motors')

    #outputs
    mass = Float(0.0, iotype='out', units='kg', desc='overall component mass')
    cm = Array(np.array([0.0, 0.0, 0.0]), iotype='out', desc='center of mass of the component in [x,y,z] for an arbitrary coordinate system')
    I = Array(np.array([0.0, 0.0, 0.0]), iotype='out', desc=' moments of Inertia for the component [Ixx, Iyy, Izz] around its center of mass')    


    def __init__(self):
        ''' Initializes yaw system
        '''
        super(YawSystem_drive, self).__init__()

    def execute(self):
        
        size_YawSystem(self)

        #-------------------------------------------------------------------------------

class Transformer_drive(Component):
    ''' Transformer class
            The transformer class is used to represent the transformer of a wind turbine drivetrain.
            It contains the general properties for a wind turbine component as well as additional design load and dimentional attributes as listed below.
            It contains an update method to determine the mass, mass properties, and dimensions of the component if it is in fact uptower'''

    #inputs
    machine_rating = Float(iotype='in', units='kW', desc='machine rating of the turbine')
    uptower_transformer = Bool(iotype='in', desc = 'uptower or downtower transformer')
    tower_top_diameter = Float(iotype = 'in', units = 'm', desc = 'tower top diameter for comparision of nacelle CM')
    rotor_mass = Float(iotype='in', units='kg', desc='rotor mass')
    overhang = Float(iotype='in', units='m', desc='rotor overhang distance')
    generator_cm = Array(iotype='in', desc='center of mass of the generator in [x,y,z]')
    rotor_diameter = Float(iotype='in',units='m', desc='rotor diameter of turbine')
    RNA_mass = Float(iotype = 'in', units='kg', desc='mass of total RNA')
    RNA_cm = Float(iotype='in', units='m', desc='RNA CM along x-axis')

    #outputs
    mass = Float(0.0, iotype='out', units='kg', desc='overall component mass')
    cm = Array(np.array([0.0, 0.0, 0.0]), iotype='out', desc='center of mass of the component in [x,y,z] for an arbitrary coordinate system')
    I = Array(np.array([0.0, 0.0, 0.0]), iotype='out', desc=' moments of Inertia for the component [Ixx, Iyy, Izz] around its center of mass')    

    def __init__(self):
        '''
        Initializes transformer component
        '''

        super(Transformer_drive, self).__init__()

        self.missing_deriv_policy = 'assume_zero'

    def execute(self):

        size_Transformer(self)

#-------------------------------------------------------------------

class HighSpeedSide_drive(Component):
    '''
    HighSpeedShaft class
          The HighSpeedShaft class is used to represent the high speed shaft and mechanical brake components of a wind turbine drivetrain.
          It contains the general properties for a wind turbine component as well as additional design load and dimentional attributes as listed below.
          It contains an update method to determine the mass, mass properties, and dimensions of the component.
    '''

    # variables
    rotor_diameter = Float(iotype='in', units='m', desc='rotor diameter')
    rotor_torque = Float(iotype='in', units='N*m', desc='rotor torque at rated power')
    gear_ratio = Float(iotype='in', desc='overall gearbox ratio')
    lss_diameter = Float(iotype='in', units='m', desc='low speed shaft outer diameter')
    gearbox_length = Float(iotype = 'in', units = 'm', desc='gearbox length')
    gearbox_height = Float(iotype='in', units = 'm', desc = 'gearbox height')
    gearbox_cm = Array(iotype = 'in', units = 'm', desc = 'gearbox cm [x,y,z]')
    length_in = Float(iotype = 'in', units = 'm', desc = 'high speed shaft length determined by user. Default 0.5m')

    # returns
    mass = Float(0.0, iotype='out', units='kg', desc='overall component mass')
    cm = Array(np.array([0.0, 0.0, 0.0]), iotype='out', desc='center of mass of the component in [x,y,z] for an arbitrary coordinate system')
    I = Array(np.array([0.0, 0.0, 0.0]), iotype='out', desc=' moments of Inertia for the component [Ixx, Iyy, Izz] around its center of mass')
    length = Float(iotype='out', desc='length of high speed shaft')

    def __init__(self):
        '''
        Initializes high speed side component
        '''

        super(HighSpeedSide_drive, self).__init__()

        #controls what happens if derivatives are missing
        self.missing_deriv_policy = 'assume_zero'

    def execute(self):

        size_HighSpeedSide(self)

#----------------------------------------------------------------------------------------------

class Generator_drive(Component):
    '''Generator class
          The Generator class is used to represent the generator of a wind turbine drivetrain.
          It contains the general properties for a wind turbine component as well as additional design load and dimentional attributes as listed below.
          It contains an update method to determine the mass, mass properties, and dimensions of the component.
    '''

    # variables
    rotor_diameter = Float(iotype='in', units='m', desc='rotor diameter')
    machine_rating = Float(iotype='in', units='kW', desc='machine rating of generator')
    gear_ratio = Float(iotype='in', desc='overall gearbox ratio')
    highSpeedSide_length = Float( iotype = 'in', units = 'm', desc='length of high speed shaft and brake')
    highSpeedSide_cm = Array(np.array([0.0,0.0,0.0]), iotype = 'in', units = 'm', desc='cm of high speed shaft and brake')
    rotor_speed = Float(iotype='in', units='rpm', desc='Speed of rotor at rated power')

    # parameters
    drivetrain_design = Enum('geared', ('geared', 'single_stage', 'multi_drive', 'pm_direct_drive'), iotype='in')

    # returns
    mass = Float(0.0, iotype='out', units='kg', desc='overall component mass')
    cm = Array(np.array([0.0, 0.0, 0.0]), iotype='out', desc='center of mass of the component in [x,y,z] for an arbitrary coordinate system')
    I = Array(np.array([0.0, 0.0, 0.0]), iotype='out', desc=' moments of Inertia for the component [Ixx, Iyy, Izz] around its center of mass')

    def __init__(self):
        '''
        Initializes generator component
        '''

        super(Generator_drive, self).__init__()

        #controls what happens if derivatives are missing
        self.missing_deriv_policy = 'assume_zero'

    def execute(self):
        size_Generator(self)

#-------------------------------------------------------------------------------

class AboveYawMassAdder_drive(Component):

    # variables
    machine_rating = Float(iotype = 'in', units='kW', desc='machine rating')
    lss_mass = Float(iotype = 'in', units='kg', desc='component mass')
    main_bearing_mass = Float(iotype = 'in', units='kg', desc='component mass')
    second_bearing_mass = Float(iotype = 'in', units='kg', desc='component mass')
    gearbox_mass = Float(iotype = 'in', units='kg', desc='component mass')
    hss_mass = Float(iotype = 'in', units='kg', desc='component mass')
    generator_mass = Float(iotype = 'in', units='kg', desc='component mass')
    bedplate_mass = Float(iotype = 'in', units='kg', desc='component mass')
    bedplate_length = Float(iotype = 'in', units='m', desc='component length')
    bedplate_width = Float(iotype = 'in', units='m', desc='component width')
    transformer_mass = Float(iotype = 'in', units='kg', desc='component mass')

    # parameters
    crane = Bool(iotype='in', desc='flag for presence of crane')

    # returns
    electrical_mass = Float(iotype = 'out', units='kg', desc='component mass')
    vs_electronics_mass = Float(iotype = 'out', units='kg', desc='component mass')
    hvac_mass = Float(iotype = 'out', units='kg', desc='component mass')
    controls_mass = Float(iotype = 'out', units='kg', desc='component mass')
    platforms_mass = Float(iotype = 'out', units='kg', desc='component mass')
    crane_mass = Float(iotype = 'out', units='kg', desc='component mass')
    mainframe_mass = Float(iotype = 'out', units='kg', desc='component mass')
    cover_mass = Float(iotype = 'out', units='kg', desc='component mass')
    above_yaw_mass = Float(iotype = 'out', units='kg', desc='total mass above yaw system')
    length = Float(iotype = 'out', units='m', desc='component length')
    width = Float(iotype = 'out', units='m', desc='component width')
    height = Float(iotype = 'out', units='m', desc='component height')

    def __init__(self):
        ''' Initialize above yaw mass adder component
        '''

        super(AboveYawMassAdder_drive, self).__init__()

        #controls what happens if derivatives are missing
        self.missing_deriv_policy = 'assume_zero'

    def execute(self):

        add_AboveYawMass(self)

#--------------------------------------------
class RNASystemAdder_drive(Component):
    ''' RNASystem class
          This analysis is only to be used in placing the transformer of the drivetrain.
          The Rotor-Nacelle-Assembly class is used to represent the RNA of the turbine without the transformer and bedplate (to resolve circular dependency issues).
          It contains the general properties for a wind turbine component as well as additional design load and dimentional attributes as listed below.
          It contains an update method to determine the mass, mass properties, and dimensions of the component. 
    '''
    #inputs
    yawMass = Float(iotype='in', units='kg', desc='mass of yaw system')
    lss_mass = Float(iotype = 'in', units='kg', desc='component mass')
    main_bearing_mass = Float(iotype = 'in', units='kg', desc='component mass')
    second_bearing_mass = Float(iotype = 'in', units='kg', desc='component mass')
    gearbox_mass = Float(iotype = 'in', units='kg', desc='component mass')
    hss_mass = Float(iotype = 'in', units='kg', desc='component mass')
    generator_mass = Float(iotype = 'in', units='kg', desc='component mass')
    lss_cm = Array(np.array([0.0,0.0,0.0]),iotype = 'in', units='kg', desc='component CM')
    main_bearing_cm = Array(np.array([0.0,0.0,0.0]),iotype = 'in', units='kg', desc='component CM')
    second_bearing_cm = Array(np.array([0.0,0.0,0.0]),iotype = 'in', units='kg', desc='component CM')
    gearbox_cm = Array(np.array([0.0,0.0,0.0]),iotype = 'in', units='kg', desc='component CM')
    hss_cm = Array(np.array([0.0,0.0,0.0]),iotype = 'in', units='kg', desc='component CM')
    generator_cm = Array(np.array([0.0,0.0,0.0]),iotype = 'in', units='kg', desc='component CM')
    overhang = Float(iotype = 'in', units='m', desc='nacelle overhang')
    rotor_mass = Float(iotype = 'in', units='kg', desc='component mass')
    machine_rating = Float(iotype = 'in', units = 'kW', desc = 'machine rating ')

    #returns
    RNA_mass = Float(iotype = 'out', units='kg', desc='mass of total RNA')
    RNA_cm = Float(iotype='out', units='m', desc='RNA CM along x-axis')

    def __init__(self):
        ''' Initialize RNA Adder component
        '''

        super(RNASystemAdder_drive , self).__init__()

        #controls what happens if derivatives are missing
        self.missing_deriv_policy = 'assume_zero'

    def execute(self):

        add_RNA(self)
        

#--------------------------------------------
class NacelleSystemAdder_drive(Component): #added to drive to include transformer
    ''' NacelleSystem class
          The Nacelle class is used to represent the overall nacelle of a wind turbine.
          It contains the general properties for a wind turbine component as well as additional design load and dimentional attributes as listed below.
          It contains an update method to determine the mass, mass properties, and dimensions of the component.
    '''
    # variables
    above_yaw_mass = Float(iotype='in', units='kg', desc='mass above yaw system')
    yawMass = Float(iotype='in', units='kg', desc='mass of yaw system')
    lss_mass = Float(iotype = 'in', units='kg', desc='component mass')
    main_bearing_mass = Float(iotype = 'in', units='kg', desc='component mass')
    second_bearing_mass = Float(iotype = 'in', units='kg', desc='component mass')
    gearbox_mass = Float(iotype = 'in', units='kg', desc='component mass')
    hss_mass = Float(iotype = 'in', units='kg', desc='component mass')
    generator_mass = Float(iotype = 'in', units='kg', desc='component mass')
    bedplate_mass = Float(iotype = 'in', units='kg', desc='component mass')
    mainframe_mass = Float(iotype = 'in', units='kg', desc='component mass')
    lss_cm = Array(np.array([0.0,0.0,0.0]),iotype = 'in', units='kg', desc='component CM')
    main_bearing_cm = Array(np.array([0.0,0.0,0.0]),iotype = 'in', units='kg', desc='component CM')
    second_bearing_cm = Array(np.array([0.0,0.0,0.0]),iotype = 'in', units='kg', desc='component CM')
    gearbox_cm = Array(np.array([0.0,0.0,0.0]),iotype = 'in', units='kg', desc='component CM')
    hss_cm = Array(np.array([0.0,0.0,0.0]),iotype = 'in', units='kg', desc='component CM')
    generator_cm = Array(np.array([0.0,0.0,0.0]),iotype = 'in', units='kg', desc='component CM')
    bedplate_cm = Array(np.array([0.0,0.0,0.0]),iotype = 'in', units='kg', desc='component CM')
    lss_I = Array(np.array([0.0,0.0,0.0]),iotype = 'in', units='kg', desc='component I')
    main_bearing_I = Array(np.array([0.0,0.0,0.0]),iotype = 'in', units='kg', desc='component I')
    second_bearing_I = Array(np.array([0.0,0.0,0.0]),iotype = 'in', units='kg', desc='component I')
    gearbox_I = Array(np.array([0.0,0.0,0.0]),iotype = 'in', units='kg', desc='component I')
    hss_I = Array(np.array([0.0,0.0,0.0]),iotype = 'in', units='kg', desc='component I')
    generator_I = Array(np.array([0.0,0.0,0.0]),iotype = 'in', units='kg', desc='component I')
    bedplate_I = Array(np.array([0.0,0.0,0.0]),iotype = 'in', units='kg', desc='component I')
    transformer_mass = Float(iotype = 'in', units='kg', desc='component mass')
    transformer_cm = Array(np.array([0.0,0.0,0.0]),iotype = 'in', units='kg', desc='component CM')
    transformer_I = Array(np.array([0.0,0.0,0.0]),iotype = 'in', units='kg', desc='component I')

    # returns
    nacelle_mass = Float(0.0, iotype='out', units='kg', desc='overall component mass')
    nacelle_cm = Array(np.array([0.0, 0.0, 0.0]), units='m', iotype='out', desc='center of mass of the component in [x,y,z] for an arbitrary coordinate system')
    nacelle_I = Array(np.array([0.0, 0.0, 0.0]), units='kg*m**2', iotype='out', desc=' moments of Inertia for the component [Ixx, Iyy, Izz] around its center of mass')

    def __init__(self):
        ''' Initialize above yaw mass adder component
        '''

        super(NacelleSystemAdder_drive , self).__init__()

        #controls what happens if derivatives are missing
        self.missing_deriv_policy = 'assume_zero'

    def execute(self):

        add_Nacelle(self)


if __name__ == '__main__':
     pass
