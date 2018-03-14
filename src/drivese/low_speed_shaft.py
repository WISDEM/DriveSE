
from openmdao.api import Component

import numpy as np
from math import pi, cos, sqrt, radians, sin, exp, log10, log, floor, ceil
import algopy
import scipy as scp
import scipy.optimize as opt
from scipy import integrate

from drivese_utils import fatigue_for_bearings, resize_for_bearings, get_rotor_mass, get_L_rb, get_My, get_Mz


class LowSpeedShaft_Base(Component):

    def __init__(self):
        super(LowSpeedShaft_Base, self).__init__()

        # variables
        self.add_param('rotor_bending_moment_x', val=0.0, units='N*m', desc='The bending moment about the x axis')
        self.add_param('rotor_bending_moment_y', val=0.0, units='N*m', desc='The bending moment about the y axis')
        self.add_param('rotor_bending_moment_z', val=0.0, units='N*m', desc='The bending moment about the z axis')
        self.add_param('rotor_force_x', val=0.0, units='N', desc='The force along the x axis applied at hub center')
        self.add_param('rotor_force_y', val=0.0, units='N', desc='The force along the y axis applied at hub center')
        self.add_param('rotor_force_z', val=0.0, units='N', desc='The force along the z axis applied at hub center')
        self.add_param('rotor_mass', val=0.0, units='kg', desc='rotor mass')
        self.add_param('rotor_diameter', val=0.0, units='m', desc='rotor diameter')
        self.add_param('machine_rating', val=0.0, units='kW', desc='machine_rating machine rating of the turbine')
        self.add_param('gearbox_mass', val=0.0, units='kg', desc='Gearbox mass')
        self.add_param('carrier_mass', val=0.0, units='kg', desc='Carrier mass')
        self.add_param('overhang', val=0.0, units='m', desc='Overhang distance')
        self.add_param('L_rb', val=0.0, units='m', desc='distance between hub center and upwind main bearing')
        self.add_param('DrivetrainEfficiency', val=0.0 desc='overall drivettrain efficiency')

        # fatigue1 variables
        self.add_param('rotor_freq', val=0.0, units='rpm', desc='rated rotor speed')
        self.add_param('fatigue_exponent', val=0.0 desc='fatigue exponent of material')
        self.add_param('S_ut', val=0.0700e6, units='Pa', desc='ultimate tensile strength of material')
        self.add_param('weibull_A', val=0.0, units='m/s', desc='weibull scale parameter "A" of 10-minute windspeed probability distribution')
        self.add_param('weibull_k', val=0.0 desc='weibull shape parameter "k" of 10-minute windspeed probability distribution')
        self.add_param('blade_number', val=0.0 desc='number of blades on rotor, 2 or 3')
        self.add_param('cut_in', val=0.0, units='m/s', desc='cut-in windspeed')
        self.add_param('cut_out', val=0.0, units='m/s', desc='cut-out windspeed')
        self.add_param('Vrated', val=0.0, units='m/s', desc='rated windspeed')
        self.add_param('T_life', val=0.0, units='yr', desc='design life')
        self.add_param('availability', val=0.0.95, desc='turbine availability')

        # fatigue2 variables
        self.add_param('rotor_thrust_distribution', val=np.array([]), units='N', desc='thrust distribution across turbine life')
        self.add_param('rotor_thrust_count', val=np.array([]), desc='corresponding cycle array for thrust distribution')
        self.add_param('rotor_Fy_distribution', val=np.array([]), units='N', desc='Fy distribution across turbine life')
        self.add_param('rotor_Fy_count', val=np.array([]), desc='corresponding cycle array for Fy distribution')
        self.add_param('rotor_Fz_distribution', val=np.array([]), units='N', desc='Fz distribution across turbine life')
        self.add_param('rotor_Fz_count', val=np.array([]), desc='corresponding cycle array for Fz distribution')
        self.add_param('rotor_torque_distribution', val=np.array([]), units='N*m', desc='torque distribution across turbine life')
        self.add_param('rotor_torque_count', val=np.array([]), desc='corresponding cycle array for torque distribution')
        self.add_param('rotor_My_distribution', val=np.array([]), units='N*m', desc='My distribution across turbine life')
        self.add_param('rotor_My_count', val=np.array([]), desc='corresponding cycle array for My distribution')
        self.add_param('rotor_Mz_distribution', val=np.array([]), units='N*m', desc='Mz distribution across turbine life')
        self.add_param('rotor_Mz_count', val=np.array([]), desc='corresponding cycle array for Mz distribution')

        # parameters
        self.add_param('shrink_disc_mass', val=0.0, units='kg', desc='Mass of the shrink disc')
        self.add_param('gearbox_cm', val=np.array([]), units='m', desc='center of mass of gearbox')
        self.add_param('gearbox_length', val=0.0, units='m', desc='gearbox length')
        self.add_param('flange_length', val=0.0, units='m', desc='flange length')
        self.add_param('shaft_angle', val=0.0, units='rad', desc='Angle of the LSS inclindation with respect to the horizontal')
        self.add_param('shaft_ratio', val=0.0 desc='Ratio of inner diameter to outer diameter.  Leave zero for solid LSS')
        self.add_param('mb1Type', val='SRB', ('CARB', 'TRB1', 'TRB2', 'SRB', 'CRB', 'RB'), desc='Main bearing type')
        self.add_param('mb2Type', val='SRB', ('CARB', 'TRB1', 'TRB2', 'SRB', 'CRB', 'RB'), desc='Second bearing type')
        self.add_param('check_fatigue', val=0, (0, 1, 2), desc='turns on and off fatigue check')
        self.add_param('IEC_Class', val='A', ('A', 'B', 'C'), desc='IEC class letter: A, B, or C')

        
        # outputs
        self.add_output('design_torque', val=0.0,  units='N*m', desc='lss design torque')
        self.add_output('design_bending_load', val=0.0,  units='N', desc='lss design bending load')
        self.add_output('length', val=0.0, units='m', desc='lss length')
        self.add_output('diameter1', val=0.0, units='m',  desc='lss outer diameter at main bearing')
        self.add_output('diameter2', val=0.0, units='m',  desc='lss outer diameter at second bearing')
        self.add_output('mass', val=0.0, units='kg', desc='overall component mass')
        self.add_output('cm', val=np.array([0.0, 0.0, 0.0]), desc='center of mass of the component in [x,y,z] for an arbitrary coordinate system')
        self.add_output('I', val=np.array([0.0, 0.0, 0.0]), desc=' moments of Inertia for the component [Ixx, Iyy, Izz] around its center of mass')
        self.add_output('FW_mb1', val=0.0, units='m',  desc='facewidth of upwind main bearing')
        self.add_output('FW_mb2', val=0.0, units='m',  desc='facewidth of main bearing')
        self.add_output('bearing_mass1', val=0.0,  units='kg', desc='main bearing mass')
        self.add_output('bearing_mass2', val=0.0, units='kg',  desc='second bearing mass')
        self.add_output('bearing_location1', val=np.array([0, 0, 0]), units='m', desc='main bearing 1 center of mass')
        self.add_output('bearing_location2', val=np.array([0, 0, 0]), units='m', desc='main bearing 2 center of mass')

        
    def get_Damage_Brng2(self):
        I = (pi / 64.0) * (self.D_med**4 - self.D_in**4)
        J = I * 2
        Area = pi / 4. * (self.D_med**2 - self.D_in**2)
        self.LssWeight = self.density * 9.81 * \
            (((pi / 12) * (self.D_max**2 + self.D_med**2 + self.D_max *
                           self.D_med) * (self.L_mb)) - (pi / 4 * self.L_mb * self.D_in**2))

        self.Fz1stoch = (-self.My_stoch) / (self.L_mb)
        self.Fy1stoch = self.Mz_stoch / self.L_mb
        self.My2stoch = 0.  # My_stoch - abs(Fz1stoch)*self.L_mb #=0
        self.Mz2stoch = 0.  # Mz_stoch - abs(Fy1stoch)*self.L_mb #=0

        # create stochastic loads across N
        stoch_bend2 = (self.My2stoch**2 + self.Mz2stoch **
                       2)**(0.5) * self.D_med / (2. * I)
        stoch_shear2 = abs(self.Mx_stoch * self.D_med / (2. * J))
        # all normal force held by downwind bearing
        stoch_normal2 = self.Fx_stoch / Area * cos(self.shaft_angle)
        stoch_stress2 = ((stoch_bend2 + stoch_normal2) **
                         2 + 3. * stoch_shear2**2)**(0.5)
        # print stoch_stress2

        # create mean loads
        # Fz_mean*self.L_rb*self.D_med/(2.*I) #not mean, but deterministic
        mean_bend2 = 0.
        mean_shear2 = self.Mx_mean * self.D_med / (2. * J)
        mean_normal2 = self.Fx_mean / Area * \
            cos(self.shaft_angle) + (self.rotorWeight +
                                     self.LssWeight) * sin(self.shaft_angle)
        mean_stress2 = ((mean_bend2 + mean_normal2) **
                        2 + 3. * mean_shear2**2)**(0.5)
        # apply Goodman with compressive (-) mean stress
        S_mod_stoch2 = Goodman(stoch_stress2, -mean_stress2, self.S_ut)

        # Use Palmgren-Miner linear damage rule to add damage from stochastic
        # load ranges
        DEL_y = self.Fx_stoch.copy()  # initialize
        for i in range(self.num_pts):
            DEL_y[i] = self.N[i] / \
                (Ninterp(S_mod_stoch2[i], self.SN_a, self.SN_b))

        # damage from stochastic loading
        self.Damage = scp.integrate.simps(DEL_y, x=self.N, even='avg')

        # create deterministic loads occurring N_rotor times
        self.Fz1determ = (self.gbxWeight * self.L_gb - self.LssWeight * .5 *
                          self.L_mb - self.rotorWeight * (self.L_mb + self.L_rb)) / (self.L_mb)
        # -rotorWeight*(self.L_rb+self.L_mb) + Fz1determ*self.L_mb - self.LssWeight*.5*self.L_mb + self.gbxWeight*self.L_gb
        self.My2determ = self.gbxWeight * self.L_gb
        self.determ_stress2 = abs(self.My2determ * self.D_med / (2. * I))

        S_mod_determ2 = Goodman(self.determ_stress2, -mean_stress2, self.S_ut)

        if S_mod_determ2 > 0:
            self.Damage += self.N_rotor / \
                (Ninterp(S_mod_determ2, self.SN_a, self.SN_b))
        # print 'max stochastic:', np.max(S_mod_stoch2)
        # print ''
        # print 'Downwind Bearing Diameter:', self.D_med
        # print 'self.Damage:', self.Damage

    def get_Damage_Brng1(self):
        self.D_in = self.shaft_ratio * self.D_max
        self.D_max = (self.D_max**4 + self.D_in**4)**0.25
        self.D_min = (self.D_min**4 + self.D_in**4)**0.25
        I = (pi / 64.0) * (self.D_max**4 - self.D_in**4)
        J = I * 2
        Area = pi / 4. * (self.D_max**2 - self.D_in**2)
        self.LssWeight = self.density * 9.81 * \
            (((pi / 12) * (self.D_max**2 + self.D_min**2 + self.D_max *
                           self.D_min) * (self.L_ms)) - (pi / 4 * self.L_ms * self.D_in**2))

        # create stochastic loads across N
        stoch_bend1 = (self.My_stoch**2 + self.Mz_stoch **
                       2)**(0.5) * self.D_max / (2. * I)
        stoch_shear1 = abs(self.Mx_stoch * self.D_max / (2. * J))
        stoch_normal1 = self.Fx_stoch / Area * cos(self.shaft_angle)
        stoch_stress1 = ((stoch_bend1 + stoch_normal1) **
                         2 + 3. * stoch_shear1**2)**(0.5)

        # create mean loads
        # Fz_mean*self.L_rb*self.D_max/(2.*I) #not mean, but deterministic
        mean_bend1 = 0
        mean_shear1 = self.Mx_mean * self.D_max / (2. * J)
        mean_normal1 = self.Fx_mean / Area * \
            cos(self.shaft_angle) + (self.rotorWeight +
                                     self.LssWeight) * sin(self.shaft_angle)
        mean_stress1 = ((mean_bend1 + mean_normal1) **
                        2 + 3. * mean_shear1**2)**(0.5)

        # apply Goodman with compressive (-) mean stress
        S_mod_stoch1 = Goodman(stoch_stress1, -mean_stress1, self.S_ut)

        # Use Palmgren-Miner linear damage rule to add damage from stochastic
        # load ranges
        DEL_y = self.Fx_stoch.copy()  # initialize
        for i in range(self.num_pts):
            DEL_y[i] = self.N[i] / \
                (Ninterp(S_mod_stoch1[i], self.SN_a, self.SN_b))

        # damage from stochastic loading
        self.Damage = scp.integrate.simps(DEL_y, x=self.N, even='avg')

        # create deterministic loads occurring N_rotor times
        # only deterministic stress at mb1 is bending due to weights
        determ_stress1 = abs(
            self.rotorWeight * cos(self.shaft_angle) * self.L_rb * self.D_max / (2. * I))

        S_mod_determ = Goodman(determ_stress1, -mean_stress1, self.S_ut)
        # print 'before deterministic self.Damage:', self.Damage

        self.Damage += self.N_rotor / \
            (Ninterp(S_mod_determ, self.SN_a, self.SN_b))

    def setup_Fatigue_Loads(self):
        R = self.rotor_diameter / 2.0
        rotor_torque = (self.machine_rating * 1000 /
                        self.DrivetrainEfficiency) / (self.rotor_freq * (pi / 30))
        Tip_speed_ratio = self.rotor_freq / 30. * pi * R / self.Vrated
        rho_air = 1.225  # kg/m^3 density of air TODO add as input
        p_o = 4. / 3 * rho_air * ((4 * pi * self.rotor_freq / 60 * R / 3)**2 + self.Vrated**2) * (
            pi * R / (self.blade_number * Tip_speed_ratio * (Tip_speed_ratio**2 + 1)**(.5)))
        # print 'po:',p_o
        # characteristic frequency on rotor from turbine of given blade number
        # [Hz]
        n_c = self.blade_number * self.rotor_freq / 60
        # number of rotor rotations based off of weibull curve. .827 comes from
        # lower rpm than rated at lower wind speeds
        self.N_f = self.availability * n_c * (self.T_life * 365 * 24 * 60 * 60) * exp(-(
            self.cut_in / self.weibull_A)**self.weibull_k) - exp(-(self.cut_out / self.weibull_A)**self.weibull_k)

        k_b = 2.5  # calculating rotor pressure from all three blades. Use kb=1 for individual blades

        if self.IEC_Class == 'A':  # From IEC 61400-1 TODO consider calculating based off of 10-minute windspeed and weibull parameters, include neighboring wake effects?
            I_t = 0.18
        elif self.IEC_Class == 'B':
            I_t = 0.14
        else:
            I_t = 0.12

        Beta = 0.11 * k_b * (I_t + 0.1) * (self.weibull_A + 4.4)

        # for analysis with N on log scale, makes larger loads contain finer
        # step sizes
        self.num_pts = 100
        # with zeros: N=np.logspace(log10(1.0),log10(N_f),endpoint=True,num=self.num_pts)
        self.N = np.logspace((log10(self.N_f) - (2 * k_b - 0.18) / Beta),
                             log10(self.N_f), endpoint=True, num=self.num_pts)
        self.N_rotor = self.N_f / 3.
        F_stoch = self.N.copy()

        # print N

        k_r = 0.8  # assuming natural frequency of rotor is significantly larger than rotor rotational frequency

        for i in range(self.num_pts):
            F_stoch[i] = standardrange(self.N[i], self.N_f, Beta, k_b)
        # print 'Standard1:'
        # print F_stoch

        Fx_factor = (.3649 * log(self.rotor_diameter) - 1.074)
        Mx_factor = (.0799 * log(self.rotor_diameter) - .2577)
        My_factor = (.172 * log(self.rotor_diameter) - .5943)
        Mz_factor = (.1659 * log(self.rotor_diameter) - .5795)

        self.Fx_stoch = (F_stoch.copy() * 0.5 * p_o * (R)) * Fx_factor
        self.Mx_stoch = (F_stoch.copy() * 0.45 * p_o *
                         (R)**2) * Mx_factor  # *0.31
        self.My_stoch = (F_stoch.copy() * 0.33 * p_o *
                         k_r * (R)**2) * My_factor  # *0.25
        self.Mz_stoch = (F_stoch.copy() * 0.33 * p_o *
                         k_r * (R)**2) * Mz_factor  # *0.25

        self.Fx_mean = 0.5 * p_o * R * self.blade_number * Fx_factor
        self.Mx_mean = 0.5 * rotor_torque * Mx_factor
        self.rotorWeight = self.rotor_mass * self.g

#-------------------------------------------------------------------------
# Components
#-------------------------------------------------------------------------


class LowSpeedShaft_drive4pt(LowSpeedShaft_Base):
    ''' LowSpeedShaft class
          The LowSpeedShaft class is used to represent the low speed shaft component of a wind turbine drivetrain. 
          It contains the general properties for a wind turbine component as well as additional design load and dimentional attributes as listed below.
          It contains an update method to determine the mass, mass properties, and dimensions of the component.
    '''

    def __init__(self):
        super(LowSpeedShaft_drive4pt, self).__init__()

    def size_LSS_4pt_Loop_1(self):
        # Distances
        self.L_as = self.L_ms / 2.0  # distance from main bearing to shaft center
        # distance from upwind main bearing to upwind carrier bearing 0.5 meter
        # is an estimation # to add as an input
        self.L_cu = self.L_ms + 0.5
        # distance from upwind main bearing to downwind carrier bearing 0.5
        # meter is an estimation # to add as an input
        self.L_cd = self.L_cu + 0.5

        # Weight properties
        self.rotorWeight = self.rotor_mass * self.g  # rotor weight
        self.lssWeight = pi / 3.0 * (self.D_max**2 + self.D_min**2 +
                                     self.D_max * self.D_min) * self.L_ms * self.density * self.g / 4.0
        self.lss_mass = self.lssWeight / self.g
        self.gbxWeight = self.gearbox_mass * self.g  # gearbox weight
        self.gbxWeight = self.gbxWeight  # needed in fatigue functions
        self.carrierWeight = self.carrier_mass * self.g  # carrier weight
        self.shrinkDiscWeight = self.shrink_disc_mass * self.g

        # define LSS
        x_ms = np.linspace(self.L_rb, self.L_ms + self.L_rb, self.len_pts)
        x_rb = np.linspace(0.0, self.L_rb, self.len_pts)
        y_gp = np.linspace(0, self.L_gp, self.len_pts)

        F_mb_x = -self.rotor_force_x - self.rotorWeight * sin(self.shaft_angle)
        self.F_mb_y = +self.rotor_bending_moment_z / self.L_bg - \
            self.rotor_force_y * (self.L_bg + self.L_rb) / self.L_bg
        self.F_mb_z = (-self.rotor_bending_moment_y + self.rotorWeight * (cos(self.shaft_angle) * (self.L_rb + self.L_bg)
                                                                          + sin(self.shaft_angle) * self.H_gb) + self.lssWeight * (self.L_bg - self.L_as)
                       * cos(self.shaft_angle) + self.shrinkDiscWeight * cos(self.shaft_angle)
                       * (self.L_bg - self.L_ms) - self.gbxWeight * cos(self.shaft_angle) * self.L_gb - self.rotor_force_z * cos(self.shaft_angle) * (self.L_bg + self.L_rb)) / self.L_bg

        F_gb_x = -(self.lssWeight + self.shrinkDiscWeight +
                   self.gbxWeight) * sin(self.shaft_angle)
        F_gb_y = -self.F_mb_y - self.rotor_force_y
        F_gb_z = -self.F_mb_z + (self.shrinkDiscWeight + self.rotorWeight +
                                 self.gbxWeight + self.lssWeight) * cos(self.shaft_angle) - self.rotor_force_z

        My_ms = np.zeros(2 * self.len_pts)
        Mz_ms = np.zeros(2 * self.len_pts)

        for k in range(self.len_pts):
            My_ms[k] = -self.rotor_bending_moment_y + self.rotorWeight * cos(self.shaft_angle) * x_rb[
                k] + 0.5 * self.lssWeight / self.L_ms * x_rb[k]**2 - self.rotor_force_z * x_rb[k]
            Mz_ms[k] = -self.rotor_bending_moment_z - \
                self.rotor_force_y * x_rb[k]

        for j in range(self.len_pts):
            My_ms[j + self.len_pts] = -self.rotor_force_z * x_ms[j] - self.rotor_bending_moment_y + self.rotorWeight * \
                cos(self.shaft_angle) * x_ms[j] - self.F_mb_z * (
                    x_ms[j] - self.L_rb) + 0.5 * self.lssWeight / self.L_ms * x_ms[j]**2
            Mz_ms[j + self.len_pts] = -self.rotor_bending_moment_z - \
                self.F_mb_y * (x_ms[j] - self.L_rb) - \
                self.rotor_force_y * x_ms[j]

        x_shaft = np.concatenate([x_rb, x_ms])

        MM_max = np.amax((My_ms**2 + Mz_ms**2)**0.5)
        Index = np.argmax((My_ms**2 + Mz_ms**2)**0.5)

        MM_min = ((My_ms[-1]**2 + Mz_ms[-1]**2)**0.5)
        # Design shaft OD
        MM = MM_max
        self.D_max = (16.0 * self.n_safety / pi / self.Sy * (4.0 * (MM * self.u_knm_inlb / 1000)**2 +
                                                             3.0 * (self.rotor_bending_moment_x * self.u_knm_inlb / 1000)**2)**0.5)**(1.0 / 3.0) * self.u_in_m

        # OD at end
        MM = MM_min
        self.D_min = (16.0 * self.n_safety / pi / self.Sy * (4.0 * (MM * self.u_knm_inlb / 1000)**2 +
                                                             3.0 * (self.rotor_bending_moment_x * self.u_knm_inlb / 1000)**2)**0.5)**(1.0 / 3.0) * self.u_in_m

        # Estimate ID
        self.D_in = self.shaft_ratio * self.D_max
        self.D_max = (self.D_max**4 + self.D_in**4)**0.25
        self.D_min = (self.D_min**4 + self.D_in**4)**0.25

        self.lssWeight_new = ((pi / 3) * (self.D_max**2 + self.D_min**2 + self.D_max * self.D_min) * (
            self.L_ms) * self.density / 4 + (-pi / 4 * (self.D_in**2) * self.density * (self.L_ms))) * self.g

        def deflection(F_z, W_r, gamma, M_y, f_mb_z, L_rb, W_ms, L_ms, z):
            return -F_z * z**3 / 6.0 + W_r * cos(gamma) * z**3 / 6.0 - M_y * z**2 / 2.0 - f_mb_z * (z - L_rb)**3 / 6.0 + W_ms / (L_ms + L_rb) / 24.0 * z**4

        D1 = deflection(self.rotor_force_z, self.rotorWeight, self.shaft_angle, self.rotor_bending_moment_y,
                        self.F_mb_z, self.L_rb, self.lssWeight_new, self.L_ms, self.L_rb + self.L_ms)
        D2 = deflection(self.rotor_force_z, self.rotorWeight, self.shaft_angle,
                        self.rotor_bending_moment_y, self.F_mb_z, self.L_rb, self.lssWeight_new, self.L_ms, self.L_rb)
        C1 = -(D1 - D2) / self.L_ms
        C2 = D2 - C1 * (self.L_rb)

        I_2 = pi / 64.0 * (self.D_max**4 - self.D_in**4)

        def gx(F_z, W_r, gamma, M_y, f_mb_z, L_rb, W_ms, L_ms, C1, z):
            return -F_z * z**2 / 2.0 + W_r * cos(gamma) * z**2 / 2.0 - M_y * z - f_mb_z * (z - L_rb)**2 / 2.0 + W_ms / (L_ms + L_rb) / 6.0 * z**3 + C1

        self.theta_y = np.zeros(self.len_pts)
        d_y = np.zeros(self.len_pts)

        for kk in range(self.len_pts):
            self.theta_y[kk] = gx(self.rotor_force_z, self.rotorWeight, self.shaft_angle, self.rotor_bending_moment_y,
                                  self.F_mb_z, self.L_rb, self.lssWeight_new, self.L_ms, C1, x_ms[kk]) / self.E / I_2
            d_y[kk] = (deflection(self.rotor_force_z, self.rotorWeight, self.shaft_angle, self.rotor_bending_moment_y,
                                  self.F_mb_z, self.L_rb, self.lssWeight_new, self.L_ms, x_ms[kk]) + C1 * x_ms[kk] + C2) / self.E / I_2

    def size_LSS_4pt_Loop_2(self):

        # Distances
        L_as = (self.L_ms_gb + self.L_mb) / 2.0
        L_cu = (self.L_ms_gb + self.L_mb) + 0.5
        L_cd = L_cu + 0.5

        # Weight
        self.lssWeight_new = ((pi / 3) * (self.D_max**2 + self.D_min**2 + self.D_max * self.D_min) * (self.L_ms_gb + self.L_mb)
                              * self.density / 4 + (-pi / 4 * (self.D_in**2) * self.density * (self.L_ms_gb + self.L_mb))) * self.g

        # define LSS
        x_ms = np.linspace(self.L_rb + self.L_mb, self.L_ms_gb +
                           self.L_mb + self.L_rb, self.len_pts)
        x_mb = np.linspace(self.L_rb, self.L_mb + self.L_rb, self.len_pts)
        x_rb = np.linspace(0.0, self.L_rb, self.len_pts)
        y_gp = np.linspace(0, self.L_gp, self.len_pts)

        F_mb2_x = -self.rotor_force_x - \
            self.rotorWeight * sin(self.shaft_angle)
        F_mb2_y = -self.rotor_bending_moment_z / self.L_mb + \
            self.rotor_force_y * (self.L_rb) / self.L_mb
        F_mb2_z = (self.rotor_bending_moment_y - self.rotorWeight * cos(self.shaft_angle) * self.L_rb
                   - self.lssWeight * L_as * cos(self.shaft_angle) - self.shrinkDiscWeight * (
                       self.L_mb + self.L_ms_0) * cos(self.shaft_angle)
                   + self.gbxWeight * cos(self.shaft_angle) * self.L_gb + self.rotor_force_z * cos(self.shaft_angle) * self.L_rb) / self.L_mb

        F_mb1_x = 0.0
        F_mb1_y = -self.rotor_force_y - F_mb2_y
        F_mb1_z = (self.rotorWeight + self.lssWeight + self.shrinkDiscWeight) * \
            cos(self.shaft_angle) - self.rotor_force_z - F_mb2_z

        F_gb_x = -(self.lssWeight + self.shrinkDiscWeight +
                   self.gbxWeight) * sin(self.shaft_angle)
        F_gb_y = -self.F_mb_y - self.rotor_force_y
        F_gb_z = -self.F_mb_z + (self.shrinkDiscWeight + self.rotorWeight +
                                 self.gbxWeight + self.lssWeight) * cos(self.shaft_angle) - self.rotor_force_z

        My_ms = np.zeros(3 * self.len_pts)
        Mz_ms = np.zeros(3 * self.len_pts)

        for k in range(self.len_pts):
            My_ms[k] = -self.rotor_bending_moment_y + self.rotorWeight * cos(self.shaft_angle) * x_rb[
                k] + 0.5 * self.lssWeight / (self.L_mb + self.L_ms_0) * x_rb[k]**2 - self.rotor_force_z * x_rb[k]
            Mz_ms[k] = -self.rotor_bending_moment_z - \
                self.rotor_force_y * x_rb[k]

        for j in range(self.len_pts):
            My_ms[j + self.len_pts] = -self.rotor_force_z * x_mb[j] - self.rotor_bending_moment_y + self.rotorWeight * \
                cos(self.shaft_angle) * x_mb[j] - F_mb1_z * (x_mb[j] - self.L_rb) + \
                0.5 * self.lssWeight / (self.L_mb + self.L_ms_0) * x_mb[j]**2
            Mz_ms[j + self.len_pts] = -self.rotor_bending_moment_z - \
                F_mb1_y * (x_mb[j] - self.L_rb) - self.rotor_force_y * x_mb[j]

        for l in range(self.len_pts):
            My_ms[l + 2 * self.len_pts] = -self.rotor_force_z * x_ms[l] - self.rotor_bending_moment_y + self.rotorWeight * cos(self.shaft_angle) * x_ms[l] - F_mb1_z * (
                x_ms[l] - self.L_rb) - F_mb2_z * (x_ms[l] - self.L_rb - self.L_mb) + 0.5 * self.lssWeight / (self.L_mb + self.L_ms_0) * x_ms[l]**2
            Mz_ms[l + 2 * self.len_pts] = -self.rotor_bending_moment_z - \
                self.F_mb_y * (x_ms[l] - self.L_rb) - \
                self.rotor_force_y * x_ms[l]

        x_shaft = np.concatenate([x_rb, x_mb, x_ms])

        MM_max = np.amax((My_ms**2 + Mz_ms**2)**0.5)
        Index = np.argmax((My_ms**2 + Mz_ms**2)**0.5)

        MM_min = ((My_ms[-1]**2 + Mz_ms[-1]**2)**0.5)

        MM_med = ((My_ms[-1 - self.len_pts]**2 +
                   Mz_ms[-1 - self.len_pts]**2)**0.5)

        # Design Shaft OD using static loading and distortion energy theory
        MM = MM_max
        self.D_max = (16.0 * self.n_safety / pi / self.Sy * (4.0 * (MM * self.u_knm_inlb / 1000)**2 +
                                                             3.0 * (self.rotor_bending_moment_x * self.u_knm_inlb / 1000)**2)**0.5)**(1.0 / 3.0) * self.u_in_m

        # OD at end
        MM = MM_min
        self.D_min = (16.0 * self.n_safety / pi / self.Sy * (4.0 * (MM * self.u_knm_inlb / 1000)**2 +
                                                             3.0 * (self.rotor_bending_moment_x * self.u_knm_inlb / 1000)**2)**0.5)**(1.0 / 3.0) * self.u_in_m

        MM = MM_med
        self.D_med = (16.0 * self.n_safety / pi / self.Sy * (4.0 * (MM * self.u_knm_inlb / 1000)**2 +
                                                             3.0 * (self.rotor_bending_moment_x * self.u_knm_inlb / 1000)**2)**0.5)**(1.0 / 3.0) * self.u_in_m

        # Estimate ID
        self.D_in = self.shaft_ratio * self.D_max
        self.D_max = (self.D_max**4 + self.D_in**4)**0.25
        self.D_min = (self.D_min**4 + self.D_in**4)**0.25
        self.D_med = (self.D_med**4 + self.D_in**4)**0.25

        self.lssWeight_new = (self.density * pi / 12.0 * self.L_mb * (self.D_max**2 + self.D_med **
                                                                      2 + self.D_max * self.D_med) - self.density * pi / 4.0 * self.D_in**2 * self.L_mb) * self.g

        # deflection between mb1 and mb2
        def deflection1(F_r_z, W_r, gamma, M_y, f_mb1_z, L_rb, W_ms, L_ms, L_mb, z):
            return -F_r_z * z**3 / 6.0 + W_r * cos(gamma) * z**3 / 6.0 - M_y * z**2 / 2.0 - f_mb1_z * (z - L_rb)**3 / 6.0 + W_ms / (L_ms + L_mb) / 24.0 * z**4

        D11 = deflection1(self.rotor_force_z, self.rotorWeight, self.shaft_angle, self.rotor_bending_moment_y,
                          F_mb1_z, self.L_rb, self.lssWeight_new, self.L_ms, self.L_mb, self.L_rb + self.L_mb)
        D21 = deflection1(self.rotor_force_z, self.rotorWeight, self.shaft_angle, self.rotor_bending_moment_y,
                          F_mb1_z, self.L_rb, self.lssWeight_new, self.L_ms, self.L_mb, self.L_rb)
        C11 = -(D11 - D21) / self.L_mb
        C21 = -D21 - C11 * (self.L_rb)

        I_2 = pi / 64.0 * (self.D_max**4 - self.D_in**4)

        def gx1(F_r_z, W_r, gamma, M_y, f_mb1_z, L_rb, W_ms, L_ms, L_mb, C11, z):
            return -F_r_z * z**2 / 2.0 + W_r * cos(gamma) * z**2 / 2.0 - M_y * z - f_mb1_z * (z - L_rb)**2 / 2.0 + W_ms / (L_ms + L_mb) / 6.0 * z**3 + C11

        self.theta_y = np.zeros(2 * self.len_pts)
        d_y = np.zeros(2 * self.len_pts)

        for kk in range(self.len_pts):
            self.theta_y[kk] = gx1(self.rotor_force_z, self.rotorWeight, self.shaft_angle, self.rotor_bending_moment_y,
                                   F_mb1_z, self.L_rb, self.lssWeight_new, self.L_ms, self.L_mb, C11, x_mb[kk]) / self.E / I_2
            d_y[kk] = (deflection1(self.rotor_force_z, self.rotorWeight, self.shaft_angle, self.rotor_bending_moment_y,
                                   F_mb1_z, self.L_rb, self.lssWeight_new, self.L_ms, self.L_mb, x_mb[kk]) + C11 * x_mb[kk] + C21) / self.E / I_2

        # Deflection between mb2 and gbx
        def deflection2(F_z, W_r, gamma, M_y, f_mb1_z, f_mb2_z, L_rb, W_ms, L_ms, L_mb, z):
            return -F_z * z**3 / 6.0 + W_r * cos(gamma) * z**3 / 6.0 - M_y * z**2 / 2.0 - f_mb1_z * (z - L_rb)**3 / 6.0 + -f_mb2_z * (z - L_rb - L_mb)**3 / 6.0 + W_ms / (L_ms + L_mb) / 24.0 * z**4

        def gx2(F_z, W_r, gamma, M_y, f_mb1_z, f_mb2_z, L_rb, W_ms, L_ms, L_mb, z):
            return -F_z * z**2 / 2.0 + W_r * cos(gamma) * z**2 / 2.0 - M_y * z - f_mb1_z * (z - L_rb)**2 / 2.0 - f_mb2_z * (z - L_rb - L_mb)**2 / 2.0 + W_ms / (L_ms + L_mb) / 6.0 * z**3

        D12 = deflection2(self.rotor_force_z, self.rotorWeight, self.shaft_angle, self.rotor_bending_moment_y,
                          F_mb1_z, F_mb2_z, self.L_rb, self.lssWeight_new, self.L_ms, self.L_mb, self.L_rb + self.L_mb)
        D22 = gx2(self.rotor_force_z, self.rotorWeight, self.shaft_angle, self.rotor_bending_moment_y,
                  F_mb1_z, F_mb2_z, self.L_rb, self.lssWeight_new, self.L_ms, self.L_mb, self.L_rb + self.L_mb)
        C12 = gx1(self.rotor_force_z, self.rotorWeight, self.shaft_angle, self.rotor_bending_moment_y,
                  F_mb1_z, self.L_rb, self.lssWeight_new, self.L_ms, self.L_mb, C11, x_mb[-1]) - D22
        C22 = -D12 - C12 * (self.L_rb + self.L_mb)

        for kk in range(self.len_pts):
            self.theta_y[kk + self.len_pts] = (gx2(self.rotor_force_z, self.rotorWeight, self.shaft_angle, self.rotor_bending_moment_y,
                                                   F_mb1_z, F_mb2_z, self.L_rb, self.lssWeight_new, self.L_ms, self.L_mb, x_ms[kk]) + C12) / self.E / I_2
            d_y[kk + self.len_pts] = (deflection2(self.rotor_force_z, self.rotorWeight, self.shaft_angle, self.rotor_bending_moment_y,
                                                  F_mb1_z, F_mb2_z, self.L_rb, self.lssWeight_new, self.L_ms, self.L_mb, x_ms[kk]) + C12 * x_ms[kk] + C22) / self.E / I_2

    def solve_nonlinear(self, params, unknowns, resids):

        # input parameters
        self.g = 9.81

        if self.L_rb == 0:  # distance from hub center to main bearing
            L_rb = 0.007835 * self.rotor_diameter + 0.9642
        else:
            L_rb = self.L_rb

        # If user does not know important moments, crude approx
        if self.rotor_mass > 0 and self.rotor_bending_moment_y == 0:
            self.rotor_bending_moment_y = get_My(self.rotor_mass, L_rb)

        if self.rotor_mass > 0 and self.rotor_bending_moment_z == 0:
            self.rotor_bending_moment_z = get_Mz(self.rotor_mass, L_rb)

        if self.rotor_mass == 0:
            [self.rotor_mass] = get_rotor_mass(self.machine_rating, False)

        if self.flange_length == 0:
            self.flange_length = 0.3 * \
                (self.rotor_diameter / 100.0)**2.0 - \
                0.1 * (self.rotor_diameter / 100.0) + 0.4

        # initialization for iterations
        self.L_ms_new = 0.0
        self.L_ms_0 = 0.5  # main shaft length downwind of main bearing
        self.L_ms = self.L_ms_0
        self.len_pts = 101
        self.D_max = 1
        self.D_min = 0.2

        tol = 1e-4
        check_limit = 1.0
        dL = 0.05
        counter = 0
        N_count = 50
        N_count_2 = 2

        # Distances
        # distance from first main bearing to gearbox yokes  # to add as an
        # input
        self.L_bg = 6.11 - L_rb
        self.L_as = self.L_ms / 2.0  # distance from main bearing to shaft center
        self.L_gb = 0.0  # distance to gbx center from trunnions in x-dir # to add as an input
        self.H_gb = 1.0  # distance to gbx center from trunnions in z-dir # to add as an input
        self.L_gp = 0.825  # distance from gbx coupling to gbx trunnions
        # distance from upwind main bearing to upwind carrier bearing 0.5 meter
        # is an estimation # to add as an input
        self.L_cu = self.L_ms + 0.5
        # distance from upwind main bearing to downwind carrier bearing 0.5
        # meter is an estimation # to add as an input
        self.L_cd = self.L_cu + 0.5

        # material properties
        self.E = 2.1e11
        self.density = 7800.0
        self.n_safety = 2.5  # According to AGMA, takes into account the peak load safety factor
        self.Sy = 66000  # *self.S_ut/700e6 #66000 #psi

        # unit conversion
        self.u_knm_inlb = 8850.745454036
        self.u_in_m = 0.0254000508001

        # Main bearing defelection check
        if self.mb1Type == 'TRB1' or 'TRB2':
            Bearing_Limit = 3.0 / 60.0 / 180.0 * pi
        elif self.mb1Type == 'CRB':
            Bearing_Limit = 4.0 / 60.0 / 180.0 * pi
        elif self.mb1Type == 'SRB' or 'RB':
            Bearing_Limit = 0.078
        elif self.mb1Type == 'RB':
            Bearing_Limit = 0.002
        elif self.mb1Type == 'CARB':
            Bearing_Limit = 0.5 / 180 * pi
        else:
            Bearing_Limit = False

        # Second bearing defelection check
        if self.mb2Type == 'TRB1' or 'TRB2':
            Bearing_Limit2 = 3.0 / 60.0 / 180.0 * pi
        elif self.mb2Type == 'CRB':
            Bearing_Limit2 = 4.0 / 60.0 / 180.0 * pi
        elif self.mb2Type == 'SRB' or 'RB':
            Bearing_Limit2 = 0.078
        elif self.mb2Type == 'RB':
            Bearing_Limit2 = 0.002
        elif self.mb2Type == 'CARB':
            Bearing_Limit2 = 0.5 / 180 * pi
        else:
            Bearing_Limit2 = False

        self.n_safety_brg = 1.0

        length_max = self.overhang - L_rb + \
            (self.gearbox_cm[0] - self.gearbox_length /
             2.)  # modified length limit 7/29/14

        while abs(check_limit) > tol and self.L_ms_new < length_max:
            counter = counter + 1
            if self.L_ms_new > 0:
                self.L_ms = self.L_ms_new
            else:
                self.L_ms = self.L_ms_0

            self.size_LSS_4pt_Loop_1()

            check_limit = abs(
                abs(self.theta_y[-1]) - Bearing_Limit / self.n_safety_brg)

            if check_limit < 0:
                self.L_ms_new = self.L_ms + dL

            else:
                self.L_ms_new = self.L_ms + dL

         # Initialization
        self.L_mb = self.L_ms_new
        counter_ms = 0
        check_limit_ms = 1.0
        self.L_mb_new = 0.0
        self.L_mb_0 = self.L_mb  # main shaft length
        self.L_ms = self.L_ms_new
        dL_ms = 0.05
        dL = 0.0025

        while abs(check_limit_ms) > tol and self.L_mb_new < length_max:
            counter_ms = counter_ms + 1
            if self.L_mb_new > 0:
                self.L_mb = self.L_mb_new
            else:
                self.L_mb = self.L_mb_0

            counter = 0.0
            check_limit = 1.0
            self.L_ms_gb_new = 0.0
            self.L_ms_0 = 0.5  # mainshaft length
            self.L_ms = self.L_ms_0

            while abs(check_limit) > tol and counter < N_count_2:
                counter = counter + 1
                if self.L_ms_gb_new > 0.0:
                    self.L_ms_gb = self.L_ms_gb_new
                else:
                    self.L_ms_gb = self.L_ms_0

                self.size_LSS_4pt_Loop_2()

                check_limit = abs(
                    abs(self.theta_y[-1]) - Bearing_Limit / self.n_safety_brg)

                if check_limit < 0:
                    self.L_ms__gb_new = self.L_ms_gb + dL
                else:
                    self.L_ms__gb_new = self.L_ms_gb + dL

                check_limit_ms = abs(
                    abs(self.theta_y[-1]) - Bearing_Limit2 / self.n_safety_brg)

                if check_limit_ms < 0:
                    self.L_mb_new = self.L_mb + dL_ms
                else:
                    self.L_mb_new = self.L_mb + dL_ms

        # fatigue check Taylor Parsons 6/14
        if self.check_fatigue == 1 or self.check_fatigue == 2:
            #start_time = time.time()

            # checks to make sure all inputs are reasonable
            if self.rotor_mass < 100:
                [self.rotor_mass] = get_rotor_mass(self.machine_rating, False)

            # material properties 34CrNiMo6 steel +QT, large diameter
            self.n_safety = 2.5
            if self.S_ut <= 0:
                self.S_ut = 700.0e6  # Pa

            # calculate material props for fatigue
            Sm = 0.9 * self.S_ut  # for bending situations, material strength at 10^3 cycles

            if self.fatigue_exponent != 0:
                if self.fatigue_exponent > 0:
                    self.SN_b = - self.fatigue_exponent
                else:
                    self.SN_b = self.fatigue_exponent
            else:
                C_size = 0.6  # diameter larger than 10"
                # machined surface 272*(self.S_ut/1e6)**-.995 #forged
                C_surf = 4.51 * (self.S_ut / 1e6)**-.265
                C_temp = 1  # normal operating temps
                C_reliab = 0.814  # 99% reliability
                C_envir = 1.  # enclosed environment
                Se = C_size * C_surf * C_temp * C_reliab * C_envir * .5 * \
                    self.S_ut  # modified endurance limit for infinite life (should be Sf)\
                Nfinal = 5e8  # point where fatigue limit occurs under hypothetical S-N curve TODO adjust to fit actual data
                # assuming no endurance limit (high strength steel)
                z = log10(1e3) - log10(Nfinal)
                self.SN_b = 1 / z * log10(Sm / Se)
            self.SN_a = Sm / (1000.**self.SN_b)
            # print 'fatigue_exponent:',self.SN_b
            # print 'm:', -1/self.SN_b
            # print 'a:', self.SN_a
            if self.check_fatigue == 1:

                setup_Fatigue_Loads(self)

                # upwind bearing calculations
                iterationstep = 0.001
                diameter_limit = 5.0
                while True:
                    get_Damage_Brng1(self)
                    if self.Damage < 1 or self.D_max >= diameter_limit:
                        break
                    else:
                        self.D_max += iterationstep

                # downwind bearing calculations
                diameter_limit = 5.0
                iterationstep = 0.001
                while True:
                    get_Damage_Brng2(self)
                    if self.Damage < 1 or self.D_med >= diameter_limit:
                        break
                    else:
                        self.D_med += iterationstep

                # begin bearing calculations
                # counts per rotation (not defined by characteristic frequency
                # 3n_rotor)
                N_bearings = self.N / self.blade_number

                # radial stochastic + deterministic mean
                Fr1_range = ((abs(self.Fz1stoch) + abs(self.Fz1determ))
                             ** 2 + self.Fy1stoch**2)**.5
                Fa1_range = np.zeros(len(self.Fy1stoch))

                #...calculate downwind forces
                lss_weight = self.density * 9.81 * \
                    (((pi / 12) * (self.D_max**2 + self.D_med**2 + self.D_max *
                                   self.D_med) * (self.L_mb)) - (pi / 4 * self.L_mb * self.D_in**2))
                Fy2stoch = -self.Mz_stoch / (self.L_mb)  # = -Fy1 - Fy_stoch
                Fz2stoch = -(lss_weight * 2. / 3. * self.L_mb - self.My_stoch) / (self.L_mb) + (lss_weight + self.shrinkDiscWeight + self.gbxWeight) * \
                    cos(self.shaft_angle) - \
                    self.rotorWeight  # -Fz1 +Weights*cos(gamma)-Fz_stoch+Fz_mean (Fz_mean is in negative direction)
                Fr2_range = (Fy2stoch**2 + (Fz2stoch + abs(-self.rotorWeight * L_rb +
                                                           0.5 * lss_weight + self.gbxWeight * self.L_gb / self.L_mb))**2)**0.5
                Fa2_range = self.Fx_stoch * cos(self.shaft_angle) + (
                    self.rotorWeight + lss_weight) * sin(self.shaft_angle)  # axial stochastic + mean

                life_bearing = self.N_f / self.blade_number

                [self.D_max_a, FW_max, bearing1mass] = fatigue_for_bearings(
                    self.D_max, Fr1_range, Fa1_range, N_bearings, life_bearing, self.mb1Type, False)
                [self.D_med_a, FW_med, bearing2mass] = fatigue_for_bearings(
                    self.D_med, Fr2_range, Fa2_range, N_bearings, life_bearing, self.mb2Type, False)

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
            # return (L_ult/(.5*L_range))**m #TODO double-check that the input
            # will be the load RANGE instead of load amplitudes. May also
            # include means

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

        else:  # if fatigue_check is not true, resize based on diameter
            [self.D_max_a, FW_max, bearing1mass] = resize_for_bearings(
                self.D_max,  self.mb1Type, False)
            [self.D_med_a, FW_med, bearing2mass] = resize_for_bearings(
                self.D_med,  self.mb2Type, False)

        # end fatigue code additions 6/2014

        lss_mass_new = (pi / 3) * (self.D_max_a**2 + self.D_med_a**2 + self.D_max_a * self.D_med_a) * (self.L_mb - (FW_max + FW_med) / 2) * self.density / 4 + \
            (pi / 4) * (self.D_max_a**2 - self.D_in**2) * self.density * FW_max +\
            (pi / 4) * (self.D_med_a**2 - self.D_in**2) * self.density * FW_med -\
            (pi / 4) * (self.D_in**2) * self.density * \
            (self.L_mb + (FW_max + FW_med) / 2)

        # begin bearing routine with updated shaft mass
        # add facewidths and flange
        self.length = self.L_mb_new + \
            (FW_max + FW_med) / 2 + self.flange_length
        # print ("self.L_mb: {0}").format(self.L_mb)
        # print ("LSS length, m: {0}").format(self.length)
        self.D_outer = self.D_max
        # print ("Upwind MB OD, m: {0}").format(self.D_max_a)
        # print ("Dnwind MB OD, m: {0}").format(self.D_med_a)
        # print ("self.D_min: {0}").format(self.D_min)
        self.D_in = self.D_in
        self.mass = lss_mass_new * 1.33  # add flange mass
        self.diameter1 = self.D_max_a
        self.diameter2 = self.D_med_a

        # calculate mass properties
        downwind_location = np.array([self.gearbox_cm[
                                     0] - self.gearbox_length / 2., self.gearbox_cm[1], self.gearbox_cm[2]])

        bearing_location1 = np.array([0., 0., 0.])  # upwind
        bearing_location1[0] = downwind_location[0] - \
            (self.L_mb_new + FW_med / 2) * cos(self.shaft_angle)
        bearing_location1[1] = downwind_location[1]
        bearing_location1[2] = downwind_location[2] + \
            (self.L_mb_new + FW_med / 2) * sin(self.shaft_angle)
        self.bearing_location1 = bearing_location1

        bearing_location2 = np.array([0., 0., 0.])  # downwind
        bearing_location2[0] = downwind_location[
            0] - FW_med * .5 * cos(self.shaft_angle)
        bearing_location2[1] = downwind_location[1]
        bearing_location2[2] = downwind_location[
            2] + FW_med * .5 * sin(self.shaft_angle)
        self.bearing_location2 = bearing_location2

        cm = np.array([0.0, 0.0, 0.0])
        # From solid models, center of mass with flange (not including shrink
        # disk) very nearly .65*total_length
        cm[0] = downwind_location[0] - 0.65 * \
            self.length * cos(self.shaft_angle)
        cm[1] = downwind_location[1]
        cm[2] = downwind_location[2] + 0.65 * \
            self.length * sin(self.shaft_angle)

        # including shrink disk mass
        self.cm[0] = (cm[0] * self.mass + downwind_location[0] *
                      self.shrink_disc_mass) / (self.mass + self.shrink_disc_mass)
        self.cm[1] = cm[1]
        self.cm[2] = (cm[2] * self.mass + downwind_location[2] *
                      self.shrink_disc_mass) / (self.mass + self.shrink_disc_mass)
        self.mass += self.shrink_disc_mass

        I = np.array([0.0, 0.0, 0.0])
        I[0] = self.mass * (self.D_in ** 2.0 + self.D_outer ** 2.0) / 8.0
        I[1] = self.mass * (self.D_in ** 2.0 + self.D_outer **
                            2.0 + (4.0 / 3.0) * (self.length ** 2.0)) / 16.0
        I[2] = I[1]
        self.I = I

        self.FW_mb1 = FW_max
        self.FW_mb2 = FW_med

        self.bearing_mass1 = bearing1mass
        self.bearing_mass2 = bearing2mass

#-------------------------------------------------------------------------


class LowSpeedShaft_drive3pt(LowSpeedShaft_Base):
    ''' LowSpeedShaft class
          The LowSpeedShaft class is used to represent the low speed shaft component of a wind turbine drivetrain. 
          It contains the general properties for a wind turbine component as well as additional design load and dimentional attributes as listed below.
          It contains an update method to determine the mass, mass properties, and dimensions of the component.
    '''
    def __init__(self):
        super(LowSpeedShaft_drive3pt, self).__init__()


    def size_LSS_3pt(self):
        # Distances
        # distance from hub center to gearbox yokes
        L_bg = 6.11 * (self.machine_rating / 5.0e3)
        L_as = self.L_ms / 2.0  # distance from main bearing to shaft center
        H_gb = 1.0  # distance to gbx center from trunnions in z-dir
        L_gp = 0.825  # distance from gbx coupling to gbx trunnions
        L_cu = self.L_ms + 0.5
        L_cd = L_cu + 0.5
        self.L_gb = 0

        # Weight properties
        weightRotor = self.rotor_mass * self.g
        massLSS = pi / 3 * (self.D_max**2.0 + self.D_min**2.0 +
                            self.D_max * self.D_min) * self.L_ms * self.density / 4.0
        self.weightLSS = massLSS * self.g  # LSS weight
        self.weightShrinkDisc = self.shrink_disc_mass * self.g  # shrink disc weight
        self.weightGbx = self.gearbox_mass * self.g  # gearbox weight
        weightCarrier = self.carrier_mass * self.g

        len_pts = 101
        x_ms = np.linspace(self.L_rb, self.L_ms + self.L_rb, len_pts)
        x_rb = np.linspace(0.0, self.L_rb, len_pts)
        y_gp = np.linspace(0, L_gp, len_pts)

        #len_my = np.arange(1,len(self.rotor_bending_moment_y)+1)
        #print ("self.rotor_force_x: {0}").format(self.rotor_force_x)
        #print ("self.rotor_force_y: {0}").format(self.rotor_force_y)
        #print ("self.rotor_force_z: {0}").format(self.rotor_force_z)
        #print ("self.rotor_bending_moment_x: {0}").format(self.rotor_bending_moment_x)
        #print ("self.rotor_bending_moment_y: {0}").format(self.rotor_bending_moment_y)
        #print ("self.rotor_bending_moment_z: {0}").format(self.rotor_bending_moment_z)
        F_mb_x = -self.rotor_force_x - weightRotor * sin(self.shaft_angle)
        self.F_mb_y = self.rotor_bending_moment_z / L_bg - \
            self.rotor_force_y * (L_bg + self.L_rb) / L_bg
        self.F_mb_z = (-self.rotor_bending_moment_y + weightRotor * (cos(self.shaft_angle) * (self.L_rb + L_bg)
                                                                     + sin(self.shaft_angle) * H_gb) + self.weightLSS * (L_bg - L_as)
                       * cos(self.shaft_angle) + self.weightShrinkDisc * cos(self.shaft_angle)
                       * (L_bg - self.L_ms) - self.weightGbx * cos(self.shaft_angle) * self.L_gb - self.rotor_force_z * cos(self.shaft_angle) * (L_bg + self.L_rb)) / L_bg

        F_gb_x = -(self.weightLSS + self.weightShrinkDisc +
                   self.weightGbx) * sin(self.shaft_angle)
        F_gb_y = -self.F_mb_y - self.rotor_force_y
        F_gb_z = -self.F_mb_z + (self.weightLSS + self.weightShrinkDisc +
                                 self.weightGbx + weightRotor) * cos(self.shaft_angle) - self.rotor_force_z

        # print 'radial force ', (F_gb_y**2+F_gb_z**2)**0.5
        # print 'axial force ', F_gb_x

        # carrier bearing loads
        F_cu_z = (self.weightLSS * cos(self.shaft_angle) + self.weightShrinkDisc * cos(self.shaft_angle) + self.weightGbx * cos(self.shaft_angle)) - self.F_mb_z - self.rotor_force_z - \
            (-self.rotor_bending_moment_y - self.rotor_force_z * cos(self.shaft_angle) * self.L_rb + self.weightLSS *
             (L_bg - L_as) * cos(self.shaft_angle) - weightCarrier * cos(self.shaft_angle) * self.L_gb) / (1 - L_cu / L_cd)

        F_cd_z = (self.weightLSS * cos(self.shaft_angle) + self.weightShrinkDisc * cos(self.shaft_angle) +
                  self.weightGbx * cos(self.shaft_angle)) - self.F_mb_z - self.rotor_force_z - F_cu_z

        My_ms = np.zeros(2 * len_pts)
        Mz_ms = np.zeros(2 * len_pts)

        for k in range(len_pts):
            My_ms[k] = -self.rotor_bending_moment_y + weightRotor * cos(self.shaft_angle) * x_rb[
                k] + 0.5 * self.weightLSS / self.L_ms * x_rb[k]**2 - self.rotor_force_z * x_rb[k]
            Mz_ms[k] = -self.rotor_bending_moment_z - \
                self.rotor_force_y * x_rb[k]

        for j in range(len_pts):
            My_ms[j + len_pts] = -self.rotor_force_z * x_ms[j] - self.rotor_bending_moment_y + weightRotor * \
                cos(self.shaft_angle) * x_ms[j] - self.F_mb_z * (
                    x_ms[j] - self.L_rb) + 0.5 * self.weightLSS / self.L_ms * x_ms[j]**2
            Mz_ms[j + len_pts] = -self.rotor_bending_moment_z - self.F_mb_y * \
                (x_ms[j] - self.L_rb) - self.rotor_force_y * x_ms[j]

        x_shaft = np.concatenate([x_rb, x_ms])

        MM_max = np.amax((My_ms**2 + Mz_ms**2)**0.5 / 1000.0)
        Index = np.argmax((My_ms**2 + Mz_ms**2)**0.5 / 1000.0)

        # print 'Max Moment kNm:'
        # print MM_max
        # print 'Max moment location m:'
        # print x_shaft[Index]

        MM_min = ((My_ms[-1]**2 + Mz_ms[-1]**2)**0.5 / 1000.0)

        # print 'Max Moment kNm:'
        # print MM_min
        #print 'Max moment location m:'#
        # print x_shaft[-1]

        # Design shaft OD using distortion energy theory

        MM = MM_max
        self.D_max = (16.0 * self.n_safety / pi / self.Sy * (4.0 * (MM * self.u_knm_inlb)**2 + 3.0 * (
            self.rotor_bending_moment_x / 1000.0 * self.u_knm_inlb)**2)**0.5)**(1.0 / 3.0) * self.u_in_m

        # OD at end
        MM = MM_min
        self.D_min = (16.0 * self.n_safety / pi / self.Sy * (4.0 * (MM * self.u_knm_inlb)**2 + 3.0 * (
            self.rotor_bending_moment_x / 1000.0 * self.u_knm_inlb)**2)**0.5)**(1.0 / 3.0) * self.u_in_m

        # Estimate ID
        self.D_in = self.shaft_ratio * self.D_max
        self.D_max = (self.D_in**4.0 + self.D_max**4.0)**0.25
        self.D_min = (self.D_in**4.0 + self.D_min**4.0)**0.25
        # print'Max shaft OD m:'
        # print self.D_max
        # print 'Min shaft OD m:'
        # print self.D_min
        # print'Shaft ID:', self.D_in

        self.weightLSS_new = (self.density * pi / 12.0 * self.L_ms * (self.D_max**2.0 + self.D_min**2.0 + self.D_max * self.D_min) - self.density * pi / 4.0 * self.D_in**2.0 * self.L_ms +
                              self.density * pi / 4.0 * self.D_max**2 * self.L_rb) * self.g
        massLSS_new = self.weightLSS_new / self.g

        # print 'Old LSS mass kg:'
        # print massLSS
        # print 'New LSS mass kg:'
        # print massLSS_new

        def fx(F_r_z, W_r, gamma, M_y, f_mb_z, L_rb, W_ms, L_ms, z):
            return -F_r_z * z**3 / 6.0 + W_r * cos(gamma) * z**3 / 6.0 - M_y * z**2 / 2.0 - f_mb_z * (z - L_rb)**3 / 6.0 + W_ms / (L_ms + L_rb) / 24.0 * z**4

        D1 = fx(self.rotor_force_z, weightRotor, self.shaft_angle, self.rotor_bending_moment_y,
                self.F_mb_z, self.L_rb, self.weightLSS_new, self.L_ms, self.L_rb + self.L_ms)
        D2 = fx(self.rotor_force_z, weightRotor, self.shaft_angle, self.rotor_bending_moment_y,
                self.F_mb_z, self.L_rb, self.weightLSS_new, self.L_ms, self.L_rb)
        C1 = -(D1 - D2) / self.L_ms
        C2 = -D2 - C1 * (self.L_rb)

        I_2 = pi / 64.0 * (self.D_max**4 - self.D_in**4)

        def gx(F_r_z, W_r, gamma, M_y, f_mb_z, L_rb, W_ms, L_ms, C1, z):
            return -F_r_z * z**2 / 2.0 + W_r * cos(gamma) * z**2 / 2.0 - M_y * z - f_mb_z * (z - L_rb)**2 / 2.0 + W_ms / (L_ms + L_rb) / 6.0 * z**3 + C1

        self.theta_y = np.zeros(len_pts)
        d_y = np.zeros(len_pts)

        for kk in range(len_pts):
            self.theta_y[kk] = gx(self.rotor_force_z, weightRotor, self.shaft_angle, self.rotor_bending_moment_y,
                                  self.F_mb_z, self.L_rb, self.weightLSS_new, self.L_ms, C1, x_ms[kk]) / self.E / I_2
            d_y[kk] = (fx(self.rotor_force_z, weightRotor, self.shaft_angle, self.rotor_bending_moment_y,
                          self.F_mb_z, self.L_rb, self.weightLSS_new, self.L_ms, x_ms[kk]) + C1 * x_ms[kk] + C2) / self.E / I_2

    def solve_nonlinear(self, params, unknowns, resids):

        # input parameters
        if self.flange_length == 0:
            self.flange_length = 0.3 * \
                (self.rotor_diameter / 100.0)**2.0 - \
                0.1 * (self.rotor_diameter / 100.0) + 0.4

        if self.L_rb == 0:  # distance from hub center to main bearing
            L_rb = get_L_rb(self.rotor_diameter, False)[0]
        else:
            L_rb = self.L_rb

        # If user does not know important moments, crude approx
        if self.rotor_mass > 0 and self.rotor_bending_moment_y == 0:
            self.rotor_bending_moment_y = get_My(self.rotor_mass, L_rb)

        if self.rotor_mass > 0 and self.rotor_bending_moment_z == 0:
            self.rotor_bending_moment_z = get_Mz(self.rotor_mass, L_rb)

        self.g = 9.81  # m/s
        self.density = 7850.0

        self.L_ms_new = 0.0
        self.L_ms_0 = 0.5  # main shaft length downwind of main bearing
        self.L_ms = self.L_ms_0
        tol = 1e-4
        check_limit = 1.0
        dL = 0.05
        self.D_max = 1.0
        self.D_min = 0.2

        T = self.rotor_bending_moment_x / 1000.0

        # Main bearing defelection check
        if self.mb1Type == 'TRB1' or 'TRB2':
            Bearing_Limit = 3.0 / 60.0 / 180.0 * pi
        elif self.mb1Type == 'CRB':
            Bearing_Limit = 4.0 / 60.0 / 180.0 * pi
        elif self.mb1Type == 'SRB' or 'RB':
            Bearing_Limit = 0.078
        elif self.mb1Type == 'RB':
            Bearing_Limit = 0.002
        elif self.mb1Type == 'CARB':
            Bearing_Limit = 0.5 / 180 * pi
        else:
            Bearing_Limit = False

        self.n_safety_brg = 1.0
        self.n_safety = 2.5
        self.Sy = 66000  # *self.S_ut/700e6 #psi
        self.E = 2.1e11
        N_count = 50

        self.u_knm_inlb = 8850.745454036
        self.u_in_m = 0.0254000508001
        counter = 0
        length_max = self.overhang - L_rb + \
            (self.gearbox_cm[0] - self.gearbox_length /
             2.)  # modified length limit 7/29

        while abs(check_limit) > tol and self.L_ms_new < length_max:
            counter = counter + 1
            if self.L_ms_new > 0:
                self.L_ms = self.L_ms_new
            else:
                self.L_ms = self.L_ms_0

            #-----------------------
            self.size_LSS_3pt()
            #-----------------------

            check_limit = abs(
                abs(self.theta_y[-1]) - Bearing_Limit / self.n_safety_brg)
            # print 'deflection slope'
            # print Bearing_Limit
            # print 'threshold'
            # print theta_y[-1]
            self.L_ms_new = self.L_ms + dL

        # fatigue check Taylor Parsons 6/2014
        if self.check_fatigue == 1 or 2:
            #start_time = time.time()
            # material properties 34CrNiMo6 steel +QT, large diameter
            self.E = 2.1e11
            self.density = 7800.0
            self.n_safety = 2.5
            if self.S_ut <= 0:
                self.S_ut = 700.0e6  # Pa
            Sm = 0.9 * self.S_ut  # for bending situations, material strength at 10^3 cycles
            C_size = 0.6  # diameter larger than 10"
            # machined surface 272*(self.S_ut/1e6)**-.995 #forged
            C_surf = 4.51 * (self.S_ut / 1e6)**-.265
            C_temp = 1  # normal operating temps
            C_reliab = 0.814  # 99% reliability
            C_envir = 1.  # enclosed environment
            Se = C_size * C_surf * C_temp * C_reliab * C_envir * .5 * \
                self.S_ut  # modified endurance limit for infinite life

            if self.fatigue_exponent != 0:
                if self.fatigue_exponent > 0:
                    self.SN_b = - self.fatigue_exponent
                else:
                    self.SN_b = self.fatigue_exponent
            else:
                Nfinal = 5e8  # point where fatigue limit occurs under hypothetical S-N curve TODO adjust to fit actual data
                # assuming no endurance limit (high strength steel)
                z = log10(1e3) - log10(Nfinal)
                self.SN_b = 1 / z * log10(Sm / Se)
            self.SN_a = Sm / (1000.**self.SN_b)
            # print 'm:', -1/self.SN_b
            # print 'a:', self.SN_a

            if self.check_fatigue == 1:
                # checks to make sure all inputs are reasonable
                if self.rotor_mass < 100:
                    [self.rotor_mass] = get_rotor_mass(
                        self.machine_rating, False)

                # Rotor Loads calculations using DS472
                setup_Fatigue_Loads(self)

                # upwind diameter calculations
                iterationstep = 0.001
                diameter_limit = 1.5
                while True:

                    get_Damage_Brng1(self)

                    if self.Damage < 1 or self.D_max >= diameter_limit:
                        break
                    else:
                        self.D_max += iterationstep

                # begin bearing calculations
                N_bearings = self.N / self.blade_number  # rotation number

                Fz1stoch = (-self.My_stoch) / (self.L_ms)
                Fy1stoch = self.Mz_stoch / self.L_ms
                Fz1determ = (self.weightGbx * self.L_gb - self.LssWeight * .5 *
                             self.L_ms - self.rotorWeight * (self.L_ms + L_rb)) / (self.L_ms)

                # radial stochastic + deterministic mean
                Fr_range = ((abs(Fz1stoch) + abs(Fz1determ))
                            ** 2 + Fy1stoch**2)**.5
                Fa_range = self.Fx_stoch * cos(self.shaft_angle) + (
                    self.rotorWeight + self.LssWeight) * sin(self.shaft_angle)  # axial stochastic + mean

                life_bearing = self.N_f / self.blade_number

                [self.D_max_a, FW_max, bearingmass] = fatigue_for_bearings(
                    self.D_max, Fr_range, Fa_range, N_bearings, life_bearing, self.mb1Type, False)

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
            # return (L_ult/(.5*L_range))**m #TODO double-check that the input
            # will be the load RANGE instead of load amplitudes. Also, may
            # include means?

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

        # resize bearing if no fatigue check
        if self.check_fatigue == 0:
            [self.D_max_a, FW_max, bearingmass] = resize_for_bearings(
                self.D_max,  self.mb1Type, False)

        # mb2 is a representation of the gearbox connection
        [self.D_min_a, FW_min, trash] = resize_for_bearings(
            self.D_min,  self.mb2Type, False)

        lss_mass_new = (pi / 3) * (self.D_max_a**2 + self.D_min_a**2 + self.D_max_a * self.D_min_a) * (self.L_ms - (FW_max + FW_min) / 2) * self.density / 4 + \
            (pi / 4) * (self.D_max_a**2 - self.D_in**2) * self.density * FW_max +\
            (pi / 4) * (self.D_min_a**2 - self.D_in**2) * self.density * FW_min -\
            (pi / 4) * (self.D_in**2) * self.density * \
            (self.L_ms + (FW_max + FW_min) / 2)
        lss_mass_new *= 1.35  # add flange and shrink disk mass
        self.length = self.L_ms_new + \
            (FW_max + FW_min) / 2 + self.flange_length
        #print ("self.L_ms: {0}").format(self.L_ms)
        #print ("LSS length, m: {0}").format(self.length)
        self.D_outer = self.D_max
        #print ("Upwind MB OD, m: {0}").format(self.D_max_a)
        #print ("CB OD, m: {0}").format(self.D_min_a)
        #print ("self.D_min: {0}").format(self.D_min)
        self.D_in = self.D_in
        self.mass = lss_mass_new
        self.diameter1 = self.D_max_a
        self.diameter2 = self.D_min_a
        # self.length=self.L_ms
        # print self.length
        self.D_outer = self.D_max_a
        self.diameter = self.D_max_a

        # calculate mass properties
        downwind_location = np.array([self.gearbox_cm[
                                     0] - self.gearbox_length / 2., self.gearbox_cm[1], self.gearbox_cm[2]])

        bearing_location1 = np.array([0., 0., 0.])  # upwind
        bearing_location1[0] = downwind_location[
            0] - self.L_ms * cos(self.shaft_angle)
        bearing_location1[1] = downwind_location[1]
        bearing_location1[2] = downwind_location[
            2] + self.L_ms * sin(self.shaft_angle)
        self.bearing_location1 = bearing_location1

        self.bearing_location2 = np.array(
            [0., 0., 0.])  # downwind does not exist

        cm = np.array([0.0, 0.0, 0.0])
        # From solid models, center of mass with flange (not including shrink
        # disk) very nearly .65*total_length
        cm[0] = downwind_location[0] - 0.65 * \
            self.length * cos(self.shaft_angle)
        cm[1] = downwind_location[1]
        cm[2] = downwind_location[2] + 0.65 * \
            self.length * sin(self.shaft_angle)

        # including shrink disk mass
        self.cm[0] = (cm[0] * self.mass + downwind_location[0] *
                      self.shrink_disc_mass) / (self.mass + self.shrink_disc_mass)
        self.cm[1] = cm[1]
        self.cm[2] = (cm[2] * self.mass + downwind_location[2] *
                      self.shrink_disc_mass) / (self.mass + self.shrink_disc_mass)
        # print 'shaft before shrink disk:', self.mass
        self.mass += self.shrink_disc_mass

        I = np.array([0.0, 0.0, 0.0])
        I[0] = self.mass * (self.D_in ** 2.0 + self.D_outer ** 2.0) / 8.0
        I[1] = self.mass * (self.D_in ** 2.0 + self.D_outer **
                            2.0 + (4.0 / 3.0) * (self.length ** 2.0)) / 16.0
        I[2] = I[1]
        self.I = I

        # print 'self.L_rb %8.f' %(self.L_rb) #*(self.machine_rating/5.0e3)   #distance from hub center to main bearing scaled off NREL 5MW
        # print 'L_bg %8.f' %(L_bg) #*(self.machine_rating/5.0e3)         #distance from hub center to gearbox yokes
        # print 'L_as %8.f' %(L_as) #distance from main bearing to shaft center

        self.FW_mb = FW_max
        self.bearing_mass1 = bearingmass
        self.bearing_mass2 = 0.

#-------------------------------------------------------------------------


class LowSpeedShaft_drive(Component):
    ''' LowSpeedShaft class
          The LowSpeedShaft class is used to represent the low speed shaft component of a wind turbine drivetrain. This model is outdated and does not contain fatigue analysis
          It contains the general properties for a wind turbine component as well as additional design load and dimentional attributes as listed below.
          It contains an update method to determine the mass, mass properties, and dimensions of the component.
    '''

    def __init__(self):
        super(LowSpeedShaft_drive, self).__init__()

        # variables
        self.add_param('rotor_torque', val=0.0, units='N*m', desc='The torque load due to aerodynamic forces on the rotor')
        self.add_param('rotor_bending_moment', val=0.0, units='N*m', desc='The bending moment from uneven aerodynamic loads')
        self.add_param('rotor_mass', val=0.0, units='kg', desc='rotor mass')
        self.add_param('rotor_diameter', val=0.0, units='m', desc='rotor diameter')
        self.add_param('rotor_speed', val=0.0, units='rpm', desc='rotor speed at rated power')
        self.add_param('machine_rating', val=0.0, units='kW', desc='machine_rating machine rating of the turbine')

        # parameters
        self.add_param('shaft_angle', val=0.0, units='deg', desc='Angle of the LSS inclindation with respect to the horizontal')
        self.add_param('shaft_length', val=0.0, units='m', desc='length of low speed shaft')
        self.add_param('shaftD1', val=0.0, units='m', desc='Fraction of LSS distance from gearbox to downwind main bearing')
        self.add_param('shaftD2', val=0.0, units='m', desc='raction of LSS distance from gearbox to upwind main bearing')
        self.add_param('shaft_ratio', val=0.0 desc='Ratio of inner diameter to outer diameter.  Leave zero for solid LSS')

        # outputs
        self.add_output('design_torque', val=0.0,  units='N*m', desc='lss design torque')
        self.add_output('design_bending_load', val=0.0,  units='N', desc='lss design bending load')
        self.add_output('length', val=0.0, units='m', desc='lss length')
        self.add_output('diameter', val=0.0, units='m', desc='lss outer diameter')
        self.add_output('mass', val=0.0, units='kg', desc='overall component mass')
        self.add_output('cm', val=np.array([0.0, 0.0, 0.0]), desc='center of mass of the component in [x,y,z] for an arbitrary coordinate system')
        self.add_output('I', val=np.array([0.0, 0.0, 0.0]), desc=' moments of Inertia for the component [Ixx, Iyy, Izz] around its center of mass')

    def calc_mass(rotor_torque, rotor_bending_moment, rotor_mass, rotorDiaemeter, rotor_speed, shaft_angle, shaft_length, shaftD1, shaftD2, machine_rating, shaft_ratio):

            # Second moment of area for hollow shaft
        def Imoment(d_o, d_i):
            I = (pi / 64.0) * (d_o**4 - d_i**4)
            return I

        # Second polar moment for hollow shaft
        def Jmoment(d_o, d_i):
            J = (pi / 32.0) * (d_o**4 - d_i**4)
            return J

        # Bending stress
        def bendingStress(M, y, I):
            sigma = M * y / I
            return sigma

        # Shear stress
        def shearStress(T, r, J):
            tau = T * r / J
            return tau

        # Find the necessary outer diameter given a diameter ratio and max
        # stress
        def outerDiameterStrength(shaft_ratio, maxFactoredStress):
            D_outer = (16.0 / (pi * (1.0 - shaft_ratio**4.0) * maxFactoredStress) * (factoredTotalRotorMoment +                                                                   sqrt(factoredTotalRotorMoment**2.0 + factoredrotor_torque**2.0)))**(1.0 / 3.0)
            return D_outer

        #[rotor_torque, rotor_bending_moment, rotor_mass, rotorDiaemeter, rotor_speed, shaft_angle, shaft_length, shaftD1, shaftD2, machine_rating, shaft_ratio] = x

        # torque check
        if rotor_torque == 0:
            # rotational speed in rad/s at rated power
            omega = rotor_speed / 60 * (2 * pi)
            eta = 0.944  # drivetrain efficiency
            rotor_torque = machine_rating / (omega * eta)  # torque

        # self.length=shaft_length

        # compute masses, dimensions and cost
        # static overhanging rotor moment (need to adjust for CM of rotor not
        # just distance to end of LSS)
        L2 = shaft_length * shaftD2  # main bearing to end of mainshaft
        alpha = shaft_angle * pi / 180.0  # shaft angle
        # horizontal distance from main bearing to hub center of mass
        L2 = L2 * cos(alpha)
        staticRotorMoment = rotor_mass * L2 * 9.81  # static bending moment from rotor

        # assuming 38CrMo4 / AISI 4140 from
        # http://www.efunda.com/materials/alloys/alloy_steels/show_alloy.cfm?id=aisi_4140&prop=all&page_title=aisi%204140
        yieldStrength = 417.0 * 10.0**6.0  # Pa
        steelDensity = 8.0 * 10.0**3

        # Safety Factors
        gammaAero = 1.35
        gammaGravity = 1.35  # some talk of changing this to 1.1
        gammaFavorable = 0.9
        gammaMaterial = 1.25  # most conservative

        maxFactoredStress = yieldStrength / gammaMaterial
        factoredrotor_torque = rotor_torque * gammaAero
        factoredTotalRotorMoment = rotor_bending_moment * \
            gammaAero - staticRotorMoment * gammaFavorable

        self.D_outer = outerDiameterStrength(
            self.shaft_ratio, maxFactoredStress)
        self.D_in = shaft_ratio * self.D_outer

        # print "LSS outer diameter is %f m, inner diameter is %f m"
        # %(self.D_outer, self.D_in)

        J = Jmoment(self.D_outer, self.D_in)
        I = Imoment(self.D_outer, self.D_in)

        sigmaX = bendingStress(factoredTotalRotorMoment, self.D_outer / 2.0, I)
        tau = shearStress(rotor_torque, self.D_outer / 2.0, J)

        # print "Max unfactored normal bending stress is %g MPa" % (sigmaX/1.0e6)
        # print "Max unfactored shear stress is %g MPa" % (tau/1.0e6)

        volumeLSS = ((self.D_outer / 2.0)**2.0 -
                     (self.D_in / 2.0)**2.0) * pi * shaft_length
        mass = volumeLSS * steelDensity

        return mass

    def solve_nonlinear(self, params, unknowns, resids):

        self.mass = calc_mass(self.rotor_torque, self.rotor_bending_moment, self.rotor_mass, self.rotor_diameter, self.rotor_speed,
                              self.shaft_angle, self.shaft_length, self.shaftD1, self.shaftD2, self.machine_rating, self.shaft_ratio)

        self.design_torque = self.rotor_torque
        self.design_bending_load = self.rotor_bending_moment
        self.length = self.shaft_length
        self.diameter = self.D_outer

        # calculate mass properties
        cm = np.array([0.0, 0.0, 0.0])
        # cm based on WindPACT work - halfway between locations of two main
        # bearings TODO change!
        cm[0] = - (0.035 - 0.01) * self.rotor_diameter
        cm[1] = 0.0
        cm[2] = 0.025 * self.rotor_diameter
        self.cm = cm

        I = np.array([0.0, 0.0, 0.0])
        I[0] = self.mass * (self.D_in ** 2.0 + self.D_outer ** 2.0) / 8.0
        I[1] = self.mass * (self.D_in ** 2.0 + self.D_outer **
                            2.0 + (4.0 / 3.0) * (self.length ** 2.0)) / 16.0
        I[2] = I[1]
        self.I = I
