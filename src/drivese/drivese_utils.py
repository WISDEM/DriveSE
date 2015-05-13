"""
driveSE_utilis.py
Utilities and functions used in DriveSE

Created by Taylor Parsons 2014.
Copyright (c) NREL. All rights reserved.
"""

from openmdao.main.api import Component
from openmdao.main.datatypes.api import Float, Array
import numpy as np
from math import pi, cos, sqrt, radians, sin, exp, log10, log, floor, ceil
import algopy
import scipy as scp

def size_LSS_4pt_Loop_1(self):
  #Distances
  self.L_as = self.L_ms/2.0     #distance from main bearing to shaft center
  self.L_cu = self.L_ms + 0.5   #distance from upwind main bearing to upwind carrier bearing 0.5 meter is an estimation # to add as an input
  self.L_cd = self.L_cu + 0.5   #distance from upwind main bearing to downwind carrier bearing 0.5 meter is an estimation # to add as an input

  #Weight properties
  self.rotorWeight=self.rotor_mass*self.g                             #rotor weight
  self.lssWeight = pi/3.0*(self.D_max**2 + self.D_min**2 + self.D_max*self.D_min)*self.L_ms*self.density*self.g/4.0 #
  self.lss_mass = self.lssWeight/self.g
  self.gbxWeight = self.gearbox_mass*self.g                           #gearbox weight
  self.gbxWeight = self.gbxWeight                                     #needed in fatigue functions
  self.carrierWeight = self.carrier_mass*self.g                       #carrier weight
  self.shrinkDiscWeight = self.shrink_disc_mass*self.g

  #define LSS
  x_ms = np.linspace(self.L_rb, self.L_ms+self.L_rb, self.len_pts)
  x_rb = np.linspace(0.0, self.L_rb, self.len_pts)
  y_gp = np.linspace(0, self.L_gp, self.len_pts)

  F_mb_x = -self.rotor_force_x - self.rotorWeight*sin(self.shaft_angle)
  self.F_mb_y = +self.rotor_bending_moment_z/self.L_bg - self.rotor_force_y*(self.L_bg + self.L_rb)/self.L_bg
  self.F_mb_z = (-self.rotor_bending_moment_y + self.rotorWeight*(cos(self.shaft_angle)*(self.L_rb + self.L_bg)\
             + sin(self.shaft_angle)*self.H_gb) + self.lssWeight*(self.L_bg - self.L_as)\
             * cos(self.shaft_angle) + self.shrinkDiscWeight*cos(self.shaft_angle)\
             *(self.L_bg - self.L_ms) - self.gbxWeight*cos(self.shaft_angle)*self.L_gb - self.rotor_force_z*cos(self.shaft_angle)*(self.L_bg + self.L_rb))/self.L_bg

  F_gb_x = -(self.lssWeight+self.shrinkDiscWeight+self.gbxWeight)*sin(self.shaft_angle)
  F_gb_y = -self.F_mb_y - self.rotor_force_y
  F_gb_z = -self.F_mb_z + (self.shrinkDiscWeight+self.rotorWeight+self.gbxWeight + self.lssWeight)*cos(self.shaft_angle) - self.rotor_force_z

  My_ms = np.zeros(2*self.len_pts)
  Mz_ms = np.zeros(2*self.len_pts)

  for k in range(self.len_pts):
      My_ms[k] = -self.rotor_bending_moment_y + self.rotorWeight*cos(self.shaft_angle)*x_rb[k] + 0.5*self.lssWeight/self.L_ms*x_rb[k]**2 - self.rotor_force_z*x_rb[k]
      Mz_ms[k] = -self.rotor_bending_moment_z - self.rotor_force_y*x_rb[k]

  for j in range(self.len_pts):
      My_ms[j+self.len_pts] = -self.rotor_force_z*x_ms[j] - self.rotor_bending_moment_y + self.rotorWeight*cos(self.shaft_angle)*x_ms[j] - self.F_mb_z*(x_ms[j]-self.L_rb) + 0.5*self.lssWeight/self.L_ms*x_ms[j]**2
      Mz_ms[j+self.len_pts] = -self.rotor_bending_moment_z - self.F_mb_y*(x_ms[j]-self.L_rb) -self.rotor_force_y*x_ms[j]

  x_shaft = np.concatenate([x_rb, x_ms])

  MM_max=np.amax((My_ms**2+Mz_ms**2)**0.5)
  Index=np.argmax((My_ms**2+Mz_ms**2)**0.5)

  MM_min = ((My_ms[-1]**2+Mz_ms[-1]**2)**0.5)
  #Design shaft OD 
  MM=MM_max
  self.D_max=(16.0*self.n_safety/pi/self.Sy*(4.0*(MM*self.u_knm_inlb/1000)**2+3.0*(self.rotor_bending_moment_x*self.u_knm_inlb/1000)**2)**0.5)**(1.0/3.0)*self.u_in_m

  #OD at end
  MM=MM_min
  self.D_min=(16.0*self.n_safety/pi/self.Sy*(4.0*(MM*self.u_knm_inlb/1000)**2+3.0*(self.rotor_bending_moment_x*self.u_knm_inlb/1000)**2)**0.5)**(1.0/3.0)*self.u_in_m

  #Estimate ID
  self.D_in=self.shaft_ratio*self.D_max
  self.D_max = (self.D_max**4 + self.D_in**4)**0.25
  self.D_min = (self.D_min**4 + self.D_in**4)**0.25
 
  self.lssWeight_new=((pi/3)*(self.D_max**2+self.D_min**2+self.D_max*self.D_min)*(self.L_ms)*self.density/4+(-pi/4*(self.D_in**2)*self.density*(self.L_ms)))*self.g

  def deflection(F_z,W_r,gamma,M_y,f_mb_z,L_rb,W_ms,L_ms,z):
      return -F_z*z**3/6.0 + W_r*cos(gamma)*z**3/6.0 - M_y*z**2/2.0 - f_mb_z*(z-L_rb)**3/6.0 + W_ms/(L_ms + L_rb)/24.0*z**4
  
           
  D1 = deflection(self.rotor_force_z,self.rotorWeight,self.shaft_angle,self.rotor_bending_moment_y,self.F_mb_z,self.L_rb,self.lssWeight_new,self.L_ms,self.L_rb+self.L_ms)
  D2 = deflection(self.rotor_force_z,self.rotorWeight,self.shaft_angle,self.rotor_bending_moment_y,self.F_mb_z,self.L_rb,self.lssWeight_new,self.L_ms,self.L_rb)
  C1 = -(D1-D2)/self.L_ms;
  C2 = D2-C1*(self.L_rb);
  
  I_2=pi/64.0*(self.D_max**4 - self.D_in**4)

  def gx(F_z,W_r,gamma,M_y,f_mb_z,L_rb,W_ms,L_ms,C1,z):
      return -F_z*z**2/2.0 + W_r*cos(gamma)*z**2/2.0 - M_y*z - f_mb_z*(z-L_rb)**2/2.0 + W_ms/(L_ms + L_rb)/6.0*z**3 + C1

  self.theta_y = np.zeros(self.len_pts)
  d_y = np.zeros(self.len_pts)

  for kk in range(self.len_pts):
      self.theta_y[kk]=gx(self.rotor_force_z,self.rotorWeight,self.shaft_angle,self.rotor_bending_moment_y,self.F_mb_z,self.L_rb,self.lssWeight_new,self.L_ms,C1,x_ms[kk])/self.E/I_2
      d_y[kk]=(deflection(self.rotor_force_z,self.rotorWeight,self.shaft_angle,self.rotor_bending_moment_y,self.F_mb_z,self.L_rb,self.lssWeight_new,self.L_ms,x_ms[kk])+C1*x_ms[kk]+C2)/self.E/I_2

def size_LSS_4pt_Loop_2(self):

  #Distances
  L_as = (self.L_ms_gb+self.L_mb)/2.0
  L_cu = (self.L_ms_gb + self.L_mb) + 0.5
  L_cd = L_cu + 0.5

  #Weight
  self.lssWeight_new=((pi/3)*(self.D_max**2+self.D_min**2+self.D_max*self.D_min)*(self.L_ms_gb + self.L_mb)*self.density/4+(-pi/4*(self.D_in**2)*self.density*(self.L_ms_gb + self.L_mb)))*self.g

  #define LSS
  x_ms = np.linspace(self.L_rb + self.L_mb, self.L_ms_gb + self.L_mb +self.L_rb, self.len_pts)
  x_mb = np.linspace(self.L_rb, self.L_mb+self.L_rb, self.len_pts)
  x_rb = np.linspace(0.0, self.L_rb, self.len_pts)
  y_gp = np.linspace(0, self.L_gp, self.len_pts)

  F_mb2_x = -self.rotor_force_x - self.rotorWeight*sin(self.shaft_angle)
  F_mb2_y = -self.rotor_bending_moment_z/self.L_mb + self.rotor_force_y*(self.L_rb)/self.L_mb
  F_mb2_z = (self.rotor_bending_moment_y - self.rotorWeight*cos(self.shaft_angle)*self.L_rb \
            -self.lssWeight*L_as*cos(self.shaft_angle) - self.shrinkDiscWeight*(self.L_mb+self.L_ms_0)*cos(self.shaft_angle) \
             + self.gbxWeight*cos(self.shaft_angle)*self.L_gb + self.rotor_force_z*cos(self.shaft_angle)*self.L_rb)/self.L_mb

  F_mb1_x = 0.0
  F_mb1_y = -self.rotor_force_y - F_mb2_y
  F_mb1_z = (self.rotorWeight + self.lssWeight + self.shrinkDiscWeight)*cos(self.shaft_angle) - self.rotor_force_z - F_mb2_z

  F_gb_x = -(self.lssWeight+self.shrinkDiscWeight+self.gbxWeight)*sin(self.shaft_angle)
  F_gb_y = -self.F_mb_y - self.rotor_force_y
  F_gb_z = -self.F_mb_z + (self.shrinkDiscWeight+self.rotorWeight+self.gbxWeight + self.lssWeight)*cos(self.shaft_angle) - self.rotor_force_z

  My_ms = np.zeros(3*self.len_pts)
  Mz_ms = np.zeros(3*self.len_pts)

  for k in range(self.len_pts):
      My_ms[k] = -self.rotor_bending_moment_y + self.rotorWeight*cos(self.shaft_angle)*x_rb[k] + 0.5*self.lssWeight/(self.L_mb+self.L_ms_0)*x_rb[k]**2 - self.rotor_force_z*x_rb[k]
      Mz_ms[k] = -self.rotor_bending_moment_z - self.rotor_force_y*x_rb[k]

  for j in range(self.len_pts):
      My_ms[j+self.len_pts] = -self.rotor_force_z*x_mb[j] - self.rotor_bending_moment_y + self.rotorWeight*cos(self.shaft_angle)*x_mb[j] - F_mb1_z*(x_mb[j]-self.L_rb) + 0.5*self.lssWeight/(self.L_mb+self.L_ms_0)*x_mb[j]**2
      Mz_ms[j+self.len_pts] = -self.rotor_bending_moment_z - F_mb1_y*(x_mb[j]-self.L_rb) -self.rotor_force_y*x_mb[j]

  for l in range(self.len_pts):
      My_ms[l + 2*self.len_pts] = -self.rotor_force_z*x_ms[l] - self.rotor_bending_moment_y + self.rotorWeight*cos(self.shaft_angle)*x_ms[l] - F_mb1_z*(x_ms[l]-self.L_rb) -F_mb2_z*(x_ms[l] - self.L_rb - self.L_mb) + 0.5*self.lssWeight/(self.L_mb+self.L_ms_0)*x_ms[l]**2
      Mz_ms[l + 2*self.len_pts] = -self.rotor_bending_moment_z - self.F_mb_y*(x_ms[l]-self.L_rb) -self.rotor_force_y*x_ms[l]

  x_shaft = np.concatenate([x_rb, x_mb, x_ms])

  MM_max=np.amax((My_ms**2+Mz_ms**2)**0.5)
  Index=np.argmax((My_ms**2+Mz_ms**2)**0.5)

  MM_min = ((My_ms[-1]**2+Mz_ms[-1]**2)**0.5)

  MM_med = ((My_ms[-1 - self.len_pts]**2 + Mz_ms[-1 - self.len_pts]**2)**0.5)

  #Design Shaft OD using static loading and distortion energy theory
  MM=MM_max
  self.D_max=(16.0*self.n_safety/pi/self.Sy*(4.0*(MM*self.u_knm_inlb/1000)**2+3.0*(self.rotor_bending_moment_x*self.u_knm_inlb/1000)**2)**0.5)**(1.0/3.0)*self.u_in_m

  #OD at end
  MM=MM_min
  self.D_min=(16.0*self.n_safety/pi/self.Sy*(4.0*(MM*self.u_knm_inlb/1000)**2+3.0*(self.rotor_bending_moment_x*self.u_knm_inlb/1000)**2)**0.5)**(1.0/3.0)*self.u_in_m

  MM=MM_med
  self.D_med=(16.0*self.n_safety/pi/self.Sy*(4.0*(MM*self.u_knm_inlb/1000)**2+3.0*(self.rotor_bending_moment_x*self.u_knm_inlb/1000)**2)**0.5)**(1.0/3.0)*self.u_in_m

  #Estimate ID
  self.D_in=self.shaft_ratio*self.D_max
  self.D_max = (self.D_max**4 + self.D_in**4)**0.25
  self.D_min = (self.D_min**4 + self.D_in**4)**0.25
  self.D_med = (self.D_med**4 + self.D_in**4)**0.25

  self.lssWeight_new = (self.density*pi/12.0*self.L_mb*(self.D_max**2+self.D_med**2 + self.D_max*self.D_med) - self.density*pi/4.0*self.D_in**2*self.L_mb)*self.g

  #deflection between mb1 and mb2
  def deflection1(F_r_z,W_r,gamma,M_y,f_mb1_z,L_rb,W_ms,L_ms,L_mb,z):
      return -F_r_z*z**3/6.0 + W_r*cos(gamma)*z**3/6.0 - M_y*z**2/2.0 - f_mb1_z*(z-L_rb)**3/6.0 + W_ms/(L_ms + L_mb)/24.0*z**4
  
  D11 = deflection1(self.rotor_force_z,self.rotorWeight,self.shaft_angle,self.rotor_bending_moment_y,F_mb1_z,self.L_rb,self.lssWeight_new,self.L_ms,self.L_mb,self.L_rb+self.L_mb)
  D21 = deflection1(self.rotor_force_z,self.rotorWeight,self.shaft_angle,self.rotor_bending_moment_y,F_mb1_z,self.L_rb,self.lssWeight_new,self.L_ms,self.L_mb,self.L_rb)
  C11 = -(D11-D21)/self.L_mb
  C21 = -D21-C11*(self.L_rb)

  I_2=pi/64.0*(self.D_max**4 - self.D_in**4)

  def gx1(F_r_z,W_r,gamma,M_y,f_mb1_z,L_rb,W_ms,L_ms,L_mb,C11,z):
      return -F_r_z*z**2/2.0 + W_r*cos(gamma)*z**2/2.0 - M_y*z - f_mb1_z*(z - L_rb)**2/2.0 + W_ms/(L_ms + L_mb)/6.0*z**3 + C11

  self.theta_y = np.zeros(2*self.len_pts)
  d_y = np.zeros(2*self.len_pts)

  for kk in range(self.len_pts):
      self.theta_y[kk]=gx1(self.rotor_force_z,self.rotorWeight,self.shaft_angle,self.rotor_bending_moment_y,F_mb1_z,self.L_rb,self.lssWeight_new,self.L_ms,self.L_mb,C11,x_mb[kk])/self.E/I_2
      d_y[kk]=(deflection1(self.rotor_force_z,self.rotorWeight,self.shaft_angle,self.rotor_bending_moment_y,F_mb1_z,self.L_rb,self.lssWeight_new,self.L_ms,self.L_mb,x_mb[kk])+C11*x_mb[kk]+C21)/self.E/I_2

  #Deflection between mb2 and gbx
  def deflection2(F_z,W_r,gamma,M_y,f_mb1_z,f_mb2_z,L_rb,W_ms,L_ms,L_mb,z):
      return -F_z*z**3/6.0 + W_r*cos(gamma)*z**3/6.0 - M_y*z**2/2.0 - f_mb1_z*(z-L_rb)**3/6.0 + -f_mb2_z*(z - L_rb - L_mb)**3/6.0 + W_ms/(L_ms + L_mb)/24.0*z**4

  def gx2(F_z,W_r,gamma,M_y,f_mb1_z,f_mb2_z,L_rb,W_ms,L_ms,L_mb,z):
      return -F_z*z**2/2.0 + W_r*cos(gamma)*z**2/2.0 - M_y*z - f_mb1_z*(z - L_rb)**2/2.0 - f_mb2_z*(z - L_rb - L_mb)**2/2.0 + W_ms/(L_ms + L_mb)/6.0*z**3

  D12 = deflection2(self.rotor_force_z,self.rotorWeight,self.shaft_angle,self.rotor_bending_moment_y,F_mb1_z,F_mb2_z,self.L_rb,self.lssWeight_new,self.L_ms,self.L_mb,self.L_rb+self.L_mb)
  D22 = gx2(self.rotor_force_z,self.rotorWeight,self.shaft_angle,self.rotor_bending_moment_y,F_mb1_z,F_mb2_z,self.L_rb,self.lssWeight_new,self.L_ms,self.L_mb,self.L_rb+self.L_mb)
  C12 = gx1(self.rotor_force_z,self.rotorWeight,self.shaft_angle,self.rotor_bending_moment_y,F_mb1_z,self.L_rb,self.lssWeight_new,self.L_ms,self.L_mb,C11,x_mb[-1])-D22
  C22 = -D12-C12*(self.L_rb + self.L_mb);

  for kk in range(self.len_pts):
      self.theta_y[kk + self.len_pts]=(gx2(self.rotor_force_z,self.rotorWeight,self.shaft_angle,self.rotor_bending_moment_y,F_mb1_z,F_mb2_z,self.L_rb,self.lssWeight_new,self.L_ms,self.L_mb,x_ms[kk]) + C12)/self.E/I_2
      d_y[kk + self.len_pts]=(deflection2(self.rotor_force_z,self.rotorWeight,self.shaft_angle,self.rotor_bending_moment_y,F_mb1_z,F_mb2_z,self.L_rb,self.lssWeight_new,self.L_ms,self.L_mb,x_ms[kk])+C12*x_ms[kk]+C22)/self.E/I_2


def get_Damage_Brng2(self):
  I=(pi/64.0)*(self.D_med**4-self.D_in**4)
  J=I*2
  Area=pi/4.*(self.D_med**2-self.D_in**2)
  self.LssWeight=self.density*9.81*(((pi/12)*(self.D_max**2+self.D_med**2+self.D_max*self.D_med)*(self.L_mb))-(pi/4*self.L_mb*self.D_in**2))
  
  self.Fz1stoch = (-self.My_stoch)/(self.L_mb)
  self.Fy1stoch = self.Mz_stoch/self.L_mb
  self.My2stoch = 0. #My_stoch - abs(Fz1stoch)*self.L_mb #=0
  self.Mz2stoch = 0. #Mz_stoch - abs(Fy1stoch)*self.L_mb #=0

  #create stochastic loads across N
  stoch_bend2 = (self.My2stoch**2+self.Mz2stoch**2)**(0.5)*self.D_med/(2.*I)
  stoch_shear2 = abs(self.Mx_stoch*self.D_med/(2.*J))
  stoch_normal2 = self.Fx_stoch/Area*cos(self.shaft_angle) #all normal force held by downwind bearing
  stoch_stress2 = ((stoch_bend2+stoch_normal2)**2+3.*stoch_shear2**2)**(0.5)
  #print stoch_stress2
  
  #create mean loads
  mean_bend2 = 0. #Fz_mean*self.L_rb*self.D_med/(2.*I) #not mean, but deterministic
  mean_shear2 = self.Mx_mean*self.D_med/(2.*J)
  mean_normal2 = self.Fx_mean/Area*cos(self.shaft_angle)+(self.rotorWeight+self.LssWeight)*sin(self.shaft_angle)
  mean_stress2 = ((mean_bend2+mean_normal2)**2+3.*mean_shear2**2)**(0.5)
  #apply Goodman with compressive (-) mean stress
  S_mod_stoch2=Goodman(stoch_stress2,-mean_stress2,self.S_ut)

  #Use Palmgren-Miner linear damage rule to add damage from stochastic load ranges
  DEL_y=self.Fx_stoch.copy() #initialize
  for i in range(self.num_pts):
      DEL_y[i] = self.N[i]/(Ninterp(S_mod_stoch2[i],self.SN_a,self.SN_b))

  self.Damage = scp.integrate.simps(DEL_y, x=self.N , even='avg') #damage from stochastic loading

  #create deterministic loads occurring N_rotor times
  self.Fz1determ = (self.gbxWeight*self.L_gb - self.LssWeight*.5*self.L_mb - self.rotorWeight*(self.L_mb+self.L_rb)) / (self.L_mb)
  self.My2determ = self.gbxWeight*self.L_gb #-rotorWeight*(self.L_rb+self.L_mb) + Fz1determ*self.L_mb - self.LssWeight*.5*self.L_mb + self.gbxWeight*self.L_gb
  self.determ_stress2 = abs(self.My2determ*self.D_med/(2.*I))

  S_mod_determ2=Goodman(self.determ_stress2,-mean_stress2,self.S_ut)

  if S_mod_determ2 > 0:
    self.Damage += self.N_rotor/(Ninterp(S_mod_determ2,self.SN_a,self.SN_b))
  # print 'max stochastic:', np.max(S_mod_stoch2)
  # print ''
  # print 'Downwind Bearing Diameter:', self.D_med
  # print 'self.Damage:', self.Damage

def get_Damage_Brng1(self):
  self.D_in=self.shaft_ratio*self.D_max
  self.D_max = (self.D_max**4 + self.D_in**4)**0.25
  self.D_min = (self.D_min**4 + self.D_in**4)**0.25
  I=(pi/64.0)*(self.D_max**4-self.D_in**4)
  J=I*2
  Area=pi/4.*(self.D_max**2-self.D_in**2)
  self.LssWeight=self.density*9.81*(((pi/12)*(self.D_max**2+self.D_min**2+self.D_max*self.D_min)*(self.L_ms))-(pi/4*self.L_ms*self.D_in**2))

  #create stochastic loads across N
  stoch_bend1 = (self.My_stoch**2+self.Mz_stoch**2)**(0.5)*self.D_max/(2.*I)
  stoch_shear1 = abs(self.Mx_stoch*self.D_max/(2.*J))
  stoch_normal1 = self.Fx_stoch/Area*cos(self.shaft_angle)
  stoch_stress1 = ((stoch_bend1+stoch_normal1)**2+3.*stoch_shear1**2)**(0.5)
  
  #create mean loads
  mean_bend1 = 0 #Fz_mean*self.L_rb*self.D_max/(2.*I) #not mean, but deterministic
  mean_shear1 = self.Mx_mean*self.D_max/(2.*J)
  mean_normal1 = self.Fx_mean/Area*cos(self.shaft_angle)+(self.rotorWeight+self.LssWeight)*sin(self.shaft_angle)
  mean_stress1 = ((mean_bend1+mean_normal1)**2+3.*mean_shear1**2)**(0.5)

  #apply Goodman with compressive (-) mean stress
  S_mod_stoch1=Goodman(stoch_stress1,-mean_stress1,self.S_ut)

  #Use Palmgren-Miner linear damage rule to add damage from stochastic load ranges
  DEL_y=self.Fx_stoch.copy() #initialize
  for i in range(self.num_pts):
      DEL_y[i] = self.N[i]/(Ninterp(S_mod_stoch1[i],self.SN_a,self.SN_b))

  self.Damage = scp.integrate.simps(DEL_y,x= self.N, even='avg') #damage from stochastic loading

  #create deterministic loads occurring N_rotor times
  determ_stress1 = abs(self.rotorWeight*cos(self.shaft_angle)*self.L_rb*self.D_max/(2.*I)) #only deterministic stress at mb1 is bending due to weights

  S_mod_determ=Goodman(determ_stress1,-mean_stress1,self.S_ut)
  # print 'before deterministic self.Damage:', self.Damage

  self.Damage += self.N_rotor/(Ninterp(S_mod_determ,self.SN_a,self.SN_b))

def setup_Fatigue_Loads(self):
  R=self.rotor_diameter/2.0
  rotor_torque = (self.machine_rating * 1000 / self.DrivetrainEfficiency) / (self.rotor_freq * (pi/30))
  Tip_speed_ratio= self.rotor_freq/30.*pi*R/self.Vrated
  rho_air= 1.225 #kg/m^3 density of air TODO add as input
  p_o = 4./3*rho_air*((4*pi*self.rotor_freq/60*R/3)**2+self.Vrated**2)*(pi*R/(self.blade_number*Tip_speed_ratio*(Tip_speed_ratio**2+1)**(.5)))
  # print 'po:',p_o
  n_c=self.blade_number*self.rotor_freq/60 #characteristic frequency on rotor from turbine of given blade number [Hz]
  self.N_f=self.availability*n_c*(self.T_life*365*24*60*60)*exp(-(self.cut_in/self.weibull_A)**self.weibull_k)-exp(-(self.cut_out/self.weibull_A)**self.weibull_k) #number of rotor rotations based off of weibull curve. .827 comes from lower rpm than rated at lower wind speeds

  k_b= 2.5 #calculating rotor pressure from all three blades. Use kb=1 for individual blades

  if self.IEC_Class == 'A': # From IEC 61400-1 TODO consider calculating based off of 10-minute windspeed and weibull parameters, include neighboring wake effects?
    I_t=0.18 
  elif self.IEC_Class == 'B':
    I_t=0.14
  else:
    I_t=0.12

  Beta=0.11*k_b*(I_t+0.1)*(self.weibull_A+4.4)

  #for analysis with N on log scale, makes larger loads contain finer step sizes
  self.num_pts=100
  self.N=np.logspace( (log10(self.N_f)-(2*k_b-0.18)/Beta) , log10(self.N_f) , endpoint=True , num=self.num_pts) # with zeros: N=np.logspace(log10(1.0),log10(N_f),endpoint=True,num=self.num_pts)
  self.N_rotor = self.N_f/3.
  F_stoch=self.N.copy()

  #print N

  k_r=0.8 #assuming natural frequency of rotor is significantly larger than rotor rotational frequency

  for i in range(self.num_pts):
    F_stoch[i] = standardrange(self.N[i],self.N_f,Beta,k_b)
  # print 'Standard1:'
  # print F_stoch

  Fx_factor = (.3649*log(self.rotor_diameter)-1.074)
  Mx_factor = (.0799*log(self.rotor_diameter)-.2577)
  My_factor = (.172*log(self.rotor_diameter)-.5943)
  Mz_factor = (.1659*log(self.rotor_diameter)-.5795)

  self.Fx_stoch = (F_stoch.copy()*0.5*p_o*(R))*Fx_factor
  self.Mx_stoch = (F_stoch.copy()*0.45*p_o*(R)**2)*Mx_factor#*0.31
  self.My_stoch = (F_stoch.copy()*0.33*p_o*k_r*(R)**2)*My_factor#*0.25
  self.Mz_stoch = (F_stoch.copy()*0.33*p_o*k_r*(R)**2)*Mz_factor#*0.25        

  self.Fx_mean=0.5*p_o*R*self.blade_number*Fx_factor
  self.Mx_mean=0.5*rotor_torque*Mx_factor
  self.rotorWeight=self.rotor_mass*self.g

def standardrange(N, N_f, Beta, k_b): 
    F_delta=(Beta*(log10(N_f)-log10(N)))+0.18
    if F_delta>=2*k_b:
      F_delta=0.
    return F_delta

def Ninterp(S,a,b):
    return (S/a)**(1/b)

def Goodman(S_alt,S_mean,Sut):
    return S_alt/(1-(S_mean/Sut))

def size_LSS_3pt(self):
  #Distances
  L_bg = 6.11 *(self.machine_rating/5.0e3)         #distance from hub center to gearbox yokes
  L_as = self.L_ms/2.0     #distance from main bearing to shaft center
  H_gb = 1.0          #distance to gbx center from trunnions in z-dir     
  L_gp = 0.825        #distance from gbx coupling to gbx trunnions
  L_cu = self.L_ms + 0.5
  L_cd = L_cu + 0.5
  self.L_gb=0

  #Weight properties
  weightRotor=self.rotor_mass*self.g
  massLSS = pi/3*(self.D_max**2.0 + self.D_min**2.0 + self.D_max*self.D_min)*self.L_ms*self.density/4.0
  self.weightLSS = massLSS*self.g       #LSS weight
  self.weightShrinkDisc = self.shrink_disc_mass*self.g                #shrink disc weight
  self.weightGbx = self.gearbox_mass*self.g                              #gearbox weight
  weightCarrier = self.carrier_mass*self.g

  len_pts=101;
  x_ms = np.linspace(self.L_rb, self.L_ms+self.L_rb, len_pts)
  x_rb = np.linspace(0.0, self.L_rb, len_pts)
  y_gp = np.linspace(0, L_gp, len_pts)

  #len_my = np.arange(1,len(self.rotor_bending_moment_y)+1)
  #print ("self.rotor_force_x: {0}").format(self.rotor_force_x)
  #print ("self.rotor_force_y: {0}").format(self.rotor_force_y)
  #print ("self.rotor_force_z: {0}").format(self.rotor_force_z)
  #print ("self.rotor_bending_moment_x: {0}").format(self.rotor_bending_moment_x)
  #print ("self.rotor_bending_moment_y: {0}").format(self.rotor_bending_moment_y)
  #print ("self.rotor_bending_moment_z: {0}").format(self.rotor_bending_moment_z)
  F_mb_x = -self.rotor_force_x - weightRotor*sin(self.shaft_angle)
  self.F_mb_y = self.rotor_bending_moment_z/L_bg - self.rotor_force_y*(L_bg + self.L_rb)/L_bg
  self.F_mb_z = (-self.rotor_bending_moment_y + weightRotor*(cos(self.shaft_angle)*(self.L_rb + L_bg)\
  + sin(self.shaft_angle)*H_gb) + self.weightLSS*(L_bg - L_as)\
  * cos(self.shaft_angle) + self.weightShrinkDisc*cos(self.shaft_angle)\
  *(L_bg - self.L_ms) - self.weightGbx*cos(self.shaft_angle)*self.L_gb - self.rotor_force_z*cos(self.shaft_angle)*(L_bg + self.L_rb))/L_bg

  F_gb_x = -(self.weightLSS + self.weightShrinkDisc + self.weightGbx)*sin(self.shaft_angle)
  F_gb_y = -self.F_mb_y - self.rotor_force_y
  F_gb_z = -self.F_mb_z + (self.weightLSS + self.weightShrinkDisc + self.weightGbx + weightRotor)*cos(self.shaft_angle) - self.rotor_force_z

  # print 'radial force ', (F_gb_y**2+F_gb_z**2)**0.5
  # print 'axial force ', F_gb_x

  #carrier bearing loads
  F_cu_z = (self.weightLSS*cos(self.shaft_angle) + self.weightShrinkDisc*cos(self.shaft_angle) + self.weightGbx*cos(self.shaft_angle)) - self.F_mb_z - self.rotor_force_z- \
  (-self.rotor_bending_moment_y - self.rotor_force_z*cos(self.shaft_angle)*self.L_rb + self.weightLSS*(L_bg - L_as)*cos(self.shaft_angle) - weightCarrier*cos(self.shaft_angle)*self.L_gb)/(1 - L_cu/L_cd)

  F_cd_z = (self.weightLSS*cos(self.shaft_angle) + self.weightShrinkDisc*cos(self.shaft_angle) + self.weightGbx*cos(self.shaft_angle)) - self.F_mb_z - self.rotor_force_z - F_cu_z 


  My_ms = np.zeros(2*len_pts)
  Mz_ms = np.zeros(2*len_pts)

  for k in range(len_pts):
      My_ms[k] = -self.rotor_bending_moment_y + weightRotor*cos(self.shaft_angle)*x_rb[k] + 0.5*self.weightLSS/self.L_ms*x_rb[k]**2 - self.rotor_force_z*x_rb[k]
      Mz_ms[k] = -self.rotor_bending_moment_z - self.rotor_force_y*x_rb[k]

  for j in range(len_pts):
      My_ms[j+len_pts] = -self.rotor_force_z*x_ms[j] - self.rotor_bending_moment_y + weightRotor*cos(self.shaft_angle)*x_ms[j] - self.F_mb_z*(x_ms[j]-self.L_rb) + 0.5*self.weightLSS/self.L_ms*x_ms[j]**2
      Mz_ms[j+len_pts] = -self.rotor_bending_moment_z - self.F_mb_y*(x_ms[j]-self.L_rb) - self.rotor_force_y*x_ms[j]

  x_shaft = np.concatenate([x_rb, x_ms])

  MM_max=np.amax((My_ms**2 + Mz_ms**2)**0.5/1000.0)
  Index=np.argmax((My_ms**2 + Mz_ms**2)**0.5/1000.0)
      
  #print 'Max Moment kNm:'
  #print MM_max
  #print 'Max moment location m:'
  #print x_shaft[Index]

  MM_min = ((My_ms[-1]**2+Mz_ms[-1]**2)**0.5/1000.0)

  #print 'Max Moment kNm:'
  #print MM_min
  #print 'Max moment location m:'#
  #print x_shaft[-1]

  #Design shaft OD using distortion energy theory
  
 
  MM=MM_max
  self.D_max=(16.0*self.n_safety/pi/self.Sy*(4.0*(MM*self.u_knm_inlb)**2 + 3.0*(self.rotor_bending_moment_x/1000.0*self.u_knm_inlb)**2)**0.5)**(1.0/3.0)*self.u_in_m

  #OD at end
  MM=MM_min
  self.D_min=(16.0*self.n_safety/pi/self.Sy*(4.0*(MM*self.u_knm_inlb)**2 + 3.0*(self.rotor_bending_moment_x/1000.0*self.u_knm_inlb)**2)**0.5)**(1.0/3.0)*self.u_in_m

  #Estimate ID
  self.D_in=self.shaft_ratio*self.D_max
  self.D_max=(self.D_in**4.0 + self.D_max**4.0)**0.25
  self.D_min=(self.D_in**4.0 + self.D_min**4.0)**0.25
  #print'Max shaft OD m:'
  #print self.D_max
  #print 'Min shaft OD m:'
  #print self.D_min
  #print'Shaft ID:', self.D_in
  

  self.weightLSS_new = (self.density*pi/12.0*self.L_ms*(self.D_max**2.0 + self.D_min**2.0 + self.D_max*self.D_min) - self.density*pi/4.0*self.D_in**2.0*self.L_ms + \
                    self.density*pi/4.0*self.D_max**2*self.L_rb)*self.g
  massLSS_new = self.weightLSS_new/self.g

  #print 'Old LSS mass kg:' 
  #print massLSS
  #print 'New LSS mass kg:'
  #print massLSS_new

  def fx(F_r_z,W_r,gamma,M_y,f_mb_z,L_rb,W_ms,L_ms,z):
      return -F_r_z*z**3/6.0 + W_r*cos(gamma)*z**3/6.0 - M_y*z**2/2.0 - f_mb_z*(z-L_rb)**3/6.0 + W_ms/(L_ms + L_rb)/24.0*z**4
  
             
  D1 = fx(self.rotor_force_z,weightRotor,self.shaft_angle,self.rotor_bending_moment_y,self.F_mb_z,self.L_rb,self.weightLSS_new,self.L_ms,self.L_rb+self.L_ms)
  D2 = fx(self.rotor_force_z,weightRotor,self.shaft_angle,self.rotor_bending_moment_y,self.F_mb_z,self.L_rb,self.weightLSS_new,self.L_ms,self.L_rb)
  C1 = -(D1-D2)/self.L_ms;
  C2 = -D2-C1*(self.L_rb);
  
  
  I_2=pi/64.0*(self.D_max**4 - self.D_in**4)

  def gx(F_r_z,W_r,gamma,M_y,f_mb_z,L_rb,W_ms,L_ms,C1,z):
      return -F_r_z*z**2/2.0 + W_r*cos(gamma)*z**2/2.0 - M_y*z - f_mb_z*(z-L_rb)**2/2.0 + W_ms/(L_ms + L_rb)/6.0*z**3 + C1

  self.theta_y = np.zeros(len_pts)
  d_y = np.zeros(len_pts)

  for kk in range(len_pts):
      self.theta_y[kk]=gx(self.rotor_force_z,weightRotor,self.shaft_angle,self.rotor_bending_moment_y,self.F_mb_z,self.L_rb,self.weightLSS_new,self.L_ms,C1,x_ms[kk])/self.E/I_2
      d_y[kk]=(fx(self.rotor_force_z,weightRotor,self.shaft_angle,self.rotor_bending_moment_y,self.F_mb_z,self.L_rb,self.weightLSS_new,self.L_ms,x_ms[kk])+C1*x_ms[kk]+C2)/self.E/I_2



def add_AboveYawMass(self):
  # electronic systems, hydraulics and controls
  self.electrical_mass = 0.0

  self.vs_electronics_mass = 0 #2.4445*self.machine_rating + 1599.0 accounted for in transformer calcs

  self.hvac_mass = 0.08 * self.machine_rating

  self.controls_mass     = 0.0

  # mainframe system including bedplate, platforms, crane and miscellaneous hardware
  self.platforms_mass = 0.125 * self.bedplate_mass

  if (self.crane):
      self.crane_mass =  3000.0
  else:
      self.crane_mass = 0.0

  self.mainframe_mass  = self.bedplate_mass + self.crane_mass + self.platforms_mass

  nacelleCovArea      = 2 * (self.bedplate_length ** 2)              # this calculation is based on Sunderland
  self.cover_mass = (84.1 * nacelleCovArea) / 2          # this calculation is based on Sunderland - divided by 2 in order to approach CSM

  # yaw system weight calculations based on total system mass above yaw system
  self.above_yaw_mass =  self.lss_mass + \
              self.main_bearing_mass + self.second_bearing_mass + \
              self.gearbox_mass + \
              self.hss_mass + \
              self.generator_mass + \
              self.mainframe_mass + \
              self.transformer_mass + \
              self.electrical_mass + \
              self.vs_electronics_mass + \
              self.hvac_mass + \
              self.cover_mass

  self.length      = self.bedplate_length                              # nacelle length [m] based on bedplate length
  self.width       = self.bedplate_width                        # nacelle width [m] based on bedplate width
  self.height      = (2.0 / 3.0) * self.length                         # nacelle height [m] calculated based on cladding area

def add_Nacelle(self):
  # aggregation of nacelle mass
  self.nacelle_mass = (self.above_yaw_mass + self.yawMass)

  # calculation of mass center and moments of inertia
  cm = np.array([0.0,0.0,0.0])
  for i in (range(0,3)):
      # calculate center of mass (use mainframe_mass in place of bedplate_mass - assume lumped around bedplate_cm)
      cm[i] = (self.lss_mass * self.lss_cm[i] + self.transformer_cm[i] * self.transformer_mass + \
              self.main_bearing_mass * self.main_bearing_cm[i] + self.second_bearing_mass * self.second_bearing_cm[i] + \
              self.gearbox_mass * self.gearbox_cm[i] + self.hss_mass * self.hss_cm[i] + \
              self.generator_mass * self.generator_cm[i] + self.mainframe_mass * self.bedplate_cm[i] ) / \
              (self.lss_mass + self.main_bearing_mass + self.second_bearing_mass + \
              self.gearbox_mass + self.hss_mass + self.generator_mass + self.mainframe_mass)
  self.nacelle_cm = cm

  I = np.zeros(6)
  for i in (range(0,3)):                        # calculating MOI, at nacelle center of gravity with origin at tower top center / yaw mass center, ignoring masses of non-drivetrain components / auxiliary systems
      # calculate moments around CM
      # sum moments around each components CM (adjust for mass of mainframe) # TODO: add yaw MMI
      I[i]  =  self.lss_I[i] + self.main_bearing_I[i] + self.second_bearing_I[i] + self.gearbox_I[i] + self.transformer_I[i] +\
                    self.hss_I[i] + self.generator_I[i] + self.bedplate_I[i] * (self.mainframe_mass / self.bedplate_mass)
      # translate to nacelle CM using parallel axis theorem (use mass of mainframe en lieu of bedplate to account for auxiliary equipment)
      for j in (range(0,3)):
          if i != j:
              I[i] +=  self.lss_mass * (self.lss_cm[j] - cm[j]) ** 2 + \
                            self.main_bearing_mass * (self.main_bearing_cm[j] - cm[j]) ** 2 + \
                            self.second_bearing_mass * (self.second_bearing_cm[j] - cm[j]) ** 2 + \
                            self.gearbox_mass * (self.gearbox_cm[j] - cm[j]) ** 2 + \
                            self.transformer_mass * (self.transformer_cm[j] - cm[j]) ** 2 + \
                            self.hss_mass * (self.hss_cm[j] - cm[j]) ** 2 + \
                            self.generator_mass * (self.generator_cm[j] - cm[j]) ** 2 + \
                            self.mainframe_mass * (self.bedplate_cm[j] - cm[j]) ** 2
  self.nacelle_I = I

def add_RNA(self):
  if self.rotor_mass>0:
      rotor_mass = self.rotor_mass
  else:
      [rotor_mass] = get_rotor_mass(self.machine_rating,False)

  masses = np.array([rotor_mass, self.yawMass, self.lss_mass, self.main_bearing_mass,self.second_bearing_mass,self.gearbox_mass,self.hss_mass,self.generator_mass])
  cms = np.array([(-self.overhang), 0.0, self.lss_cm[0], self.main_bearing_cm[0], self.second_bearing_cm[0], self.gearbox_cm[0], self.hss_cm[0], self.generator_cm[0]])

  self.RNA_mass = np.sum(masses)
  self.RNA_cm = np.sum(masses*cms)/np.sum(masses)
  # print self.RNA_mass
  # print self.RNA_cm

def size_Transformer(self):

  def combine_CM(mass1,CM1,mass2,CM2):
      return (mass1*CM1+mass2*CM2)/(mass1+mass2)

  if self.uptower_transformer == True:
      #function places transformer where tower top CM is within tower bottom OD to reduce tower moments
      if self.rotor_mass:
          rotor_mass = self.rotor_mass
      else:
          [rotor_mass] = get_rotor_mass(self.machine_rating,False)

      bottom_OD = self.tower_top_diameter*1.7 #approximate average from industry data
      # print bottom_OD

      self.mass = 2.4445*(self.machine_rating) + 1599.0

      if self.RNA_cm <= -(bottom_OD)/2: #upwind of acceptable. Most likely
          transformer_x = (bottom_OD/2.*(self.RNA_mass+self.mass) - (self.RNA_mass*self.RNA_cm))/(self.mass)
          if transformer_x > self.generator_cm[0]*3:
              print '\n ---------transformer location manipulation not suitable for overall Nacelle CM changes: rear distance excessively large------- \n'
              transformer_x = self.generator_cm[0] + (1.6 * 0.015 * self.rotor_diameter) #assuming generator and transformer approximately same length
      else:
          transformer_x = self.generator_cm[0] + (1.8 * 0.015 * self.rotor_diameter) #assuming generator and transformer approximately same length

      cm = np.array([0.,0.,0.])
      cm[0] = transformer_x
      cm[1] = self.generator_cm[1]
      cm[2] = self.generator_cm[2]/.75*.5 #same height as gearbox CM
      self.cm = cm

      width = self.tower_top_diameter+.5
      height = 0.016*self.rotor_diameter #similar to gearbox
      length = .012*self.rotor_diameter #similar to gearbox

      def get_I(d1,d2,mass):
          return mass*(d1**2 + d2**2)/12.

      I = np.array([0.,0.,0.])
      I[0] = get_I(height,width,self.mass)
      I[1] = get_I(length, height, self.mass)
      I[2] = get_I(length, width, self.mass)
      self.I = I

  else:
      self.cm = np.array([0.,0.,0.])
      self.I = self.cm.copy()
      self.mass = 0.

#functions used in bedplate sizing
def midDeflection(totalLength,loadLength,load,E,I):
  defl = load*loadLength**2.0*(3.0*totalLength - loadLength)/(6.0*E*I)
  return defl

  #tip deflection for distributed load
def distDeflection(totalLength,distWeight,E,I):
  defl = distWeight*totalLength**4.0/(8.0*E*I)
  return defl

def setup_Bedplate(self):
  #Model bedplate as 2 parallel I-beams with a rear steel frame and a front cast frame
  #Deflection constraints applied at each bedplate end
  #Stress constraint checked at root of front and rear bedplate sections

  self.g = 9.81
  self.E = 2.1e11
  self.density = 7800

  if self.L_rb>0:
      self.L_rb = self.L_rb
  else:  
      [self.L_rb] = get_L_rb(self.rotor_diameter,False)

  #component weights and locations
  if self.transformer_mass>0: #only if uptower transformer
      self.transLoc = self.transformer_location.item()
      self.convMass = 0.3*self.transformer_mass
  else:
      self.transLoc = 0
      self.convMass = (2.4445*(self.machine_rating) + 1599.0)*0.3 #(transformer mass * .3)

  self.convLoc = self.generator_location * 2.0
  self.mb1_location = abs(self.mb1_location) #abs(self.gbx_length/2.0) + abs(self.lss_length)
  self.mb2_location = abs(self.mb2_location) #abs(self.gbx_length/2.0)
  self.lss_location= abs(self.lss_location)

  if self.transLoc > 0:
    self.rearTotalLength = self.transLoc*1.1
  else:
    self.rearTotalLength = self.generator_location*4.237/2.886 -self.tower_top_diameter/2.0 #scaled off of GE1.5

  self.frontTotalLength = self.mb1_location + self.FW_mb1/2.

  #rotor weights and loads
  self.rotorLoc = self.mb1_location + self.L_rb
  self.rotorFz=abs(self.rotor_force_z)
  self.rotorMy=abs(self.rotor_bending_moment_y)

  #If user does not know important moment, crude approx
  if self.rotor_mass > 0 and self.rotorMy == 0: 
      self.rotorMy=get_My(self.rotor_mass,self.L_rb)

  #initial I-beam dimensions
  self.tf = 0.01905
  self.tw = 0.0127
  self.h0 = 0.6096
  self.b0 = self.h0/2.0
  
  #Rear Steel Frame:
  if self.gbx_location ==0:
      self.gbx_location = 0
      self.gbx_mass = 0
  else: 
      self.gbx_location = self.gbx_location
      self.gbx_mass = self.gbx_mass

  self.rootStress = 250e6
  self.totalTipDefl = 1.0
  self.stressTol = 5e5
  self.deflTol = 1e-4
  self.defl_denom = 1500. #factor in deflection check
  self.stress_mult = 8. #modified to fit industry data

  self.stressMax = 620e6 #yeild of alloy steel
  self.deflMax = self.rearTotalLength/self.defl_denom

def characterize_Bedplate_Rear(self):
  self.bi = (self.b0-self.tw)/2.0
  self.hi = self.h0-2.0*self.tf
  self.I_b = self.b0*self.h0**3/12.0 - 2*self.bi*self.hi**3/12.0
  self.A = self.b0*self.h0 - 2.0*self.bi*self.hi
  self.w=self.A*self.density
  #Tip Deflection for load not at end
  
  self.hssTipDefl = midDeflection(self.rearTotalLength,self.hss_location,self.hss_mass*self.g/2,self.E,self.I_b)
  self.genTipDefl = midDeflection(self.rearTotalLength,self.generator_location,self.generator_mass*self.g/2,self.E,self.I_b)
  self.convTipDefl = midDeflection(self.rearTotalLength,self.convLoc,self.convMass*self.g/2,self.E,self.I_b)
  self.transTipDefl = midDeflection(self.rearTotalLength,self.transLoc,self.transformer_mass*self.g/2,self.E,self.I_b)
  self.gbxTipDefl = midDeflection(self.rearTotalLength,self.gbx_location,self.gbx_mass*self.g/2,self.E,self.I_b)
  self.selfTipDefl = distDeflection(self.rearTotalLength,self.w*self.g,self.E,self.I_b)

  self.totalTipDefl = self.hssTipDefl + self.genTipDefl + self.convTipDefl + self.transTipDefl +  self.selfTipDefl + self.gbxTipDefl
  
  #root stress
  self.totalBendingMoment=(self.hss_location*self.hss_mass + self.generator_location*self.generator_mass + self.convLoc*self.convMass + self.transLoc*self.transformer_mass + self.w*self.rearTotalLength**2/2.0)*self.g
  self.rootStress = self.totalBendingMoment*self.h0/(2.*self.I_b)

  #mass
  self.steelVolume = self.A*self.rearTotalLength
  self.steelMass = self.steelVolume*self.density

  #2 parallel I beams
  self.totalSteelMass = 2.0*self.steelMass

  self.rearTotalTipDefl=self.totalTipDefl
  self.rearBendingStress=self.rootStress

def setup_Bedplate_Front(self):
  if self.gbx_location < 0:
      self.gbx_location = abs(self.gbx_location)
      self.gbx_mass = self.gbx_mass
  else: 
      self.gbx_location = 0
      self.gbx_mass = 0
  self.E=169e9 #EN-GJS-400-18-LT http://www.claasguss.de/html_e/pdf/THBl2_engl.pdf
  self.castDensity = 7100

  self.tf = 0.01905
  self.tw = 0.0127
  self.h0 = 0.6096
  self.b0 = self.h0/2.0

  self.rootStress = 250e6
  self.totalTipDefl = 1.0

  self.deflMax = self.frontTotalLength/self.defl_denom
  self.stressMax = 200e6

def characterize_Bedplate_Front(self):
  self.bi = (self.b0-self.tw)/2.0
  self.hi = self.h0-2.0*self.tf
  self.I_b = self.b0*self.h0**3/12.0 - 2*self.bi*self.hi**3/12.0
  self.A = self.b0*self.h0 - 2.0*self.bi*self.hi
  self.w=self.A*self.castDensity

  #Tip Deflection for load not at end
  self.gbxTipDefl = midDeflection(self.frontTotalLength,self.gbx_mass,self.gbx_mass*self.g/2.0,self.E,self.I_b)
  self.mb1TipDefl = midDeflection(self.frontTotalLength,self.mb1_location,self.mb1_mass*self.g/2.0,self.E,self.I_b)
  self.mb2TipDefl = midDeflection(self.frontTotalLength,self.mb2_location,self.mb2_mass*self.g/2.0,self.E,self.I_b)
  self.lssTipDefl = midDeflection(self.frontTotalLength,self.lss_location,self.lss_mass*self.g/2.0,self.E,self.I_b)
  self.rotorTipDefl = midDeflection(self.frontTotalLength,self.rotorLoc,self.rotor_mass*self.g/2.0,self.E,self.I_b)
  self.rotorFzTipDefl = midDeflection(self.frontTotalLength,self.rotorLoc,self.rotorFz/2.0,self.E,self.I_b)
  self.selfTipDefl = distDeflection(self.frontTotalLength,self.w*self.g,self.E,self.I_b)
  self.rotorMyTipDefl = self.rotorMy/2.0*self.frontTotalLength**2/(2.0*self.E*self.I_b)

  self.totalTipDefl = self.mb1TipDefl + self.mb2TipDefl + self.lssTipDefl  + self.rotorTipDefl + self.selfTipDefl +\
    self.rotorMyTipDefl + self.rotorFzTipDefl + self.gbxTipDefl

  #root stress
  self.totalBendingMoment=(self.mb1_location*self.mb1_mass/2.0 + self.mb2_location*self.mb2_mass/2.0 + self.lss_location*\
    self.lss_mass/2.0 + self.w*self.frontTotalLength**2/2.0 + self.rotorLoc*self.rotor_mass/2.0)*self.g + self.rotorLoc*\
    self.rotorFz/2.0 +self.rotorMy/2.0
  self.rootStress = self.totalBendingMoment*self.h0/2/self.I_b

  #mass
  self.castVolume = self.A*self.frontTotalLength
  self.castMass = self.castVolume*self.castDensity

  #2 parallel I-beams
  self.totalCastMass = 2.0*self.castMass
  self.frontTotalTipDefl=self.totalTipDefl
  self.frontBendingStress=self.rootStress

def size_Bedplate(self):
  self.frontHeight = self.h0

  #frame multiplier for front support
  self.support_multiplier = 1.1+5e13*self.rotor_diameter**(-8) # based on solidworks estimates for GRC and GE bedplates. extraneous mass percentage decreases for larger machines
  # print self.rotor_diameter
  # print support_multiplier
  self.totalCastMass *= self.support_multiplier
  self.totalSteelMass *= self.support_multiplier
  self.mass = self.totalCastMass+ self.totalSteelMass

  # print 'rotor mass', self.rotor_mass
  # print 'rotor bending moment_y', self.rotor_bending_moment_y
  # print 'rotor fz', self.rotor_force_z 
  # print 'rear bedplate length: ', rearTotalLength
  # print 'front bedplate length: ', frontTotalLength
  # print'rear bedplate tip deflection', rearTotalTipDefl
  # print'front bedplate tip deflection', frontTotalTipDefl
  # print 'bending stress [MPa] at root of rear bedplate:', rearBendingStress/1.0e6
  # print 'bending stress [MPa] at root of front bedplate:', frontBendingStress/1.0e6
  # print 'front bedplate bedplate mass [kg]:', totalCastMass
  # print 'rear bedplate mass [kg]:', totalSteelMass
  # print 'total bedplate mass:', totalSteelMass+ totalCastMass

  self.length = self.frontTotalLength + self.rearTotalLength
  self.width = self.b0 + self.tower_top_diameter
  if self.rearHeight >= self.frontHeight:
      self.height = self.rearHeight
  else:
      self.height = self.frontHeight

  # calculate mass properties
  cm = np.array([0.0,0.0,0.0])
  cm[0] = (self.totalSteelMass*self.rearTotalLength/2 - self.totalCastMass*self.frontTotalLength/2)/(self.mass) #previously 0.
  cm[1] = 0.0
  cm[2] = -self.height/2.
  self.cm = cm

  self.depth = (self.length / 2.0)

  I = np.array([0.0, 0.0, 0.0])
  I[0]  = self.mass * (self.width ** 2 + self.depth ** 2) / 8
  I[1]  = self.mass * (self.depth ** 2 + self.width ** 2 + (4/3) * self.length ** 2) / 16
  I[2]  = I[1]
  self.I = I


def size_LowSpeedShaft(self):
  def calc_mass(rotor_torque, rotor_bending_moment, rotor_mass, rotorDiaemeter, rotor_speed, shaft_angle, shaft_length, shaftD1, shaftD2, machine_rating, shaft_ratio):

        # Second moment of area for hollow shaft
    def Imoment(d_o,d_i):
        I=(pi/64.0)*(d_o**4-d_i**4)
        return I
    
    # Second polar moment for hollow shaft
    def Jmoment(d_o,d_i):
        J=(pi/32.0)*(d_o**4-d_i**4)
        return J
    
    # Bending stress
    def bendingStress(M, y, I):
        sigma=M*y/I
        return sigma
    
    # Shear stress
    def shearStress(T, r, J):
        tau=T*r/J
        return tau
    
    #Find the necessary outer diameter given a diameter ratio and max stress
    def outerDiameterStrength(shaft_ratio,maxFactoredStress):
        D_outer=(16.0/(pi*(1.0-shaft_ratio**4.0)*maxFactoredStress)*(factoredTotalRotorMoment+sqrt(factoredTotalRotorMoment**2.0+factoredrotor_torque**2.0)))**(1.0/3.0)
        return D_outer

    #[rotor_torque, rotor_bending_moment, rotor_mass, rotorDiaemeter, rotor_speed, shaft_angle, shaft_length, shaftD1, shaftD2, machine_rating, shaft_ratio] = x

    #torque check
    if rotor_torque == 0:
        omega=rotor_speed/60*(2*pi)      #rotational speed in rad/s at rated power
        eta=0.944                 #drivetrain efficiency
        rotor_torque=machine_rating/(omega*eta)         #torque

    #self.length=shaft_length
        
    # compute masses, dimensions and cost
    #static overhanging rotor moment (need to adjust for CM of rotor not just distance to end of LSS)
    L2=shaft_length*shaftD2                   #main bearing to end of mainshaft
    alpha=shaft_angle*pi/180.0           #shaft angle
    L2=L2*cos(alpha)                  #horizontal distance from main bearing to hub center of mass
    staticRotorMoment=rotor_mass*L2*9.81      #static bending moment from rotor
  
    #assuming 38CrMo4 / AISI 4140 from http://www.efunda.com/materials/alloys/alloy_steels/show_alloy.cfm?id=aisi_4140&prop=all&page_title=aisi%204140
    yieldStrength=417.0*10.0**6.0 #Pa
    steelDensity=8.0*10.0**3
    
    #Safety Factors
    gammaAero=1.35
    gammaGravity=1.35 #some talk of changing this to 1.1
    gammaFavorable=0.9
    gammaMaterial=1.25 #most conservative
    
    maxFactoredStress=yieldStrength/gammaMaterial
    factoredrotor_torque=rotor_torque*gammaAero
    factoredTotalRotorMoment=rotor_bending_moment*gammaAero-staticRotorMoment*gammaFavorable

    self.D_outer=outerDiameterStrength(self.shaft_ratio,maxFactoredStress)
    self.D_in=shaft_ratio*self.D_outer

    #print "LSS outer diameter is %f m, inner diameter is %f m" %(self.D_outer, self.D_in)
    
    J=Jmoment(self.D_outer,self.D_in)
    I=Imoment(self.D_outer,self.D_in)
    
    sigmaX=bendingStress(factoredTotalRotorMoment, self.D_outer/2.0, I)
    tau=shearStress(rotor_torque, self.D_outer/2.0, J)
    
    #print "Max unfactored normal bending stress is %g MPa" % (sigmaX/1.0e6)
    #print "Max unfactored shear stress is %g MPa" % (tau/1.0e6)
    
    volumeLSS=((self.D_outer/2.0)**2.0-(self.D_in/2.0)**2.0)*pi*shaft_length
    mass=volumeLSS*steelDensity
    
    return mass

  self.mass = calc_mass(self.rotor_torque, self.rotor_bending_moment, self.rotor_mass, self.rotor_diameter, self.rotor_speed, \
                              self.shaft_angle, self.shaft_length, self.shaftD1, self.shaftD2, self.machine_rating, self.shaft_ratio)


  self.design_torque = self.rotor_torque
  self.design_bending_load = self.rotor_bending_moment
  self.length = self.shaft_length
  self.diameter = self.D_outer

  # calculate mass properties
  cm = np.array([0.0,0.0,0.0])
  cm[0] = - (0.035 - 0.01) * self.rotor_diameter            # cm based on WindPACT work - halfway between locations of two main bearings TODO change!
  cm[1] = 0.0
  cm[2] = 0.025 * self.rotor_diameter
  self.cm = cm

  I = np.array([0.0, 0.0, 0.0])
  I[0]  = self.mass * (self.D_in ** 2.0 + self.D_outer ** 2.0) / 8.0
  I[1]  = self.mass * (self.D_in ** 2.0 + self.D_outer ** 2.0 + (4.0 / 3.0) * (self.length ** 2.0)) / 16.0
  I[2]  = I[1]
  self.I = I

def size_YawSystem(self):
  if self.yaw_motors_number == 0 :
    if self.rotor_diameter < 90.0 :
      self.yaw_motors_number = 4
    elif self.rotor_diameter < 120.0 :
      self.yaw_motors_number = 6
    else:
      self.yaw_motors_number = 8

  #Assume friction plate surface width is 1/10 the diameter
  #Assume friction plate thickness scales with rotor diameter
  frictionPlateVol=pi*self.tower_top_diameter*(self.tower_top_diameter*0.10)*(self.rotor_diameter/1000.0)
  steelDensity=8000.0
  frictionPlateMass=frictionPlateVol*steelDensity
  
  #Assume same yaw motors as Vestas V80 for now: Bonfiglioli 709T2M
  yawMotorMass=190.0
  
  totalYawMass=frictionPlateMass + (self.yaw_motors_number*yawMotorMass)
  self.mass= totalYawMass

  # calculate mass properties
  # yaw system assumed to be collocated to tower top center
  cm = np.array([0.0,0.0,0.0])
  cm[2] = -self.bedplate_height
  self.cm = cm

  # assuming 0 MOI for yaw system (ie mass is nonrotating)
  I = np.array([0.0, 0.0, 0.0])
  self.I = I


def size_HighSpeedSide(self):
  # compute masses, dimensions and cost
  design_torque = self.rotor_torque / self.gear_ratio               # design torque [Nm] based on rotor torque and Gearbox ratio
  massFact = 0.025                                 # mass matching factor default value
  highSpeedShaftMass = (massFact * design_torque)

  mechBrakeMass = (0.5 * highSpeedShaftMass)      # relationship derived from HSS multiplier for University of Sunderland model compared to NREL CSM for 750 kW and 1.5 MW turbines

  self.mass = (mechBrakeMass + highSpeedShaftMass)

  diameter = (1.5 * self.lss_diameter)                     # based on WindPACT relationships for full HSS / mechanical brake assembly
  if self.length_in == 0:
      self.length = 0.5+self.rotor_diameter/127.
  else:
      self.length = self.length_in
  length = self.length

  matlDensity = 7850. # material density kg/m^3

  # calculate mass properties
  cm = np.array([0.0,0.0,0.0])
  cm[0]   = self.gearbox_cm[0]+self.gearbox_length/2+length/2
  cm[1]   = self.gearbox_cm[1]
  cm[2]   = self.gearbox_cm[2]+self.gearbox_height*0.2
  self.cm = cm

  I = np.array([0.0, 0.0, 0.0])
  I[0]    = 0.25 * length * 3.14159 * matlDensity * (diameter ** 2) * (self.gear_ratio**2) * (diameter ** 2) / 8.
  I[1]    = self.mass * ((3/4.) * (diameter ** 2) + (length ** 2)) / 12.
  I[2]    = I[1]
  self.I = I

def size_Generator(self):
  massCoeff = [None, 6.4737, 10.51 ,  5.34  , 37.68  ]
  massExp   = [None, 0.9223, 0.9223,  0.9223, 1      ]

  if self.rotor_speed !=0:
    CalcRPM = self.rotor_speed
  else:
    CalcRPM    = 80 / (self.rotor_diameter*0.5*pi/30)
  CalcTorque = (self.machine_rating*1.1) / (CalcRPM * pi/30)

  if self.drivetrain_design == 'geared':
      drivetrain_design = 1
  elif self.drivetrain_design == 'single_stage':
      drivetrain_design = 2
  elif self.drivetrain_design == 'multi_drive':
      drivetrain_design = 3
  elif self.drivetrain_design == 'pm_direct_drive':
      drivetrain_design = 4

  if (drivetrain_design < 4):
      self.mass = (massCoeff[drivetrain_design] * self.machine_rating ** massExp[drivetrain_design])
  else:  # direct drive
      self.mass = (massCoeff[drivetrain_design] * CalcTorque ** massExp[drivetrain_design])

  # calculate mass properties
  length = (1.8 * 0.015 * self.rotor_diameter)
  d_length_d_rotor_diameter = 1.8*.015

  depth = (0.015 * self.rotor_diameter)
  d_depth_d_rotor_diameter = 0.015

  width = (0.5 * depth)
  d_width_d_depth = 0.5

  # print self.highSpeedSide_cm
  cm = np.array([0.0,0.0,0.0])
  cm[0]  = self.highSpeedSide_cm[0] + self.highSpeedSide_length/2. + length/2.
  cm[1]  = self.highSpeedSide_cm[1]
  cm[2]  = self.highSpeedSide_cm[2]
  self.cm = cm

  I = np.array([0.0, 0.0, 0.0])
  I[0]   = ((4.86 * (10. ** (-5))) * (self.rotor_diameter ** 5.333)) + (((2./3.) * self.mass) * (depth ** 2 + width ** 2) / 8.)
  I[1]   = (I[0] / 2.) / (self.gear_ratio ** 2) + ((1./3.) * self.mass * (length ** 2) / 12.) + (((2. / 3.) * self.mass) * \
             (depth ** 2. + width ** 2. + (4./3.) * (length ** 2.)) / 16. )
  I[2]   = I[1]
  self.I = I 



class blade_moment_transform(Component): 
    ''' Blade_Moment_Transform class          
          The Blade_Moment_Transform class is used to transform moments from the WISDEM rotor models to driveSE.
    '''
    # variables
    # ensure angles are in radians. Azimuth is 3-element array with blade azimuths; b1, b2, b3 are 3-element arrays for each blade moment (Mx, My, Mz); pitch and cone are floats
    azimuth_angle = Array(np.array([0,2*pi/3,4*pi/3]),iotype='in',units='rad',desc='azimuth angles for each blade')
    pitch_angle = Float(iotype ='in', units = 'rad', desc = 'pitch angle at each blade, assumed same')
    cone_angle = Float(iotype='in', units='rad', desc='cone angle at each blade, assumed same')
    b1 = Array(iotype='in', units='N*m', desc='moments in x,y,z directions along local blade coordinate system')
    b2 = Array(iotype='in', units='N*m', desc='moments in x,y,z directions along local blade coordinate system')
    b3 = Array(iotype='in', units='N*m', desc='moments in x,y,z directions along local blade coordinate system')

    # returns
    Mx = Float(iotype='out',units='N*m', desc='rotor moment in x-direction')
    My = Float(iotype='out',units='N*m', desc='rotor moment in y-direction')
    Mz = Float(iotype='out',units='N*m', desc='rotor moment in z-direction')
    
    def __init__(self):
        
        super(blade_moment_transform, self).__init__()
    
    def execute(self):
        # print "input blade loads:"
        # i=0
        # while i<3:
        #   print 'b1:', self.b1[i]
        #   print 'b2:', self.b2[i]
        #   print 'b3:', self.b3[i]
        #   i+=1
        # print

        #nested function for transformations
        def trans(alpha,con,phi,bMx,bMy,bMz):
            Mx = bMx*cos(con)*cos(alpha) - bMy*(sin(con)*cos(alpha)*sin(phi)-sin(alpha)*cos(phi)) + bMz*(sin(con)*cos(alpha)*cos(phi)-sin(alpha)*sin(phi))
            My = bMx*cos(con)*sin(alpha) - bMy*(sin(con)*sin(alpha)*sin(phi)+cos(alpha)*cos(phi)) + bMz*(sin(con)*sin(alpha)*cos(phi)+cos(alpha)*sin(phi))
            Mz = bMx*(-sin(alpha)) - bMy*(-cos(alpha)*sin(phi)) + bMz*(cos(alpha)*cos(phi))
            # print 
            # print Mx
            # print My
            # print Mz
            # print
            return [Mx,My,Mz]

        [b1Mx,b1My,b1Mz] = trans(self.pitch_angle,self.cone_angle,self.azimuth_angle[0],self.b1[0],self.b1[1],self.b1[2])
        [b2Mx,b2My,b2Mz] = trans(self.pitch_angle,self.cone_angle,self.azimuth_angle[1],self.b2[0],self.b2[1],self.b2[2])
        [b3Mx,b3My,b3Mz] = trans(self.pitch_angle,self.cone_angle,self.azimuth_angle[2],self.b3[0],self.b3[1],self.b3[2])

        self.Mx = b1Mx+b2Mx+b3Mx
        self.My = b1My+b2My+b3My
        self.Mz = b1Mz+b2Mz+b3Mz

        # print 'azimuth:', self.azimuth_angle/pi*180.
        # print 'pitch:', self.pitch_angle/pi*180.
        # print 'cone:', self.cone_angle/pi*180.

        # print "Total Moments:"
        # print self.Mx
        # print self.My
        # print self.Mz
        # print


# returns FW, mass for bearings without fatigue analysis
def resize_for_bearings(D_shaft,type,deriv):
# assume low load rating for bearing
  if type == 'CARB': #p = Fr, so X=1, Y=0
    out = [D_shaft,.2663*D_shaft+.0435,1561.4*D_shaft**2.6007]
    if deriv== True:
      out.extend([1.,.2663,1561.4*2.6007*D_shaft**1.6007])
  elif type == 'SRB':
    out=[D_shaft,.2762*D_shaft,876.7*D_shaft**1.7195]
    if deriv== True:
      out.extend([1.,.2762,876.7*1.7195*D_shaft**0.7195])    
  elif type == 'TRB1':
    out = [D_shaft,.0740,92.863*D_shaft**.8399]
    if deriv == True:
      out.extend([1.,0.,92.863*0.8399*D_shaft**(0.8399-1.)])
  elif type == 'CRB':
    out = [D_shaft,.1136*D_shaft,304.19*D_shaft**1.8885]
    if deriv == True:
      out.extend([1.,.1136,304.19*1.8885*D_shaft**0.8885])    
  elif type == 'TRB2':
    out = [D_shaft,.1499*D_shaft,543.01*D_shaft**1.9043]
    if deriv == True:
      out.extend([1.,.1499,543.01*1.9043*D_shaft**.9043])
  elif type == 'RB': #factors depend on ratio Fa/C0, C0 depends on bearing... TODO: add this functionality
    out = [D_shaft,.0839,229.47*D_shaft**1.8036]
    if deriv == True:
      out.extend([1.0,0.0,229.47*1.8036*D_shaft**0.8036])

  return out #shaft diameter, FW, mass. if deriv==True, provides derivatives.


# fatigue analysis for bearings
def fatigue_for_bearings(D_shaft,F_r,F_a,N_array,life_bearing,type,deriv):
  #deriv is boolean, defines if derivatives are returned
  if type == 'CARB': #p = Fr, so X=1, Y=0
    if (np.max(F_a)) > 0:
      print '---------------------------------------------------------'
      print "error: axial loads too large for CARB bearing application"
      print '---------------------------------------------------------'
    else:
      e = 1
      Y1 = 0.
      X2 = 1.
      Y2 = 0.
      p = 10./3
    C_min = C_calc(F_a,F_r,N_array,p,e,Y1,Y2,X2,life_bearing)
    if C_min > 13980*D_shaft**1.5602:
        out = [D_shaft,0.4299*D_shaft+0.0382,3682.8*D_shaft**2.7676]
        if deriv:
          out.extend([1.,0.4299,3682.8*2.7676*D_shaft**1.7676])
    else:
        out = [D_shaft,.2663*D_shaft+.0435,1561.4*D_shaft**2.6007]
        if deriv:
          out.extend([1.,.2663,1561.4*2.6007*D_shaft**1.6007])

  elif type == 'SRB':
    e = 0.32
    Y1 = 2.1
    X2 = 0.67
    Y2 = 3.1
    p = 10./3
    C_min = C_calc(F_a,F_r,N_array,p,e,Y1,Y2,X2,life_bearing)
    if C_min >  13878*D_shaft**1.0796:
        out = [D_shaft,.4801*D_shaft,2688.3*D_shaft**1.8877]
        if deriv:
          out.extend([1.,.4801,2688.3*1.8877*D_shaft**0.8877])
    else:
        out = [D_shaft,.2762*D_shaft,876.7*D_shaft**1.7195]
        if deriv:
          out.extend([1.,.2762,876.7*1.7195*D_shaft**0.7195])


  elif type == 'TRB1':
    e = .37
    Y1 = 0
    X2 = .4
    Y2 = 1.6
    p = 10./3
    C_min = C_calc(F_a,F_r,N_array,p,e,Y1,Y2,X2,life_bearing)
    if C_min >  670*D_shaft+1690:
        out = [D_shaft,.1335,269.83*D_shaft**.441]
        if deriv:
          out.extend([1.,0.,269.83*0.441*D_shaft**(0.441-1.)])
    else:
        out = [D_shaft,.0740,92.863*D_shaft**.8399]
        if deriv:
          out.extend([1.,0.,92.863*0.8399*D_shaft**(0.8399-1.)])

  elif type == 'CRB':
    if (np.max(F_a)/np.max(F_r)>=.5) or (np.min(F_a)/(np.min(F_r))>=.5):
      print '--------------------------------------------------------'
      print "error: axial loads too large for CRB bearing application"
      print '--------------------------------------------------------'
    else:
        e = 0.2
        Y1 = 0
        X2 = 0.92
        Y2 = 0.6
        p = 10./3
        C_min = C_calc(F_a,F_r,N_array,p,e,Y1,Y2,X2,life_bearing)
        if C_min > 4526.5*D_shaft**.9556 :
            out = [D_shaft,.2603*D_shaft,1070.8*D_shaft**1.8278]
            if deriv:
              out.extend([1.,.2603,1070.8*1.8278*D_shaft**0.8278])
        else:
            out = [D_shaft,.1136*D_shaft,304.19*D_shaft**1.8885]
            if deriv:
              out.extend([1.,.1136,304.19*1.8885*D_shaft**0.8885]) 

  elif type == 'TRB2':
    e = 0.4
    Y1 = 2.5
    X2 = 0.4
    Y2 = 1.75
    p = 10./3
    C_min = C_calc(F_a,F_r,N_array,p,e,Y1,Y2,X2,life_bearing)
    if C_min > 6579.9*D_shaft**.8592 :
        out = [D_shaft,.3689*D_shaft,1442.6*D_shaft**1.8932]
        if deriv:
          out.extend([1.,.3689,1442.6*1.8932*D_shaft**.8932]) 
    else:
        out = [D_shaft,.1499*D_shaft,543.01*D_shaft**1.9043]
        if deriv:
          out.extend([1.,.1499,543.01*1.9043*D_shaft**.9043])


  elif type == 'RB': #factors depend on ratio Fa/C0, C0 depends on bearing... TODO: add this functionality
    e = 0.4
    Y1 = 1.6
    X2 = 0.75
    Y2 = 2.15
    p = 3.
    C_min = C_calc(F_a,F_r,N_array,p,e,Y1,Y2,X2,life_bearing)
    if C_min > 884.5*D_shaft**.9964 :
        out = [D_shaft,.1571,646.46*D_shaft**2.]
        if deriv:
          out.extend([1.,0.,646.46*2.*D_shaft])
    else:
        out = [D_shaft,.0839,229.47*D_shaft**1.8036]
        if deriv:
          out.extend([1.0,0.0,229.47*1.8036*D_shaft**0.8036])

  return out


#calculate required dynamic load rating, C
def C_calc(F_a,F_r,N_array,p,e,Y1,Y2,X2,life_bearing):
  Fa_ref = np.max(F_a) #used in comparisons Fa/Fr <e
  Fr_ref = np.max(F_r)

  if Fa_ref/Fr_ref <=e:
    P = F_r + Y1*F_a
  else:
    P = X2*F_r + Y2*F_a

  P_eq = ((scp.integrate.simps((P**p),x=N_array,even='avg'))/(N_array[-1]-N_array[0]))**(1/p)
  C_min = P_eq*(life_bearing/1e6)**(1./p)/1000 #kN
  return C_min


# -------------------------------------------------

# def fatigue2_for_bearings(D_shaft,type,Fx,n_Fx,Fy_Fy,n_Fy,Fz_Fz,n_Fz,Fz_My,n_My,Fy_Mz,n_Mz,life_bearing):
# #takes in the effects of individual forces and moments on the radial and axial bearing forces, computes C from sum of bearing life reductions

#   if type == 'CARB': #p = Fr, so X=1, Y=0
#     e = 1
#     Y1 = 0.
#     X2 = 1.
#     Y2 = 0.
#     p = 10./3

#   elif type == 'SRB':
#     e = 0.32
#     Y1 = 2.1
#     X2 = 0.67
#     Y2 = 3.1
#     p = 10./3

#   elif type == 'TRB1':
#     e = .37
#     Y1 = 0
#     X2 = .4
#     Y2 = 1.6
#     p = 10./3

#   elif type == 'CRB':
#     e = 0.2
#     Y1 = 0
#     X2 = 0.92
#     Y2 = 0.6
#     p = 10./3

#   elif type == 'TRB2':
#     e = 0.4
#     Y1 = 2.5
#     X2 = 0.4
#     Y2 = 1.75
#     p = 10./3

#   elif type == 'RB': #factors depend on ratio Fa/C0, C0 depends on bearing... TODO: add this functionality?
#   #idea: select bearing based off of bore, then calculate Fa/C0, see if life is feasible, if not, iterate?
#     e = 0.4
#     Y1 = 1.6
#     X2 = 0.75
#     Y2 = 2.15
#     p = 3.

#   #Dynamic load rating calculation:
#   #reference axial and radial force to find which calculation factor to use-- assume this ratio is relatively consistent across bearing life
#   Fa_ref = np.max(Fx)
#   Fr_ref = ((np.max(Fy_Fy)+np.max(Fy_Mz))**2+(np.max(Fz_Fz)+np.max(Fz_My))**2)**.5

#   if Fa_ref/Fr_ref <=e:
#     #P = F_r + Y1*F_a
#     P_fx =Y1*Fx #equivalent P due to Fx
#     P_fy =Fy_Fy #equivalent P due to Fy... etc
#     P_fz =Fz_Fz
#     P_my =Fz_My
#     P_mz =Fy_Mz
#   else:
#     #P = X2*F_r + Y2*F_a
#     P_fx =Y2*Fx #equivalent P due to Fx
#     P_fy =X2*Fy_Fy #equivalent P due to Fy... etc
#     P_fz =X2*Fz_Fz
#     P_my =X2*Fz_My
#     P_mz =X2*Fy_Mz

#   P_eq = ((scp.integrate.simps((P_fx**p),x=n_Fx,even='avg'))/(np.max(n_Fx)-np.min(n_Fx)))**(1/p)\
#   +((scp.integrate.simps((P_fy**p),x=n_Fy,even='avg'))/(np.max(n_Fy)-np.min(n_Fy)))**(1/p)\
#   +((scp.integrate.simps((P_fz**p),x=n_Fz,even='avg'))/(np.max(n_Fz)-np.min(n_Fz)))**(1/p)\
#   +((scp.integrate.simps((P_my**p),x=n_My,even='avg'))/(np.max(n_My)-np.min(n_My)))**(1/p)\
#   +((scp.integrate.simps((P_mz**p),x=n_Mz,even='avg'))/(np.max(n_Mz)-np.min(n_Mz)))**(1/p)

#   C_min = P_eq*(life_bearing/1e6)**(1./p)/1000 #kN


#   if type == 'CARB': #p = Fr, so X=1, Y=0
#     if C_min > 13980*D_shaft**1.5602:
#         return [D_shaft,0.4299*D_shaft+0.0382,3682.8*D_shaft**2.7676]
#     else:
#         return [D_shaft,.2663*D_shaft+.0435,1561.4*D_shaft**2.6007]

#   elif type == 'SRB':
#     if C_min >  13878*D_shaft**1.0796:
#         return [D_shaft,.4801*D_shaft,2688.3*D_shaft**1.8877]
#     else:
#         return [D_shaft,.2762*D_shaft,876.7*D_shaft**1.7195]

#   elif type == 'TRB1':
#     if C_min >  670*D_shaft+1690:
#         return [D_shaft,.1335,269.83*D_shaft**.441]
#     else:
#         return [D_shaft,.0740,92.863*D_shaft**.8399]

#   elif type == 'CRB':
#     if C_min > 4526.5*D_shaft**.9556 :
#         return [D_shaft,.2603*D_shaft,1070.8*D_shaft**1.8278]
#     else:
#         return [D_shaft,.1136*D_shaft,304.19*D_shaft**1.8885]

#   elif type == 'TRB2':
#     if C_min > 6579.9*D_shaft**.8592 :
#         return [D_shaft,.3689*D_shaft,1442.6*D_shaft**1.8932]
#     else:
#         return [D_shaft,.1499*D_shaft,543.01*D_shaft**1.9043]

#   elif type == 'RB': #factors depend on ratio Fa/C0, C0 depends on bearing... TODO: add this functionality
#     if C_min > 884.5*D_shaft**.9964:
#         return [D_shaft,.1571,646.46*D_shaft**2.]
#     else:
#         return [D_shaft,.0839,229.47*D_shaft**1.8036]

# -------------------------------------------------


def get_rotor_mass(machine_rating,deriv): #if user inputs forces and zero rotor mass
    out = [23.566*machine_rating]
    if deriv:
      out.extend([23.566])
    return out


def get_L_rb(rotor_diameter,deriv):
    out = [0.007835*rotor_diameter+0.9642]
    if deriv:
      out.extend([.007835])
    return out

def get_My(rotor_mass,L_rb): #moments taken to scale approximately with force (rotor mass) and distance (L_rb)
    if L_rb == 0:
      L_rb = get_L_rb((rotor_mass+49089)/1170.6) #approximate rotor diameter from rotor mass
    return 59.7*rotor_mass*L_rb

def get_Mz(rotor_mass,L_rb): #moments taken to scale roughly with force (rotor mass) and distance (L_rb)
    if L_rb == 0:
      L_rb = get_L_rb((rotor_mass-49089)/1170.6) #approximate rotor diameter from rotor mass
    return 53.846*rotor_mass*L_rb


def sys_print(nace):
    print
    print '-------------Nacelle system model results--------------------'

    print 'Low speed shaft %8.1f kg %6.2f m %6.2f Ixx %6.2f Iyy %6.2f Izz %6.2f CGx %6.2f CGy %6.2f CGz '\
          % (nace.lowSpeedShaft.mass-nace.lowSpeedShaft.shrink_disc_mass , nace.lowSpeedShaft.length, nace.lowSpeedShaft.I[0], nace.lowSpeedShaft.I[1], nace.lowSpeedShaft.I[2], nace.lowSpeedShaft.cm[0], nace.lowSpeedShaft.cm[1], nace.lowSpeedShaft.cm[2])
    print 'LSS diameters:', 'upwind', nace.lowSpeedShaft.diameter1   , 'downwind', nace.lowSpeedShaft.diameter2 , 'inner', nace.lowSpeedShaft.diameter1*nace.shaft_ratio
    print 'Main bearing upwind   %8.1f kg. cm %8.1f %8.1f %8.1f' % (nace.mainBearing.mass ,nace.mainBearing.cm[0],nace.mainBearing.cm[1],nace.mainBearing.cm[2])
    print 'Second bearing downwind   %8.1f kg. cm %8.1f %8.1f %8.1f' % (nace.secondBearing.mass ,nace.secondBearing.cm[0],nace.secondBearing.cm[1],nace.secondBearing.cm[2])
    print 'Gearbox         %8.1f kg %6.2f Ixx %6.2f Iyy %6.2f Izz %6.2f CGx %6.2f CGy %6.2f CGz' \
          % (nace.gearbox.mass, nace.gearbox.I[0], nace.gearbox.I[1], nace.gearbox.I[2], nace.gearbox.cm[0], nace.gearbox.cm[1], nace.gearbox.cm[2] )
    print '     gearbox stage masses: %8.1f kg  %8.1f kg %8.1f kg' % (nace.gearbox.stage_masses[0], nace.gearbox.stage_masses[1], nace.gearbox.stage_masses[2])
    print 'High speed shaft & brakes  %8.1f kg %6.2f Ixx %6.2f Iyy %6.2f Izz %6.2f CGx %6.2f CGy %6.2f CGz' \
          % (nace.highSpeedSide.mass, nace.highSpeedSide.I[0], nace.highSpeedSide.I[1], nace.highSpeedSide.I[2], nace.highSpeedSide.cm[0], nace.highSpeedSide.cm[1], nace.highSpeedSide.cm[2])
    print 'Generator       %8.1f kg %6.2f Ixx %6.2f Iyy %6.2f Izz %6.2f CGx %6.2f CGy %6.2f CGz' \
          % (nace.generator.mass, nace.generator.I[0], nace.generator.I[1], nace.generator.I[2], nace.generator.cm[0], nace.generator.cm[1], nace.generator.cm[2])
    print 'Variable speed electronics %8.1f kg' % (nace.above_yaw_massAdder.vs_electronics_mass)
    print 'Transformer mass %8.1f kg' % (nace.transformer.mass)
    print 'Overall mainframe %8.1f kg' % (nace.above_yaw_massAdder.mainframe_mass)
    print 'Bedplate     %8.1f kg %8.1f m length %6.2f Ixx %6.2f Iyy %6.2f Izz %6.2f CGx %6.2f CGy %6.2f CGz' \
         % (nace.bedplate.mass, nace.bedplate.length, nace.bedplate.I[0], nace.bedplate.I[1], nace.bedplate.I[2], nace.bedplate.cm[0], nace.bedplate.cm[1], nace.bedplate.cm[2])
    print 'electrical connections  %8.1f kg' % (nace.above_yaw_massAdder.electrical_mass)
    print 'HVAC system     %8.1f kg' % (nace.above_yaw_massAdder.hvac_mass )
    print 'Nacelle cover:   %8.1f kg %6.2f m Height %6.2f m Width %6.2f m Length' % (nace.above_yaw_massAdder.cover_mass , nace.above_yaw_massAdder.height, nace.above_yaw_massAdder.width, nace.above_yaw_massAdder.length)
    print 'Yaw system      %8.1f kg' % (nace.yawSystem.mass )
    print 'Overall nacelle:  %8.1f kg .cm %6.2f %6.2f %6.2f I %6.2f %6.2f %6.2f' % (nace.nacelle_mass, nace.nacelle_cm[0], nace.nacelle_cm[1], nace.nacelle_cm[2], nace.nacelle_I[0], nace.nacelle_I[1], nace.nacelle_I[2]  )
    # print
    # print 'Mx:', nace.rotor_torque
    # print 'My:',nace.rotor_bending_moment_y
    # print 'Mz:',nace.rotor_bending_moment_z
    # print 
