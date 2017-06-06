from drive_DFIG import sys_print, Drive3pt
from scipy.interpolate import interp1d
import numpy as np
import pandas as pd
from openmdao.main.api import Assembly
from openmdao.lib.drivers.api import SLSQPdriver, COBYLAdriver

# NREL 5 MW Rotor Variables
print '----- NREL 5 MW Turbine - 3 Point Suspension -----'

nace = Drive3pt() 
nace.rotor_diameter = 130.0 # m
nace.rotor_speed = 11.753 # #rpm m/s
nace.machine_rating = 3600.0
nace.Target_Efficiency = 93
nace.Gearbox_efficiency=0.955
nace.Gearbox_output=nace.machine_rating*nace.Gearbox_efficiency
nace.rotor_torque =  1.5 * (nace.machine_rating * 1000 / nace.Gearbox_efficiency)/(nace.rotor_speed * (np.pi / 30)) # 
nace.rotor_thrust = 1.120e6 # N
nace.rotor_mass = 0.0 #accounted for in F_z # kg
nace.rotor_speed = 11.753 #rpm
nace.rotor_bending_moment = 1.11e7 # Nm same as rotor_bending_moment_y
nace.rotor_bending_moment_x = 3.83e6 # Nm
nace.rotor_bending_moment_y = 1.11e7 # Nm
nace.rotor_bending_moment_z = 1.17e7 # Nm
nace.rotor_force_x = 1.120e6 # N
nace.rotor_force_y = 2.51e5 # N
nace.rotor_force_z = -1.03e6 # N


# NREL 5 MW Drivetrain variables
nace.drivetrain_design = 'DFIG' # geared 3-stage Gearbox with induction generator machine
nace.machine_rating = 3600.0 # kW
nace.gear_ratio = 102.1 # 97:1 as listed in the 5 MW reference document
nace.gear_configuration = 'eep' # epicyclic-epicyclic-parallel
nace.crane = True # onboard crane present
nace.shaft_angle = 5.0 #deg
nace.shaft_ratio = 0.10
nace.Np = [3,3,1]
nace.ratio_type = 'optimal'
nace.shaft_type = 'normal'
nace.uptower_transformer=True
nace.shrink_disc_mass = 333.3*nace.machine_rating/1000.0 # estimated
nace.carrier_mass = 8000.0 # estimated
nace.mb1Type = 'SRB'
nace.mb2Type = 'SRB'
nace.flange_length = 0.5
nace.overhang = 5.0
nace.L_rb = 0 # length from hub center to main bearing, leave zero if unknow

nace.check_fatigue = 0 #0 if no fatigue check, 1 if parameterized fatigue check, 2 if known loads inputs

# Generator design variables for DFIG
nace.DFIG_r_s= 0.4  #meter
nace.DFIG_l_s= 1.4 #meter
nace.DFIG_h_s = 0.1 #meter
nace.DFIG_h_r = 0.1 #meter
nace.DFIG_I_0 = 29.5  # Ampere
nace.DFIG_B_symax = 1. # Tesla
nace.DFIG_S_N = -0.1 # Tesla

            
#variables if check_fatigue = 1:
nace.blade_number=3
nace.cut_in=3. #cut-in m/s
nace.cut_out=25. #cut-out m/s
nace.Vrated=11.4 #rated windspeed m/s
nace.weibull_k = 2.2 # windepeed distribution shape parameter
nace.weibull_A = 9. # windspeed distribution scale parameter
nace.T_life=20. #design life in years
nace.IEC_Class_Letter = 'A'

#variables if check_fatigue =2:
#test distribution
p_o = 6444.24
R = nace.rotor_diameter*.5
count = np.logspace(np.log10(328),np.log10(288984470), endpoint=True , num=100)
standard = count.copy()
for i in range(len(count)):
    standard[i] = .11*2.5*.28*13.4*(np.log10(288984470)-np.log10(count[i]))+0.18
Fx_factor = (.3649*np.log(nace.rotor_diameter)-1.074)
Mx_factor = (.0799*np.log(nace.rotor_diameter)-.2577)
My_factor = (.172*np.log(nace.rotor_diameter)-.5943)
Mz_factor = (.1659*np.log(nace.rotor_diameter)-.5795)
nace.rotor_thrust_distribution = standard.copy()**0.5*p_o*(R)*Fx_factor
nace.rotor_thrust_count = np.logspace(np.log10(328),np.log10(288984470), endpoint=True , num=100)
nace.rotor_Fy_distribution = np.zeros(100)
nace.rotor_Fy_count = nace.rotor_thrust_count.copy()
nace.rotor_Fz_distribution = np.zeros(100)
nace.rotor_Fz_count = nace.rotor_thrust_count.copy()
nace.rotor_torque_distribution = standard.copy()*0.45*p_o*(R)**2*Mx_factor
nace.rotor_torque_count = nace.rotor_thrust_count.copy()
nace.rotor_My_distribution = standard.copy()*0.33*p_o*.8*(R)**2*My_factor
nace.rotor_My_count = nace.rotor_thrust_count.copy()
nace.rotor_Mz_distribution = standard.copy()*0.33*p_o*.8*(R)**2*Mz_factor
nace.rotor_Mz_count = nace.rotor_thrust_count.copy() 

# NREL 5 MW Tower Variables
nace.tower_top_diameter = 4.73 # m
nace.shaft_ratio = 0.6

class NOpt(Assembly):
    def configure(self):
        self.add('Nacelle', nace)
        self.add('driver',COBYLAdriver())
        self.driver.workflow.add('Nacelle')
        self.driver.accuracy = 1.0e-6
        self.driver.maxiter = 5000

        self.driver.add_parameter('Nacelle.shaft_ratio', low=0.02, high=0.9)
        self.driver.add_objective('Nacelle.nacelle_mass')


nace.run() 
mass_before = nace.nacelle_mass 
sr_before = nace.shaft_ratio
Opt = NOpt()
Opt.run()
print "SR before: ", sr_before
print "SR after: ", Opt.Nacelle.shaft_ratio
print "Mass before: ", mass_before
print "Mass after : ", Opt.Nacelle.nacelle_mass
quit()
raw_data = {'Parameters': ['Rating','Objective function','Air gap diameter', "Stator length","Lambda ratio","Diameter ratio", "Pole pitch(tau_p)", " Number of Stator Slots","Stator slot height(h_s)","Slots/pole/phase","Stator slot width(b_s)", " Stator slot aspect ratio","Stator tooth width(b_t)", "Stator yoke height(h_ys)","Rotor slots", "Rotor yoke height(h_yr)", "Rotor slot height(h_r)", "Rotor slot width(b_r)"," Rotor Slot aspect ratio", "Rotor tooth width(b_t)", "Peak air gap flux density","Peak air gap flux density fundamental","Peak stator yoke flux density","Peak rotor yoke flux density","Peak Stator tooth flux density","Peak rotor tooth flux density","Pole pairs", "Generator output frequency", "Generator output phase voltage", "Generator Output phase current","Optimal Slip","Stator Turns","Conductor cross-section","Stator Current density","Specific current loading","Stator resistance", "Stator leakage inductance", "Excited magnetic inductance"," Rotor winding turns","Conductor cross-section","Magnetization current","I_mag/Is"," Rotor Current density","Rotor resitance", " Rotor leakage inductance", "Generator Efficiency","Iron mass","Copper mass","Structural Steel mass","Total Mass","Total Material Cost"], 'Values': [Opt.Nacelle.comp.DFIG.P_gennom/1e6,nace.comp.Objective_function,2*nace.comp.DFIG.DFIG_r_s,nace.comp.DFIG.DFIG_l_s,nace.comp.DFIG.lambda_ratio,nace.comp.DFIG.D_ratio,nace.comp.DFIG.tau_p*1000,nace.comp.DFIG.N_slots,nace.comp.DFIG.DFIG_h_s*1000,nace.comp.DFIG.q1,nace.comp.DFIG.b_s*1000,nace.comp.DFIG.Slot_aspect_ratio1,nace.comp.DFIG.b_t*1000,nace.comp.DFIG.h_ys*1000,nace.comp.DFIG.Q_r,nace.comp.DFIG.h_yr*1000,nace.comp.DFIG.DFIG_h_r*1000,nace.comp.DFIG.b_r*1000,nace.comp.DFIG.Slot_aspect_ratio2,nace.comp.DFIG.b_tr*1000,nace.comp.DFIG.B_g,nace.comp.DFIG.B_g1,nace.comp.DFIG.DFIG_B_symax,nace.comp.DFIG.B_rymax,nace.comp.DFIG.B_tsmax,nace.comp.DFIG.B_trmax,nace.comp.DFIG.p,nace.comp.DFIG.f,nace.comp.DFIG.E_p,nace.comp.DFIG.I_s,nace.comp.DFIG.DFIG_S_N,nace.comp.DFIG.W_1a,nace.comp.DFIG.A_Cuscalc,nace.comp.DFIG.J_s,nace.comp.DFIG.A_1/1000,nace.comp.DFIG.R_s,nace.comp.DFIG.L_s,nace.comp.DFIG.L_sm,nace.comp.DFIG.W_2,nace.comp.DFIG.A_Curcalc,nace.comp.DFIG.DFIG_I_0,nace.comp.DFIG.Current_ratio,nace.comp.DFIG.J_r,nace.comp.DFIG.R_R,nace.comp.DFIG.L_r,nace.comp.DFIG.gen_eff,nace.comp.DFIG.Iron/1000,nace.comp.DFIG.Cu/1000,nace.comp.DFIG.Structure/1000,nace.comp.DFIG.M_actual/1000,nace.comp.DFIG.Costs/1000],
                    'Limit': ['','','','','(0.2-1.5)','(1.37-1.4)','','','','','','(4-10)','','','','','','','(4-10)','','(0.7-1.2)','','2','2.1','2.1','2.1','','','(500-5000)','','(-0.002-0.3)','','','(3-6)','<60','','','','','','','(0.1-0.3)','(3-6)','','','>93','','','','',''],
                            'Units':['MW','','m','m','-','-','mm','-','mm','','mm','','mm','mm','-','mm','mm','mm','-','mm','T','T','T','T','T','T','-','Hz','V','A','','turns','mm^2','A/mm^2','kA/m','ohms','p.u','p.u','turns','mm^2','A','','A/mm^2','ohms','p.u','%','Tons','Tons','Tons','Tons','$1000']}
df=pd.DataFrame(raw_data, columns=['Parameters','Values','Limit','Units'])
print(df)

#cm_print(nace)
sys_print(Opt.Nacelle)
