
"""
test_Turbine_CostsSE.py

Created by Katherine Dykes on 2014-01-07.
Copyright (c) NREL. All rights reserved.
"""

import unittest
import numpy as np
from commonse.utilities import check_gradient_unit_test

from drivese.drive_smooth import BearingSmooth, YawSystemSmooth, BedplateSmooth

class TestBearingSmooth(unittest.TestCase):

    def test1(self):
        comp = BearingSmooth()
        comp.bearing_type = 'SRB'
        comp.lss_diameter = 0.721049014299
        comp.rotor_diameter = 125.740528176
        comp.bearing_switch = 'main'

        check_gradient_unit_test(self, comp)


class TestYawSystemSmooth(unittest.TestCase):

    def test1(self):
        comp = YawSystemSmooth()
        comp.rotor_diameter = 125.740528176
        comp.tower_top_diameter = 3.87

        check_gradient_unit_test(self, comp)


class TestBedplateSmooth(unittest.TestCase):

    def test1(self):
        comp = BedplateSmooth()
        comp.hss_location = 0.785878301101
        comp.hss_mass = 2288.26758514
        comp.generator_location = 1.5717566022
        comp.generator_mass = 16699.851325
        comp.lss_location = -3.14351320441
        comp.lss_mass = 12546.3193435
        comp.mb1_location = -1.25740528176
        comp.mb1_mass = 3522.06734168
        comp.mb2_location = -4.40091848617
        comp.mb2_mass = 5881.81400444
        comp.tower_top_diameter = 3.87
        comp.rotor_diameter = 125.740528176
        comp.machine_rating = 5000.0
        comp.rotor_mass = 93910.5225629
        comp.rotor_bending_moment_y = -2325000.0
        comp.rotor_force_z = -921262.226342
        comp.h0_rear = 1.35
        comp.h0_front = 1.7

        check_gradient_unit_test(self, comp)
        
if __name__ == "__main__":
    unittest.main()
    