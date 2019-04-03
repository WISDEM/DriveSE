# DriveSE Changelog

## 0.2.0 ([04/01/2019])

[Garrett Barter](mailto: garrett.barter@nrel.gov)

- OpenMDAO1 release

## 0.1.0 ([09/30/2014])

[Yi Guo](mailto: yi.guo@nrel.gov)

- initial release

## 0.1.1 ([02/09/2015])

[Taylor Parsons](mailto: taylor.parsons@nrel.gov)

[Fixes]

- bearing tables replaced by interpolation functions

- user-defined fatigue code commented out

- transformer mass added back into above yaw mass

- flange length equations updated to be quadratic

## 0.1.2 ([07/08/2015])

[Taylor Parsons](mailto: taylor.parsons@nrel.gov)

[Changes]

- Updates made to remove functions to separate drivese_utils file so that they can be used by both the gradient-friendly smooth version of DriveSE and normal DriveSE

- Added hooks to handle full suite of rotor loads coming from RotorSE

