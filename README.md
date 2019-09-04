# DEPRECATED
------------

**THIS REPOSITORY IS DEPRECATED AND WILL BE ARCHIVED (READ-ONLY) IN NOVEMBER 2019.**

WISDEM has moved to a single, integrated repository at https://github.com/wisdem/wisdem

---------------
# DriveSE

DriveSE is a set of models to size wind turbine components from the hub system, drivetrain and overall nacelle.  It replaces the Drive WindPACT (DriveWPACT) model which was based on older technology and empirical data.  The new models are physics-based and provide sizing of components based off of key system configuration parameters as well as the aerodynamic loads from the rotor. 

Author: [NREL WISDEM Team](mailto:systems.engineering@nrel.gov) 

## Documentation

See local documentation in the `docs`-directory or access the online version at <http://wisdem.github.io/DriveSE/>

## Installation

For detailed installation instructions of WISDEM modules see <https://github.com/WISDEM/WISDEM> or to install DriveSE by itself do:

    $ python setup.py install

## Run Unit Tests

To check if installation was successful try to import the package:

	$ python
	> import drivese.drive
	> import drivese.hub

You may also run the unit tests which include functional and gradient tests.  Analytic gradients are provided for variables only so warnings will appear for missing gradients on model input parameters; these can be ignored.

	$ python src/test/test_DriveSE.py

For software issues please use <https://github.com/WISDEM/DriveSE/issues>.  For functionality and theory related questions and comments please use the NWTC forum for [Systems Engineering Software Questions](https://wind.nrel.gov/forum/wind/viewtopic.php?f=34&t=1002).
