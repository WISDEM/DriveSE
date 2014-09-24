DriveSE is a set of models to size wind turbine components from the hub system, drivetrain and overall nacelle.

Author: [Y. Guo](mailto:yi.guo@nrel.gov)

## Version

This software is a beta version 0.1.0.

## Detailed Documentation

For detailed documentation see <http://wisdem.github.io/DriveSE/>

## Prerequisites

NumPy, SciPy, FUSED-Wind, OpenMDAO

## Installation

Install DriveSE within an activated OpenMDAO environment

	$ plugin install

It is not recommended to install the software outside of OpenMDAO.

## Run Unit Tests

To check if installation was successful try to import the module

	$ python
	> import drivese.drive
	> import drivese.hub

You may also run the unit tests.

	$ python src/test/test_DriveSE_gradients.py

