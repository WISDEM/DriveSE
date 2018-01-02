DriveSE is a set of models to size wind turbine components from the hub system, drivetrain and overall nacelle.  It replaces the Drive WindPACT (DriveWPACT) model which was based on older technology and empirical data.  The new models are physics-based and provide sizing of components based off of key system configuration parameters as well as the aerodynamic loads from the rotor. 

Author: [Y. Guo, R. King and T. Parsons](nrel.wisdem+drivese@gmail.com)

## Version

This software is a beta version 0.1.3.

## Detailed Documentation

For detailed documentation see <http://wisdem.github.io/DriveSE/>

## Prerequisites

General: NumPy, SciPy, Swig, pyWin32, MatlPlotLib, Lxml, OpenMDAO

## Dependencies:

Wind Plant Framework: [FUSED-Wind](http://fusedwind.org) (Framework for Unified Systems Engineering and Design of Wind Plants)

Sub-Models: CommonSE, DriveWPACT

Supporting python packages: Pandas, Algopy, Zope.interface, Sphinx, Xlrd, PyOpt, py2exe, Pyzmq, Sphinxcontrib-bibtex, Sphinxcontrib-zopeext, Numpydoc, Ipython

## Installation

First, clone the [repository](https://github.com/WISDEM/DriveSE)
or download the releases and uncompress/unpack (DriveSE.py-|release|.tar.gz or DriveSE.py-|release|.zip) from the website link at the bottom the [DriveSE site](http://nwtc.nrel.gov/DriveSE).

Install DriveSE within an activated OpenMDAO environment:

	$ plugin install

It is not recommended to install the software outside of OpenMDAO.

## Run Unit Tests

To check if installation was successful try to import the module within an activated OpenMDAO environment.

	$ python
	> import drivese.drive
	> import drivese.hub

You may also run the unit tests which include functional and gradient tests.  Analytic gradients are provided for variables only so warnings will appear for missing gradients on model input parameters; these can be ignored.

	$ python src/test/test_DriveSE.py

For software issues please use <https://github.com/WISDEM/DriveSE/issues>.  For functionality and theory related questions and comments please use the NWTC forum for [Systems Engineering Software Questions](https://wind.nrel.gov/forum/wind/viewtopic.php?f=34&t=1002).
