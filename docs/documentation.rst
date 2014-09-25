.. _documentation-label:

.. currentmodule:: drivese.drive

Documentation
--------------

.. only:: latex

    An HTML version of this documentation is available which is better formatted for reading the code documentation and contains hyperlinks to the source code.


Turbine component sizing models for hub and drivetrain components are described along with mass-cost models for the full set of turbine components from the rotor to tower and foundation.

Documentation for DriveSE
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The following inputs and outputs are defined for DriveWPACT:

.. literalinclude:: ../src/drivese/drive.py
    :language: python
    :start-after: Drive4pt(Assembly)
    :end-before: def configure(self)
    :prepend: class Drive4pt(Assembly):

Implemented Base Model
=========================
.. module:: drivewpact.drive
.. class:: NacelleBase

Referenced Sub-System Modules 
==============================
.. module:: drivese.drivese_components
.. class:: LowSpeedShaft_drive
.. class:: LowSpeedShaft_drive4pt
.. class:: LowSpeedShaft_drive3pt
.. class:: MainBearing_drive
.. class:: SecondBearing_drive
.. class:: Gearbox_drive
.. class:: HighSpeedSide_drive
.. class:: Generator_drive
.. class:: Bedplate_drive
.. class:: AboveYawMassAdder_drive
.. class:: YawSystem_drive
.. class:: NacelleSystemAdder_drive


.. currentmodule:: drivese.hub

Documentation for HubSE
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The following inputs and outputs are defined for HubWPACT:

.. literalinclude:: ../src/drivese/hub.py
    :language: python
    :start-after: HubSE(Assembly)
    :end-before: def configure(self)
    :prepend: class HubSE(Assembly):

Implemented Base Model
=========================
.. module:: drivewpact.hub
.. class:: HubBase

Referenced Sub-System Modules 
==============================
.. module:: drivewpact.hub
.. class:: PitchSystem
.. class:: Spinner
.. class:: HubSystemAdder
.. module:: drivese.hub
.. class:: Hub_drive
