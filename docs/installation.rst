Installation
------------

.. admonition:: prerequisites
   :class: warning

   NumPy, SciPy, FUSED-Wind, OpenMDAO, CommonSE, DriveWPACT

Clone the repository at `<https://github.com/WISDEM/DriveSE>`_
or download the releases and uncompress/unpack (DriveSE.py-|release|.tar.gz or DriveSE.py-|release|.zip)

To install DriveSE, first activate the OpenMDAO environment and then install with the following command.

.. code-block:: bash

   $ plugin install

To check if installation was successful try to import the module

.. code-block:: bash

    $ python

.. code-block:: python

    > import drivese.drive
    > import drivese.hub

or run the unit tests which include functional and gradient tests.  Analytic gradients are provided for variables only so warnings will appear for missing gradients on model input parameters; these can be ignored.

.. code-block:: bash

   $ python src/test/test_DriveSE.py

An "OK" signifies that all the tests passed.

For software issues please use `<https://github.com/WISDEM/DriveSE/issues>`_.  For functionality and theory related questions and comments please use the NWTC forum for `Systems Engineering Software Questions <https://wind.nrel.gov/forum/wind/viewtopic.php?f=34&t=1002>`_.

.. only:: latex

    An HTML version of this documentation that contains further details and links to the source code is available at `<http://wisdem.github.io/DriveSE>`_