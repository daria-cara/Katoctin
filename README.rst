.. image:: https://img.shields.io/badge/License-MIT-blue.svg
   :target: https://github.com/daria-cara/Katoctin/blob/master/LICENSE.txt
   :alt: MIT license

.. image:: https://img.shields.io/badge/Contributor%20Covenant-v1.4%20adopted-ff69b4.svg
   :target: CODE_OF_CONDUCT.md
   :alt: Contributor Covenant

.. image:: https://img.shields.io/badge/python-3.5%20%203.6%20%203.7-blue.svg
   :alt: Python version support


Katoctin
========

``katoctin`` is a package that automates multi-planet search in archival
K2 light curve data. The algorithm is based on
`Transit Least Squares (TLS) <https://github.com/hippke/tls>`_
to find planet transits in K2 light curves. The code uses a custom detrending
algorithm based on Savitzky–Golay filtering to normalize light curves.

This is my summer project at Space Telescope Science Institute. Many
thanks to Dr. Susan Mullally, who mentored me, and STScI for giving me the
opportunity to create this.


Documentation
=============

``planet_search``
-----------------

Command line script for planet search.

:Usage:

.. code-block:: bash

  planet_search [-h] (-f FILE | -s STARS) -c CAMPAIGN [-l LOG]
                [-p PROVENANCE] [-m MAX_PLANETS] [-i]

Find multiple planets around each star in the input list.

:Optional arguments:
  -h, --help            show this help message and exit
  -f FILE, --file FILE  Name of a CVS file containing EPIC star IDs.
  -s STARS, --stars STARS
                        Comma-separated list of EPIC star IDs.
  -c CAMPAIGN, --campaign CAMPAIGN
                        Campaign number.
  -l LOG, --log LOG     File name of the log file containing results.
  -p PROVENANCE, --provenance PROVENANCE
                        Find planets using data from the specified provenance.
  -m MAX_PLANETS, --max_planets MAX_PLANETS
                        Maximum number of planets to find for each star.
  -i, --interactive     Whether or not to be interactive plots that allow for
                        zooming in.

:Examples:

.. code-block:: bash

    $ ./planet_search -s 228863643 -i -c 1
    $ ./planet_search -s '228863643,228867558' -i
    $ ./planet_search -s '228863643,228867558' -i -c '*'
    $ ./planet_search -s '228863643,228867558' -i -c '*' -m 2


``fcomp``
---------

A script for comparing planet periods found as a result of planet search
and saved in the summary log with periods downloaded from MAST.

:Usage:

.. code-block:: bash

  fcomp [-h] -s SUMMARY -m MAST -o OUTPUT

Compare found transit periods with known values from MAST.

:Optional arguments:

  -h, --help            show this help message and exit
  -s SUMMARY, --summary SUMMARY
                        File name of the summary log with computed results.
  -m MAST, --mast MAST  File name of the mast file containing the confirmed
                        period and data.
  -o OUTPUT, --output OUTPUT
                        File name to which to save the results of the
                        comparison.

:Examples:

.. code-block:: bash

    $ ./fcomp -s logs/summary_log.csv -m mast_data/k2_mast.csv -o comp_results.csv
