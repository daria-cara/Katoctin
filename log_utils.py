"""
Utility functions for logging results to different file formats.

:License: :doc:`LICENSE`

.. moduleauthor:: Daria Cara <daria.cara.2@gmail.com>

"""
import os
from datetime import date
from collections import OrderedDict

import numpy as np
import pandas as pd


def create_summary_log(log_file_name):
    """
    Creates a CSV log file in the ``log/`` sub-directory to which the
    summary log for a planet search will be saved.

    Parameters
    ----------
    log_file_name: str
        File name of the CSV log file to be created.

    Returns
    -------
    f: file object
        File object of the summary log file with write access and initialized
        with appropriate column names.

    """
    log_file_path = os.path.join('logs/', log_file_name)

    output_colnames = [
        'star_id',
        'campaign',
        'planet_id',
        'project_name',
        'period',
        'transit_depth',
        'transit_midpoint',
        'duration',
        'SDE',
    ]

    pdf = pd.DataFrame(columns=output_colnames)
    f = open(log_file_path, 'w')
    pdf.to_csv(f, header=True, index=False)
    f.flush()

    return f


def append_summary_log(log_file, star_id, campaign,
                       planet_id, provenance, tls_results):
    """ Appends a new entry to the summary log file.

    Parameters
    ----------
    log_file: file object
        An open file object of the summary log.

    star_id: int, str
        EPIC ID of the star.

    campaign: str
        K2 campaign of the observation of the star.

    planet_id: int
        The number of the planet whose transit is currently being logged.

    provenance: str
        Provenance name in MAST archive, e.g., ``'K2'``, ``'EVEREST'``,
        ``'K2SFF'``.

    tls_results: dict
        The results from the TLS calculations.

    """
    pdf = pd.DataFrame(
        OrderedDict(
            [
                ('star_id', [star_id]),
                ('campaign', [campaign]),
                ('planet_id', [planet_id]),
                ('prov_name', [provenance]),
                ('period', [tls_results.period]),
                ('transit_depth', [tls_results.depth]),
                ('transit_midpoint', [np.mean(tls_results.transit_times)]),
                ('duration', [tls_results.duration]),
                ('SDE', [tls_results.SDE]),
            ]
        ),
        index=None
    )
    pdf.to_csv(log_file, header=False, index=False)
    log_file.flush()


def log_results(star_id, planet_id, provenance, tls_results):
    """
    Create a text log file and log main results of a planet search.
    The log file name will be constructed using the following pattern:

        ``star_id + '_' + provenance + '_' + today_date + '.log'``

    The file will be created in the ``log/`` sub-directory.

    Parameters
    ----------
    star_id: int, str
        EPIC ID of the star.

    planet_id: int
        The number of the planet whose transit is currently being logged.

    provenance: str
        Provenance name in MAST archive, e.g., ``'K2'``, ``'EVEREST'``,
        ``'K2SFF'``.

    tls_results: dict
        The results from the TLS calculations.

    """
    star_id = str(star_id)
    if planet_id:
        star_id = '_'.join([star_id, str(planet_id)])

    today = date.today()
    today = today.strftime("%Y%m%d")
    file_name = os.path.join('logs/', '_'.join([star_id, provenance, today]) + '.log')
    with open(file_name, 'w') as f:
        f.write('Star ID: {}\n'.format(star_id))
        if planet_id:
            f.write('Planet ID: {}\n'.format(planet_id))
        f.write('Provenance: {}\n'.format(provenance))
        f.write('Period (d): {}\n'.format(tls_results.period))
        f.write('Transit Times: {}\n'.format(tls_results.transit_times))
        f.write('Transit Depth: {}\n'.format(tls_results.depth))
        f.write('Duration: {}\n'.format(tls_results.duration))
        f.write('SDE: {}\n'.format(tls_results.SDE))
