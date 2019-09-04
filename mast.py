"""
Tools for retrieving K2 data from MAST archive.

:License: :doc:`LICENSE`

.. moduleauthor:: Daria Cara <daria.cara.2@gmail.com>

"""
import numpy as np
from astroquery.mast import Observations


_MISSION_MAPPING = {
    'K2': 'K2',
    'EVEREST': 'HLSP',
    'K2SFF': 'HLSP',
}


def get_mast_file_list(star_id, provenance, sequence_name):
    """
    Queries MAST to retrieve archive data based on object name (``star_id``),
    data provenance, and sequence name.

    Parameters
    ----------
    star_id: int, str
        EPIC ID of the star.

    provenance: str
        Provenance name in MAST archive, e.g., ``'K2'``, ``'EVEREST'``,
        ``'K2SFF'``.

    sequence_name: str
        Campaign number.

    Returns
    -------
    file_names: list of str
        A Python list of all the file names retrieved from archive.

    """
    star_id = str(star_id)
    target_name = '*' + star_id.split()[-1]

    # Make sure that there are data for the criteria
    mission = _MISSION_MAPPING[provenance]

    print("\n=====  Retrieving data for observation:  =====")
    print("** star_id: {}\n** provenance: {}\n** sequence_name: {}"
          .format(star_id, provenance, sequence_name))

    if sequence_name == '*':
        sequence_name = ''

    obs_count = Observations.query_criteria_count(
        obs_collection=mission,
        dataproduct_type=["timeseries"],
        instrument_name="Kepler",
        objectname=star_id,
        target_name=target_name,
        project="K2",
        provenance_name=provenance,
        sequence_number=sequence_name + '*'
    )
    if obs_count == 0:
        raise RuntimeError("No data found in archive.")

    obs_table = Observations.query_criteria(
        obs_collection=mission,
        dataproduct_type=["timeseries"],
        instrument_name="Kepler",
        objectname=star_id,
        target_name=target_name,
        project="K2",
        provenance_name=provenance,
        sequence_number=sequence_name + '*'
    )

    data_products = Observations.get_product_list(obs_table)

    lc_mask = ["lightcurve" in x or "light curve" in x for x in
               map(str.lower, data_products['description'])]
    if not any(lc_mask):
        raise RuntimeError("Retrieved data products do not contain light "
                           "curve data.")

    data_products = data_products[lc_mask]  # keep only rows with light curves
    manifest = Observations.download_products(data_products)
    files = list(manifest['Local Path'])  # get local file names

    # sort results:
    idx = np.argsort(files)
    file_names = [files[i] for i in idx]

    print("Download Status: SUCCESS\n")

    return file_names
