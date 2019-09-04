"""
This module provides main functionality for planet search.

:License: :doc:`LICENSE`

.. moduleauthor:: Daria Cara <daria.cara.2@gmail.com>

"""
# import glob
import os
import pandas as pd
from astropy.io import fits
from astropy.stats import sigma_clip
from scipy import signal
import numpy as np
from transitleastsquares import (transitleastsquares as tls,
                                 transit_mask, cleaned_array)

from mast import get_mast_file_list
from log_utils import create_summary_log, append_summary_log, log_results
from plot_tools import plot_lc, plot_tls


def detect_outliers(data, sigma=3, window_length=25, polyorder=3):
    """
    Detect outliers using sigma-clipping algorithm on "detrended" data.
    Trend is computed by applying a Savitzky–Golay filter to the input ``data``.

    Parameters
    ----------
    data: numpy.ndarray
        1D array of corrected flux data.

    sigma: float
        The number of standard deviations to use for both the lower and upper
        clipping limit. This parameter will be passed to
        `~astropy.stats.sigma_clip`.

    polyorder: int
        The order of the polynomial used to fit the samples. ``polyorder`` must
        be less than ``window_length``. This parameter will
        be passed to `~scipy.signal.savgol_filter`.

    window_length: int
        The length of the Savitzky–Golay filter window.
        ``window_length`` must be a positive odd integer. This parameter will
        be passed to `~scipy.signal.savgol_filter`.

    Returns
    -------
    lmask: numpy.ndarray
        An array of boolean type whose `True` values indicate data points
        that were below the lower clipping limit. For light curve data, these
        points may be both "true" outliers as well as transit points.

    umask: numpy.ndarray
        An array of boolean type whose `True` values indicate data points
        that were above the upper clipping limit.

    """
    assert np.all(np.isfinite(data))
    trend = signal.savgol_filter(
        data,
        window_length,
        polyorder=polyorder
    )
    norm_data = data / trend
    _, lower, upper = sigma_clip(
        norm_data, sigma=sigma, masked=False, return_bounds=True
    )
    lmask = np.less(norm_data, lower)
    umask = np.greater(norm_data, upper)
    return lmask, umask


def normalize_lc(data, lmask=None, window_length=25, polyorder=3):
    """
    Normalizes corrected flux data by dividing them by the trend. The trend is
    computed by applying a Savitzky–Golay filter to the input ``data``, except
    for the data that should be clipped according to ``lmask``. For those data
    a median filter is used instead with the kernel size given by
    ``window_length``.

    Parameters
    ----------
    data: numpy.ndarray
        1D array of corrected flux data.

    lmask: numpy.ndarray
        A 1D array of boolean type whose `True` values indicate data points
        that were below the lower clipping limit as returned
        by :py:func:`detect_outliers`. For light curve data, these
        points may be either actual outliers or transit points.

    polyorder: int
        The order of the polynomial used to fit the samples. ``polyorder`` must
        be less than ``window_length``. This parameter will
        be passed to `~scipy.signal.savgol_filter`.

    window_length: int
        The length of the Savitzky–Golay filter window or the size of the
        kernel of the median filter. ``window_length`` must be a positive odd
        integer. This parameter will be passed directly to either
        `~scipy.signal.savgol_filter` or `~scipy.signal.medfilt` depending of
        the value of ``lmask``.

    Returns
    -------
    norm_data: numpy.ndarray
        A 1D array of the same length as input data containing "detrended"
        data, i.e., data normalized by trend.

    trend: numpy.ndarray
        A 1D array of the same length as input data containing computed trend.

    """
    assert np.all(np.isfinite(data))
    if lmask is None:
        lmask = np.zeros_like(data, dtype=np.bool)
    data_med_for_tran = data.copy()

    # remove transits and other lower outliers from data:
    udata = data[np.logical_not(lmask)]

    # apply median filter to data without transits. This will be used to
    # replace clipped data.
    med_trend = signal.medfilt(udata, kernel_size=window_length)

    # replace masked data with median values:
    missing = np.flatnonzero(lmask)
    i = np.cumsum(np.logical_not(lmask)) - 1
    data_med_for_tran[missing] = med_trend[i[missing]]

    # find trend for all points with "transits" replaced by nearby medians:
    trend = signal.savgol_filter(
        data_med_for_tran,
        window_length,
        polyorder=polyorder
    )

    norm_data = data / trend

    return norm_data, trend


def get_lc(lc_data, provenance, use_quality=True):
    """
    Extract light curve data (raw and corrected fluxes, data quality, and time
    stamps) from Kepler data products.

    Parameters
    ----------
    lc_data: numpy.ndarray, str
        A structured array containing data series or a string file name
        of the file containing the data.

    provenance: str
        Provenance name in MAST archive, e.g., ``'K2'``, ``'EVEREST'``,
        ``'K2SFF'``.

    use_quality: bool
        If enabled, data flagged in the data quality array will be removed.

    Returns
    -------
    time: numpy.ndarray
        Time series.

    raw_flux: numpy.ndarray
        Raw flux series.

    cor_flux: numpy.ndarray
        Corrected flux series.

    """
    if isinstance(lc_data, str):
        lc_data = fits.getdata(lc_data, ext=1)

    if provenance == 'K2':
        raw_flux = lc_data['SAP_FLUX']
        cor_flux = lc_data['PDCSAP_FLUX']
        quality_kwd = 'SAP_QUALITY'
        time = lc_data['TIME']
        dq_bits = list(range(1, 22))

    elif provenance == 'EVEREST':
        raw_flux = lc_data['FRAW']
        cor_flux = lc_data['FLUX']
        quality_kwd = 'QUALITY'
        time = lc_data['TIME']
        dq_bits = list(range(1, 26))

    elif provenance == 'K2SFF':
        raw_flux = lc_data['FRAW']
        cor_flux = lc_data['FCOR']
        time = lc_data['T']
        quality_kwd = None  # No data quality information available for K2SFF

    else:
        raise ValueError("Unsupported provenance.")

    if quality_kwd and use_quality:
        quality = lc_data[quality_kwd]
        dq_mask = sum(1 << (bit - 1) for bit in dq_bits)
        mask = np.logical_not(quality & dq_mask)  # good data mask
        raw_flux = raw_flux[mask]
        cor_flux = cor_flux[mask]
        time = time[mask]

    mask = np.isfinite(time) & np.isfinite(raw_flux) & np.isfinite(cor_flux)

    return time[mask], raw_flux[mask], cor_flux[mask]


def _calc_stats(tls_results, sde_threshold=5):
    """
    Decide whether a detection is significant. Currently this function simply
    checks if ``results.SDE`` is larger than specified ``sde_threshold``.

    Parameters
    ----------
    tls_results: dict
        The results from the TLS calculations.

    sde_threshold: float
        Threshold for SDE for a detection to be considered significant.

    """
    significant = tls_results.SDE > sde_threshold
    return significant


def run_tls(time, normalized_flux):
    """
    Run TLS and return results. Print a summary of the results if TLS was
    successful.

    Parameters
    ----------
    time: numpy.ndarray
        Time series.

    normalized_flux: numpy.ndarray
        Detrended corrected flux as returned by :py:func:`normalize_lc`.

    Returns
    -------
    tls_results: dict
        The results from the TLS calculations.

    success: bool
        Indicates whether TLS was successful.

    """
    model = tls(time, normalized_flux)
    tls_results = model.power()
    if not np.isfinite(tls_results.period):
        print("INVALID RESULTS")
        return tls_results, False

    print("\n=====  RESULTS:  =====")
    print('Period: {:.5f} days'.format(tls_results.period))
    print('Transit depth: {:.5f}'.format(tls_results.depth))
    print('Best duration: {:.5f} days'.format(tls_results.duration))
    print('Signal detection efficiency (SDE): {:.5g}'.format(tls_results.SDE))

    print("Found {:d} transit times in time series:"
          .format(len(tls_results.transit_times)))
    ntransits = len(tls_results.transit_times)
    for i, t in enumerate(tls_results.transit_times[:5]):
        print("   T{:d}: {:0.5f}".format(i + 1, t))
    if ntransits == 6:
        print("   T{:d}: {:0.5f}".format(6, tls_results.transit_times[5]))
    elif ntransits > 6:
        print("   ... and {:d} more transit times: {}"
              .format(ntransits - 5, tls_results.transit_times[5:]))

    return tls_results, True


def find_planets(time, raw_flux, cor_flux, campaign_no,
                 star_id, provenance, summary_log,
                 max_planets=1, plotno=1, interactive=False):
    """
    Find one or more planet transits given input time, raw flux and corrected
    flux series. Save the results to the log files and display plots of
    light curves and folded light curves from TLS.

    Parameters
    ----------
    time: numpy.ndarray
        Time series.

    raw_flux: numpy.ndarray
        Raw flux series.

    cor_flux: numpy.ndarray
        Corrected flux series.

    campaign_no: int, str
        Campaign number or "sequence number". Can be a wildcard character `'*'`
        to indicate that all campaigns should be searched. Used for logging.

    star_id: int, str
        EPIC ID of the star. Used for logging.

    provenance: str
        Provenance name in MAST archive, e.g., ``'K2'``, ``'EVEREST'``,
        ``'K2SFF'``. Used for logging.

    summary_log: file object
        File object of the summary log file with write access and initialized
        with appropriate column names.

    max_planets: int
        Maximum number of planets to find.

    plotno: int
        Plot number to be passed directly to `~matplotlib.pyplot.figure`.
        If `None`, a new figure will be created.

    interactive: bool
        Indicates whether or not to draw figures on screen. When ``interactive``
        is `False`, figures are saved to files in the ``'Graphs/tls/'`` and
        ``'Graphs/lc/'`` sub-directories. Figure's file names will be
        constructed using the following pattern:

            ``star_id + '_' + provenance + '_tls.log'``
            ``star_id + '_' + provenance + '_lc.log'``
    """
    planet_id = 1
    while (max_planets and planet_id <= max_planets) or max_planets is None:
        lmask, umask = detect_outliers(cor_flux)
        # lmask indicates lower outliers and possible transits
        # umask indicates indicates outliers above the trend
        # remove upper outliers:
        umask = np.logical_not(umask)
        lmask = lmask[umask]
        time = time[umask]
        raw_flux = raw_flux[umask]
        cor_flux = cor_flux[umask]

        # "detrend" corrected flux:
        normalized_cor_flux, trend = normalize_lc(cor_flux, lmask)

        # run TLS:
        tls_results, valid = run_tls(time, normalized_cor_flux)
        if not valid:
            break

        # if the detection is significant, plot it and log it:
        if _calc_stats(tls_results):
            if isinstance(tls_results.transit_times, list):
                plot_lc(time, raw_flux, cor_flux, normalized_cor_flux,
                        trend, star_id, planet_id, provenance, tls_results,
                        interactive=interactive)
                plotno += 1

                plot_tls(star_id, planet_id, provenance, plotno, tls_results,
                         interactive=interactive)
                plotno += 1

                log_results(star_id, planet_id, provenance, tls_results)
                append_summary_log(summary_log, star_id, campaign_no,
                                   planet_id, provenance, tls_results)

        # Remove detected transit and remove it from the data series:
        intransit = transit_mask(
            time, tls_results.period, 2 * tls_results.duration, tls_results.T0
        )
        raw_flux = raw_flux[~intransit]
        cor_flux = cor_flux[~intransit]
        time = time[~intransit]
        time, cor_flux = cleaned_array(time, cor_flux)
        planet_id += 1


def find_planets_around_stars(stars_df, campaign_no, max_planets=None,
                              provenance=["EVEREST", "K2SFF", "K2"],
                              log_file='summary_log.csv', interactive=False):
    """
    Find one or more planet transits given input time, raw flux and corrected
    flux series. Save the results to the log files and display plots of
    light curves and folded light curves from TLS.

    Parameters
    ----------
    stars_df: str, pandas.DataFrame, list of str or int, tuple of str or int
        String file name of a CSV file containing a column named `'EPIC'`.
        The first line in the file must be column header and the second line
        is ignored. Alternatively, ``stars_df`` can be a `~pandas.DataFrame`
        containing a column named `'EPIC'`. Other options are a list or tuple
        of string or integer EPIC IDs of the stars around which to search for
        transits.

    campaign_no: int, str
        Campaign number or "sequence number". Can be a wildcard character `'*'`
        to indicate that all campaigns should be searched.

    max_planets: int
        Maximum number of planets to find.

    star_id: int, str
        EPIC ID of the star.

    provenance: str
        Provenance name in MAST archive, e.g., ``'K2'``, ``'EVEREST'``,
        ``'K2SFF'``.

    log_file: str
        File name of the summary log file to which all planet search results
        will be logged.

    interactive: bool
        Indicates whether or not to draw figures on screen. When ``interactive``
        is `False`, figures are saved to files in the ``'Graphs/tls/'`` and
        ``'Graphs/lc/'`` sub-directories. Figure's file names will be
        constructed using the following pattern:

            ``star_id + '_' + provenance + '_tls.log'``
            ``star_id + '_' + provenance + '_lc.log'``

    """
    os.makedirs('logs/', exist_ok=True)
    os.makedirs('Graphs', exist_ok=True)
    os.makedirs('Graphs/tls/', exist_ok=True)
    os.makedirs('Graphs/lc/', exist_ok=True)

    sequence_name = str(campaign_no).strip()

    if isinstance(stars_df, str):
        stars_df = pd.read_csv(stars_df, header=[0], skiprows=[1])

    if isinstance(stars_df, (list, tuple)):
        stars_df = pd.DataFrame({'EPIC': stars_df})

    elif not isinstance(stars_df, pd.DataFrame):
        raise TypeError("'stars_df' must be either a Pandas DataFrame or a "
                        "string file name.")

    if 'EPIC' not in stars_df.columns.values:
        raise ValueError("Input list of stars does not have an 'EPIC' column.")

    catalog = stars_df['EPIC'].unique().tolist()

    if isinstance(provenance, str):
        provenance = [provenance]

    summary_log = create_summary_log(log_file)

    plotno = 1

    for star_id in catalog:
        for prov in provenance:
            try:
                data_files = get_mast_file_list(
                    str(star_id), prov, sequence_name
                )
                # DEBUG:
                # In order to bypass MAST and use already locally downloaded
                # data files uncomment lines below and the 'glob' import
                # at the beginning of the module:
                # data_files = glob.glob(
                #     '/.../mastDownload/K2/*/*{:s}*{:s}*.fits'
                #     .format(prov, star_id)
                # )

            except Exception as e:
                print("There was an issue retrieving the files for {} in "
                      "the {} data set.".format(star_id, prov))
                print("Reported error: '{}'".format(e.args[0]))
                continue

            try:
                time, raw_flux, cor_flux = get_lc(data_files[0], prov)
                if time.size < 10:
                    print("WARNING: Too few data to find transit.")
                    continue

            except ValueError:
                continue

            find_planets(
                time, raw_flux, cor_flux, campaign_no=sequence_name,
                star_id=star_id, provenance=prov, summary_log=summary_log,
                max_planets=max_planets, plotno=plotno, interactive=interactive
            )

    summary_log.close()
