"""
This module provides functions for creating ``matplotlib`` plots of
"raw" and calibrated light curves as well as folded time series.

:License: :doc:`LICENSE`

.. moduleauthor:: Daria Cara <daria.cara.2@gmail.com>

"""
import os

import numpy as np
import matplotlib.pyplot as plt


def _bin_folded_points(tls_results, duration_period_ratio=1.7, nbins='auto'):
    values = tls_results.folded_y
    phase = tls_results.folded_phase

    delta = duration_period_ratio * (tls_results.duration / tls_results.period)
    mask = np.logical_and(phase > (0.5 - delta), phase < (0.5 + delta))
    values = values[mask]
    phase = phase[mask]

    bin_edges = np.histogram_bin_edges(phase, bins=nbins)

    idx = np.digitize(phase, bin_edges)

    binned_values = []
    binned_phase = []
    for k in range(bin_edges.size):
        mask = idx == k
        if np.any(mask):
            binned_values.append(np.mean(values[mask]))
            binned_phase.append(np.mean(phase[mask]))

    return binned_values, binned_phase


def plot_lc(time, raw, cor, norm, trend, star_id, planet_id, provenance,
            tls_results, plotno=None, interactive=False):
    """
    Plot raw, corrected, and normalized light curves.

    Parameters
    ----------
    time: numpy.ndarray
        Time-stamp series.

    raw: numpy.ndarray
        Raw flux series.

    cor: numpy.ndarray
        "Detrended" flux series.

    norm: numpy.ndarray
        Normalized flux.

    trend: numpy.ndarray
        Trend data.

    star_id: int, str
        EPIC ID of the star.

    planet_id: int
        The number of the planet whose transit is currently being logged.

    provenance: str
        Provenance name in MAST archive, e.g., ``'K2'``, ``'EVEREST'``,
        ``'K2SFF'``.

    tls_results: dict
        The results from the TLS calculations.

    plotno: int
        Plot number to be passed directly to `~matplotlib.pyplot.figure`.
        If `None`, a new figure will be created.

    interactive: bool
        Indicates whether or not to create interactive plots. When
        ``interactive`` is `False`, figure will be saved to a file in the
        ``'Graphs/lc/'`` sub-directory. The figure's file name will be
        constructed using the following pattern:

            ``star_id + '_' + provenance + '_lc.log'``

    """
    star_id = str(star_id)
    if planet_id:
        star_id = '_'.join([star_id, str(planet_id)])

    fig = plt.figure(plotno, figsize=(11, 5))

    fig.add_subplot(311)
    plt.title(provenance, loc='right')
    plt.plot(time, raw, 'r.', label='RAW')
    plt.legend(loc='best')
    plt.title('Lightcurves for ' + star_id)

    fig.add_subplot(312)
    plt.plot(time, cor, 'b.', label='COR')
    plt.plot(time, trend, '#f97306')
    plt.legend(loc='best')

    fig.add_subplot(313)
    for tt in tls_results.transit_times:
        plt.axvline(x=tt, color='black', ls='--')
    plt.plot(time, norm, 'g.', label='NORM')
    plt.legend(loc='best')
    plt.xlabel('TIME (BTJD)\nPeriod: {:.6g}  Duration:  {:.5g}'
               .format(tls_results.period, tls_results.duration))

    plt.tight_layout()

    if interactive:
        plt.show()

    else:
        plt.savefig(
            os.path.join('Graphs/lc/',
                         '_'.join([star_id, provenance, '_lc.png']))
        )

    plt.close(fig)


def plot_tls(star_id, planet_id, provenance, plotno, tls_results, interactive):
    """
    Plot folded light curves from TLS results.
    raw, corrected, and normalized light curves.

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

    plotno: int
        Plot number to be passed directly to `~matplotlib.pyplot.figure`.
        If `None`, a new figure will be created.

    interactive: bool
        Indicates whether or not to create interactive plots. When
        ``interactive`` is `False`, figure will be saved to a file in the
        ``'Graphs/tls/'`` sub-directory. The figure's file name will be
        constructed using the following pattern:

            ``star_id + '_' + provenance + '_tls.log'``

    """
    DURATION_PERIOD_RATIO = 1.7
    star_id = str(star_id)
    if planet_id:
        star_id = '_'.join([star_id, str(planet_id)])

    fig = plt.figure(plotno, figsize=(9, 7))

    fig.add_subplot(313)
    plt.title('SDE vs. Period for ' + star_id)
    ax = plt.gca()
    ax.axvline(tls_results.period, alpha=0.4, lw=3)
    plt.xlim(np.min(tls_results.periods), np.max(tls_results.periods))
    for n in range(2, 10):
        ax.axvline(n * tls_results.period, alpha=0.4, lw=1, linestyle="dashed")
        ax.axvline(tls_results.period / n, alpha=0.4, lw=1, linestyle="dashed")
    plt.ylabel('SDE')
    plt.xlabel('Period (days)\nPeriod: {:.6g}    Duration {:.5g}'
               .format(tls_results.period, tls_results.duration))
    plt.plot(tls_results.periods, tls_results.power, color='black', lw=0.5)
    plt.xlim(0, max(tls_results.periods))

    fig.add_subplot(312)
    plt.title('Folded Transit Curve for ' + star_id)
    binned_y, binned_x = _bin_folded_points(tls_results)
    plt.plot(tls_results.model_folded_phase, tls_results.model_folded_model,
             color='red')
    plt.scatter(tls_results.folded_phase, tls_results.folded_y,
                color='blue', s=10, alpha=0.5, zorder=2)
    plt.scatter(binned_x, binned_y, color='yellow', s=40, alpha=1, zorder=2,
                edgecolors='black')
    delta = DURATION_PERIOD_RATIO * (tls_results.duration / tls_results.period)
    plt.xlim(0.5 - delta, 0.5 + delta)
    plt.xlabel('Phase')
    plt.ylabel('Relative flux')

    fig.add_subplot(311)
    plt.title(provenance, loc='right')
    plt.title('Folded Transit Curve for ' + star_id)
    plt.plot(tls_results.model_folded_phase, tls_results.model_folded_model,
             color='red')
    plt.scatter(tls_results.folded_phase, tls_results.folded_y, color='blue',
                s=10, alpha=0.5, zorder=2)
    plt.xlim(0.0, 1)
    plt.xlabel('Phase')
    plt.ylabel('Relative flux')

    plt.tight_layout()

    if interactive:
        plt.show()

    else:
        plt.savefig(
            os.path.join('Graphs/tls/',
                         '_'.join([star_id, provenance, '_tls.png']))
        )

    plt.close(fig)
