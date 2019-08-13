from datetime import date
from collections import OrderedDict

import pandas as pd
from astroquery.mast import Observations
from astropy.io import fits
from astropy.stats import sigma_clip
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
from transitleastsquares import transitleastsquares
from transitleastsquares import transit_mask, cleaned_array
import os
import argparse
import time


start_time = time.time()



def create_summary_log(log_file):
    # os.makedirs('logs', exist_ok=True)
    log_file_path = os.path.join('logs/', log_file)

    output_colnames = [
        'star_id',
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


def append_summary_log(log_file, star_id, planet_id, project, results):
    pdf = pd.DataFrame(
        OrderedDict(
            [
                ('star_id', [star_id]),
                ('planet_id', [planet_id]),
                ('project_name', [project]),
                ('period', [results.period]),
                ('transit_depth', [results.depth]),
                ('transit_midpoint', [np.mean(results.transit_times)]),
                ('duration', [results.duration]),
                ('SDE', [results.SDE]),
            ]
        ),
        index=None
    )
    pdf.to_csv(log_file, header=False, index=False)
    log_file.flush()


def log_results(star_id, planet_id, project, results):
    if planet_id:
        star_id = '_'.join([star_id, str(planet_id)])

    # os.makedirs('logs', exist_ok=True)
    today = date.today()
    today = today.strftime("%Y%m%d")
    file_name = os.path.join('logs/', '_'.join([star_id, project, today]) + '.log')
    with open(file_name, 'w') as f:
        f.write('Star ID: {}\n'.format(star_id))
        if planet_id:
            f.write('Planet ID: {}\n'.format(planet_id))
        f.write('Project Name: {}\n'.format(project))
        f.write('Period (d): {}\n'.format(results.period))
        f.write('Transit Times: {}\n'.format(results.transit_times))
        f.write('Transit Depth: {}\n'.format(results.depth))
        f.write('Duration: {}\n'.format(results.duration))
        f.write('SDE: {}\n'.format(results.SDE))


def get_file(star, project):
    target_name = '*' + star.split()[1]
    # Make sure that there are data for the criteria
    obs_count = Observations.query_criteria_count(
        obs_collection="K2",
        dataproduct_type=["timeseries"],
        instrument_name="Kepler",
        objectname=star,
        target_name=target_name,
        project=project
    )
    if obs_count == 0:
        raise RuntimeError("No data found in archive.")

    obs_table = Observations.query_criteria(
        obs_collection="K2",
        dataproduct_type=["timeseries"],
        instrument_name="Kepler",
        objectname=star,
        target_name=target_name,
        project=project
    )

    data_products = Observations.get_product_list(obs_table)
    lc_mask = ["lightcurve" in x or "light curve" in x for x in map(str.lower, data_products['description'])]
    if not any(lc_mask):
        raise RuntimeError("Retrieved data products do not contain light curve data.")

    data_products = data_products[lc_mask]  # keep only rows with light curves
    manifest = Observations.download_products(data_products)
    files = list(manifest['Local Path'])  # get local file names

    # sort results:
    idx = np.argsort(files)
    sorted_files = [files[i] for i in idx]

    return sorted_files, data_products[idx], obs_table[idx]


def detrend_lc(data,  median_kernel_size=25):
    proc_data = signal.medfilt(data, kernel_size=median_kernel_size)
    detrend_data = data / proc_data
    detrend_data = sigma_clip(detrend_data, sigma_upper=2, sigma_lower=float('inf'))
    return detrend_data


def bits_to_mask(bits):
    return sum(1 << (bit - 1) for bit in bits)


def get_lc(lc_data, project, use_quality=True):
    if isinstance(lc_data, str):
        lc_data = fits.getdata(lc_data, ext=1)

    if project == "k2":
        raw_flux = lc_data['SAP_FLUX']
        cor_flux = lc_data['PDCSAP_FLUX']
        quality = lc_data['SAP_QUALITY']
        time = lc_data['TIME']
        dq_bits = list(range(1, 22))  # [1, 2, 3, 4, 5, 6, 8, 10, 12]

    elif project == "hlsp_everest":
        raw_flux = lc_data['FRAW']
        cor_flux = lc_data['FLUX']
        quality = lc_data['QUALITY']
        time = lc_data['TIME']
        dq_bits = list(range(1, 22)) + [23, 24, 25]  # [1, 2, 3, 4, 5, 6, 8, 10, 12, 13, 14, 20, 23, 24, 25]

    elif project == "hlsp_k2sff":
        raw_flux = lc_data['FRAW']
        cor_flux = lc_data['FCOR']
        # quality = lc_data['QUALITY']
        time = lc_data['T']
        dq_bits = [1, 2, 3, 4, 5, 6, 8, 10, 12]

    else:
        raise ValueError("Unknown project.")

    if (project == "hlsp_everest" or project == "k2") and use_quality:
        dq_mask = bits_to_mask(dq_bits)
        mask = np.logical_not(np.bitwise_and(quality, dq_mask))  # good data mask
        raw_flux = raw_flux[mask]
        cor_flux = cor_flux[mask]
        time = time[mask]

    mask = np.isfinite(time) & np.isfinite(raw_flux) & np.isfinite(cor_flux)

    return time[mask], raw_flux[mask], cor_flux[mask]


def plot_lc(t, raw, cor, detr, star, planet_id, project, transit_times, plotno=None):
    if planet_id:
        star = '_'.join([star, str(planet_id)])

    fig = plt.figure(plotno, figsize=(11, 5))

    fig.add_subplot(311)
    plt.title(project, loc = 'right')
    plt.plot(t, raw, 'r.', label='RAW')
    plt.legend(loc='best')
    plt.title('Lightcurves for ' + star)

    fig.add_subplot(312)
    plt.plot(t, cor, 'b.', label='COR')
    plt.legend(loc='best')

    fig.add_subplot(313)
    plt.plot(t, detr, 'g.', label='DETREND')
    for time in transit_times:
        plt.axvline(x=time, color = 'black', ls = '--')
    plt.legend(loc='best')
    plt.xlabel('TIME (BTJD)')

    plt.tight_layout()

    plt.savefig(os.path.join('Graphs/lc/', '_'.join([star, project, '_lc.png'])))
    plt.close(fig)


def calc_stats():
    return 1


def calc_tls(t, flux, star):
    print("\n\nt: {}\nflux: {}\nstar: {}\n\n".format(t, flux, star))
    model = transitleastsquares(t, flux)
    results = model.power()
    if not np.isfinite(results.period):
        print("INVALID RESULTS")
        return results, False
    print("\n\nmodel:\n{}\n\n".format(model))
    print("\n\nRESULTS:\n{}\n\n".format(results))
    print('Period', format(results.period, '.5f'), 'd')
    print(len(results.transit_times), 'transit times in time series:',
          ['{0:0.5f}'.format(i) for i in results.transit_times])
    print('Transit depth', format(results.depth, '.5f'))
    print('Best duration (days)', format(results.duration, '.5f'))
    print('Signal detection efficiency (SDE):', results.SDE)
    return results, True


def tls_plot(star, planet_id, project, plotno, results):
    if planet_id:
        star = '_'.join([star, str(planet_id)])
    fig = plt.figure(plotno, figsize=(9, 7))
    fig.add_subplot(313)
    plt.title('SDE vs. Period for ' + star)
    ax = plt.gca()
    ax.axvline(results.period, alpha=0.4, lw=3)
    plt.xlim(np.min(results.periods), np.max(results.periods))
    for n in range(2, 10):
        ax.axvline(n * results.period, alpha=0.4, lw=1, linestyle="dashed")
        ax.axvline(results.period / n, alpha=0.4, lw=1, linestyle="dashed")
    plt.ylabel('SDE')
    plt.xlabel('Period (days)')
    plt.plot(results.periods, results.power, color='black', lw=0.5)
    plt.xlim(0, max(results.periods))

    fig.add_subplot(312)
    plt.title('Folded Transit Curve for ' + star)
    plt.plot(results.model_folded_phase, results.model_folded_model, color='red')
    plt.scatter(results.folded_phase, results.folded_y, color='blue', s=10, alpha=0.5, zorder=2)
    plt.xlim(0.5-(1.5*(results.duration/results.period)), 0.5+(1.5*(results.duration/results.period)))
    plt.xlabel('Phase')
    plt.ylabel('Relative flux');

    fig.add_subplot(311)
    plt.title(project, loc='right')
    plt.title('Folded Transit Curve for ' + star)
    plt.plot(results.model_folded_phase, results.model_folded_model, color='red')
    plt.scatter(results.folded_phase, results.folded_y, color='blue', s=10, alpha=0.5, zorder=2)
    plt.xlim(0.0, 1)
    plt.xlabel('Phase')
    plt.ylabel('Relative flux')
    plt.tight_layout()
    plt.savefig(os.path.join('Graphs/tls/', '_'.join([star, project, '_tls.png'])))
    plt.close(fig)
    return results


def find_planet(time, raw_flux, cor_flux, star_id, planet_id, project, plotno,
                summary_log):
    fcor_flux = detrend_lc(cor_flux)
    if hasattr(fcor_flux, 'mask'):
        mask = fcor_flux.mask
    else:
        mask = np.zeros(len(fcor_flux), dtype=np.bool)
    fcor_flux = fcor_flux[~mask].data
    time = time[~mask]
    raw_flux = raw_flux[~mask]

    results, valid = calc_tls(time, fcor_flux, star_id)
    plot_lc(time, raw_flux, cor_flux[~mask], fcor_flux, star_id, planet_id,
            project, transit_times=results.transit_times)
    plotno += 1

    if valid:
        tls_plot(star_id, planet_id, project, plotno, results)
        plotno += 1

        log_results(star_id, planet_id, project, results)
        append_summary_log(summary_log, star_id, planet_id, project, results)

    return results, valid, plotno


def find_planets(time, raw_flux, cor_flux, max_planets, star_id, project,
                 plotno, summary_log):
    valid = True
    planet_id = 1
    while valid and ((max_planets and planet_id <= max_planets) or max_planets is None):
        results, valid, plotno = find_planet(time, raw_flux, cor_flux, star_id,
                                             'pl{:d}'.format(planet_id),
                                             project, plotno, summary_log)

        intransit = transit_mask(time, results.period, 2 * results.duration, results.T0)

        raw_flux = raw_flux[~intransit]
        cor_flux = cor_flux[~intransit]
        time = time[~intransit]
        time, cor_flux = cleaned_array(time, cor_flux)
        planet_id += 1


def find_planets_around_stars(stars='k2names.csv',
                              top_nstars = None,
                              max_planets=None,
                              log_file='summary_log.csv',
                              projects=("hlsp_everest", "hlsp_k2sff", "k2")):
    os.makedirs('logs', exist_ok=True)
    os.makedirs('Graphs', exist_ok=True)
    os.makedirs('Graphs/tls', exist_ok=True)
    os.makedirs('Graphs/lc', exist_ok=True)

    if isinstance(stars, str):
        stars_df = pd.read_csv(stars)
        if 'epic_name' not in stars_df.columns.values.tolist():
            raise ValueError("Input list of stars does not have an 'epic_name'"
                             "column.")
        catalog = stars_df['epic_name'].unique().tolist()

    summary_log = create_summary_log(log_file)

    plotno = 1

    if top_nstars is not None:
        top_nstars += 1

    for star_id in catalog[:top_nstars]:
        for project in projects:
            try:
                data_files, data_prod, _ = get_file(star_id, project)
                print("THIS LIGHT CURVE USES PROJECT: " + project)
            except Exception:
                print("There was an issue retrieving the files for " + star_id + " in the " + project + " data set.")
                continue

            try:
                time, raw_flux, cor_flux = get_lc(data_files[0], project)
                print("TIME: ", type(time), time.shape)
                print("raw_flux: ", type(raw_flux), raw_flux.shape)
                print("cor_flux: ", type(cor_flux), cor_flux.shape)
                if time.size < 10:
                    print("WARNING: Too few data to find transit.")
                    continue
            except ValueError:
                continue

            find_planets(
                time, raw_flux, cor_flux, max_planets, star_id, project, plotno,
                summary_log
            )

    summary_log.close()

if __name__ == "__main__":
    # Parse input parameters:
    parser = argparse.ArgumentParser(
        description='Find multiple planets around each star in the input list.'
    )

    parser.add_argument(
        '-f', '--file', default='k2names.csv', action='store',
        help='File name of a list of star names.'
    )

    parser.add_argument(
        '-t', '--top', default=None, action='store', type=int,
        help='The number of stars from the top of the input list to process.'
    )

    parser.add_argument(
        '-l', '--log', default='summary_log.csv', action='store',
        help='File name of the log file containing results.'
    )

    parser.add_argument(
        '-p', '--projects', default="hlsp_everest,hlsp_k2sff,k2", action='store',
        help='Find planets using data from the specified project.'
    )

    parser.add_argument(
        '-m', '--max_planets', default=4, action='store', type=int,
        help='Maximum number of planets to find for each star.'
    )

    options = parser.parse_args()

    file_name = options.file
    top_star = options.top
    log_file = options.log
    max_planets = options.max_planets
    projects = tuple(p for p in options.projects.split(',') if p)

    find_planets_around_stars(
        stars=file_name, top_nstars=20, max_planets=max_planets,
        log_file=log_file, projects=projects
    )


print("--- %s seconds ---" % (time.time() - start_time))