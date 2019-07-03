from astroquery.mast import Observations
from astropy.io import fits
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
import statistics
from datetime import date
from transitleastsquares import transitleastsquares


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

    return files, data_products, obs_table


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

    return time, raw_flux, cor_flux


def plot_lc(t, raw, cor, detr, star):
    fig = plt.figure(figsize=(11, 5))

    fig.add_subplot(311)
    plt.plot(t, raw, 'r.', label='RAW')
    plt.legend(loc='best')
    plt.title('Lightcurves for ' + star)

    fig.add_subplot(312)
    plt.plot(t, cor, 'b.', label='COR')
    plt.legend(loc='best')

    fig.add_subplot(313)
    plt.plot(t, detr, 'g.', label='DETREND')
    plt.legend(loc='best')
    plt.xlabel('TIME (BTJD)')

    plt.tight_layout()

    plt.show()


def detrend_lc(data, median_kernel_size=25, not_everest=True):
    proc_data = signal.medfilt(data, kernel_size=median_kernel_size)
    if not_everest:
        detrend_data = (data / proc_data)
    else:
        detrend_data = (data / proc_data) + statistics.median(data)
    return detrend_data


def calc_stats():
    return 1


def tls_plot(t, flux, star):
    model = transitleastsquares(t, flux)
    results = model.power()
    print('Period', format(results.period, '.5f'), 'd')
    print(len(results.transit_times), 'transit times in time series:', \
          ['{0:0.5f}'.format(i) for i in results.transit_times])
    print('Transit depth', format(results.depth, '.5f'))
    print('Best duration (days)', format(results.duration, '.5f'))
    print('Signal detection efficiency (SDE):', results.SDE)

    fig = plt.figure(figsize=(9, 7))
    fig.add_subplot(313)
    plt.title('SDE vs. Period for ' + star)
    ax = plt.gca()
    ax.axvline(results.period, alpha=0.4, lw=3)
    plt.xlim(np.min(results.periods), np.max(results.periods))
    for n in range(2, 10):
        ax.axvline(n * results.period, alpha=0.4, lw=1, linestyle="dashed")
        ax.axvline(results.period / n, alpha=0.4, lw=1, linestyle="dashed")
    plt.ylabel(r'SDE')
    plt.xlabel('Period (days)')
    plt.plot(results.periods, results.power, color='black', lw=0.5)
    plt.xlim(0, max(results.periods))

    fig.add_subplot(312)
    plt.title('Folded Transit Curve for ' + star)
    plt.plot(results.model_folded_phase, results.model_folded_model, color='red')
    plt.scatter(results.folded_phase, results.folded_y, color='blue', s=10, alpha=0.5, zorder=2)
    plt.xlim(0.48, 0.52)
    plt.xlabel('Phase')
    plt.ylabel('Relative flux');

    fig.add_subplot(311)
    plt.title('Folded Transit Curve for ' + star)
    plt.plot(results.model_folded_phase, results.model_folded_model, color='red')
    plt.scatter(results.folded_phase, results.folded_y, color='blue', s=10, alpha=0.5, zorder=2)
    plt.xlim(0.0, 1)
    plt.xlabel('Phase')
    plt.ylabel('Relative flux');

    plt.tight_layout()


def log_results(star, project):
    today = date.today()
    print(today.strftime("%Y%m%d"))


with open("/Users/dcara/project/Katoctin/k2name.csv") as f:
    catalog = []
    for line in f.readlines():
        sline = line.strip()
        if sline:
            catalog.append(sline)
    catalog = list(set(catalog))

project_list = ["hlsp_k2sff", "k2"]
for project in project_list:
    for star_id in catalog[0:1]:
        try:
            data_files, data_prod, _ = get_file(star_id, project)
            print("THIS LIGHT CURVE USES PROJECT: " + project)
        except Exception:
            print("There was an issue retrieving the files for " + star_id + " in the " + project + " data set.")
            continue
        try:
            time, raw_flux, cor_flux = get_lc(data_files[0], project)
        except ValueError as e:
            print("{} Project: {}".format(e.args[0], project))
            continue
        if calc_stats() > 0:
            if project == "hlsp_k2sff" or project == "k2":
                fcor_flux = detrend_lc(cor_flux)
            else:
                fcor_flux = detrend_lc(cor_flux, not_everest=False)

            plot_lc(time, raw_flux, cor_flux, fcor_flux, star_id)
            tls_plot(time, fcor_flux, star_id)