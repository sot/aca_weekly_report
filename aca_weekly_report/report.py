# coding: utf-8
# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""
Make a report of performance during recent observations.
"""
import os
import argparse
import numpy as np
from jinja2 import Template
from pathlib import Path
import pickle
import gzip
from astropy.table import Table, join, vstack
from collections import defaultdict
import logging

import Ska.DBI
from Chandra.Time import DateTime
import kadi.paths
from kadi import events
import kadi.commands
from Ska.engarchive import fetch_sci
from Ska.engarchive.fetch import get_time_range
from Ska.engarchive.utils import logical_intervals
import mica.starcheck
import mica.stats.acq_stats
import mica.stats.guide_stats
from mica.utils import load_name_to_mp_dir
from proseco.acq import get_p_man_err

SKA = os.environ['SKA']
MP_STARCATS = None
ACQ_STATS = None
GUIDE_STATS = None
HI_BGD = None
MP_DIR = Path(SKA) / 'data' / 'mpcrit1' / 'mplogs'


def get_options():
    parser = argparse.ArgumentParser(description="Run weekly report making")
    parser.add_argument("--out",
                        default=".",
                        help="output directory, default is '.'")
    parser.add_argument("--start",
                        help="start time for search for manvrs for report")
    parser.add_argument("--stop",
                        help="stop time for search for manvrs for report")
    parser.add_argument("--days-back",
                        default=10,
                        type=float,
                        help="number of days back from 'now' for standard report (default 10)")
    opt = parser.parse_args()
    return opt


def get_all_starcats():
    """
    Get a table of the times all commanded star catalogs and the source mp_dir for the mission
    from the beginning of the kadi cmds and cmd_states tables.

    :returns: astropy Table
    """
    starcats = kadi.commands.get_cmds(type="MP_STARCAT")

    # In case we ever include manually-commanded star catalogs, filter those out.
    starcats = starcats[starcats["source"] != "CMD_EVT"]

    # Translate from e.g. DEC2506C to /2006/DEC2506/oflsc/.
    mp_dirs = [load_name_to_mp_dir(sc["source"]) for sc in starcats]
    mp_starcats = Table([starcats["date"], mp_dirs], names=["date", "mp_dir"])

    return mp_starcats


def man_ok(one_shot, p_man_err):
    return p_man_err > 0.025 or one_shot < 60


def get_proseco_catalog(manvr):
    """
    Find and load (from the mission planning pkl) the proseco star catalog commanded
    before the end of the supplied manvr.

    :param manvr: kadi manvr event
    :returns: proseco star ACACatalogTable
    """
    mp_starcat = MP_STARCATS[MP_STARCATS['date'] <= manvr.stop][-1]
    pfiles = sorted((MP_DIR / mp_starcat['mp_dir'][1:] / 'output').glob('*proseco.pkl.gz'))
    if len(pfiles) == 1:
        acas = pickle.load(gzip.open(pfiles[0], 'rb'))
        times = [DateTime(acas[obsid].meta['date']).secs for obsid in acas]
        obsids = [obsid for obsid in acas]
        ptable = Table([times, obsids], names=['times', 'obsid'])
        ptable.sort('times')

        # So far, only obsid 47188 in recovery shows a manvr time after the catalog time
        # but I've added a minute of slop/padding to the operation to grab the catalog.
        obsid = int(ptable[ptable['times'] < (DateTime(manvr.stop).secs + 60)][-1]['obsid'])
        pcat = acas[obsid]
    else:
        raise ValueError("No proseco catalog available.")
    return pcat


def get_proseco_data(pcat):
    """
    Extra proseco predictions about guide and acq success and the expected temperatures.

    :param pcat: proseco ACACatalogTable
    :returns: dictionary with p2, guide_count, and the predicted t_ccd_acq and t_ccd_guide
    """
    if hasattr(pcat, 'acqs'):
        p2 = -np.log10(pcat.acqs.calc_p_safe())
        pcar = pcat.get_review_table()
        pcar.run_aca_review()
        guide_count = pcar.guide_count
        pdata = {'pred_t_ccd_acq': pcat.call_args['t_ccd_acq'],
                 'pred_t_ccd_guide': pcat.call_args['t_ccd_guide'],
                 'p2': p2,
                 'guide count': guide_count}
    else:
        pdata = {'pred_t_ccd_acq': np.nan,
                 'pred_t_ccd_guide': np.nan,
                 'p2': np.nan,
                 'guide count': np.nan}
    return pdata


def get_fid_data(pcat):
    """
    Fetch the fid tracking metrics from the Sybase track stats table and combine with the
    proseco fid catalog information (which has a spoiler_score / prediction of amount of spoiling).

    :param pcat: proseco ACACatalogTable
    :returns: Table with fid data including expected and observed values
    """
    if len(pcat.fids) == 0:
        return []
    with Ska.DBI.DBI(dbi='sybase', server='sybase', user='aca_read') as db:
        fids = db.fetchall(
            f"select * from trak_stats_data where obsid = {pcat.obsid} and type = 'FID'")
    if len(fids) > 0:
        fids = Table(fids)
        fids['f_track'] = (fids['n_samples'] - fids['not_tracking_samples']) / fids['n_samples']
        fids = fids['slot', 'f_track']
        fids = join(pcat.fids, fids, keys=['slot'])
        for col in ['f_track', 'yang', 'zang']:
            fids[col].format = '.2f'
        return fids['slot', 'id', 'yang', 'zang', 'spoiler_score', 'f_track']
    else:
        return []


def get_acq_data(pcat):
    """
    Fetch the acq success metrics from mica acq stats table and combine with the
    proseco acq catalog information (which has p_acq).

    :param pcat: proseco ACACatalogTable
    :returns: Table with acq data including expected (proseco) and observed values
    """
    obs_acq_stats = Table(ACQ_STATS[ACQ_STATS['obsid'] == pcat.obsid])[
        'agasc_id', 'acqid', 'mag_obs', 'dy', 'dz', 'cdy', 'cdz', 'sat_pix', 'ion_rad']
    obs_acq_stats.rename_column('agasc_id', 'id')
    if len(obs_acq_stats) == 0:
        return []
    acqs = pcat.acqs.copy()
    acqs = acqs['id', 'slot', 'yang', 'zang', 'mag', 'halfw', 'p_acq']
    acqs = join(acqs, obs_acq_stats)
    for col in ['yang', 'zang', 'mag', 'dy', 'dz', 'cdy', 'cdz', 'mag_obs']:
        acqs[col].format = '.1f'
    acqs['p_acq'].format = '.2f'
    return acqs


def get_acq_anoms(acqs):
    for acq in acqs:
        if ((abs(acq['dy']) > acq['halfw'] + 20) or
                (abs(acq['dz']) > acq['halfw'] + 20)):
            return True


def get_guide_data(pcat):
    """
    Fetch the guide success metrics from mica guide stats table and combine with the
    proseco guide catalog information.

    :param pcat: proseco ACACatalogTable
    :returns: Table with guide data including expected (proseco) and observed values
    """
    guides = pcat.guides.copy()
    guides = guides['id', 'slot', 'yang', 'zang', 'mag']
    global GUIDE_STATS
    obs_gui_stats = Table(GUIDE_STATS[GUIDE_STATS['obsid'] == pcat.obsid])[
        'agasc_id', 'type', 'f_track', 'f_within_3', 'f_within_5', 'dy_mean', 'dz_mean']
    obs_gui_stats.rename_column('agasc_id', 'id')
    if len(obs_gui_stats) == 0:
        return []
    guides = join(guides, obs_gui_stats)
    for col in ['f_track', 'f_within_3', 'f_within_5', 'dy_mean', 'dz_mean']:
        guides[col].format = '.2f'
    for col in ['zang', 'yang', 'mag']:
        guides[col].format = '.1f'
    return guides


def get_max_tccd(start, stop):
    """
    Get the max AACCCDPT over the interval.

    :param start: start time of interval
    :param stop: stop time of interval
    :returns: max value from fetch or np.nan if data not available
    """
    if stop is None or DateTime(stop).secs > get_time_range('AACCCDPT')[1]:
        return np.nan
    else:
        return np.max(fetch_sci.Msid('AACCCDPT', start, stop).vals)


def get_manvr_data(manvr):
    """
    Calculate likelihood of one shot from proseco's get_p_man_err and include that, the manvr
    angles before and after, and the one shots before and after in a dictionary for use in
    the report.

    :param manvr: kadi manvr to the dwell being reported upon
    :returns: dictionary with maneuver observed quantities and probability of one shot size
    """
    p_man_err_before = get_p_man_err(manvr.one_shot,
                                     manvr.angle)
    if manvr.get_next() is False:
        one_shot_after = -1
        man_angle_after = -1
        p_man_err_after = 0.0
    else:
        one_shot_after = manvr.get_next().one_shot
        man_angle_after = manvr.get_next().angle
        p_man_err_after = get_p_man_err(one_shot_after,
                                        man_angle_after)
    return {'man angle before': manvr.angle,
            'one shot before': manvr.one_shot,
            'p man err before': p_man_err_before,
            'one shot after': one_shot_after,
            'man angle after': man_angle_after,
            'p man err after': p_man_err_after}


def kalman_ok(kalman_data):
    return kalman_data['consec_lt_two'] <= 27.0


def get_kalman_data(manvr):
    """
    Duplicate the kalman_watch calculation to look for intervals of low AOKALSTR.
    This searches over the dwell or dwells associated with the manvr until the next
    manvr start.

    :param manvr: kadi manvr
    :returns: dictionary with longest low kalman interval and lowest number of kal stars
    """
    kalman_data = {'consec_lt_two': 0,
                   'min_kalstr': np.nan}
    trange = get_time_range('AOKALSTR')
    if manvr.get_next() is False or trange[1] < DateTime(manvr.get_next().start).secs:
        return kalman_data
    dat = fetch_sci.Msidset(['AOKALSTR', 'AOPCADMD', 'AOACASEQ'],
                            manvr.guide_start,
                            manvr.get_next().start)
    dat.interpolate(1.025)
    ok = (dat['AOACASEQ'].vals == 'KALM') & (dat['AOPCADMD'].vals == 'NPNT')
    kalman_data['min_kalstr'] = np.min(dat['AOKALSTR'].vals[ok].astype(int))
    low_kal = (dat['AOKALSTR'].vals.astype(int) <= 1) & ok
    if np.any(low_kal):
        intervals = logical_intervals(dat['AOKALSTR'].times,
                                      low_kal,
                                      max_gap=10)
        kalman_data['consec_lt_two'] = np.max(intervals['duration'])
    return kalman_data


def is_cdy_acq_anom(acq):
    return abs(acq['cdy']) > (acq['halfw'] + 10)


def is_cdz_acq_anom(acq):
    return abs(acq['cdz']) > (acq['halfw'] + 10)


def is_p_acq_anom(acq):
    return ((acq['p_acq'] > 0.95) and (not acq['acqid']))


def is_guide_poorly_tracked(guide):
    return guide['f_within_3'] < 0.95


def is_acquired(acq):
    return acq['acqid']


def is_fid_poorly_tracked(fid):
    return fid['f_track'] < 0.95


def get_n_acq(acqs):
    if len(acqs) == 0:
        return -1
    else:
        return np.count_nonzero(np.array([is_acquired(acq) for acq in acqs]))


def get_n_good_guides(guides):
    if len(guides) == 0:
        return -1
    else:
        return np.count_nonzero(np.array([not is_guide_poorly_tracked(guide)
                                          for guide in guides]))


def get_n_bad_fids(fids):
    if len(fids) == 0:
        return -1
    else:
        return np.count_nonzero(np.array([is_fid_poorly_tracked(fid) for fid in fids]))


def check_cat_data(cat, warn_funcs, warn_cols):
    """
    Create string-ified version of the table for use in the detailed report.
    Run row-based checks on a submitted acq, guide, or fid table.
    Make a formatting/status table based on the checks.

    There is some duplication with the code to make the top-level formatted table.

    :param cat: astropy table of fid, guide, or acq data
    :param warn_funcs: array of functions to be run as tests on the rows of the table
    :param warn_cols: columns to mark as "red" on failure of the function/tests in
                      warn_funcs.  warn_funcs and warn_cols should be associated 1-to-1.

    :returns: string-ified catalog table, markup/formatting table, boolean status on any fail
    """
    if len(cat) == 0:
        return Table(), Table(), False
    print_cat = Table(np.zeros((len(cat), len(cat.colnames))).astype(str),
                      names=cat.colnames)

    # Set up table string formatting
    for name in cat.colnames:
        for row_idx in range(len(cat)):
            if cat[name].format is not None:
                print_cat[name][row_idx] = cat[name].pformat(show_name=False)[row_idx]
            else:
                print_cat[name][row_idx] = f'{cat[name][row_idx]}'

    # Substitute in links to kadi for the ids for stars (which I think will be greater
    # than 20)
    id_strs = []
    for row in cat:
        if row['id'] > 20:
            id_strs.append(
                f'<A HREF="https://kadi.cfa.harvard.edu/star_hist/?agasc_id={row["id"]}">'
                + f'{row["id"]}</A>')
        else:
            id_strs.append(str(row['id']))
    print_cat['id'] = id_strs

    # Make a status dictionary (just bad status for now).
    has_warn = False
    status = defaultdict(list)
    for row_idx, row in enumerate(cat):
        for func, name in zip(warn_funcs, warn_cols):
            if func(row):
                status[name].append(row_idx)
                has_warn = True

    # Save status and CSS formatting in a table
    td_class = Table()
    for name in cat.colnames:
        td_classes_list = []
        for ii in range(len(cat)):
            td_classes = []
            if name in status and ii in status[name]:
                td_classes.append('bad')
            if cat[name].dtype.kind in ('i', 'f'):
                td_classes.append('align-right')
            td_classes_list.append(td_classes)
        td_class[name] = [' '.join(x) for x in td_classes_list]

    return print_cat, td_class, has_warn


def get_and_check_cats(pcat):
    """
    Fetch the catalogs and observed data.
    Run the code to make formatted and marked up tables from that data.
    Return top level status and all the individual tables in dictionaries.

    :param pcat: proseco ACACatalogTable
    :returns: dict of top level catalog status items, dict of all the catalogs and markup
    """
    cat_data = {}
    fid_data = get_fid_data(pcat)
    cats = {}
    cat_data['n bad fid'] = get_n_bad_fids(fid_data)
    fid_print_data, fid_tds, fid_warn = check_cat_data(
        fid_data, [is_fid_poorly_tracked], ['f_track'])
    cats['fid'] = {'table': fid_print_data,
                   'markup': fid_tds,
                   'name': 'FID'}
    cat_data['fid_warn'] = fid_warn

    guide_data = get_guide_data(pcat)
    cat_data['n good guide'] = get_n_good_guides(guide_data)
    guide_print_data, guide_tds, guide_warn = check_cat_data(
        guide_data,
        [is_guide_poorly_tracked], ['f_within_3'])
    cats['guide'] = {'table': guide_print_data,
                     'markup': guide_tds,
                     'name': 'GUIDE'}
    cat_data['guide_warn'] = guide_warn

    acq_data = get_acq_data(pcat)
    cat_data['n_acq'] = get_n_acq(acq_data)
    acq_print_data, acq_tds, acq_warn = check_cat_data(
        acq_data,
        [is_cdy_acq_anom, is_cdz_acq_anom, is_p_acq_anom],
        ['cdy', 'cdz', 'acqid'])
    cats['acq'] = {'table': acq_print_data,
                   'markup': acq_tds,
                   'name': 'ACQ'}
    cat_data['acq_warn'] = acq_warn
    return cat_data, cats


def t_ccd_ok(dat):
    return dat['t_ccd'] < (dat['pred_t_ccd_guide'] + 3.0)


def get_obsmetrics(manvr):
    """
    For the given maneuver, get the associated catalogs, the observed quantities at acquisition
    and over the dwell, and return dictionaries describing everything.

    :param manvr: kadi manvr
    :returns: dict of all top-level obs/manvr metrics,
              dict of the proseco catalogs and observed quantities on each row,
              dict associating potential warnings with links for this obs/manvr
    """
    metric = {'obsid': manvr.obsid,
              'start': manvr.acq_start}
    if manvr.next_nman_start is not False:
        metric['t_ccd'] = get_max_tccd(manvr.acq_start, manvr.next_nman_start)
    else:
        metric['t_ccd'] = np.nan

    pcat = get_proseco_catalog(manvr)
    obsid = int(pcat.obsid)
    proseco_data = get_proseco_data(pcat)
    manvr_data = get_manvr_data(manvr)
    kal_data = get_kalman_data(manvr)
    strobs = f'{obsid:05d}'
    cat_data, cats = get_and_check_cats(pcat)
    for dat in (manvr_data, proseco_data, kal_data, cat_data):
        metric.update(dat)

    links = {'detail_url': f'./obs_{metric["obsid"]}_{metric["start"]}.html',
             'dash':
             f'https://icxc.cfa.harvard.edu/aspect/centroid_reports/{strobs[0:2]}/{strobs}/',
             'mica':
             f'https://icxc.cfa.harvard.edu/aspect/mica_reports/{strobs[0:2]}/{strobs}/'}

    metric.update(links)
    warn_map = {
        'KAL': 'http://cxc.cfa.harvard.edu/mta/ASPECT/kalman_watch/',
        'ANOM': metric['mica'],
        'HI_BGD':
        f'https://cxc.cfa.harvard.edu/mta/ASPECT/aca_hi_bgd_mon/events/obs_{strobs}/index.html',
        'MANVR': metric['dash'],
        'ACQ': metric['detail_url'],
        'GUIDE': metric['detail_url'],
        'FID': metric['detail_url'],
        'T_CCD': metric['mica'],
    }

    return metric, cats, warn_map


def make_metric_print(dat, warn_map):
    """
    For the data/dictionary for a given obs/manvr, run some checks, and get together the pieces
    for a formatted table.

    :param dat: dictionary with just about eveyrthing for the obs/manvr
    :param warn_map: dictionary with mapping concerns/warning to hopefully relevant links
    :returns: astropy Table of string-ified content, Table of markup/formatting
    """
    # Make a status dictionary (just bad status for now).
    status = {}
    end_warns = []
    for ctype in ['FID', 'GUIDE', 'ACQ']:
        if dat[f'{ctype.lower()}_warn']:
            end_warns.append((ctype, warn_map[ctype]))
    if not np.isnan(dat['t_ccd']) and not t_ccd_ok(dat):
        end_warns.append(('T_CCD', warn_map['T_CCD']))
        status['t_ccd'] = True
    if not man_ok(dat['one shot before'], dat['p man err before']):
        end_warns.append(('MANVR', warn_map['MANVR']))
        status['one shot before'] = True
    if not man_ok(dat['one shot after'], dat['p man err after']):
        status['one shot after'] = True
    if not kalman_ok(dat):
        end_warns.append(('KAL', warn_map['KAL']))
    if dat['obsid'] in HI_BGD['obsid']:
        end_warns.append(('HI_BGD', warn_map['HI_BGD']))
    href_warns = []
    href_warns = ','.join([f"<A HREF='{w[1]}'>{w[0]}</A>" for w in end_warns])

    print_cols = ['t_ccd', 'man angle before', 'man angle after',
                  'one shot before', 'one shot after',
                  'p man err before', 'p man err after',
                  'p2', 'n_acq',
                  'guide count', 'n good guide']
    formats = {'t_ccd': '.2f', 'p man err before': '.2f', 'p man err after': '.2f',
               'man angle before': '.0f', 'man angle after': '.0f',
               'one shot before': '.0f',
               'one shot after': '.0f', 'p2': '.2f',
               'guide count': '.1f'}

    print_table = {}
    # Add some long URL fields to the dictionary before making an astropy.table
    print_table['obsid'] = (
        f"<A HREF='{dat['detail_url']}'>{dat['obsid']}</A>")
    print_table['mica'] = (
        f"<A HREF='{dat['mica']}'>mica</A>")
    print_table['dash'] = (
        f"<A HREF='{dat['dash']}'>dash</A>")
    print_table['start'] = dat['start']

    # Add the rest
    for col in print_cols:
        print_table[col] = dat[col]

    print_table = Table([print_table])
    print_table['warns'] = href_warns

    # Save status and CSS formatting in a table
    td_class = {}
    for name in print_cols:
        td_classes = []
        if name in status and status[name]:
            td_classes.append('bad')
        if name in print_table.colnames and print_table[name].dtype.kind in ('i', 'f'):
            td_classes.append('align-right')
        td_class[name] = ' '.join(td_classes)

    # Update the formats of the columns to be more stringy
    for col in print_cols:
        if col in formats:
            print_table[col] = f'{dat[col]:{formats[col]}}'
        else:
            print_table[col] = str(dat[col])

    # Put in order
    print_table = print_table[['obsid', 'start', 'dash', 'mica']
                              + print_cols + ['warns']]
    return print_table, td_class


def main():
    """
    Run data fetching over a time interval and make aca_weekly_reports for the obs/manvrs in
    in the interval.
    """
    opt = get_options()
    outdir = opt.out
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    stop = DateTime(opt.stop)
    if opt.start is None:
        start = DateTime() - opt.days_back
    else:
        start = DateTime(opt.start)

    # these globals are cop-outs but ...
    global ACQ_STATS
    ACQ_STATS = mica.stats.acq_stats.get_stats()
    global GUIDE_STATS
    GUIDE_STATS = mica.stats.guide_stats.get_stats()
    global HI_BGD
    HI_BGD = Table.read(Path(SKA) / 'data' / 'aca_hi_bgd_mon' / 'bgd_events.dat',
                        format='ascii')
    global MP_STARCATS
    MP_STARCATS = get_all_starcats()

    manvrs = events.manvrs.filter(start=start, stop=stop)

    file_dir = Path(__file__).parent
    detail_template = Template(open(
        file_dir / "detail_template.html", 'r').read())

    logger = logging.getLogger('aca_weekly_report')
    metrics = []
    obss = []
    metric_rows = []
    markups = []
    for manvr in manvrs:
        if manvr.acq_start is None:
            logger.info(f"Skip {manvr.obsid} has no acquisition")
            continue
        logger.debug(manvr.start, manvr.stop, manvr.obsid)
        try:
            metric, obs_cats, warn_map = get_obsmetrics(manvr)
            metric_row, markup = make_metric_print(metric, warn_map)
            obss.append(obs_cats)
            metrics.append(metric)
            metric_rows.append(metric_row)
            markups.append(markup)
        except Exception:
            logger.warn(f"Skip {manvr.obsid} at {manvr.start}.  Error processing")

    markups = Table(markups)
    metric_print = vstack(metric_rows)

    for i, row in enumerate(metric_print):
        obs_page = detail_template.render(
            obs=row,
            obs_cats=obss[i])
        f = open(
            os.path.join(outdir,
                         f'obs_{metrics[i]["obsid"]}_{metrics[i]["start"]}.html'), "w")
        f.write(obs_page)
        f.close()

    for col in ['one shot after', 'man angle after', 'p man err after',
                'man angle before', 'p man err before']:
        metric_print.remove_column(col)

    template = Template(open(file_dir / "top_level_template.html", 'r').read())
    page = template.render(metrics=metric_print, markup=markups)
    f = open(os.path.join(outdir, "index.html"), "w")
    f.write(page)
    f.close()


if __name__ == '__main__':
    main()
