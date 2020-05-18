import os
from glob import glob
import numpy as np
from jinja2 import Template
from pathlib import Path
import pickle
import gzip
import logging
from astropy.table import Table, join
from collections import defaultdict
import tables

import Ska.DBI
from Chandra.Time import DateTime
from kadi import events
from Ska.engarchive import fetch_sci
from Ska.engarchive.fetch import get_time_range
from Ska.engarchive.utils import logical_intervals
from chandra_aca.centroid_resid import CentroidResiduals

from mica.common import MICA_ARCHIVE
import mica.starcheck
import mica.stats.acq_stats
import mica.stats.guide_stats
import proseco
import proseco.core
from proseco.acq import get_p_man_err

MP_STARCATS = None
ACQ_STATS = None
GUIDE_STATS = None
HI_BGD = None
MP_DIR = '/data/mpcrit1/mplogs'


def get_all_starcats():
    with tables.open_file('/proj/sot/ska/data/kadi/cmds.h5', 'r') as h5:
        all_cmds = Table(h5.root.data[:])
    kadi_starcats = all_cmds[all_cmds['type'] == 'MP_STARCAT']

    with Ska.DBI.DBI(dbi='sqlite', server='/proj/sot/ska/data/cmd_states/cmd_states.db3') as db:
        timelines = Table(db.fetchall("select * from timeline_loads"))
    timelines.rename_column('id', 'timeline_id')

    kadi_starcats['date'] = kadi_starcats['date'].astype(str)
    mp_starcats = join(kadi_starcats['date', 'timeline_id'],
                       timelines['timeline_id', 'mp_dir'], 
                       keys=['timeline_id'],
                       )['date', 'mp_dir']
    mp_starcats.sort('date')
    return mp_starcats


def man_ok(one_shot, p_man_err):
    return p_man_err > 0.025 or one_shot < 60


def get_proseco_catalog(manvr):
    mp_starcat = MP_STARCATS[MP_STARCATS['date'] <= manvr.stop][-1]
    pfiles = glob(os.path.join(MP_DIR, mp_starcat['mp_dir'][1:], 'output', '*proseco.pkl.gz'))
    if len(pfiles) == 1:
        acas = pickle.load(gzip.open(pfiles[0], 'rb'))
        dates =[acas[obsid].meta['date'] for obsid in acas]
        obsids = [obsid for obsid in acas]
        ptable = Table([dates, obsids], names=['dates', 'obsid'])
        ptable.sort('dates')
        obsid = int(ptable[ptable['dates'] < manvr.stop][-1]['obsid'])
        pcat = acas[obsid]
    else:
        raise ValueError("No proseco catalog available.")
        #obsid, pcat = fake_pcat(manvr.obsid, mp_starcat['mp_dir'])
    return pcat


def get_proseco_data(pcat):
    if hasattr(pcat, 'acqs'):
        p2 = -np.log10(pcat.acqs.calc_p_safe())
        pcar = pcat.get_review_table()
        pcar.run_aca_review()
        guide_count = pcar.guide_count
        pdata = {'pred_t_ccd_acq': pcat.call_args['t_ccd_acq'],
                 'pred_t_ccd_guide': pcat.call_args['t_ccd_guide'],
                 'p2': p2,
                 'guide_count': guide_count}
    else:
        pdata = {'pred_t_ccd_acq': np.nan,
                 'pred_t_ccd_guide': np.nan,
                 'p2': np.nan,
                 'guide_count': np.nan}
    return pdata


def get_fid_data(pcat):
    if len(pcat.fids) == 0:
        return []
    with Ska.DBI.DBI(dbi='sybase', server='sybase', user='aca_read') as db:
        fids =  db.fetchall(
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
    obs_acq_stats = Table(ACQ_STATS[ACQ_STATS['obsid'] == pcat.obsid])[
        'agasc_id', 'acqid', 'mag_obs', 'cdy', 'cdz', 'sat_pix', 'ion_rad']
    obs_acq_stats.rename_column('agasc_id', 'id')
    if len(obs_acq_stats) == 0:
        return []
    acqs = pcat.acqs.copy()
    acqs = acqs['id', 'slot', 'yang', 'zang', 'mag', 'halfw', 'p_acq']
    acqs = join(acqs, obs_acq_stats)
    for col in ['yang', 'zang', 'mag', 'cdy', 'cdz', 'mag_obs']:
        acqs[col].format = '.1f'
    acqs['p_acq'].format = '.2f'
    return acqs


def get_acq_anoms(acqs):
    for acq in acqs:
        if ((abs(acq['cdy']) > acq['halfw'] + 10) or
            (abs(acq['cdz']) > acq['halfw'] + 10)):
            return True


def get_guide_data(pcat):
    guides = pcat.guides.copy()
    guides = guides['id', 'slot', 'yang', 'zang', 'mag']
    global GUIDE_STATS
    obs_gui_stats = Table(GUIDE_STATS[GUIDE_STATS['obsid'] == pcat.obsid])[
        'agasc_id', 'f_track', 'f_within_3', 'f_within_5', 'dy_mean', 'dz_mean']
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
    if stop is None or DateTime(stop).secs > get_time_range('AACCCDPT')[1]:
        return np.nan
    else:
        return np.max(fetch_sci.Msid('AACCCDPT', start, stop).vals)


def get_manvr_data(manvr):
    p_man_err_before = proseco.acq.get_p_man_err(manvr.one_shot,
                                                 manvr.angle)
    if manvr.get_next() is False:
        one_shot_after = -1
        man_angle_after = -1
        p_man_err_after = 0
    else:
        one_shot_after = manvr.get_next().one_shot
        man_angle_after = manvr.get_next().angle
        p_man_err_after = proseco.acq.get_p_man_err(one_shot_after,
                                                    man_angle_after)
    return {'man_angle_before': manvr.angle,
            'one_shot_before': manvr.one_shot,
            'p_man_err_before': p_man_err_before,
            'one_shot_after': one_shot_after,
            'man_angle_after': man_angle_after,
            'p_man_err_after': p_man_err_after}


def make_cat_page(cat_data, out='./acq.html'):
    if len(cat_data) == 0:
        return
    file_dir = Path(__file__).parent
    template = Template(open(file_dir / "cat_template.html", 'r').read())
    formats = {col: f'%{cat_data[col].format}' for col in cat_data.colnames
               if cat_data[col].format is not None}
    page = template.render(table=cat_data)
    f = open(out, "w")
    f.write(page)
    f.close()


def kalman_ok(kalman_data):
    return kalman_data['consec_lt_two'] <= 27.0


def get_kalman_data(manvr):
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
    return ((acq['p_acq'] > 0.95) & (acq['acqid'] == False))


def is_guide_poorly_tracked(guide):
    return guide['f_within_3'] < 0.90


def is_acquired(acq):
    return acq['acqid'] == True


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
    if len(cat) == 0:
        return [], False
    print_cat = Table(np.zeros((len(cat), len(cat.colnames))).astype(str),
                      names=cat.colnames)
    for col in cat.colnames:
        for row_idx in range(len(cat)):
            if cat[col].format is not None:
                print_cat[col][row_idx] = cat[col].pformat(show_name=False)[row_idx]
            else:
                print_cat[col][row_idx] = f'{cat[col][row_idx]}'
    had_warn = False
    for row_idx in range(len(cat)):
        for func, col in zip(warn_funcs, warn_cols):
            if func(cat[row_idx]):
                print_cat[col][row_idx] = f'<font color="red">{print_cat[col][row_idx]}</font>'
                had_warn = True
    return print_cat, had_warn


def get_and_check_cats(pcat, strobs):
    cat_data = {}
    fid_data = get_fid_data(pcat)
    cat_data['n_bad_fid'] = get_n_bad_fids(fid_data)
    fid_print_data, fid_warn = check_cat_data(fid_data,
                                              [is_fid_poorly_tracked],
                                              ['f_track'])
    make_cat_page(fid_print_data, f'./reps/{strobs}_fid.html')

    guide_data = get_guide_data(pcat)
    cat_data['n_good_guide'] = get_n_good_guides(guide_data)
    guide_print_data, guide_warn = check_cat_data(guide_data,
                                                      [is_guide_poorly_tracked],
                                                      ['f_within_3'])
    make_cat_page(guide_print_data, f'./reps/{strobs}_guide.html')

    acq_data = get_acq_data(pcat)
    cat_data['n_acq'] = get_n_acq(acq_data)
    acq_print_data, acq_warn = check_cat_data(
        acq_data,
        [is_cdy_acq_anom, is_cdz_acq_anom, is_p_acq_anom],
        ['cdy', 'cdz', 'acqid'])
    make_cat_page(acq_print_data, f'./reps/{strobs}_acq.html')

    cat_data.update({'fid_warn': fid_warn,
                     'guide_warn': guide_warn,
                     'acq_warn': acq_warn})
    return cat_data


def t_ccd_ok(dat):
    return dat['t_ccd'] < (dat['pred_t_ccd_guide'] + 3.0)


def get_obsmetrics(manvr):
    warns = []
    metric = {'obsid': manvr.obsid,
              'start': manvr.acq_start}
    if manvr.get_next() is not False:
        metric['t_ccd'] = get_max_tccd(manvr.acq_start, manvr.get_next().start)
    else:
        metric['t_ccd'] = np.nan

    pcat = get_proseco_catalog(manvr)
    obsid = int(pcat.obsid)
    proseco_data = get_proseco_data(pcat)
    manvr_data = get_manvr_data(manvr)
    kal_data = get_kalman_data(manvr)
    strobs = f'{obsid:05d}'
    cat_data  = get_and_check_cats(pcat, strobs)
    for dat in (manvr_data, proseco_data, kal_data, cat_data): 
        metric.update(dat)

    links = {'acq_url': f"./reps/{strobs}_acq.html",
             'guide_url': f"./reps/{strobs}_guide.html",
             'fid_url' : f"./reps/{strobs}_fid.html",
             'dash_url':
             f'https://icxc.cfa.harvard.edu/aspect/centroid_dashboard/{strobs[0:2]}/{strobs}/',
             'mica_url':
             f'https://icxc.cfa.harvard.edu/aspect/mica_reports/{strobs[0:2]}/{strobs}/'}

    metric.update(links)
    warn_map = {
        'KAL': 'http://cxc.cfa.harvard.edu/mta/ASPECT/kalman_watch/',
        'ANOM': metric['mica_url'],
        'HI_BGD':
        f'https://cxc.cfa.harvard.edu/mta/ASPECT/aca_hi_bgd_mon/events/obs_{strobs}/index.html',
        'MANVR': metric['dash_url'],
        'ACQ': metric['acq_url'],
        'GUIDE': metric['guide_url'],
        'FID': metric['fid_url'],
        'T_CCD': metric['mica_url'],
    }
    return metric, warn_map


def make_metric_print(metrics, warn_maps):
    print_cols = ['obsid', 'start', 'mica_url', 'dash_url', 'acq_url', 'guide_url',
                  't_ccd',
                  'one_shot_before', 'one_shot_after',
                  'p2', 'n_acq',
                  'guide_count', 'n_good_guide']
    formats = {'t_ccd': '.2f', 'one_shot_before': '.0f',
               'one_shot_after': '.0f', 'p2': '.2f',
               'guide_count': '.1f'}
    print_table = defaultdict(list)
    for col in print_cols:
        for row in metrics:
            if col in formats:
                print_table[col].append(f'{row[col]:{formats[col]}}')
            else:
                print_table[col].append(str(row[col]))

    all_warns = []
    # And this whole thing in a function
    for row_idx in range(len(metrics)):
        row_warns = []
        row = metrics[row_idx]
        for ctype in ['FID', 'GUIDE', 'ACQ']:
            if row[f'{ctype.lower()}_warn']:
                row_warns.append((ctype, warn_maps[row_idx][ctype]))
        if not np.isnan(row['t_ccd']) and not t_ccd_ok(row):
            row_warns.append(('T_CCD', warn_maps[row_idx]['T_CCD']))
            print_table['t_ccd'][row_idx] = (
                f'<font color="red">{print_table["t_ccd"][row_idx]}</font>')
        if not man_ok(row['one_shot_before'], row['p_man_err_before']):
            row_warns.append(('MANVR', warn_maps[row_idx]['MANVR']))
            print_table['one_shot_before'][row_idx] = (
                f'<font color="red">{print_table["one_shot_before"][row_idx]}</font>')
        if not man_ok(row['one_shot_after'], row['p_man_err_after']):
            row_warns.append(('MANVR', warn_maps[row_idx]['MANVR']))
            print_table['one_shot_after'][row_idx] = (
                f'<font color="red">{print_table["one_shot_after"][row_idx]}</font>')
        if not kalman_ok(row):
            row_warns.append(('KAL', warn_maps[row_idx]['KAL']))
        if row['obsid'] in HI_BGD['obsid']:
            row_warns.append(('HI_BGD', warn_maps[row_idx]['HI_BGD']))
        href_warns = []
        for w in row_warns:
            href_warns.append(f"<A HREF='{w[1]}'>{w[0]}</A>")
        all_warns.append(','.join(href_warns))
        
    print_table['warns'] = all_warns
    return Table(print_table)


def main():
    # these globals are cop-outs but ...

    global ACQ_STATS
    ACQ_STATS = mica.stats.acq_stats.get_stats()
    global GUIDE_STATS
    GUIDE_STATS = mica.stats.guide_stats.get_stats()
    global HI_BGD
    HI_BGD = Table.read('/proj/sot/ska/data/aca_hi_bgd_mon/bgd_events.dat',
                        format='ascii')
    global MP_STARCATS
    MP_STARCATS = get_all_starcats()


    # should get start time from last processed for continuity
    start = DateTime() - 7
    manvrs = events.manvrs.filter(start=start)


    metrics = []
    warn_maps = []
    for manvr in manvrs:
        if manvr.acq_start is None:
            print(f"Skip {manvr.obsid} has no acquisition")
            continue
        print(manvr.start, manvr.stop, manvr.obsid)
        metric, warn_map = get_obsmetrics(manvr)
        metrics.append(metric)
        warn_maps.append(warn_map)

    metrics = Table(metrics)

    metric_print = make_metric_print(metrics, warn_maps)

    outdir = "."
    file_dir = Path(__file__).parent
    template = Template(open(file_dir / "top_level_template.html", 'r').read())
    page = template.render(metrics=metric_print)
    f = open(os.path.join(outdir, "index_draft.html"), "w")
    f.write(page)
    f.close()

    #reports = 'newreports.dat'
    #Table(metrics).write(reports)


if __name__ == '__main__':
    main()




