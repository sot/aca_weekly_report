import os
from glob import glob
import numpy as np
from jinja2 import Template
from pathlib import Path
import pickle
import gzip
import logging
from astropy.table import Table, join
import shelve
import tables

import Ska.DBI
from Chandra.Time import DateTime
from kadi import events
#import kadi.commands
#import kadi.commands.states as kadi_states
from Ska.engarchive import fetch_sci
from Ska.engarchive.fetch import get_time_range
from Ska.engarchive.utils import logical_intervals
from chandra_aca.centroid_resid import CentroidResiduals
import mica.vv
from mica.common import MICA_ARCHIVE
import mica.starcheck
import mica.stats.acq_stats
import mica.stats.guide_stats
import proseco
import proseco.core
from proseco.acq import get_p_man_err
import sparkles


with tables.open_file('/proj/sot/ska/data/kadi/cmds.h5', 'r') as h5:
    all_cmds = Table(h5.root.data[:])
kadi_starcats = all_cmds[all_cmds['type'] == 'MP_STARCAT']
del all_cmds


with Ska.DBI.DBI(dbi='sqlite', server='/proj/sot/ska/data/cmd_states/cmd_states.db3') as db:
    timelines = Table(db.fetchall("select * from timeline_loads"))
timelines.rename_column('id', 'timeline_id')
kadi_starcats['date'] = kadi_starcats['date'].astype(str)
MP_STARCATS = join(kadi_starcats['date', 'timeline_id'],
                   timelines['timeline_id', 'mp_dir'], 
                   keys=['timeline_id'],
                   )['date', 'mp_dir']
del kadi_starcats
del timelines
MP_STARCATS.sort('date')


ACQ_STATS = mica.stats.acq_stats.get_stats()
GUIDE_STATS = mica.stats.guide_stats.get_stats()


HI_BGD = Table.read('/proj/sot/ska/data/aca_hi_bgd_mon/bgd_events.dat',
                    format='ascii')

reports = 'newreports.dat'


MP_DIR = '/data/mpcrit1/mplogs'

def fake_pcat(obsid, mp_dir):
    sc = mica.starcheck.get_starcheck_catalog(obsid, mp_dir=mp_dir)
    raise ValueError
    

def get_proseco_cat(manvr):
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
        obsid, pcat = fake_pcat(manvr.obsid, mp_starcat['mp_dir'])
    return obsid, pcat


def get_proseco_data(manvr):
    obsid, pcat = get_proseco_cat(manvr)
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
    return obsid, pdata, pcat


def get_fid_data(obsid, pcat):
    with Ska.DBI.DBI(dbi='sybase', server='sybase', user='aca_read') as db:
        fids =  Table(db.fetchall(
            f"select * from trak_stats_data where obsid = {obsid} and type = 'FID'"))
    fids['f_track'] = (fids['n_samples'] - fids['not_tracking_samples']) / fids['n_samples']
    fids = fids['slot', 'f_track']
    fids = join(pcat.fids, fids, keys=['slot'])
    for col in ['f_track']:
        fids[col].format = '.2f'
    return fids['slot', 'id', 'yang', 'zang', 'spoiler_score', 'f_track']


def get_acq_data(obsid, pcat):
    obs_acq_stats = Table(ACQ_STATS[ACQ_STATS['obsid'] == obsid])[
        'agasc_id', 'acqid', 'mag_obs', 'cdy', 'cdz', 'sat_pix', 'ion_rad']
    obs_acq_stats.rename_column('agasc_id', 'id')
    if len(obs_acq_stats) == 0:
        return []
    acqs = pcat.acqs.copy()
    acqs = acqs['id', 'slot', 'yang', 'zang', 'mag', 'halfw', 'p_acq']
    acqs = join(acqs, obs_acq_stats)
    acqs['cdy'].format = '.1f'
    acqs['cdz'].format = '.1f'
    return acqs


def check_acq_data(acqs):
    acq_warns = []
    if np.any(acqs[acqs['p_acq'] > 0.95]['acqid'] == False):
        for fails in acqs[acqs['acqid'] == False]:
            if fails['p_acq'] > 0.95:
                acq_warns.append(f'Did not acquire star {fails["id"]} '
                                 + f'p_acq {fails["p_acq"]:.3f}')
    n_acq = np.count_nonzero(acqs['acqid'])
    if n_acq <= 3:
        acq_warns.append(f'Only acquired {n_acq} stars')
    for acq in acqs:
        if ((abs(acq['cdy']) > acq['halfw'] + 10) or
            (abs(acq['cdz']) > acq['halfw'] + 10)):
            acq_warns.append('Looks like acquisition anomaly')
    return acq_warns


def get_guide_data(obsid, pcat):
    guides = pcat.guides.copy()
    guides = guides['id', 'slot', 'yang', 'zang', 'mag']
    obs_gui_stats = Table(GUIDE_STATS[GUIDE_STATS['obsid'] == obsid])[
        'agasc_id', 'f_track', 'f_within_3', 'f_within_5', 'dy_mean', 'dz_mean']
    obs_gui_stats.rename_column('agasc_id', 'id')
    if len(obs_gui_stats) == 0:
        return []
    guides = join(guides, obs_gui_stats)
    for col in ['f_track', 'f_within_3', 'f_within_5', 'dy_mean', 'dz_mean']:
        guides[col].format = '.2f'
    return guides


def check_guide_data(guides):
    guide_warns = []
    if np.any(guides['f_within_3'] < 0.90):
        for star in guides[guides['f_within_3'] < 0.90]:
            guide_warns.append(f'Star {star["id"]} f_within_3 = {star["f_within_3"]}')
    return guide_warns


def get_temperatures(manvr):
    trange = get_time_range('AACCCDPT')
    if manvr.get_next() is not False and trange[1] > DateTime(manvr.get_next().start).secs:
        t_ccd_acq = np.max(fetch_sci.Msid(
                'AACCCDPT', manvr.acq_start, DateTime(manvr.acq_start).secs + 360).vals)
        t_ccd_guide = np.max(fetch_sci.Msid(
            'AACCCDPT', manvr.guide_start, manvr.get_next().start).vals)
        return {'t_ccd_acq': t_ccd_acq,
                't_ccd_guide': t_ccd_guide}
    else:
        return {'t_ccd_acq': np.nan,
                't_ccd_guide': np.nan}



def get_manvr_data(manvr, pcat):
    p_man_err = proseco.acq.get_p_man_err(manvr.one_shot,
                                          pcat.man_angle)
    if (p_man_err > 0.025) or (manvr.one_shot < 60):
        man_ok = True
    else:
        man_ok = False
    return manvr, {'proseco_man_angle': pcat.man_angle,
                   'one_shot': manvr.one_shot,
                   'p_man_err': p_man_err,
                   'man_ok': man_ok}



def make_cat_page(cat_data, warns, out='./acq.html'):
    file_dir = Path(__file__).parent
    template = Template(open(file_dir / "cat_template.html", 'r').read())
    if len(cat_data) > 0:
        cat_data._base_repr_()
        cat = '\n'.join(cat_data.pformat(max_width=-1, max_lines=-1))
    else:
        cat = ""
    page = template.render(table=cat, warns=warns)
    f = open(out, "w")
    f.write(page)
    f.close()
    

def get_kalman_data(manvr):
    kalman_data = {'consec_lt_two': np.nan,
                   'min_kalstr': np.nan,
                   'kalman_ok': True}
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
    if kalman_data['consec_lt_two'] > 27.0:
        kalman_data['kalman_ok'] = False
    return kalman_data


def check_fid_data(fids):
    warns = []
    if np.any(fids['f_track'] < 0.95):
        for fid in fids[fids['f_track'] < 0.95]:
            warns.append(f'Fid {fid["id"]} f_track = {fid["f_track"]}')
    return warns



def get_obsmetrics(manvr):
    metric = {'obsid': manvr.obsid}
    obs_warns = []
    #obsid = manvr.obsid
    obsid, proseco_data, pcat = get_proseco_data(manvr)
    strobs = f'{obsid:05d}'
    manvr, manvr_data = get_manvr_data(manvr, pcat)
    temperatures = get_temperatures(manvr)
    if (not np.isnan(temperatures['t_ccd_acq']) and
        (temperatures['t_ccd_acq'] > pcat.t_ccd_acq + 3)):
        obs_warns.append('HI_TEMP')
    if (not np.isnan(temperatures['t_ccd_guide']) and
        (temperatures['t_ccd_guide'] > pcat.t_ccd_guide + 3)):
        obs_warns.append('HI_TEMP')

    if len(pcat.fids) > 0:
        fid_data = get_fid_data(obsid, pcat)
        fid_warns = check_fid_data(fid_data)
        make_cat_page(fid_data, fid_warns, f'./{strobs}_fid.html')
        if len(fid_warns) > 0:
            obs_warns.append('FID')

    guide_data = get_guide_data(obsid, pcat)
    if len(guide_data) == 0:
        guide_warns = ['No observed guide data']
    else:
        guide_warns = check_guide_data(guide_data)

    acq_data = get_acq_data(obsid, pcat)
    if len(acq_data) == 0:
        acq_warns = ['No observed acq data']
        metric['n_acq'] = -1
    else:
        acq_warns = check_acq_data(acq_data)
        metric['n_acq'] = np.count_nonzero(acq_data['acqid'] == True)


    make_cat_page(acq_data, acq_warns, f'./{strobs}_acq.html')
    make_cat_page(guide_data, guide_warns, f'./{strobs}_guide.html')

    kal_data = get_kalman_data(manvr)
    if kal_data['kalman_ok'] == False:
        obs_warns.append('KAL')

    for dat in (manvr_data, temperatures, proseco_data, kal_data): # acq_data, guide_data):
        metric.update(dat)

    if metric['man_ok'] == False:
        obs_warns.append('MANVR')

    if len(acq_warns) > 0:
        obs_warns.append('ACQ')

    if len(guide_warns) > 0:
        obs_warns.append('GUIDE')

    if obsid in HI_BGD['obsid']:
        obs_warns.append('HI_BGD')

    metric['acq_url'] = f"{strobs}_acq.html"
    metric['guide_url'] = f"{strobs}_guide.html"
    metric['fid_url'] = f"{strobs}_fid.html"
    metric['dash_url'] = (
        f'https://icxc.cfa.harvard.edu/aspect/centroid_dashboard/{strobs[0:2]}/{strobs}/')
    metric['mica_url'] = (
        f'https://icxc.cfa.harvard.edu/aspect/mica_reports/{strobs[0:2]}/{strobs}/')

    warn_map = {'KAL': 'http://cxc.cfa.harvard.edu/mta/ASPECT/kalman_watch/',
                'HI_BGD': 'https://cxc.cfa.harvard.edu/mta/ASPECT/aca_hi_bgd_mon/',
                'MANVR': metric['dash_url'],
                'ACQ': metric['acq_url'],
                'GUIDE': metric['guide_url'],
                'FID': metric['fid_url'],
                'HI_TEMP': metric['mica_url']}

    url_warns = [f"<A HREF='{warn_map[warn]}'>{warn}</A>" for warn in obs_warns]
    metric['warns'] = ",".join(url_warns)
    return metric


# should get start time from last processed for continuity
#start = DateTime() - 14.5
#dwells = events.dwells.filter(start=start, stop=start + 1)
#manvrs = events.manvrs.filter(start='2020:106:05:50:25.185', stop='2020:108:05:50:25.185')
#manvrs = events.manvrs.filter(DateTime() - 14)
#manvrs = events.manvrs.filter(start='2020:078', stop='2020:081')
#manvrs = events.manvrs.filter(obsid=22500)
manvrs = events.manvrs.filter(start='2020:058', stop='2020:061')
#manvrs = events.manvrs.filter(obsid=23037)
#dwells = events.dwells.filter(obsid=22643)
#manvrs = events.manvrs.filter('2019:353', '2019:354')
#manvrs = events.manvrs.filter('2012:246', '2012:251')
#raise ValueError
#logger = logging.getLogger('vv')
#logger.setLevel(50)

metrics = []
for manvr in manvrs:
    print(manvr.start, manvr.stop, manvr.obsid)
    metrics.append(get_obsmetrics(manvr))

outdir = "."
file_dir = Path(__file__).parent
template = Template(open(file_dir / "top_level_template.html", 'r').read())
page = template.render(metrics=metrics)
f = open(os.path.join(outdir, "index_draft.html"), "w")
f.write(page)
f.close()

#Table(metrics).write(reports)


#def get_res(obsid, pcat):
#    #ds = events.dwells.filter(obsid=obsid)
#    #start = ds[0].start
#    #stop = ds[len(ds) - 1].stop
#    att_source = 'obc' if obsid > 38000 else 'ground'
#    slot_res = {}
#    for slot in pcat.guides['slot']:
#        val = CentroidResiduals.for_slot(obsid=obsid, slot=slot,
#                                         att_source=att_source, centroid_source='obc')
#        out = {}
#        if len(val.dyags) > 0 and len(val.dzags) > 0:
#            out['std_dy'] = np.std(val.dyags)
#            out['std_dz'] = np.std(val.dzags)
#            out['median_dy'] = np.median(val.dyags)
#            out['median_dz'] = np.median(val.dzags)
#            drs = np.sqrt((val.dyags ** 2) + (val.dzags ** 2))
#            out['median_dr'] = np.median(drs)
#            for dist in ['1.5', '5.0']:
#                out[f'f_within_{dist}'] = np.count_nonzero(drs < float(dist)) / len(drs)
#        else:
#            for metric in ['median_dr', 'median_dy', 'median_dz', 'std_dy', 'std_dz']:
#                out[metric] = -9999
#            for metric in ['f_within_5.0', 'f_within_1.5']:
#                out[metric] = 0
#        slot_res[slot] = out
#    return slot_res
#        

#def get_cen_dash(obsid):
    #per_obs = CEN_DASH_OBSID[CEN_DASH_OBSID['obsid'] == obsid]
    #slot_data = CEN_DASH_SLOT[CEN_DASH_SLOT['obsid'] == obsid]
    #raise ValueError
