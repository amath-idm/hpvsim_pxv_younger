"""
Extract plot-ready CSVs from hpvsim_pxv_younger .obj source files.

Usage:
  python save_baselines.py [--source results/raw_v2/results] [--outdir results/v2.0.x_published]
"""

import argparse
import json
from datetime import date
from pathlib import Path

import hpvsim as hpv
import numpy as np
import pandas as pd
import sciris as sc

from run_scenarios import coverage_arr, efficacy_dict


# --- figS1: sexual behavior ---
def save_figS1(src, dst):
    for name in ['model_sb_AFS', 'model_sb_prop_married', 'model_age_diffs', 'model_casual']:
        df = sc.loadobj(f'{src}/{name}.obj')
        df.to_csv(f'{dst}/{name}.csv', index=False)

    partners = sc.loadobj(f'{src}/partners.obj')
    rows = [{'sex': s, 'partner_count': int(v)} for s, arr in partners.items() for v in arr]
    pd.DataFrame(rows).to_csv(f'{dst}/partners.csv', index=False)


# --- figS2: calibration (reduced) ---
def save_figS2(src, dst, res_to_plot=100):
    calib = sc.loadobj(f'{src}/nigeria_calib_reduced.obj')
    n = min(res_to_plot, len(calib.analyzer_results))

    rows = []
    for pos in range(n):
        cancers = calib.analyzer_results[pos]['cancers'][2020]
        for bi, val in enumerate(cancers):
            rows.append({'trial_pos': pos, 'bin': bi, 'value': float(val)})
    pd.DataFrame(rows).to_csv(f'{dst}/figS2_cancers_by_age.csv', index=False)

    for rkey in ['cin_genotype_dist', 'cancerous_genotype_dist']:
        rows = []
        for pos in range(n):
            run = calib.sim_results[pos][rkey]
            arr = [run] if sc.isnumber(run) else list(run)
            for bi, val in enumerate(arr):
                rows.append({'trial_pos': pos, 'bin': bi, 'value': float(val)})
        pd.DataFrame(rows).to_csv(f'{dst}/figS2_{rkey}.csv', index=False)

    for ti, name in enumerate(['cancers', 'cin_genotype', 'cancerous_genotype']):
        if ti < len(calib.target_data):
            calib.target_data[ti].to_csv(f'{dst}/figS2_target_{name}.csv', index=False)


# --- figS3: age pyramids ---
def save_figS3(src, dst, years=('2025', '2050', '2075', '2100')):
    sim = sc.loadobj(f'{src}/nigeria.sim')
    a = sim.get_analyzer('age_pyramid')

    model_rows = []
    for yi, yr in enumerate(years):
        p = sc.odict(a.age_pyramids)[yi]
        for bin_val, m_val, f_val in zip(p['bins'], p['m'], p['f']):
            model_rows.append({'year': yr, 'bin': int(bin_val), 'm': int(m_val), 'f': int(f_val)})
    pd.DataFrame(model_rows).to_csv(f'{dst}/figS3_model.csv', index=False)

    data = a.data.copy()
    data.columns = data.columns.str[0].str.lower()
    data = data[data['y'].isin([float(y) for y in years])]
    data.to_csv(f'{dst}/figS3_data.csv', index=False)


# --- fig2: bars (vx_scens_all) and fig3: time series (vx_scens_equiv) ---
_FIG3_METRICS = ['asr_cancer_incidence', 'cancers', 'cancer_deaths']


def _flatten_scens(scens_obj):
    rows = []
    for scen_label, mres in scens_obj.items():
        years = mres['year']
        for mi, metric in enumerate(_FIG3_METRICS):
            series = mres[metric]
            for yi, yr in enumerate(years):
                rows.append({
                    'scenario': scen_label, 'year': float(yr), 'metric': metric,
                    'value': float(series.values[yi]),
                    'low': float(series.low[yi]),
                    'high': float(series.high[yi]),
                })
    return pd.DataFrame(rows)


def _flatten_precin(scens_obj):
    rows = []
    ts = 0.67
    for scen_label, mres in scens_obj.items():
        years = mres['year']
        precin = mres['n_precin_by_age']
        females = mres['n_females_alive_by_age']
        for yi, yr in enumerate(years):
            val = precin.values[3:11, yi].sum() / females.values[3:11, yi].sum() * ts
            lo = precin.low[3:11, yi].sum() / females.low[3:11, yi].sum() * ts
            hi = precin.high[3:11, yi].sum() / females.high[3:11, yi].sum() * ts
            rows.append({'scenario': scen_label, 'year': float(yr),
                         'metric': 'precin_incidence', 'value': val, 'low': lo, 'high': hi})
    return pd.DataFrame(rows)


def save_fig23(src, dst):
    for kind in ['all', 'equiv']:
        fname = f'vx_scens_{kind}.obj'
        scens = sc.loadobj(f'{src}/{fname}')
        df = _flatten_scens(scens)
        precin_df = _flatten_precin(scens)
        pd.concat([df, precin_df], ignore_index=True).to_csv(
            f'{dst}/fig23_scens_{kind}.csv', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', default='results/raw_v2/results')
    parser.add_argument('--outdir', default='results/v2.0.x_published')
    args = parser.parse_args()

    dst = Path(args.outdir)
    dst.mkdir(parents=True, exist_ok=True)

    save_figS1(args.source, str(dst))
    save_figS2(args.source, str(dst))
    save_figS3(args.source, str(dst))
    save_fig23(args.source, str(dst))

    manifest = {
        'figures': ['fig2', 'fig3', 'figS1', 'figS2', 'figS3'],
        'source': args.source,
        'extracted_on_hpvsim_version': hpv.__version__,
        'date': date.today().isoformat(),
        'note': 'Source .obj files dated Nov 2024, likely hpvsim 2.0.x era.',
    }
    (dst / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    print(f'Saved baselines to {dst}')
