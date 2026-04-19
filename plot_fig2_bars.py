"""
Fig 2: cumulative impact of infant vaccination scenarios.

Plots from plot-ready CSV `fig23_scens_all.csv` produced by run_scenarios.py.
"""
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sciris as sc
import seaborn as sns

import utils as ut
from run_scenarios import coverage_arr, efficacy_dict


def preprocess_data(scens_df, start_year=2025):
    """Compute cancers/cancer_deaths averted vs adolescent-matched baseline."""
    metrics = ['cancers', 'cancer_deaths']
    efficacy_arr = efficacy_dict['all']
    records = []

    for cn, cov_val in enumerate(coverage_arr):
        base_label = f'Adolescent: {np.round(cov_val, decimals=1)} coverage'
        base = scens_df[(scens_df.scenario == base_label) & (scens_df.year >= start_year)]

        for en, eff_val in enumerate(efficacy_arr):
            scen_label = f'Infants: {np.round(eff_val, decimals=3)} efficacy'
            scen = scens_df[(scens_df.scenario == scen_label) & (scens_df.year >= start_year)]

            for metric in metrics:
                base_vals = base[base.metric == metric].sort_values('year')['value'].values
                scen_vals = scen[scen.metric == metric].sort_values('year')['value'].values
                n_averted = float((base_vals - scen_vals).sum())
                records.append({
                    'coverage': int(round(cov_val, 1) * 100),
                    'efficacy': int(round(eff_val, 1) * 100),
                    'metric': metric.replace('_', ' ').capitalize(),
                    'val': n_averted,
                })
    return pd.DataFrame(records)


def plot_fig2(df, outpath='figures/fig2_vx_impact.png'):
    sns.set_style('whitegrid')
    ut.set_font(30)
    g = sns.catplot(
        data=df.loc[df.metric != 'cost'],
        kind='bar', x='efficacy', y='val', row='metric',
        hue='coverage', palette='rocket_r', sharey=False,
        height=5, aspect=3,
    )
    g.set_axis_labels('Vaccine efficacy for infants (%)', '')
    g.set_titles('{row_name} averted')
    for ax in g.axes.flat:
        sc.SIticks(ax)
    g.legend.set_title('Adolescent\ncoverage (%)')
    plt.savefig(outpath, dpi=100)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--resfolder', default='results')
    parser.add_argument('--outpath', default='figures/fig2_vx_impact.png')
    args = parser.parse_args()

    scens_df = pd.read_csv(f'{args.resfolder}/fig23_scens_all.csv')
    df = preprocess_data(scens_df)
    plot_fig2(df, outpath=args.outpath)
    print('Done.')
