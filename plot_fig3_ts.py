"""
Fig 3: comparable-efficacy infant vaccination scenarios.

Plots from plot-ready CSV `fig23_scens_equiv.csv` produced by run_scenarios.py.
"""
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sciris as sc

import utils as ut
from run_scenarios import coverage_arr


def _select(df, scenario, metric, start_year, end_year, smooth=True):
    sub = df[(df.scenario == scenario) & (df.metric == metric)
             & (df.year >= start_year) & (df.year < end_year)].sort_values('year')
    years = sub['year'].values
    best = sub['value'].values
    low = sub['low'].values
    high = sub['high'].values
    if smooth:
        best = np.convolve(best, np.ones(5), 'valid') / 5
        low = np.convolve(low, np.ones(5), 'valid') / 5
        high = np.convolve(high, np.ones(5), 'valid') / 5
        years = years[4:]
    return years, best, low, high


def plot_fig3(scens_df, outpath='figures/fig3_vx_scens.png'):
    ut.set_font(20)
    plot_coverage_arr = coverage_arr[::2]
    plot_efficacy_arr = 0.95 * plot_coverage_arr / 0.9
    colors = sc.vectocolor(len(plot_efficacy_arr), reverse=True)
    plot_dict = sc.objdict(
        precin_incidence='Detectable HPV prevalence, females 15+',
        asr_cancer_incidence='ASR cancer incidence',
    )

    fig, axes = plt.subplots(len(plot_dict), 2, figsize=(17, 10))
    start_year, end_year = 2016, 2100

    for rn, (metric, plot_label) in enumerate(plot_dict.items()):
        # Column 0: adolescent scenarios
        ax = axes[rn, 0]
        yrs, best, *_ = _select(scens_df, 'Baseline', metric, start_year, end_year)
        ax.plot(yrs, best, color='k', label='Baseline')
        for cvn, cov_val in enumerate(plot_coverage_arr):
            label = f'Adolescent: {np.round(cov_val, decimals=1)} coverage'
            yrs, best, *_ = _select(scens_df, label, metric, start_year, end_year)
            ax.plot(yrs, best, color=colors[cvn],
                    label=f'{int(np.floor(cov_val*100))}% coverage')
        ax.set_ylim(bottom=0)
        ax.set_ylabel(plot_label)
        ax.set_title(f'Adolescent vaccination scenarios\n{plot_label}')
        if metric == 'asr_cancer_incidence':
            ax.axhline(y=4, color='k', ls='--')
        if rn == 0:
            ax.legend(frameon=False)

        # Column 1: infant scenarios
        ax = axes[rn, 1]
        yrs, best, *_ = _select(scens_df, 'Baseline', metric, start_year, end_year)
        ax.plot(yrs, best, color='k', label='Baseline')
        for ie, eff_val in enumerate(plot_efficacy_arr):
            label = f'Infants: {np.round(eff_val, decimals=3)} efficacy'
            yrs, best, *_ = _select(scens_df, label, metric, start_year, end_year)
            ax.plot(yrs, best, color=colors[ie],
                    label=f'{int(np.ceil(eff_val*100))}% efficacy')
        ax.set_ylim(bottom=0)
        ax.set_ylabel(plot_label)
        ax.set_title(f'Equivalent infant vaccination scenarios\n{plot_label}')
        if metric == 'asr_cancer_incidence':
            ax.axhline(y=4, color='k', ls='--')
        if rn == 0:
            ax.legend(frameon=False)

    fig.tight_layout()
    plt.savefig(outpath, dpi=100)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--resfolder', default='results')
    parser.add_argument('--outpath', default='figures/fig3_vx_scens.png')
    args = parser.parse_args()

    scens_df = pd.read_csv(f'{args.resfolder}/fig23_scens_equiv.csv')
    plot_fig3(scens_df, outpath=args.outpath)
    print('Done.')
