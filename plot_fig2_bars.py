"""
Fig 2: cumulative cancers/cancer deaths averted by coverage × efficacy.

Loads plot-ready CSV `fig2_averted.csv` (precomputed per (coverage, efficacy, metric)).
"""
import argparse

import matplotlib.pyplot as plt
import pandas as pd
import sciris as sc
import seaborn as sns

import utils as ut


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
    parser.add_argument('--resfolder', default='results/v2.0.x_published')
    parser.add_argument('--outpath', default='figures/fig2_vx_impact.png')
    args = parser.parse_args()
    df = pd.read_csv(f'{args.resfolder}/fig2_averted.csv')
    plot_fig2(df, outpath=args.outpath)
    print('Done.')
