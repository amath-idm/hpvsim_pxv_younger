"""
Plot 2: cumulative impact of infant vaccination scenarios
"""

import pandas as pd
import sciris as sc
from run_scenarios import coverage_arr, efficacy_arr
import utils as ut
import seaborn as sns
import numpy as np


def preprocess_data(msim_dict, cost_dict=None):

    # What to store
    start_year = 2025
    metrics = ['cancers', 'cancer_deaths']
    records = sc.autolist()

    for cn, cov_val in enumerate(coverage_arr):
        base_label = f'Adolescent: {np.round(cov_val, decimals=1)} coverage'
        si = sc.findinds(msim_dict[base_label].year, start_year)[0]

        for en, eff_val in enumerate(efficacy_arr):
            # cov_val = eff_val*0.9/0.95
            base_label = f'Adolescent: {np.round(cov_val, decimals=1)} coverage'
            scen_label = f'Infants: {np.round(eff_val, decimals=3)} efficacy'

            for pn, metric in enumerate(metrics):
                base_vals = msim_dict[base_label][metric].values[si:]
                scen_vals = msim_dict[scen_label][metric].values[si:]
                n_averted = sum(base_vals - scen_vals)
                records += {'coverage': int(round(cov_val, 1)*100), 'efficacy': int(round(eff_val, 1)*100), 'metric': f'{metric.replace("_"," ").capitalize()}', 'val': n_averted}

    df = pd.DataFrame.from_dict(records)

    return df


def plot_fig2(df):

    sns.set_style("whitegrid")
    ut.set_font(30)
    g = sns.catplot(
        data=df.loc[df.metric != 'cost'],
        kind="bar",
        x="efficacy",
        y="val",
        row="metric",
        hue="coverage",
        palette="rocket_r",
        sharey=False,
        height=5, aspect=3,
    )
    g.set_axis_labels("Vaccine efficacy for infants (%)", "")
    g.set_titles("{row_name} averted")

    for ax in g.axes.flat:
        sc.SIticks(ax)
    g.legend.set_title("Adolescent\ncoverage (%)")

    # fig.tight_layout()
    fig_name = 'figures/fig2_vx_impact.png'
    sc.savefig(fig_name, dpi=100)
    return


# %% Run as a script
if __name__ == '__main__':

    # Load scenarios and construct figure
    msim_dict = sc.loadobj('results/vx_scens_all.obj')
    df = preprocess_data(msim_dict)

    plot_fig2(df)
