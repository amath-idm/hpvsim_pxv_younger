"""
Plot 2 and 3 for infant vaccination scenarios
"""

import pandas as pd
import sciris as sc
from run_scenarios import coverage_arr, efficacy_arr
import utils as ut
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as tkr

def preprocess_data(msim_dict, cost_dict):

    # What to store
    start_year = 2025
    metrics = ['cancers', 'cancer_deaths']
    records = sc.autolist()

    for cn, cov_val in enumerate(coverage_arr):
        base_label = f'Adolescent: {cov_val} coverage'
        si = sc.findinds(msim_dict[base_label].year, start_year)[0]
        di = sc.findinds(msim_dict[base_label].dalys.index.values, start_year)[0]
        base_dalys = msim_dict[base_label].dalys.dalys.values[di:]

        for en, eff_val in enumerate(efficacy_arr):
            scen_label = f'Adolescents: {cov_val} coverage, Infants: {eff_val} efficacy'
            scen_dalys = msim_dict[scen_label].dalys.dalys.values[di:]
            dalys_averted = sum(base_dalys - scen_dalys)
            records += {'coverage': cov_val, 'efficacy': eff_val, 'metric':'DALYs', 'val': dalys_averted}

            for pn, metric in enumerate(metrics):
                base_vals = msim_dict[base_label][metric].values[si:]
                scen_vals = msim_dict[scen_label][metric].values[si:]
                n_averted = sum(base_vals - scen_vals)
                records += {'coverage': cov_val, 'efficacy': eff_val, 'metric': f'{metric.replace("_"," ").capitalize()}', 'val': n_averted}

            # Costs
            scen_costs = 0
            base_costs = 0
            for cname, cost in cost_dict.items():
                if msim_dict[scen_label].get(cname):
                    scen_costs += msim_dict[scen_label][cname].values * cost
                if msim_dict[base_label].get(cname):
                    base_costs += msim_dict[scen_label][cname].values * cost

            total_scen_cost = sum([i / 1.03 ** t for t, i in enumerate(scen_costs)])
            total_base_cost = sum([i / 1.03 ** t for t, i in enumerate(base_costs)])
            costs_averted = total_base_cost - total_scen_cost

            records += {'coverage': cov_val, 'efficacy': eff_val, 'metric': 'cost', 'val': costs_averted}

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
        palette="rocket",
        sharey=False,
        height=5, aspect=3,
    )
    g.set_axis_labels("Vaccine efficacy for infants", "")
    g.set_titles("{row_name} averted")

    for ax in g.axes.flat:
        ax.yaxis.set_major_formatter(tkr.FuncFormatter(lambda x, p: format(int(x), ',')))
    g.legend.set_title("Adolescent\ncoverage")

    fig_name = 'figures/vx_impact.png'
    sc.savefig(fig_name, dpi=100)
    return


def plot_fig3(df):
    sns.set_style("whitegrid")
    ut.set_font(14)
    df2 = df.loc[df['metric'].isin(['DALYs', 'cost'])]
    df2 = df2.groupby(['coverage', 'efficacy', 'metric']).val.first().unstack().reset_index()
    df2['Cost per DALY averted'] = df2['cost'] / df2['DALYs']
    df2['DALYs'] = df2['DALYs']/1e6
    df2['Adolescent coverage'] = df2['coverage']*100

    g = sns.FacetGrid(
        data=df2,
        col="Adolescent coverage",
        col_wrap=2,
        hue="efficacy",
        palette="rocket",
    )
    g.map(sns.scatterplot, "DALYs", "Cost per DALY averted")
    g.add_legend(frameon=False)
    g.legend.set_title("Infant\nefficacy")
    for ax in g.axes.flat:
        ax.yaxis.set_major_formatter(tkr.FuncFormatter(lambda x, p: format(int(x), ',')))
        ax.xaxis.set_major_formatter(tkr.FuncFormatter(lambda x, p: format(int(x))))
        ax.set_xlim(-20, 20)

    fig_name = 'figures/vx_econ_impact.png'
    sc.savefig(fig_name, dpi=100)
    return df2


# %% Run as a script
if __name__ == '__main__':

    # Load scenarios and construct figure
    msim_dict = sc.loadobj('results/vx_scens.obj')
    cost_dict = sc.objdict({
        'Routine vx': 9,
        'Catchup vx': 9,
        'Infant vx': 5,
        'excision': 41.76,
        'ablation': 11.76,
        'radiation': 450
    })
    df = preprocess_data(msim_dict, cost_dict)

    plot_fig2(df)
    df2 = plot_fig3(df)
