"""
Plot calibration to India
"""
import hpvsim as hpv
import hpvsim.utils as hpu
import hpvsim.parameters as hppar
import pylab as pl
import pandas as pd
from scipy.stats import lognorm, norm
import numpy as np
import sciris as sc
import utils as ut
import seaborn as sns

import run_sim as rs


# %% Functions

def plot_calib(calib, res_to_plot=100):

    ut.set_font(size=24)
    fig = pl.figure(layout="tight", figsize=(12, 11))
    prev_col = '#5f5cd2'
    canc_col = '#c1981d'
    ms = 80
    gen_cols = sc.gridcolors(4)

    # Make 2 rows, with 2 panels in the top row and 3 in the bottom
    gs0 = fig.add_gridspec(2, 1)
    gs00 = gs0[0].subgridspec(1, 1)
    gs01 = gs0[1].subgridspec(1, 4)

    # Pull out the analyzer and sim results
    analyzer_results = calib.analyzer_results
    sim_results = calib.sim_results

    # ###############
    # # Panel A: HPV prevalence by age
    # ###############
    res = sc.loadobj(f'results/nigeria_msim.obj')
    year = 2020
    ind = sc.findinds(res['year'], year)[0]

    pre_cins = dict()
    ts = 1  #0.5  # detection rate / sensitivity
    for which in ['values', 'low', 'high']:
        this_res = res['n_precin_by_age'][which][:, ind]
        pre_cins[which] = [
            sum(this_res[3:5]) / sum(res['n_females_alive_by_age'][3:5, ind]),
            sum(this_res[5:7])*ts / sum(res['n_females_alive_by_age'][5:7, ind]),
            sum(this_res[7:9])*ts / sum(res['n_females_alive_by_age'][7:9, ind]),
            sum(this_res[9:11])*ts / sum(res['n_females_alive_by_age'][9:11, ind]),
            sum(this_res[11:13])*ts / sum(res['n_females_alive_by_age'][11:13, ind]),
            sum(this_res[13:])*ts / sum(res['n_females_alive_by_age'][13:, ind]),
        ]

    ax = fig.add_subplot(gs01[:2])

    # Extract data
    # datadf = calib.target_data[0]
    # best = datadf.value.values
    age_labels = ['15-25', '25-34', '35-44', '45-54', '55-64', '65+']
    x = np.arange(len(age_labels))
    best = np.array([.30, .25, .25, .23, .23, .25])

    # Pull out lower and upper bounds from Figure 54 here: https://hpvcentre.net/statistics/reports/IND.pdf
    lowererr = np.array([0.08, 0.08, 0.08, 0.08, 0.08, 0.08 ])
    uppererr = np.array([0.08, 0.08, 0.08, 0.08, 0.08, 0.08 ])
    err = [lowererr, uppererr]

    # # Extract model results
    # bins = []
    # values = []
    # for run_num, run in enumerate(analyzer_results):
    #     bins += x.tolist()
    #     values += list(run['hpv_prevalence'][2020])
    # modeldf = pd.DataFrame({'bins': bins, 'values': values})

    # # Plot model
    # sns.lineplot(ax=ax, x='bins', y='values', data=modeldf, color=prev_col, errorbar=('pi', 95))

    # Plot model from msim:
    ax.plot(x, pre_cins['values'], color=prev_col)
    ax.fill_between(x, pre_cins['low'], pre_cins['high'], color=prev_col, alpha=0.3)

    # Plot data
    ax.errorbar(x, best, yerr=err, ls='none', marker='d', markersize=ms/10, color='k')

    # Axis sttings
    ax.set_ylim([0, 0.75])
    ax.set_xticks(x, age_labels)
    ax.set_ylabel('')
    ax.set_xlabel('Age')
    ax.set_title('Detectable HPV\n prevalence, 2020')
    # ax.set_title('Detectable HPV prevalence,\n normal cervical cytology, 2020')

    ###############
    # Panel B: Cancers by age
    ###############
    # ax = fig.add_subplot(gs01[0])
    ax = fig.add_subplot(gs00[0])

    # Data
    datadf = calib.target_data[0]
    age_labels = ['0', '15', '20', '25', '30', '35', '40', '45', '50', '55', '60', '65', '70', '75', '80', '85']
    x = np.arange(len(age_labels))
    best = datadf.value.values

    # Extract model results
    bins = []
    values = []
    for run_num, run in enumerate(analyzer_results):
        bins += x.tolist()
        values += list(run['cancers'][2020])
    modeldf = pd.DataFrame({'bins': bins, 'values': values})

    sns.lineplot(ax=ax, x='bins', y='values', data=modeldf, color=canc_col, errorbar=('pi', 95))
    ax.scatter(x, best, marker='d', s=ms, color='k')

    ax.set_ylim([0, 2_500])
    ax.set_xticks(x, age_labels)
    sc.SIticks(ax)
    ax.set_ylabel('')
    ax.set_xlabel('Age')
    ax.set_title('Cancers by age, 2020')

    # CINS and cancer by genotype
    rkeys = ['cin_genotype_dist', 'cancerous_genotype_dist']
    rlabels = ['HSILs', 'Cancers']  #['HSILs by genotype', 'Cancers by genotype']
    for ai, rkey in enumerate(rkeys):
        ax = fig.add_subplot(gs01[ai+2])

        # Plot data
        datadf = calib.target_data[ai+1]
        ydata = datadf.value.values
        x = np.arange(len(ydata))

        # Extract model results
        bins = []
        values = []
        for run_num, run in enumerate(sim_results):
            bins += x.tolist()
            if sc.isnumber(run[rkey]):
                values += sc.promotetolist(run[rkey])
            else:
                values += run[rkey].tolist()
        modeldf = pd.DataFrame({'bins': bins, 'values': values})

        # Plot model
        sns.boxplot(ax=ax, x='bins', y='values', data=modeldf, palette=gen_cols, showfliers=False)
        ax.scatter(x, ydata, color='k', marker='d', s=ms)

        ax.set_ylim([0, 1])
        ax.set_xticks(np.arange(4), ['16', '18', 'Hi5', 'OHR'])
        ax.set_ylabel('')
        ax.set_xlabel('')
        ax.set_title(rlabels[ai])

    fig.tight_layout()
    pl.savefig(f"figures/figS2.png", dpi=300)

    return

# %% Run as a script
if __name__ == '__main__':

    location = 'india'
    calib = sc.loadobj(f'results/nigeria_calib_reduced.obj')
    plot_calib(calib)

    print('Done.') 
