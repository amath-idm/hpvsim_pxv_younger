"""
Plot comparable efficacy if infant vaccination scenarios
"""


import pylab as pl
import sciris as sc
import numpy as np
from run_scenarios import coverage_arr, efficacy_arr
import utils as ut


def plot_single(ax, mres, to_plot, si, ei, color, label=None, smooth=True, al=None):
    years = mres.year[si:ei]
    ts = 0.67  # detection rate / sensitivity

    if to_plot == 'precin_incidence':
        best = mres.n_precin_by_age[3:11, si:ei].sum(axis=0) / mres.n_females_alive_by_age[3:11, si:ei].sum(axis=0) * ts
        low = mres.n_precin_by_age.low[3:11, si:ei].sum(axis=0) / mres.n_females_alive_by_age.low[3:11, si:ei].sum(axis=0) * ts
        high = mres.n_precin_by_age.high[3:11, si:ei].sum(axis=0) / mres.n_females_alive_by_age.high[3:11, si:ei].sum(axis=0) * ts
    else:
        best = mres[to_plot][si:ei]
        low = mres[to_plot].low[si:ei]
        high = mres[to_plot].high[si:ei]

    if smooth:
        best = np.convolve(list(best), np.ones(5), "valid")/5
        low = np.convolve(list(low), np.ones(5), "valid")/5
        high = np.convolve(list(high), np.ones(5), "valid")/5
        years = years[4:]

    ax.plot(years, best, color=color, label=label)
    # ax.fill_between(years, low, high, alpha=0.1, color=color)
    return ax


def plot_fig3(msim_dict):

    ut.set_font(20)  # 16 for paper
    plot_coverage_arr = coverage_arr[::2]  #[[0,4,8]]  #[[0,4,8]]  # which ones to plot
    plot_efficacy_arr = 0.95*plot_coverage_arr/.9  #[[0,4,8]]  #[[0,4,8]]  # which ones to plot
    colors = sc.vectocolor(len(plot_efficacy_arr), reverse=True)
    # covcolors = sc.vectocolor(len(plot_coverage_arr), reverse=True)
    plot_dict = sc.objdict(
        precin_incidence='Detectable HPV prevalence, females 15+',
        asr_cancer_incidence='ASR cancer incidence'
    )

    fig, axes = pl.subplots(len(plot_dict), 2, figsize=(17, 10))
    # fig, axes = pl.subplots(len(plot_dict), 2, figsize=(11, 10))

    # What to plot
    start_year = 2016
    end_year = 2100
    mbase = msim_dict['Baseline']
    si = sc.findinds(mbase.year, start_year)[0]
    ei = sc.findinds(mbase.year, end_year)[0]

    for rn, to_plot, plot_label in plot_dict.enumitems():

        # Plot adolescent scenarios
        cn = 0
        ax = axes[rn, cn]

        # Plot baseline
        baseline_label = 'Baseline'
        mres = msim_dict[baseline_label]
        ax = plot_single(ax, mres, to_plot, si, ei, 'k', label='Baseline')

        # Plot adolescents
        for cvn, cov_val in enumerate(plot_coverage_arr):
            adolescent_label = f'Adolescent: {np.round(cov_val, decimals=1)} coverage'
            mres = msim_dict[adolescent_label]
            ax = plot_single(ax, mres, to_plot, si, ei, colors[cvn], label=f'{int(np.floor(cov_val*100))}% coverage', al=adolescent_label)

        ax.set_ylim(bottom=0)  #, top=23)
        ax.set_ylabel(plot_label)
        ax.set_title(f'Adolescent vaccination scenarios\n{plot_label}')
        if rn==0: ax.legend(frameon=False)
        if to_plot == 'asr_cancer_incidence': ax.axhline(y=4, color='k', ls='--')
        cn += 1

        # Plot infants
        ax = axes[rn, cn]

        # Plot baseline
        baseline_label = 'Baseline'
        mres = msim_dict[baseline_label]
        ax = plot_single(ax, mres, to_plot, si, ei, 'k', label='Baseline')

        for ie, eff_val in enumerate(plot_efficacy_arr):
            cov_val = eff_val*0.9/0.95
            infant_label = f'Infants: {np.round(eff_val, decimals=3)} efficacy'
            mres = msim_dict[infant_label]
            ax = plot_single(ax, mres, to_plot, si, ei, colors[ie], label=f'{int(np.ceil(eff_val*100))}% efficacy')

        ax.set_ylim(bottom=0)  #, top=23)
        ax.set_ylabel(plot_label)
        ax.set_title(f'Equivalent infant vaccination scenarios\n{plot_label}')
        if to_plot == 'asr_cancer_incidence': ax.axhline(y=4, color='k', ls='--')
        if rn==0: ax.legend(frameon=False)

    fig.tight_layout()
    fig_name = 'figures/fig3_vx_scens.png'
    sc.savefig(fig_name, dpi=100)

    return


# %% Run as a script
if __name__ == '__main__':

    # Load scenarios and construct figure
    msim_dict = sc.loadobj('results/vx_scens_equiv.obj')
    plot_fig3(msim_dict)


    print('Done.')

