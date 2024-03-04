"""
Plot 1 for infant vaccination scenarios
"""


import pylab as pl
import sciris as sc
from run_scenarios import coverage_arr, efficacy_arr
import utils as ut


def plot_single(ax, mres, to_plot, si, color, label=None):
    years = mres.year[si:]
    best = mres[to_plot][si:]
    low = mres[to_plot].low[si:]
    high = mres[to_plot].high[si:]
    ax.plot(years, best, color=color, label=label)
    ax.fill_between(years, low, high, alpha=0.5, color=color)
    return ax


def plot_fig1(msim_dict):

    ut.set_font(16)
    colors = sc.vectocolor(len(efficacy_arr), reverse=True)

    n_plots = len(coverage_arr)
    n_rows, n_cols = sc.get_rows_cols(n_plots)
    fig, axes = pl.subplots(n_rows, n_cols, figsize=(11, 10))
    if n_plots > 1: axes = axes.flatten()

    # What to plot
    start_year = 2020
    to_plot = 'asr_cancer_incidence'

    for pn, cov_val in enumerate(coverage_arr):

        ax = axes[pn] if n_plots > 1 else axes

        # Plot adolescents
        adolescent_label = f'Adolescent: {cov_val} coverage'
        mres = msim_dict[adolescent_label]
        si = sc.findinds(mres.year, start_year)[0]
        ax = plot_single(ax, mres, to_plot, si, 'k', label='Adolescents only')

        for ie, eff_val in enumerate(efficacy_arr):
            infant_label = f'Adolescents: {cov_val} coverage, Infants: {eff_val} efficacy'
            mres = msim_dict[infant_label]
            ax = plot_single(ax, mres, to_plot, si, colors[ie], label=f'Infants, {int(eff_val*100)}% efficacy')

        ax.set_ylim(bottom=0, top=23)
        ax.set_ylabel('ASR cancer incidence')
        ax.set_title(f'{int(cov_val*100)}% vaccine coverage')
        ax.axhline(y=4, color='k', ls='--')
        if pn == 0: ax.legend(frameon=False)

    fig.tight_layout()
    fig_name = 'figures/vx_scens.png'
    sc.savefig(fig_name, dpi=100)

    return


# %% Run as a script
if __name__ == '__main__':

    # Load scenarios and construct figure
    msim_dict = sc.loadobj('results/vx_scens.obj')
    plot_fig1(msim_dict)




