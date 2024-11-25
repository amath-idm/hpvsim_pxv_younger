"""
Plot thresholds for effective coverage
"""


import pylab as pl
import numpy as np
import sciris as sc
import utils as ut


def plot_required_efficacy():

    ut.set_font(16)
    fig, ax = pl.subplots(1, 1, figsize=(8, 5))

    vea = 95  # Vaccine efficacy for adults
    vca_arr = np.linspace(10, 100, 10)  # Vaccine coverage for adults
    vci = np.linspace(1, 100, 100)  # Vaccine coverage for infants
    colors = sc.vectocolor(vca_arr)

    for li, vca in enumerate(vca_arr):
        vei = vea * vca/vci
        pl.plot(vci, vei, c=colors[li], alpha=1., lw=3, label=f'VCA={vca}')
        pl.text(101, 95*vca/100-1, f'VCA={int(vca)}', color=colors[li], fontsize=10)

    ax.set_ylim(bottom=0, top=100)
    ax.set_xlim([0, 110])
    ax.set_xlabel('Infant vaccine coverage')
    ax.set_ylabel('Required vaccine efficacy')
    ax.set_title(f'Vaccine efficacies required at time of debut')
    # ax.legend()

    fig.tight_layout()
    fig_name = 'figures/vei_needed.png'
    sc.savefig(fig_name, dpi=100)

    return


def plot_required_coverage():

    ut.set_font(16)
    fig, ax = pl.subplots(1, 1, figsize=(8, 5))

    vea = 95  # Vaccine efficacy for adults
    vca_arr = np.linspace(10, 90, 9)  # Vaccine coverage for adults
    vei = np.linspace(1, 95, 95)  # Vaccine efficacy for infants
    colors = sc.vectocolor(vca_arr)

    for li, vca in enumerate(vca_arr):
        vci = vea * vca/vei
        pl.plot(vei, vci, c=colors[li], alpha=1., lw=3, label=f'VCA={vca}')
        pl.text(96, 95*vca/95-1, f'VCA={int(vca)}', color=colors[li], fontsize=10)

    ax.set_ylim(bottom=0, top=100)
    ax.set_xlim([0, 110])
    ax.set_xlabel('Infant vaccine efficacy')
    ax.set_ylabel('Required infant vaccine coverage')
    ax.set_title(f'Required infant vaccine coverage for equivalency')
    # ax.legend()

    fig.tight_layout()
    fig_name = 'figures/vci_needed.png'
    sc.savefig(fig_name, dpi=100)

    return


# %% Run as a script
if __name__ == '__main__':

    plot_required_coverage()
    plot_required_efficacy()


