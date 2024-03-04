"""
Plot 2 for infant vaccination scenarios
"""
import pandas as pd


import pylab as pl
import sciris as sc
from run_scenarios import coverage_arr, efficacy_arr
import utils as ut


def preprocess_fig2(msim_dict):

    ut.set_font(16)

    # What to store
    start_year = 2025
    metrics = ['cancers', 'cancer_deaths', 'dalys']
    records = sc.autolist()

    for pn, metric in enumerate(metrics):

        for cn, cov_val in enumerate(coverage_arr):
            si = sc.findinds(msim_dict[base_label].year, start_year)[0]

            base_label = f'Adolescent: {cov_val} coverage'
            base_vals = msim_dict[base_label][metric].values[si:]

            for en, eff_val in enumerate(efficacy_arr):
                scen_label = f'Adolescent: {cov_val} coverage, Infants {eff_val} efficacy'
                scen_vals = msim_dict[scen_label][metric].values[si:]

                n_averted = sum(base_vals - scen_vals)

                records += {'coverage': cov_val, 'efficacy': eff_val, f'{metric}_averted': n_averted}

    df = pd.DataFrame(records)
    return


# %% Run as a script
if __name__ == '__main__':

    # Load scenarios and construct figure
    msim_dict = sc.loadobj('results/vx.scens')
    df = preprocess_fig2(msim_dict)





