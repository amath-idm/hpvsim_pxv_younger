"""
This file is used to run calibrations for TxV 10-country analysis.

Instructions: Go to the CONFIGURATIONS section on lines 29-36 to set up the script before running it.
"""
#%%
# Additions to handle numpy multithreading
import os

os.environ.update(
    OMP_NUM_THREADS='1',
    OPENBLAS_NUM_THREADS='1',
    NUMEXPR_NUM_THREADS='1',
    MKL_NUM_THREADS='1',
)

# Standard imports
import sciris as sc
import hpvsim as hpv
import pylab as pl
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# Imports from this repository
import run_sim as rs
import utils as ut
from run_calibration import *

########################################################################
# Functions
########################################################################
def load_calib(location=None, do_plot=True, which_pars=0, save_pars=True, filestem=''):
    fnlocation = location.replace(' ', '_')
    filename = f'{fnlocation}_calib{filestem}'
    calib = sc.load(f'results/{filename}.obj')
    if do_plot:
        # sc.options(font='Libertinus Sans')
        n_to_plot = len(calib.df[calib.df['mismatch']<1.8])
        fig = calib.plot(res_to_plot=n_to_plot, plot_type='sns.boxplot', do_save=False)
        fig.suptitle(f'Calibration results, {location.capitalize()}')
        fig.tight_layout()
        fig.savefig(f'figures/{filename}.png')

    if save_pars:
        calib_pars = calib.trial_pars_to_sim_pars(which_pars=which_pars)
        trial_pars = sc.autolist()
        for i in range(10):
            trial_pars += calib.trial_pars_to_sim_pars(which_pars=i)
        sc.save(f'results/{location}_pars{filestem}.obj', trial_pars)

    return calib

# Read failed attempts from tmp files
def read_calibration_with_tmp(filename):
    import optuna as op
    import os
    storage = f'sqlite:///{filename}.db'
    study = op.load_study(storage=storage, study_name=filename)
    print("Study loaded")
    tmp_filename = 'tmp_calibration_%05i.obj'
    calib = run_calib(location='india',n_trials=8000, n_workers=40, k=3, run_calib = False)
    calib.glabels = [g.upper() for g in calib.sim['genotype_map'].values()]
    calib.sim_results = []
    calib.analyzer_results = []
    calib.extra_sim_results = []
    calib.extra_analyzer_results = []
    for trial in study.trials:
        n = trial.number
        filename_tmp = tmp_filename % trial.number
        if os.path.exists(filename_tmp):
            results = sc.load(filename_tmp)
            calib.sim_results.append(results['sim'])
            calib.analyzer_results.append(results['analyzer'])
            calib.extra_sim_results.append(results['extra_sim_results'])
    print("Added trials to calib")
    calib.parse_study(study)
    print("Parsed study")
    # sc.saveobj(f'results/{filename}.obj', calib)
    return (calib)

def plot_density(dfs, labels=None, max_mismatch=None, show_points = False, plot_index=None, test_dist = False):
    from scipy import stats

    ''' Plot kernel densities '''
    plt.rcdefaults()
    fig, axes = pl.subplots(8, 4, figsize=(20, 16))
    colors = sc.gridcolors(len(dfs))
    colors = sc.vectocolor([i for i in range(len(dfs))], cmap='coolwarm_r')
    # rearrange column orders to align better
    new_columns = ['hpv_control_prob', 'hpv_reactivation', 'cell_imm_init_par1', 'own_imm_hr', 'f_cross_layer', 
                        'f_partners_c_par1','m_cross_layer', 'm_partners_c_par1', 
                        'beta',  'hpv18_rel_beta', 'hi5_rel_beta', 'ohr_rel_beta',
                        'sev_dist_par1', 'age_risk_age', 'age_risk_risk',
                        'hpv16_dur_cin_par1', 'hpv18_dur_cin_par1', 'hi5_dur_cin_par1', 'ohr_dur_cin_par1',
                        'hpv16_dur_cin_par2', 'hpv18_dur_cin_par2', 'hi5_dur_cin_par2',  'ohr_dur_cin_par2',
                        'hpv16_cancer_fn_ld50', 'hpv18_cancer_fn_ld50', 'hi5_cancer_fn_ld50', 'ohr_cancer_fn_ld50']
    
    samp1 =list(list() for _ in range(len(new_columns)))
    samp2 =list(list() for _ in range(len(new_columns)))

    for i, df in enumerate(dfs):
        if max_mismatch is not None:
            inds = hpv.true(df['mismatch'].values <= max_mismatch)
            df = df.iloc[inds, :].copy()

        df = df.drop(columns=['index','mismatch'])
        df = df[new_columns]

        pars = df.columns
        for ia, ax in enumerate(axes.flat):
            if ia <= len(pars) and ia!=15:
                df_to_plot = df.iloc[:, ia - 1] if ia > 14 else df.iloc[:, ia]  # Check for out-of-bounds
                if test_dist:
                    if i==0:  samp1[ia - (ia > 14)] = df_to_plot.values
                    if i==1:  samp2[ia - (ia > 14)] = df_to_plot.values
                if df_to_plot.var() > 0:
                        sns.kdeplot(df_to_plot, color=colors[i], ax=ax, fill=True, label=None if labels is None else labels[i], alpha=0.5)       
                else:
                    ax.axvline(df_to_plot.iloc[0], color=colors[i], linestyle='-', linewidth=10, label=None if labels is None else labels[i], alpha=0.5)
                ax.set_xlabel(pars[ia-1] if ia>14 else pars[ia], fontsize=15)              
                if show_points: 
                    points = df_to_plot.values[0:10]
                    if plot_index is not None:
                        ax.scatter(x=[points[i] for i in plot_index], y=np.ones(len(plot_index))* (1.1 * ax.get_ylim()[1]), color=colors[i], alpha=0.4)                            
                        for j, point in enumerate(points):
                            if j in plot_index:
                                ax.text(point, (1.1) * ax.get_ylim()[1], str(j), ha='center', va='center', fontsize=10)
                    else:
                        ax.scatter(x=points, y=np.ones(len(points))* (1.1 * ax.get_ylim()[1]), color=colors[i], alpha=0.4)                            
                       
                # ax.legend()
            else:
                ax.set_axis_off()

    results = []

    # Perform KS test and store results - to identify if two distributions are statistically different
    if test_dist:
        for k in range(len(samp1)):
            ks_statistic, p_value = stats.ks_2samp(samp1[k], samp2[k])
            results.append({
                'new_column': new_columns[k],
                'ks_statistic': ks_statistic,
                'p_value': p_value
            })
        # Sort the results based on p-values in descending order
        sorted_results = sorted(results, key=lambda x: x['p_value'], reverse=False)
        # Print or further process the sorted results
        for result in sorted_results:
            print(result['new_column'], "KS test statistic:", round(result['ks_statistic'],2), "KS test p-value:", result['p_value'])
    if labels is not None:
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.05), ncol=len(labels), title='hpv_control_prob', title_fontsize=20, fontsize=15)
    fig.tight_layout()
    fig_name = f'figures/calib_densities.png'
    # sc.savefig(fig_name, dpi=300)
    return

def pairplotpars(df, inds=None, color_column=None, cmap='parula', bins=None, edgecolor='w', facecolor='#F8A493',
                 figsize=(20, 16)):
    ''' Plot scatterplots, histograms, and kernel densities '''
    if inds is not None:
        df = df.iloc[inds, :].copy()

    # Choose the colors
    if color_column:
        colors = sc.vectocolor(df[color_column].values, cmap=cmap)
    else:
        colors = [facecolor for _ in range(len(df))]
    df['color_column'] = [sc.rgb2hex(rgba[:-1]) for rgba in colors]

    def hide_current_axis(*args, **kwds):
        pl.gca().set_visible(False)

    # Make the plot
    df = df.drop(columns=['mismatch'])
    grid = sns.PairGrid(df, diag_sharey=False)
    grid = grid.map_lower(pl.scatter, **{'facecolors': df['color_column'], 's':1})
    grid = grid.map_diag(pl.hist, bins=bins, edgecolor=edgecolor, facecolor=facecolor)
    grid = grid.map_upper(hide_current_axis)
    grid.fig.set_size_inches(figsize)
    grid.fig.tight_layout()

    return grid

# %% Read files

# calib0 = load_calib('india',  filestem='_jan10_hpv_control_0') # Save top 10 or top 5 param sets using this function
calib0 = sc.load(f'results/india_calib_jan10_hpv_control_0.obj')
calib25 = sc.load(f'results/india_calib_jan10_hpv_control_25.obj')
calib50 = sc.load(f'results/india_calib_jan10_hpv_control_50.obj')
calib75 = sc.load(f'results/india_calib_jan10_hpv_control_75.obj')
calib100 = sc.load(f'results/india_calib_jan10_hpv_control_100.obj')

calib0_df = calib0.df
calib0_df['hpv_reactivation'] = 0.0
calib25_df = calib25.df
calib50_df = calib50.df
calib75_df = calib75.df
calib100_df = calib100.df
all_df =  pd.concat([calib0_df, calib25_df, calib75_df, calib100_df])

#%% Filter by condition and plot densities
test_col = 'hpv_reactivation'
test_col_val = 0.5
mismatch = 2.0
condition = lambda df: df[df['mismatch'] < mismatch]
condition2 = lambda df: df[(df['mismatch'] < mismatch) & (df[test_col] < test_col_val)]
condition3 = lambda df: df[(df['mismatch'] < mismatch) & (df[test_col] >= test_col_val)]

plot_density([condition(calib0_df),condition(calib100_df)], labels=['0%','100%'],show_points = True)

#%% Observe outcomes - for parameter exploration
from test_parameter_exploration import *

calib0_df_res = organize_results(calib0)
calib25_df_res = organize_results(calib25)
calib50_df_res = organize_results(calib50)
calib75_df_res = organize_results(calib75)
calib100_df_res = organize_results(calib100)
all_res = pd.concat([calib0_df_res, calib25_df_res, calib75_df_res, calib100_df_res])

result_df = all_res
param_cols = [col for col in calib0_df.columns.values if col not in ['mismatch', 'index']]
X = result_df[param_cols]
Y = result_df[result_df.columns.difference(param_cols+['index'])]
param_importance = fit_model('LinearRegression', X, Y) # choose method from 'RandomForest', 'LinearRegression', and 'XBGoost'
outcomes = get_interest_outcome(calib0, Y, outcome_level=1)
heatmap(param_importance, outcomes, save_plot = False, sort_by=['mismatch'])

#%% Identify correlations between parameters
corr = condition(all_df).corr()
corr_long = corr.unstack().reset_index()
corr_long.columns = ['param1', 'param2', 'corr_score']

corr_long_filtered = corr_long[
    (corr_long['param1'] != corr_long['param2']) &
    ~corr_long['param1'].str.contains('index') &
    ~corr_long['param2'].str.contains('index') &
    (corr_long['param1'] < corr_long['param2'])  # Additional condition for removing redundant rows
]
corr_long_filtered['abs_score'] = np.abs(corr_long_filtered['corr_score'])
corr_long_filtered['Notes'] = corr_long['param1'] + ' vs ' + corr_long['param2']
# corr_long_filtered.to_csv()
param_interest = ['hpv_control_prob','f_cross_layer','cell_imm_init_par1','age_risk_risk','hpv_reactivation','f_partners_c_par1','m_partners_c_par1']
filtered_param = corr_long_filtered[
    (corr_long_filtered['param1'].isin(param_interest)) & 
    (corr_long_filtered['param2'].isin(param_interest))
]

#%%