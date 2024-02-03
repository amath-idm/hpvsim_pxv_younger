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
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import run_sim as rs
import utils as ut
from run_calibration import *

#%% read simulation results (generated from run_scenarios - organize_msim_results)
no_df = pd.read_csv('results/india_scenarios_jan10_hpv_control_0_df.csv')
hund_df = pd.read_csv('results/india_scenarios_jan10_hpv_control_100_df.csv')
df1 = pd.read_csv('results/india_scenarios_calib50_lowact_df.csv')
df2 = pd.read_csv('results/india_scenarios_calib50_highact_df.csv')

no_df['latency'] = 0
df1['latency'] = 50
df2['latency'] = 50
hund_df['latency'] = 100

df1['name'] = '50_latency_low_react'
df2['name'] = '50_latency_high_react'
no_df['name'] = '0_latency'
hund_df['name'] = '100_latency'

all_df = pd.concat([no_df, df1, df2, hund_df])
plot_df = all_df
#%% Filter with not-vaccinated case. For each outcomes, year, plot results by age(x-axis) by 'name' (latency)
def generate_age_label(column_name):
    if column_name.startswith(plot_val+'_by_age'):
        age_range = column_name.replace(plot_val+'_by_age_', '')
        age_range = age_range.replace('_', '-')
        return f'{5*int(age_range)}~{5*int(age_range)+4}'
    else:
        return column_name

plot_df = all_df

# filtering
filter_by = 'vx_scen'
filter_val = [plot_df[filter_by].unique()[0]]
plot_df = plot_df[plot_df[filter_by].isin(filter_val)]

# plot lines by
plot_by = 'name'

#plotting outcomes
plot_val_list = ['infections','cancers']

# additional filtering to show by year outcomes
filter_by = 'year'
filter_val_list = [1990, 2020, 2050, 2100]
num_rows = len(filter_val_list)

# for each simultion outcomes
for plot_val in plot_val_list:
    plot_idx = plot_val_list.index(plot_val)
    fig, axes = plt.subplots(num_rows, 1, figsize=(7, 3 * num_rows))
    # for each filtering (year)
    for idx, filter_val in enumerate(filter_val_list):
        real_plot_df = plot_df[plot_df[filter_by] == filter_val]
        # by age
        selected_cols = [col for col in real_plot_df.columns if col.startswith(plot_val + '_by_age') and not col.endswith('_low') and not col.endswith('_high')]
        grouped_data = real_plot_df.groupby(plot_by)[selected_cols].mean().reset_index()
        melted_data = pd.melt(grouped_data, id_vars=[plot_by], var_name='x', value_name='value')
        melted_data['x'] = melted_data['x'].apply(generate_age_label)
        palette = sns.color_palette('coolwarm_r', n_colors=len(melted_data[plot_by].unique()))

        # reoder
        if plot_by == 'name':
            custom_order = ['0_latency', '100_latency', '50_latency_low_react', '50_latency_high_react']
            melted_data['name'] = pd.Categorical(melted_data['name'], categories=custom_order, ordered=True)

        ax = sns.pointplot(data=melted_data, x='x', y='value', hue=plot_by, palette=palette, ax=axes[idx])
        ax.tick_params(axis='x', labelsize=6)
        ax.set_xlabel('Age Group')
        ax.set_ylabel(plot_val)
        ax.xaxis.grid(True)
        # ax.set_ylim(0, 45000)  # Adjust the y-axis limit as needed
        ax.get_legend().set_visible(False)
        ax.set_title(plot_val + ' by age at Year ' + str(filter_val))

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=len(labels), title='Latency', title_fontsize=14, fontsize=12)
    plt.tight_layout()
    plt.show()
#%% Compare incremental differences.
import matplotlib.ticker as ticker

# Define scenarios to compare 
plot_df = all_df
filter_by_scenario = 'vx_scen'
scenarios_to_compare = [plot_df[filter_by_scenario].unique()[0], plot_df[filter_by_scenario].unique()[-1]] # compare with no vaccine and most vaccine

plot_by = 'name'
plot_val_list = ['cancers', 'infections']
num_rows = len(plot_val_list)

fig, axes = plt.subplots(num_rows, 1, figsize=(7, 3 * num_rows))
for plot_val in plot_val_list:
    plot_idx = plot_val_list.index(plot_val)
    for idx, filter_val in enumerate(filter_val_list):
        melted_data_glob = pd.DataFrame()
        for i, scenario in enumerate(scenarios_to_compare):
            real_plot_df = plot_df[(plot_df[filter_by_scenario] == scenario)]
            grouped_data = real_plot_df.groupby([plot_by, 'year'])[plot_val].mean().reset_index()
            palette = sns.color_palette('coolwarm_r', n_colors=len(grouped_data[plot_by].unique()))
            if plot_by == 'name':
                custom_order = ['0_latency', '100_latency', '50_latency_low_react', '50_latency_high_react']
                melted_data['name'] = pd.Categorical(melted_data['name'], categories=custom_order, ordered=True)
            if i == 0:
                melted_data_glob = grouped_data
            melted_data_glob[scenario] = grouped_data[plot_val] # add scenario outcomes results

        # Calculate and plot differences between the first and second scenario
        melted_data_glob['difference'] = melted_data_glob[scenarios_to_compare[0]] - melted_data_glob[scenarios_to_compare[1]]
        ax = sns.lineplot(data=melted_data_glob, x='year', y='difference', hue=plot_by, ax=axes[plot_idx], palette=palette, marker='o', markevery=20,linestyle='-', linewidth = 3, legend=True if idx==0 else False)
        ax.set_title('Averted ' + plot_val + ' by year')
        ax.tick_params(axis='x', labelsize=6)
        # ax.set_ylim(0, 30000)  # Adjust the y-axis limit as needed
        ax.set_ylabel('Difference')
        ax.set_xlabel('Year')
        ax.xaxis.grid(True)
        ax.get_legend().set_visible(False)

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=len(labels), title='Latency')

plt.tight_layout()
plt.show()

#%%