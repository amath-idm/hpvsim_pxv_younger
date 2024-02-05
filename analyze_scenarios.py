#%% import packages
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

def generate_age_label(column_name):
    if column_name.startswith(plot_val+'_by_age'):
        age_range = column_name.replace(plot_val+'_by_age_', '')
        age_range = age_range.replace('_', '-')
        return f'{5*int(age_range)}~{5*int(age_range)+4}'
    else:
        return column_name

def custom_sort_order(df):
    custom_order = ['0_latency', '100_latency', '50_latency_low_react', '50_latency_high_react']
    df['name_cat']= pd.Categorical(df['name'], categories=custom_order, ordered=True)
    df = df.sort_values(by='name_cat')
    df = df.drop('name_cat', axis=1)
    return df

#%% read simulation results (generated from run_scenarios - organize_msim_results)
no_df = pd.read_csv('results/india_scenarios_jan10_hpv_control_0_df.csv')
hund_df = pd.read_csv('results/india_scenarios_jan10_hpv_control_100_df.csv')
df1 = pd.read_csv('results/india_scenarios_calib50_lowact_df.csv')
df2 = pd.read_csv('results/india_scenarios_calib50_highact_df.csv')

no_df['latency'] = 0
df1['latency'] = 50
df2['latency'] = 50
hund_df['latency'] = 100

no_df['name'] = '0_latency'
df1['name'] = '50_latency_low_react'
df2['name'] = '50_latency_high_react'
hund_df['name'] = '100_latency'

all_df = pd.concat([no_df, df1, df2, hund_df])
all_df = custom_sort_order(all_df)
#%% Filter with not-vaccinated case. For each outcomes, plot results by year(x-axis) by 'name' (latency)
plot_df = all_df

# filtering
filter_by = 'vx_scen'
filter_val = plot_df[filter_by].unique()[0]
plot_df = plot_df[plot_df[filter_by]==filter_val]

# plot lines by
plot_by = 'name'

#plotting outcomes
plot_val_list = ['n_susceptible', 'n_infectious','n_precin','n_cin','cancers']

#plot setting
show_bound = True

fig, axes = plt.subplots(len(plot_val_list), 1, figsize=(7, 3 * len(plot_val_list)))
# for each simultion outcomes
for idx, plot_val in enumerate(plot_val_list):
    plot_idx = plot_val_list.index(plot_val)
    plot_by_list = plot_df[plot_by].unique()
    palette = sns.color_palette('coolwarm_r', n_colors=len(plot_by_list))
    data_mean = plot_df.groupby([plot_by, 'year'])[plot_val].mean().reset_index()
    ax = sns.lineplot(data=data_mean, x='year', y=plot_val, hue=plot_by, palette=palette, marker='o', markevery=20, linestyle='-', ax=axes[idx])

    # Use plt.fill_between for filling between low and high
    if show_bound:
        data_low = plot_df.groupby([plot_by, 'year'])[plot_val+'_low'].mean().reset_index()
        data_high = plot_df.groupby([plot_by, 'year'])[plot_val+'_high'].mean().reset_index()
        for sub_idx, group in data_low.groupby(plot_by):
            ax.fill_between(group['year'], group[plot_val+'_low'], data_high[data_high[plot_by] == sub_idx][plot_val+'_high'], alpha=0.3, color=palette[list(plot_by_list).index(sub_idx)])
    ax.set_xlabel('Year')
    ax.set_ylabel(plot_val)
    ax.xaxis.grid(True)
    # ax.set_ylim(0, 45000)  # Adjust the y-axis limit as needed
    ax.get_legend().set_visible(False)
    ax.set_title(plot_val + ' by year with ' + str(filter_val))
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=len(labels), title=plot_by, title_fontsize=14, fontsize=12)
plt.tight_layout()
plt.show()

#%% Compare total outcomes
plot_val_list = ['n_susceptible', 'n_infectious','n_precin','n_cin','cancers']
total_df = plot_df.groupby('name')[plot_val_list].sum().reset_index()[plot_val_list]
total_df /= np.mean(np.array(total_df), axis=0)
total_df['p(cin|precin)'] = total_df['n_cin']/total_df['n_precin']
total_df['p(cancer|cin)'] = total_df['cancers']/total_df['n_cin']
total_df['p(cancer|infectious)'] = total_df['cancers']/total_df['n_infectious']

unique_names = plot_df['name'].unique()
fig_width = len(unique_names) * 2  # Adjust the multiplier as needed
fig_height = len(plot_val_list) * 0.5  # Adjust the multiplier as needed
plt.figure(figsize=(fig_width, fig_height))

sns.heatmap(total_df,cmap='Blues', annot=True)
plt.xticks(np.arange(len(total_df.columns))+0.5, total_df.columns)
plt.yticks(np.arange(len(unique_names))+0.5, unique_names, rotation=0)
plt.axvline(x=5, color='red', linestyle='-', linewidth=2)
plt.title('Total occurrence divided by mean value')
plt.show()


#%%  For each outcomes and year, plot results by age(x-axis) by 'name' (latency model). Filter with not-vaccinated case. 
plot_df = all_df
filter_by = 'vx_scen'
filter_val = [plot_df[filter_by].unique()[0]]
plot_df = plot_df[plot_df[filter_by].isin(filter_val)]

# plot lines by
plot_by = 'name'

#plotting outcomes
plot_val_list = ['infections']

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
        selected_cols = [col for col in real_plot_df.columns if col.startswith(plot_val + '_by_age') and not col.endswith('_low') and not col.endswith('_high')]
        grouped_data = real_plot_df.groupby(plot_by)[selected_cols].mean().reset_index()
        melted_data = pd.melt(grouped_data, id_vars=[plot_by], var_name='x', value_name='value')
        melted_data['x'] = melted_data['x'].apply(generate_age_label)
        palette = sns.color_palette('coolwarm_r', n_colors=len(melted_data[plot_by].unique()))

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
            palette = sns.color_palette('coolwarm_r', n_colors=len(plot_df[plot_by].unique()))
            if plot_by == 'name':
                grouped_data = custom_sort_order(grouped_data)
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