'''
Utilities for multicalibration
'''

# Standard imports
import sciris as sc
import hpvsim as hpv
import hpvsim.parameters as hppar
import pandas as pd
import numpy as np
import pylab as pl
import seaborn as sns
from matplotlib.gridspec import GridSpec
import math
from scipy.stats import lognorm, norm

import analyzers as an
import run_sim as rs

resfolder = 'results'
figfolder = 'figures'
datafolder = 'data'

def set_font(size=None, font='Libertinus Sans'):
    ''' Set a custom font '''
    sc.fonts(add=sc.thisdir(aspath=True) / 'assets' / 'LibertinusSans-Regular.otf')
    sc.options(font=font, fontsize=size)
    return


def map_sb_loc(location):
    ''' Map between different representations of country names '''
    location = location.title()
    if location == "Cote Divoire": location = "Cote d'Ivoire"
    if location == "Drc": location = 'Congo Democratic Republic'
    return location


def rev_map_sb_loc(location):
    ''' Map between different representations of country names '''
    location = location.lower()
    # location = location.replace(' ', '_')
    if location == 'congo democratic republic': location = "drc"
    if location == "cote d'ivoire": location = 'cote divoire'
    return location


def make_sb_data(location=None, dist_type='lognormal', debut_bias=[0,0]):

    sb_location = map_sb_loc(location)

    # Read in data
    sb_data_f = pd.read_csv(f'data/sb_pars_women_{dist_type}.csv')
    sb_data_m = pd.read_csv(f'data/sb_pars_men_{dist_type}.csv')

    try:
        distf = sb_data_f.loc[sb_data_f["location"]==sb_location,"dist"].iloc[0]
        par1f = sb_data_f.loc[sb_data_f["location"]==sb_location,"par1"].iloc[0]
        par2f = sb_data_f.loc[sb_data_f["location"]==sb_location,"par2"].iloc[0]
        distm = sb_data_m.loc[sb_data_m["location"]==sb_location,"dist"].iloc[0]
        par1m = sb_data_m.loc[sb_data_m["location"]==sb_location,"par1"].iloc[0]
        par2m = sb_data_m.loc[sb_data_m["location"]==sb_location,"par2"].iloc[0]
    except:
        print(f'No data for {sb_location=}, {location=}')

    debut = dict(
        f=dict(dist=distf, par1=par1f+debut_bias[0], par2=par2f),
        m=dict(dist=distm, par1=par1m+debut_bias[1], par2=par2m),
    )

    return debut


def make_datafiles(locations):
    ''' Get the relevant datafiles for the selected locations '''
    datafiles = dict()
    asr_locs            = ['drc', 'ethiopia', 'kenya', 'nigeria', 'tanzania', 'uganda']
    cancer_type_locs    = ['ethiopia', 'kenya', 'nigeria', 'tanzania', 'india', 'uganda']
    cin_type_locs       = ['nigeria', 'tanzania', 'india']

    for location in locations:
        dflocation = location.replace(' ','_')
        datafiles[location] = [
            f'data/{dflocation}_cancer_cases.csv',
        ]

        if location in asr_locs:
            datafiles[location] += [f'data/{dflocation}_asr_cancer_incidence.csv']

        if location in cancer_type_locs:
            datafiles[location] += [f'data/{dflocation}_cancer_types.csv']

        if location in cin_type_locs:
            datafiles[location] += [f'data/{dflocation}_cin_types.csv']

    return datafiles



def lognorm_params(par1, par2):
    """
    Given the mean and std. dev. of the log-normal distribution, this function
    returns the shape and scale parameters for scipy's parameterization of the
    distribution.
    """
    mean = np.log(par1 ** 2 / np.sqrt(par2 ** 2 + par1 ** 2))  # Computes the mean of the underlying normal distribution
    sigma = np.sqrt(np.log(par2 ** 2 / par1 ** 2 + 1))  # Computes sigma for the underlying normal distribution

    scale = np.exp(mean)
    shape = sigma
    return shape, scale



def plot_vx_impact(location=None, background_scen=None, adolescent_coverages=None, infant_coverage=None,
                   infant_efficacies=None, discounting=False):

    set_font(size=16)

    bigdf = sc.loadobj(f'{resfolder}/{location}.obj')
    econdf = sc.loadobj(f'{resfolder}/{location}_econ.obj')

    colors = sc.gridcolors(20)
    standard_le = 88.8
    markers = ['s', 'p', '*']
    ls = ['dashed', 'dotted']
    xes = np.arange(len(infant_efficacies))
    fig, axes = pl.subplots(nrows=1, ncols=3, figsize=(12, 8))

    for ia, adolescent_coverage in enumerate(adolescent_coverages):
        vx_scen_label = f'Vx, {adolescent_coverage}% cov, 9-14'
        screen_scen_label = background_scen['screen_scen']
        no_infant_df = bigdf[(bigdf.screen_scen == screen_scen_label) & (bigdf.vx_scen == vx_scen_label)].groupby('year')[
                ['asr_cancer_incidence', 'asr_cancer_incidence_low', 'asr_cancer_incidence_high',
                 'cancers', 'cancers_low', 'cancers_high', 'cancer_deaths', 'cancer_deaths_low', 'cancer_deaths_high']].sum()

        no_infant_econdf_cancers = econdf[(econdf.screen_scen == screen_scen_label) & (econdf.vx_scen == vx_scen_label)].groupby('year')[
                ['new_cancers', 'new_cancer_deaths']].sum()

        no_infant_econdf_ages = econdf[(econdf.screen_scen == screen_scen_label) & (econdf.vx_scen == vx_scen_label)].groupby('year')[
                ['av_age_cancer_deaths', 'av_age_cancers']].mean()

        if discounting:
            cancers_no_infant = np.array([i / 1.03 ** t for t, i in enumerate(no_infant_econdf_cancers['new_cancers'].values)])
            cancer_deaths_no_infant = np.array(
                [i / 1.03 ** t for t, i in enumerate(no_infant_econdf_cancers['new_cancer_deaths'].values)])
        else:
            cancers_no_infant = no_infant_econdf_cancers['new_cancers'].values
            cancer_deaths_no_infant = no_infant_econdf_cancers['new_cancer_deaths'].values

        avg_age_ca_death = np.mean(no_infant_econdf_ages['av_age_cancer_deaths'])
        avg_age_ca = np.mean(no_infant_econdf_ages['av_age_cancers'])
        ca_years = avg_age_ca_death - avg_age_ca
        yld_no_infant = np.sum(np.sum([0.54 * .1, 0.049 * .5, 0.451 * .3, 0.288 * .1]) * ca_years * cancers_no_infant)
        yll_no_infant = np.sum((standard_le - avg_age_ca_death) * cancer_deaths_no_infant)
        dalys_no_infant = yll_no_infant + yld_no_infant

        ys = sc.findinds(no_infant_df.index, 2025)[0]
        ye = sc.findinds(no_infant_df.index, 2100)[0]
        years = no_infant_df.index[ys:ye]
        cancer_ts = np.array(no_infant_df['asr_cancer_incidence'])[ys:ye]
        cancer_ts_low = np.array(no_infant_df['asr_cancer_incidence_low'])[ys:ye]
        cancer_ts_high = np.array(no_infant_df['asr_cancer_incidence_high'])[ys:ye]

        for ieff, inf_efficacy in enumerate(infant_efficacies):
            vx_scen_label_to_use = f'{vx_scen_label}, {infant_coverage}% cov, infant, {inf_efficacy}% efficacy'
            df = bigdf[(bigdf.screen_scen == screen_scen_label) & (bigdf.vx_scen == vx_scen_label_to_use)].groupby('year')[
                ['asr_cancer_incidence', 'asr_cancer_incidence_low', 'asr_cancer_incidence_high',
                 'cancers', 'cancers_low', 'cancers_high', 'cancer_deaths', 'cancer_deaths_low',
                 'cancer_deaths_high']].sum()

            econdf_cancers = \
            econdf[(econdf.screen_scen == screen_scen_label) & (econdf.vx_scen == vx_scen_label_to_use)].groupby('year')[
                ['new_cancers', 'new_cancer_deaths']].sum()

            econdf_ages = \
            econdf[(econdf.screen_scen == screen_scen_label) & (econdf.vx_scen == vx_scen_label_to_use)].groupby('year')[
                ['av_age_cancer_deaths', 'av_age_cancers']].mean()

            if discounting:
                cancers = np.array([i / 1.03 ** t for t, i in enumerate(econdf_cancers['new_cancers'].values)])
                cancer_deaths = np.array(
                    [i / 1.03 ** t for t, i in enumerate(econdf_cancers['new_cancer_deaths'].values)])
            else:
                cancers = econdf_cancers['new_cancers'].values
                cancer_deaths = econdf_cancers['new_cancer_deaths'].values

            avg_age_ca_death = np.mean(econdf_ages['av_age_cancer_deaths'])
            avg_age_ca = np.mean(econdf_ages['av_age_cancers'])
            ca_years = avg_age_ca_death - avg_age_ca
            yld = np.sum(np.sum([0.54 * .1, 0.049 * .5, 0.451 * .3, 0.288 * .1]) * ca_years * cancers)
            yll = np.sum((standard_le - avg_age_ca_death) * cancer_deaths)
            dalys = yll + yld
            dalys_averted = dalys_no_infant - dalys

            axes[2].scatter(xes[ieff], dalys_averted, color=colors[ieff+1], marker=markers[ia], s=200)
            cancers_averted = np.sum(np.array(no_infant_df['cancers'])[ys:ye]) - np.sum(np.array(df['cancers'])[ys:ye])
            if ieff==0:
                axes[0].scatter(xes[ieff], cancers_averted, color=colors[ieff+1], marker=markers[ia], s=200, label=f'{adolescent_coverage}%')
            else:
                axes[0].scatter(xes[ieff], cancers_averted, color=colors[ieff + 1], marker=markers[ia], s=200)
            cancer_deaths_averted = np.sum(np.array(no_infant_df['cancer_deaths'])[ys:ye]) - np.sum(np.array(df['cancer_deaths'])[ys:ye])
            axes[1].scatter(xes[ieff], cancer_deaths_averted, color=colors[ieff+1], marker=markers[ia], s=200)

    axes[0].axhline(y=0, color='black')
    axes[1].axhline(y=0, color='black')
    axes[2].axhline(y=0, color='black')
    axes[0].set_xticks([r for r in range(len(xes))], infant_efficacies)
    axes[1].set_xticks([r for r in range(len(xes))], infant_efficacies)
    axes[2].set_xticks([r for r in range(len(xes))], infant_efficacies)
    axes[0].set_xlabel('Vaccine efficacy for infants (%)')
    axes[1].set_xlabel('Vaccine efficacy for infants (%)')
    axes[2].set_xlabel('Vaccine efficacy for infants (%)')
    # ax1.set_ylabel('Cervical cancer incidence (per 100k)')
    axes[2].set_ylabel('DALYs averted relative to 9yo (2025-2100)')
    axes[1].set_ylabel('Cancer deaths averted relative to 9yo (2025-2100)')
    axes[0].set_ylabel('Cancers averted relative to 9yo (2025-2100)')
    for ax in axes:
        sc.SIticks(ax)

    axes[0].legend(title='9-14 coverage')
    fig.tight_layout()
    fig_name = f'{figfolder}/{location}_vx_impact.png'
    fig.savefig(fig_name, dpi=100)

    return

def plot_CEA(location=None, background_scen=None, adolescent_coverages=None, infant_coverage=None,
                   infant_efficacies=None, discounting=False):

    set_font(size=24)

    econ_df = sc.loadobj(f'{resfolder}/{location}_econ.obj')

    cost_dict = dict(
        adolescent_pxv=9,
        infant_pxv=5,
        leep=41.76,
        ablation=11.76,
        cancer=450
    )

    standard_le = 88.8
    colors = sc.gridcolors(20)
    markers = ['s', 'p', '*']
    fig, ax = pl.subplots(figsize=(12, 12))
    screen_scen_label = background_scen['screen_scen']
    handles = []
    for ia, adolescent_coverage in enumerate(adolescent_coverages):
        vx_scen_label = f'Vx, {adolescent_coverage}% cov, 9-14'
        no_infant_econdf_counts = econ_df[(econ_df.screen_scen == screen_scen_label) & (econ_df.vx_scen == vx_scen_label)].groupby(
            'year')[
            ['new_vaccinations', 'new_infant_vaccinations', 'new_thermal_ablations', 'new_leeps',
             'new_cancer_treatments', 'new_cancers', 'new_cancer_deaths']].sum()

        no_infant_econdf_means = econ_df[(econ_df.screen_scen == screen_scen_label) & (econ_df.vx_scen == vx_scen_label)].groupby(
            'year')[
            ['av_age_cancer_deaths', 'av_age_cancers']].mean()

        if discounting:
            cancers = np.array([i / 1.03 ** t for t, i in enumerate(no_infant_econdf_counts['new_cancers'].values)])
            cancer_deaths = np.array(
                [i / 1.03 ** t for t, i in enumerate(no_infant_econdf_counts['new_cancer_deaths'].values)])
        else:
            cancers = no_infant_econdf_counts['new_cancers'].values
            cancer_deaths = no_infant_econdf_counts['new_cancer_deaths'].values
        avg_age_ca_death = np.mean(no_infant_econdf_means['av_age_cancer_deaths'])
        avg_age_ca = np.mean(no_infant_econdf_means['av_age_cancers'])
        ca_years = avg_age_ca_death - avg_age_ca
        yld = np.sum(np.sum([0.54 * .1, 0.049 * .5, 0.451 * .3, 0.288 * .1]) * ca_years * cancers)
        yll = np.sum((standard_le - avg_age_ca_death) * cancer_deaths)
        daly_no_infant = yll + yld
        total_cost_no_infant = (no_infant_econdf_counts['new_vaccinations'].values * cost_dict['adolescent_pxv']) + \
                           (no_infant_econdf_counts['new_thermal_ablations'].values * cost_dict['ablation']) + \
                           (no_infant_econdf_counts['new_leeps'].values * cost_dict['leep']) + \
                           (no_infant_econdf_counts['new_cancer_treatments'].values * cost_dict['cancer'])
        if discounting:
            cost_no_infant = np.sum([i / 1.03 ** t for t, i in enumerate(total_cost_no_infant)])
        else:
            cost_no_infant = np.sum(total_cost_no_infant)

        for ieff, inf_efficacy in enumerate(infant_efficacies):
            vx_scen_label_to_use = f'{vx_scen_label}, {infant_coverage}% cov, infant, {inf_efficacy}% efficacy'

            econdf_cancers = \
                econ_df[(econ_df.screen_scen == screen_scen_label) & (econ_df.vx_scen == vx_scen_label_to_use)].groupby(
                    'year')[
                    ['new_vaccinations', 'new_infant_vaccinations', 'new_thermal_ablations', 'new_leeps',
                     'new_cancer_treatments', 'new_cancers', 'new_cancer_deaths']].sum()

            econdf_ages = \
                econ_df[(econ_df.screen_scen == screen_scen_label) & (econ_df.vx_scen == vx_scen_label_to_use)].groupby(
                    'year')[
                    ['av_age_cancer_deaths', 'av_age_cancers']].mean()

            if discounting:
                cancers = np.array([i / 1.03 ** t for t, i in enumerate(econdf_cancers['new_cancers'].values)])
                cancer_deaths = np.array(
                    [i / 1.03 ** t for t, i in enumerate(econdf_cancers['new_cancer_deaths'].values)])
            else:
                cancers = econdf_cancers['new_cancers'].values
                cancer_deaths = econdf_cancers['new_cancer_deaths'].values
            avg_age_ca_death = np.mean(econdf_ages['av_age_cancer_deaths'])
            avg_age_ca = np.mean(econdf_ages['av_age_cancers'])
            ca_years = avg_age_ca_death - avg_age_ca
            yld = np.sum(np.sum([0.54 * .1, 0.049 * .5, 0.451 * .3, 0.288 * .1]) * ca_years * cancers)
            yll = np.sum((standard_le - avg_age_ca_death) * cancer_deaths)
            daly_infant = yll + yld

            total_cost_infant = (econdf_cancers['new_vaccinations'].values * cost_dict['adolescent_pxv']) + \
                                (econdf_cancers['new_infant_vaccinations'].values * cost_dict['infant_pxv']) + \
                                (econdf_cancers['new_thermal_ablations'].values * cost_dict['ablation']) + \
                                (econdf_cancers['new_leeps'].values * cost_dict['leep']) + \
                                (econdf_cancers['new_cancer_treatments'].values * cost_dict['cancer'])
            if discounting:
                cost_infant = np.sum([i / 1.03 ** t for t, i in enumerate(total_cost_infant)])
            else:
                cost_infant = np.sum(total_cost_infant)

            dalys_averted = daly_no_infant - daly_infant
            additional_cost = cost_infant - cost_no_infant
            cost_daly_averted = additional_cost / dalys_averted

            handle, = ax.plot(dalys_averted / 1e6, cost_daly_averted, color=colors[ieff + 1], marker=markers[ia],
                    linestyle='None', markersize=20)

            handles.append(handle)

    legend1 = ax.legend([handles[0], handles[3], handles[6]], adolescent_coverages, title='9-14 coverage', loc=1)
    ax.legend([handles[0], handles[1], handles[2]], infant_efficacies, title='Infant efficacy', loc='center right')
    pl.gca().add_artist(legend1)

    sc.SIticks(ax)
    # ax.legend(title='Background intervention scale-up')
    ax.set_xlabel('DALYs averted (millions), 2025-2100')
    ax.set_ylabel('Incremental costs/DALY averted, $USD 2025-2100')
    # ax.axhline(y=0, color='black')
    # ax.axvline(x=0, color='black')
    ax.set_ylim(bottom=0)
    ax.set_xlim(left=0)
    # fig.suptitle(f'TxV CEA for {locations}', fontsize=18)
    fig.tight_layout()
    fig_name = f'{figfolder}/CEA.png'
    fig.savefig(fig_name, dpi=100)

    return

def plot_resource_use(location=None, background_scen=None, adolescent_coverages=None, infant_coverage=None,
                   infant_efficacies=None):

    set_font(size=24)

    econ_df = sc.loadobj(f'{resfolder}/{location}_econ.obj')

    colors = sc.gridcolors(20)
    df = sc.dcp(econ_df)
    df['year'] = df.index
    df = df.pivot(index='year', columns='vx_scen', values='new_vaccinations')
    fig, axes = pl.subplots(2,1, figsize=(12, 12))
    screen_scen_label = background_scen['screen_scen']
    width = 0.1
    r1 = np.arange(len(adolescent_coverages))
    r2 = [x + width for x in r1]
    r3 = [x + width for x in r2]
    xes = [r1, r2, r3]

    for icov, adolescent_coverage in enumerate(adolescent_coverages):
        for ieff, inf_efficacy in enumerate(infant_efficacies):
            vx_scen_label_to_use = f'Vx, {adolescent_coverage}% cov, 9-14, {infant_coverage}% cov, infant, {inf_efficacy}% efficacy'

            econdf_cancers = \
            econ_df[(econ_df.screen_scen == screen_scen_label) & (econ_df.vx_scen == vx_scen_label_to_use)].groupby('year')[
                ['new_vaccinations', 'new_infant_vaccinations']].sum()
            adolsecent_pxv = np.sum(econdf_cancers['new_vaccinations'])
            infant_pxv = np.sum(econdf_cancers['new_infant_vaccinations'])

            if icov == 0:
                axes[0].scatter(xes[ieff][icov], adolsecent_pxv/1e6, color=colors[ieff+1], s=200, label=inf_efficacy)
            else:
                axes[0].scatter(xes[ieff][icov], adolsecent_pxv/1e6, s=200, color=colors[ieff + 1])
            axes[1].scatter(xes[ieff][icov], infant_pxv/1e9, s=200, color=colors[ieff + 1])

    axes[0].set_ylabel('Doses, 2025-2100 (millions)')
    axes[1].set_ylabel('Doses, 2025-2100 (billions)')
    axes[0].set_title('Adolescent vaccine doses')
    axes[1].set_title('Infant vaccine doses')
    axes[0].set_xticks([r + 1.5*width for r in range(len(r1))], adolescent_coverages)
    axes[1].set_xticks([r + 1.5 * width for r in range(len(r1))], adolescent_coverages)
    # sc.SIticks(axes[0])
    # sc.SIticks(axes[1])
    axes[0].legend()
    fig.tight_layout()
    fig_name = f'{figfolder}/resource_use.png'
    fig.savefig(fig_name, dpi=100)

    return
