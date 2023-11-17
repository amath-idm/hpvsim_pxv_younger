"""
Read in sexual behavior data
"""

# Import packages
import sciris as sc
import numpy as np
import pylab as pl
import pandas as pd
import seaborn as sns
from scipy.stats import norm, lognorm
import hpvsim as hpv
import hpvsim.utils as hpu

# Imports from this repository
import utils as ut
import run_sim as rs
import locations as set

def percentiles_to_pars(x1, p1, x2, p2):
    """ Find the parameters of a normal distribution where:
            P(X < p1) = x1
            P(X < p2) = x2
    """
    p1ppf = norm.ppf(p1)
    p2ppf = norm.ppf(p2)

    location = ((x1 * p2ppf) - (x2 * p1ppf)) / (p2ppf - p1ppf)
    scale = (x2 - x1) / (p2ppf - p1ppf)
    return location, scale


def logn_percentiles_to_pars(x1, p1, x2, p2):
    """ Find the parameters of a lognormal distribution where:
            P(X < p1) = x1
            P(X < p2) = x2
    """
    x1 = np.log(x1)
    x2 = np.log(x2)
    p1ppf = norm.ppf(p1)
    p2ppf = norm.ppf(p2)
    s = (x2 - x1) / (p2ppf - p1ppf)
    mean = ((x1 * p2ppf) - (x2 * p1ppf)) / (p2ppf - p1ppf)
    scale = np.exp(mean)
    return s, scale


def read_debut_data(dist_type='lognormal'):
    '''
    Read in dataframes taken from DHS and return them in a plot-friendly format,
    optionally saving the distribution parameters
    '''

    df1 = pd.read_csv('data/afs_dist.csv')
    df2 = pd.read_csv('data/afs_median.csv')

    # Deal with median data
    df2['y'] = 50

    # Rearrange data into a plot-friendly format
    dff = {}
    rvs = {'Women': {}, 'Men': {}}

    for sex in ['Women', 'Men']:

        dfw = df1[['Country', f'{sex} 15', f'{sex} 18', f'{sex} 20', f'{sex} 22', f'{sex} 25', f'{sex} never']]
        dfw = dfw.melt(id_vars='Country', value_name='Percentage', var_name='AgeStr')

        # Add values for proportion ever having sex
        countries = dfw.Country.unique()
        n_countries = len(countries)
        vals = []
        for country in countries:
            val = 100-dfw.loc[(dfw['AgeStr'] == f'{sex} never') & (dfw['Country'] == country) , 'Percentage'].iloc[0]
            vals.append(val)

        data_cat = {'Country': countries, 'AgeStr': [f'{sex} 60']*n_countries}
        data_cat["Percentage"] = vals
        df_cat = pd.DataFrame.from_dict(data_cat)
        dfw = pd.concat([dfw,df_cat])

        conditions = [
            (dfw['AgeStr'] == f"{sex} 15"),
            (dfw['AgeStr'] == f"{sex} 18"),
            (dfw['AgeStr'] == f"{sex} 20"),
            (dfw['AgeStr'] == f"{sex} 22"),
            (dfw['AgeStr'] == f"{sex} 25"),
            (dfw['AgeStr'] == f"{sex} 60"),
        ]
        values = [15, 18, 20, 22, 25, 60]
        dfw['Age'] = np.select(conditions, values)

        dff[sex] = dfw

        res = dict()
        res["location"] = []
        res["par1"] = []
        res["par2"] = []
        res["dist"] = []
        for pn,country in enumerate(countries):
            dfplot = dfw.loc[(dfw["Country"] == country) & (dfw["AgeStr"] != f'{sex} never') & (dfw["AgeStr"] != f'{sex} 60')]
            x1 = 15
            p1 = dfplot.loc[dfplot["Age"] == x1, 'Percentage'].iloc[0] / 100
            x2 = df2.loc[df2["Country"]==country,f"{sex} median"].iloc[0]
            p2 = .50
            # x2 = 25
            # p2 = dfplot.loc[dfplot["Age"] == x2, 'Percentage'].iloc[0] / 100
            res["location"].append(country)
            res["dist"].append(dist_type)

            if dist_type=='normal':
                loc, scale = percentiles_to_pars(x1, p1, x2, p2)
                rv = norm(loc=loc, scale=scale)
                res["par1"].append(loc)
                res["par2"].append(scale)
            elif dist_type=='lognormal':
                s, scale = logn_percentiles_to_pars(x1, p1, x2, p2)
                rv = lognorm(s=s, scale=scale)
                res["par1"].append(rv.mean())
                res["par2"].append(rv.std())

            rvs[sex][country] = rv

        pd.DataFrame.from_dict(res).to_csv(f'data/sb_pars_{sex.lower()}_{dist_type}.csv')

    return countries, dff, df2, rvs


def read_marriage_data():
    dfraw = pd.read_csv('data/prop_married.csv')
    df = dfraw.melt(id_vars=['Country', 'Survey'], value_name='Percentage', var_name='AgeRange')
    return df


def get_sb_from_sims(dist_type='lognormal', marriage_scale=1, debut_bias=[0,0],
                     verbose=-1, debug=False):
    '''
    Run sims with the sexual debut parameters inferred from DHA data, and save
    the proportion of people of each age who've ever had sex
    '''

    locations = set.locations
    countries_to_run = locations
    sims = rs.run_sims(
        locations=countries_to_run,
        age_pyr=True,
        analyzers=[ut.AFS(),ut.prop_married(),hpv.snapshot(timepoints=['2020'])],
        debug=debug,
        dist_type=dist_type,
        marriage_scale=marriage_scale,
        debut_bias=debut_bias,
        verbose=verbose,
    )

    # Save output on age at first sex (AFS)
    dfs = sc.autolist()
    for country in countries_to_run:
        a = sims[country].get_analyzer('AFS')
        for cs,cohort_start in enumerate(a.cohort_starts):
            df = pd.DataFrame()
            df['age'] = a.bins
            df['cohort'] = cohort_start
            df['model_prop_f'] = a.prop_active_f[cs,:]
            df['model_prop_m'] = a.prop_active_m[cs,:]
            df['country'] = country
            dfs += df
    afs_df = pd.concat(dfs)
    sc.saveobj(f'results/model_sb_AFS.obj', afs_df)

    # Save output on proportion married
    dfs = sc.autolist()
    for country in countries_to_run:
        a = sims[country].get_analyzer('prop_married')
        df = a.df
        df['country'] = country
        dfs += df
    pm_df = pd.concat(dfs)
    sc.saveobj(f'results/model_sb_prop_married.obj', pm_df)

    # Save output on age differences between partners
    dfs = sc.autolist()
    for country in countries_to_run:
        df = pd.DataFrame()
        snapshot = sims[country].get_analyzer('snapshot')
        ppl = snapshot.snapshots[0]
        age_diffs = ppl.contacts['m']['age_m'] - ppl.contacts['m']['age_f']
        df['age_diffs'] = age_diffs
        df['country'] = country
        dfs += df
    agediff_df = pd.concat(dfs)
    sc.saveobj(f'results/model_age_diffs.obj', agediff_df)


    # Save output on the number of casual relationships
    binspan = 5
    bins = np.arange(15, 50, binspan)
    dfs = sc.autolist()
    for country in countries_to_run:
        snapshot = sims[country].get_analyzer('snapshot')
        ppl = snapshot.snapshots[0]
        conditions = {}
        general_conditions = ppl.is_female * ppl.alive * ppl.level0 * ppl.is_active
        for ab in bins:
            conditions[ab] = (ppl.age >= ab) * (ppl.age < ab + binspan) * general_conditions

        casual_partners = {(0,1): sc.autolist(), (1,2):sc.autolist(), (2,3):sc.autolist(), (3,5):sc.autolist(), (5,50):sc.autolist()}
        for cp in casual_partners.keys():
            for ab,age_cond in conditions.items():
                this_condition = conditions[ab] * (ppl.current_partners[1,:]>=cp[0]) * (ppl.current_partners[1,:]<cp[1])
                casual_partners[cp] += len(hpu.true(this_condition))

        popsize = sc.autolist()
        for ab, age_cond in conditions.items():
            popsize += len(hpu.true(age_cond))

        # Construct dataframe
        n_bins = len(bins)
        partners = np.repeat([0, 1, 2, 3, 5], n_bins)
        allbins = np.tile(bins, 5)
        counts = np.concatenate([val for val in casual_partners.values()])
        allpopsize = np.tile(popsize, 5)
        shares = counts / allpopsize
        datadict = dict(bins=allbins, partners=partners, counts=counts, popsize=allpopsize, shares=shares)
        df = pd.DataFrame.from_dict(datadict)
        df['country'] = country
        dfs += df

    casual_df = pd.concat(dfs)
    sc.saveobj(f'results/model_casual.obj', casual_df)


    return sims, afs_df, pm_df, agediff_df, casual_df


def plot_sb(dist_type='lognormal'):
    '''
    Create plots of sexual behavior inputs and outputs
    '''

    ut.set_font(12)

    data_countries, dff, df2, rvs = read_debut_data(dist_type=dist_type)
    alldf = sc.loadobj(f'results/model_sb_AFS.obj')
    countries = alldf['country'].unique()
    n_countries = len(countries)
    n_rows, n_cols = sc.get_rows_cols(n_countries)

    for sk,sex in {'f':'Women', 'm':'Men'}.items():
        fig, axes = pl.subplots(n_rows, n_cols, figsize=(8,11))
        axes = axes.flatten()
        dfw = dff[sex]

        for pn,country in enumerate(countries):
            ax = axes[pn]
            data_country = ut.map_sb_loc(country)
            dfplot = dfw.loc[(dfw["Country"]==data_country)&(dfw["AgeStr"]!=f'{sex} never')&(dfw["AgeStr"]!=f'{sex} 60')]
            dfmed = df2.loc[df2["Country"] == data_country]
            if len(dfplot)>0:
                sns.scatterplot(ax=ax, data=dfplot, x="Age", y="Percentage")
                sns.scatterplot(ax=ax, data=dfmed, x=f"{sex} median", y="y")

                rv = rvs[sex][data_country]
                xx = np.arange(12,30.1,0.1)
                xxx = np.arange(12,31,1)
                ax.plot(xx, rv.cdf(xx)*100, 'k--', lw=2)

            for cohort in alldf["cohort"].unique():
                modely = np.array(alldf.loc[(alldf["country"]==country)&(alldf["cohort"]==cohort)][f'model_prop_{sk}'])
                ax.plot(xxx, modely*100, 'b-', lw=1)
            title_country = country.title()
            if title_country == 'Congo Democratic Republic':
                title_country = 'DRC'

            ax.set_title(title_country)
            ax.set_ylabel('')
            ax.set_xlabel('')

        fig.tight_layout()
        pl.savefig(f"figures/SMs/fig_sb_{sex.lower()}.png", dpi=100)

    return


def plot_prop_married():
    '''
    Create plots of sexual behavior inputs and outputs
    '''

    ut.set_font(12)
    # Read in data and model results
    df = read_marriage_data()
    modeldf = sc.loadobj(f'results/model_sb_prop_married.obj')
    modeldf.reset_index()

    # Plot
    countries = modeldf.country.unique()
    n_countries = len(countries)
    n_rows, n_cols = sc.get_rows_cols(n_countries)
    colors = sc.gridcolors(1)

    fig, axes = pl.subplots(n_rows, n_cols, figsize=(8, 11))
    if n_countries>1:
        axes = axes.flatten()
    else:
        axes = axes

    for pn, country in enumerate(countries):
        if n_countries > 1:
            ax = axes[pn]
        else:
            ax = axes

        # Plot data
        d_country = ut.map_sb_loc(country)
        dfplot_d = df.loc[(df["Country"] == d_country)]
        sns.scatterplot(ax=ax, data=dfplot_d, x="AgeRange", y="Percentage")

        # Plot model
        location = ut.rev_map_sb_loc(country)
        dfplot_m = modeldf.loc[modeldf['country']==location]
        # dfplot_m['val'] = dfplot_m['val'].multiply(100)
        dfplot_m['val'] = dfplot_m['val'].apply(lambda x: x * 100)
        sns.boxplot(data=dfplot_m, x="age", y="val", color=colors[0], ax=ax)

        title_country = country.title()
        if title_country == 'Drc':
            title_country = 'DRC'

        ax.set_title(title_country)
        ax.set_ylabel('')
        ax.set_xlabel('')

    fig.tight_layout()
    pl.savefig(f"figures/SMs/fig_prop_married.png", dpi=100)

    return



def plot_age_diffs():
    '''
    Plot the age differences between marital partners
    '''

    ut.set_font(12)
    agediffs = sc.loadobj(f'results/model_age_diffs.obj')

    # Plot
    countries = agediffs.country.unique()
    n_countries = len(countries)
    n_rows, n_cols = sc.get_rows_cols(n_countries)
    colors = sc.gridcolors(1)

    fig, axes = pl.subplots(n_rows, n_cols, figsize=(8, 11))
    if n_countries>1:
        axes = axes.flatten()
    else:
        axes = axes

    for pn, country in enumerate(countries):
        if n_countries > 1:
            ax = axes[pn]
        else:
            ax = axes

        # Plot model
        dfplot_m = agediffs.loc[agediffs['country']==country]
        sns.kdeplot(data=dfplot_m, color=colors[0], ax=ax)
        ax.legend([], [], frameon=False)

        title_country = country.title()
        if title_country == 'Drc':
            title_country = 'DRC'

        ax.set_title(title_country)
        ax.set_ylabel('')
        ax.set_xlabel('')

    fig.tight_layout()
    pl.savefig(f"figures/SMs/fig_age_diffs.png", dpi=100)

    return


def plot_casuals():
    '''
    Plot the number of casual partners by age
    '''

    ut.set_font(12)
    casual_df = sc.loadobj(f'results/model_casual.obj')
    data_df = pd.read_excel(f'data/casuals.xlsx')
    data_df = data_df.melt(id_vars=['Country', 'Survey'], value_name='Percentage', var_name='AgeStr')

    # Plot
    countries = casual_df.country.unique()
    n_countries = len(countries)
    n_rows, n_cols = sc.get_rows_cols(n_countries)

    fig, axes = pl.subplots(n_rows, n_cols, figsize=(8, 11))
    if n_countries>1:
        axes = axes.flatten()
    else:
        axes = axes

    for pn, country in enumerate(countries):
        if n_countries > 1:
            ax = axes[pn]
        else:
            ax = axes

        # Plot model
        dfplot_m = casual_df.loc[(casual_df['country']==country) & (casual_df['partners'] > 0)]
        dfplot_m['shares'] = dfplot_m['shares'].apply(lambda x: x * 100)
        dfpm = pd.DataFrame({'bins':dfplot_m.bins.unique(), 'shares':dfplot_m.groupby(['bins', 'country']).sum()['shares'].values})
        sns.barplot(data=dfpm, x="bins", y="shares", ax=ax, color='b', alpha=0.5)  #hue="partners", ax=ax)
        ax.legend([], [], frameon=False)

        # Plot data
        data_country = ut.map_sb_loc(country)
        dfplot_d = data_df.loc[data_df['Country']==data_country]
        sns.scatterplot(data=dfplot_d, x="AgeStr", y="Percentage", ax=ax, marker="D", color="k", s=50)

        title_country = country.title()
        if title_country == 'Drc':
            title_country = 'DRC'

        ax.set_title(title_country)
        ax.set_ylabel('')
        ax.set_xlabel('')
        ax.tick_params(axis='both', which='major', labelsize=10)

    fig.tight_layout()
    pl.savefig(f"figures/SMs/fig_casual.png", dpi=100)

    return


#%% Run as a script
if __name__ == '__main__':

    # countries, dff, df2, rvs = read_debut_data(dist_type=dist_type)

    dist_type = 'lognormal'
    do_run = False

    if do_run:
        sims, afs_df, pm_df, agediff_df, casual_df = get_sb_from_sims(
            dist_type=dist_type,
            marriage_scale=1,
            debut_bias=[-1,-1],
            debug=False,
            verbose=0.1
        )

    # Plotting functions
    # plot_sb(dist_type=dist_type)
    # plot_prop_married()
    # plot_age_diffs()
    plot_casuals()

    print('Done.')
