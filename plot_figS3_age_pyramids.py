""" Plot age pyramids """

# Import packages
import sciris as sc
import numpy as np
import pylab as pl
import pandas as pd
import seaborn as sns
import utils as ut


#%% Functions
def plot_pops(years, percentages=True):

    n_years = len(years)
    n_rows, n_cols = sc.get_rows_cols(n_years)
    ut.set_font(size=14)
    fig, axes = pl.subplots(n_rows, n_cols, figsize=(10, 10))
    if n_years > 1:
        axes = axes.flatten()
    else:
        axes = axes

    m_color = '#4682b4'
    f_color = '#ee7989'
    xlabel = 'Share of population by sex' if percentages else 'Population by sex'

    for c, syear in enumerate(years):
        sim = sc.loadobj('results/nigeria.sim')
        a = sim.get_analyzer('age_pyramid')
        pyramid = sc.odict(a.age_pyramids)[c]
        labels = list(reversed(a.age_labels))

        bins = pyramid['bins']
        ax = axes[c]

        # Prepare data
        pydf = pd.DataFrame(pyramid)
        if percentages:
            pydf['m'] = pydf['m'] / sum(pydf['m'])
            pydf['f'] = pydf['f'] / sum(pydf['f'])
        pydf['f'] = -pydf['f']  # Reverse values for females to get on same axis

        # Start making plot
        sns.barplot(x='m', y='bins', data=pydf, order=np.flip(bins), orient='h', ax=ax, color=m_color)
        sns.barplot(x='f', y='bins', data=pydf, order=np.flip(bins), orient='h', ax=ax, color=f_color)

        datadf = a.data[a.data.year == float(syear)]
        datadf.columns = datadf.columns.str[0]
        datadf.columns = datadf.columns.str.lower()
        if percentages:
            datadf = datadf.assign(m=datadf['m'] / sum(datadf['m']), f=datadf['f'] / sum(datadf['f']))
        datadf = datadf.assign(f=-datadf['f'])
        sns.pointplot(x='m', y='a', data=datadf, order=np.flip(bins), orient='h', ax=ax, color='k', linestyles='')
        sns.pointplot(x='f', y='a', data=datadf, order=np.flip(bins), orient='h', ax=ax, color='k', linestyles='')

        ax.set_xlabel(xlabel)
        ax.set_ylabel('')
        if c in [0, 2]:
            ax.set_yticklabels(labels[1:])
        else:
            ax.set_yticklabels([])
        ax.set_xlim([-0.15, 0.15])
        xticks = ax.get_xticks()
        if percentages:
            xlabels = [f'{abs(i)*100:.1f}%' for i in xticks]
        else:
            xlabels = [f'{sc.sigfig(abs(i), sigfigs=2, SI=True)}' for i in xticks]
        if c > 1:
            ax.set_xticks(xticks, xlabels)
        else:
            ax.set_xticks(xticks, [])
        ax.set_xlabel('')
        ax.set_title(syear)

    fig.tight_layout()
    sc.savefig(f'figures/figS3_age_pyramids.png', dpi=100)


#%% Run as a script
if __name__ == '__main__':

    plot_pops(['2025', '2050', '2075', '2100'])

    print('Done.')
