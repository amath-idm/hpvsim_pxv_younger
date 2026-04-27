"""
Plot Nigeria age pyramids.

Two modes:
  python plot_figS3_age_pyramids.py --run-sim   # run sim + save pyramid CSVs (VM)
  python plot_figS3_age_pyramids.py             # plot from saved CSVs (local)
"""
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sciris as sc
import seaborn as sns

import run_sims as rs
import utils as ut


AGE_LABELS = [
    '0-4', '5-9', '10-14', '15-19', '20-24', '25-29', '30-34', '35-39', '40-44',
    '45-49', '50-54', '55-59', '60-64', '65-69', '70-74', '75-79', '80-84', '85+'
]


def save_figS3_data(sim, resfolder='results',
                    years=('2025', '2050', '2075', '2100')):
    a = sim.get_analyzer('age_pyramid')
    model_rows = []
    for yi, yr in enumerate(years):
        p = sc.odict(a.age_pyramids)[yi]
        for bin_val, m_val, f_val in zip(p['bins'], p['m'], p['f']):
            model_rows.append({'year': yr, 'bin': int(bin_val),
                               'm': int(m_val), 'f': int(f_val)})
    pd.DataFrame(model_rows).to_csv(f'{resfolder}/figS3_model.csv', index=False)

    data = a.data.copy()
    data.columns = data.columns.str[0].str.lower()
    data = data[data['y'].isin([float(y) for y in years])]
    data.to_csv(f'{resfolder}/figS3_data.csv', index=False)


def plot_pops(years, percentages=True, resfolder='results',
              outpath='figures/figS3_age_pyramids.png'):
    n_years = len(years)
    n_rows, n_cols = sc.get_rows_cols(n_years)
    ut.set_font(size=14)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 10))
    axes = np.array(axes).flatten() if n_years > 1 else [axes]

    m_color = '#4682b4'
    f_color = '#ee7989'
    xlabel = 'Share of population by sex' if percentages else 'Population by sex'

    model_df = pd.read_csv(f'{resfolder}/figS3_model.csv')
    data_df = pd.read_csv(f'{resfolder}/figS3_data.csv')
    bins = sorted(model_df['bin'].unique())
    labels = list(reversed(AGE_LABELS))[:len(bins)]

    for c, syear in enumerate(years):
        ax = axes[c]
        pydf = model_df[model_df['year'] == int(syear)].copy()
        if pydf.empty:
            pydf = model_df[model_df['year'].astype(str) == str(syear)].copy()
        if percentages:
            pydf['m'] = pydf['m'] / pydf['m'].sum()
            pydf['f'] = pydf['f'] / pydf['f'].sum()
        pydf['f'] = -pydf['f']

        sns.barplot(x='m', y='bin', data=pydf, order=list(reversed(bins)),
                    orient='h', ax=ax, color=m_color)
        sns.barplot(x='f', y='bin', data=pydf, order=list(reversed(bins)),
                    orient='h', ax=ax, color=f_color)

        datadf = data_df[data_df['y'] == float(syear)].copy()
        if percentages and not datadf.empty:
            datadf['m'] = datadf['m'] / datadf['m'].sum()
            datadf['f'] = datadf['f'] / datadf['f'].sum()
        if not datadf.empty:
            datadf['f'] = -datadf['f']
            sns.pointplot(x='m', y='a', data=datadf, order=list(reversed(bins)),
                          orient='h', ax=ax, color='k', linestyles='')
            sns.pointplot(x='f', y='a', data=datadf, order=list(reversed(bins)),
                          orient='h', ax=ax, color='k', linestyles='')

        ax.set_xlabel(xlabel if c > 1 else '')
        ax.set_ylabel('')
        if c in [0, 2]:
            ax.set_yticklabels(labels)
        else:
            ax.set_yticklabels([])
        ax.set_xlim([-0.15, 0.15])
        xticks = ax.get_xticks()
        if percentages:
            xlabels = [f'{abs(i)*100:.1f}%' for i in xticks]
        else:
            xlabels = [f'{sc.sigfig(abs(i), sigfigs=2, SI=True)}' for i in xticks]
        ax.set_xticks(xticks, xlabels if c > 1 else [''] * len(xticks))
        ax.set_title(syear)

    fig.tight_layout()
    sc.savefig(outpath, dpi=100)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-sim', action='store_true',
                        help='Run baseline sim with age_pyramid analyzer (VM-side)')
    parser.add_argument('--resfolder', default='results/v2.0.x_published',
                        help='Dir with plot-ready CSVs (for plot mode only)')
    parser.add_argument('--outpath', default='figures/figS3_age_pyramids.png')
    parser.add_argument('--years', nargs='+', default=['2025', '2050', '2075', '2100'])
    args = parser.parse_args()

    if args.run_sim:
        import hpvsim as hpv
        calib_pars = sc.loadobj(f'{args.resfolder}/nigeria_pars.obj')
        sim = rs.run_sim(
            calib_pars=calib_pars,
            analyzers=[hpv.age_pyramid(timepoints=args.years, edges=np.arange(0, 81, 10), datafile='data/nigeria_age_pyramid_reduced.csv')],
            do_save=False,
            end=int(args.years[-1]),
        )
        save_figS3_data(sim, resfolder='results', years=args.years)
        print('Saved figS3 CSVs to results/ (copy to a versioned baseline dir to commit)')
    else:
        plot_pops(args.years, resfolder=args.resfolder, outpath=args.outpath)
        print('Done.')
